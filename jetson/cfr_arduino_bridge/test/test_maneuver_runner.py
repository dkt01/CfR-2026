"""Tests for the characterization run's state machine.

ManeuverRunner is the interlock between a profile and a car that can drive
itself: it will not leave ARMING for a moving profile until the Arduino is in
AUTO_ACTIVE and the ZED's odometry is fresh, and it aborts a run the instant
either one drops out -- see docs/characterization.md and the module docstring
in maneuver_runner_node.py. That is the thing that stops a car with no
odometry from just continuing on the last command it had.

Unlike lap_counter's and the detector's tests, this one needs the built
workspace: ManeuverRunner is a real rclpy Node (there is no ROS-free module to
import on its own), so these construct one and drive it with synthetic
cfr_interfaces/nav_msgs messages. Every timeout in the state machine reads
time.monotonic() directly rather than a clock passed in, so a fake clock is
patched into the module for the whole file -- that is what lets every test
below run instantly, with no real waiting for a countdown or a timeout.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import pytest
import rclpy
import yaml
from cfr_interfaces.msg import ArduinoStatus
from nav_msgs.msg import Odometry

MODULE = Path(__file__).parents[1] / "src" / "maneuver_runner_node.py"
_spec = importlib.util.spec_from_file_location("maneuver_runner_node", MODULE)
mrn = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mrn
_spec.loader.exec_module(mrn)


# --------------------------------------------------------------------- clock


class FakeClock:
    """A monotonic clock the test drives, so no timeout needs a real wait."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(mrn.time, "monotonic", fake)
    return fake


# -------------------------------------------------------------------- runner


def write_profile(tmp_path, **overrides):
    data = {
        "name": "test",
        "arming": overrides.pop("arming", "estop_cycle"),
        "limits": overrides.pop(
            "limits", {"max_distance": 40.0, "max_duration": 180.0, "max_speed": 3.5}
        ),
        "steps": overrides.pop(
            "steps", [{"label": "hold", "hold": 1.0, "velocity": 1.0}]
        ),
        "return": overrides.pop("return", {"mode": "none"}),
    }
    data.update(overrides)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


@pytest.fixture
def build(tmp_path, monkeypatch, request, clock):
    """A ManeuverRunner wired to a throwaway run directory, no subprocesses.

    Bag recording and the parameter-dump snapshot both shell out to `ros2`;
    patching `shutil.which` away takes the same no-op path a laptop with no
    `ros2` on PATH already takes, so neither runs during a test.
    """
    monkeypatch.setattr(mrn.shutil, "which", lambda _cmd: None)

    def make(*, params=None, **profile_overrides):
        profile_path = write_profile(tmp_path, **profile_overrides)
        values = {
            "profile": profile_path,
            "run_dir": str(tmp_path / "run"),
            "record_bag": False,
        }
        values.update(params or {})
        args = ["--ros-args"]
        for name, value in values.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            args += ["-p", f"{name}:={value}"]
        rclpy.init(args=args)
        node = mrn.ManeuverRunner()
        request.addfinalizer(
            lambda: (node.shutdown(), node.destroy_node(), rclpy.shutdown())
        )
        return node

    return make


def status(
    *,
    mode=ArduinoStatus.MODE_AUTO_ACTIVE,
    link_ok=True,
    estop=False,
    auto_arm=True,
    gains_applied=True,
    battery_level=200,
):
    msg = ArduinoStatus()
    msg.link_ok = link_ok
    msg.estop = estop
    msg.auto_arm = auto_arm
    msg.mode = mode
    msg.gains_applied = gains_applied
    msg.battery_level = battery_level
    return msg


def odom(x=0.0, y=0.0, yaw=0.0):
    msg = Odometry()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
    msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
    return msg


def feed(node, *, status_msg=None, odom_msg=None):
    """Push fresh samples in, the way the subscriptions would."""
    if status_msg is not None:
        node._on_status(status_msg)
    if odom_msg is not None:
        node._on_odom(odom_msg)


def start_running(node, clock, *, status_msg=None, odom_msg=None):
    """Drive an ARMING node straight into RUNNING (countdown must be 0)."""
    node._set_phase(node.ARMING)
    feed(node, status_msg=status_msg or status(), odom_msg=odom_msg or odom())
    node._tick_arming(clock())  # requests the (empty) profile gains
    node._tick_arming(clock())  # gains settle instantly; countdown is 0
    assert node.phase == node.RUNNING


# ------------------------------------------------------------------ WAIT_LINK


def test_link_up_with_estop_cycle_required_waits_for_estop(build):
    node = build()
    assert node.phase == node.WAIT_LINK

    feed(node, status_msg=status())
    node._tick_wait_link(time.monotonic())

    assert node.phase == node.WAIT_ESTOP_ASSERTED


def test_a_non_arming_profile_skips_the_estop_cycle(build):
    node = build(arming="none", steps=[{"label": "hold", "hold": 1.0, "velocity": 0.0}])

    feed(node, status_msg=status())
    node._tick_wait_link(time.monotonic())

    assert node.phase == node.ARMING


def test_require_estop_cycle_false_skips_straight_to_arming(build):
    node = build(params={"require_estop_cycle": False})

    feed(node, status_msg=status())
    node._tick_wait_link(time.monotonic())

    assert node.phase == node.ARMING


def test_no_link_yet_stays_at_wait_link(build):
    node = build()
    node._tick_wait_link(time.monotonic())
    assert node.phase == node.WAIT_LINK


# --------------------------------------------------------------- estop cycle


def test_estop_must_assert_then_clear_with_auto_armed(build, clock):
    node = build()
    feed(node, status_msg=status())
    node._tick_wait_link(clock())
    assert node.phase == node.WAIT_ESTOP_ASSERTED

    # Not asserted yet: keeps waiting.
    feed(node, status_msg=status(estop=False))
    node._tick_wait_estop_asserted(clock())
    assert node.phase == node.WAIT_ESTOP_ASSERTED

    feed(node, status_msg=status(estop=True, mode=ArduinoStatus.MODE_ESTOP))
    node._tick_wait_estop_asserted(clock())
    assert node.phase == node.WAIT_ESTOP_CLEARED

    # Cleared, but auto is not armed on the controller: keeps waiting.
    feed(
        node,
        status_msg=status(
            estop=False, auto_arm=False, mode=ArduinoStatus.MODE_AUTO_ARMED
        ),
    )
    node._tick_wait_estop_cleared(clock())
    assert node.phase == node.WAIT_ESTOP_CLEARED
    assert node.auto_ready  # ready for the AUTO_ARMED -> AUTO_ACTIVE handshake

    feed(
        node,
        status_msg=status(
            estop=False, auto_arm=True, mode=ArduinoStatus.MODE_AUTO_ARMED
        ),
    )
    node._tick_wait_estop_cleared(clock())
    assert node.phase == node.ARMING


# ----------------------------------------------------------- ARMING / ZED gate


def test_arming_will_not_start_with_no_odometry_at_all(build, clock):
    """The core interlock: no ZED odometry, ever, so the run cannot start."""
    node = build(params={"countdown": 0.0})
    node._set_phase(node.ARMING)
    feed(node, status_msg=status())  # link and mode fine; no odom fed at all

    node._tick_arming(clock())

    assert node.phase == node.ARMING
    assert node.gains_requested_at is None  # never got past the odometry gate


def test_arming_will_not_start_on_stale_odometry(build, clock):
    """Odometry that stopped updating is treated the same as none at all."""
    node = build(params={"countdown": 0.0})
    node._set_phase(node.ARMING)
    feed(node, status_msg=status(), odom_msg=odom())
    clock.advance(node.odom_timeout + 0.1)  # ZED odom has gone stale

    node._tick_arming(clock())

    assert node.phase == node.ARMING
    assert node.gains_requested_at is None


def test_arming_waits_for_auto_active_before_odometry_even_matters(build, clock):
    node = build(params={"countdown": 0.0})
    node._set_phase(node.ARMING)
    feed(node, status_msg=status(mode=ArduinoStatus.MODE_AUTO_ARMED), odom_msg=odom())

    node._tick_arming(clock())

    assert node.phase == node.ARMING
    assert node.gains_requested_at is None


def test_fresh_odometry_and_auto_active_let_the_run_start(build, clock):
    node = build(params={"countdown": 0.0})
    node._set_phase(node.ARMING)
    feed(node, status_msg=status(), odom_msg=odom(1.0, 2.0, 0.3))

    node._tick_arming(clock())  # requests the (empty) profile gains
    assert node.phase == node.ARMING
    assert node.gains_requested_at is not None

    node._tick_arming(clock())  # gains settle instantly; countdown is 0

    assert node.phase == node.RUNNING
    assert node.origin == pytest.approx((1.0, 2.0, 0.3))


def test_a_non_arming_profile_still_needs_odometry_to_start(build, clock):
    """`arming: none` skips the AUTO_ACTIVE wait, never the odometry gate."""
    node = build(
        arming="none",
        steps=[{"label": "hold", "hold": 1.0, "velocity": 0.0}],
        params={"countdown": 0.0, "require_estop_cycle": False},
    )
    node._set_phase(node.ARMING)
    feed(node, status_msg=status(estop=True, mode=ArduinoStatus.MODE_ESTOP))

    node._tick_arming(clock())
    assert node.phase == node.ARMING
    assert node.gains_requested_at is None

    feed(node, odom_msg=odom())
    node._tick_arming(clock())
    node._tick_arming(clock())
    assert node.phase == node.RUNNING


# --------------------------------------------------------------- RUNNING aborts


def test_stale_odometry_aborts_a_running_maneuver(build, clock):
    """The failure this whole interlock exists for: odom drops mid-run."""
    node = build(params={"countdown": 0.0})
    start_running(node, clock)

    clock.advance(node.odom_timeout + 0.1)
    feed(node, status_msg=status())  # keep the link fresh; only odom goes stale
    node._check_aborts(clock())

    assert node.result == "aborted"
    assert "odometry stale" in node.result_detail
    assert node.phase == node.STOPPING
    assert node.command == (0.0, 0.0)


def test_link_loss_aborts_a_running_maneuver(build, clock):
    node = build(params={"countdown": 0.0})
    start_running(node, clock)

    clock.advance(1.1)  # past the 1.0 s link timeout; no fresh status fed
    node._check_aborts(clock())

    assert node.result == "aborted"
    assert "Arduino link lost" in node.result_detail


def test_estop_asserted_mid_run_aborts(build, clock):
    node = build(params={"countdown": 0.0})
    start_running(node, clock)

    feed(node, status_msg=status(estop=True, mode=ArduinoStatus.MODE_ESTOP))
    node._check_aborts(clock())

    assert "E-Stop asserted" in node.result_detail


def test_leaving_auto_active_mid_run_aborts(build, clock):
    node = build(params={"countdown": 0.0})
    start_running(node, clock)

    feed(node, status_msg=status(mode=ArduinoStatus.MODE_AUTO_ARMED))
    node._check_aborts(clock())

    assert "left AUTO_ACTIVE" in node.result_detail


def test_exceeding_max_duration_aborts(build, clock):
    node = build(
        params={"countdown": 0.0},
        limits={"max_distance": 40.0, "max_duration": 10.0, "max_speed": 3.5},
    )
    start_running(node, clock)

    clock.advance(10.1)
    feed(node, status_msg=status(), odom_msg=odom())
    node._check_aborts(clock())

    assert "exceeded max_duration" in node.result_detail


def test_exceeding_max_distance_aborts(build, clock):
    node = build(
        params={"countdown": 0.0},
        limits={"max_distance": 5.0, "max_duration": 180.0, "max_speed": 3.5},
    )
    start_running(node, clock)

    feed(node, status_msg=status(), odom_msg=odom(10.0, 0.0, 0.0))
    node._check_aborts(clock())

    assert "exceeded max_distance" in node.result_detail


def test_a_non_arming_profile_ignores_estop_but_not_stale_odometry(build, clock):
    """A stationary recording tolerates E-Stop; it still needs live odometry."""
    node = build(
        arming="none",
        steps=[{"label": "hold", "hold": 1.0, "velocity": 0.0}],
        params={"countdown": 0.0, "require_estop_cycle": False},
    )
    start_running(
        node, clock, status_msg=status(estop=True, mode=ArduinoStatus.MODE_ESTOP)
    )

    feed(node, status_msg=status(estop=True, mode=ArduinoStatus.MODE_ESTOP))
    node._check_aborts(clock())
    assert node.phase == node.RUNNING  # E-Stop is the normal state for this profile

    clock.advance(node.odom_timeout + 0.1)
    node._check_aborts(clock())
    assert node.result == "aborted"
    assert "odometry stale" in node.result_detail


# ------------------------------------------------------------------- RUNNING


def test_steps_hold_then_advance_to_the_next(build, clock):
    node = build(
        params={"countdown": 0.0},
        steps=[
            {"label": "a", "hold": 1.0, "velocity": 1.0},
            {"label": "b", "hold": 1.0, "velocity": 2.0},
        ],
    )
    start_running(node, clock)
    assert node.step_index == 0

    node._tick_running(clock())  # hold not elapsed yet
    assert node.step_index == 0
    assert node.command == (0.0, 1.0)

    clock.advance(1.0)
    node._tick_running(clock())  # holds step 0's command while flipping the index
    assert node.step_index == 1
    assert node.command == (0.0, 1.0)

    node._tick_running(clock())  # step 1's command is set on the tick after
    assert node.command == (0.0, 2.0)


def test_the_last_step_with_no_return_leg_finishes_the_run(build, clock):
    node = build(
        params={"countdown": 0.0}, steps=[{"label": "a", "hold": 1.0, "velocity": 1.0}]
    )
    start_running(node, clock)

    clock.advance(1.0)
    node._tick_running(clock())

    assert node.phase == node.STOPPING
    assert node.result == "ok"
    assert node.command == (0.0, 0.0)


# ------------------------------------------------------------------ RETURNING


def test_return_to_start_reverses_until_back_near_the_line(build, clock):
    node = build(
        params={"countdown": 0.0},
        steps=[{"label": "a", "hold": 1.0, "velocity": 1.0}],
        **{
            "return": {
                "mode": "reverse_to_start",
                "speed": 1.2,
                "tolerance": 1.0,
                "timeout": 90.0,
            }
        },
    )
    start_running(node, clock)  # origin captured at (0, 0, 0)

    clock.advance(1.0)
    node._tick_running(
        clock()
    )  # flips to RETURNING; the reverse command lands next tick
    assert node.phase == node.RETURNING

    feed(node, odom_msg=odom(5.0, 0.0, 0.0))  # still 5 m out
    node._tick_returning(clock())
    assert node.phase == node.RETURNING
    assert node.command == (0.0, -1.2)

    feed(node, odom_msg=odom(0.5, 0.0, 0.0))  # within the 1 m tolerance
    node._tick_returning(clock())
    assert node.phase == node.STOPPING
    assert "returned to within" in node.result_detail


def test_return_leg_times_out_and_stops_where_it_is(build, clock):
    node = build(
        params={"countdown": 0.0},
        steps=[{"label": "a", "hold": 1.0, "velocity": 1.0}],
        **{"return": {"mode": "reverse_to_start", "timeout": 5.0}},
    )
    start_running(node, clock)

    clock.advance(1.0)
    node._tick_running(clock())
    assert node.phase == node.RETURNING

    feed(node, odom_msg=odom(5.0, 0.0, 0.0))  # never gets back
    clock.advance(5.1)
    node._tick_returning(clock())

    assert node.phase == node.STOPPING
    assert "timed out" in node.result_detail


# ------------------------------------------------------------------- STOPPING


def test_stopping_holds_auto_ready_briefly_then_finishes(build, clock):
    node = build(
        params={"countdown": 0.0}, steps=[{"label": "a", "hold": 1.0, "velocity": 1.0}]
    )
    start_running(node, clock)
    clock.advance(1.0)
    node._tick_running(clock())
    assert node.phase == node.STOPPING

    node._tick_stopping(clock())
    assert node.auto_ready  # a commanded stop, not a watchdog timeout
    assert node.phase == node.STOPPING

    clock.advance(1.1)
    node._tick_stopping(clock())
    assert not node.auto_ready
    assert node.phase == node.STOPPING

    clock.advance(1.0)
    node._tick_stopping(clock())
    assert node.phase == node.FINISHED
    assert node.done()


# ---------------------------------------------------------------- full chain


def test_the_full_chain_from_link_up_to_running(build, clock):
    """The transitions compose, driven through the real `_tick` dispatcher."""
    node = build(params={"countdown": 0.0})

    feed(node, status_msg=status())
    node._tick()
    assert node.phase == node.WAIT_ESTOP_ASSERTED

    feed(node, status_msg=status(estop=True, mode=ArduinoStatus.MODE_ESTOP))
    node._tick()
    assert node.phase == node.WAIT_ESTOP_CLEARED

    feed(
        node,
        status_msg=status(
            estop=False, auto_arm=True, mode=ArduinoStatus.MODE_AUTO_ARMED
        ),
    )
    node._tick()
    assert node.phase == node.ARMING

    feed(node, status_msg=status(), odom_msg=odom())
    node._tick()  # requests gains
    node._tick()  # settles and starts

    assert node.phase == node.RUNNING
