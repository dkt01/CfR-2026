"""Tests for hoop pass/fail detection.

Synthetic traces, built against hoop_0's real numbers from
obstacle_course_layout.yaml (yaw 1.57080, nominal [-3.9405, -1.7118]), so
these run with no simulator and no ROS. What they cannot check is the yaw
convention itself -- whether the uprights really do sit on the stored yaw's
axis in the running sim, as hoop_monitor.py's docstring assumes -- only
driving the course does that.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

MODULE = Path(__file__).parents[1] / "src" / "hoop_monitor.py"
_spec = importlib.util.spec_from_file_location("hoop_monitor", MODULE)
hoop_monitor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hoop_monitor
_spec.loader.exec_module(hoop_monitor)

Pose2D = hoop_monitor.Pose2D
Hoop = hoop_monitor.Hoop
HoopMonitor = hoop_monitor.HoopMonitor

HOOP_0 = Hoop(name="hoop_0", x=-3.9405, y=-1.7118, yaw=1.57080, half_width=0.275)


def line(start, end, step=0.1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    span = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    count = max(int(span / step), 1)
    return [
        Pose2D(start[0] + dx * i / count, start[1] + dy * i / count, yaw)
        for i in range(count + 1)
    ]


def feed(monitor: HoopMonitor, poses: list[Pose2D]):
    for pose in poses:
        monitor.update(pose)


# The gate's travel axis is yaw + 90 deg = pi, i.e. world -x. Approaching
# from +x and driving through the center should score a pass.


def test_center_line_passes():
    monitor = HoopMonitor([HOOP_0])
    feed(monitor, line((HOOP_0.x + 2.0, HOOP_0.y), (HOOP_0.x - 2.0, HOOP_0.y)))
    assert monitor.state["hoop_0"].passed
    assert not monitor.state["hoop_0"].missed
    assert monitor.all_passed
    assert not monitor.any_missed


def test_off_to_the_side_misses():
    monitor = HoopMonitor([HOOP_0])
    wide_y = HOOP_0.y + 1.0  # well outside the 0.275 m gate half-width
    feed(monitor, line((HOOP_0.x + 2.0, wide_y), (HOOP_0.x - 2.0, wide_y)))
    assert monitor.state["hoop_0"].missed
    assert not monitor.state["hoop_0"].passed
    assert monitor.any_missed
    assert not monitor.all_passed


def test_just_inside_gate_passes():
    monitor = HoopMonitor([HOOP_0])
    edge_y = HOOP_0.y + HOOP_0.half_width - 0.02
    feed(monitor, line((HOOP_0.x + 2.0, edge_y), (HOOP_0.x - 2.0, edge_y)))
    assert monitor.state["hoop_0"].passed


def test_just_outside_gate_misses():
    monitor = HoopMonitor([HOOP_0])
    edge_y = HOOP_0.y + HOOP_0.half_width + 0.02
    feed(monitor, line((HOOP_0.x + 2.0, edge_y), (HOOP_0.x - 2.0, edge_y)))
    assert monitor.state["hoop_0"].missed


def test_reverse_direction_still_passes():
    """The rules only ask that the car pass through -- not which way."""
    monitor = HoopMonitor([HOOP_0])
    feed(monitor, line((HOOP_0.x - 2.0, HOOP_0.y), (HOOP_0.x + 2.0, HOOP_0.y)))
    assert monitor.state["hoop_0"].passed


def test_never_reaching_the_plane_stays_unresolved():
    monitor = HoopMonitor([HOOP_0])
    feed(monitor, line((HOOP_0.x + 2.0, HOOP_0.y), (HOOP_0.x + 0.5, HOOP_0.y)))
    assert not monitor.state["hoop_0"].passed
    assert not monitor.state["hoop_0"].missed
    assert not monitor.any_missed
    assert not monitor.all_passed


def test_once_resolved_stays_resolved():
    """A hoop that has been missed cannot be rescued by driving back through it."""
    monitor = HoopMonitor([HOOP_0])
    wide_y = HOOP_0.y + 1.0
    feed(monitor, line((HOOP_0.x + 2.0, wide_y), (HOOP_0.x - 2.0, wide_y)))
    assert monitor.state["hoop_0"].missed
    feed(monitor, line((HOOP_0.x - 2.0, HOOP_0.y), (HOOP_0.x + 2.0, HOOP_0.y)))
    assert monitor.state["hoop_0"].missed
    assert not monitor.state["hoop_0"].passed


def test_multiple_hoops_track_independently():
    hoop_1 = Hoop(name="hoop_1", x=-6.3703, y=-2.4049, yaw=1.57080, half_width=0.275)
    monitor = HoopMonitor([HOOP_0, hoop_1])
    feed(monitor, line((HOOP_0.x + 2.0, HOOP_0.y), (HOOP_0.x - 2.0, HOOP_0.y)))
    assert monitor.state["hoop_0"].passed
    assert not monitor.state["hoop_1"].resolved
    assert not monitor.any_missed  # hoop_1 unresolved, not missed
    assert not monitor.all_passed


def test_crossing_the_plane_far_from_the_hoop_resolves_nothing():
    """The plane through a hoop is infinite; the course is not.

    Driving somewhere else entirely that happens to lie on the same plane
    is not an attempt at this hoop, so it must resolve neither way.
    """
    monitor = HoopMonitor([HOOP_0])
    far_y = HOOP_0.y - 7.9  # the gravel-run-to-bank turn, see hoop_monitor
    feed(monitor, line((HOOP_0.x + 2.0, far_y), (HOOP_0.x - 2.0, far_y)))
    assert not monitor.state["hoop_0"].resolved
    assert not monitor.any_missed


def test_the_real_lane_does_not_trip_hoop_0():
    """Regression: the actual course centerline through the gravel run.

    These are the anchors obstacle_course_path.py drives between, and they
    cross hoop_0's plane 7.9 m south of it. Before the attempt gate this
    marked hoop_0 missed and made a clean lap impossible.
    """
    monitor = HoopMonitor([HOOP_0])
    feed(monitor, line((-2.60, -9.88), (-4.15, -9.30)))
    assert not monitor.state["hoop_0"].resolved


def test_the_real_lane_does_not_trip_hoop_2():
    """Regression: the car wash run back to the finish line.

    hoop_2 sits at yaw 0, so its plane is y = 0.1489, which the start
    straight crosses about 7.2 m east of the hoop.
    """
    hoop_2 = Hoop(name="hoop_2", x=-8.1350, y=0.1489, yaw=0.0, half_width=0.275)
    monitor = HoopMonitor([hoop_2])
    feed(monitor, line((-1.65, 0.18), (0.00, 0.10)))
    assert not monitor.state["hoop_2"].resolved


def test_swerving_around_a_hoop_still_misses():
    """The gate must not become a way to skip hoops for free.

    Just outside the opening, and a metre out, both still count as missed --
    only crossings far enough away to be unrelated are ignored.
    """
    for offset in (0.5, 1.0, 1.4):
        monitor = HoopMonitor([HOOP_0])
        wide_y = HOOP_0.y + offset
        feed(monitor, line((HOOP_0.x + 2.0, wide_y), (HOOP_0.x - 2.0, wide_y)))
        assert monitor.state["hoop_0"].missed, f"{offset} m out should be a miss"


def test_update_hoop_repositions_without_losing_yaw():
    monitor = HoopMonitor([HOOP_0])
    monitor.update_hoop("hoop_0", x=HOOP_0.x - 1.0, y=HOOP_0.y)
    moved = monitor.hoops["hoop_0"]
    assert moved.yaw == HOOP_0.yaw
    assert moved.half_width == HOOP_0.half_width
    feed(monitor, line((moved.x + 2.0, moved.y), (moved.x - 2.0, moved.y)))
    assert monitor.state["hoop_0"].passed
