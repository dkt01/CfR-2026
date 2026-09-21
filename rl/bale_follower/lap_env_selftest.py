#!/usr/bin/env python3
"""Drive `LapRacerEnv` without Gazebo, ROS, or a GPU.

A training run costs five hours and a simulator that dies at the end of them,
so the bookkeeping this environment does -- arc-length projection, lap
counting, the recovery state machine, the steering integrator, the shape and
bounds of the observation -- should be wrong on a laptop, in two seconds,
rather than in a log at 3 a.m.

ROS and Gymnasium are stubbed (they live in the sim container, not here) and
Gazebo is replaced by the same kinematic bicycle the env commands, integrated
at the control rate. That makes this a test of the environment's LOGIC, not
of the vehicle: contact physics, tire slip, sensor timing and the real-time
factor are all out of scope and belong to the run itself. What it does prove
is that a lap is detected once per lap, that a wedged car enters recovery and
leaves it by reversing, that reward terms fire where they should, and that
nothing raises.

    python3 lap_env_selftest.py
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------- ROS stubs


def _install_stubs() -> None:
    class _Publisher:
        def __init__(self) -> None:
            self.last = None

        def publish(self, message) -> None:
            self.last = message

    class _Node:
        def __init__(self, name: str) -> None:
            self.name = name

        def create_publisher(self, *_args, **_kwargs):
            return _Publisher()

        def create_subscription(self, *_args, **_kwargs):
            return None

        def destroy_node(self) -> None:
            return None

    class _Executor:
        def add_node(self, _node) -> None:
            return None

        def spin(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.init = lambda **_kwargs: None
    rclpy.shutdown = lambda: None
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = _Node
    rclpy.executors = types.ModuleType("rclpy.executors")
    rclpy.executors.SingleThreadedExecutor = _Executor
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.qos.QoSProfile = lambda depth=1: types.SimpleNamespace(depth=depth,
                                                                 reliability=None)
    rclpy.qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1)

    def _message(name):
        module = types.ModuleType(name)
        for attribute in ("Twist", "TFMessage", "PointCloud2", "Clock",
                          "ArduinoStatus"):
            setattr(module, attribute, type(attribute, (), {
                "__init__": lambda self: setattr(self, "linear",
                                                 types.SimpleNamespace(x=0.0))
                or setattr(self, "angular", types.SimpleNamespace(z=0.0)),
            }))
        return module

    spaces = types.ModuleType("gymnasium.spaces")

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

        def contains(self, value):
            return (value.shape == self.shape and float(value.min()) >= self.low
                    and float(value.max()) <= self.high)

    spaces.Box = _Box
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.spaces = spaces

    class _Env:
        metadata: dict = {}

        def reset(self, *, seed=None, options=None):
            return None

    gymnasium.Env = _Env

    for name, module in {
        "rclpy": rclpy,
        "rclpy.node": rclpy.node,
        "rclpy.executors": rclpy.executors,
        "rclpy.qos": rclpy.qos,
        "geometry_msgs": types.ModuleType("geometry_msgs"),
        "geometry_msgs.msg": _message("geometry_msgs.msg"),
        "tf2_msgs": types.ModuleType("tf2_msgs"),
        "tf2_msgs.msg": _message("tf2_msgs.msg"),
        "sensor_msgs": types.ModuleType("sensor_msgs"),
        "sensor_msgs.msg": _message("sensor_msgs.msg"),
        "rosgraph_msgs": types.ModuleType("rosgraph_msgs"),
        "rosgraph_msgs.msg": _message("rosgraph_msgs.msg"),
        "cfr_interfaces": types.ModuleType("cfr_interfaces"),
        "cfr_interfaces.msg": _message("cfr_interfaces.msg"),
        "gymnasium": gymnasium,
        "gymnasium.spaces": spaces,
    }.items():
        sys.modules.setdefault(name, module)


_install_stubs()

import yaml  # noqa: E402

import bale_geometry  # noqa: E402
import env as env_module  # noqa: E402
from env import Pose2D, MAX_STEERING_ANGLE, WHEELBASE  # noqa: E402
from lap_driver import PursuitDriver  # noqa: E402
from lap_env import LapRacerEnv  # noqa: E402
from lap_reward import LapRewardConfig  # noqa: E402
from zed_sim import ZedSimConfig  # noqa: E402

env_module._unpause_world = lambda *_args, **_kwargs: True


# ------------------------------------------------------- the fake simulator


@dataclass
class FakeCar:
    """Kinematic bicycle, integrated at the control rate.

    Deliberately the same model the env uses to convert (speed, steering)
    into a twist, so this tests bookkeeping rather than physics -- with one
    exception that matters for the recovery test: the car cannot drive
    through a bale. A command that would put the footprint inside one leaves
    it where it was, which is what being wedged feels like.
    """

    bales: list
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    sim_time: float = 0.0

    def step(self, speed: float, steer_fraction: float, dt: float) -> None:
        delta = steer_fraction * MAX_STEERING_ANGLE
        x = self.x + speed * math.cos(self.yaw) * dt
        y = self.y + speed * math.sin(self.yaw) * dt
        yaw = self.yaw + (speed / WHEELBASE) * math.tan(delta) * dt
        self.sim_time += dt
        if not bale_geometry.check_collision(self.bales, x, y, yaw):
            self.x, self.y, self.yaw = x, y, yaw

    def pose(self) -> Pose2D:
        return Pose2D(x=self.x, y=self.y, yaw=self.yaw, stamp=self.sim_time,
                      sim_stamp=self.sim_time)


def make_env(config: dict, sdf_path: str, **overrides) -> tuple[LapRacerEnv, FakeCar]:
    # No ArduinoStatus publisher without the ROS stack, so the policy's speed
    # channel falls back to the true one here; the tachometer model is
    # exercised against the live simulator instead.
    env_config = {**config["env"], "speed_source": "truth", **overrides}
    bales = bale_geometry.parse_bales(sdf_path)
    car = FakeCar(bales=bales)
    dt = 1.0 / env_config["control_hz"]

    # Gazebo, the teleport service and the wall clock, replaced.
    def teleport(self, x, y, heading_deg, attempts=3):
        car.x, car.y, car.yaw = x, y, math.radians(heading_deg)

    def advance(self):
        twist = self._cmd_pub.last
        speed = twist.linear.x if twist is not None else 0.0
        car.step(speed, self._cmd_steer_fraction, dt)

    LapRacerEnv._teleport = teleport
    LapRacerEnv._advance_sim = advance
    LapRacerEnv._settle = lambda self, seconds: None
    LapRacerEnv._wait_for_pose = lambda self, since: car.pose()

    environment = LapRacerEnv(
        sdf_path=sdf_path,
        lap_reward_config=LapRewardConfig(**config["reward"]),
        zed_config=ZedSimConfig(**config.get("zed_sim", {})),
        scan_source="analytic",  # no point cloud without Gazebo
        **{k: v for k, v in env_config.items() if k != "scan_source"},
    )
    return environment, car


# ------------------------------------------------------------- the driver
# (PursuitDriver lives in lap_driver.py, shared with lap_live_check.py)


def pure_pursuit(environment: LapRacerEnv, car: FakeCar, lookahead: float,
                 target_speed: float) -> np.ndarray:
    """The shared scripted driver, so this test and the live one agree."""
    return PursuitDriver(environment, lookahead, target_speed).action(
        car.x, car.y, car.yaw)


def main() -> None:
    from pathlib import Path

    here = Path(__file__).resolve().parent
    config = yaml.safe_load((here / "config_lap.yaml").read_text())
    sdf = str(Path(here).parents[1] / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf")

    # ------------------------------------------------- 0: body clearance
    # The measurement the whole "never touch a bale" side of the reward rests
    # on, against a single synthetic bale where the right answer is known.
    bale = bale_geometry.Bale(index=0, x=0.0, y=0.0, yaw=0.0,
                              half_x=0.4572, half_y=0.2286)
    for gap in (0.0, 0.1, 0.25, 0.5):
        nose = bale.half_x + bale_geometry.CHASSIS_LENGTH / 2 + gap
        measured = bale_geometry.body_clearance([bale], nose, 0.0, 0.0)
        assert abs(measured - gap) < 1e-6, (gap, measured)
    # A bale 8 cm off the flank: invisible to the forward fan, which reads it
    # as 0.28 m of "clearance" from the car's centre. That gap is why the lap
    # reward does not use the scan for contact.
    flank = bale.half_y + bale_geometry.CHASSIS_WIDTH / 2 + 0.08
    body = bale_geometry.body_clearance([bale], 0.0, flank, 0.0)
    scan_min = float(bale_geometry.lidar_scan(
        [bale], 0.0, flank, 0.0, 36, 110.0, 6.0).min())
    assert abs(body - 0.08) < 1e-6 and scan_min > 0.25, (body, scan_min)
    print(f"body clearance: 8 cm off the flank reads {body:.3f} m, "
          f"where the forward scan reads {scan_min:.3f} m")

    # ---------------------------------------------------- 1: a flying lap
    environment, car = make_env(config, sdf, randomize_start=False,
                                wedged_start_prob=0.0, episode_time_limit_s=120.0,
                                max_laps=1)
    observation, _ = environment.reset()
    assert environment.observation_space.contains(observation), observation.shape
    assert observation.shape == (config["env"]["num_lidar_bins"]
                                 * (1 + config["env"]["scan_history"]) + 4,)

    totals: dict[str, float] = {}
    reward_total = 0.0
    rates: list[float] = []
    clearances: list[float] = []
    scans: list[np.ndarray] = []
    steps = 0
    info: dict = {}
    while True:
        action = pure_pursuit(environment, car, lookahead=1.2, target_speed=3.2)
        observation, reward, terminated, truncated, info = environment.step(action)
        assert environment.observation_space.contains(observation)
        reward_total += reward
        for key, value in info["reward_terms"].items():
            totals[key] = totals.get(key, 0.0) + value
        rates.append(info["steer_rate"])
        clearances.append(info["min_clearance"])
        scans.append(observation[:config["env"]["num_lidar_bins"]].copy())
        steps += 1
        if terminated or truncated:
            break

    print(f"lap run: {steps} steps, {info['laps']} lap(s) in "
          f"{environment._episode_time:.1f} s sim, "
          f"{info['s_progress']:.1f} m of arc, reward {reward_total:.0f}")
    print(f"  lap times {info['lap_times']}, min body clearance "
          f"{min(clearances):.3f} m, mean servo rate "
          f"{sum(rates) / len(rates):.2f} rad/s")
    print("  reward by term: " + ", ".join(
        f"{k} {v:+.0f}" for k, v in sorted(totals.items())))

    # The camera is slower than the control loop, so the policy must be
    # seeing a repeated scan on some steps -- roughly 1 - 15/20 of them.
    repeated = sum(
        1 for a, b in zip(scans, scans[1:]) if np.array_equal(a, b)
    ) / max(1, len(scans) - 1)
    expected = max(0.0, 1.0 - config["env"]["scan_update_hz"]
                   / config["env"]["control_hz"])
    print(f"  scan held between camera frames on {repeated:.0%} of steps "
          f"(camera {config['env']['scan_update_hz']:.0f} Hz against a "
          f"{config['env']['control_hz']:.0f} Hz loop, expected ~{expected:.0%})")
    assert abs(repeated - expected) < 0.12, (
        f"scan repeated on {repeated:.0%} of steps, expected ~{expected:.0%}: "
        "the camera-rate hold is not doing what it claims")

    assert info["laps"] == 1, f"scripted driver did not complete a lap: {info}"
    assert not info["collided"], "scripted driver hit a bale"
    assert totals["lap_bonus"] > 0.0, "a completed lap paid no bonus"
    assert totals["progress"] > 600.0, totals["progress"]
    # The clock runs all lap; the only stalled step should be the standstill
    # the episode starts from.
    assert totals["time"] < 0.0, totals
    assert totals["stall"] > -1.0, ("stalled mid-lap", totals)
    assert min(clearances) > 0.0, "clearance must be positive on a clean lap"

    # A second lap of the same driving must score close to the first: if it
    # does not, the arc-length bookkeeping is drifting round the loop.
    environment.max_laps = 2
    observation, _ = environment.reset()
    first = second = None
    while True:
        action = pure_pursuit(environment, car, lookahead=1.2, target_speed=3.2)
        observation, _, terminated, truncated, info = environment.step(action)
        if info["lap_time"] is not None:
            first, second = (info["lap_time"], second) if first is None else \
                (first, info["lap_time"])
        if terminated or truncated:
            break
    assert first and second and abs(first - second) < 1.5, (first, second)
    print(f"  two laps: {first:.2f} s, {second:.2f} s")

    # ------------------------------------------------ 2: wedged, and out
    # Started from the env's own wedged spawn -- angled across the corridor
    # with a bale close in front -- which tests the sampler at the same time.
    environment.close()

    def wedged_env():
        """A fresh env, parked in a wedge with the recovery flag up."""
        environment, car = make_env(config, sdf, randomize_start=True,
                                    wedged_start_prob=1.0,
                                    episode_time_limit_s=30.0)
        environment.reset(seed=7)
        for step in range(120):
            _, _, terminated, truncated, info = environment.step(
                np.array([1.0, 0.0], dtype=np.float32))
            if info["recovering"]:
                return environment, car, step + 1
            if terminated or truncated:
                break
        raise AssertionError(
            "a car that cannot move never entered recovery "
            f"(collided={info.get('collided')}, stuck={info.get('stuck')})")

    def drive(environment, action, steps):
        total = 0.0
        for _ in range(steps):
            _, reward, terminated, truncated, _ = environment.step(action)
            total += reward
            if terminated or truncated:
                break
        return total

    environment, car, entered_at = wedged_env()
    print(f"  wedged spawn: heading error {environment._prev_heading_error:+.2f} "
          f"rad, clearance "
          f"{bale_geometry.body_clearance(car.bales, car.x, car.y, car.yaw):.3f} m; "
          f"recovery entered after {entered_at} steps of no progress")

    # Reversing inverts the steering: yaw rate is (v/L)*tan(delta), so with
    # v < 0 the lock that closes the heading error is the one that SIGNS WITH
    # it. That is the manoeuvre the reward has to pay for, measured against
    # the alternative the policy is actually choosing between: sitting there.
    before = environment._prev_heading_error
    lock = math.copysign(1.0, before)
    backing_out = drive(environment, np.array([-1.0, lock], dtype=np.float32), 30)
    turned = environment._prev_heading_error

    idle_env, _, _ = wedged_env()
    idle = drive(idle_env, idle_env.encode_action(0.0, 0.0), 30)
    idle_env.close()

    print(f"  reversing out: heading error {before:+.2f} -> {turned:+.2f} rad, "
          f"reward {backing_out:+.1f} against {idle:+.1f} for sitting still")
    assert abs(turned) < abs(before), (
        "reversing with lock on did not turn the car back down the course: "
        f"{before:+.2f} -> {turned:+.2f}")
    assert backing_out > idle, (
        "backing out must pay better than sitting in the wedge, or the "
        "policy has no reason to try")

    # And driving away again has to clear the flag, after
    # `recovery_exit_distance` of forward travel. The car is put back on the
    # line first: the stand-in physics here have no sliding contact (a
    # command into a bale simply does not move the car), so a stub car that
    # clips a wall at 0.14 rad stops dead where a real one would scrub past,
    # and the state machine would never get its metre. Gazebo is where that
    # part gets tested.
    car.x, car.y, car.yaw = environment.track.pose_at(environment._path_index)
    environment._prev_pose = car.pose()
    left_recovery = False
    for _ in range(120):
        action = pure_pursuit(environment, car, lookahead=1.2, target_speed=2.0)
        _, _, terminated, truncated, info = environment.step(action)
        if not info["recovering"]:
            left_recovery = True
            break
        if terminated or truncated:
            break
    assert left_recovery, "recovery never ended despite the car moving again"
    print("  recovery cleared once the car drove away")

    # ------------------------------------------- 3: the steering integrator
    environment.reset()
    environment._cmd_steer_fraction = 0.0
    per_step = environment.steer_rate_fraction_per_s / environment.control_hz
    environment.step(np.array([0.0, 1.0], dtype=np.float32))
    assert abs(environment._cmd_steer_fraction - per_step) < 1e-6, (
        environment._cmd_steer_fraction, per_step)
    lock_steps = math.ceil(1.0 / per_step)
    for _ in range(lock_steps):
        environment.step(np.array([0.0, 1.0], dtype=np.float32))
    assert environment._cmd_steer_fraction == 1.0
    print(f"  steering integrator: {per_step:.3f} of full lock per step, "
          f"lock to lock in {2 * lock_steps / environment.control_hz:.2f} s")
    environment.close()

    print("\nself-test passed")


if __name__ == "__main__":
    main()
