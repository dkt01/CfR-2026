#!/usr/bin/env python3
"""Run a trained bale-following policy against the simulation.

Drop-in alternative to path_follower_node: subscribes to /zed/zed_node/odom,
computes the same analytic lidar observation used during training, and
publishes geometry_msgs/Twist on /cmd_vel.

Run against the normal (unpaused) simulation:

    ros2 launch cfr_arduino_bridge simulation.launch.py
    python run_policy.py --checkpoint checkpoints/final_model.zip
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from stable_baselines3 import PPO
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

import bale_geometry
from env import (
    MAX_STEERING_ANGLE,
    WHEELBASE,
    _unpause_world,
    _wrap_to_pi,
    _yaw_from_quaternion,
)

if TYPE_CHECKING:
    # casadi is only needed for the --smoother legacy path (v5 and earlier
    # checkpoints); this import never runs, it only gives the CommandSmoother
    # annotation below something real to resolve against.
    from casadi_smoother import CommandSmoother

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"


class PolicyRunner(Node):
    def __init__(
        self,
        model: PPO,
        bales: list,
        env_config: dict,
        world_name: str,
        smoother: CommandSmoother | None = None,
        require_start_signal: bool = True,
        go_topic: str = "/start_signal_detector/go",
        done_topic: str = "/lap_counter/done",
    ) -> None:
        super().__init__("bale_follower_policy")
        self.model = model
        self.bales = bales
        self.smoother = smoother

        # Gating on the same latched signals the rest of the stack uses
        # (start_signal_detector's ~/go, lap_counter's ~/done), so the
        # policy waits behind the line instead of driving the instant it is
        # launched, and stops once the lap counter says the run is over --
        # neither of which the policy itself has any notion of.
        self.armed = not require_start_signal
        self.finished = False
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        if require_start_signal:
            self.create_subscription(Bool, go_topic, self._on_go, latched)
        self.create_subscription(Bool, done_topic, self._on_done, latched)
        self.num_lidar_bins = env_config["num_lidar_bins"]
        self.lidar_fov_deg = env_config["lidar_fov_deg"]
        self.lidar_max_range = env_config["lidar_max_range"]
        self.max_speed = env_config["max_speed"]
        self.reverse_speed = env_config.get("reverse_speed", 0.0)
        self.control_hz = env_config["control_hz"]

        self.obs_speed_scale = self.max_speed

        self._lock = threading.Lock()
        self._pose = None
        self._prev_pose = None

        # Stuck recovery: if the car has barely moved for stuck_window_s
        # (typically nosed against a bale in a hairpin, where a forward-only
        # policy can never free itself), reverse briefly while steering
        # toward the obstacle side -- backing up with the wheels turned
        # toward the wall swings the nose away from it -- then hand control
        # back to the policy.
        self.recovery_enabled = True
        self.stuck_window_s = 1.5
        self.stuck_distance = 0.12
        self.recovery_duration_s = 1.2
        self.recovery_cooldown_s = 2.0
        self.recovery_reverse_speed = max(self.reverse_speed, 0.6)
        self._pose_history: list[tuple[float, float, float]] = []
        self._recovery_until = 0.0
        self._recovery_delta = 0.0
        self._recovery_start = 0.0
        self._no_trigger_until = 0.0
        # Escalation keys off actual displacement, not wall-clock spacing
        # between triggers: a car ping-ponging out of a pocket and straight
        # back into it re-triggers on a roughly constant cycle, so a
        # time-since-last-recovery gate (the previous approach) never sees
        # a long-enough gap to reset and never climbs past 2x either --
        # measured on the speed course's first hairpin: real trigger gaps of
        # 4.6-12.1 s straddled a 6 s cutoff randomly, escalation stayed
        # between 1x-2x for over 90 s straight and the car never broke free.
        # Tracking net progress since the last recovery began answers the
        # actual question ("did that reverse help?") instead of guessing at
        # it from timing.
        # A small wiggle inside the same pocket can clear the raw 0.12 m
        # stuck_distance gate without netting real progress -- measured on
        # the speed course's first turn: escalation reset to 0 after a
        # partial reverse, then re-triggered within 4 s at the same spot.
        # Half a car length is a better bar for "that recovery worked."
        self.min_recovery_progress = 0.6
        self.max_escalation_count = 5  # 2**5 = 32x base duration, ~38 s reverse
        self._recovery_anchor: tuple[float, float] | None = None
        self._escalation_count = 0
        self._escalation = 1.0
        # Phase 1 (straight reverse, no steer) clears the pocket the nose is
        # wedged into before phase 2 points the car away -- steering while
        # still jammed against the obstacle just re-noses into the same spot,
        # which is what a fixed nearest-scan bearing does when the car is
        # stuck symmetrically (a corner or a pocket, not a single wall). Only
        # engaged once plain escalation has already failed twice.
        self.straight_phase_fraction = 0.4

        # Same ground-truth pose source as training -- the Ackermann plugin's
        # odometry is dead-reckoned and drifts from the true pose.
        self.create_subscription(
            TFMessage, f"/world/{world_name}/dynamic_pose/info", self._on_pose, 10
        )
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / self.control_hz, self._on_control_tick)
        if self.armed:
            self.get_logger().info("bale-following policy ready, waiting for pose")
        else:
            self.get_logger().info(
                f"bale-following policy ready, waiting for pose and start signal on {go_topic}"
            )

    def _on_go(self, message: Bool) -> None:
        if message.data and not self.armed:
            self.get_logger().info("start signal latched; driving")
        self.armed = self.armed or message.data

    def _on_done(self, message: Bool) -> None:
        if message.data and not self.finished:
            self.get_logger().info("lap counter done; stopping")
        self.finished = message.data

    def _on_pose(self, msg: TFMessage) -> None:
        if not msg.transforms:
            return
        t = msg.transforms[0].transform
        q = t.rotation
        with self._lock:
            self._pose = (
                t.translation.x,
                t.translation.y,
                _yaw_from_quaternion(q.x, q.y, q.z, q.w),
            )

    def _check_stuck(self, now: float, x: float, y: float) -> bool:
        self._pose_history.append((now, x, y))
        while (
            self._pose_history
            and now - self._pose_history[0][0] > self.stuck_window_s + 1.0
        ):
            self._pose_history.pop(0)
        if not self.recovery_enabled or now < self._no_trigger_until:
            return False
        oldest = self._pose_history[0]
        if now - oldest[0] < self.stuck_window_s:
            return False
        return math.hypot(x - oldest[1], y - oldest[2]) < self.stuck_distance

    def _start_recovery(self, now: float, x: float, y: float, scan: np.ndarray) -> None:
        if self._recovery_anchor is not None:
            progress = math.hypot(
                x - self._recovery_anchor[0], y - self._recovery_anchor[1]
            )
        else:
            progress = math.inf
        if progress < self.min_recovery_progress:
            self._escalation_count = min(
                self._escalation_count + 1, self.max_escalation_count
            )
        else:
            self._escalation_count = 0
        self._recovery_anchor = (x, y)
        self._escalation = 2.0**self._escalation_count

        if self._escalation_count >= 2:
            # Plain escalation already failed twice: alternate the escape
            # side deterministically instead of trusting the nearest-scan
            # bearing, which is exactly the signal that kept picking a side
            # that walked the car straight back into the same pocket.
            sign = 1.0 if self._escalation_count % 2 == 0 else -1.0
        else:
            # Steer toward the side the nearest obstacle is on: in reverse
            # the nose swings away from the wheels' direction. Bin 0 is the
            # rightmost ray, positive bearings are left.
            nearest = int(np.argmin(scan))
            half_fov = math.radians(self.lidar_fov_deg) / 2.0
            bearing = -half_fov + nearest * 2.0 * half_fov / max(
                1, self.num_lidar_bins - 1
            )
            sign = math.copysign(1.0, bearing if abs(bearing) > 1e-3 else 1.0)
        self._recovery_delta = sign * MAX_STEERING_ANGLE

        duration = self.recovery_duration_s * self._escalation
        self._recovery_start = now
        self._recovery_until = now + duration
        phase = (
            f"{self.straight_phase_fraction * duration:.1f}s straight + "
            f"{(1 - self.straight_phase_fraction) * duration:.1f}s steered"
            if self._escalation_count >= 2
            else f"{duration:.1f}s steered"
        )
        self.get_logger().info(
            f"stuck (moved <{self.stuck_distance} m in {self.stuck_window_s} s, "
            f"escalation {self._escalation_count}); reversing {phase}, "
            f"steer {self._recovery_delta:+.2f} rad"
        )

    def _publish_recovery(self, now: float) -> None:
        twist = Twist()
        twist.linear.x = -self.recovery_reverse_speed
        duration = self._recovery_until - self._recovery_start
        elapsed = now - self._recovery_start
        straight_phase = (
            self._escalation_count >= 2
            and elapsed < self.straight_phase_fraction * duration
        )
        delta = 0.0 if straight_phase else self._recovery_delta
        twist.angular.z = (twist.linear.x / WHEELBASE) * math.tan(delta)
        self._cmd_pub.publish(twist)

    def _on_control_tick(self) -> None:
        with self._lock:
            pose = self._pose
        if pose is None:
            return

        if self.finished:
            self._cmd_pub.publish(Twist())
            return
        if not self.armed:
            self._cmd_pub.publish(Twist())
            # Don't accumulate stuck/speed history while held at the line --
            # it would read as "hasn't moved in stuck_window_s" the instant
            # the signal goes green.
            self._prev_pose = pose
            self._pose_history.clear()
            return

        x, y, yaw = pose
        dt = 1.0 / self.control_hz
        if self._prev_pose is None:
            linear_x, angular_z = 0.0, 0.0
        else:
            px, py, pyaw = self._prev_pose
            linear_x = math.hypot(x - px, y - py) / dt
            angular_z = _wrap_to_pi(yaw - pyaw) / dt
        self._prev_pose = pose
        scan = bale_geometry.lidar_scan(
            self.bales,
            x,
            y,
            yaw,
            self.num_lidar_bins,
            self.lidar_fov_deg,
            self.lidar_max_range,
        )

        now = time.monotonic()
        if now < self._recovery_until:
            self._publish_recovery(now)
            return
        if (
            self._recovery_until
            and now - self._recovery_until < 1.0 / self.control_hz + 0.1
        ):
            # Recovery just ended: forget the stuck history and the smoother's
            # reversed state, and give the policy a grace period to move off.
            self._pose_history.clear()
            self._no_trigger_until = now + self.recovery_cooldown_s
            if self.smoother is not None:
                self.smoother.reset()
        if self._check_stuck(now, x, y):
            self._start_recovery(now, x, y, scan)
            self._publish_recovery(now)
            return

        observation = np.concatenate(
            [
                (scan / self.lidar_max_range),
                [
                    np.clip(linear_x / self.obs_speed_scale, 0.0, 1.0),
                    np.clip((angular_z + 1.0) / 2.0, 0.0, 1.0),
                ],
            ]
        ).astype(np.float32)

        action, _ = self.model.predict(observation, deterministic=True)
        # Same normalized-action decoding the env trains with, including the
        # reverse range when the checkpoint was trained with one.
        fraction = (float(np.clip(action[0], -1.0, 1.0)) + 1.0) / 2.0
        speed = -self.reverse_speed + fraction * (self.max_speed + self.reverse_speed)
        steer_fraction = float(np.clip(action[1], -1.0, 1.0))
        delta = steer_fraction * MAX_STEERING_ANGLE

        if self.smoother is not None:
            speed, delta = self.smoother.smooth(speed, delta)

        twist = Twist()
        twist.linear.x = speed
        if abs(speed) > 1e-3:
            twist.angular.z = (speed / WHEELBASE) * math.tan(delta)
        self._cmd_pub.publish(twist)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    parser.add_argument("--world-name", default="cfr_speed_course")
    # Off by default. env.py enforces the servo slew, the traction clamp and
    # the friction circle during training, so a v6-or-later policy's raw
    # commands are already executable -- measured 3.2 rad/s of steering
    # against the servo's 3.5 rad/s limit. The smoother then only adds its
    # own lag: it brakes predictively across a 0.8 s horizon where training
    # clamped greedily, and that costs far more than it saves (120.5 m raw
    # and 1/5 collisions, against 91.5 m and 2/5 through the smoother, even
    # after retuning it to track as tightly as its weights allow).
    #
    # The smoother's place is in front of path_racer.py, whose reference is a
    # plan with no notion of actuator limits. A policy trained against those
    # limits does not need it. Use --smoother only for a checkpoint trained
    # WITHOUT env-side actuator limits (v5 and earlier).
    parser.add_argument(
        "--smoother",
        action="store_true",
        help="filter commands through the CasADi smoother; needed only "
        "for checkpoints trained without env-side actuator limits",
    )
    parser.add_argument("--no-smoother", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="disable the scripted reverse-out stuck recovery",
    )
    parser.add_argument(
        "--free-run",
        action="store_true",
        help="drive immediately instead of waiting for start_signal_detector's "
        "~/go (matching lap_counter's free_run); still stops on ~/done",
    )
    parser.add_argument(
        "--go-topic",
        default="/start_signal_detector/go",
        help="latched std_msgs/Bool that arms driving",
    )
    parser.add_argument(
        "--done-topic",
        default="/lap_counter/done",
        help="latched std_msgs/Bool that stops driving once the target lap "
        "count is reached",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=None,
        help="override the trained speed cap (m/s); affects both the "
        "action decoding and the speed observation scaling, same "
        "as evaluate.py's override",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.exists():
        raise SystemExit(
            f"missing {metadata_path}; it records the observation geometry the "
            "checkpoint was trained with and is written by train.py"
        )
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    env_config = metadata["env"]
    if args.max_speed is not None:
        env_config["max_speed"] = args.max_speed

    bales = bale_geometry.parse_bales(args.sdf_path)
    model = PPO.load(str(checkpoint))

    smoother = None
    if args.smoother and not args.no_smoother:
        # Imported lazily: casadi is only needed for this legacy path (v5
        # and earlier checkpoints), so a deployment running a v6+ checkpoint
        # without --smoother never needs casadi installed at all.
        from casadi_smoother import smoother_from_metadata

        # Same construction evaluate.py uses, so "watch it drive" and the
        # metrics run send identical commands.
        smoother = smoother_from_metadata(
            metadata,
            env_config["control_hz"],
            env_config.get("traction", 0.6),
            env_config["max_speed"],
            WHEELBASE,
            MAX_STEERING_ANGLE,
        )

    if not _unpause_world(args.world_name):
        raise SystemExit(f"could not start world '{args.world_name}' running")

    rclpy.init()
    node = PolicyRunner(
        model,
        bales,
        env_config,
        args.world_name,
        smoother,
        require_start_signal=not args.free_run,
        go_topic=args.go_topic,
        done_topic=args.done_topic,
    )
    if args.no_recovery:
        node.recovery_enabled = False
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("stopping, sending zero velocity")
        node._cmd_pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
