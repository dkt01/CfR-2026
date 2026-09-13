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

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from stable_baselines3 import PPO
from tf2_msgs.msg import TFMessage

import bale_geometry
from casadi_smoother import CommandSmoother, smoother_from_metadata
from env import MAX_STEERING_ANGLE, WHEELBASE, _unpause_world, _wrap_to_pi, _yaw_from_quaternion

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
    ) -> None:
        super().__init__("bale_follower_policy")
        self.model = model
        self.bales = bales
        self.smoother = smoother
        self.num_lidar_bins = env_config["num_lidar_bins"]
        self.lidar_fov_deg = env_config["lidar_fov_deg"]
        self.lidar_max_range = env_config["lidar_max_range"]
        self.max_speed = env_config["max_speed"]
        self.reverse_speed = env_config.get("reverse_speed", 0.0)
        self.control_hz = env_config["control_hz"]

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
        self._no_trigger_until = 0.0
        # Escalation: re-sticking right after a recovery means the policy is
        # driving straight back into the same bale, so each quick re-trigger
        # doubles the reverse time (up to 4x) to break the oscillation.
        self._last_recovery_end = 0.0
        self._escalation = 1.0

        # Same ground-truth pose source as training -- the Ackermann plugin's
        # odometry is dead-reckoned and drifts from the true pose.
        self.create_subscription(
            TFMessage, f"/world/{world_name}/dynamic_pose/info", self._on_pose, 10
        )
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / self.control_hz, self._on_control_tick)
        self.get_logger().info("bale-following policy ready, waiting for pose")

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
        while self._pose_history and now - self._pose_history[0][0] > self.stuck_window_s + 1.0:
            self._pose_history.pop(0)
        if not self.recovery_enabled or now < self._no_trigger_until:
            return False
        oldest = self._pose_history[0]
        if now - oldest[0] < self.stuck_window_s:
            return False
        return math.hypot(x - oldest[1], y - oldest[2]) < self.stuck_distance

    def _start_recovery(self, now: float, scan: np.ndarray) -> None:
        # Steer toward the side the nearest obstacle is on: in reverse the
        # nose swings away from the wheels' direction. Bin 0 is the rightmost
        # ray, positive bearings are left.
        nearest = int(np.argmin(scan))
        half_fov = math.radians(self.lidar_fov_deg) / 2.0
        bearing = -half_fov + nearest * 2.0 * half_fov / max(1, self.num_lidar_bins - 1)
        self._recovery_delta = math.copysign(MAX_STEERING_ANGLE, bearing if abs(bearing) > 1e-3 else 1.0)
        if now - self._last_recovery_end < 6.0:
            self._escalation = min(self._escalation * 2.0, 4.0)
        else:
            self._escalation = 1.0
        self._recovery_until = now + self.recovery_duration_s * self._escalation
        self.get_logger().info(
            f"stuck (moved <{self.stuck_distance} m in {self.stuck_window_s} s); "
            f"reversing with steer {self._recovery_delta:+.2f} rad"
        )

    def _publish_recovery(self) -> None:
        twist = Twist()
        twist.linear.x = -self.recovery_reverse_speed
        twist.angular.z = (twist.linear.x / WHEELBASE) * math.tan(self._recovery_delta)
        self._cmd_pub.publish(twist)

    def _on_control_tick(self) -> None:
        with self._lock:
            pose = self._pose
        if pose is None:
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
            self.bales, x, y, yaw, self.num_lidar_bins, self.lidar_fov_deg, self.lidar_max_range
        )

        now = time.monotonic()
        if now < self._recovery_until:
            self._publish_recovery()
            return
        if self._recovery_until and now - self._recovery_until < 1.0 / self.control_hz + 0.1:
            # Recovery just ended: forget the stuck history and the smoother's
            # reversed state, and give the policy a grace period to move off.
            self._pose_history.clear()
            self._no_trigger_until = now + self.recovery_cooldown_s
            self._last_recovery_end = now
            if self.smoother is not None:
                self.smoother.reset()
        if self._check_stuck(now, x, y):
            self._start_recovery(now, scan)
            self._publish_recovery()
            return

        observation = np.concatenate(
            [
                (scan / self.lidar_max_range),
                [
                    np.clip(linear_x / self.max_speed, 0.0, 1.0),
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
    parser.add_argument("--no-smoother", action="store_true",
                        help="publish raw policy commands without CasADi filtering")
    parser.add_argument("--no-recovery", action="store_true",
                        help="disable the scripted reverse-out stuck recovery")
    parser.add_argument("--max-speed", type=float, default=None,
                        help="override the trained speed cap (m/s); affects both the "
                             "action decoding and the speed observation scaling, same "
                             "as evaluate.py's override")
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
    if not args.no_smoother:
        # Same construction evaluate.py uses, so "watch it drive" and the
        # metrics run send identical commands.
        smoother = smoother_from_metadata(
            metadata, env_config["control_hz"], env_config.get("traction", 0.6),
            env_config["max_speed"], WHEELBASE, MAX_STEERING_ANGLE,
        )

    if not _unpause_world(args.world_name):
        raise SystemExit(f"could not start world '{args.world_name}' running")

    rclpy.init()
    node = PolicyRunner(model, bales, env_config, args.world_name, smoother)
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
