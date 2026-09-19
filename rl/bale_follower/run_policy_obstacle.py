#!/usr/bin/env python3
"""Run a trained Obstacle Course policy against the simulation (or the car).

Sibling to run_policy.py -- see that file for the Speed Course's driver, and
obstacle_env.py's module docstring for why this one perceives through the
ZED's point cloud (cloud_scan.py) instead of analytic course geometry.

Starts driving on the first of two signals, not one:

  * `start_signal_detector`'s latched `~/go` (the camera watching the
    course's visual start signal), or
  * the Arduino's `manual_start` field in `cfr_interfaces/ArduinoStatus`
    (the bench/manual override, for a run with no camera signal to watch --
    a hand on the transmitter, effectively).

Whichever arrives first arms the driver; both are OR'd rather than one
gating the other, matching how the run should actually be startable at the
venue. Stops the same way run_policy.py does: on lap_counter's latched
`~/done`, once the loop counter reports every lap complete.

    ros2 launch cfr_arduino_bridge obstacle_course.launch.py sensors:=true
    python run_policy_obstacle.py --checkpoint checkpoints_obstacle/final_model.zip

This does not include run_policy.py's scripted stuck-recovery: that
sequence was tuned against the Speed Course's hairpins specifically, and
retuning it blind against a course this different in shape (ramps, a
tunnel, a helix) would be guessing. A car that gets stuck here currently
just sits there -- add recovery once real or simulated runs show where it
actually gets stuck and on what.
"""

from __future__ import annotations

import argparse
import json
import collections
import math
import threading
from pathlib import Path

import numpy as np
import rclpy
from cfr_interfaces.msg import ArduinoStatus
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from stable_baselines3 import PPO
from std_msgs.msg import Bool

from cloud_scan import points_from_pointcloud2, scan_from_points
from env import (
    MAX_STEERING_ANGLE,
    WHEELBASE,
    _unpause_world,
    _wrap_to_pi,
    _yaw_from_quaternion,
)
from obstacle_env import YAW_RATE_SCALE

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/obstacle_course.sdf"


class ObstaclePolicyRunner(Node):
    def __init__(
        self,
        model: PPO,
        env_config: dict,
        world_name: str = "cfr_obstacle_course",
        require_start_signal: bool = True,
        go_topic: str = "/start_signal_detector/go",
        status_topic: str = "/arduino_bridge/status",
        done_topic: str = "/lap_counter/done",
        pose_topic: str = "/zed/zed_node/pose",
        cloud_topic: str = "/zed/zed_node/point_cloud/cloud_registered",
    ) -> None:
        super().__init__("obstacle_course_policy")
        self.model = model

        self.armed = not require_start_signal
        self.finished = False
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        if require_start_signal:
            self.create_subscription(Bool, go_topic, self._on_go, latched)
            self.create_subscription(ArduinoStatus, status_topic, self._on_status, 10)
        self.create_subscription(Bool, done_topic, self._on_done, latched)

        self.num_lidar_bins = env_config["num_lidar_bins"]
        self.lidar_fov_deg = env_config["lidar_fov_deg"]
        self.lidar_max_range = env_config["lidar_max_range"]
        self.max_speed = env_config["max_speed"]
        self.reverse_speed = env_config.get("reverse_speed", 0.0)
        self.control_hz = env_config["control_hz"]

        self.obs_speed_scale = self.max_speed
        self.traction = env_config.get("traction", 0.6)
        # Must match ObstacleCourseEnv exactly. A policy fed a differently
        # shaped or differently scaled observation than it trained on is not
        # the policy that was trained.
        self.frame_stack = int(env_config.get("frame_stack", 1))
        self._frames: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.frame_stack
        )
        self._cmd_speed = 0.0

        self._pose_lock = threading.Lock()
        self._pose = None
        self._tilt = (0.0, 0.0)
        self._prev_pose = None
        self._cloud_lock = threading.Lock()
        self._latest_cloud = None

        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)
        qos = rclpy.qos.QoSProfile(depth=1)
        qos.reliability = rclpy.qos.ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(PointCloud2, cloud_topic, self._on_cloud, qos)

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / self.control_hz, self._on_control_tick)
        if self.armed:
            self.get_logger().info("obstacle course policy ready, waiting for pose")
        else:
            self.get_logger().info(
                f"obstacle course policy ready, waiting for pose and a start "
                f"signal: {go_topic} or manual_start on {status_topic}"
            )

    # ------------------------------------------------------------ start/stop

    def _on_go(self, message: Bool) -> None:
        if message.data and not self.armed:
            self.get_logger().info("start signal latched; driving")
        self.armed = self.armed or message.data

    def _on_status(self, message: ArduinoStatus) -> None:
        if message.manual_start and not self.armed:
            self.get_logger().info("manual start override; driving")
        self.armed = self.armed or bool(message.manual_start)

    def _on_done(self, message: Bool) -> None:
        if message.data and not self.finished:
            self.get_logger().info("lap counter done; stopping")
        self.finished = message.data

    # ---------------------------------------------------------------- input

    def _on_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        sin_pitch = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        pitch = math.asin(sin_pitch)
        roll = math.atan2(
            2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        )
        with self._pose_lock:
            self._pose = (p.x, p.y, _yaw_from_quaternion(q.x, q.y, q.z, q.w))
            self._tilt = (pitch, roll)

    def _on_cloud(self, msg: PointCloud2) -> None:
        with self._cloud_lock:
            self._latest_cloud = msg

    def _apply_traction(self, speed: float, steer_fraction: float) -> float:
        a_max = self.traction * 9.81
        dt = 1.0 / self.control_hz
        speed = self._cmd_speed + min(
            max(speed - self._cmd_speed, -a_max * dt), a_max * dt
        )
        tan_delta = abs(math.tan(steer_fraction * MAX_STEERING_ANGLE))
        if tan_delta > 1e-6:
            grip_speed = math.sqrt(a_max * WHEELBASE / tan_delta)
            speed = min(max(speed, -grip_speed), grip_speed)
        self._cmd_speed = max(-self.reverse_speed, min(speed, self.max_speed))
        return self._cmd_speed

    def _scan(self) -> np.ndarray:
        with self._cloud_lock:
            msg = self._latest_cloud
        if msg is None:
            return np.full(self.num_lidar_bins, self.lidar_max_range, dtype=np.float32)
        with self._pose_lock:
            pitch, roll = self._tilt
        return scan_from_points(
            points_from_pointcloud2(msg),
            self.num_lidar_bins,
            self.lidar_fov_deg,
            self.lidar_max_range,
            pitch=pitch,
            roll=roll,
        )

    # --------------------------------------------------------------- control

    def _on_control_tick(self) -> None:
        with self._pose_lock:
            pose = self._pose
        if pose is None:
            return

        if self.finished:
            self._cmd_pub.publish(Twist())
            return
        if not self.armed:
            self._cmd_pub.publish(Twist())
            self._prev_pose = pose
            return

        x, y, yaw = pose
        dt = 1.0 / self.control_hz
        if self._prev_pose is None:
            linear_x, angular_z = 0.0, 0.0
        else:
            px, py, pyaw = self._prev_pose
            # Signed longitudinal travel, matching the env. A magnitude here
            # would make reversing look like standing still.
            linear_x = ((x - px) * math.cos(pyaw) + (y - py) * math.sin(pyaw)) / dt
            angular_z = _wrap_to_pi(yaw - pyaw) / dt
        self._prev_pose = pose

        scan = self._scan()
        frame = np.concatenate(
            [
                (scan / self.lidar_max_range),
                [
                    np.clip((linear_x / self.obs_speed_scale + 1.0) / 2.0, 0.0, 1.0),
                    np.clip((angular_z / YAW_RATE_SCALE + 1.0) / 2.0, 0.0, 1.0),
                ],
            ]
        ).astype(np.float32)
        if not self._frames:
            self._frames.extend([frame] * self.frame_stack)
        else:
            self._frames.append(frame)
        observation = np.concatenate(self._frames).astype(np.float32)

        action, _ = self.model.predict(observation, deterministic=True)
        fraction = (float(np.clip(action[0], -1.0, 1.0)) + 1.0) / 2.0
        speed = -self.reverse_speed + fraction * (self.max_speed + self.reverse_speed)
        steer_fraction = float(np.clip(action[1], -1.0, 1.0))
        delta = steer_fraction * MAX_STEERING_ANGLE

        # The same acceleration and cornering-grip clamp the env applies.
        # Publishing an unclamped speed here is what split train from deploy
        # on the Speed Course: trained at one speed, driven at another.
        speed = self._apply_traction(speed, steer_fraction)

        twist = Twist()
        twist.linear.x = speed
        if abs(speed) > 1e-3:
            twist.angular.z = (speed / WHEELBASE) * math.tan(delta)
        self._cmd_pub.publish(twist)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--world-name", default="cfr_obstacle_course")
    parser.add_argument(
        "--free-run",
        action="store_true",
        help="drive immediately instead of waiting for a start signal; "
        "still stops on ~/done",
    )
    parser.add_argument("--go-topic", default="/start_signal_detector/go")
    parser.add_argument("--status-topic", default="/arduino_bridge/status")
    parser.add_argument("--done-topic", default="/lap_counter/done")
    parser.add_argument("--pose-topic", default="/zed/zed_node/pose")
    parser.add_argument(
        "--cloud-topic", default="/zed/zed_node/point_cloud/cloud_registered"
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.exists():
        raise SystemExit(
            f"missing {metadata_path}; it records the observation geometry the "
            "checkpoint was trained with and is written by train_obstacle.py"
        )
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    env_config = metadata["env"]

    model = PPO.load(str(checkpoint))

    if not _unpause_world(args.world_name):
        raise SystemExit(f"could not start world '{args.world_name}' running")

    rclpy.init()
    node = ObstaclePolicyRunner(
        model,
        env_config,
        args.world_name,
        require_start_signal=not args.free_run,
        go_topic=args.go_topic,
        status_topic=args.status_topic,
        done_topic=args.done_topic,
        pose_topic=args.pose_topic,
        cloud_topic=args.cloud_topic,
    )
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
