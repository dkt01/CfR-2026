#!/usr/bin/env python3
"""Drive a synthetic run through the car launch and check it starts and stops.

The unit tests cover the lap counting and the start signal's decision; what
they cannot cover is the chain obstacle_racer_car.launch.py wires on the car:
the racer waits, goes on the start signal or on the Arduino's Manual Start
bit, and takes the throttle off when lap_counter latches done after the
configured laps.  This checks exactly that, with no Gazebo, no ZED and no
Arduino: it plays the bridge and the camera itself, and moves a synthetic
pose round a loop only while the racer commands a speed.

    source ~/ros2_ws/install/setup.bash
    ros2 launch rl/obstacleRacer/obstacle_racer_car.launch.py record:=false \\
        policy:=rl/obstacleRacer/bestModel/v8/policy.npz \\
        config:=rl/obstacleRacer/bestModel/v8/config.yaml
    python3 rl/obstacleRacer/check_car_chain.py --start visual   # or manual

It checks that the car holds zero speed until the start, drives after it,
is still driving after the first lap, and has zero speed from the moment
lap_counter reports done -- with done after exactly `--laps` laps (2, the
launch file's default).  The Manual Start run raises ArduinoStatus.manual_start
from 0 to 1, as the Arduino's Manual Start bit does; nothing here stands in
for an RC control.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from cfr_interfaces.msg import ArduinoStatus, DriveCommand, LapCount
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
RATE_HZ = 100.0
# Metres the synthetic car moves per pose while commanded to drive: well
# under lap_counter's max_step (1.0 m), or each sample reads as a loop closure.
STEP = 0.12
# A loop about the Obstacle Course's length, starting at the parked pose and
# heading +x, so the line lap_counter places 0.70 m ahead is crossed once on
# the way out and once per lap after.
CORNERS = [(0.0, 0.0), (22.0, 0.0), (22.0, -12.0), (-4.0, -12.0), (-4.0, 0.0)]
WAIT_S = 3.0  # parked, before the start: nothing may move the car


def loop_path():
    points = []
    for a, b in zip(CORNERS, CORNERS[1:] + CORNERS[:1]):
        span = math.hypot(b[0] - a[0], b[1] - a[1])
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        n = max(int(span / STEP), 1)
        points += [
            (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, yaw)
            for i in range(1, n + 1)
        ]
    return points


def floor_cloud():
    """Open floor ahead of the camera, in its frame (x forward, z up)."""
    xs, ys = np.meshgrid(np.linspace(0.5, 5.0, 40), np.linspace(-2.0, 2.0, 30))
    zs = np.full(xs.shape, -0.25)
    return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(np.float32)


class Check(Node):
    def __init__(self, start: str, laps: int) -> None:
        super().__init__("check_car_chain")
        self.start, self.laps = start, laps
        # Reliable, as the ZED wrapper publishes it: lap_counter subscribes
        # reliable, and the racer's best-effort subscription takes either.
        self.pose_pub = self.create_publisher(PoseStamped, "/zed/zed_node/pose", 10)
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            "/zed/zed_node/point_cloud/cloud_registered",
            qos_profile_sensor_data,
        )
        self.status_pub = self.create_publisher(
            ArduinoStatus, "/arduino_bridge/status", 10
        )
        self.go_pub = self.create_publisher(Bool, "/start_signal_detector/go", LATCHED)
        self.create_subscription(
            DriveCommand, "/drive_cmd", self.on_drive, qos_profile_sensor_data
        )
        self.create_subscription(LapCount, "/lap_counter/count", self.on_count, 10)
        self.create_subscription(Bool, "/lap_counter/done", self.on_done, LATCHED)

        self.path = loop_path()
        self.index = 0
        self.cloud = floor_cloud()
        self.velocity = 0.0
        self.active = False  # the Arduino's AUTO_ACTIVE, once it is driven
        self.t0 = None
        self.triggered_at = None
        self.moved_before = 0  # nonzero speed commands before the start
        self.first_drive = None
        self.laps_seen = 0
        self.lap_times = []
        self.v_at_lap = {}
        self.done_at = None
        self.moved_after_done = 0
        self.after_done = 0
        self.create_timer(1.0 / RATE_HZ, self.tick)
        self.create_timer(1.0 / 15.0, self.publish_cloud)

    def now(self) -> float:
        return time.monotonic()

    def on_drive(self, msg: DriveCommand) -> None:
        self.velocity = float(msg.velocity)
        if self.triggered_at is None:
            self.moved_before += self.velocity != 0.0
        elif self.velocity != 0.0 and self.first_drive is None:
            self.first_drive = self.now() - self.triggered_at
        if self.done_at is not None:
            self.after_done += 1
            self.moved_after_done += self.velocity != 0.0

    def on_count(self, msg: LapCount) -> None:
        if msg.laps > self.laps_seen:
            self.laps_seen = int(msg.laps)
            self.lap_times.append(self.now())
            self.v_at_lap[self.laps_seen] = self.velocity
            print(
                f"  lap {msg.laps}/{msg.target}, still commanding {self.velocity:+.2f} m/s"
            )

    def on_done(self, msg: Bool) -> None:
        if msg.data and self.done_at is None:
            # The racer reacts on its next 20 Hz tick; give it two.
            self.done_at = self.now() + 0.1
            print(f"  lap_counter done after {self.laps_seen} laps")

    def publish_cloud(self) -> None:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "zed_left_camera_frame"
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, self.cloud))

    def tick(self) -> None:
        if self.t0 is None:
            # Start the clock once the racer and lap_counter are listening.
            if (
                self.pose_pub.get_subscription_count() < 2
                or self.status_pub.get_subscription_count() < 2
            ):
                return
            self.t0 = self.now()
            print("racer and lap_counter up; parked")
        t = self.now() - self.t0
        if self.velocity != 0.0 and self.triggered_at is not None:
            self.active = True
            self.index += 1
        x, y, yaw = (
            (0.0, 0.0, 0.0)
            if self.index == 0
            else self.path[(self.index - 1) % len(self.path)]
        )
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x, pose.pose.position.y = x, y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_pub.publish(pose)

        trigger = t >= WAIT_S
        if trigger and self.triggered_at is None:
            self.triggered_at = self.now()
            print(f"start: {self.start}")
            if self.start == "visual":
                self.go_pub.publish(Bool(data=True))
        status = ArduinoStatus()
        status.header.stamp = pose.header.stamp
        status.link_ok = True
        status.auto_arm = True
        status.manual_start = self.start == "manual" and trigger
        status.mode = (
            ArduinoStatus.MODE_AUTO_ACTIVE
            if self.active
            else ArduinoStatus.MODE_AUTO_ARMED
        )
        status.speed = float(self.velocity)
        self.status_pub.publish(status)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", choices=("visual", "manual"), required=True)
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    rclpy.init()
    node = Check(args.start, args.laps)
    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.done_at is not None and time.monotonic() > node.done_at + 3.0:
            break
    first_lap_driving = node.v_at_lap.get(1, 0.0) != 0.0
    checks = [
        (
            "holds zero speed before the start",
            node.t0 is not None and node.moved_before == 0,
            f"{node.moved_before} nonzero commands while parked",
        ),
        (
            f"drives on the {args.start} start",
            node.first_drive is not None,
            f"first nonzero command {node.first_drive and round(node.first_drive, 2)} s after it",
        ),
        (
            "still driving after lap 1",
            first_lap_driving,
            f"commanding {node.v_at_lap.get(1)} m/s as lap 1 counted",
        ),
        (
            f"lap_counter done after exactly {args.laps} laps",
            node.done_at is not None and node.laps_seen == args.laps,
            f"done {'latched' if node.done_at else 'never'}, {node.laps_seen} laps",
        ),
        (
            "zero speed from done on",
            node.after_done > 0 and node.moved_after_done == 0,
            f"{node.moved_after_done} of {node.after_done} commands after done were nonzero",
        ),
    ]
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}  ({detail})")
        failed += not ok
    node.destroy_node()
    rclpy.try_shutdown()
    print("PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
