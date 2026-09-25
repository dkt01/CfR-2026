#!/usr/bin/env python3
"""A car on ROS topics, without Gazebo -- now with a ZED depth camera.

formulaOne's loopback (pose, tachometer, /drive_cmd into `Plant`), plus a
depth image rendered from the car's true pose: a full-resolution 640x360
32FC1 image and its CameraInfo, on the topics the ZED wrapper uses, at
`depth_hz`.  The render is the training renderer run at full resolution, so
this checks the NODE's half of the depth path -- decoding, the intrinsics,
sampling the grid, the stack, the timing -- against an image whose geometry
is known exactly.

Failure injection, for the watchdog (`depth_mode`, from `depth_fail_after`
seconds after green):

    ok       depth throughout
    stop     depth stops arriving
    nan      frames keep arriving, every pixel NaN
    none     no depth, ever
    blip     depth stops for `depth_blip` seconds, then comes back

Like formulaOne's, it is NOT a substitute for Gazebo: the car here is the
model the policy trained on.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool

from cfr_interfaces.msg import ArduinoStatus, DriveCommand

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track as track_mod  # noqa: E402
from perception import Camera  # noqa: E402
from plant import Plant  # noqa: E402
from world import World  # noqa: E402

HERE = Path(__file__).resolve().parent
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
WIDTH, HEIGHT = 640, 360


class Loopback(Node):
    def __init__(self, **kwargs):
        super().__init__("ros_loopback", **kwargs)
        self.declare_parameter("config", str(HERE / "config.yaml"))
        self.declare_parameter("repo_root", str(HERE.parents[1]))
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("go_after", 2.0)
        self.declare_parameter("depth_hz", 15.0)
        self.declare_parameter("depth_mode", "ok")  # ok | stop | nan | none
        self.declare_parameter("depth_fail_after", 30.0)  # s after green
        self.declare_parameter("depth_blip", 0.6)  # s, for depth_mode blip

        cfg = yaml.safe_load(Path(self.get_parameter("config").value).read_text())
        cfg = {**cfg, "randomize": {**cfg["randomize"], "enabled": False}}
        self.cfg = cfg
        self.track = track_mod.build(cfg, Path(self.get_parameter("repo_root").value))
        rng = np.random.default_rng(0)
        # The nominal world: bales where the map says.
        self.world = World(self.track, cfg, 1, rng)
        self.world.sample(np.array([True]), enabled=False)

        # The same camera, sampled at every pixel of the full image.
        full = copy.deepcopy(cfg)
        full["camera"].update(col_stride=1, row_stride=1, row_slope_limit=10.0)
        self.cam = Camera(full)
        self.f = (WIDTH / 2) / math.tan(math.radians(float(cfg["camera"]["hfov_deg"])) / 2)

        i = int(np.clip(np.searchsorted(self.track.s, self.track.start_station), 0, len(self.track.s) - 1))
        self.plant = Plant(cfg, 1, rng)
        self.plant.reset(
            np.array([True]),
            np.array([self.track.x[i]]),
            np.array([self.track.y[i]]),
            np.array([math.atan2(self.track.ty[i], self.track.tx[i])]),
            np.zeros(1),
        )
        self.rate = float(self.get_parameter("rate").value)
        self.dt = np.full(1, 1.0 / self.rate)
        self.command = np.zeros(2)
        self.active = False
        self.mode = str(self.get_parameter("depth_mode").value)
        self.fail_after = float(self.get_parameter("depth_fail_after").value)
        self.blip = float(self.get_parameter("depth_blip").value)
        self.go_time = None

        self.create_subscription(DriveCommand, "/drive_cmd", self.on_cmd, qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, "/zed/zed_node/pose", 10)
        self.status_pub = self.create_publisher(ArduinoStatus, "/arduino_bridge/status", 10)
        self.depth_pub = self.create_publisher(Image, "/zed/zed_node/depth/depth_registered", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/zed/zed_node/depth/camera_info", 10)
        self.go_pub = self.create_publisher(Bool, "/start_signal_detector/go", LATCHED)
        self.go_pub.publish(Bool(data=False))
        self.create_timer(1.0 / self.rate, self.tick)
        self.create_timer(1.0 / float(self.get_parameter("depth_hz").value), self.publish_depth)
        self.create_timer(float(self.get_parameter("go_after").value), self.release)
        self.released = False
        self.prev_tick = None
        self.depth_sent = 0
        self.get_logger().info(f"loopback car up, depth mode {self.mode}")

    def release(self):
        if not self.released:
            self.released = True
            self.go_time = self.get_clock().now().nanoseconds * 1e-9
            self.go_pub.publish(Bool(data=True))
            self.get_logger().info("GREEN")

    def on_cmd(self, msg):
        self.active = bool(msg.auto_ready)
        self.command = np.array([float(msg.steering), float(msg.velocity)])

    def failed(self):
        if self.mode == "none":
            return True
        if self.mode == "ok" or self.go_time is None:
            return False
        since = self.get_clock().now().nanoseconds * 1e-9 - self.go_time
        if self.mode == "blip":
            return self.fail_after <= since < self.fail_after + self.blip
        return since >= self.fail_after

    def publish_depth(self):
        failed = self.failed()
        if failed and self.mode in ("stop", "none", "blip"):
            return
        now = self.get_clock().now()
        p = self.plant
        yaw = float(p.yaw[0])
        ox = np.array([p.x[0] + self.cam.x * math.cos(yaw)])
        oy = np.array([p.y[0] + self.cam.x * math.sin(yaw)])
        angles = yaw + self.cam.azimuth[None, :]
        r_h = self.world.raycast(ox, oy, angles, self.cam.scan_max + 2.0)
        r_h = np.where(np.isfinite(r_h), r_h, self.cam.scan_max + 5.0)
        depth = self.cam.render(r_h, np.zeros(1), np.full(1, self.cam.z))[0]
        if failed and self.mode == "nan":
            depth = np.full_like(depth, np.nan)
        img = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
        img[self.cam.rows] = depth.astype(np.float32)

        msg = Image()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "zed_left_camera_optical_frame"
        msg.height, msg.width = HEIGHT, WIDTH
        msg.encoding = "32FC1"
        msg.is_bigendian = sys.byteorder == "big"
        msg.step = WIDTH * 4
        msg.data = img.tobytes()
        self.depth_pub.publish(msg)

        info = CameraInfo()
        info.header = msg.header
        info.height, info.width = HEIGHT, WIDTH
        cx, cy = WIDTH / 2 - 0.5, HEIGHT / 2 - 0.5
        info.k = [self.f, 0.0, cx, 0.0, self.f, cy, 0.0, 0.0, 1.0]
        info.p = [self.f, 0.0, cx, 0.0, 0.0, self.f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)
        self.depth_sent += 1

    def tick(self):
        steer = np.array([self.command[0]])
        speed = np.array([self.command[1] if self.active else 0.0])
        # Step by the wall time that actually passed -- formulaOne's loopback
        # explains why a fixed step here reads as a bug in the node.
        now = self.get_clock().now()
        wall = now.nanoseconds * 1e-9
        step = self.dt[0] if self.prev_tick is None else wall - self.prev_tick
        self.prev_tick = wall
        dt = np.full(1, float(np.clip(step, self.dt[0], 4.0 * self.dt[0])))
        self.plant.substep(steer, speed, dt)
        stamp = now.to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        pose.pose.position.x = float(self.plant.x[0])
        pose.pose.position.y = float(self.plant.y[0])
        pose.pose.orientation.z = math.sin(self.plant.yaw[0] / 2)
        pose.pose.orientation.w = math.cos(self.plant.yaw[0] / 2)
        self.pose_pub.publish(pose)

        status = ArduinoStatus()
        status.header.stamp = stamp
        status.header.frame_id = "base_link"
        status.link_ok = True
        status.mode = ArduinoStatus.MODE_AUTO_ACTIVE
        status.auto_arm = True
        status.battery_level = 255
        v = float(self.plant.speed[0])
        status.speed = 0.0 if v < 0.3 else v
        status.target_speed = float(self.plant.target[0])
        self.status_pub.publish(status)


def main():
    rclpy.init()
    node = Loopback()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
