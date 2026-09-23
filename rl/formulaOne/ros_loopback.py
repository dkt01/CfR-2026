#!/usr/bin/env python3
"""A car on ROS topics, without Gazebo.

Publishes /zed/zed_node/pose and /arduino_bridge/status, consumes /drive_cmd,
and moves the same `Plant` the trainer uses.  Two jobs:

  * It lets `formula_one_node.py` be exercised end to end -- anchoring,
    observation assembly, action scaling, lap counting, the DriveCommand it
    puts on the wire -- in a second, with no renderer and no physics engine.
    That is what `node_selftest.py` runs.
  * It isolates blame.  If the policy drives here and not in Gazebo, the gap
    is contacts, tire friction or timing.  If it does not drive here either,
    the gap is in this repository's own plumbing, and that is a much shorter
    search.

It is NOT a substitute for `validate.sh`.  The plant it runs is the same model
the policy was trained against, so of course the policy agrees with it; the
check that means something is the one against Gazebo's physics.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Bool

from cfr_interfaces.msg import ArduinoStatus, DriveCommand

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track as track_mod  # noqa: E402
from plant import Plant  # noqa: E402

HERE = Path(__file__).resolve().parent
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Loopback(Node):
    def __init__(self):
        super().__init__("ros_loopback")
        self.declare_parameter("config", str(HERE / "config.yaml"))
        self.declare_parameter("repo_root", str(HERE.parents[1]))
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("randomize", False)
        self.declare_parameter("go_after", 2.0)

        cfg = yaml.safe_load(Path(self.get_parameter("config").value).read_text())
        if not self.get_parameter("randomize").value:
            cfg = {**cfg, "randomize": {**cfg["randomize"], "enabled": False}}
        self.cfg = cfg
        self.track = track_mod.build(cfg, Path(self.get_parameter("repo_root").value))

        i = int(
            np.clip(
                np.searchsorted(self.track.s, self.track.start_station),
                0,
                len(self.track.s) - 1,
            )
        )
        self.plant = Plant(cfg, 1, np.random.default_rng(0))
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

        self.create_subscription(
            DriveCommand, "/drive_cmd", self.on_cmd, qos_profile_sensor_data
        )
        # RELIABLE, depth 10 -- what the ros_gz bridge publishes in Gazebo and
        # what the ZED wrapper publishes on the car.  The loopback is only
        # useful as a stand-in if it stands in on the same QoS.
        self.pose_pub = self.create_publisher(PoseStamped, "/zed/zed_node/pose", 10)
        self.status_pub = self.create_publisher(
            ArduinoStatus, "/arduino_bridge/status", 10
        )
        self.go_pub = self.create_publisher(Bool, "/start_signal_detector/go", LATCHED)
        self.go_pub.publish(Bool(data=False))
        self.create_timer(1.0 / self.rate, self.tick)
        self.create_timer(float(self.get_parameter("go_after").value), self.release)
        self.released = False
        self.prev_tick = None
        self.get_logger().info("loopback car up, will go green shortly")

    def release(self):
        if not self.released:
            self.released = True
            self.go_pub.publish(Bool(data=True))
            self.get_logger().info("GREEN")

    def on_cmd(self, msg):
        self.active = bool(msg.auto_ready)
        self.command = np.array([float(msg.steering), float(msg.velocity)])

    def tick(self):
        steer = np.array([self.command[0]])
        speed = np.array([self.command[1] if self.active else 0.0])
        # Step by the WALL time that actually passed, not by the nominal
        # period.  This node stamps its poses with the wall clock and the
        # driver runs on `use_sim_time:=false`, so advancing a fixed 1/rate
        # per tick silently decouples the two the moment anything else is
        # using the CPU: the timer falls behind, the car moves slower than
        # the clock it is timestamped with, and the driver -- which
        # differences those stamps to get a yaw rate -- reads a yaw rate
        # several times too small and drives into a bale.  That looked
        # exactly like a bug in the node.  It was a bug in this file.
        #
        # Clamped because a large step is still an integration error, and
        # because a stalled executor must not teleport the car.
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
        # Same blind spot the real tachometer has, so the policy is fed the
        # instrument it trained on rather than a clean crawl.
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
