#!/usr/bin/env python3
"""Drive short left/right arcs and report Gazebo odometry yaw signs."""

import statistics
import time

import rclpy
from rclpy.node import Node
from cfr_interfaces.msg import DriveCommand
from nav_msgs.msg import Odometry


class Probe(Node):
    def __init__(self):
        super().__init__("steering_sim_validator")
        self.publisher = self.create_publisher(DriveCommand, "/drive_cmd", 10)
        self.yaw_rates = []
        self.create_subscription(Odometry, "/zed/zed_node/odom", self.on_odom, 10)

    def on_odom(self, msg):
        self.yaw_rates.append(msg.twist.twist.angular.z)

    def arc(self, steering, duration=5):
        self.yaw_rates.clear()
        end = time.monotonic() + duration
        while time.monotonic() < end:
            cmd = DriveCommand()
            cmd.auto_ready = True
            cmd.velocity = 1.5
            cmd.steering = steering
            self.publisher.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.04)
        values = self.yaw_rates[len(self.yaw_rates) // 2 :]
        if not values:
            raise RuntimeError("no Gazebo odometry")
        return statistics.median(values)


def main():
    rclpy.init()
    probe = Probe()
    left = probe.arc(0.5)
    right = probe.arc(-0.5)
    probe.arc(0.0, 2)
    print(f"left command: median yaw rate {left:+.3f} rad/s")
    print(f"right command: median yaw rate {right:+.3f} rad/s")
    assert left > 0 and right < 0
    probe.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
