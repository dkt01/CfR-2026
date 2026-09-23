#!/usr/bin/env python3
"""TFMessage ground truth -> nav_msgs/Odometry, in the course frame.

Stands in for the QuestNav republisher path_racer.py's `--pose-msg odom`
docstring names but that does not exist yet anywhere in this repo (checked:
no QuestNav driver, bridge, or frame transform exists in jetson/ or
elsewhere). This does NOT validate a Quest headset or its calibration -- it
validates that `path_racer.py`'s `_on_odom` handler is wired correctly and
drives identically to the TF path, which is the part of "does --pose-msg odom
work" that is actually testable without the physical Orin.
"""

from __future__ import annotations

import argparse

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class OdomRepublisher(Node):
    def __init__(self, world: str, out_topic: str) -> None:
        super().__init__("odom_republisher")
        self.pub = self.create_publisher(Odometry, out_topic, 10)
        self.create_subscription(
            TFMessage, f"/world/{world}/dynamic_pose/info", self.on_tf, 10
        )

    def on_tf(self, msg: TFMessage) -> None:
        if not msg.transforms:
            return
        transform = msg.transforms[0].transform
        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "course"
        out.child_frame_id = "base_link"
        out.pose.pose.position.x = transform.translation.x
        out.pose.pose.position.y = transform.translation.y
        out.pose.pose.position.z = transform.translation.z
        out.pose.pose.orientation = transform.rotation
        self.pub.publish(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="cfr_speed_course")
    parser.add_argument("--out-topic", default="/sim_ground_truth/odom")
    args = parser.parse_args()
    rclpy.init()
    node = OdomRepublisher(args.world, args.out_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
