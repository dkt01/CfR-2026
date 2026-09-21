#!/usr/bin/env python3
"""TFMessage ground truth -> geometry_msgs/PoseStamped.

`lap_counter_node` wants a PoseStamped (what the real ZED's map-frame `~/pose`
publishes); the RL/path_racer training stack only bridges the Slash's ground
truth as TFMessage on `/world/<world>/dynamic_pose/info` (see env.py's
docstring on why: it is teleport-safe, unlike the Ackermann plugin's
dead-reckoned odometry). This republishes one as the other so the same
training-stack launch can drive the REAL lap_counter node, rather than
path_racer.py's own arc-length lap log, without pulling in the full
simulation.launch.py stack (which would double-publish /cmd_vel via
path_follower_node).

    ros2 run --prefix 'python3' pose_republisher.py   # or just: python pose_republisher.py
"""
from __future__ import annotations

import argparse

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class PoseRepublisher(Node):
    def __init__(self, world: str, out_topic: str) -> None:
        super().__init__("pose_republisher")
        self.pub = self.create_publisher(PoseStamped, out_topic, 10)
        self.create_subscription(
            TFMessage, f"/world/{world}/dynamic_pose/info", self.on_tf, 10
        )

    def on_tf(self, msg: TFMessage) -> None:
        if not msg.transforms:
            return
        transform = msg.transforms[0].transform  # index 0 is the Slash
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"
        out.pose.position.x = transform.translation.x
        out.pose.position.y = transform.translation.y
        out.pose.position.z = transform.translation.z
        out.pose.orientation = transform.rotation
        self.pub.publish(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="cfr_speed_course")
    parser.add_argument("--out-topic", default="/sim_ground_truth/pose")
    args = parser.parse_args()

    rclpy.init()
    node = PoseRepublisher(args.world, args.out_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
