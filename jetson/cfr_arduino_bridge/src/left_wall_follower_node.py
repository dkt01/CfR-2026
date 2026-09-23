#!/usr/bin/env python3
"""Drive the speed course after either visual or manual start; stop after three laps."""

import math
import sys
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from cfr_interfaces.msg import StartSignal
from sensor_msgs.msg import PointCloud2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from left_wall_follower import WHEELBASE, CourseFollower, load_bales  # noqa: E402
from left_wall_cloud import cloud_command, ranges_from_cloud  # noqa: E402


class LeftWallFollower(Node):
    def __init__(self):
        super().__init__("left_wall_follower")
        self.declare_parameter("source", "simulation")
        self.declare_parameter(
            "cloud_topic", "/zed/zed_node/point_cloud/cloud_registered"
        )
        self.source = self.get_parameter("source").value
        if self.source not in ("simulation", "cloud"):
            raise ValueError("source must be 'simulation' or 'cloud'")
        if self.source == "simulation":
            sdf = (
                Path(get_package_share_directory("cfr_arduino_bridge"))
                / "worlds/speed_course.sdf"
            )
            self.bales = load_bales(sdf)
            self.controller = CourseFollower(
                sdf.parent.parent / "config/speed_course_path.json"
            )
        else:
            self.bales = None
            self.controller = None
        self.go = False
        self.manual_go = False
        self.manual_stop = False
        self.signal_green = False
        self.done = False
        self.pose = None
        self.pose_time = None
        self.cloud_scan = None
        self.cloud_time = None
        self.signal_requested = False
        self.declare_parameter("auto_start_signal", True)
        self.signal_client = (
            self.create_client(SetBool, "/obstacle_randomizer/start_signal")
            if self.source == "simulation"
            else None
        )
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, "/start_signal_detector/go", self.on_go, latched)
        self.create_subscription(Bool, "/lap_counter/done", self.on_done, latched)
        if self.source == "simulation":
            self.create_subscription(
                Bool,
                "/obstacle_randomizer/start_signal_green",
                self.on_signal_green,
                latched,
            )
            self.create_subscription(
                StartSignal, "/start_signal_detector/state", self.on_signal_state, 10
            )
            self.create_subscription(
                PoseStamped, "/zed/zed_node/pose", self.on_pose, 10
            )
        else:
            self.create_subscription(
                PointCloud2,
                self.get_parameter("cloud_topic").value,
                self.on_cloud,
                qos_profile_sensor_data,
            )
        self.manual_publisher = self.create_publisher(
            Bool, "/left_wall_follower/manual_go", latched
        )
        self.create_service(SetBool, "~/manual_start", self.on_manual_start)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f"{self.source} wall follower waiting for start")

    def on_go(self, msg):
        self.go = msg.data

    def on_manual_start(self, request, response):
        self.manual_go = request.data
        self.manual_stop = not request.data
        self.manual_publisher.publish(Bool(data=request.data))
        if not request.data:
            self.publisher.publish(Twist())
        response.success = True
        response.message = "manual start" if request.data else "manual stop"
        return response

    def on_signal_green(self, msg):
        if msg.data and not self.signal_green and self.go:
            self.manual_stop = False
        self.signal_green = msg.data

    def on_signal_state(self, msg):
        # Wait for a confirmed red frame before turning the simulated arm.
        if (
            self.get_parameter("auto_start_signal").value
            and msg.armed
            and not self.signal_requested
            and self.signal_client.service_is_ready()
        ):
            self.signal_requested = True
            self.signal_client.call_async(SetBool.Request(data=True))

    def on_done(self, msg):
        self.done = msg.data
        if self.done:
            self.publisher.publish(Twist())
            self.get_logger().info("lap counter complete; stopped")

    def on_pose(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)
        self.pose_time = self.get_clock().now()

    def on_cloud(self, msg):
        try:
            self.cloud_scan = ranges_from_cloud(msg)
            self.cloud_time = self.get_clock().now()
        except ValueError as error:
            self.get_logger().warning(f"unusable ZED cloud: {error}")

    def tick(self):
        twist = Twist()
        visual_start = self.go and (
            self.signal_green if self.source == "simulation" else True
        )
        started = self.manual_go or visual_start
        if started and not self.manual_stop and not self.done:
            if (
                self.source == "simulation"
                and self.pose is not None
                and self.pose_time is not None
            ):
                age = (self.get_clock().now() - self.pose_time).nanoseconds * 1e-9
                if 0 <= age < 0.5:
                    speed, steer = self.controller.command(self.bales, *self.pose)
                    twist.linear.x = speed
                    twist.angular.z = speed * math.tan(steer) / WHEELBASE
            elif (
                self.source == "cloud"
                and self.cloud_scan is not None
                and self.cloud_time is not None
            ):
                age = (self.get_clock().now() - self.cloud_time).nanoseconds * 1e-9
                if 0 <= age < 0.5:
                    speed, steer = cloud_command(*self.cloud_scan)
                    twist.linear.x = speed
                    twist.angular.z = speed * math.tan(steer) / WHEELBASE
        self.publisher.publish(twist)


def main():
    rclpy.init()
    node = LeftWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publisher.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
