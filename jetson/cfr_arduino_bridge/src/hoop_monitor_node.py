#!/usr/bin/env python3
"""Watch every hoop on the Obstacle Course and say whether the car threaded it.

Run alongside the bridge and the ZED, same as lap_counter:

    ros2 launch cfr_arduino_bridge hoop_monitor.launch.py

In the Gazebo Obstacle Course this comes up already, wired to the same
layout file the randomizer draws hoop positions from.

| Interface        | Type                        | Direction                    |
| ----------------- | --------------------------- | ----------------------------- |
| `pose`            | `geometry_msgs/PoseStamped` | subscribed (`/zed/.../pose`)  |
| `hoop_layout`      | `cfr_interfaces/HoopLayout` | subscribed (current hoop poses) |
| `~/status`         | `cfr_interfaces/HoopStatus` | published every pose sample   |
| `~/reset`          | `std_srvs/Trigger`          | service, clear state for a re-run |

The pose is the same ground-truth ZED map-frame topic lap_counter reads --
see that node's docstring for why it is not `~/odom`.

Hoop positions come from `obstacle_randomizer_node`'s `~/hoop_layout` topic
rather than being read once at startup: the randomizer moves the hoops along
their lines on every `~/randomize` or `~/reset`, and this node has to track
wherever they currently stand, not where the course drawing nominally put
them.

The crossing logic itself is in `hoop_monitor.py`, which is free of ROS and
tested against synthetic traces -- including the caveat in its docstring
about the yaw convention not yet being checked against a real sim run. This
file is the wiring: parameters, the two subscriptions, and the logging that
makes a missed hoop diagnosable afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import rclpy
from cfr_interfaces.msg import HoopLayout, HoopStatus
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hoop_monitor as hm  # noqa: E402


def _yaw_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    import math

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class HoopMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__(
            "hoop_monitor", automatically_declare_parameters_from_overrides=True
        )
        self.declare_parameter_if_missing("gate_half_width", hm.DEFAULT_GATE_HALF_WIDTH)
        self.declare_parameter_if_missing("pose_timeout", 0.5)

        hoops = [
            hm.Hoop(
                name=name,
                x=float(self.hoop_param(name, "nominal")[0]),
                y=float(self.hoop_param(name, "nominal")[1]),
                yaw=float(self.hoop_param(name, "yaw")),
                half_width=float(self.get_parameter("gate_half_width").value),
            )
            for name in self.hoop_names()
        ]
        self.monitor = hm.HoopMonitor(hoops)

        self.pose_time = None
        self.warned_no_pose = False
        self.status_publisher = self.create_publisher(HoopStatus, "~/status", 10)
        self.create_service(Trigger, "~/reset", self.on_reset)

        self.create_subscription(PoseStamped, "pose", self.on_pose, 10)
        self.create_subscription(
            HoopLayout,
            "hoop_layout",
            self.on_hoop_layout,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_timer(1.0, self.on_watchdog)

        if hoops:
            self.get_logger().info(
                f"watching {len(hoops)} hoops: {', '.join(h.name for h in hoops)}"
            )
        else:
            self.get_logger().info(
                "no hoops declared for this course; publishing an always-clear status"
            )
        self.publish_status()

    def declare_parameter_if_missing(self, name: str, default) -> None:
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    # ---------------------------------------------------------------- params

    def hoop_names(self) -> list[str]:
        if not self.has_parameter("hoops.names"):
            return []
        return list(self.get_parameter("hoops.names").value)

    def hoop_param(self, hoop: str, name: str):
        return self.get_parameter(f"hoops.{hoop}.{name}").value

    # ---------------------------------------------------------------- inputs

    def on_hoop_layout(self, message: HoopLayout) -> None:
        for name, position in zip(message.names, message.positions):
            self.monitor.update_hoop(name, position.x, position.y)

    def on_pose(self, message: PoseStamped) -> None:
        self.pose_time = self.get_clock().now()
        self.warned_no_pose = False

        orientation = message.pose.orientation
        pose = hm.Pose2D(
            x=message.pose.position.x,
            y=message.pose.position.y,
            yaw=_yaw_from_quaternion(
                orientation.w, orientation.x, orientation.y, orientation.z
            ),
        )
        resolved = self.monitor.update(pose)
        for name in resolved["passed"]:
            self.get_logger().info(f"{name} passed")
        for name in resolved["missed"]:
            self.get_logger().warning(f"{name} MISSED -- run failed")
        if resolved["passed"] or resolved["missed"]:
            self.publish_status()

    def on_reset(self, request, response):
        del request
        self.monitor.reset()
        self.publish_status()
        self.get_logger().info("reset; watching for the next run")
        response.success = True
        response.message = "hoop monitor reset"
        return response

    def on_watchdog(self) -> None:
        if self.pose_time is None:
            if not self.warned_no_pose and self.hoop_names():
                self.warned_no_pose = True
                self.get_logger().warning(
                    f"no pose yet on {self.resolve_topic_name('pose')}"
                )
            return
        age = (self.get_clock().now() - self.pose_time).nanoseconds * 1e-9
        if age > self.get_parameter("pose_timeout").value and not self.warned_no_pose:
            self.warned_no_pose = True
            self.get_logger().warning(f"pose stale for {age:.1f} s")

    # ---------------------------------------------------------------- output

    def publish_status(self) -> None:
        message = HoopStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        names = list(self.monitor.hoops.keys())
        message.names = names
        message.passed = [self.monitor.state[name].passed for name in names]
        message.missed = [self.monitor.state[name].missed for name in names]
        message.any_missed = self.monitor.any_missed
        message.all_passed = self.monitor.all_passed
        self.status_publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = HoopMonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
