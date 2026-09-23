#!/usr/bin/env python3
"""Make the simulation visible in RViz2: TF tree, chassis marker, bale markers.

RViz needs three things this stack does not otherwise provide.

1. A TF TREE. The only ground-truth transform the sim bridges is
   /world/<world>/dynamic_pose/info as tf2_msgs/TFMessage, and ros_gz's
   Pose_V -> TFMessage conversion does not populate frame_id or
   child_frame_id at all (gazebosim/ros_gz#172 and #410, the latter closed
   "not planned"). RViz cannot resolve a transform whose frames are empty
   strings, so that topic is unusable as a TF source. This republishes the
   unambiguous single-vehicle pose (/zed/zed_node/pose) as a real
   world -> base_link transform instead.

2. A ROBOT TO LOOK AT. There is no URDF anywhere in this repo and no
   robot_state_publisher, so RViz's RobotModel display has nothing to load.
   The chassis is drawn as a CUBE marker built from bale_geometry's own
   CHASSIS_LENGTH/WIDTH instead, so it matches the footprint the controller
   actually reasons about rather than a second, drifting copy of the
   dimensions.

3. THE CLOUD'S OWN FRAME. The ZED point cloud arrives stamped with whatever
   frame_id gz-sim gave the sensor. Rather than hard-code a guess, this
   subscribes to the cloud, reads the frame_id off the first message that
   arrives, and publishes base_link -> <that frame> from the camera mount
   pose in the SDF. Self-configuring, so it cannot be wrong about a name
   nobody has verified.

The bales are published once as a MarkerArray so the corridor is visible
around the car; they are static, so this is latched rather than streamed.

    python rviz_bridge.py            # alongside a running sim
    rviz2 -d sim.rviz

NOTE: the point cloud only exists when the sim is launched with
`sensors:=true` (see jetson/cfr_arduino_bridge/launch/sensors_world.py) --
without it the ZED is never rendered and that display stays empty. Everything
else here works either way.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import bale_geometry

DEFAULT_SDF = (
    Path(__file__).resolve().parents[2]
    / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"
)
# The ZED's mount on the chassis, from the rgbd_camera <pose> in
# sensors_world.py. Same number medley_racer.py uses for the point cloud it
# renders in the browser viewer.
CAMERA_MOUNT = (0.315, 0.0, 0.20)
WORLD_FRAME = "world"
BASE_FRAME = "base_link"


class RvizBridge(Node):
    def __init__(self, args) -> None:
        super().__init__("rviz_bridge")
        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._cloud_frame: str | None = None

        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._markers = self.create_publisher(MarkerArray, "/sim_markers", latched)

        self.create_subscription(PoseStamped, args.pose_topic, self._on_pose, 10)
        # Only to learn the frame the cloud is stamped with; the cloud itself
        # goes straight to RViz, this never republishes it.
        self.create_subscription(PointCloud2, args.cloud_topic, self._on_cloud, 1)

        self._publish_bales(str(args.sdf))
        self.get_logger().info(
            f"rviz_bridge: {WORLD_FRAME} -> {BASE_FRAME} from {args.pose_topic}, "
            f"bale markers on /sim_markers, watching {args.cloud_topic} for its frame"
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = WORLD_FRAME
        t.child_frame_id = BASE_FRAME
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self._tf.sendTransform(t)
        self._publish_chassis(msg.header.stamp)

    def _on_cloud(self, msg: PointCloud2) -> None:
        frame = msg.header.frame_id
        if not frame or frame == self._cloud_frame:
            return
        # First cloud (or a changed frame): wire it under base_link at the
        # camera's mount pose so RViz can place it against the vehicle.
        self._cloud_frame = frame
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = BASE_FRAME
        t.child_frame_id = frame
        t.transform.translation.x = CAMERA_MOUNT[0]
        t.transform.translation.y = CAMERA_MOUNT[1]
        t.transform.translation.z = CAMERA_MOUNT[2]
        t.transform.rotation.w = 1.0
        self._static_tf.sendTransform(t)
        self.get_logger().info(
            f"point cloud frame is '{frame}'; published {BASE_FRAME} -> {frame}"
        )

    def _publish_chassis(self, stamp) -> None:
        m = Marker()
        m.header.frame_id = BASE_FRAME
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = "chassis", 0, Marker.CUBE, Marker.ADD
        m.pose.position.z = 0.09
        m.pose.orientation.w = 1.0
        m.scale.x = bale_geometry.CHASSIS_LENGTH
        m.scale.y = bale_geometry.CHASSIS_WIDTH
        m.scale.z = 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = 0.85, 0.08, 0.04, 0.9
        array = MarkerArray()
        array.markers.append(m)
        # A nose arrow, because a symmetric box gives no clue which way the
        # car is pointing -- the single most useful thing to see in a replay.
        nose = Marker()
        nose.header.frame_id = BASE_FRAME
        nose.header.stamp = stamp
        nose.ns, nose.id, nose.type, nose.action = (
            "chassis",
            1,
            Marker.ARROW,
            Marker.ADD,
        )
        nose.pose.position.z = 0.18
        nose.pose.orientation.w = 1.0
        nose.scale.x, nose.scale.y, nose.scale.z = 0.5, 0.06, 0.06
        nose.color.r, nose.color.g, nose.color.b, nose.color.a = 0.05, 1.0, 0.08, 1.0
        array.markers.append(nose)
        self._markers.publish(array)

    def _publish_bales(self, sdf_path: str) -> None:
        bales = bale_geometry.parse_bales(sdf_path)
        array = MarkerArray()
        for bale in bales:
            m = Marker()
            m.header.frame_id = WORLD_FRAME
            m.ns, m.id, m.type, m.action = "bales", bale.index, Marker.CUBE, Marker.ADD
            m.pose.position.x = bale.x
            m.pose.position.y = bale.y
            m.pose.position.z = 0.1778
            m.pose.orientation.z = math.sin(bale.yaw / 2.0)
            m.pose.orientation.w = math.cos(bale.yaw / 2.0)
            m.scale.x = bale.half_x * 2.0
            m.scale.y = bale.half_y * 2.0
            m.scale.z = 0.3556
            m.color.r, m.color.g, m.color.b, m.color.a = 0.72, 0.48, 0.12, 0.85
            array.markers.append(m)
        self._markers.publish(array)
        self.get_logger().info(f"published {len(bales)} bale markers")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pose-topic", default="/zed/zed_node/pose")
    parser.add_argument(
        "--cloud-topic", default="/zed/zed_node/point_cloud/cloud_registered"
    )
    parser.add_argument("--sdf", default=str(DEFAULT_SDF))
    args = parser.parse_args()

    rclpy.init()
    node = RvizBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
