#!/usr/bin/env python3
"""Run cloud_segmentation on the live ZED cloud and republish it by class.

Subscribes to the ZED's registered cloud and its pose, classifies every
point, and publishes the cloud again with each point's color set by its
class, so RViz and the browser viewer can show what the segmenter decided
rather than what the camera saw. The pose is only read for its pitch and
roll -- on the car the ZED's map frame is gravity-aligned, in the sim the
bridged ground-truth pose stands in for it -- which is what cloud_segmentation
needs to level the cloud.

Colors: ground gray-green, obstacle red, hoop magenta, car wash cyan,
overhead blue, unknown dark gray.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField

import cloud_segmentation as cs

CLASS_COLORS = {
    cs.GROUND: (96, 128, 96),
    cs.OBSTACLE: (230, 50, 40),
    cs.HOOP: (240, 60, 240),
    cs.CARWASH: (40, 220, 230),
    cs.OVERHEAD: (70, 110, 255),
    cs.UNKNOWN: (60, 60, 60),
}


def packed_colors(labels: np.ndarray) -> np.ndarray:
    """uint32 0x00RRGGBB per point, the PCL/ROS packed-rgb convention."""
    table = np.zeros(256, dtype=np.uint32)
    for label, (r, g, b) in CLASS_COLORS.items():
        table[label] = (r << 16) | (g << 8) | b
    return table[labels]


def xyz_from_cloud(msg: PointCloud2) -> np.ndarray:
    offsets = {f.name: f.offset for f in msg.fields}
    count = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=count * msg.point_step)
    raw = raw.reshape(count, msg.point_step)
    endian = ">f4" if msg.is_bigendian else "<f4"
    return np.stack(
        [
            raw[:, offsets[a] : offsets[a] + 4].copy().view(endian).ravel()
            for a in "xyz"
        ],
        axis=1,
    )


def pitch_roll(q) -> tuple[float, float]:
    """Nose-down pitch and left-up roll from a REP-103 orientation."""
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    roll = math.atan2(
        2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    )
    return pitch, roll


class CloudSegmentation(Node):
    def __init__(self):
        super().__init__("cloud_segmentation")
        # Segmentation runs on the full cloud -- thin structure at range is
        # what the segmenter needs every point for -- and only what is
        # published is thinned, to every stride-th row and column, because
        # the browser draws it and a quarter of 230k points is plenty to see.
        self.stride = int(self.declare_parameter("stride", 2).value)
        self.tilt = (0.0, 0.0)
        self.publisher = self.create_publisher(PointCloud2, "~/cloud", 1)
        self.create_subscription(PoseStamped, "pose", self.on_pose, 10)
        self.create_subscription(
            PointCloud2, "cloud", self.on_cloud, qos_profile_sensor_data
        )

    def on_pose(self, msg: PoseStamped) -> None:
        self.tilt = pitch_roll(msg.pose.orientation)

    def on_cloud(self, msg: PointCloud2) -> None:
        points = xyz_from_cloud(msg)
        labels = cs.segment(points, *self.tilt).labels
        if msg.height > 1 and self.stride > 1:
            grid = (msg.height, msg.width)
            points = points.reshape(*grid, 3)[:: self.stride, :: self.stride].reshape(
                -1, 3
            )
            labels = labels.reshape(grid)[:: self.stride, :: self.stride].ravel()
        keep = labels != cs.UNKNOWN
        points = points[keep].astype(np.float32)
        colors = packed_colors(labels[keep])

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = len(points)
        out.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        out.is_bigendian = False
        out.point_step = 16
        out.row_step = 16 * len(points)
        out.is_dense = True
        packed = np.empty((len(points), 4), dtype="<u4")
        packed[:, :3] = points.view("<u4")
        packed[:, 3] = colors
        out.data = packed.tobytes()
        self.publisher.publish(out)


def main():
    rclpy.init()
    node = CloudSegmentation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
