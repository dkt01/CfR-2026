#!/usr/bin/env python3
"""Apply a simple stereo-like depth error to Gazebo's perfect RGB-D cloud."""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


def corrupt_cloud(msg, rng, noise_a=0.01, noise_b=0.008, dropout=0.03):
    """Perturb range along each camera ray; preserve bearing and packed color."""
    offsets = {field.name: field.offset for field in msg.fields}
    if not all(axis in offsets for axis in ("x", "y", "z")) or msg.is_bigendian:
        raise ValueError("expected a little endian XYZ cloud")
    data = bytearray(msg.data)
    shape = (msg.height, msg.width)
    strides = (msg.row_step, msg.point_step)
    xyz = [
        np.ndarray(shape, dtype="<f4", buffer=data, offset=offsets[a], strides=strides)
        for a in ("x", "y", "z")
    ]
    x, y, z = xyz
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0)
    original_x = x[valid].copy()
    delta = rng.normal(0.0, noise_a + noise_b * original_x**2)
    scale = np.maximum(0.01, original_x + delta) / original_x
    for axis in xyz:
        axis[valid] *= scale
    lost = valid & (rng.random(shape) < dropout)
    for axis in xyz:
        axis[lost] = np.nan
    result = PointCloud2()
    result.header = msg.header
    result.height = msg.height
    result.width = msg.width
    result.fields = msg.fields
    result.is_bigendian = msg.is_bigendian
    result.point_step = msg.point_step
    result.row_step = msg.row_step
    result.is_dense = False
    result.data = bytes(data)
    return result


class ZedCloudNoise(Node):
    def __init__(self):
        super().__init__("zed_cloud_noise")
        self.rng = np.random.default_rng()
        self.publisher = self.create_publisher(
            PointCloud2, "/zed/zed_node/point_cloud/cloud_registered", 10
        )
        self.create_subscription(PointCloud2, "/zed/gz/rgbd/points", self.on_cloud, 10)

    def on_cloud(self, msg):
        self.publisher.publish(corrupt_cloud(msg, self.rng))


def main():
    rclpy.init()
    node = ZedCloudNoise()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
