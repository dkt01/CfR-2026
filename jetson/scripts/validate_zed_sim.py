#!/usr/bin/env python3
"""Sample the live simulated ZED cloud and report geometry and temporal noise."""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class Sampler(Node):
    def __init__(self):
        super().__init__("zed_sim_validator")
        self.frames = []
        self.frame_id = None
        self.create_subscription(
            PointCloud2,
            "/zed/zed_node/point_cloud/cloud_registered",
            self.on_cloud,
            10,
        )

    def on_cloud(self, msg):
        fields = {f.name: f.offset for f in msg.fields}
        if not all(k in fields for k in ("x", "y", "z")):
            raise RuntimeError(f"missing XYZ fields: {fields}")
        if msg.is_bigendian:
            raise RuntimeError("big endian cloud not supported")
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": ["<f4"] * 3,
                "offsets": [fields[k] for k in ("x", "y", "z")],
                "itemsize": msg.point_step,
            }
        )
        points = np.frombuffer(msg.data, dtype=dtype)
        xyz = np.stack([points[k] for k in ("x", "y", "z")], axis=-1)
        self.frames.append(xyz)
        self.frame_id = msg.header.frame_id


def main():
    rclpy.init()
    node = Sampler()
    end = time.monotonic() + 30
    while len(node.frames) < 8 and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=1)
    if len(node.frames) < 2:
        raise RuntimeError(f"only {len(node.frames)} cloud frames in 30 seconds")
    frames = np.stack(node.frames)
    valid = np.isfinite(frames).all(axis=2)
    p = frames[0, valid[0]]
    print(
        f"frames={len(frames)} frame_id={node.frame_id} shape={frames.shape[1:]} valid={valid.mean():.3f}"
    )
    for i, axis in enumerate("xyz"):
        print(f"{axis}: p1/median/p99={np.percentile(p[:, i], [1, 50, 99])}")
    both = valid.all(axis=0)
    static = frames[:, both, 0]
    print(
        f"repeat pixels={static.shape[1]} x temporal std median={np.median(np.std(static, axis=0)):.6f} m"
    )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
