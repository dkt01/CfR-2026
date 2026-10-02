#!/usr/bin/env python3
"""Republish one ZED depth frame every 1/rate_hz s, small enough to bag.

The ZED's depth is ~0.9-1.8 MB a frame at 12-15 Hz and its rate cannot be
turned down, so a characterization bag cannot take it raw.  This keeps the
same thinned stream record_run.py puts in drive bags (16-bit millimeters on
/run_recorder/depth, PNG-compressed when OpenCV is there), so the Run Lab's
camera checks read both kinds of run the same way.  It is its own process so
an image-handling failure cannot touch the maneuver runner.

Subscribed best-effort, depth 1: it must never hold up a driver's reliable
subscription to the same topic.
"""

import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

OUT = "/run_recorder/depth"


class DepthThinner(Node):
    def __init__(self):
        super().__init__("depth_thinner")
        self.rate_hz = float(self.declare_parameter("rate_hz", 2.0).value)
        source = self.declare_parameter(
            "source", "/zed/zed_node/depth/depth_registered"
        ).value
        try:
            import cv2  # noqa: F401

            self.png = True
            self.pub = self.create_publisher(CompressedImage, OUT + "/compressed", 10)
        except ImportError:
            self.png = False
            self.pub = self.create_publisher(Image, OUT, 10)
        self.last = None
        self.create_subscription(
            Image,
            source,
            self.on_depth,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )

    def on_depth(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last is not None and 0 <= stamp - self.last < 1.0 / self.rate_hz - 0.02:
            return
        if msg.encoding == "32FC1":
            rows = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.step // 4
            )
            with np.errstate(invalid="ignore"):
                mm = np.nan_to_num(rows[:, : msg.width] * 1000.0, nan=0.0, posinf=0.0)
            mm = np.clip(mm, 0, 65535).astype(np.uint16)
        elif msg.encoding in ("16UC1", "mono16"):
            mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.step // 2
            )
            mm = mm[:, : msg.width]
        else:
            return
        if msg.is_bigendian != (sys.byteorder == "big"):
            mm = mm.byteswap()
        self.last = stamp
        if self.png:
            import cv2

            ok, png = cv2.imencode(".png", np.ascontiguousarray(mm))
            if not ok:
                return
            out = CompressedImage(
                header=msg.header, format="16UC1; png", data=png.tobytes()
            )
        else:
            out = Image(header=msg.header, height=mm.shape[0], width=mm.shape[1])
            out.encoding = "16UC1"
            out.is_bigendian = sys.byteorder == "big"
            out.step = 2 * out.width
            out.data = np.ascontiguousarray(mm).tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DepthThinner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
