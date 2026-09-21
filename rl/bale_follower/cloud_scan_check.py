#!/usr/bin/env python3
"""Is the observation the policy trains on actually a picture of the course?

Subscribes to the rendered ZED cloud and the ground-truth pose, builds the
36-bin scan two ways at the same instant -- `cloud_scan.scan_from_points`
from the camera, and `bale_geometry.lidar_scan` ray-cast against the true
bale geometry -- and compares them. Publishes nothing, so it is safe to run
alongside a training or collection process.

The policy sees only the first of these. If it does not resemble the second,
no amount of reward shaping will teach it to steer, because it cannot see
where the bales are.

    python cloud_scan_check.py --samples 40
"""

from __future__ import annotations

import argparse
import math
import threading
import time

import numpy as np
import rclpy
import yaml
from pathlib import Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage

import bale_geometry
from cloud_scan import points_from_pointcloud2, scan_from_points
from env import _yaw_from_quaternion

HERE = Path(__file__).resolve().parent
SDF = HERE.parents[1] / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--world", default="cfr_speed_course")
    parser.add_argument("--topic",
                        default="/zed/zed_node/point_cloud/cloud_registered")
    args = parser.parse_args()

    config = yaml.safe_load((HERE / "config_lap.yaml").read_text())["env"]
    bins = config["num_lidar_bins"]
    fov = config["lidar_fov_deg"]
    max_range = config["lidar_max_range"]
    bales = bale_geometry.parse_bales(str(SDF))

    rclpy.init()
    node = Node("cloud_scan_check")
    state = {"pose": None, "tilt": (0.0, 0.0), "cloud": None, "clouds": 0}
    lock = threading.Lock()

    def on_pose(msg: TFMessage) -> None:
        if not msg.transforms:
            return
        t = msg.transforms[0].transform
        q = t.rotation
        sin_pitch = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        with lock:
            state["pose"] = (t.translation.x, t.translation.y,
                             _yaw_from_quaternion(q.x, q.y, q.z, q.w))
            state["tilt"] = (math.asin(sin_pitch),
                             math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                                        1.0 - 2.0 * (q.x * q.x + q.y * q.y)))

    def on_cloud(msg: PointCloud2) -> None:
        with lock:
            state["cloud"] = msg
            state["clouds"] += 1

    node.create_subscription(TFMessage, f"/world/{args.world}/dynamic_pose/info",
                             on_pose, 10)
    qos = rclpy.qos.QoSProfile(depth=1)
    qos.reliability = rclpy.qos.ReliabilityPolicy.BEST_EFFORT
    node.create_subscription(PointCloud2, args.topic, on_cloud, qos)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        with lock:
            ready = state["pose"] is not None and state["cloud"] is not None
        if ready:
            break
        time.sleep(0.2)
    else:
        raise SystemExit("no pose and/or cloud arrived in 20 s -- "
                         f"is the sim up with sensors:=true? clouds seen: "
                         f"{state['clouds']}")

    errors, cloud_empty, truth_empty, point_counts, nan_fractions = [], [], [], [], []
    per_bin = np.zeros(bins)
    samples = 0
    while samples < args.samples:
        with lock:
            pose, tilt, msg = state["pose"], state["tilt"], state["cloud"]
        points = points_from_pointcloud2(msg)
        finite = np.isfinite(points).all(axis=1) if len(points) else np.zeros(0, bool)
        nan_fractions.append(1.0 - (finite.mean() if len(points) else 1.0))
        point_counts.append(len(points))
        camera = scan_from_points(points, bins, fov, max_range,
                                  pitch=tilt[0], roll=tilt[1])
        truth = bale_geometry.lidar_scan(bales, pose[0], pose[1], pose[2],
                                         bins, fov, max_range)
        errors.append(np.abs(camera - truth))
        per_bin += np.abs(camera - truth)
        cloud_empty.append((camera >= max_range - 1e-6).mean())
        truth_empty.append((truth >= max_range - 1e-6).mean())
        samples += 1
        time.sleep(0.15)

    errors = np.asarray(errors)
    print(f"\n{samples} paired scans, {np.mean(point_counts):.0f} cloud points each, "
          f"{np.mean(nan_fractions):.1%} of them non-finite")
    print(f"mean |camera - truth| per bin : {errors.mean():.3f} m")
    print(f"median                        : {np.median(errors):.3f} m")
    print(f"90th percentile               : {np.percentile(errors, 90):.3f} m")
    print(f"bins reading 'nothing there'  : camera {np.mean(cloud_empty):.1%}, "
          f"truth {np.mean(truth_empty):.1%}")
    print(f"worst bins (index: mean error): " + ", ".join(
        f"{i}:{per_bin[i] / samples:.2f}" for i in np.argsort(per_bin)[-5:][::-1]))
    agree = (errors < 0.5).mean()
    print(f"bins agreeing within 0.5 m    : {agree:.1%}")
    print("\nverdict:", "camera scan tracks the geometry"
          if agree > 0.8 else "CAMERA SCAN DOES NOT MATCH THE GEOMETRY")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
