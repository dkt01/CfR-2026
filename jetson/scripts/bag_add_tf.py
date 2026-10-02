#!/usr/bin/env python3
"""Give a recorded run the TF tree RViz needs to draw the car and the ZED.

    ./scripts/bag_add_tf.py ~/cfr_runs/<run>                # writes <run>/bag_tf + <run>/frames.rviz
    ./scripts/bag_add_tf.py ~/cfr_runs/<run> --overwrite           # redo after a mount change
    ./scripts/bag_add_tf.py ~/cfr_runs/<run> --pose-frame zed_camera_link
    ros2 bag play ~/cfr_runs/<run>/bag_tf --clock --loop &
    rviz2 -d ~/cfr_runs/<run>/frames.rviz --ros-args -p use_sim_time:=true

record_run.py does not record /tf, so a bag has poses but no frames.  This
copies every message into a new bag and adds

    /tf         map -> base_link, one per /zed/zed_node/pose
    /tf_static  base_link -> zed_camera_link -> zed_left_camera_frame
                -> zed_left_camera_optical_frame

The static chain is vehicle.yaml's camera_mount and lens offset, read through
camera_extrinsics.py, so it is the same geometry the drivers use.  base_link
is the vehicle frame: ground under the wheelbase midpoint, x forward, z up.

What /zed/zed_node/pose reports differs by source.  On the car it is the ZED's
camera_link, so base_link is that pose times the inverse mount.  In Gazebo it
is the slash model origin, which is already base_link.  --pose-frame auto
picks Gazebo when every pose sits at exactly z = 0, which the real camera
never does.

A car bag that recorded /tf already has the ZED wrapper's own tree, map ->
odom -> zed_camera_link -> zed_camera_center -> zed_left_camera_frame ->
zed_left_camera_frame_optical, with the factory lens offsets.  zed_camera_link
already has a parent there, so base_link goes in as its child instead, the
inverse mount, appended to the recorded /tf_static.  Nothing else is added.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_extrinsics import load_mount, rotation  # noqa: E402

import rosbag2_py  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.serialization import deserialize_message, serialize_message  # noqa: E402
from rclpy.time import Time  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

POSE_TOPIC = "/zed/zed_node/pose"
BASE = "base_link"
CAMERA = "zed_camera_link"
LENS = "zed_left_camera_frame"
OPTICAL = "zed_left_camera_optical_frame"


def quat_of(r):
    """3x3 rotation -> (x, y, z, w)."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(
        math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2]
    )
    y = math.copysign(
        math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0]
    )
    z = math.copysign(
        math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1]
    )
    return x, y, z, w


def matrix_of(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform(parent, child, stamp, r, t):
    msg = TransformStamped()
    msg.header.frame_id, msg.child_frame_id = parent, child
    msg.header.stamp = stamp
    (
        msg.transform.translation.x,
        msg.transform.translation.y,
        msg.transform.translation.z,
    ) = (float(v) for v in t)
    q = msg.transform.rotation
    q.x, q.y, q.z, q.w = quat_of(r)
    return msg


def static_chain(mount, stamp):
    r_mount = rotation(mount.roll, mount.pitch, mount.yaw)
    # REP-103 body frame to camera optical frame (z forward, x right, y down).
    r_optical = rotation(-math.pi / 2, 0.0, -math.pi / 2)
    return [
        transform(BASE, CAMERA, stamp, r_mount, (mount.x, mount.y, mount.z)),
        transform(CAMERA, LENS, stamp, np.eye(3), mount.lens),
        transform(LENS, OPTICAL, stamp, r_optical, (0.0, 0.0, 0.0)),
    ]


def open_reader(uri):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    return reader


def sim_poses(uri):
    """True when every pose sits at exactly z = 0, i.e. Gazebo ground truth."""
    reader = open_reader(uri)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[POSE_TOPIC]))
    pose_type = get_message("geometry_msgs/msg/PoseStamped")
    seen = False
    while reader.has_next():
        _, data, _ = reader.read_next()
        seen = True
        if deserialize_message(data, pose_type).pose.position.z != 0.0:
            return False
    return seen


RVIZ = """\
Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Global Options:
    Fixed Frame: map
    Frame Rate: 30
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Cell Size: 1
      Plane Cell Count: 60
      Reference Frame: map
    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: true
      Show Names: true
      Show Arrows: true
      Show Axes: true
      Marker Scale: 0.6
      Frame Timeout: 1000
    - Class: rviz_default_plugins/Path
      Name: Centerline
      Enabled: true
      Topic:
        Value: /formula_one/centerline
        Durability Policy: Transient Local
    - Class: rviz_default_plugins/MarkerArray
      Name: Speed cap
      Enabled: true
      Topic:
        Value: /formula_one/markers
        Durability Policy: Transient Local
    - Class: rviz_default_plugins/Pose
      Name: ZED pose (raw)
      Enabled: true
      Topic:
        Value: /zed/zed_node/pose
      Shape: Axes
      Axes Length: 0.4
      Axes Radius: 0.02
    - Class: rviz_default_plugins/Camera
      Name: Depth frustum
      Enabled: false
      Topic:
        Value: /zed/zed_node/depth/camera_info
  Tools:
    - Class: rviz_default_plugins/MoveCamera
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Target Frame: base_link
      Distance: 3
      Pitch: 0.5
      Yaw: 2.4
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, help="run directory holding bag/")
    ap.add_argument(
        "--pose-frame",
        choices=("auto", BASE, CAMERA),
        default="auto",
        help="what point /zed/zed_node/pose reports (default: auto)",
    )
    ap.add_argument(
        "--vehicle", type=Path, default=None, help="vehicle.yaml to read the mount from"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="output bag (default <run>/bag_tf)"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="replace an existing output bag"
    )
    args = ap.parse_args()

    src = args.run / "bag"
    out = args.out or args.run / "bag_tf"
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    mount = load_mount(args.vehicle) if args.vehicle else load_mount()
    if mount is None:
        sys.exit("no camera_mount in vehicle.yaml")
    pose_frame = args.pose_frame
    if pose_frame == "auto":
        pose_frame = BASE if sim_poses(src) else CAMERA
    if args.pose_frame == BASE and "/tf" in {
        t.name for t in open_reader(src).get_all_topics_and_types()
    }:
        sys.exit("this bag recorded the ZED's /tf, so its pose is zed_camera_link")
    print(
        f"{POSE_TOPIC} taken as {pose_frame}; mount ({mount.provenance}) "
        f"x {mount.x:.3f} y {mount.y:+.3f} z {mount.z:.3f} m, "
        f"rpy {math.degrees(mount.roll):+.1f} {math.degrees(mount.pitch):+.1f} "
        f"{math.degrees(mount.yaw):+.1f} deg, lens {tuple(mount.lens)}"
    )

    r_mount = rotation(mount.roll, mount.pitch, mount.yaw)
    t_mount = np.array([mount.x, mount.y, mount.z])

    reader = open_reader(src)
    topics = reader.get_all_topics_and_types()
    types = {t.name: t.type for t in topics}
    has_tree = "/tf" in types
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(out), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    for t in topics:
        writer.create_topic(t)
    if not has_tree:
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                0,
                "/tf",
                "tf2_msgs/msg/TFMessage",
                "cdr",
            )
        )
    if "/tf_static" not in types:
        # Latched, so a late RViz still gets it.
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                0,
                "/tf_static",
                "tf2_msgs/msg/TFMessage",
                "cdr",
                offered_qos_profiles=[_latched_qos()],
            )
        )

    def ours(stamp):
        if has_tree:
            # camera_link_T_base = inv(base_T_camera_link)
            return [transform(CAMERA, BASE, stamp, r_mount.T, -r_mount.T @ t_mount)]
        return static_chain(mount, stamp)

    pose_type = get_message(types[POSE_TOPIC])
    wrote_static, n_tf = False, 0
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == "/tf_static" and not wrote_static:
            # Into the recorded message rather than beside it: a depth-1
            # latched topic would hand a late subscriber only one of the two.
            tf_static = deserialize_message(data, TFMessage)
            tf_static.transforms.extend(ours(tf_static.transforms[0].header.stamp))
            data, wrote_static = serialize_message(tf_static), True
        elif "/tf_static" not in types and not wrote_static:
            # At the bag's first instant, so playback from any offset that
            # starts at 0 has it.  Static TF ignores the stamp.
            tf_static = TFMessage(transforms=ours(Time(nanoseconds=t_ns).to_msg()))
            writer.write("/tf_static", serialize_message(tf_static), t_ns)
            wrote_static = True
        writer.write(topic, data, t_ns)
        if topic != POSE_TOPIC or has_tree:
            continue
        msg = deserialize_message(data, pose_type)
        p, o = msg.pose.position, msg.pose.orientation
        r = matrix_of((o.x, o.y, o.z, o.w))
        t = np.array([p.x, p.y, p.z])
        if pose_frame == CAMERA:
            # map_T_base = map_T_cam * inv(base_T_cam)
            r = r @ r_mount.T
            t = t - r @ t_mount
        tf = TFMessage(
            transforms=[
                transform(msg.header.frame_id or "map", BASE, msg.header.stamp, r, t)
            ]
        )
        writer.write("/tf", serialize_message(tf), t_ns)
        n_tf += 1
    del writer

    (args.run / "frames.rviz").write_text(RVIZ)
    added = (
        f"{CAMERA}->{BASE} into the recorded tree"
        if has_tree
        else f"{n_tf} map->{BASE} transforms"
    )
    print(f"wrote {out} ({added}) and {args.run / 'frames.rviz'}")


def _latched_qos():
    return rosbag2_py._storage.QoS(1).transient_local().reliable()


if __name__ == "__main__":
    main()
