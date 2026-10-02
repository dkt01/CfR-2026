#!/usr/bin/env python3
"""Film a driver on a CfR course inside the sim container (video.sh runs this).

    capture.py patch-world --course obstacle --rtf 0.1 [--sensors-plugin]
    capture.py gazebo --course obstacle --seed 104 --out /work/out --label "..."
    capture.py replay --course speed --track /work/track.npz --out /work/out --label "..."

gazebo  The real stack drives under Gazebo physics (launched by video.sh).
        Once set up, the capture steps the world 10 ms at a time, paced to
        --rtf, and places a level, yaw-following chase camera on the car
        before every step.  The run ends on lap_counter's done, a rollover,
        no progress, or the timeout.
replay  A numpy-sim track (rollout.py) is re-posed frame by frame with Gazebo
        paused: a static ghost of the car carries the ZED view, the chase
        camera follows it, and each frame steps the world one control period so
        both cameras render.  Gazebo's physics plays no part.

Both write <out>/frames/*.jpg, <out>/list.txt (an ffmpeg concat list whose
durations are sim time, so the video plays at real speed) and summary.json.
Before either, a top-down camera photographs the course for the minimap.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path("/repo")
SHARE = (
    Path.home() / "ros2_ws/install/cfr_arduino_bridge/share/cfr_arduino_bridge/worlds"
)
COURSES = {
    "obstacle": dict(
        world="cfr_obstacle_course", sdf="obstacle_course.sdf", chase=(1.5, 0.95)
    ),
    "speed": dict(world="cfr_speed_course", sdf="speed_course.sdf", chase=(2.2, 1.3)),
}
SYSTEM_MARKER = "<!-- cfr:sensors-system -->"
SENSORS_PLUGIN = (
    '<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">'
    "<render_engine>ogre2</render_engine></plugin>"
)
ZED_OFFSET = (0.315, 0.0, 0.20)  # sensors_world.SENSORS_CAMERA, chassis frame
W, H = 1280, 720


# ------------------------------------------------------------------ world file


def patch_world(args):
    """Fresh copy of the source world into the install tree, then the edits.

    The install tree is the container's own (build.sh copies, it does not
    symlink), so nothing here touches the repo.  real_time_factor caps how fast
    Gazebo runs: with a training run holding the CPU, the ROS pipeline falls
    behind sim time and the drivers' stale-sensor watchdogs stop the car.
    """
    c = COURSES[args.course]
    text = (REPO / "jetson/cfr_arduino_bridge/worlds" / c["sdf"]).read_text()
    text, n = re.subn(
        r"<real_time_factor>[^<]*</real_time_factor>",
        f"<real_time_factor>{args.rtf}</real_time_factor>",
        text,
        count=1,
    )
    assert n == 1, "no <real_time_factor> in the world"
    if args.sensors_plugin:
        # The film cameras need the Sensors system even when the launch runs
        # sensors:=false, which would otherwise leave the marker empty.
        assert text.count(SYSTEM_MARKER) == 1
        text = text.replace(SYSTEM_MARKER, SENSORS_PLUGIN)
    (SHARE / c["sdf"]).write_text(text)
    print(
        f"patched {SHARE / c['sdf']}: rtf {args.rtf}, sensors plugin {args.sensors_plugin}"
    )


def car_start(course):
    root = ET.parse(
        REPO / "jetson/cfr_arduino_bridge/worlds" / COURSES[course]["sdf"]
    ).getroot()
    car = [m for m in root.iter("model") if m.get("name") == "slash"][0]
    x, y, z, _, _, yaw = (float(v) for v in car.findtext("pose").split())
    return x, y, yaw


def course_bounds(course):
    if course == "speed":
        d = json.loads(
            (
                REPO / "jetson/cfr_arduino_bridge/config/speed_course_path.json"
            ).read_text()
        )
        m = 2.5
        return min(d["x"]) - m, max(d["x"]) + m, min(d["y"]) - m, max(d["y"]) + m
    return -10.8, 10.9, -12.3, 3.9  # rl/obstacleRacer course_model's static grid


# ------------------------------------------------------------------ geometry


def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rpy_of(q):
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    roll = math.atan2(
        2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    )
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


def camera_sdf(
    name, topic, width, height, hfov, hz, pitch=0.0, far=40.0, near=0.1, rgbd=False
):
    return (
        f'<sdf version="1.9"><model name="{name}"><static>true</static><link name="link">'
        f'<sensor name="{name}" type="{"rgbd_camera" if rgbd else "camera"}"><pose>0 0 0 0 {pitch} 0</pose>'
        f"<always_on>1</always_on><update_rate>{hz}</update_rate><topic>{topic}</topic>"
        f"<camera><horizontal_fov>{hfov}</horizontal_fov>"
        f"<image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>"
        f"<clip><near>{near}</near><far>{far}</far></clip></camera></sensor></link></model></sdf>"
    )


def ghost_sdf(course, hz, segmentation=False):
    """The car's visuals only, static, carrying a ZED-matched camera."""
    root = ET.parse(
        REPO / "jetson/cfr_arduino_bridge/worlds" / COURSES[course]["sdf"]
    ).getroot()
    car = [m for m in root.iter("model") if m.get("name") == "slash"][0]
    links = []
    for link in car.findall("link"):
        vis = link.findall("visual")
        if not vis:
            continue
        body = "".join(ET.tostring(v, encoding="unicode") for v in vis)
        if link.get("name") == "chassis":
            x, y, z = ZED_OFFSET
            body += (
                f'<sensor name="zedview" type="{"rgbd_camera" if segmentation else "camera"}"><pose>{x} {y} {z} 0 0 0</pose>'
                f"<always_on>1</always_on><update_rate>{hz}</update_rate><topic>/video/zed</topic>"
                "<camera><horizontal_fov>1.91986</horizontal_fov>"
                "<image><width>640</width><height>360</height><format>R8G8B8</format></image>"
                "<clip><near>0.2</near><far>20</far></clip></camera></sensor>"
            )
        links.append(
            f'<link name="{link.get("name")}"><pose>{link.findtext("pose") or "0 0 0 0 0 0"}</pose>{body}</link>'
        )
    return f'<sdf version="1.9"><model name="ghost"><static>true</static>{"".join(links)}</model></sdf>'


# ------------------------------------------------------------------ the node


def make_node(args):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from ros_gz_interfaces.srv import (
        ControlWorld,
        DeleteEntity,
        SetEntityPose,
        SpawnEntity,
    )
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import Bool

    from cfr_interfaces.msg import DriveCommand, HoopStatus, LapCount

    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

    class Film(Node):
        def __init__(self):
            super().__init__(
                "driver_video",
                parameter_overrides=[
                    rclpy.parameter.Parameter(
                        "use_sim_time", rclpy.Parameter.Type.BOOL, True
                    )
                ],
            )
            self.world = COURSES[args.course]["world"]
            self.img, self.stamp = {}, {}
            # Reliable, not sensor-data QoS: a best-effort 1.5 MB image is
            # fragmented and dropped under load, and in replay each pose is
            # rendered exactly once, so a drop would stall the film.
            for k in ("chase", "map"):
                self.stamp[k] = -1.0
                self.create_subscription(
                    Image, f"/video/{k}", lambda m, k=k: self.on_img(k, m), 10
                )
            self.stamp["zed"] = -1.0
            zed_topic = getattr(args, "zed_topic", "") or (
                "/video/zed/image" if args.segmentation else "/video/zed"
            )
            self.create_subscription(
                Image,
                zed_topic,
                lambda m: self.on_img("zed", m),
                qos_profile_sensor_data if getattr(args, "zed_topic", "") else 10,
            )
            self.cloud = None
            if args.segmentation:
                self.create_subscription(
                    PointCloud2,
                    "/zed/zed_node/point_cloud/cloud_registered"
                    if getattr(args, "zed_topic", "")
                    else "/video/zed/points",
                    lambda m: setattr(self, "cloud", m),
                    1,
                )
            self.pose = None
            self.speed = 0.0
            self.cmd = None
            self.hoops = None
            self.laps = None
            self.done = False
            self.pose_t = None
            self.yaw_rate = 0.0
            self.create_subscription(
                PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
            )
            self.create_subscription(
                Odometry,
                "/zed/zed_node/odom",
                self.on_odom,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                DriveCommand,
                "/drive_cmd",
                lambda m: setattr(self, "cmd", (m.steering, m.velocity)),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                HoopStatus,
                "/hoop_monitor/status",
                lambda m: setattr(self, "hoops", m),
                10,
            )
            self.create_subscription(
                LapCount, "/lap_counter/count", lambda m: setattr(self, "laps", m), 10
            )
            self.create_subscription(
                Bool,
                "/lap_counter/done",
                lambda m: setattr(self, "done", bool(m.data)),
                latched,
            )
            w = f"/world/{self.world}"
            self.cli = {
                "control": self.create_client(ControlWorld, f"{w}/control"),
                "pose": self.create_client(SetEntityPose, f"{w}/set_pose"),
                "create": self.create_client(SpawnEntity, f"{w}/create"),
                "remove": self.create_client(DeleteEntity, f"{w}/remove"),
            }
            self.types = dict(
                control=ControlWorld,
                pose=SetEntityPose,
                create=SpawnEntity,
                remove=DeleteEntity,
            )

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_img(self, k, m):
            self.img[k] = m
            self.stamp[k] = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

        def on_pose(self, m):
            p = m.pose.position
            r, pch, y = rpy_of(m.pose.orientation)
            self.pose = (p.x, p.y, p.z, y, pch, r)
            self.pose_t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

        def on_odom(self, m):
            self.speed = m.twist.twist.linear.x
            self.yaw_rate = m.twist.twist.angular.z

        def pose_at(self, t):
            """The car's pose carried forward to sim time t.

            The pose publisher runs at 30 Hz, so the latest pose can be 33 ms
            old -- 17 cm at 5 m/s, which a camera placed from it shows.
            """
            x, y, z, yaw, pitch, roll = self.pose
            dt = min(max(t - self.pose_t, 0.0), 0.1)
            yaw_mid = yaw + 0.5 * self.yaw_rate * dt
            return (
                x + self.speed * math.cos(yaw_mid) * dt,
                y + self.speed * math.sin(yaw_mid) * dt,
                z,
                yaw + self.yaw_rate * dt,
                pitch,
                roll,
            )

        def wait_until(self, cond, timeout):
            t0 = time.monotonic()
            while not cond() and time.monotonic() - t0 < timeout:
                rclpy.spin_once(self, timeout_sec=0.05)
            return cond()

        def call(self, srv_type, name, req, timeout=60.0):
            c = self.create_client(srv_type, name)
            if not c.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f"{name} is not up")
            f = c.call_async(req)
            rclpy.spin_until_future_complete(self, f, timeout_sec=timeout)
            self.destroy_client(c)
            if f.result() is None:
                raise RuntimeError(f"{name} did not answer")
            return f.result()

        def gz(self, key, req, wait=True):
            c = self.cli[key]
            if not c.wait_for_service(timeout_sec=60):
                raise RuntimeError(f"gz service {key} is not bridged")
            f = c.call_async(req)
            if wait:
                rclpy.spin_until_future_complete(self, f, timeout_sec=60)
                return f.result()
            return f

        def set_pose(self, name, x, y, z, roll=0.0, pitch=0.0, yaw=0.0, wait=True):
            r = SetEntityPose.Request()
            r.entity.name = name
            r.entity.type = 2  # MODEL
            r.pose.position.x, r.pose.position.y, r.pose.position.z = (
                float(x),
                float(y),
                float(z),
            )
            (
                r.pose.orientation.x,
                r.pose.orientation.y,
                r.pose.orientation.z,
                r.pose.orientation.w,
            ) = quat(roll, pitch, yaw)
            return self.gz("pose", r, wait)

        def spawn(self, name, sdf, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
            r = SpawnEntity.Request()
            r.entity_factory.name = name
            r.entity_factory.sdf = sdf
            r.entity_factory.allow_renaming = False
            p = r.entity_factory.pose
            p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
            (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w) = quat(
                roll, pitch, yaw
            )
            return self.gz("create", r).success

        def remove(self, name):
            r = DeleteEntity.Request()
            r.entity.name = name
            r.entity.type = 2
            return self.gz("remove", r).success

        def control(self, pause=None, steps=0):
            r = ControlWorld.Request()
            if pause is not None:
                r.world_control.pause = pause
            if steps:
                r.world_control.pause = True
                r.world_control.multi_step = int(steps)
            return self.gz("control", r).success

    rclpy.init()
    return Film()


def to_bgr(msg):
    import cv2

    a = (
        np.frombuffer(msg.data, np.uint8)
        .reshape(msg.height, msg.step)[:, : msg.width * 3]
        .reshape(msg.height, msg.width, 3)
    )
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else a.copy()


def segmentation_view(msg, pitch, roll):
    """Classify the organized RGB-D cloud with the car's compiled segmenter."""
    import sys

    module_dir = str(REPO / "jetson/cfr_arduino_bridge/src")
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    import cloud_segmentation as seg

    if msg.height < 2:
        raise ValueError("segmentation view needs an organized point cloud")
    fields = {field.name: field for field in msg.fields}
    order = ">" if msg.is_bigendian else "<"
    shape = (msg.height, msg.width)
    strides = (msg.row_step, msg.point_step)

    def field(name, dtype):
        return np.ndarray(
            shape,
            dtype=order + dtype,
            buffer=msg.data,
            offset=fields[name].offset,
            strides=strides,
        )

    xyz = np.stack([field(name, "f4") for name in "xyz"], axis=-1)
    color_name = next((name for name in ("rgb", "rgba") if name in fields), None)
    rgb = field(color_name, "u4").reshape(-1) if color_name else None
    labels = seg.segment(xyz.reshape(-1, 3), pitch, roll, rgb=rgb).labels.reshape(shape)
    # BGR values match cloud_segmentation_node.cpp's class colors.
    palette = np.array(
        [(96, 128, 96), (40, 50, 230), (240, 60, 240), (230, 220, 40), (255, 110, 70)],
        dtype=np.uint8,
    )
    out = np.full((*shape, 3), 24, dtype=np.uint8)
    known = labels < len(palette)
    out[known] = palette[labels[known]]
    return out


def latest_segmentation(node, pose, cache):
    cloud = node.cloud
    if cloud is None:
        return None
    stamp = (cloud.header.stamp.sec, cloud.header.stamp.nanosec)
    if cache["stamp"] != stamp:
        cache["image"] = segmentation_view(cloud, pose[4], pose[5])
        cache["stamp"] = stamp
    return cache["image"]


# ------------------------------------------------------------------ minimap


class Minimap:
    """A photo of the course from straight above, and world -> pixel for it."""

    HFOV = 1.2

    def __init__(self, node, course, paused):
        import cv2

        x0, x1, y0, y1 = course_bounds(course)
        wx, wy = x1 - x0, y1 - y0
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.w = 1000
        self.h = int(round(self.w * wy / wx))
        height = (wx / 2) / math.tan(self.HFOV / 2) + 1.0
        self.mpp = 2 * height * math.tan(self.HFOV / 2) / self.w  # at z = 0
        # Pitch +90 deg looks straight down; yaw +90 deg then puts +x to the
        # right of the image and +y up.
        node.spawn(
            "mapcam",
            camera_sdf(
                "mapcam",
                "/video/map",
                self.w,
                self.h,
                self.HFOV,
                5,
                far=height + 5,
                near=1.0,
            ),
            self.cx,
            self.cy,
            height,
            0.0,
            math.pi / 2,
            math.pi / 2,
        )
        before = node.stamp["map"]
        for _ in range(40):
            if paused:
                node.control(steps=200)
            if node.wait_until(lambda: node.stamp["map"] > before, 15.0):
                break
        if node.stamp["map"] <= before:
            raise RuntimeError("the map camera never rendered")
        self.img = to_bgr(node.img["map"])
        node.remove("mapcam")
        scale = min(320 / self.w, 300 / self.h)
        self.small = cv2.resize(
            self.img,
            (int(self.w * scale), int(self.h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        self.scale = scale

    def px(self, x, y):
        return (
            int((self.w / 2 + (x - self.cx) / self.mpp) * self.scale),
            int((self.h / 2 - (y - self.cy) / self.mpp) * self.scale),
        )


# ------------------------------------------------------------------ frames


class Compositor:
    def __init__(self, out, title, subtitle, note, minimap, segmentation=False):
        self.out = Path(out)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        self.title, self.subtitle, self.note = title, subtitle, note
        self.map = minimap
        self.segmentation = segmentation
        self.trail = []
        self.n = 0
        self.durations = []

    def frame(self, chase, zed, pose, lines, banner="", duration=None, segmented=None):
        import cv2

        c = np.full((H, W + (320 if self.segmentation else 0), 3), 24, np.uint8)
        c[:540, :960] = cv2.resize(chase, (960, 540))
        if zed is not None:
            c[:180, 960:1280] = cv2.resize(zed, (320, 180))
            cv2.putText(
                c,
                "ZED view",
                (966, 172),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if self.segmentation:
            if segmented is not None:
                c[:180, 1280:1600] = cv2.resize(segmented, (320, 180))
            cv2.putText(
                c,
                "Point cloud segmentation",
                (1286, 172),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            legend = (
                ("ground", (96, 128, 96)),
                ("obstacle", (40, 50, 230)),
                ("hoop", (240, 60, 240)),
                ("car wash", (230, 220, 40)),
                ("overhead", (255, 110, 70)),
            )
            for i, (label, color) in enumerate(legend):
                y = 212 + 28 * i
                cv2.rectangle(c, (1290, y - 11), (1306, y + 5), color, -1)
                cv2.putText(
                    c,
                    label,
                    (1316, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        mp = self.map.small.copy()
        if pose is not None:
            self.trail.append(self.map.px(pose[0], pose[1]))
        if len(self.trail) > 1:
            cv2.polylines(
                mp,
                [np.array(self.trail, np.int32)],
                False,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        if pose is not None:
            x, y, yaw = pose[0], pose[1], pose[3]
            p = self.map.px(x, y)
            cv2.circle(mp, p, 4, (0, 255, 0), -1)
            cv2.line(
                mp,
                p,
                self.map.px(x + 1.2 * math.cos(yaw), y + 1.2 * math.sin(yaw)),
                (0, 255, 0),
                2,
            )
        ox = 960 + (320 - mp.shape[1]) // 2
        c[190 : 190 + mp.shape[0], ox : ox + mp.shape[1]] = mp
        y0 = 190 + mp.shape[0] + 30
        for i, s in enumerate(lines):
            cv2.putText(
                c,
                s,
                (970, y0 + 28 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            c,
            self.title,
            (16, 582),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            c,
            self.subtitle,
            (16, 614),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        if self.note:
            cv2.putText(
                c,
                self.note,
                (16, 640),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (140, 170, 255),
                1,
                cv2.LINE_AA,
            )
        if banner:
            cv2.putText(
                c,
                banner,
                (16, 692),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(
            str(self.out / "frames" / f"{self.n:06d}.jpg"),
            c,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        self.durations.append(duration)
        self.n += 1

    def finish(self, stamps=None, first=0, hold=2.0, summary=None):
        """The concat list: sim-time durations, so playback is real time."""
        d = list(self.durations)
        if stamps is not None:
            d = [max(0.01, b - a) for a, b in zip(stamps, stamps[1:])] + [1 / 15]
        d[-1] = hold
        with open(self.out / "list.txt", "w") as f:
            for i in range(first, self.n):
                f.write(f"file 'frames/{i:06d}.jpg'\nduration {d[i]:.4f}\n")
            f.write(f"file 'frames/{self.n - 1:06d}.jpg'\n")
        (self.out / "summary.json").write_text(json.dumps(summary or {}, indent=1))


class Follow:
    """A level chase camera that trails the car's heading, smoothed."""

    def __init__(self, back, up, tau=0.2):
        self.back, self.up, self.tau = back, up, tau
        self.yaw = self.z = None

    def update(self, pose, dt):
        x, y, z, yaw = pose[:4]
        if self.yaw is None:
            self.yaw, self.z = yaw, z
        a = 1 - math.exp(-max(dt, 0.0) / self.tau)
        self.yaw += a * math.remainder(yaw - self.yaw, math.tau)
        self.z += a * (z - self.z)
        return (
            x - self.back * math.cos(self.yaw),
            y - self.back * math.sin(self.yaw),
            self.z + self.up,
            0.0,
            0.0,
            self.yaw,
        )


def set_layout(node, seed):
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    from std_srvs.srv import Trigger

    req = SetParameters.Request()
    req.parameters = [
        Parameter(
            name="seed",
            value=ParameterValue(
                type=ParameterType.PARAMETER_INTEGER, integer_value=int(seed)
            ),
        )
    ]
    node.call(SetParameters, "/obstacle_randomizer/set_parameters", req)
    res = node.call(Trigger, "/obstacle_randomizer/randomize", Trigger.Request(), 150)
    node.get_logger().info(f"layout: {res.message}")


def set_target_laps(node, laps):
    """speed_course.launch.py pins lap_counter to 3 laps whatever laps:= says;
    the driver stops after its own config's count, so match it at runtime."""
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters

    req = SetParameters.Request()
    req.parameters = [
        Parameter(
            name="target_laps",
            value=ParameterValue(
                type=ParameterType.PARAMETER_INTEGER, integer_value=int(laps)
            ),
        )
    ]
    ok = (
        node.call(SetParameters, "/lap_counter/set_parameters", req)
        .results[0]
        .successful
    )
    node.get_logger().info(f"lap_counter target_laps {laps}: {ok}")


def progress_line(node, course):
    if course == "obstacle":
        if node.hoops is None:
            return "hoops 0/3"
        return f"hoops {sum(node.hoops.passed)}/{len(node.hoops.passed)}" + (
            "  MISSED" if node.hoops.any_missed else ""
        )
    if node.laps is not None:
        return f"laps {node.laps.laps}/{node.laps.target}"
    return ""


# ------------------------------------------------------------------ modes


def run_gazebo(args):
    import rclpy
    from std_srvs.srv import SetBool

    n = make_node(args)
    log = n.get_logger().info
    if not n.wait_until(lambda: n.pose is not None, 120):
        raise RuntimeError("no /zed/zed_node/pose -- is the course up?")
    if args.course == "obstacle":
        set_layout(n, args.seed)
        x, y, yaw = car_start("obstacle")
        body = json.dumps(
            {"x": x, "y": y, "heading": math.degrees(yaw), "z": 0.0}
        ).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:9003/api/sim/teleport",
            body,
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=150) as r:
            log(f"teleport: {r.read()[:80]}")
    if args.laps:
        set_target_laps(n, args.laps)
    t0 = n.now()
    n.wait_until(lambda: n.now() - t0 > 2.0, 120)  # settle
    mm = Minimap(n, args.course, paused=False)
    back, up = COURSES[args.course]["chase"]
    follow = Follow(back, up)
    n.spawn(
        "chasecam",
        camera_sdf("chasecam", "/video/chase", 960, 540, 1.25, args.hz, pitch=0.42),
        *follow.update(n.pose, 0),
    )
    use_zedcam = not args.zed_topic
    if use_zedcam:
        n.spawn(
            "zedcam",
            camera_sdf(
                "zedcam",
                "/video/zed",
                640,
                360,
                1.91986,
                args.hz,
                near=0.2,
                far=20,
                rgbd=args.segmentation,
            ),
        )
    # From here the capture drives the clock.  Moving the cameras while the
    # world runs free lands each move a varying few ms late, so the car
    # creeps across the frame and snaps back; moving them between steps, and
    # waiting for the move, puts them on the car for every render.
    n.control(pause=True)
    step_s = args.step_ms / 1000.0

    def place_cameras(t):
        p = n.pose_at(t)
        n.set_pose("chasecam", *follow.update(p, step_s))
        if use_zedcam:
            x, y, z, yaw, pitch, roll = p
            o = rot(roll, pitch, yaw) @ np.array(ZED_OFFSET)
            n.set_pose("zedcam", x + o[0], y + o[1], z + o[2], roll, pitch, yaw)

    def step():
        t, w = n.now(), time.monotonic()
        place_cameras(t + step_s / 2)  # a render lands somewhere in the step
        n.control(steps=args.step_ms)
        n.wait_until(lambda: n.now() >= t + step_s - 1e-4, 10.0)
        # The ROS nodes run in wall time: give every step at least step/rtf of
        # it.  Pacing the average instead lets steps after a slow render run
        # back to back, and the segmenter falls behind (stale-cloud holds).
        while time.monotonic() - w < step_s / args.rtf:
            rclpy.spin_once(n, timeout_sec=0.005)

    def advance(seconds):
        end = n.now() + seconds
        while n.now() < end:
            step()

    comp = Compositor(
        args.out, args.title, args.subtitle, args.note, mm, args.segmentation
    )
    seg_cache = {"stamp": None, "image": None}
    stamps, rec = [], {"on": False, "t_go": None, "status": ""}
    maxv = [0.0]

    def on_chase(msg):
        if not rec["on"] or n.pose is None:
            return
        maxv[0] = max(maxv[0], n.speed)
        t = n.now() - rec["t_go"]
        lines = [
            f"t = {t:6.1f} s (sim time)",
            f"speed {n.speed:5.2f} m/s  max {maxv[0]:4.2f}",
        ]
        if n.cmd:
            lines.append(f"cmd v {n.cmd[1]:5.2f} m/s  steer {n.cmd[0]:+5.2f}")
        p = progress_line(n, args.course)
        if p:
            lines.append(p)
        zed = to_bgr(n.img["zed"]) if "zed" in n.img else None
        segmented = (
            latest_segmentation(n, n.pose, seg_cache) if args.segmentation else None
        )
        comp.frame(to_bgr(msg), zed, n.pose, lines, rec["status"], segmented=segmented)
        stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    orig = n.on_img

    def on_img(k, m):
        orig(k, m)
        if k == "chase":
            on_chase(m)

    n.on_img = on_img  # the subscriptions look on_img up at call time
    for _ in range(int(5.0 / step_s)):
        if n.stamp["chase"] > 0:
            break
        step()
    else:
        raise RuntimeError("the chase camera never rendered")
    if args.segmentation and n.cloud is None:
        for _ in range(int(5.0 / step_s)):
            step()
            if n.cloud is not None:
                break
        else:
            raise RuntimeError("the ZED point cloud never arrived")

    rec.update(on=True, t_go=n.now(), status="waiting for the start")
    advance(1.5)
    if args.course == "obstacle":
        n.call(SetBool, "/obstacle_randomizer/start_signal", SetBool.Request(data=True))
        manual = "/obstacle_racer/manual_start"
    else:
        manual = "/formula_one/manual_start"
    start_xy = n.pose[:2]
    t_green = n.now()
    moved_at, last_xy, last_move = None, start_xy, n.now()
    kicked = args.course != "obstacle"
    if kicked:
        n.call(SetBool, manual, SetBool.Request(data=True))
    outcome = "timeout"
    while True:
        # --timeout is driving time: the wait for the start (up to the manual
        # fallback at 10 s) must not eat into a short clip.
        if moved_at is None and n.now() - t_green > 30.0:
            outcome = "no_start"
            break
        if moved_at is not None and n.now() - moved_at >= args.timeout:
            break
        step()
        x, y, _, _, pitch, roll = n.pose
        if math.hypot(x - last_xy[0], y - last_xy[1]) > 0.15:
            last_xy, last_move = (x, y), n.now()
            if moved_at is None:
                moved_at = n.now()
                rec["status"] = ""
        if not kicked and n.now() - t_green > 10:
            log("start signal not seen after 10 s; manual start")
            n.call(SetBool, manual, SetBool.Request(data=True))
            kicked, last_move = True, n.now()
        if n.done:
            outcome = "finish"
            break
        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            outcome = "rollover"
            break
        if n.now() - last_move > args.stuck_s:
            outcome = "stuck"
            break
    t_end = n.now()
    rec["status"] = {
        "finish": "FINISHED",
        "rollover": "ROLLED OVER",
        "stuck": f"STUCK - no progress for {args.stuck_s:.0f} s",
        "timeout": f"TIME LIMIT - {args.timeout:.0f} s of driving",
        "no_start": "NEVER STARTED",
    }[outcome]
    advance(args.tail)
    rec["on"] = False
    first = 0
    if moved_at is not None:  # trim the idle wait for the start
        first = next((i for i, s in enumerate(stamps) if s >= moved_at - 1.0), 0)
    summary = dict(
        mode="gazebo",
        course=args.course,
        outcome=outcome,
        time=(t_end - (moved_at or t_green)),
        max_speed=maxv[0],
        end=list(n.pose),
        progress=progress_line(n, args.course),
        frames=comp.n,
        first_frame=first,
    )
    comp.finish(stamps=stamps, first=first, hold=1 / 15, summary=summary)
    log(json.dumps(summary))
    rclpy.shutdown()


def run_replay(args):
    import rclpy

    d = np.load(args.track)
    meta = json.loads(Path(str(args.track) + ".json").read_text())
    t, pose = d["t"], d["pose"]
    dt = float(meta["dt"])
    steps = int(round(dt * 1000))
    hz = round(1 / dt)
    n = make_node(args)
    log = n.get_logger().info
    if args.course == "obstacle" and meta.get("seed") is not None:
        set_layout(n, meta["seed"])
    n.set_pose("slash", 80.0, 80.0, 0.05)  # the real car, out of shot
    n.control(pause=True)
    mm = Minimap(n, args.course, paused=True)
    x, y, z, yaw, pitch, roll = pose[0]
    log(
        f"ghost: {n.spawn('ghost', ghost_sdf(args.course, hz, args.segmentation), x, y, z, roll, pitch, yaw)}"
    )
    back, up = COURSES[args.course]["chase"]
    follow = Follow(back, up)
    log(
        f"chasecam: {n.spawn('chasecam', camera_sdf('chasecam', '/video/chase', 960, 540, 1.25, hz, pitch=0.42), *follow.update(pose[0], 0))}"
    )
    comp = Compositor(
        args.out, args.title, args.subtitle, args.note, mm, args.segmentation
    )
    seg_cache = {"stamp": None, "image": None}
    total = len(t) if not args.limit else min(args.limit, len(t))
    label, ptotal, vlabel = (
        meta["progress_label"],
        meta["progress_total"],
        meta.get("vcmd_label", "cmd v"),
    )
    tick = time.monotonic()
    for k in range(-5, total):  # five warm-up frames, not written
        i = max(k, 0)
        x, y, z, yaw, pitch, roll = pose[i]
        n.set_pose("ghost", x, y, z, roll, pitch, yaw)
        n.set_pose("chasecam", *follow.update(pose[i], dt if k > -5 else 0))
        before = n.stamp["chase"]
        cloud_before = n.cloud.header.stamp if n.cloud is not None else None
        n.control(steps=steps)
        for _ in range(6):
            if n.wait_until(
                lambda: (
                    n.stamp["chase"] > before
                    and (
                        not args.segmentation
                        or (
                            n.cloud is not None and n.cloud.header.stamp != cloud_before
                        )
                    )
                ),
                8.0,
            ):
                break
            n.control(steps=steps)  # a dropped frame: render this pose again
        else:
            raise RuntimeError("the chase camera or ZED point cloud stopped rendering")
        for _ in range(3):
            rclpy.spin_once(n, timeout_sec=0.0)
        if k < 0:
            continue
        lines = [
            f"t = {t[i]:6.1f} s",
            f"speed {d['v'][i]:5.2f} m/s  max {d['v'][: i + 1].max():4.2f}",
            f"{vlabel} {d['vcmd'][i]:5.2f} m/s  steer {d['steer'][i]:+5.2f}",
            f"{'laps' if label == 'lap' else label} {d['progress'][i]}/{ptotal}",
        ]
        last = i == total - 1
        banner = ""
        if last and total == len(t):
            banner = (
                f"FINISHED  {meta['time']:.1f} s"
                if meta.get("finished")
                else f"{meta.get('outcome', 'ended').upper()} at {meta['time']:.1f} s"
            )
        zed = to_bgr(n.img["zed"]) if "zed" in n.img else None
        segmented = (
            latest_segmentation(n, pose[i], seg_cache) if args.segmentation else None
        )
        comp.frame(
            to_bgr(n.img["chase"]),
            zed,
            pose[i],
            lines,
            banner,
            duration=dt,
            segmented=segmented,
        )
        if k % 50 == 0:
            rate = (k + 1) / max(time.monotonic() - tick, 1e-6)
            print(
                f"frame {k}/{total}  {rate:.1f} fps  eta {(total - k) / max(rate, 1e-6) / 60:.1f} min",
                flush=True,
            )
    summary = dict(
        mode="replay",
        course=args.course,
        frames=comp.n,
        **{k: meta.get(k) for k in ("outcome", "time", "finished", "policy", "seed")},
    )
    comp.finish(hold=2.0, summary=summary)
    log(json.dumps(summary))
    rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("patch-world")
    p.add_argument("--course", choices=COURSES, required=True)
    p.add_argument("--rtf", type=float, default=0.1)
    p.add_argument("--sensors-plugin", action="store_true")
    for name in ("gazebo", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--course", choices=COURSES, required=True)
        p.add_argument(
            "--segmentation",
            action="store_true",
            help="show the classified ZED point cloud",
        )
        p.add_argument("--out", required=True)
        p.add_argument("--title", default="")
        p.add_argument("--subtitle", default="")
        p.add_argument("--note", default="")
        if name == "gazebo":
            p.add_argument("--seed", type=int, default=104)
            p.add_argument(
                "--timeout", type=float, default=180.0, help="sim s of driving"
            )
            p.add_argument("--stuck-s", type=float, default=12.0)
            p.add_argument("--tail", type=float, default=3.0)
            p.add_argument("--hz", type=int, default=15)
            p.add_argument(
                "--rtf", type=float, default=0.1, help="wall pacing of the steps"
            )
            p.add_argument(
                "--step-ms", type=int, default=10, help="sim ms per camera update"
            )
            p.add_argument(
                "--zed-topic",
                default="",
                help="use the rendered ZED instead of a film camera",
            )
            p.add_argument(
                "--laps",
                type=int,
                default=0,
                help="set lap_counter's target (0 leaves it)",
            )
        else:
            p.add_argument("--track", required=True)
            p.add_argument(
                "--limit", type=int, default=0, help="render only the first N frames"
            )
    args = ap.parse_args()
    {"patch-world": patch_world, "gazebo": run_gazebo, "replay": run_replay}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
