#!/usr/bin/env python3
"""Render the point-cloud segmentation test fixtures in Gazebo.

For every scenario in `cfr_arduino_bridge/test/segmentation_scenarios.py`
this renders what the ZED sees from that pose -- depth, color, and a
per-pixel label saying which part of the course each pixel belongs to -- and
writes it to `test/fixtures/segmentation/<name>.npz`. The unit tests then
score `cloud_segmentation` against those labels without needing Gazebo.

Why a camera rig and not the car: the rig is a static model carrying the
same rgbd sensor `sensors_world.py` gives the car, plus a segmentation
camera co-located with it. Static models go exactly where `set_pose` puts
them, so a scenario on a 19% ramp does not roll back down it while the
frame renders, and the pose stored with the fixture is the pose the frame
was rendered from. The car's own pitch and roll come from the course: each
wheel is dropped onto the highest drivable surface under it (ramp, deck,
helix, bank, pothole board, recess, bump, gravel) and the chassis is fitted
through the four contacts.

Where the labels come from: every visual in the world is given a
`gz::sim::systems::Label` plugin carrying its PART_* id, in a copy of the
world written for this run only. The committed worlds are not touched.

Run inside the Gazebo image, with the repository mounted read-write:

    docker run --rm -v "$PWD:/repo" -w /repo \
        unfrobotics/docker-ros2-jazzy-gz-rviz2:latest \
        bash -c 'source /opt/ros/jazzy/setup.bash && LIBGL_ALWAYS_SOFTWARE=1 \
            python3 jetson/scripts/capture_segmentation_fixtures.py'

`--only name ...` re-renders a subset; `--preview DIR` also writes a PNG per
fixture (color, labels, depth side by side) for eyeballing a new scenario.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "cfr_arduino_bridge"
sys.path.insert(0, str(PACKAGE / "test"))
sys.path.insert(0, str(PACKAGE / "launch"))

import segmentation_scenarios as sc  # noqa: E402
from sensors_world import SENSORS_CAMERA, SENSORS_SYSTEM, SYSTEM_MARKER  # noqa: E402

WORLDS = {
    "speed": ("speed_course.sdf", "cfr_speed_course"),
    "obstacle": ("obstacle_course.sdf", "cfr_obstacle_course"),
}
OUT = PACKAGE / "test" / "fixtures" / "segmentation"

# Parts a wheel may stand on. Everything else -- bales, rails, the bank's
# back wall, which is part of the bank mesh but faces sideways -- is filtered
# out by keeping only upward-facing triangles.
DRIVABLE = {
    sc.PART_FLOOR,
    sc.PART_RAMP,
    sc.PART_HELIX,
    sc.PART_BANK,
    sc.PART_POTHOLE_BOARD,
    sc.PART_POTHOLE_BUMP,
    sc.PART_GRAVEL,
    sc.PART_SMALL_RAMP,
    sc.PART_CARWASH_BASE,
}
# Highest a surface may be for a car that is not `elevated`: above the
# pothole bumps and the bank's top edge, below the deck over the tunnel.
GROUND_LEVEL_LIMIT = 0.30

RIG = "segmentation_rig"


# ---------------------------------------------------------------- geometry


def rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SDF's fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    m = matrix
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(
        math.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2]
    )
    y = math.copysign(
        math.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0]
    )
    z = math.copysign(
        math.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1]
    )
    return x, y, z, w


def parse_pose(element) -> np.ndarray:
    """4x4 transform of an SDF <pose> child, identity if absent."""
    transform = np.eye(4)
    pose = element.find("pose") if element is not None else None
    if pose is not None and pose.text:
        x, y, z, roll, pitch, yaw = (float(v) for v in pose.text.split())
        transform[:3, :3] = rotation(roll, pitch, yaw)
        transform[:3, 3] = (x, y, z)
    return transform


def stl_triangles(path: Path) -> np.ndarray:
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, 80)
    record = np.dtype([("normal", "<3f4"), ("v", "<9f4"), ("attr", "<u2")])
    return (
        np.frombuffer(data, dtype=record, count=count, offset=84)["v"]
        .reshape(-1, 3, 3)
        .astype(float)
    )


def box_triangles(size) -> np.ndarray:
    sx, sy, sz = (s / 2 for s in size)
    corners = np.array(
        [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]
    )
    # Two triangles per face; winding does not matter, only the top faces are
    # used and they are picked by normal, not by order.
    faces = [
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
    ]
    tris = []
    for a, b, c, d in faces:
        tris += [corners[[a, b, c]], corners[[a, c, d]]]
    return np.array(tris)


class Surface:
    """Height of the highest drivable visual under a plan point."""

    def __init__(self, triangles: np.ndarray):
        normal = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        # Faces within ~45 degrees of up, whichever way they were wound.
        up = np.abs(normal[:, 2]) > 0.7 * np.linalg.norm(normal, axis=1)
        self.tris = triangles[up]
        self.lo = self.tris[:, :, :2].min(axis=1)
        self.hi = self.tris[:, :, :2].max(axis=1)

    def height(self, x: float, y: float, limit: float) -> float:
        near = (
            (self.lo[:, 0] <= x)
            & (self.hi[:, 0] >= x)
            & (self.lo[:, 1] <= y)
            & (self.hi[:, 1] >= y)
        )
        best = 0.0
        for a, b, c in self.tris[near]:
            ux, uy = b[0] - a[0], b[1] - a[1]
            vx, vy = c[0] - a[0], c[1] - a[1]
            area = ux * vy - uy * vx
            if abs(area) < 1e-12:
                continue
            wx, wy = x - a[0], y - a[1]
            s = (wx * vy - wy * vx) / area
            t = (ux * wy - uy * wx) / area
            if s < -1e-9 or t < -1e-9 or s + t > 1 + 1e-9:
                continue
            z = a[2] + s * (b[2] - a[2]) + t * (c[2] - a[2])
            if best < z <= limit:
                best = z
        return best


def car_pose(scenario: sc.Scenario, surface: Surface):
    """(position, rotation) of the chassis origin, standing on the course."""
    yaw = math.radians(scenario.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    limit = 10.0 if scenario.elevated else GROUND_LEVEL_LIMIT
    contacts = {}
    for wheel, (ox, oy) in sc.WHEEL_CONTACTS.items():
        wx = scenario.x + c * ox - s * oy
        wy = scenario.y + s * ox + c * oy
        z = scenario.wheels.get(wheel)
        contacts[wheel] = z if z is not None else surface.height(wx, wy, limit)
    front = (contacts["front_left"] + contacts["front_right"]) / 2
    rear = (contacts["rear_left"] + contacts["rear_right"]) / 2
    left = (contacts["front_left"] + contacts["rear_left"]) / 2
    right = (contacts["front_right"] + contacts["rear_right"]) / 2
    wheelbase = 2 * sc.WHEEL_CONTACTS["front_left"][0]
    track = 2 * sc.WHEEL_CONTACTS["front_left"][1]
    # Least-squares plane through four symmetric contacts. Pitch is SDF's,
    # positive nose-down; roll positive raises the left side.
    pitch = -math.atan2(front - rear, wheelbase)
    roll = math.atan2(left - right, track)
    z = sum(contacts.values()) / 4
    return np.array([scenario.x, scenario.y, z]), rotation(roll, pitch, yaw), contacts


def helix_scenario(scenario: sc.Scenario, gen) -> sc.Scenario:
    """Resolve a helix scenario's (x, y, yaw) from how far down it is."""
    fraction = sc.HELIX_FRACTIONS[scenario.name]
    cx, cy = gen.to_world(*gen.HELIX_CENTRE)
    radius = (gen.HELIX_INNER_R + gen.HELIX_OUTER_R) / 2 * gen.FOOT
    angle = (
        math.radians(gen.HELIX_START_DEG)
        + math.pi
        + math.radians(gen.HELIX_SWEEP_DEG) * fraction
    )
    return sc.Scenario(
        scenario.name,
        scenario.course,
        cx + radius * math.cos(angle),
        cy + radius * math.sin(angle),
        math.degrees(angle + math.pi / 2),
        scenario.note,
        scenario.wheels,
        scenario.moves,
        scenario.expect,
        scenario.elevated,
    )


# ------------------------------------------------------------------- world


def camera_geometry() -> dict:
    camera = ET.fromstring(SENSORS_CAMERA)
    return {
        "hfov": float(camera.find("camera/horizontal_fov").text),
        "width": int(camera.find("camera/image/width").text),
        "height": int(camera.find("camera/image/height").text),
        "near": float(camera.find("camera/clip/near").text),
        "far": float(camera.find("camera/clip/far").text),
        "offset": [float(v) for v in camera.find("pose").text.split()],
    }


def rig_model() -> str:
    """The rgbd camera the car carries, and a label camera matched to it."""
    geometry = camera_geometry()
    optics = (
        f"<horizontal_fov>{geometry['hfov']}</horizontal_fov>"
        f"<image><width>{geometry['width']}</width><height>{geometry['height']}</height>"
        "{format}</image>"
        f"<clip><near>{geometry['near']}</near><far>{geometry['far']}</far></clip>"
    )
    return f"""
    <model name="{RIG}"><static>true</static><pose>0 0 30 0 0 0</pose>
      <link name="link">
        <sensor name="rgbd" type="rgbd_camera">
          <always_on>1</always_on><update_rate>5</update_rate><topic>/seg/rgbd</topic>
          <camera>{optics.format(format="<format>R8G8B8</format>")}</camera>
        </sensor>
        <sensor name="labels" type="segmentation">
          <always_on>1</always_on><update_rate>5</update_rate><topic>/seg/labels</topic>
          <camera><segmentation_type>semantic</segmentation_type>{optics.format(format="")}</camera>
        </sensor>
      </link>
    </model>"""


def labeled_world(course: str) -> tuple[str, dict, list]:
    """World text with every visual labeled and the car swapped for the rig.

    Also returns the original pose of every movable model, so a scenario
    that moves one can put it back, and the world-frame triangles of every
    drivable visual for the surface model.
    """
    text = (PACKAGE / "worlds" / WORLDS[course][0]).read_text()
    text = text.replace(SYSTEM_MARKER, SENSORS_SYSTEM)
    root = ET.fromstring(text)
    world = root.find("world")
    world.remove(next(m for m in world.findall("model") if m.get("name") == "slash"))

    poses, drivable, unlabeled = {}, [], []
    for model in world.findall("model"):
        name = model.get("name")
        model_tf = parse_pose(model)
        poses[name] = (
            model.find("pose").text if model.find("pose") is not None else "0 0 0 0 0 0"
        )
        for link in model.findall("link"):
            link_tf = model_tf @ parse_pose(link)
            for visual in link.findall("visual"):
                part = sc.part_for_visual(name, visual.get("name"))
                if part is None:
                    unlabeled.append(f"{name}/{visual.get('name')}")
                    continue
                plugin = ET.SubElement(
                    visual,
                    "plugin",
                    {
                        "filename": "gz-sim-label-system",
                        "name": "gz::sim::systems::Label",
                    },
                )
                ET.SubElement(plugin, "label").text = str(part)
                if part not in DRIVABLE:
                    continue
                geometry = visual.find("geometry")
                tf = link_tf @ parse_pose(visual)
                if geometry.find("mesh") is not None:
                    uri = geometry.find("mesh/uri").text
                    tris = stl_triangles(PACKAGE / "meshes" / uri.rsplit("/", 1)[1])
                elif geometry.find("box") is not None:
                    tris = box_triangles(
                        [float(v) for v in geometry.find("box/size").text.split()]
                    )
                elif geometry.find("plane") is not None:
                    sx, sy = (
                        float(v) for v in geometry.find("plane/size").text.split()
                    )
                    tris = box_triangles((sx, sy, 0.0))
                elif geometry.find("cylinder") is not None:
                    r = float(geometry.find("cylinder/radius").text)
                    h = float(geometry.find("cylinder/length").text)
                    tris = box_triangles((2 * r, 2 * r, h))
                else:
                    continue
                flat = tris.reshape(-1, 3)
                flat = flat @ tf[:3, :3].T + tf[:3, 3]
                drivable.append(flat.reshape(-1, 3, 3))
    if unlabeled:
        raise RuntimeError(
            "visuals with no entry in segmentation_scenarios.VISUAL_PARTS: "
            + ", ".join(unlabeled[:10])
        )
    world.append(ET.fromstring(rig_model()))
    return ET.tostring(root, encoding="unicode"), poses, np.concatenate(drivable)


# --------------------------------------------------------------- transport


def set_pose(world: str, name: str, position, matrix) -> None:
    qx, qy, qz, qw = quaternion(matrix)
    request = (
        f'name: "{name}" position {{x: {position[0]} y: {position[1]} z: {position[2]}}} '
        f"orientation {{x: {qx} y: {qy} z: {qz} w: {qw}}}"
    )
    for _ in range(5):
        result = subprocess.run(
            [
                "gz",
                "service",
                "-s",
                f"/world/{world}/set_pose",
                "--reqtype",
                "gz.msgs.Pose",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "3000",
                "--req",
                request,
            ],
            capture_output=True,
            text=True,
        )
        if "true" in result.stdout:
            return
        time.sleep(1.0)
    raise RuntimeError(f"set_pose {name} failed: {result.stdout} {result.stderr}")


def pose_from_text(text: str):
    x, y, z, roll, pitch, yaw = (float(v) for v in text.split())
    return (x, y, z), rotation(roll, pitch, yaw)


class Frames:
    """Latest depth, color, label and cloud messages, grouped by stamp."""

    TOPICS = {
        "depth": "/seg/rgbd/depth_image",
        "rgb": "/seg/rgbd/image",
        "labels": "/seg/labels/labels_map",
        "points": "/seg/rgbd/points",
        "info": "/seg/rgbd/camera_info",
    }

    def __init__(self):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2

        rclpy.init()
        self.node = rclpy.create_node("segmentation_capture")
        self.lock = threading.Lock()
        self.by_stamp: dict = {}
        self.info = None
        types = {"depth": Image, "rgb": Image, "labels": Image, "points": PointCloud2}
        for key, kind in types.items():
            self.node.create_subscription(
                kind,
                self.TOPICS[key],
                lambda msg, key=key: self._on(key, msg),
                qos_profile_sensor_data,
            )
        self.node.create_subscription(
            CameraInfo, self.TOPICS["info"], self._on_info, qos_profile_sensor_data
        )
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    def _on(self, key, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.by_stamp.setdefault(stamp, {})[key] = msg
            for old in sorted(self.by_stamp)[:-12]:
                del self.by_stamp[old]

    def _on_info(self, msg):
        self.info = msg

    def latest_stamp(self) -> float:
        with self.lock:
            return max(self.by_stamp, default=-1.0)

    def complete_after(self, after: float, timeout: float = 30.0) -> dict:
        """First frame stamped after `after` with every stream present."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                for stamp in sorted(self.by_stamp):
                    group = self.by_stamp[stamp]
                    # The cloud is optional: it only checks the reprojection,
                    # and at 3.7 MB a frame it is the stream best-effort QoS
                    # drops first.
                    if stamp > after and all(
                        k in group for k in ("depth", "rgb", "labels")
                    ):
                        return {"stamp": stamp, **group}
            time.sleep(0.05)
        with self.lock:
            recent = {
                round(k, 2): sorted(v) for k, v in sorted(self.by_stamp.items())[-4:]
            }
        raise TimeoutError(
            f"no complete frame from the rig after {after:.2f} s -- is the render "
            f"context up? latest stamps and streams: {recent}"
        )


def image_array(msg) -> np.ndarray:
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "mono8": 1}[msg.encoding]
    array = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, channels
    )
    return array[..., ::-1] if msg.encoding == "bgr8" else array


def cloud_xyz(msg) -> np.ndarray:
    offsets = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, msg.point_step
    )
    return np.stack(
        [
            raw[..., offsets[a] : offsets[a] + 4].copy().view(np.float32)[..., 0]
            for a in "xyz"
        ],
        axis=-1,
    )


# ----------------------------------------------------------------- preview


def write_png(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


PALETTE = np.array(
    [
        [0, 0, 0],
        [70, 110, 60],
        [220, 160, 40],
        [240, 240, 240],
        [255, 0, 255],
        [0, 255, 255],
        [255, 60, 60],
        [120, 120, 255],
        [150, 100, 60],
        [255, 255, 0],
        [200, 120, 60],
        [160, 160, 180],
        [140, 70, 20],
        [180, 140, 90],
        [255, 128, 0],
        [110, 110, 110],
        [90, 200, 90],
        [0, 120, 255],
    ]
    + [[255, 255, 255]] * 238,
    dtype=np.uint8,
)


def preview(path: Path, rgb, labels, depth) -> None:
    finite = np.where(np.isfinite(depth), depth, 0.0)
    shade = (255 * (1 - np.clip(finite / 8.0, 0, 1))).astype(np.uint8)
    shade[~np.isfinite(depth)] = 0
    write_png(
        path,
        np.concatenate(
            [rgb[..., :3], PALETTE[labels], np.repeat(shade[..., None], 3, -1)], axis=1
        ),
    )


# -------------------------------------------------------------------- main


def capture_course(course: str, scenarios, args, gen, frames: "Frames") -> None:
    world_text, original_poses, drivable = labeled_world(course)
    surface = Surface(drivable)
    scratch = Path(tempfile.mkdtemp(prefix="cfr_seg_"))
    world_file = scratch / WORLDS[course][0]
    world_file.write_text(world_text)
    world = WORLDS[course][1]

    env = dict(os.environ)
    env["GZ_SIM_RESOURCE_PATH"] = (
        f"{PACKAGE.parent}:{env.get('GZ_SIM_RESOURCE_PATH', '')}"
    )
    log = open(scratch / "gz.log", "w")
    server = subprocess.Popen(
        ["gz", "sim", "-s", "-r", "-v", "2", str(world_file)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    bridge = subprocess.Popen(
        [
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            "/seg/rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/seg/rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/seg/rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/seg/rgbd/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/seg/labels/labels_map@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        with frames.lock:
            frames.by_stamp.clear()
        frames.complete_after(-1.0, timeout=180.0)
        print(f"[{course}] rig is rendering", flush=True)
        for scenario in scenarios:
            if scenario.name in sc.HELIX_FRACTIONS:
                scenario = helix_scenario(scenario, gen)
            for model, (mx, my, myaw) in scenario.moves.items():
                set_pose(
                    world, model, (mx, my, 0.0), rotation(0, 0, math.radians(myaw))
                )
            position, chassis, contacts = car_pose(scenario, surface)
            offset = np.array(sc.CAMERA_OFFSET)
            camera_position = position + chassis @ offset
            set_pose(world, RIG, camera_position, chassis)
            # Two frame periods of settling: a frame already in flight when
            # the pose changed may have rendered from the old one.
            time.sleep(0.3)
            for attempt in range(3):
                try:
                    frame = frames.complete_after(frames.latest_stamp() + 0.35)
                    break
                except TimeoutError:
                    if attempt == 2:
                        raise
                    set_pose(world, RIG, camera_position, chassis)
            for model in scenario.moves:
                set_pose(world, model, *pose_from_text(original_poses[model]))

            depth = image_array(frame["depth"]).astype(np.float32)
            labels = image_array(frame["labels"])[..., 0].copy()
            rgb = image_array(frame["rgb"])[..., :3].copy()
            info = frames.info
            # Gazebo's cloud measures from pixel centers, so its principal
            # point is half a pixel short of the one camera_info reports
            # (319.5 against 320 at 640 wide). Store the one that reproduces
            # the cloud, since the cloud is what the car consumes.
            fx, fy = info.k[0], info.k[4]
            cx, cy = info.k[2] - 0.5, info.k[5] - 0.5
            # The fixture keeps depth, not the cloud: the tests rebuild the
            # cloud with these intrinsics. Check that reproduces Gazebo's.
            rows, cols = np.indices(depth.shape)
            rebuilt = np.stack(
                [depth, -(cols - cx) * depth / fx, -(rows - cy) * depth / fy], axis=-1
            )
            mismatch = 0.0
            if "points" in frame:
                points = cloud_xyz(frame["points"])
                both = np.isfinite(rebuilt).all(-1) & np.isfinite(points).all(-1)
                mismatch = (
                    float(np.abs(rebuilt[both] - points[both]).max())
                    if both.any()
                    else 0.0
                )
            if mismatch > 0.002:
                raise RuntimeError(
                    f"{scenario.name}: depth reprojection is {mismatch:.3f} m off Gazebo's cloud"
                )

            depth_mm = np.where(
                np.isfinite(depth) & (depth > 0), np.round(depth * 1000), 0
            )
            roll = math.atan2(chassis[2, 1], chassis[2, 2])
            pitch = -math.asin(max(-1.0, min(1.0, chassis[2, 0])))
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                OUT / f"{scenario.name}.npz",
                depth_mm=depth_mm.astype(np.uint16),
                labels=labels.astype(np.uint8),
                rgb=rgb.astype(np.uint8),
                intrinsics=np.array([fx, fy, cx, cy]),
                camera_position=camera_position,
                camera_rotation=chassis,
                car_pose=np.array([*position, roll, pitch, math.radians(scenario.yaw)]),
                course=course,
                name=scenario.name,
            )
            seen = {
                sc.PART_NAMES.get(int(p), str(p)): int(n)
                for p, n in zip(*np.unique(labels, return_counts=True))
            }
            print(
                f"  {scenario.name:28s} z={position[2]:.3f} roll={math.degrees(roll):+5.1f} "
                f"pitch={math.degrees(pitch):+5.1f}  {seen}",
                flush=True,
            )
            if args.preview:
                preview(
                    Path(args.preview) / f"{scenario.name}.png",
                    rgb,
                    labels,
                    np.where(depth_mm > 0, depth_mm / 1000.0, np.inf),
                )
    finally:
        for process in (bridge, server):
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10)
            except Exception:
                os.killpg(process.pid, signal.SIGKILL)
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="*", help="scenario names (default: all)")
    parser.add_argument("--course", choices=sorted(WORLDS), help="one course only")
    parser.add_argument(
        "--preview", help="also write a PNG per fixture into this directory"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(HERE))
    import generate_obstacle_course as gen

    wanted = [
        s
        for s in sc.SCENARIOS
        if (not args.only or s.name in args.only)
        and (not args.course or s.course == args.course)
    ]
    if args.only and len(wanted) != len(set(args.only)):
        missing = set(args.only) - {s.name for s in wanted}
        raise SystemExit(f"unknown scenarios: {sorted(missing)}")
    if args.preview:
        Path(args.preview).mkdir(parents=True, exist_ok=True)
    frames = Frames()
    for course in WORLDS:
        subset = [s for s in wanted if s.course == course]
        if subset:
            capture_course(course, subset, args, gen, frames)


if __name__ == "__main__":
    main()
