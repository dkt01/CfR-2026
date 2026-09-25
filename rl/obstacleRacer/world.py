"""The Obstacle Course as geometry: every collision and visual in the world.

Gazebo keeps two descriptions of each obstacle and uses them for different
things, so this does too:

  * COLLISION primitives are what physics touches -- oriented boxes (many of
    them tilted: the ramps, the 24 helix segments, the bank, the entry and
    exit wedges), upright cylinders and the ground plane.  The plant's wheels
    stand on them and the chassis hits them.
  * VISUALS are what the ZED renders -- mostly STL meshes, and several of them
    have no collision at all: the car-wash ribbons and arches, the hoops' top
    bars, the tunnel's foil roof, the pothole bump domes.  The sensor model
    sees these and only these.

Each collision is tagged with a ROLE, by name, because the name is what says
what the thing is for:

  support   drivable: the wheels stand on it.  Its top face is road.
  obstacle  everything the chassis must not touch.

The tags are listed explicitly below rather than inferred from shape -- a
bale's top is as flat as the bridge deck -- and `parse` refuses a world with
an element it cannot place, so a new obstacle in the generator is a loud
failure here instead of a silent hole in the model.

The movable models (buckets, hoops, gap bales, Wide Section bales) are split out as DYNAMIC and
re-posed per layout by `apply_layout`; everything else is STATIC.
"""

from __future__ import annotations

import math
import re
import struct
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = REPO / "jetson" / "cfr_arduino_bridge"
WORLD_SDF = PACKAGE / "worlds" / "obstacle_course.sdf"
MESH_DIR = PACKAGE / "meshes"

# Tire mu is 1.0 in the vehicle model and ODE takes the smaller of the two
# surfaces' mu, so a surface only matters here when it is below the tire's.
TIRE_MU = 1.0
DEFAULT_MU = 1.0

# (model regex, element regex) -> role.  First match wins.  Every collision
# in the world must match one of these; see `parse`.
COLLISION_ROLES = [
    (r"ground", r"collision", "support"),
    (r"bridge", r"(ramp|deck)_collision", "support"),
    (r"bridge", r"(ramp|deck)_rail_-?1_collision", "obstacle"),
    (r"helix", r"deck_\d+_collision", "support"),
    (r"helix", r"(inner|outer)_rail_\d+_collision", "obstacle"),
    (r"bank", r"surface_collision", "support"),
    (r"bank", r"wall_(outer|end_a|end_b)_collision", "obstacle"),
    (r"gravel_box", r"(base|surface|pebble_\d+)_collision", "support"),
    (r"gravel_box", r"rail_[ns]_collision", "obstacle"),
    (r"(gravel|pothole)_(entry|exit)_ramp", r".*wedge_collision", "support"),
    (r"pothole_section", r"(base|top_\d+|bump_\d+_body)_collision", "support"),
    (r"hoop_\d+", r"base_collision", "support"),
    (r"hoop_\d+", r"upright_-?1_collision", "obstacle"),
    (r"car_wash", r"base_collision", "support"),
    (r"car_wash", r"upright_\d+_-?1_collision", "obstacle"),
    (r"course_bales", r"bale_\d+_collision", "obstacle"),
    (r"gap_bale_\d+", r"bale_collision", "obstacle"),
    (r"wide_bale_\d+", r"bale_collision", "obstacle"),
    (r"bucket_\d+", r"body_collision", "obstacle"),
    (r"tunnel", r"wall_-?1_collision", "obstacle"),
    (r"start_signal_frame", r"post_collision", "obstacle"),
]

# Visuals the sensor model skips.  The start-signal arms are 0.8 m up on a
# post beside the start straight and move; the ground plane is the surface
# itself.
IGNORED_VISUALS = [(r"start_signal_arms", r".*"), (r"ground", r".*")]

# Tagged for the sensor model's gate output; the segmenter finds these as
# gates and does not let their spans block.
VISUAL_CLASSES = [
    (r"car_wash", r"strip_\d+_\d+_visual", "carwash"),
]

DYNAMIC = re.compile(r"(bucket_\d+|hoop_\d+|gap_bale_\d+|wide_bale_\d+)$")
SKIPPED_MODELS = re.compile(r"slash$")


@dataclass
class Prim:
    model: str
    name: str
    source: str  # "collision" | "visual"
    kind: str  # "box" | "cylinder" | "plane" | "mesh"
    pose: np.ndarray  # 4x4, element in model frame
    model_pose: np.ndarray  # 4x4, model in world
    size: tuple = ()  # box (x, y, z) | cylinder (radius, length)
    mesh: str = ""
    role: str = ""  # collision: support | obstacle
    vclass: str = "solid"  # visual: solid | carwash
    mu: float = DEFAULT_MU

    @property
    def world(self) -> np.ndarray:
        return self.model_pose @ self.pose

    @property
    def dynamic(self) -> bool:
        return bool(DYNAMIC.match(self.model))


@dataclass
class World:
    prims: list[Prim] = field(default_factory=list)

    def select(self, source=None, role=None, dynamic=None):
        return [
            p
            for p in self.prims
            if (source is None or p.source == source)
            and (role is None or p.role == role)
            and (dynamic is None or p.dynamic == dynamic)
        ]


def pose_matrix(values) -> np.ndarray:
    x, y, z, roll, pitch, yaw = (list(values) + [0.0] * 6)[:6]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # SDF: extrinsic roll, pitch, yaw about fixed x, y, z = R = Rz Ry Rx.
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (x, y, z)
    return matrix


def _pose_of(element) -> np.ndarray:
    node = element.find("pose")
    if node is None or not (node.text or "").strip():
        return np.eye(4)
    return pose_matrix(float(v) for v in node.text.split())


def _match(table, model, name):
    for model_re, name_re, value in table:
        if re.fullmatch(model_re, model) and re.fullmatch(name_re, name):
            return value
    return None


def parse(sdf_path: Path = WORLD_SDF) -> World:
    root = ElementTree.parse(sdf_path).getroot()
    world = World()
    unknown = []
    for model in root.find("world").findall("model"):
        model_name = model.get("name")
        if SKIPPED_MODELS.match(model_name):
            continue
        model_pose = _pose_of(model)
        links = model.findall("link")
        for link in links:
            link_pose = model_pose @ _pose_of(link)
            for source in ("collision", "visual"):
                for element in link.findall(source):
                    name = element.get("name")
                    geometry = element.find("geometry")[0]
                    kind = geometry.tag
                    if source == "visual" and _match(
                        [(m, n, True) for m, n in IGNORED_VISUALS], model_name, name
                    ):
                        continue
                    prim = Prim(
                        model=model_name,
                        name=name,
                        source=source,
                        kind=kind,
                        pose=_pose_of(element),
                        model_pose=link_pose,
                    )
                    if kind == "box":
                        prim.size = tuple(
                            float(v) for v in geometry.find("size").text.split()
                        )
                    elif kind == "cylinder":
                        prim.size = (
                            float(geometry.find("radius").text),
                            float(geometry.find("length").text),
                        )
                    elif kind == "mesh":
                        uri = geometry.find("uri").text
                        prim.mesh = uri.rsplit("/", 1)[-1]
                    elif kind != "plane":
                        unknown.append(f"{model_name}/{name}: geometry {kind}")
                        continue
                    if source == "collision":
                        role = _match(COLLISION_ROLES, model_name, name)
                        if role is None:
                            unknown.append(f"{model_name}/{name}: no role")
                            continue
                        prim.role = role
                        mu = element.find(".//ode/mu")
                        if mu is not None:
                            prim.mu = min(float(mu.text), TIRE_MU)
                    else:
                        prim.vclass = (
                            _match(VISUAL_CLASSES, model_name, name) or "solid"
                        )
                    world.prims.append(prim)
    if unknown:
        raise ValueError(
            "world elements this model does not know how to treat -- add them "
            "to COLLISION_ROLES / IGNORED_VISUALS in world.py:\n  "
            + "\n  ".join(unknown)
        )
    return world


def apply_layout(world: World, layout: dict, spec: dict) -> World:
    """A copy of `world` with the movable models where `layout` puts them.

    Buckets a layout leaves out, and the parked gap bale, are dropped rather
    than parked: they stand off the course, behind the start straight's wall,
    where the car can neither reach nor see them.
    """
    buckets = layout["buckets"]
    hoops = layout["hoops"]
    gap_spec = spec["gap_bales"]
    out = World()
    for prim in world.prims:
        if not prim.dynamic:
            out.prims.append(prim)
            continue
        model = prim.model
        if model.startswith("bucket_"):
            index = int(model.split("_")[1])
            if index >= len(buckets):
                continue
            x, y = buckets[index]
            pose = pose_matrix((x, y, 0.0, 0, 0, 0))
        elif model.startswith("wide_bale_"):
            # Layouts exported before the Wide Section moved keep the drawing.
            poses = layout.get("wide_bales") or {}
            x, y, yaw = poses.get(model) or spec["wide_bales"][model]["nominal"]
            pose = pose_matrix((x, y, 0.0, 0, 0, yaw))
        elif model.startswith("hoop_"):
            x, y = hoops[model]
            yaw = float(spec["hoops"][model]["yaw"])
            pose = pose_matrix((x, y, 0.0, 0, 0, yaw))
        else:
            if model == layout["gap_bale"]:
                continue
            x, y = gap_spec[model]["position"]
            pose = pose_matrix((x, y, 0.0, 0, 0, float(gap_spec[model]["yaw"])))
        out.prims.append(replace(prim, model_pose=pose))
    return out


_MESH_CACHE: dict[str, np.ndarray] = {}


def stl_triangles(name: str) -> np.ndarray:
    """(N, 3, 3) triangles of a mesh, in its own frame, meters."""
    if name in _MESH_CACHE:
        return _MESH_CACHE[name]
    data = (MESH_DIR / name).read_bytes()
    if data[:5].lower() == b"solid" and b"facet" in data[:400]:
        verts = [
            [float(v) for v in line.split()[1:4]]
            for line in data.decode(errors="replace").splitlines()
            if line.strip().startswith("vertex")
        ]
        triangles = np.asarray(verts, dtype=float).reshape(-1, 3, 3)
    else:
        (count,) = struct.unpack_from("<I", data, 80)
        dtype = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        records = np.frombuffer(data, dtype=dtype, count=count, offset=84)
        triangles = records["v"].astype(float)
    _MESH_CACHE[name] = triangles
    return triangles


def world_triangles(prim: Prim) -> np.ndarray:
    """A visual's surface as world-frame triangles, whatever its geometry."""
    matrix = prim.world
    if prim.kind == "mesh":
        local = stl_triangles(prim.mesh)
    elif prim.kind == "box":
        local = _box_triangles(prim.size)
    elif prim.kind == "cylinder":
        local = _cylinder_triangles(*prim.size)
    else:
        return np.zeros((0, 3, 3))
    flat = local.reshape(-1, 3)
    moved = flat @ matrix[:3, :3].T + matrix[:3, 3]
    return moved.reshape(-1, 3, 3)


def _box_triangles(size) -> np.ndarray:
    hx, hy, hz = (s / 2 for s in size)
    corners = np.array(
        [
            [sx * hx, sy * hy, sz * hz]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    faces = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    tris = []
    for a, b, c, d in faces:
        tris.append(corners[[a, b, c]])
        tris.append(corners[[a, c, d]])
    return np.asarray(tris)


def _cylinder_triangles(radius, length, segments=16) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, segments + 1)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    h = length / 2
    tris = []
    for i in range(segments):
        a, b = ring[i], ring[i + 1]
        tris.append([[a[0], a[1], -h], [b[0], b[1], -h], [b[0], b[1], h]])
        tris.append([[a[0], a[1], -h], [b[0], b[1], h], [a[0], a[1], h]])
        tris.append([[0, 0, h], [a[0], a[1], h], [b[0], b[1], h]])
        tris.append([[0, 0, -h], [b[0], b[1], -h], [a[0], a[1], -h]])
    return np.asarray(tris, dtype=float)
