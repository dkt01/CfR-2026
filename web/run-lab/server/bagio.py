"""Reading rosbag2 runs without ROS.

`rosbags` is pure Python and reads both MCAP and sqlite3 bags, so the Run Lab
works on a laptop with no ROS install at all.  MCAP bags carry their own
message definitions; the repo's cfr_interfaces definitions are registered on
top so an older sqlite3 bag (or one recorded before a field was added) still
decodes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

REPO = Path(__file__).resolve().parents[3]
CFR_MSGS = REPO / "jetson" / "cfr_interfaces" / "msg"


def typestore():
    store = get_typestore(Stores.ROS2_JAZZY)
    types = {}
    for path in sorted(CFR_MSGS.glob("*.msg")):
        types.update(
            get_types_from_msg(path.read_text(), f"cfr_interfaces/msg/{path.stem}")
        )
    try:
        store.register(types)
    except Exception:  # noqa: BLE001 - a definition clash just means the bag's own wins
        pass
    return store


def find_bag(run_dir: Path):
    """The bag directory inside a run: <run>/bag, or the run itself."""
    for candidate in (run_dir / "bag", run_dir):
        if (candidate / "metadata.yaml").exists() or list(candidate.glob("*.mcap")):
            return candidate
    return None


class Bag:
    """Thin wrapper: topic listing, and typed iteration by topic."""

    def __init__(self, path: Path):
        self.path = path
        self.reader = AnyReader([path], default_typestore=typestore())

    def __enter__(self):
        self.reader.open()
        return self

    def __exit__(self, *exc):
        self.reader.close()

    @property
    def start_ns(self):
        return self.reader.start_time

    @property
    def end_ns(self):
        return self.reader.end_time

    def topics(self):
        counts = defaultdict(int)
        types = {}
        for conn in self.reader.connections:
            counts[conn.topic] += conn.msgcount
            types[conn.topic] = conn.msgtype
        return {t: {"type": types[t], "count": counts[t]} for t in sorted(counts)}

    def has(self, topic):
        return any(c.topic == topic for c in self.reader.connections)

    def messages(self, *topics, every_ns=0):
        """Yield (topic, t_ns, msg) for the given topics, in log-time order.

        every_ns > 0 decimates per topic *before* deserialising, which is what
        keeps a bag full of 3.7 MB point clouds tractable.
        """
        conns = [c for c in self.reader.connections if c.topic in topics]
        if not conns:
            return
        last = {}
        for conn, t_ns, raw in self.reader.messages(connections=conns):
            if every_ns and t_ns - last.get(conn.topic, -(1 << 62)) < every_ns:
                continue
            last[conn.topic] = t_ns
            try:
                msg = self.reader.deserialize(raw, conn.msgtype)
            except Exception:  # noqa: BLE001 - one bad message must not sink a run
                continue
            yield conn.topic, t_ns, msg


# ------------------------------------------------------------------ helpers


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def yaw_of(q):
    return float(
        np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    )


def quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform_matrix(translation, rotation):
    m = np.eye(4)
    m[:3, :3] = quat_matrix(rotation)
    m[:3, 3] = [translation.x, translation.y, translation.z]
    return m


_POINTFIELD = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def cloud_to_numpy(msg, max_range=10.0):
    """PointCloud2 -> (N,3) float32 xyz and (N,3) uint8 rgb, finite points only.

    sensor_msgs numbering (FLOAT32 = 7), which is what a bag holds -- NOT the
    raw gz numbering the browser viewer decodes (see the viewer's notes).
    """
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in "xyz"):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    names, formats, offsets = [], [], []
    for name in ("x", "y", "z", "rgb", "rgba"):
        if name in fields:
            f = fields[name]
            names.append(name)
            formats.append(
                ("<" if not msg.is_bigendian else ">")
                + _POINTFIELD.get(f.datatype, "f4")
            )
            offsets.append(f.offset)
    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": msg.point_step,
        }
    )
    count = msg.width * msg.height
    data = np.frombuffer(
        bytes(msg.data) if not isinstance(msg.data, np.ndarray) else msg.data,
        dtype=dtype,
        count=count,
    )
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    ok = np.isfinite(xyz).all(axis=1)
    ok &= np.einsum("ij,ij->i", xyz, xyz) < max_range * max_range
    xyz = xyz[ok]
    color_field = "rgb" if "rgb" in names else ("rgba" if "rgba" in names else None)
    if color_field:
        packed = np.ascontiguousarray(data[color_field][ok]).view(np.uint32)
        rgb = np.stack(
            [(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=1
        ).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 200, np.uint8)
    return xyz, rgb


def voxel_downsample(xyz, rgb, voxel):
    if len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    _, first = np.unique(flat, return_index=True)
    return xyz[first], rgb[first]


class StaticTF:
    """/tf_static (and the first /tf of each pair) as a graph, for chaining
    the camera's optical frame to the frame the pose describes."""

    def __init__(self):
        self.edges = {}  # (parent, child) -> 4x4

    def add(self, msg):
        for tf in msg.transforms:
            parent = tf.header.frame_id.lstrip("/")
            child = tf.child_frame_id.lstrip("/")
            key = (parent, child)
            if key not in self.edges:
                self.edges[key] = transform_matrix(
                    tf.transform.translation, tf.transform.rotation
                )

    def lookup(self, target, source):
        """Matrix taking points in `source` into `target`, or None."""
        target, source = target.lstrip("/"), source.lstrip("/")
        if target == source:
            return np.eye(4)
        graph = defaultdict(list)
        for (parent, child), m in self.edges.items():
            graph[parent].append((child, m))  # parent <- child: p = m @ c
            graph[child].append((parent, np.linalg.inv(m)))
        # BFS from target outward, accumulating target <- frame
        frontier = [(target, np.eye(4))]
        seen = {target}
        while frontier:
            nxt = []
            for frame, acc in frontier:
                for other, m in graph[frame]:
                    if other in seen:
                        continue
                    total = acc @ m
                    if other == source:
                        return total
                    seen.add(other)
                    nxt.append((other, total))
            frontier = nxt
        return None
