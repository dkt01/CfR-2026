"""Per-point classes for the ZED cloud: ground, obstacle, hoop, car wash, overhead.

The segmenter itself is C++ -- `include/cfr_arduino_bridge/cloud_segmentation.hpp`
explains what it does and why, and `cloud_segmentation_node` runs it on the car
and in Gazebo. This module loads the same compiled library through its C
interface, so `test/test_cloud_segmentation.py` scores exactly the code that
drives, and training code can call it without a second implementation to keep
in step. It used to be the numpy original; that ran at ~75 ms a frame on a
desktop core, several times the ZED's frame period on the Orin.

The library is found, in order, at `$CFR_CLOUD_SEGMENTATION_LIB`, beside this
file or one directory up (the installed layout), and on the loader path (a
sourced workspace). `Params` is built from the library's own table of names
and defaults, so the tunables are declared once, in the header.

Frames: points arrive in REP-103 body convention relative to the camera (+x
forward, +y left, +z up). Pitch is nose-down positive and roll left-side-up
positive, as cloud_scan takes them.
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GROUND = 0
OBSTACLE = 1
HOOP = 2
CARWASH = 3
OVERHEAD = 4
# Beyond max_range, non-finite, or behind the camera.
UNKNOWN = 255

CLASS_NAMES = {
    GROUND: "ground",
    OBSTACLE: "obstacle",
    HOOP: "hoop",
    CARWASH: "carwash",
    OVERHEAD: "overhead",
    UNKNOWN: "unknown",
}

_LIBRARY = "libcfr_cloud_segmentation.so"
# More gates than any view holds; a frame that finds more is re-run larger.
_MAX_GATES = 64


def _load() -> ctypes.CDLL:
    here = Path(__file__).resolve().parent
    candidates = [os.environ.get("CFR_CLOUD_SEGMENTATION_LIB")]
    candidates += [str(here / _LIBRARY), str(here.parent / _LIBRARY), _LIBRARY]
    errors = []
    for candidate in filter(None, candidates):
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    raise ImportError(
        "cloud_segmentation needs the compiled library -- build cfr_arduino_bridge "
        "and source its workspace, or set CFR_CLOUD_SEGMENTATION_LIB.\n  "
        + "\n  ".join(errors)
    )


_lib = _load()
_lib.cfr_segmentation_param_name.restype = ctypes.c_char_p
_lib.cfr_segmentation_param_default.restype = ctypes.c_double
_lib.cfr_segment.restype = ctypes.c_int64
_lib.cfr_segment.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
]
_GATE_FIELDS = _lib.cfr_segmentation_gate_fields()


def _param_fields():
    fields = []
    for index in range(_lib.cfr_segmentation_param_count()):
        name = _lib.cfr_segmentation_param_name(index).decode()
        default = _lib.cfr_segmentation_param_default(index)
        if _lib.cfr_segmentation_param_is_int(index):
            fields.append((name, int, dataclasses.field(default=int(default))))
        else:
            fields.append((name, float, dataclasses.field(default=float(default))))
    return fields


# Tunables, in meters unless named otherwise; see Params in the header for
# what each one is measured from.
Params = dataclasses.make_dataclass(
    "Params", _param_fields(), frozen=True, module=__name__
)
_PARAM_NAMES = [f.name for f in dataclasses.fields(Params)]


@dataclass
class Gate:
    kind: int
    # Center of the span, and its two feet, in the leveled camera frame.
    center: tuple
    feet: tuple
    span: float
    top: float
    # Unit vector along the span, feet[0] to feet[1].
    axis: tuple = (0.0, 1.0)

    @property
    def bearing(self) -> float:
        return math.atan2(self.center[1], self.center[0])

    @property
    def range(self) -> float:
        return math.hypot(self.center[0], self.center[1])


@dataclass
class Segmentation:
    labels: np.ndarray
    # Height above the drivable surface under each point (nan if unknown).
    height: np.ndarray
    # Points the car cannot pass through: obstacles, and the feet of gates.
    blocking: np.ndarray
    # The cloud leveled against gravity, camera at the origin.
    level: np.ndarray
    gates: list = field(default_factory=list)


def level_points(points: np.ndarray, pitch: float, roll: float) -> np.ndarray:
    """Rotate body-frame points so +z is up: p_level = Ry(pitch) Rx(roll) p."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rotation = np.array(
        [
            [cp, sp * sr, sp * cr],
            [0.0, cr, -sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return points @ rotation.T


def segment(
    points: np.ndarray,
    pitch: float = 0.0,
    roll: float = 0.0,
    params: Params = Params(),
    rgb: np.ndarray | None = None,
) -> Segmentation:
    """Classify an (N, 3) body-frame cloud. See the header for how."""
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    n = len(points)
    if rgb is not None:
        rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint32).reshape(-1))
        if len(rgb) != n:
            raise ValueError(f"rgb has {len(rgb)} entries for {n} points")
    values = np.array([getattr(params, name) for name in _PARAM_NAMES], dtype=float)
    labels = np.empty(n, dtype=np.uint8)
    height = np.empty(n, dtype=np.float64)
    blocking = np.empty(n, dtype=np.uint8)
    level = np.empty((n, 3), dtype=np.float64)
    capacity = _MAX_GATES
    while True:
        rows = np.empty((capacity, _GATE_FIELDS), dtype=np.float64)
        found = _lib.cfr_segment(
            points.ctypes.data,
            n,
            float(pitch),
            float(roll),
            values.ctypes.data,
            len(values),
            None if rgb is None else rgb.ctypes.data,
            labels.ctypes.data,
            height.ctypes.data,
            blocking.ctypes.data,
            level.ctypes.data,
            rows.ctypes.data,
            capacity,
        )
        if found < 0:
            raise RuntimeError("cloud_segmentation library disagrees on Params")
        if found <= capacity:
            break
        capacity = int(found)
    gates = [
        Gate(
            kind=int(row[0]),
            center=(float(row[1]), float(row[2])),
            feet=((float(row[3]), float(row[4])), (float(row[5]), float(row[6]))),
            span=float(row[7]),
            top=float(row[8]),
            axis=(float(row[9]), float(row[10])),
        )
        for row in rows[:found]
    ]
    return Segmentation(labels, height, blocking.astype(bool), level, gates)


def scan_from_segmentation(
    segmentation: Segmentation,
    num_bins: int,
    fov_deg: float,
    max_range: float,
    min_range: float = 0.15,
) -> np.ndarray:
    """Nearest blocking return per bearing bin, as `cloud_scan` produces.

    Bin 0 is the rightmost bearing and empty bins read max_range, so this
    drops in wherever `scan_from_points` is used today.
    """
    scan = np.full(num_bins, max_range, dtype=np.float32)
    pts = segmentation.level[segmentation.blocking]
    if pts.size == 0:
        return scan
    ranges = np.hypot(pts[:, 0], pts[:, 1])
    bearings = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    half = fov_deg / 2
    inside = (
        (ranges <= max_range)
        & (bearings >= -half)
        & (bearings <= half)
        & (pts[:, 0] > 0)
    )
    if not inside.any():
        return scan
    edges = np.linspace(-half, half, num_bins + 1)
    index = np.clip(np.digitize(bearings[inside], edges) - 1, 0, num_bins - 1)
    np.minimum.at(scan, index, np.maximum(ranges[inside], min_range).astype(np.float32))
    return scan


def points_from_depth(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """(H*W, 3) body-frame points from a depth image, row-major.

    Non-positive or non-finite depth becomes nan, so the result stays
    aligned pixel for pixel with the image it came from.
    """
    depth = np.asarray(depth, dtype=np.float64)
    rows, cols = np.indices(depth.shape)
    bad = ~np.isfinite(depth) | (depth <= 0)
    d = np.where(bad, np.nan, depth)
    return np.stack([d, -(cols - cx) * d / fx, -(rows - cy) * d / fy], axis=-1).reshape(
        -1, 3
    )
