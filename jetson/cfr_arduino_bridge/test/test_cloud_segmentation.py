"""Checks cloud_segmentation against rendered, ground-truth-labeled views.

Two kinds of test live here.

Synthetic ones build a cloud from a handful of planes and boxes and check a
single behavior each -- the frame conventions, a flat floor, a ramp, a
hoop-shaped gate -- so a failure names the rule that broke.

Fixture ones score the segmenter on the views in segmentation_scenarios.py,
rendered in Gazebo by jetson/scripts/capture_segmentation_fixtures.py with a
label camera beside the ZED. The label says what each pixel *is* (a bale, the
ramp, a ribbon); `truth_classes` turns that into what the segmenter should
*call* it, which for the drivable structures depends on which face the
camera sees: a ramp's top is ground, its side is a wall, its underside is
overhead. Each view is then held to:

  * ground not reported as obstacle, and obstacles not reported as passable,
    by point;
  * the 36-bin scan the policy consumes matching the scan the truth implies;
  * every part the scenario exists for (`expect`) visible and classed right.

`python test_cloud_segmentation.py --report` prints the per-scenario table
these thresholds were read against, which is the thing to look at when a
change to the segmenter moves them.
"""

from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import cloud_segmentation as cs  # noqa: E402
import segmentation_scenarios as sc  # noqa: E402

FIXTURES = HERE / "fixtures" / "segmentation"
IGNORE = 254

# The policy's scan (rl/bale_follower/config_lap.yaml).
SCAN_BINS = 36
SCAN_FOV = 110.0
SCAN_RANGE = 6.0

# Per-point scores are taken inside this plan range. The scan goes to 6 m,
# but by then a floor row is half a meter deep and a hoop's tube two pixels
# wide; what matters most is the stretch the car can still react to.
EVAL_RANGE = 4.0

# Heights, world meters, used by the truth rules. Floor is z = 0 under all
# of the course's raised structure.
BASE_PLATE = 0.03
LOW_STEP = 0.05
FOOT_BAND = 0.06

GROUND_PARTS = {
    sc.PART_FLOOR,
    sc.PART_POTHOLE_BOARD,
    sc.PART_POTHOLE_BUMP,
    sc.PART_GRAVEL,
    sc.PART_SMALL_RAMP,
    sc.PART_CARWASH_BASE,
}
# Drivable on top, walls on the side, overhead underneath.
RAISED_PARTS = {sc.PART_RAMP, sc.PART_HELIX, sc.PART_BANK}
SOLID_PARTS = {
    sc.PART_BALE,
    sc.PART_BUCKET,
    sc.PART_RAIL,
    sc.PART_SIGNAL,
    sc.PART_TUNNEL,
}

# What each `expect` entry must be classed as, and how much of it.
EXPECT = {
    # The board blocks; its lamp arms stand out over open floor 0.7 m up
    # and are overhead. Judged on each point's own truth.
    "signal": (None, 0.95),
    "bale": ({cs.OBSTACLE}, 0.95),
    "bucket": ({cs.OBSTACLE}, 0.95),
    "rail": ({cs.OBSTACLE}, 0.90),
    "hoop": ({cs.HOOP}, 0.80),
    # The arch's uprights stand flush against the bale walls either side
    # and block like them; its bar is car wash. Both are right.
    "carwash_arch": ({cs.CARWASH, cs.OBSTACLE}, 0.95),
    "carwash_ribbon": ({cs.CARWASH}, 0.85),
    # Drivable parts: judged on the faces truth_classes calls ground.
    "ramp": ({cs.GROUND}, 0.95),
    "helix": ({cs.GROUND}, 0.90),
    "bank": ({cs.GROUND}, 0.95),
    "gravel": ({cs.GROUND}, 0.98),
    "small_ramp": ({cs.GROUND}, 0.98),
    "pothole_board": ({cs.GROUND}, 0.98),
    "pothole_bump": ({cs.GROUND}, 0.98),
    # Walls block, the roof is overhead: judged on each point's own truth.
    "tunnel": (None, 0.90),
}

# Views where the right answer for an expected part is looser than above,
# and why.
EXPECT_OVERRIDES = {
    # 0.4 m from the camera the hoop's top bar is above the frame: what is
    # left is two posts, and calling them obstacles is correct.
    ("obs_hoop0_close", "hoop"): ({cs.HOOP, cs.OBSTACLE}, 0.95),
    # Edge-on, the uprights line up and the span collapses to a post.
    ("obs_hoop0_edge_on", "hoop"): ({cs.HOOP, cs.OBSTACLE}, 0.95),
}

PART_BY_NAME = {name: part for part, name in sc.PART_NAMES.items()}


# ---------------------------------------------------------------- fixtures


class View:
    """One fixture, as the car and as the truth see it."""

    def __init__(self, path: Path):
        data = np.load(path)
        self.name = str(data["name"])
        self.course = str(data["course"])
        depth = data["depth_mm"].astype(np.float64) / 1000.0
        depth[depth <= 0] = np.nan
        self.shape = depth.shape
        fx, fy, cx, cy = data["intrinsics"]
        self.points = cs.points_from_depth(depth, fx, fy, cx, cy)
        self.parts = data["labels"].reshape(-1).astype(np.int64)
        rotation = data["camera_rotation"]
        # What the IMU reports: nose-down pitch, left-up roll.
        self.pitch = -math.asin(max(-1.0, min(1.0, rotation[2, 0])))
        self.roll = math.atan2(rotation[2, 1], rotation[2, 2])
        self.world = self.points @ rotation.T + data["camera_position"]
        self.normals = _normals(
            self.world.reshape(*self.shape, 3), data["camera_position"]
        )
        self.truth, self.truth_blocking = truth_classes(self)
        level = cs.level_points(self.points, self.pitch, self.roll)
        self.plan = np.hypot(level[:, 0], level[:, 1])
        self.level = level


def _normals(world: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """Unit surface normals per pixel, facing the camera; nan across edges.

    Central differences inside the image, one-sided at its border -- the
    tunnel roof is only ever seen along the top rows.
    """
    right = np.gradient(world, axis=1)
    down = np.gradient(world, axis=0)
    n = np.cross(right, down)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    distance = np.linalg.norm(world - camera, axis=-1, keepdims=True)
    # Neighbors further apart than a smooth surface allows are an edge.
    span = np.maximum(np.linalg.norm(right, axis=-1), np.linalg.norm(down, axis=-1))[
        ..., None
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.where((norm > 0) & (span < 0.025 * distance + 0.01), n / norm, np.nan)
    facing = np.sum(n * (camera - world), axis=-1, keepdims=True)
    n = np.where(facing < 0, -n, n)
    return n.reshape(-1, 3)


def _support_heights(view: View, ground: np.ndarray) -> np.ndarray:
    """World height of the true ground nearest under each point, in plan.

    Highest ground-truth return per 0.1 m plan cell, spread three cells
    outward; the floor (z = 0) where no ground was seen nearby.
    """
    cell = 0.1
    xy = view.world[:, :2]
    ok = ground & np.isfinite(xy).all(axis=1)
    if not ok.any():
        return np.zeros(len(xy))
    origin = np.nanmin(xy, axis=0) - 1.0
    ij = np.floor((np.nan_to_num(xy, nan=0.0) - origin) / cell).astype(np.int64)
    ij = np.clip(ij, 0, None)
    shape = tuple(ij.max(axis=0) + 2)
    top = np.full(shape, -np.inf)
    np.maximum.at(top, (ij[ok, 0], ij[ok, 1]), view.world[ok, 2])
    for _ in range(3):
        grown = top.copy()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown = np.maximum(grown, np.roll(np.roll(top, dx, 0), dy, 1))
        top = np.where(np.isfinite(top), top, grown)
    support = top[ij[:, 0], ij[:, 1]]
    return np.where(np.isfinite(support), support, 0.0)


def truth_classes(view: View):
    """(class, blocking) per point, from part labels and world geometry."""
    parts = view.parts
    z = view.world[:, 2]
    nz = view.normals[:, 2]
    truth = np.full(len(parts), IGNORE, dtype=np.int64)
    finite = np.isfinite(view.points).all(axis=1)

    ground = np.isin(parts, list(GROUND_PARTS))
    truth[ground] = cs.GROUND

    raised = np.isin(parts, list(RAISED_PARTS))
    top = raised & (nz >= 0.8)
    under = raised & (nz <= -0.5)
    side = raised & np.isfinite(nz) & ~top & ~under
    truth[top] = cs.GROUND
    truth[under] = np.where(z[under] > cs.Params().clearance, cs.OVERHEAD, cs.OBSTACLE)
    truth[side] = np.where(z[side] < LOW_STEP, cs.GROUND, cs.OBSTACLE)

    solid = np.isin(parts, list(SOLID_PARTS))
    high = solid & (z > cs.Params().clearance)
    truth[solid] = cs.OBSTACLE
    truth[high & (nz <= -0.5)] = cs.OVERHEAD
    # Above the car with no usable normal (an edge): roof or wall is a guess.
    truth[high & np.isnan(nz)] = IGNORE

    hoop = parts == sc.PART_HOOP
    truth[hoop] = np.where(z[hoop] < BASE_PLATE, cs.GROUND, cs.HOOP)
    arch = parts == sc.PART_CARWASH_ARCH
    truth[arch] = np.where(z[arch] < BASE_PLATE, cs.GROUND, cs.CARWASH)
    truth[parts == sc.PART_CARWASH_RIBBON] = cs.CARWASH

    truth[~finite] = IGNORE
    # The bottom few centimeters of anything standing on a surface are
    # within the segmenter's ground tolerance by design, and a car cannot
    # tell them from the surface either. Measured against the true surface
    # under them, not the floor, so a rail on the ramp is judged the same as
    # a bale on the floor.
    support = _support_heights(view, truth == cs.GROUND)
    standing = np.isin(truth, [cs.OBSTACLE, cs.HOOP, cs.CARWASH])
    foot = standing & (z - support < FOOT_BAND)
    truth[foot] = IGNORE
    # Overhead by what is under it, not only by which way it faces -- the
    # start signal's lamp arms stand out from the board 0.7 m up with floor
    # under them, and the car passes beneath them as it does the tunnel roof.
    # Heights are above the true surface under each point, not the floor.
    cell = 0.05
    ok = finite & (truth != IGNORE)
    ij = np.floor(np.nan_to_num(view.world[:, :2], nan=0.0) / cell).astype(np.int64)
    ij -= ij[ok].min(axis=0) if ok.any() else 0
    ij = np.clip(ij, 0, None)
    shape = tuple(ij[ok].max(axis=0) + 1) if ok.any() else (1, 1)
    ij = np.minimum(ij, np.array(shape) - 1)
    solid_low = np.full(shape, np.inf)
    standing_now = ok & (truth != cs.GROUND)
    np.minimum.at(
        solid_low, (ij[standing_now, 0], ij[standing_now, 1]), z[standing_now]
    )
    floor_seen = np.zeros(shape, dtype=bool)
    floor_seen[ij[ok & (truth == cs.GROUND), 0], ij[ok & (truth == cs.GROUND), 1]] = (
        True
    )
    clear = cs.Params().clearance
    overhang = (
        (truth == cs.OBSTACLE)
        & (z - support > clear)
        & (solid_low[ij[:, 0], ij[:, 1]] - support > clear)
        & floor_seen[ij[:, 0], ij[:, 1]]
    )
    truth[overhang] = cs.OVERHEAD
    # The feet of a gate block; its bar and ribbons do not.
    blocking = (truth == cs.OBSTACLE) | (
        (hoop | arch) & (truth != cs.GROUND) & (z <= cs.Params().clearance)
    )
    return truth, blocking


def _scan(level: np.ndarray, blocking: np.ndarray) -> np.ndarray:
    seg = cs.Segmentation(
        labels=np.zeros(len(level), np.uint8),
        height=np.zeros(len(level)),
        blocking=blocking & np.isfinite(level).all(axis=1),
        level=np.nan_to_num(level, nan=1e9),
    )
    return cs.scan_from_segmentation(seg, SCAN_BINS, SCAN_FOV, SCAN_RANGE)


def stereo_noise(points: np.ndarray, seed: int) -> np.ndarray:
    """zed_cloud_noise_node's model: range error a + b x^2, 3% dropout."""
    rng = np.random.default_rng(seed)
    x = points[:, 0]
    valid = np.isfinite(points).all(axis=1) & (x > 0)
    out = points.copy()
    delta = rng.normal(0.0, 0.01 + 0.008 * np.where(valid, x, 0.0) ** 2)
    scale = np.where(valid, np.maximum(0.01, x + delta) / np.where(valid, x, 1.0), 1.0)
    out *= scale[:, None]
    out[valid & (rng.random(len(x)) < 0.03)] = np.nan
    return out


def bent_streamers(view: View, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Bend each 0.425 m streamer while preserving its length.

    The rendered fixture supplies the visible points and part labels. This
    changes only streamer returns, keeping the bucket and bale truth intact;
    it cannot model newly exposed or occluded background pixels. Each strip
    has its own tip direction in the mixed and tangled cases.
    """
    ribbon = (view.parts == sc.PART_CARWASH_RIBBON) & np.isfinite(view.world).all(
        axis=1
    )
    world = view.world[ribbon]
    row = np.clip(np.rint((world[:, 1] + 0.4445) / 0.127), 0, 7).astype(int)
    arch = np.rint((world[:, 0] - np.nanmin(world[:, 0])) / 0.457).astype(int)
    if mode == "mixed":
        phase = row * 2.31 + arch * 1.73
        tip_x, tip_y = 0.30 * np.cos(phase), 0.30 * np.sin(phase)
    elif mode == "tangled":
        # Separate attachments, with free ends drawn toward the middle and
        # neighboring strips crossing there.
        tip_x = np.where(row % 2 == 0, -0.08, 0.08)
        tip_y = np.clip(-(row - 3.5) * 0.127, -0.30, 0.30)
    else:
        directions = {
            "forward": (0.30, 0.0),
            "backward": (-0.30, 0.0),
            "left": (0.0, 0.30),
            "right": (0.0, -0.30),
            "diagonal": (0.30 / math.sqrt(2), 0.30 / math.sqrt(2)),
        }
        tip_x, tip_y = directions[mode]
    drop = np.clip(0.54 - world[:, 2], 0.0, 0.425)
    factor = drop / 0.425
    length = np.hypot(tip_x, tip_y)
    lift = drop * (1.0 - np.sqrt(1.0 - (length / 0.425) ** 2))
    delta_world = np.broadcast_arrays(factor * tip_x, factor * tip_y, lift)
    rotation = np.load(FIXTURES / f"{view.name}.npz")["camera_rotation"]
    points = view.points.copy()
    points[ribbon] += np.stack(delta_world, axis=1) @ rotation

    # The real party streamers are lemon yellow; the other known objects in
    # the wash are gray buckets and straw-colored bales.
    rgb = np.full(len(view.parts), 0x808080, dtype=np.uint32)
    rgb[view.parts == sc.PART_BALE] = 0xB87A1F
    rgb[ribbon] = 0xFFF430
    return points, rgb


def score(view: View, points=None, pitch=None, roll=None) -> dict:
    """Everything the fixture tests assert on, for one run of the segmenter."""
    points = view.points if points is None else points
    seg = cs.segment(
        points,
        view.pitch if pitch is None else pitch,
        view.roll if roll is None else roll,
    )
    # Nearer than the ZED's minimum depth the real camera returns nothing,
    # and the segmenter discards it the same way.
    depth = np.linalg.norm(view.points, axis=1)
    near = (
        (view.plan <= EVAL_RANGE)
        & (depth >= cs.Params().min_depth)
        & np.isfinite(points).all(axis=1)
    )
    truth = view.truth
    predicted = seg.labels.astype(np.int64)

    ground = near & (truth == cs.GROUND)
    obstacle = near & (truth == cs.OBSTACLE)
    passable = {cs.GROUND, cs.OVERHEAD, cs.HOOP, cs.CARWASH, cs.UNKNOWN}
    result = {
        "segmentation": seg,
        "ground_points": int(ground.sum()),
        "false_obstacle": float(seg.blocking[ground].mean()) if ground.any() else 0.0,
        "obstacle_points": int(obstacle.sum()),
        "missed_obstacle": float(np.isin(predicted[obstacle], list(passable)).mean())
        if obstacle.any()
        else 0.0,
    }
    truth_scan = _scan(view.level, view.truth_blocking)
    scan = cs.scan_from_segmentation(seg, SCAN_BINS, SCAN_FOV, SCAN_RANGE)
    tolerance = 0.10 + 0.05 * truth_scan
    result["scan_agreement"] = float(np.mean(np.abs(scan - truth_scan) <= tolerance))
    result["scan"] = scan
    result["truth_scan"] = truth_scan

    parts = {}
    for name in sc.BY_NAME[view.name].expect:
        part = PART_BY_NAME[name]
        classes, _ = EXPECT_OVERRIDES.get((view.name, name), EXPECT[name])
        mask = near & (view.parts == part) & (truth != IGNORE)
        if classes is None:
            hit = predicted[mask] == truth[mask]
        else:
            # Judge drivable parts on their drivable faces only (a ramp's
            # side is supposed to be a wall), and standing parts on what
            # stands (a hoop's base plate is supposed to be ground).
            if classes == {cs.GROUND}:
                mask &= truth == cs.GROUND
            else:
                mask &= truth != cs.GROUND
            hit = np.isin(predicted[mask], list(classes))
        parts[name] = (int(mask.sum()), float(hit.mean()) if mask.any() else 0.0)
    result["parts"] = parts
    return result


_VIEWS: dict = {}


def load(name: str) -> View:
    if name not in _VIEWS:
        path = FIXTURES / f"{name}.npz"
        if not path.exists():
            pytest.fail(
                f"no fixture for scenario {name}: run "
                "jetson/scripts/capture_segmentation_fixtures.py --only " + name
            )
        _VIEWS[name] = View(path)
    return _VIEWS[name]


# ------------------------------------------------------- synthetic, by rule


def _grid(x0, x1, y0, y1, step=0.02):
    x, y = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step), indexing="ij")
    return x.ravel(), y.ravel()


def floor(height=-0.20, x0=0.3, x1=5.0):
    x, y = _grid(x0, x1, -2.0, 2.0)
    return np.c_[x, y, np.full_like(x, height)]


def wall(x, y0, y1, z0=-0.20, z1=0.156, step=0.01):
    y, z = np.meshgrid(np.arange(y0, y1, step), np.arange(z0, z1, step), indexing="ij")
    return np.c_[np.full(y.size, x), y.ravel(), z.ravel()]


def test_leveling_matches_the_imu_convention():
    # Nose-down pitch: a point straight ahead of the camera is below it.
    ahead = cs.level_points(np.array([[1.0, 0.0, 0.0]]), math.radians(10), 0.0)
    assert ahead[0, 2] < 0
    # Left-up roll: a point to the left is above the camera.
    left = cs.level_points(np.array([[0.0, 1.0, 0.0]]), 0.0, math.radians(10))
    assert left[0, 2] > 0


def test_flat_floor_is_all_ground():
    seg = cs.segment(floor())
    assert np.all(seg.labels == cs.GROUND)
    assert not seg.blocking.any()


def test_a_bale_on_the_floor_is_an_obstacle_and_the_floor_is_not():
    cloud = np.r_[floor(), wall(2.0, -0.45, 0.45)]
    seg = cs.segment(cloud)
    is_wall = np.arange(len(cloud)) >= len(floor())
    high = is_wall & (cloud[:, 2] > -0.20 + 0.08)
    assert np.all(seg.labels[high] == cs.OBSTACLE)
    assert np.all(seg.labels[~is_wall] == cs.GROUND)
    scan = cs.scan_from_segmentation(seg, 36, 110.0, 6.0)
    assert scan[17] == pytest.approx(2.0, abs=0.02) and scan[18] == pytest.approx(
        2.0, abs=0.02
    )
    assert scan[0] == 6.0


def test_a_19_percent_ramp_is_ground_all_the_way_up():
    x, y = _grid(0.3, 5.0, -0.4, 0.4)
    z = -0.20 + np.clip(x - 1.0, 0, None) * 0.19
    seg = cs.segment(np.c_[x, y, z])
    assert np.mean(seg.labels == cs.GROUND) > 0.99


def test_unleveled_pitch_is_undone():
    # The car pitched 8 degrees nose-up on a ramp foot sees the flat floor
    # ahead as rising; told its pitch, it must still call it ground.
    pitch = math.radians(-8)
    body = cs.level_points(floor(), -pitch, 0.0)  # exact inverse with no roll
    seg = cs.segment(body, pitch=pitch)
    assert np.mean(seg.labels == cs.GROUND) > 0.99


def test_scan_bin_zero_is_the_rightmost_bearing():
    cloud = np.r_[floor(), wall(1.5, -1.9, -1.5)]
    scan = cs.scan_from_segmentation(cs.segment(cloud), 36, 110.0, 6.0)
    assert scan[:10].min() < 6.0 and scan[26:].min() == 6.0


def _gate(distance, span, top, curtain=False):
    """Two posts and a bar, with an optional curtain between, across +x."""
    parts = []
    for side in (-1, 1):
        y, z = np.meshgrid(
            np.arange(-0.017, 0.017, 0.008) + side * span / 2,
            np.arange(-0.20, -0.20 + top, 0.01),
            indexing="ij",
        )
        parts.append(np.c_[np.full(y.size, distance), y.ravel(), z.ravel()])
    y, z = np.meshgrid(
        np.arange(-span / 2, span / 2, 0.01),
        np.arange(top - 0.23, top - 0.20, 0.01),
        indexing="ij",
    )
    parts.append(np.c_[np.full(y.size, distance), y.ravel(), z.ravel()])
    if curtain:
        for strip in np.arange(-span / 2 + 0.1, span / 2 - 0.1, 0.127):
            y, z = np.meshgrid(
                np.arange(strip, strip + 0.05, 0.01),
                np.arange(-0.085, top - 0.2, 0.01),
                indexing="ij",
            )
            parts.append(np.c_[np.full(y.size, distance), y.ravel(), z.ravel()])
    return np.concatenate(parts)


def test_a_hoop_is_found_and_only_its_feet_block():
    gate = _gate(2.0, 0.584, 0.537)
    cloud = np.r_[floor(), gate]
    seg = cs.segment(cloud)
    # The bottom few centimeters of any post are ground by design.
    mine = (np.arange(len(cloud)) >= len(floor())) & (cloud[:, 2] > -0.20 + 0.06)
    assert np.mean(seg.labels[mine] == cs.HOOP) > 0.95
    assert [g.kind for g in seg.gates] == [cs.HOOP]
    assert seg.gates[0].span == pytest.approx(0.55, abs=0.1)
    scan = cs.scan_from_segmentation(seg, 36, 110.0, 6.0)
    assert scan[17] == 6.0 and scan[18] == 6.0  # straight through the middle


def test_a_curtained_gate_is_a_car_wash():
    gate = _gate(2.0, 1.151, 0.537, curtain=True)
    cloud = np.r_[floor(), gate]
    seg = cs.segment(cloud)
    mine = (np.arange(len(cloud)) >= len(floor())) & (cloud[:, 2] > -0.20 + 0.06)
    assert np.mean(seg.labels[mine] == cs.CARWASH) > 0.95
    assert not seg.blocking[mine & (np.abs(cloud[:, 1]) < 0.4)].any()


def test_a_tall_board_on_two_feet_is_not_a_gate():
    # The start signal: 0.81 m wide, 1.22 m tall, with daylight under it.
    board = _gate(2.0, 0.81, 1.22)
    y, z = np.meshgrid(
        np.arange(-0.4, 0.4, 0.01), np.arange(-0.05, 1.0, 0.01), indexing="ij"
    )
    board = np.r_[board, np.c_[np.full(y.size, 2.0), y.ravel(), z.ravel()]]
    seg = cs.segment(np.r_[floor(), board])
    assert not seg.gates


def test_a_wide_wash_arch_is_found_when_thin_streamers_disappear():
    # A party streamer need not produce a stereo return in every frame. The
    # rigid 1.151 m arch still distinguishes the wash from the 0.584 m hoops.
    cloud = np.r_[floor(), _gate(2.0, 1.151, 0.537)]
    seg = cs.segment(cloud)
    assert [gate.kind for gate in seg.gates] == [cs.CARWASH]


def test_yellow_is_not_passable_without_a_car_wash_arch():
    cloud = np.r_[floor(), wall(2.0, -0.45, 0.45)]
    rgb = np.full(len(cloud), 0xFFF430, dtype=np.uint32)
    seg = cs.segment(cloud, rgb=rgb)
    face = (np.arange(len(cloud)) >= len(floor())) & (cloud[:, 2] > -0.12)
    assert np.all(seg.labels[face] == cs.OBSTACLE)
    assert np.all(seg.blocking[face])


def test_gazebo_shaded_straw_inside_the_wash_stays_blocking():
    ground = floor()
    arch = _gate(2.0, 1.151, 0.537)
    bale = wall(2.25, -0.20, 0.20)
    cloud = np.r_[ground, arch, bale]
    rgb = np.full(len(cloud), 0x808080, dtype=np.uint32)
    rgb[len(ground) + len(arch) :] = 0x554621  # observed Gazebo bale shade
    seg = cs.segment(cloud, rgb=rgb)
    face = np.arange(len(cloud)) >= len(ground) + len(arch)
    face &= cloud[:, 2] > -0.12
    assert any(gate.kind == cs.CARWASH for gate in seg.gates)
    assert np.mean(seg.blocking[face]) >= 0.95


@pytest.mark.parametrize("name", ("obs_carwash_near", "obs_carwash_inside"))
def test_gazebo_shaded_yellow_streamers_stay_passable(name):
    view = load(name)
    points, rgb = bent_streamers(view, "tangled")
    ribbon = view.parts == sc.PART_CARWASH_RIBBON
    rgb[ribbon] = 0x64622B  # observed Gazebo streamer shade
    seg = cs.segment(points, view.pitch, view.roll, rgb=rgb)
    assert np.mean(seg.blocking[ribbon & np.isfinite(points).all(axis=1)]) <= 0.01


@pytest.mark.parametrize(
    "name",
    (
        "obs_carwash_near",
        "obs_carwash_oblique",
        "obs_carwash_inside",
        "obs_carwash_bucket_behind",
    ),
)
@pytest.mark.parametrize(
    "mode", ("forward", "backward", "left", "right", "diagonal", "mixed", "tangled")
)
def test_windblown_yellow_streamers_do_not_block_the_wash(name, mode):
    view = load(name)
    points, rgb = bent_streamers(view, mode)
    seg = cs.segment(points, view.pitch, view.roll, rgb=rgb)
    level = cs.level_points(points, view.pitch, view.roll)
    near = (
        (view.parts == sc.PART_CARWASH_RIBBON)
        & np.isfinite(points).all(axis=1)
        & (np.hypot(level[:, 0], level[:, 1]) <= EVAL_RANGE)
    )
    assert near.sum() >= 1000
    assert any(gate.kind == cs.CARWASH for gate in seg.gates)
    assert np.mean(seg.blocking[near]) <= 0.01, f"{name} {mode}: ribbon points block"
    truth_scan = _scan(level, view.truth_blocking)
    scan = cs.scan_from_segmentation(seg, SCAN_BINS, SCAN_FOV, SCAN_RANGE)
    agreement = np.mean(np.abs(scan - truth_scan) <= 0.10 + 0.05 * truth_scan)
    assert agreement >= 0.85, f"{name} {mode}: scan agreement {agreement:.0%}"
    if name == "obs_carwash_bucket_behind":
        bucket = (view.parts == sc.PART_BUCKET) & np.isfinite(points).all(axis=1)
        assert np.mean(seg.blocking[bucket]) >= 0.80
        bale = (
            (view.parts == sc.PART_BALE)
            & (view.truth == cs.OBSTACLE)
            & np.isfinite(points).all(axis=1)
        )
        assert np.mean(seg.blocking[bale]) >= 0.95


# ---------------------------------------------------------- fixture views


SCENARIO_NAMES = [s.name for s in sc.SCENARIOS]

# Views the segmenter is known to get wrong, and why. Strict: when a change
# fixes one, its test fails as XPASS until it is taken off this list, so the
# list cannot quietly go stale.
_INSIDE = (
    "camera 0.19 m behind a ribbon curtain, nearer than the ZED's minimum "
    "depth; the next arch is seen only through the gaps, and the bale walls "
    "either side, their lower halves hidden by ribbons, join it"
)
_DEEP_ARCHES = (
    "arches behind the first are seen through its ribbons; with the floor "
    "under them hidden, the surface is grown onto the ribbons' lower ends, "
    "so from 2-4 m some ribbons read as standing on the ground and block"
)
KNOWN_LIMITATIONS = {
    ("obstacles", "obs_carwash_inside"): _INSIDE,
    ("noise", "obs_carwash_inside"): _INSIDE,
    ("scan", "obs_carwash_far"): _DEEP_ARCHES,
    ("noise", "obs_carwash_far"): _DEEP_ARCHES,
}


def scenarios(test: str):
    return [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True, reason=KNOWN_LIMITATIONS[(test, name)]
            ),
        )
        if (test, name) in KNOWN_LIMITATIONS
        else name
        for name in SCENARIO_NAMES
    ]


@pytest.mark.parametrize("name", scenarios("ground"))
def test_ground_is_not_called_obstacle(name):
    result = score(load(name))
    assert result["false_obstacle"] <= 0.02, (
        f"{result['false_obstacle']:.1%} of {result['ground_points']} ground points "
        "within 4 m were reported as blocking"
    )


@pytest.mark.parametrize("name", scenarios("obstacles"))
def test_obstacles_are_not_called_passable(name):
    result = score(load(name))
    assert result["missed_obstacle"] <= 0.05, (
        f"{result['missed_obstacle']:.1%} of {result['obstacle_points']} obstacle points "
        "within 4 m were reported as ground, overhead or a gate"
    )


@pytest.mark.parametrize("name", scenarios("scan"))
def test_scan_matches_the_truth(name):
    result = score(load(name))
    assert result["scan_agreement"] >= 0.90, (
        f"only {result['scan_agreement']:.0%} of bins agree\n"
        f"scan  {np.round(result['scan'], 2)}\ntruth {np.round(result['truth_scan'], 2)}"
    )


@pytest.mark.parametrize(
    "name", [s.name for s in sc.SCENARIOS if s.expect], ids=lambda n: n
)
def test_expected_parts_are_seen_and_classed(name):
    result = score(load(name))
    for part, (count, recall) in result["parts"].items():
        _, threshold = EXPECT_OVERRIDES.get((name, part), EXPECT[part])
        assert count >= 50, f"{part} is not in view ({count} points within 4 m)"
        assert recall >= threshold, (
            f"{part}: {recall:.1%} classed right, need {threshold:.0%}"
        )


@pytest.mark.parametrize("name", scenarios("noise"))
def test_holds_up_under_stereo_noise_and_imu_error(name):
    # The rendered cloud is perfect and the IMU exact; the car's are not. One
    # degree of pitch and roll error and the stereo range noise the sim adds
    # is the minimum any of this has to survive. Looser than the clean
    # tests: noise is honest error, the question is only whether the
    # segmenter degrades or collapses.
    view = load(name)
    noisy = stereo_noise(view.points, seed=zlib.crc32(name.encode()))
    result = score(
        view, noisy, view.pitch + math.radians(1.0), view.roll - math.radians(1.0)
    )
    assert result["false_obstacle"] <= 0.05, (
        f"false obstacle {result['false_obstacle']:.1%}"
    )
    assert result["missed_obstacle"] <= 0.10, (
        f"missed obstacle {result['missed_obstacle']:.1%}"
    )
    assert result["scan_agreement"] >= 0.80, (
        f"scan agreement {result['scan_agreement']:.0%}"
    )


LEGACY = HERE.parents[2] / "rl" / "bale_follower"


@pytest.mark.parametrize("name", [s.name for s in sc.SCENARIOS if s.course == "speed"])
def test_speed_course_height_band_filter_matches_the_truth(name):
    # The filter the Speed Course policy was trained on: one height band,
    # leveled by the IMU. On the flat Speed Course it should be as good as
    # the full segmenter, start signal included.
    if not (LEGACY / "cloud_scan.py").exists():
        pytest.skip("rl/bale_follower is not beside this package")
    sys.path.insert(0, str(LEGACY))
    import cloud_scan

    view = load(name)
    truth = _scan(view.level, view.truth_blocking)
    noisy = stereo_noise(view.points, seed=zlib.crc32(name.encode()))
    for label, points, pitch, roll, needed in (
        ("clean", view.points, view.pitch, view.roll, 0.90),
        # What training actually fed it: the noisy sim cloud, and an IMU a
        # degree out.
        (
            "noisy",
            noisy,
            view.pitch + math.radians(1.0),
            view.roll - math.radians(1.0),
            0.85,
        ),
    ):
        scan = cloud_scan.scan_from_points(
            points[np.isfinite(points).all(axis=1)],
            SCAN_BINS,
            SCAN_FOV,
            SCAN_RANGE,
            pitch=pitch,
            roll=roll,
        )
        agreement = float(np.mean(np.abs(scan - truth) <= 0.10 + 0.05 * truth))
        assert agreement >= needed, (
            f"{label}: {agreement:.0%}\nscan  {np.round(scan, 2)}\ntruth {np.round(truth, 2)}"
        )


def test_every_fixture_belongs_to_a_scenario():
    stale = sorted(p.stem for p in FIXTURES.glob("*.npz") if p.stem not in sc.BY_NAME)
    assert not stale, f"fixtures with no scenario: {stale}"


# -------------------------------------------------------------------- report


def report(names) -> None:
    head = f"{'scenario':30s} {'falseObs':>8s} {'missed':>7s} {'scan':>5s} {'n~scan':>6s}  parts"
    print(head)
    for name in names:
        view = load(name)
        clean = score(view)
        noisy = score(
            view,
            stereo_noise(view.points, seed=zlib.crc32(name.encode())),
            view.pitch + math.radians(1.0),
            view.roll - math.radians(1.0),
        )
        parts = " ".join(f"{p}={r:.0%}/{n}" for p, (n, r) in clean["parts"].items())
        print(
            f"{name:30s} {clean['false_obstacle']:8.1%} {clean['missed_obstacle']:7.1%} "
            f"{clean['scan_agreement']:5.0%} {noisy['scan_agreement']:6.0%}  {parts}"
        )


if __name__ == "__main__":
    if "--report" in sys.argv:
        wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
        report(wanted or SCENARIO_NAMES)
    else:
        sys.exit(pytest.main([__file__, "-q"]))
