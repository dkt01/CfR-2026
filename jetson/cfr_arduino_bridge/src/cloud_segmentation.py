"""Per-point classes for the ZED cloud: ground, obstacle, hoop, car wash, overhead.

`rl/bale_follower/cloud_scan.py` decides "obstacle or not" with one fixed
height band, which is enough on the Speed Course's flat floor and nowhere
else. The Obstacle Course breaks it four ways:

  * **The ground is not flat.** The overpass ramp climbs 0.64 m in 3.3 m, so
    2 m up it the surface is higher than a straw bale is tall. The helix
    descends, the bank rolls the car 9 degrees, the pothole board and gravel
    tray stand proud of the floor and are dished or littered on top.
  * **Hoops are to be driven through,** and seen from the car they are two
    posts and a bar at bale height.
  * **The car wash is to be driven through** although its ribbons render as
    a curtain from 0.12 m up to 0.54 m -- a wall, to a height band.
  * **Some structure is overhead:** the tunnel roof, the deck above it.

So instead of one band, this

  1. levels the cloud with the IMU's pitch and roll;
  2. grows the drivable surface outward from under the car's own wheels on
     a plan grid, accepting a cell when its lowest return continues the
     surface it borders (within a grade and a step -- ramps and dishes
     continue it, a bale face does not);
  3. measures every point's height above the surface under it: close to it
     is GROUND, below it is ground too (a lower level seen over an edge);
  4. calls a column OVERHEAD when nothing in it comes lower than the car's
     roof, and OBSTACLE otherwise;
  5. finds *gates* -- thin structures standing on two feet with the span
     between them open underneath -- and relabels them HOOP when the span
     is open to the car's height and CARWASH when it is hung with a curtain.

Every step reads only the cloud, the camera's mounting, and the IMU's
gravity vector, so this runs unchanged on the robot. Nothing reads sim pose
or the world file.

Frames: points arrive in REP-103 body convention relative to the camera
(+x forward, +y left, +z up), as Gazebo's rgbd_camera and the ZED wrapper's
`point_cloud/cloud_registered` both publish them. Pitch is nose-down
positive and roll left-side-up positive, matching `cloud_scan`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class Params:
    """Tunables, in meters unless named otherwise.

    The defaults come from the course and the car, not from fitting the
    fixtures: each says which measurement it is.
    """

    max_range: float = 6.0
    # The ZED 2i's minimum depth. Anything nearer is a stereo artifact.
    min_depth: float = 0.30
    # Non-ground returns a 5 cm column needs to be believed: at least this
    # many, and at least speckle_fill of what a face at that range gives.
    min_column_points: int = 3
    speckle_fill: float = 0.05
    # Angle between neighboring pixels: 1.92 rad over 640 columns.
    pixel_angle: float = 0.003
    # Where the camera sits relative to the chassis origin, which is on the
    # ground plane between the wheels (sensors_world.SENSORS_CAMERA).
    camera_forward: float = 0.315
    camera_height: float = 0.20
    # Wheel contacts relative to the chassis origin (generate_vehicle_model).
    half_wheelbase: float = 0.162
    half_track: float = 0.145

    # Plan grid the drivable surface is grown on.
    cell: float = 0.10
    # A neighboring cell continues the surface if its lowest return is
    # within step + grade * distance of it. The steepest drivable surface is
    # the overpass ramp at 19%; the tallest drivable step is the 2 in gravel
    # tray rail. A bale is 0.356 m, a bucket 0.38 m, a guard rail 0.152 m.
    max_grade: float = 0.25
    max_step: float = 0.06
    # How far growth may carry the surface across cells with no returns --
    # shadows behind bumps and ribbons, and the rows of floor that thin out
    # with range. Grows with range because the camera's row spacing on the
    # ground does: about 0.015 * r^2 m for a 0.2 m high camera.
    gap_base: float = 0.25
    gap_per_range_sq: float = 0.02
    # Which low percentile of a cell's heights stands for its surface.
    surface_percentile: float = 0.10
    surface_band: float = 0.04
    # Seeds: cells within this plan distance of the camera that lie on the
    # plane of the car's own wheels.
    seed_radius: float = 1.2
    seed_tolerance: float = 0.06

    # A point this high or less above the surface under it is ground. Above
    # the 19 mm pothole bumps, the 13 mm gate base plates and the pebbles;
    # below the car's 40 mm belly, so anything it would scrape is not ground.
    ground_tolerance: float = 0.05
    # A column whose lowest non-ground return is above this is overhead: the
    # car, camera housing included, stands 0.23 m; the tunnel roof is 0.62.
    clearance: float = 0.32

    # Gates. The plan grid is finer than the surface grid because a hoop's
    # tube is 34 mm across.
    gate_cell: float = 0.05
    # Hoop top 0.537 m, car wash arch top 0.537 m. The start signal board is
    # 1.22 m and the deck's guard rails top out at 0.79 m; neither may qualify.
    gate_min_top: float = 0.40
    gate_max_top: float = 0.70
    # A foot touches the ground; the span between the feet does not. A
    # post's lowest non-ground return sits just above ground_tolerance; the
    # car wash ribbons stop 0.115 m up, which over its 13 mm base plate is
    # 0.10 m above the surface under them.
    foot_height: float = 0.08
    # The camera's horizontal field of view (sensors_world.SENSORS_CAMERA),
    # and how near its edge a gate's end counts as running out of frame.
    hfov_deg: float = 110.0
    frame_margin_deg: float = 3.0
    # The span must fit the car (0.30 m) with room either side.
    gate_min_span: float = 0.40
    gate_max_span: float = 1.40
    # Thin in the direction of travel: a hoop's base is 0.4 m deep but it is
    # ground; above it the frame is one 34 mm tube.
    gate_max_thickness: float = 0.15
    # Bearing resolution of the see-through test for open space under a
    # column: about 3 cm at 3 m.
    see_through_deg: float = 0.5
    # How many gate cells from a hanging run's end its support may be: the
    # bale walls either side of the car wash stand 0.12 m off its arches.
    support_reach: int = 3
    # A grounded run longer than this is a wall, not a gate's foot; one no
    # bigger than post_size may be a post (the uprights are 34 mm tubes).
    wall_length: float = 0.30
    post_size: float = 0.15
    # The car wash is five arches at 0.457 m pitch: 1.83 m first to last.
    carwash_depth: float = 1.95
    # Of the span's length, how much hangs below car height for it to be a
    # curtain (car wash) rather than open (hoop).
    curtain_fraction: float = 0.25
    # Depth noise to allow for in the thickness test, as the stereo model
    # zed_cloud_noise_node applies: sigma = a + b r^2.
    noise_a: float = 0.01
    noise_b: float = 0.008


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


def _shift(array: np.ndarray, dx: int, dy: int, fill) -> np.ndarray:
    """array[i - dx, j - dy], padded with `fill` where that falls outside."""
    out = np.full_like(array, fill)
    h, w = array.shape
    xs = slice(max(dx, 0), h + min(dx, 0))
    xd = slice(max(-dx, 0), h + min(-dx, 0))
    ys = slice(max(dy, 0), w + min(dy, 0))
    yd = slice(max(-dy, 0), w + min(-dy, 0))
    out[xs, ys] = array[xd, yd]
    return out


_NEIGHBORS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]


def _grow_surface(zmin, seeds, cell, params, range_of_cell):
    """Drivable-surface height per cell, and which cells were reached.

    Relaxation on the grid, all cells at once: a cell joins the surface when
    a neighbor already on it predicts its lowest return within step +
    grade * distance. Empty cells carry the neighbor's height forward for
    as long as the gap allowance at that range permits, so the surface can
    cross shadows and the thinning rows of far floor without inventing a
    surface where something solid stands.
    """
    surface = np.where(seeds, zmin, np.nan)
    reached = seeds.copy()
    gap = np.where(seeds, 0.0, np.inf)
    occupied = np.isfinite(zmin)
    allowance = params.gap_base + params.gap_per_range_sq * range_of_cell**2
    for _ in range(4 * max(zmin.shape)):
        changed = False
        for dx, dy in _NEIGHBORS:
            step = cell * math.hypot(dx, dy)
            n_surface = _shift(surface, dx, dy, np.nan)
            n_gap = _shift(gap, dx, dy, np.inf)
            n_reached = _shift(reached, dx, dy, False)
            # Occupied cells: accept when the return continues the surface.
            # The tolerance grows with the gap crossed to get here, since
            # the surface may have kept climbing unseen.
            reach = step + np.where(np.isfinite(n_gap), n_gap, 0.0)
            ok = (
                occupied
                & ~reached
                & n_reached
                & (
                    np.abs(zmin - n_surface)
                    <= params.max_step + params.max_grade * reach
                )
            )
            if ok.any():
                surface[ok] = zmin[ok]
                gap[ok] = 0.0
                reached[ok] = True
                changed = True
            # Empty cells: carry the surface across, within the allowance.
            carry = (
                ~occupied
                & n_reached
                & (n_gap + step <= allowance)
                & (n_gap + step < gap)
            )
            if carry.any():
                surface[carry] = n_surface[carry]
                gap[carry] = n_gap[carry] + step
                reached[carry] = True
                changed = True
        if not changed:
            break
    return surface, reached & occupied


def _fill(surface, known, passes):
    """Spread known surface heights into unknown cells, `passes` cells deep."""
    surface = surface.copy()
    known = known.copy()
    for _ in range(passes):
        total = np.zeros_like(surface)
        count = np.zeros_like(surface)
        for dx, dy in _NEIGHBORS:
            n_known = _shift(known, dx, dy, False)
            total += np.where(
                n_known, _shift(np.where(known, surface, 0.0), dx, dy, 0.0), 0.0
            )
            count += n_known
        new = ~known & (count > 0)
        if not new.any():
            break
        surface[new] = total[new] / count[new]
        known |= new
    return surface, known


def _seen_under(x, y, z, is_ground, params):
    """Whether a ray to the ground further out passed below each point.

    Per bearing bin, the steepest ground ray beyond a point's range is the
    one that passes lowest under it; the point is seen under if that ray is
    below it at its range. Exact bearing bins only -- a ray past the edge of
    a board has not been under the board.
    """
    bins = int(round(360.0 / params.see_through_deg))
    bearing = ((np.arctan2(y, x) + math.pi) / (2 * math.pi) * bins).astype(
        np.int64
    ) % bins
    r = np.hypot(x, y)
    g_bin, g_r = bearing[is_ground], r[is_ground]
    if g_r.size == 0:
        return np.zeros(len(x), dtype=bool)
    slope = z[is_ground] / np.maximum(g_r, 1e-6)
    order = np.lexsort((g_r, g_bin))
    g_bin, g_r, slope = g_bin[order], g_r[order], slope[order]
    # Suffix minimum of slope within each bin: offsetting by bin keeps a
    # later bin's values from ever winning an earlier bin's minimum.
    offset = 10.0 * g_bin
    steepest = np.minimum.accumulate((slope + offset)[::-1])[::-1] - offset
    span = float(params.max_range) * 4 + 10.0
    keys = g_bin * span + g_r
    query = bearing * span + r + params.cell
    at = np.searchsorted(keys, query)
    ok = at < len(keys)
    at = np.minimum(at, len(keys) - 1)
    ok &= g_bin[at] == bearing
    return ok & (steepest[at] * r < z - 0.02)


def _components(occupied: np.ndarray) -> np.ndarray:
    """8-connected component index per cell (-1 where unoccupied)."""
    h, w = occupied.shape
    big = np.iinfo(np.int64).max
    label = np.where(occupied, np.arange(h * w, dtype=np.int64).reshape(h, w), big)
    while True:
        best = label
        for dx, dy in _NEIGHBORS:
            best = np.minimum(best, _shift(label, dx, dy, big))
        best = np.where(occupied, best, big)
        if np.array_equal(best, label):
            break
        label = best
    return np.where(occupied, label, -1)


def _find_gates(level, height, candidate, column_low, params):
    """Gates among the candidate points: [(Gate, member indices, foot indices)].

    A gate is found by what hangs, not by what stands. On a plan grid, a
    cell is *grounded* when its lowest return is within foot_height of the
    surface, and *hanging* when its lowest return is above that and its
    highest no taller than a gate. A gate is a thin run of hanging cells --
    a hoop's bar, a car wash arch with its ribbons -- held up at both ends
    by grounded cells: its own posts, or a wall standing against them, as
    the bale walls stand against the car wash's uprights. An end may also
    run out of the frame, which is how the car wash looks from inside it.
    """
    index = np.flatnonzero(candidate)
    if index.size == 0:
        return []
    pts = level[index]
    h = height[index]
    g = params.gate_cell
    ix = np.floor(pts[:, 0] / g).astype(np.int64)
    iy = np.floor(pts[:, 1] / g).astype(np.int64)
    x0, y0 = ix.min() - 2, iy.min() - 2
    shape = (ix.max() - x0 + 3, iy.max() - y0 + 3)
    cx, cy = ix - x0, iy - y0
    lowest = np.full(shape, np.inf)
    highest = np.full(shape, -np.inf)
    np.minimum.at(lowest, (cx, cy), h)
    np.maximum.at(highest, (cx, cy), h)
    grounded = lowest <= params.foot_height
    # Hanging needs the ground under it seen (column_low is -inf where it
    # was not); a far bucket whose foot hides behind a near one is not
    # hanging, it is merely half seen.
    evidence = np.zeros(shape, dtype=bool)
    evidence[cx, cy] = column_low[index] > params.foot_height
    hanging = (
        np.isfinite(lowest) & ~grounded & evidence & (highest <= params.gate_max_top)
    )
    # Grounded runs longer than a post are walls. They may hold a gate up,
    # but they are never part of it.
    runs = _components(grounded)
    wall = np.zeros(shape, dtype=bool)
    post = np.zeros(shape, dtype=bool)
    for run in np.unique(runs[grounded]):
        where = np.argwhere(runs == run) * g
        extent = max(np.ptp(where[:, 0]), np.ptp(where[:, 1]))
        if extent > params.wall_length:
            wall |= runs == run
        elif extent <= params.post_size:
            post |= runs == run
    support = grounded.copy()
    for dx, dy in _NEIGHBORS:
        support |= _shift(grounded, dx, dy, False)
    near_support = support
    for _ in range(params.support_reach - 1):
        grown = near_support.copy()
        for dx, dy in _NEIGHBORS:
            grown |= _shift(near_support, dx, dy, False)
        near_support = grown

    edge = math.radians(params.hfov_deg / 2 - params.frame_margin_deg)
    # Close gaps of up to four cells between hanging cells before grouping
    # them -- but only across the line of sight. Seen from inside the car
    # wash, the nearest curtain's bar is above the frame and what is left is
    # strips 51 mm wide with 76 mm between, and the arch beyond shows only
    # through those gaps, 0.15 m at a time. Along the line of sight stereo
    # noise smears each arch by several centimetres, and bridging that way
    # welds five arches 0.457 m apart into one blob.
    bearing = np.arctan2(
        (np.arange(shape[1]) + y0 + 0.5)[None, :],
        (np.arange(shape[0]) + x0 + 0.5)[:, None],
    )
    across_is_y = np.abs(bearing) < math.pi / 4
    bridged = hanging.copy()
    for dx, dy, lateral in ((0, 1, across_is_y), (1, 0, ~across_is_y)):
        before = np.zeros(shape, dtype=bool)
        after = np.zeros(shape, dtype=bool)
        for k in range(1, 5):
            before |= _shift(hanging, k * dx, k * dy, False)
            after |= _shift(hanging, -k * dx, -k * dy, False)
        bridged |= before & after & lateral
    component = np.where(hanging, _components(bridged), -1)
    point_component = component[cx, cy]
    gates = []
    for comp in np.unique(component[hanging]):
        cells = np.argwhere(component == comp)
        if len(cells) < 3:
            continue
        xy = (cells + [x0, y0] + 0.5) * g
        center = xy.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((xy - center).T))
        axis = vectors[:, 1]
        u = (xy - center) @ axis
        v = (xy - center) @ vectors[:, 0]
        span = float(u.max() - u.min()) + g
        if not params.gate_min_span <= span <= params.gate_max_span:
            continue
        distance = float(np.hypot(*center))
        sigma = params.noise_a + params.noise_b * distance**2
        if float(v.max() - v.min()) + g > params.gate_max_thickness + 2 * sigma:
            continue
        # The middle must hang over open floor. A tapered bucket's rim
        # overhangs its body by a couple of centimeters and so hangs too,
        # but right beside something grounded all along its length.
        middle = cells[np.abs(u) <= span / 4]
        if support[middle[:, 0], middle[:, 1]].any():
            continue
        # Each end must be held up: grounded cells beside its last cells, or
        # the frame's edge.
        ends = []
        for end in (u <= u.min() + g, u >= u.max() - g):
            end_cells = cells[end]
            held = bool(near_support[end_cells[:, 0], end_cells[:, 1]].any())
            end_xy = xy[end]
            cut = bool((np.abs(np.arctan2(end_xy[:, 1], end_xy[:, 0])) >= edge).any())
            ends.append("held" if held else "cut" if cut else None)
        if None in ends:
            continue
        members = point_component == comp
        top = float(h[members].max())
        if top < params.gate_min_top:
            continue
        curtain = float(np.mean(lowest[cells[:, 0], cells[:, 1]] < params.clearance))
        kind = CARWASH if curtain >= params.curtain_fraction else HOOP
        # Unseen feet are only trusted for a curtain: two posts nobody saw
        # do not make a hoop.
        if "cut" in ends and kind != CARWASH:
            continue
        # The posts: grounded, not wall, touching the hanging run's ends.
        end_mask = np.zeros(shape, dtype=bool)
        end_mask[cells[:, 0], cells[:, 1]] = True
        grown = end_mask.copy()
        for dx, dy in _NEIGHBORS:
            grown |= _shift(end_mask, dx, dy, False)
        for dx, dy in _NEIGHBORS:
            grown |= _shift(grown, dx, dy, False)
        # Only runs a post's size: a bale wall glimpsed through the ribbons
        # in short pieces is no longer long, but it is still not a post.
        posts = post & grown
        post_points = posts[cx, cy]
        foot_a = center + axis * u.min()
        foot_b = center + axis * u.max()
        gate = Gate(
            kind=kind,
            center=(float(center[0]), float(center[1])),
            feet=(tuple(map(float, foot_a)), tuple(map(float, foot_b))),
            span=span,
            top=top,
            axis=(float(axis[0]), float(axis[1])),
        )
        gates.append((gate, index[members | post_points], index[post_points]))
    return gates


def segment(
    points: np.ndarray,
    pitch: float = 0.0,
    roll: float = 0.0,
    params: Params = Params(),
) -> Segmentation:
    """Classify an (N, 3) body-frame cloud. See the module docstring."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    labels = np.full(n, UNKNOWN, dtype=np.uint8)
    height = np.full(n, np.nan)
    blocking = np.zeros(n, dtype=bool)
    level = level_points(points, pitch, roll)
    finite = np.isfinite(level).all(axis=1)
    plan = np.hypot(level[:, 0], level[:, 1])
    depth = np.linalg.norm(points, axis=1)
    valid = (
        finite
        & (plan <= params.max_range)
        & (level[:, 0] > -0.2)
        & (depth >= params.min_depth)
    )
    if not valid.any():
        return Segmentation(labels, height, blocking, level)

    # The plane the car's own wheels stand on, in the leveled frame.
    wheels = np.array(
        [
            [
                sx * params.half_wheelbase - params.camera_forward,
                sy * params.half_track,
                -params.camera_height,
            ]
            for sx in (-1, 1)
            for sy in (-1, 1)
        ]
    )
    wheels = level_points(wheels, pitch, roll)
    design = np.c_[np.ones(4), wheels[:, :2]]
    plane = np.linalg.lstsq(design, wheels[:, 2], rcond=None)[0]

    cell = params.cell
    x_lo = -0.5
    nx = int(math.ceil((params.max_range - x_lo) / cell)) + 1
    ny = int(math.ceil(2 * params.max_range / cell)) + 1
    vx, vy, vz = level[valid, 0], level[valid, 1], level[valid, 2]
    ci = np.clip(((vx - x_lo) / cell).astype(np.int64), 0, nx - 1)
    cj = np.clip(((vy + params.max_range) / cell).astype(np.int64), 0, ny - 1)
    # A low percentile of each cell's heights rather than its minimum: one
    # stereo outlier below the floor would otherwise set the cell's height,
    # and the next cell's genuine rise would then read as a step.
    # Taken over the returns in the cell's bottom band only, so a cell that
    # is mostly bale face still reads the floor at the face's foot.
    floor = np.full((nx, ny), np.inf)
    np.minimum.at(floor, (ci, cj), vz)
    band = vz <= floor[ci, cj] + params.surface_band
    zmin = np.full((nx, ny), np.nan)
    cell_id = (ci * ny + cj)[band]
    order = np.lexsort((vz[band], cell_id))
    ids, start, count = np.unique(cell_id[order], return_index=True, return_counts=True)
    pick = start + np.floor(params.surface_percentile * (count - 1)).astype(np.int64)
    zmin.reshape(-1)[ids] = vz[band][order][pick]

    gx = x_lo + (np.arange(nx) + 0.5) * cell
    gy = -params.max_range + (np.arange(ny) + 0.5) * cell
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    car_plane = plane[0] + plane[1] * GX + plane[2] * GY
    range_of_cell = np.hypot(GX, GY)
    # The car's own plane only holds under the car: pitched over the crest
    # of the ramp, it climbs 0.14 m per meter above the flat deck ahead. So
    # the tolerance opens by the grade allowance with distance past the
    # front axle.
    ahead = np.maximum(GX - (params.half_wheelbase - params.camera_forward), 0.0)
    seeds = (
        np.isfinite(zmin)
        & (range_of_cell <= params.seed_radius)
        & (np.abs(zmin - car_plane) <= params.seed_tolerance + params.max_grade * ahead)
    )
    surface, reached = _grow_surface(zmin, seeds, cell, params, range_of_cell)
    surface, known = _fill(surface, reached, passes=int(round(1.0 / cell)))
    surface = np.where(known, surface, car_plane)

    h = vz - surface[ci, cj]
    height[valid] = h
    is_ground = h <= params.ground_tolerance

    # Columns: the lowest non-ground return in each surface cell. Overhead
    # also needs evidence that the space under it is open -- ground seen in
    # or beside the column. A structure whose lower half is hidden behind
    # something nearer (the start signal behind its bale wall) has nothing
    # below it only because nothing below it was seen.
    lowest = np.full((nx, ny), np.inf)
    np.minimum.at(lowest, (ci[~is_ground], cj[~is_ground]), h[~is_ground])
    # The overhead test itself runs on the finer gate grid: in a 0.1 m cell
    # the tunnel's roof shares a column with the top of its wall.
    fine = params.gate_cell
    fi = np.floor((vx - x_lo) / fine).astype(np.int64)
    fj = np.floor((vy + params.max_range) / fine).astype(np.int64)
    fine_lowest = np.full((fi.max() + 1, fj.max() + 1), np.inf)
    np.minimum.at(fine_lowest, (fi[~is_ground], fj[~is_ground]), h[~is_ground])
    # Evidence of open space under a column: ground seen in the column
    # itself, or ground seen *beyond* it on the same bearing -- a ray that
    # reached the floor further out passed under whatever is here. Ground
    # a fine cell away does not count: as often as not it is the floor in
    # front of whatever hides this column's lower half (the start signal
    # behind its bale wall, a bucket behind a nearer one), or the one foot
    # of it that does show.
    open_below = np.zeros(fine_lowest.shape, dtype=bool)
    open_below[fi[is_ground], fj[is_ground]] = True
    seen_under = ~is_ground & _seen_under(vx, vy, vz, is_ground, params)
    open_below[fi[seen_under], fj[seen_under]] = True
    overhead = (
        ~is_ground & (fine_lowest[fi, fj] > params.clearance) & open_below[fi, fj]
    )
    plan_range = np.hypot(vx, vy)
    # Speckle: a stray return, far off the surface it belongs to, standing
    # alone in its column. Real structure puts several returns in a 5 cm
    # column even at the far end of the scan.
    # How many is several scales with range: a face fills a 5 cm column
    # with (0.05 / (range * pixel_angle))^2 returns, some 200 at 1 m and 6
    # at 6 m. Stereo error scatters a far return anywhere along its ray, so
    # a near column holding a handful of returns is that, not an object.
    fine_count = np.zeros(fine_lowest.shape, dtype=np.int64)
    np.add.at(fine_count, (fi[~is_ground], fj[~is_ground]), 1)
    expected = (
        params.gate_cell / (params.pixel_angle * np.maximum(plan_range, 0.1))
    ) ** 2
    needed = np.maximum(params.min_column_points, params.speckle_fill * expected)
    speckle = ~is_ground & (fine_count[fi, fj] < needed)

    valid_index = np.flatnonzero(valid)
    labels[valid_index] = np.where(
        is_ground,
        GROUND,
        np.where(speckle, UNKNOWN, np.where(overhead, OVERHEAD, OBSTACLE)),
    )
    blocking[valid_index] = ~is_ground & ~overhead & ~speckle

    candidate = np.zeros(n, dtype=bool)
    candidate[valid_index] = ~is_ground & ~speckle & (h <= 2.0)
    gates = []
    # What hangs, per point: the lowest return in its column is off the
    # ground, and the ground under it was actually seen.
    column_low = np.full(n, -np.inf)
    column_low[valid_index] = np.where(open_below[fi, fj], fine_lowest[fi, fj], -np.inf)
    # Beside a grounded column: a tapered bucket's rim overhangs its body by
    # a couple of centimeters with floor showing under the lip, which is not
    # hanging in any sense a car wash cares about.
    grounded_fine = fine_lowest <= params.foot_height
    beside = grounded_fine.copy()
    for dx, dy in _NEIGHBORS:
        beside |= _shift(grounded_fine, dx, dy, False)
    beside_ground = np.zeros(n, dtype=bool)
    beside_ground[valid_index] = beside[fi, fj]
    for gate, members, feet in _find_gates(
        level, height, candidate, column_low, params
    ):
        # Right beside something grounded is part of that thing -- the top
        # edge of a bale wall the gate leans on -- unless it is the gate's
        # own post.
        members = members[~beside_ground[members] | np.isin(members, feet)]
        labels[members] = gate.kind
        blocking[members] = False
        blocking[feet] = height[feet] <= params.clearance
        _claim_footprint(gate, level, height, labels, blocking, candidate, params)
        gates.append(gate)
        if gate.kind == CARWASH:
            _claim_car_wash(
                gate, level, height, labels, blocking, column_low, beside_ground, params
            )
    return Segmentation(labels, height, blocking, level, gates)


def _claim_footprint(gate, level, height, labels, blocking, candidate, params):
    """The rest of a gate's bar, in its own thin footprint.

    The hanging run stops short of each post: the post hides the floor under
    the bar beside it, so there is no evidence that stretch hangs. Inside the
    span everything above the ground belongs to the gate; past the span's
    ends only what is higher than any bale does, because the walls a gate may
    lean on stand there. None of it blocks -- the posts were claimed, and
    made blocking, with the gate itself.
    """
    axis = np.array(gate.axis)
    normal = np.array([-axis[1], axis[0]])
    rel = level[:, :2] - np.array(gate.center)
    along = np.abs(rel @ axis)
    across = np.abs(rel @ normal)
    distance = math.hypot(*gate.center)
    sigma = params.noise_a + params.noise_b * distance**2
    inside = along <= gate.span / 2
    beyond = ~inside & (
        along <= gate.span / 2 + params.support_reach * params.gate_cell
    )
    mine = (
        candidate
        & np.isin(labels, [OBSTACLE, OVERHEAD])
        & (across <= params.gate_max_thickness / 2 + 2 * sigma)
        & (height <= params.gate_max_top)
        & (inside | (beyond & (height >= params.gate_min_top)))
    )
    labels[mine] = gate.kind
    blocking[mine] = False


def _claim_car_wash(
    gate, level, height, labels, blocking, column_low, beside_ground, params
):
    """Label the rest of a car wash once one of its arches is found.

    The arches behind the first are seen through its ribbons, in pieces,
    and a piece has no feet to be recognized by. But a car wash is a row of
    arches carwash_depth deep, so what hangs inside that footprint -- above
    the ground, not standing on it -- is more of the same. Anything that
    does stand on the ground inside it keeps its own label.
    """
    axis = np.array(gate.axis)
    normal = np.array([-axis[1], axis[0]])
    mid = np.array(gate.center)
    rel = level[:, :2] - mid
    along = rel @ axis
    across = rel @ normal
    half = gate.span / 2 + params.gate_cell
    inside = (
        np.isin(labels, [OBSTACLE, OVERHEAD])
        & (np.abs(along) <= half + params.gate_cell)
        & (np.abs(across) <= params.carwash_depth)
        & (height <= params.gate_max_top)
    )
    # Only what hangs: whatever stands inside or beside it -- its own
    # uprights, the bale walls flush against them, a parked bucket -- keeps
    # the label it already has, and keeps blocking.
    hanging = inside & (column_low > params.foot_height) & ~beside_ground
    labels[hanging] = CARWASH
    blocking[hanging] = False


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
