#!/usr/bin/env python3
"""What the segmented ZED cloud tells the car, computed from the course model.

On the car and in Gazebo the observation comes from the real pipeline:
ZED cloud -> cloud_segmentation (C++) -> `scan_from_segmentation` (nearest
BLOCKING return per bearing bin, from the camera, in the leveled frame) and
`seg.gates` (hoops and car-wash arches found as two feet with an open span).
Training cannot afford to render a cloud per step, so this produces the same
two things by ray-marching the course model's sensor grid:

  * The grid already holds the segmenter's height rule per cell per exposed
    layer (course_model: a return between 0.05 and 0.32 m above the layer
    blocks; the tunnel roof and the hoop bar are overhead; ramps, the bank,
    the gravel tray and the potholes are ground; car-wash ribbons are their
    own class), baked from the VISUAL meshes the camera renders.
  * Each ray starts at the camera where the body's heave, pitch and roll put
    it, walks the plan grid on the car's own level, and returns the first
    blocking cell that is (a) inside the camera's vertical field of view
    along that bearing -- a car pitched up the ramp looks over low things
    close in; one pitched down at the crest looks into the floor -- and (b)
    not hidden behind the drivable surface it has crossed, so the far side
    of a crest is out of sight until the car is over it.  Nearer than the
    ZED's 0.30 m minimum depth nothing is seen at all.
  * Noise follows zed_cloud_noise_node's range law, plus dropouts, near
    phantoms, and the segmenter's known weakness: the car wash seen from 2 m
    or more can read as solid.  The IMU attitude the segmenter levels with
    is perturbed, which moves returns across the ground/obstacle threshold
    at range.

Gates are geometric: a hoop or arch is reported when both feet are in the
horizontal field of view, in range, and not behind something the scan hits
first -- the conditions the segmenter needs to see two feet.

    python3 sensor.py --fixtures   # model vs the real segmenter on the
                                   # Gazebo-rendered fixtures (needs the lib)
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

import course_model
import plant as P
import world as world_module

# Gate kinds, as the segmenter's.
HOOP, CARWASH = 2, 3
HOOP_HALF_SPAN = 0.292
ARCH_HALF_SPAN = 0.5755
GATE_FEATURES = 4  # valid, bearing, range, normal angle


def gate_table(model) -> np.ndarray:
    """Per layout: (kind, cx, cy, axis_x, axis_y, half_span) for every gate."""
    base = world_module.parse()
    arches = []
    for p in base.prims:
        if (
            p.model == "car_wash"
            and p.source == "visual"
            and p.name.startswith("arch_")
        ):
            m = p.world
            arches.append((m[0, 3], m[1, 3], m[0, 1], m[1, 1]))
    rows = []
    for layout in model.layouts:
        gates = []
        for name, (x, y) in layout["hoops"].items():
            yaw = float(model.spec["hoops"][name]["yaw"])
            gates.append((HOOP, x, y, math.cos(yaw), math.sin(yaw), HOOP_HALF_SPAN))
        for x, y, ax, ay in arches:
            gates.append((CARWASH, x, y, ax, ay, ARCH_HALF_SPAN))
        rows.append(gates)
    return np.asarray(rows, np.float64)


class SensorConfig:
    def __init__(self, cfg: dict):
        s = cfg["sensor"]
        self.bins = int(s["bins"])
        self.vec = np.array(
            [
                math.radians(s["fov_deg"]),
                s["max_range"],
                s["min_range"],
                math.radians(s["vfov_deg"]),
                s["camera_forward"],
                s["camera_height"],
                s["ray_step"],
                s["rays_per_bin"],
                s["noise_a"],
                s["noise_b"],
                s["bin_noise_scale"],
                s["dropout"],
                s["phantom"],
                s["carwash_misread_range"],
                s["carwash_misread_prob"],
                math.radians(s["attitude_noise_deg"]),
                s["gate_max_range"],
                math.radians(s["gate_frame_margin_deg"]),
            ],
            np.float64,
        )


# How far the followed surface may step between two cells and still be the
# same surface: the segmenter's max_step, plus its max_grade over one step.
FOLLOW_STEP = 0.06


# Open-floor skipping.  Most of a ray's 2 cm samples cross open floor, and
# nothing on open floor can stop the ray or move the surface it follows.  A
# cell is "open" when it has one exposed layer at the same height as its four
# neighbors, no car-wash ribbon, and nothing visual that reaches
# GROUND_TOLERANCE above the lowest surface a ray can follow (so it cannot
# block against any surface).  sight_skip stores, per layout and cell, how
# far a ray can go from anywhere in the cell and stay on open cells.
#
# Over such a stretch the result cannot change: no sample blocks, the
# followed surface is flat and unchanged, and the horizon it sets is either
# already at its highest (surface above the camera) or at its highest on the
# last sample, which the ray still reads (surface below the camera).
SKIP_MARGIN = 2  # cells: a point anywhere in either cell, not the centers
SKIP_BELOW = 0.05  # m under the lowest layer that a followed surface may sit


def sight_skip(model):
    """(SKIP, skip_from): per-layout skip distances in cells, uint8, on the
    static grid's (row, col), and the lowest followed surface they hold for."""
    from scipy.ndimage import distance_transform_edt

    s, w = model.static, model.window
    tops = [s["E_top"][s["E_n"] > m, m] for m in range(s["E_top"].shape[-1])]
    tops += [w["E_top"][w["E_n"] > m, m] for m in range(w["E_top"].shape[-1])]
    lowest = float(min(t.min() for t in tops if t.size))
    skip_from = lowest - SKIP_BELOW
    reach = skip_from + course_model.GROUND_TOLERANCE

    def open_(f):
        # One exposed layer, nothing that can block, no car-wash ribbon; the
        # layer's height rides along in the same array for the flat test.
        kv = np.arange(f["V_hi"].shape[-1])
        tall = ((f["V_hi"] > reach) & (kv < f["V_n"][..., None])).any(-1)
        ok = (f["E_n"] == 1) & ~tall & (f["C_n"] == 0)
        return np.where(ok, f["E_top"][..., 0], np.nan)

    out = []
    for top in P.per_layout(model, open_):
        # NaN (not open) is unequal to everything, so this is also the open
        # test; the grid's edge rows stay closed: off the grid is floor.
        flat = np.zeros(top.shape, bool)
        c = top[1:-1, 1:-1]
        flat[1:-1, 1:-1] = (
            (c == top[2:, 1:-1])
            & (c == top[:-2, 1:-1])
            & (c == top[1:-1, 2:])
            & (c == top[1:-1, :-2])
        )
        cells = np.floor(distance_transform_edt(flat)) - SKIP_MARGIN
        out.append(np.clip(cells, 0, 255).astype(np.uint8))
    return np.stack(out), skip_from


@nb.njit(cache=True, inline="always")
def _skip_steps(SKIP, G, lay, x, y, step):
    """Ray steps from (x, y) that stay on open floor, at least 1."""
    i = int(math.floor((x - G[0]) / course_model.RES))
    j = int(math.floor((y - G[1]) / course_model.RES))
    if i < 0 or j < 0 or i >= SKIP.shape[2] or j >= SKIP.shape[1]:
        return 1
    return max(1, int(SKIP[lay, j, i] * course_model.RES / step))


@nb.njit(cache=True, inline="always")
def _cast(
    LAY,
    VIS,
    WASH,
    lay,
    cx,
    cy,
    cz,
    zref,
    heading,
    bearing,
    el_center,
    vfov,
    max_range,
    min_range,
    step,
    wash_solid_from,
    SKIP,
    skip_from,
):
    """Plan range to the first visible blocking cell along one ray, or max."""
    c = math.cos(heading + bearing)
    s = math.sin(heading + bearing)
    horizon = -10.0
    half_v = 0.5 * vfov
    r = step
    while r <= max_range:
        px = cx + r * c
        py = cy + r * s
        # Follow the surface out from under the car, as the segmenter grows
        # it from the wheels: a ramp stays road all the way up, and across a
        # drop-off (the deck's edge) the surface stays where the car is.
        top = course_model.nearest_layer(LAY, lay, px, py, zref)
        if abs(top - zref) <= FOLLOW_STEP:
            zref = top
            el_ground = math.atan2(top - cz, r)
            if el_ground > horizon:
                horizon = el_ground
        if r >= min_range:
            blo, bhi = course_model.visual_band(VIS, lay, px, py, zref)
            solid = bhi >= blo
            if not solid and r >= wash_solid_from:
                blo, bhi = course_model.visual_band(WASH, lay, px, py, zref)
                solid = bhi >= blo
            if solid:
                el_hi = math.atan2(bhi - cz, r)
                el_lo = math.atan2(blo - cz, r)
                if (
                    el_hi >= el_center - half_v
                    and el_lo <= el_center + half_v
                    and el_hi > horizon
                ):
                    return r
        # Open floor ahead (sight_skip): the samples it holds can neither
        # block nor change the followed surface, so step over them.  r
        # still advances one step at a time, to land on the same values.
        if zref >= skip_from:
            for _ in range(_skip_steps(SKIP, LAY[0], lay, px, py, step) - 1):
                r += step
        r += step
    return max_range


@nb.njit(cache=True, inline="always")
def _surface_at(LAY, lay, x0, y0, z0, x1, y1, step):
    """The followed surface's height at (x1, y1), walked out from the car.

    Rays start at the camera, 0.315 m ahead of the car; on the 19% ramp the
    surface there is 0.06 m above the car's own, more than one follow step,
    so starting from the car's height reads the whole ramp as a wall.
    """
    d = math.hypot(x1 - x0, y1 - y0)
    k = max(1, int(d / step))
    zref = z0
    for i in range(1, k + 1):
        f = i / k
        top = course_model.nearest_layer(
            LAY, lay, x0 + f * (x1 - x0), y0 + f * (y1 - y0), zref
        )
        if abs(top - zref) <= FOLLOW_STEP:
            zref = top
    return zref


@nb.njit(cache=True, parallel=True)
def sense(
    LAY,
    VIS,
    WASH,
    lay,
    x,
    y,
    z,
    yaw,
    pitch,
    roll,
    cfg,
    bins,
    gates,
    wash_misread,
    SKIP,
    skip_from,
    out_scan,
    out_gate,
):
    """Fill out_scan (n, bins) with ranges and out_gate (n, 8) with gates.

    `wash_misread` (n,) is 1 where this step's car-wash view reads solid.
    Noise is drawn here; numba's per-thread generator is seeded by the env.
    """
    fov, max_range, min_range, vfov = cfg[0], cfg[1], cfg[2], cfg[3]
    cam_fwd, cam_h, step = cfg[4], cfg[5], cfg[6]
    rays = int(cfg[7])
    na, nbq, nscale = cfg[8], cfg[9], cfg[10]
    dropout, phantom = cfg[11], cfg[12]
    wash_range = cfg[13]
    att_noise = cfg[15]
    gate_range, margin = cfg[16], cfg[17]
    n = x.shape[0]
    for k in nb.prange(n):
        cp = math.cos(pitch[k])
        cx = x[k] + cam_fwd * cp * math.cos(yaw[k])
        cy = y[k] + cam_fwd * cp * math.sin(yaw[k])
        cz = z[k] + cam_fwd * math.sin(pitch[k]) + cam_h * cp
        zref0 = _surface_at(LAY, lay[k], x[k], y[k], z[k], cx, cy, step)
        # The segmenter levels with the IMU's attitude, which is not quite
        # the true one: the error tips its idea of "up" by this much.
        dp = np.random.normal(0.0, att_noise)
        dr = np.random.normal(0.0, att_noise)
        wash_from = wash_range if wash_misread[k] else 1e9
        half = 0.5 * fov
        width = fov / bins
        for b in range(bins):
            best = max_range
            for q in range(rays):
                bearing = -half + width * (b + (q + 0.5) / rays)
                el_c = pitch[k] * math.cos(bearing) + roll[k] * math.sin(bearing)
                el_c += dp * math.cos(bearing) + dr * math.sin(bearing)
                rng = _cast(
                    LAY,
                    VIS,
                    WASH,
                    lay[k],
                    cx,
                    cy,
                    cz,
                    zref0,
                    yaw[k],
                    bearing,
                    el_c,
                    vfov,
                    max_range,
                    min_range,
                    step,
                    wash_from,
                    SKIP,
                    skip_from,
                )
                if rng < best:
                    best = rng
            if best < max_range:
                best += np.random.normal(0.0, nscale * (na + nbq * best * best))
            u = np.random.random()
            if u < dropout:
                best = max_range
            elif u < dropout + phantom:
                best = min_range + np.random.random() * 0.7
            out_scan[k, b] = min(max(best, min_range), max_range)

        # Gates: nearest visible hoop, nearest visible arch.
        for kind_i in range(2):
            kind = 2.0 + kind_i
            best_r = 1e9
            feat0 = 0.0
            feat1 = 0.0
            feat2 = 0.0
            feat3 = 0.0
            L = lay[k]
            for g in range(gates.shape[1]):
                if gates[L, g, 0] != kind:
                    continue
                gx, gy = gates[L, g, 1], gates[L, g, 2]
                ax, ay, hs = gates[L, g, 3], gates[L, g, 4], gates[L, g, 5]
                ok = True
                for side in (-1.0, 1.0):
                    fx = gx + side * hs * ax - cx
                    fy = gy + side * hs * ay - cy
                    fr = math.hypot(fx, fy)
                    fb = math.atan2(fy, fx) - yaw[k]
                    fb = (fb + math.pi) % (2 * math.pi) - math.pi
                    if fr > gate_range or abs(fb) > half - margin or fr < min_range:
                        ok = False
                        break
                    # The foot has to be the first thing the scan meets on
                    # its bearing, give or take its own thickness.
                    bi = int((fb + half) / width)
                    bi = min(max(bi, 0), bins - 1)
                    if out_scan[k, bi] < fr - 0.20:
                        ok = False
                        break
                if not ok:
                    continue
                dx = gx - cx
                dy = gy - cy
                gr = math.hypot(dx, dy)
                if gr < best_r:
                    best_r = gr
                    gb = math.atan2(dy, dx) - yaw[k]
                    gb = (gb + math.pi) % (2 * math.pi) - math.pi
                    # The gate's normal, turned to point away from the car,
                    # relative to the car's heading.
                    nx, ny = -ay, ax
                    if nx * dx + ny * dy < 0:
                        nx, ny = -nx, -ny
                    na_ = math.atan2(ny, nx) - yaw[k]
                    na_ = (na_ + math.pi) % (2 * math.pi) - math.pi
                    feat0 = 1.0
                    feat1 = gb / half
                    feat2 = gr / max_range
                    feat3 = na_ / (0.5 * math.pi)
            out_gate[k, 4 * kind_i + 0] = feat0
            out_gate[k, 4 * kind_i + 1] = feat1
            out_gate[k, 4 * kind_i + 2] = feat2
            out_gate[k, 4 * kind_i + 3] = min(max(feat3, -2.0), 2.0)


@nb.njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


class Sensor:
    def __init__(self, cfg: dict, model, n: int, rng: np.random.Generator):
        self.sc = SensorConfig(cfg)
        self.cfg = cfg["sensor"]
        self.LAY, self.VIS, self.WASH = model.tables()[2]
        self.SKIP, self.skip_from = sight_skip(model)
        self.gates = gate_table(model)
        self.n = n
        self.rng = rng
        self.scan = np.zeros((n, self.sc.bins))
        self.gate = np.zeros((n, 2 * GATE_FEATURES))
        seed_numba(int(rng.integers(1 << 31)))

    def read(self, lay, state, idx=None):
        """(scan, gate) for every car, or just for cars `idx`."""
        if idx is None:
            s, L, scan, gate = state, lay, self.scan, self.gate
        else:
            s, L = state[idx], lay[idx]
            scan = np.zeros((len(idx), self.sc.bins))
            gate = np.zeros((len(idx), 2 * GATE_FEATURES))
        misread = (self.rng.random(len(s)) < self.cfg["carwash_misread_prob"]).astype(
            np.uint8
        )
        sense(
            self.LAY,
            self.VIS,
            self.WASH,
            np.ascontiguousarray(L),
            s[:, P.S_X].copy(),
            s[:, P.S_Y].copy(),
            s[:, P.S_Z].copy(),
            s[:, P.S_YAW].copy(),
            s[:, P.S_PITCH].copy(),
            s[:, P.S_ROLL].copy(),
            self.sc.vec,
            self.sc.bins,
            self.gates,
            misread,
            self.SKIP,
            self.skip_from,
            scan,
            gate,
        )
        return scan, gate
