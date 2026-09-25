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
        r += step
    return max_range


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
                    z[k],
                    yaw[k],
                    bearing,
                    el_c,
                    vfov,
                    max_range,
                    min_range,
                    step,
                    wash_from,
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
        self.gates = gate_table(model)
        self.n = n
        self.rng = rng
        self.scan = np.zeros((n, self.sc.bins))
        self.gate = np.zeros((n, 2 * GATE_FEATURES))
        seed_numba(int(rng.integers(1 << 31)))

    def read(self, lay, state, idx=None):
        """(scan, gate) for every car, or just for cars `idx`."""
        import plant as P

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
            scan,
            gate,
        )
        return scan, gate
