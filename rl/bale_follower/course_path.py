"""Plan a fastest-lap path from the known bale geometry.

The course is static and its exact geometry is already parsed from the SDF,
so unlike the RL policy -- which must rediscover the corridor every episode
through a noisy forward scan -- this module plans once, offline:

1. **Occupancy grid** (5 cm): bale OBBs inflated by half the car width plus
   a safety margin; the free cells are the drivable corridor.
2. **Centerline**: skeletonize the free space and take the longest path
   through the skeleton graph -- the corridor's medial axis from one end of
   the serpentine to the other, immune to the SDF's DXF drawing order.
3. **Racing line** (CasADi): elastic-band NLP over the centerline points --
   minimize curvature and stay smooth, subject to each point staying inside
   a per-point free disc taken from the distance transform. Straightens what
   the corridor allows and rounds the hairpins to their widest arc.
4. **Minimum-time speed profile**: v_limit = sqrt(mu*g / |curvature|), then
   a forward/backward pass with the same accel limit the CasADi command
   smoother enforces, so the plan never asks for grip the tires don't have.

Output is a JSON with (x, y, s, v, curvature) samples every ~10 cm, consumed
by path_racer.py. Regenerate only when the course SDF changes:

    python course_path.py --plot     # writes course_path.json + course_path.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import casadi
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

import bale_geometry

GRAVITY = 9.81
CAR_HALF_WIDTH = 0.15  # 0.30 m wide chassis
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "course_path.json"


def build_occupancy(bales, resolution: float, inflate: float):
    xs = [b.x for b in bales]
    ys = [b.y for b in bales]
    pad = 2.0
    x0, y0 = min(xs) - pad, min(ys) - pad
    nx = int((max(xs) - min(xs) + 2 * pad) / resolution)
    ny = int((max(ys) - min(ys) + 2 * pad) / resolution)
    occupied = np.zeros((ny, nx), dtype=bool)

    for bale in bales:
        hx, hy = bale.half_x + inflate, bale.half_y + inflate
        reach = math.hypot(hx, hy)
        ci0 = max(0, int((bale.x - reach - x0) / resolution))
        ci1 = min(nx, int((bale.x + reach - x0) / resolution) + 1)
        cj0 = max(0, int((bale.y - reach - y0) / resolution))
        cj1 = min(ny, int((bale.y + reach - y0) / resolution) + 1)
        if ci1 <= ci0 or cj1 <= cj0:
            continue
        gx, gy = np.meshgrid(
            x0 + (np.arange(ci0, ci1) + 0.5) * resolution,
            y0 + (np.arange(cj0, cj1) + 0.5) * resolution,
        )
        dx, dy = gx - bale.x, gy - bale.y
        c, s = math.cos(bale.yaw), math.sin(bale.yaw)
        u = dx * c + dy * s
        v = -dx * s + dy * c
        occupied[cj0:cj1, ci0:ci1] |= (np.abs(u) <= hx) & (np.abs(v) <= hy)

    return occupied, (x0, y0, resolution)


def _skeleton_cycle(skeleton: np.ndarray) -> np.ndarray:
    """The closed loop through the skeleton, as ordered (j, i) rows.

    The course is a loop: two corridors joined by hairpins at both ends. Its
    skeleton is that cycle plus dead-end spurs (the decorative spiral tips at
    each hairpin). Iteratively stripping degree-1 pixels removes the spurs
    and leaves only the cycle, which is then ordered by walking it.
    """
    work = skeleton.copy()
    kernel = np.ones((3, 3), dtype=int)
    while True:
        neighbor_count = ndimage.convolve(work.astype(int), kernel, mode="constant") - work
        leaves = work & (neighbor_count <= 1)
        if not leaves.any():
            break
        work &= ~leaves
    if not work.any():
        raise ValueError("skeleton contains no cycle; is the course actually a loop?")

    points = np.argwhere(work)
    index = {tuple(p): k for k, p in enumerate(points)}
    neighbors: list[list[int]] = [[] for _ in points]
    for k, (j, i) in enumerate(points):
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == di == 0:
                    continue
                other = index.get((j + dj, i + di))
                if other is not None:
                    neighbors[k].append(other)

    # Walk the cycle, always preferring an unvisited neighbor. Occasional
    # thick-skeleton pixels give a node three neighbours; skipping the extra
    # one costs a diagonal shortcut of one grid cell, nothing more.
    order = [0]
    visited = {0}
    while True:
        for other in neighbors[order[-1]]:
            if other not in visited:
                order.append(other)
                visited.add(other)
                break
        else:
            break
    return points[order]


def extract_centerline(occupied, origin, seed_xy) -> np.ndarray:
    """Medial axis of the drivable corridor.

    Free space includes the open field outside the walls (whose skeleton is
    longer than the corridor's), so only the connected component containing
    the spawn point is kept -- the corridor is sealed off from the outside.
    """
    x0, y0, res = origin
    free = ~occupied
    labels, _ = ndimage.label(free)
    seed_label = labels[int((seed_xy[1] - y0) / res), int((seed_xy[0] - x0) / res)]
    if seed_label == 0:
        raise ValueError("spawn point is inside an inflated obstacle; lower --margin")
    free = labels == seed_label
    skeleton = skeletonize(free)
    path_cells = _skeleton_cycle(skeleton)
    xy = np.stack(
        [x0 + (path_cells[:, 1] + 0.5) * res, y0 + (path_cells[:, 0] + 0.5) * res], axis=1
    )
    return xy


def resample(xy: np.ndarray, spacing: float) -> np.ndarray:
    """Uniform arc-length resampling of a closed loop (last point joins first)."""
    closed = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.arange(0.0, s[-1], spacing)
    return np.stack(
        [np.interp(samples, s, closed[:, 0]), np.interp(samples, s, closed[:, 1])], axis=1
    )


def enforce_min_radius(xy: np.ndarray, min_radius: float, iterations: int = 60) -> np.ndarray:
    """Smooth away curvature the vehicle physically cannot achieve.

    The planner is free to draw an apex tighter than the car's real minimum
    radius, and did: the first line asked for 1.3 m where the measured
    full-lock radius is 0.97 m -- feasible, but with no margin for tracking
    error, which is where the car wedged. Points exceeding the limit are
    pulled toward their neighbours' midpoint until the whole line is inside
    the vehicle's capability.
    """
    line = xy.copy()
    limit = 1.0 / min_radius
    for _ in range(iterations):
        kappa = np.abs(curvature(line))
        hot = kappa > limit
        if not hot.any():
            break
        midpoints = (np.roll(line, 1, axis=0) + np.roll(line, -1, axis=0)) / 2.0
        blend = np.clip((kappa[hot] - limit) / limit, 0.0, 1.0)[:, None] * 0.5
        line[hot] = line[hot] * (1 - blend) + midpoints[hot] * blend
    return line


def optimize_racing_line(
    centerline: np.ndarray, occupied, origin, safety: float
) -> np.ndarray:
    """Elastic-band smoothing: pull the line straight where clearance allows.

    Each point may move within a disc of radius (clearance - safety) around
    its centerline position -- clearance from the distance transform of the
    already-inflated grid, so the constraint is conservative by construction.
    """
    x0, y0, res = origin
    clearance = ndimage.distance_transform_edt(~occupied) * res
    n = len(centerline)
    radii = np.empty(n)
    for k, (px, py) in enumerate(centerline):
        j = int((py - y0) / res)
        i = int((px - x0) / res)
        radii[k] = max(0.0, clearance[j, i] - safety)

    opti = casadi.Opti()
    pts = opti.variable(n, 2)
    opti.set_initial(pts, centerline)
    # Closed loop: the bend term wraps around, no pinned endpoints. Curvature
    # dominates the cost -- this is a racing line, not a centerline beautifier.
    up = casadi.vertcat(pts[1:, :], pts[:1, :])
    down = casadi.vertcat(pts[-1:, :], pts[:-1, :])
    bend = up - 2 * pts + down
    stay = pts - centerline
    opti.minimize(
        casadi.sumsqr(bend) * 100.0 + casadi.sumsqr(stay) * 0.05
    )
    for k in range(n):
        if radii[k] > 1e-3:
            opti.subject_to(
                casadi.sumsqr(pts[k, :] - centerline[k, :][None, :]) <= radii[k] ** 2
            )
        else:
            opti.subject_to(pts[k, :] == centerline[k, :][None, :])
    opti.solver(
        "ipopt",
        {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
         "ipopt.max_iter": 300},
    )
    try:
        solution = opti.solve()
        return np.array(solution.value(pts))
    except RuntimeError:
        print("racing-line NLP failed; falling back to raw centerline")
        return centerline


def curvature(xy: np.ndarray) -> np.ndarray:
    """Signed curvature of a closed loop (cyclic central differences)."""
    d = (np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)) / 2.0
    dd = np.roll(xy, -1, axis=0) - 2 * xy + np.roll(xy, 1, axis=0)
    num = d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]
    den = (d[:, 0] ** 2 + d[:, 1] ** 2) ** 1.5
    return np.where(den > 1e-9, num / den, 0.0)


def speed_profile(xy: np.ndarray, traction: float, max_speed: float,
                  speed_margin: float = 0.85, turn_speed: float = 0.0,
                  turn_curvature: float = 0.15) -> np.ndarray:
    """Minimum-time flying-lap profile for a closed loop, on a traction ellipse.

    Three limits, applied in order:

      1. cornering -- v <= sqrt(a_plan / kappa)
      2. corner exit -- cyclic forward pass, accelerating on whatever grip the
         corner is not already using
      3. corner entry -- cyclic backward pass, braking under the same
         combined limit

    Passes run twice so the limits propagate across the wrap point. The
    standing start is not in the profile; the tracker ramps up to it.

    The combined (2, 3) step is the point of this function. Treating the
    lateral and longitudinal limits as independent -- accelerating at full
    a_max while already at the cornering limit -- asks for more grip than
    exists, and the tracker cannot deliver it: measured corner speeds came in
    at 0.67 of such a plan. The ellipse plans only what the tires can actually
    produce, so a corner exit is a speed the car can really carry.

    `speed_margin` scales the whole grip budget down. It is the tracking
    allowance: the corridor is 0.95 m wide and cross-track error grows with
    speed, so planning at the true limit leaves nothing for error and the car
    rubs the bales. This replaces an older flat cap that clamped every corner
    to one speed regardless of radius, which made 6.7 m sweepers crawl at the
    same pace as 1.3 m hairpins.
    """
    a_max = traction * GRAVITY
    a_plan = max(speed_margin, 1e-3) * a_max
    kappa = np.abs(curvature(xy))
    # Smooth curvature a little: single-sample kinks from the grid otherwise
    # punch unnecessary dips into the profile.
    kernel = np.ones(5) / 5.0
    kappa = np.convolve(np.concatenate([kappa[-2:], kappa, kappa[:2]]), kernel, mode="same")[2:-2]
    v = np.minimum(np.sqrt(a_plan / np.maximum(kappa, 1e-6)), max_speed)
    # Flat cap, retained as an option. It is cruder than the ellipse -- one
    # speed for every corner regardless of radius -- but it is the only
    # configuration with a verified single-racer measurement behind it
    # (31.70 s best, 6 laps, 0 stuck). The ellipse is better physics and
    # plans a 27.3 s lap, but measured 60-81 s laps when finally timed
    # against a single racer, so it is not yet the default anyone should
    # trust. Keep both until the ellipse has a clean measurement.
    if turn_speed > 0:
        v = np.where(kappa > turn_curvature, np.minimum(v, turn_speed), v)
    n = len(v)
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)

    def longitudinal(speed: float, k_index: int) -> float:
        """Grip left over for accelerating or braking at this speed/curvature."""
        lateral = speed * speed * kappa[k_index]
        spare = 1.0 - min(lateral / a_plan, 1.0) ** 2
        return a_plan * math.sqrt(max(spare, 0.0))

    for _ in range(2):
        for k in range(n):  # forward: corner exit
            nxt = (k + 1) % n
            a_long = longitudinal(v[k], k)
            v[nxt] = min(v[nxt], math.sqrt(v[k] ** 2 + 2 * a_long * ds[k]))
        for k in range(n - 1, -1, -1):  # backward: corner entry
            nxt = (k + 1) % n
            a_long = longitudinal(v[nxt], nxt)
            v[k] = min(v[k], math.sqrt(v[nxt] ** 2 + 2 * a_long * ds[k]))
    return v


def plan(sdf_path: str, resolution: float, spacing: float, margin: float,
         safety: float, traction: float, max_speed: float,
         min_radius: float = 0.0, speed_margin: float = 0.85,
         turn_speed: float = 0.0, turn_curvature: float = 0.15) -> dict:
    bales = bale_geometry.parse_bales(sdf_path)
    spawn = bale_geometry.parse_vehicle_spawn(sdf_path)
    occupied, origin = build_occupancy(bales, resolution, CAR_HALF_WIDTH + margin)
    centerline = resample(extract_centerline(occupied, origin, spawn), spacing)
    line = optimize_racing_line(centerline, occupied, origin, safety)
    if min_radius > 0:
        line = enforce_min_radius(line, min_radius)
    line = resample(line, spacing)
    v = speed_profile(line, traction, max_speed, speed_margin,
                      turn_speed, turn_curvature)
    ds = np.linalg.norm(np.roll(line, -1, axis=0) - line, axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds[:-1])])
    v_mid = np.maximum((v + np.roll(v, -1)) / 2.0, 0.05)
    lap_time = float(np.sum(ds / v_mid))
    return {
        "sdf": str(sdf_path),
        "closed": True,
        "traction": traction,
        "speed_margin": speed_margin,
        "max_speed": max_speed,
        "length_m": float(s[-1]),
        "estimated_time_s": lap_time,
        "x": line[:, 0].round(4).tolist(),
        "y": line[:, 1].round(4).tolist(),
        "s": s.round(4).tolist(),
        "v": v.round(3).tolist(),
        "curvature": curvature(line).round(4).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--spacing", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.05,
                        help="extra inflation beyond the car half-width (m)")
    parser.add_argument("--safety", type=float, default=0.05,
                        help="clearance the racing line must keep beyond the inflated grid (m)")
    parser.add_argument("--traction", type=float, default=0.6)
    parser.add_argument("--max-speed", type=float, default=4.0)
    parser.add_argument("--min-radius", type=float, default=1.25,
                        help="tightest radius the line may ask for (m). The vehicle "
                             "measures 0.97 m at full lock (vehicle_calibration.py), so "
                             "this leaves margin for tracking error rather than planning "
                             "apexes the car can only just make.")
    parser.add_argument("--speed-margin", type=float, default=0.85,
                        help="fraction of the grip budget the profile plans for "
                             "(0-1). The remainder is the allowance for tracking "
                             "error: cross-track grows with speed and the corridor "
                             "is only 0.95 m wide, so planning at 1.0 rubs the bales.")
    parser.add_argument("--turn-speed", type=float, default=0.0,
                        help="flat speed cap (m/s) wherever curvature exceeds "
                             "--turn-curvature; 0 uses the ellipse profile alone. "
                             "Cruder than --speed-margin, but the only setting with "
                             "a verified single-racer lap time behind it.")
    parser.add_argument("--turn-curvature", type=float, default=0.15,
                        help="curvature (1/m) above which --turn-speed applies "
                             "(0.15 = radius 6.7 m)")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    result = plan(args.sdf_path, args.resolution, args.spacing, args.margin,
                  args.safety, args.traction, args.max_speed, args.min_radius,
                  args.speed_margin, args.turn_speed, args.turn_curvature)
    with open(args.output, "w") as handle:
        json.dump(result, handle)
    v = np.array(result["v"])
    print(f"path: {result['length_m']:.1f} m, estimated lap {result['estimated_time_s']:.1f} s "
          f"(traction {args.traction}, margin {args.speed_margin}, max {args.max_speed} m/s)")
    print(f"speed: min {v.min():.2f}  mean {v.mean():.2f}  max {v.max():.2f} m/s")
    print(f"wrote {args.output}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bales = bale_geometry.parse_bales(args.sdf_path)
        fig, ax = plt.subplots(figsize=(14, 6))
        for bale in bales:
            corners = bale_geometry.obb_corners(bale.x, bale.y, bale.yaw, bale.half_x, bale.half_y)
            ax.fill(corners[:, 0], corners[:, 1], color="peru", alpha=0.8)
        points = ax.scatter(result["x"], result["y"], c=result["v"], s=4, cmap="viridis")
        fig.colorbar(points, ax=ax, label="planned speed (m/s)")
        spawn = bale_geometry.parse_vehicle_spawn(args.sdf_path)
        ax.plot(spawn[0], spawn[1], "r*", markersize=14, label="spawn")
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"{result['length_m']:.1f} m racing line, estimated {result['estimated_time_s']:.1f} s")
        png = Path(args.output).with_suffix(".png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
