#!/usr/bin/env python3
"""The Speed Course as the policy sees it: a centerline, a speed cap, and a
distance field for the bales.

Three things are precomputed here and cached, because all three have to be
IDENTICAL in training and on the car.  A policy trained against one track
object and deployed against another that rounds a corner differently is the
classic silent sim-to-real failure, so there is one builder and one cache file
and both the trainer and the ROS node load it.

  1. CENTERLINE, in the driving direction.  config/speed_course_path.json
     stores the loop in the direction of INCREASING s, which is the opposite
     of the way the car is parked (generate_speed_course.py HEADING = pi, and
     the stored tangent at the start pose points +x).  It is reversed here
     once, so nothing downstream has to remember that.

  2. SPEED CAP.  Two numbers are given by the rules we race to -- 2.5 m/s
     through the hairpins, 5.2 m/s on the straights -- and the cap is built to
     honour them exactly rather than to approximate them with a lateral-accel
     curve.  A section is a hairpin when its radius is at or below
     `hairpin_radius`; everywhere else takes the straight limit, further
     limited by sqrt(a_lat / |k|) so a fast sweeper cannot ask for more grip
     than `lateral_accel` allows.  With the committed course that taper never
     binds (the sweepers sit at R ~ 10 m, worth 6.6 m/s), so the cap is
     effectively the two rule numbers and a clean step between them.

     The step is deliberate.  THE CAR HAS NO BRAKES -- deceleration is coast
     drag only (docs/characterization-results.md), 9.3 m of track to go from
     5.2 to 2.5 m/s -- so the cap cannot be met by reacting to it.  It has to
     be met by lifting off before it, which is the one thing the policy is
     really being trained to do.  Smearing the step out would train that skill
     away.

  3. SIGNED DISTANCE FIELD over the bales.  Frenet half-widths are wrong in
     exactly the places that matter: the hairpins wrap around a free-standing
     island of bales, where the corridor's walls are not at a constant offset
     from the centerline normal.  A grid sampled off the real bale rectangles
     is right everywhere, costs one bilinear lookup per footprint point, and
     is what both the collision test and the graze penalty read.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Bale footprint, from generate_speed_course.py.  Straw bales are 36 x 18 in.
INCH = 0.0254
BALE_LENGTH = 36 * INCH
BALE_WIDTH = 18 * INCH


def load_bales(sdf_path: Path) -> np.ndarray:
    """(N, 3) of bale centre x, y and yaw, read out of the world file.

    The world is the authority, not the DXF: generate_speed_course.py only
    re-derives bales when a drawing is passed, and the drawing revision that
    produced the committed course is not the one in the repo.
    """
    text = Path(sdf_path).read_text()
    poses = re.findall(
        r'<collision name="bale_\d+_collision"><pose>([-\d\.e ]+)</pose>', text
    )
    if not poses:
        raise ValueError(f"no bales found in {sdf_path}")
    rows = np.array([[float(v) for v in p.split()] for p in poses])
    return np.column_stack([rows[:, 0], rows[:, 1], rows[:, 5]])


def _resample_loop(x: np.ndarray, y: np.ndarray, ds: float):
    """Uniform arc-length resample of a closed polyline."""
    seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    length = float(s[-1] + seg[-1])
    su = np.arange(0.0, length, ds)
    return (
        np.interp(su, s, x, period=length),
        np.interp(su, s, y, period=length),
        su,
        length,
    )


def _smooth_periodic(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window) | 1)
    kernel = np.ones(window) / window
    padded = np.r_[values[-window:], values, values[:window]]
    return np.convolve(padded, kernel, "same")[window:-window]


def _close_gaps(mask: np.ndarray, width: int) -> np.ndarray:
    """Fill runs of False shorter than `width`, on a circular mask.

    A single sample of R just over the threshold part way round a hairpin
    would otherwise split one zone into two and put a spike of 5.2 m/s in the
    middle of a corner the car is taking at 2.5.
    """
    out = mask.copy()
    n = len(mask)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    for a, b in zip(idx, np.r_[idx[1:], idx[0] + n]):
        if 0 < b - a - 1 <= width:
            out[np.arange(a + 1, b) % n] = True
    return out


def _dilate(mask: np.ndarray, width: int) -> np.ndarray:
    out = mask.copy()
    n = len(mask)
    for i in np.flatnonzero(mask):
        out[np.arange(i - width, i + width + 1) % n] = True
    return out


@dataclass
class Track:
    """Everything about the course that training and deployment must share."""

    s: np.ndarray  # (N,) arc length, uniform ds, driving direction
    x: np.ndarray  # (N,) centerline
    y: np.ndarray
    tx: np.ndarray  # (N,) unit tangent
    ty: np.ndarray
    kappa: np.ndarray  # (N,) signed curvature, + is left
    v_cap: np.ndarray  # (N,) m/s, the rule limit at this station
    hairpin: np.ndarray  # (N,) bool
    half_left: np.ndarray  # (N,) m to the bale face on the left
    half_right: np.ndarray
    length: float
    ds: float
    start_station: float  # s of the parked car
    sdf: np.ndarray  # (H, W) metres to the nearest bale, + outside
    sdf_origin: tuple  # (x0, y0) of cell (0, 0)
    sdf_res: float

    # ---------------------------------------------------------------- Frenet

    def project(self, x, y, hint):
        """Nearest station to (x, y), searched in a window around `hint`.

        Incremental rather than global on purpose.  The course doubles back on
        itself -- the two straights run 1.4 m apart -- so a global nearest
        point can and does jump to the other side of the wall.  The window is
        wide enough for any real step (4.5 m covers 0.25 s of pose dropout at
        full speed) and narrow enough that it cannot reach the parallel leg.
        """
        n = len(self.s)
        half = int(4.5 / self.ds)
        offsets = np.arange(-half, half + 1)
        idx = (hint[:, None] + offsets[None, :]) % n
        dx = x[:, None] - self.x[idx]
        dy = y[:, None] - self.y[idx]
        best = np.argmin(dx * dx + dy * dy, axis=1)
        rows = np.arange(len(x))
        near = idx[rows, best]

        # Refine to sub-sample precision along the local tangent, so progress
        # is smooth rather than quantised to ds.
        ex, ey = x - self.x[near], y - self.y[near]
        along = ex * self.tx[near] + ey * self.ty[near]
        along = np.clip(along, -self.ds, self.ds)
        lateral = -ex * self.ty[near] + ey * self.tx[near]
        station = (self.s[near] + along) % self.length
        return near, station, lateral

    def locate(self, x, y, yaw):
        """Global re-localisation, disambiguated by heading.

        `project` searches a window around a hint, which is right while the
        car is driving and wrong the instant it is teleported: the hint stays
        where the car used to be, the window never reaches the new position,
        and the station it returns is somewhere the car is not.  Everything
        downstream then steers for that place -- which looks exactly like a
        car that ignores the course and drives into the bales.

        A GLOBAL nearest point is not enough on its own, because this course
        doubles back on itself: the two straights run 1.4 m apart in opposite
        directions, so the nearest station to a car on one of them is often on
        the other.  Gating on heading picks the leg the car is actually
        pointing along.
        """
        dx = x[:, None] - self.x[None, :]
        dy = y[:, None] - self.y[None, :]
        d2 = dx * dx + dy * dy
        ref = np.arctan2(self.ty, self.tx)[None, :]
        agree = np.cos(yaw[:, None] - ref) > 0.0
        # Only fall back to ignoring heading if nothing on the loop agrees,
        # which means the car is pointing backwards and any answer is a guess.
        d2 = np.where(agree | ~agree.any(axis=1, keepdims=True), d2, np.inf)
        idx = np.argmin(d2, axis=1)
        return idx, self.s[idx]

    def heading_error(self, yaw, idx):
        ref = np.arctan2(self.ty[idx], self.tx[idx])
        return np.arctan2(np.sin(yaw - ref), np.cos(yaw - ref))

    def at(self, station, field):
        """Periodic linear sample of a per-station field at arbitrary s."""
        return np.interp(station % self.length, self.s, field, period=self.length)

    def lookahead(self, station, distances, field):
        """(B, K) of `field` sampled `distances` ahead of each station."""
        query = station[:, None] + np.asarray(distances)[None, :]
        return self.at(query.ravel(), field).reshape(query.shape)

    # ------------------------------------------------------------- clearance

    def clearance(self, x, y):
        """Metres from (x, y) to the nearest bale face; negative inside one."""
        gx = (x - self.sdf_origin[0]) / self.sdf_res
        gy = (y - self.sdf_origin[1]) / self.sdf_res
        h, w = self.sdf.shape
        gx = np.clip(gx, 0, w - 1.001)
        gy = np.clip(gy, 0, h - 1.001)
        x0, y0 = gx.astype(np.int32), gy.astype(np.int32)
        fx, fy = gx - x0, gy - y0
        g = self.sdf
        return (
            g[y0, x0] * (1 - fx) * (1 - fy)
            + g[y0, x0 + 1] * fx * (1 - fy)
            + g[y0 + 1, x0] * (1 - fx) * fy
            + g[y0 + 1, x0 + 1] * fx * fy
        )

    def body_clearance(self, x, y, yaw, half_length, half_width):
        """Clearance of the worst point of the car's rectangular footprint.

        Sampled at the four corners plus the two side midpoints.  Six points
        beat four because the corridor is 0.92 m wide against a 0.55 m car:
        the first thing to touch a bale wall the car is running parallel to is
        the middle of its flank, not a corner.
        """
        ox = np.array([1, 1, -1, -1, 0, 0]) * half_length
        oy = np.array([1, -1, 1, -1, 1, -1]) * half_width
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        px = x[:, None] + c * ox[None, :] - s * oy[None, :]
        py = y[:, None] + s * ox[None, :] + c * oy[None, :]
        return self.clearance(px.ravel(), py.ravel()).reshape(px.shape).min(axis=1)


def _build_sdf(bales: np.ndarray, res: float, margin: float):
    """Distance to the nearest bale rectangle, on a grid, positive outside."""
    cx, cy, yaw = bales[:, 0], bales[:, 1], bales[:, 2]
    x0 = float(cx.min() - margin)
    x1 = float(cx.max() + margin)
    y0 = float(cy.min() - margin)
    y1 = float(cy.max() + margin)
    xs = np.arange(x0, x1 + res, res)
    ys = np.arange(y0, y1 + res, res)
    gx, gy = np.meshgrid(xs, ys)
    flat_x, flat_y = gx.ravel(), gy.ravel()
    out = np.empty(flat_x.size, dtype=np.float32)

    cos, sin = np.cos(-yaw), np.sin(-yaw)
    hl, hw = BALE_LENGTH / 2, BALE_WIDTH / 2
    # Chunked so the (cells x bales) intermediate stays cache-sized.
    for start in range(0, flat_x.size, 40000):
        stop = min(start + 40000, flat_x.size)
        dx = flat_x[start:stop, None] - cx[None, :]
        dy = flat_y[start:stop, None] - cy[None, :]
        lx = cos[None, :] * dx - sin[None, :] * dy
        ly = sin[None, :] * dx + cos[None, :] * dy
        qx = np.abs(lx) - hl
        qy = np.abs(ly) - hw
        # Standard box SDF: outside distance, minus the inside depth.
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        out[start:stop] = (outside + inside).min(axis=1)
    return out.reshape(gx.shape), (x0, y0)


def _wall_distance(track_x, track_y, nx, ny, sample, limit=4.0, step=0.02):
    """Distance along a normal to the first bale, by marching the field."""
    out = np.full(len(track_x), limit)
    reached = np.zeros(len(track_x), dtype=bool)
    for dist in np.arange(step, limit, step):
        probe = sample(track_x + nx * dist, track_y + ny * dist)
        hit = (probe <= 0.0) & ~reached
        out[hit] = dist - step
        reached |= hit
        if reached.all():
            break
    return out


def build(config: dict, repo_root: Path, cache_dir: Path | None = None) -> Track:
    """Build (or load from cache) the Track described by `config['track']`."""
    cfg = config["track"]
    sdf_path = repo_root / cfg["world_sdf"]
    path_path = repo_root / cfg["centerline_json"]

    key = hashlib.sha256(
        (
            json.dumps(cfg, sort_keys=True)
            + sdf_path.read_text()[:200000]
            + path_path.read_text()[:200000]
        ).encode()
    ).hexdigest()[:16]
    cache_dir = cache_dir or (Path(__file__).parent / ".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"track_{key}.npz"
    if cache.exists():
        blob = np.load(cache, allow_pickle=False)
        return Track(
            s=blob["s"],
            x=blob["x"],
            y=blob["y"],
            tx=blob["tx"],
            ty=blob["ty"],
            kappa=blob["kappa"],
            v_cap=blob["v_cap"],
            hairpin=blob["hairpin"].astype(bool),
            half_left=blob["half_left"],
            half_right=blob["half_right"],
            length=float(blob["length"]),
            ds=float(blob["ds"]),
            start_station=float(blob["start_station"]),
            sdf=blob["sdf"],
            sdf_origin=tuple(blob["sdf_origin"]),
            sdf_res=float(blob["sdf_res"]),
        )

    raw = json.loads(path_path.read_text())
    # Reverse: the JSON runs against the way the car is parked.
    px = np.asarray(raw["x"], dtype=float)[::-1]
    py = np.asarray(raw["y"], dtype=float)[::-1]
    ds = float(cfg["ds"])
    x, y, s, length = _resample_loop(px, py, ds)

    dx, dy = np.gradient(x, ds), np.gradient(y, ds)
    norm = np.hypot(dx, dy)
    tx, ty = dx / norm, dy / norm
    ddx, ddy = np.gradient(dx, ds), np.gradient(dy, ds)
    kappa = (dx * ddy - dy * ddx) / np.maximum(norm**3, 1e-9)
    # Smoothed over a car length: the policy has to corner a 0.55 m rigid
    # body, not track the curvature of a point.
    kappa = _smooth_periodic(kappa, int(cfg["curvature_smooth_m"] / ds))
    radius = 1.0 / np.maximum(np.abs(kappa), 1e-9)

    hairpin = radius <= float(cfg["hairpin_radius"])
    hairpin = _close_gaps(hairpin, int(cfg["hairpin_gap_close_m"] / ds))
    hairpin = _dilate(hairpin, int(cfg["hairpin_dilate_m"] / ds))

    v_cap = np.minimum(
        float(cfg["v_straight"]), np.sqrt(float(cfg["lateral_accel"]) * radius)
    )
    v_cap[hairpin] = float(cfg["v_hairpin"])

    grid, origin = _build_sdf(
        load_bales(sdf_path), float(cfg["sdf_res"]), float(cfg["sdf_margin"])
    )
    res = float(cfg["sdf_res"])

    def sample(qx, qy):
        gx = np.clip((qx - origin[0]) / res, 0, grid.shape[1] - 1).astype(int)
        gy = np.clip((qy - origin[1]) / res, 0, grid.shape[0] - 1).astype(int)
        return grid[gy, gx]

    half_left = _wall_distance(x, y, -ty, tx, sample)
    half_right = _wall_distance(x, y, ty, -tx, sample)

    start = cfg["vehicle_start"]
    d2 = (x - start[0]) ** 2 + (y - start[1]) ** 2
    start_station = float(s[int(np.argmin(d2))])

    track = Track(
        s=s,
        x=x,
        y=y,
        tx=tx,
        ty=ty,
        kappa=kappa,
        v_cap=v_cap,
        hairpin=hairpin,
        half_left=half_left,
        half_right=half_right,
        length=length,
        ds=ds,
        start_station=start_station,
        sdf=grid.astype(np.float32),
        sdf_origin=origin,
        sdf_res=res,
    )
    np.savez_compressed(
        cache,
        s=s,
        x=x,
        y=y,
        tx=tx,
        ty=ty,
        kappa=kappa,
        v_cap=v_cap,
        hairpin=hairpin,
        half_left=half_left,
        half_right=half_right,
        length=length,
        ds=ds,
        start_station=start_station,
        sdf=track.sdf,
        sdf_origin=np.array(origin),
        sdf_res=res,
    )
    return track


if __name__ == "__main__":
    import yaml

    here = Path(__file__).resolve().parent
    root = here.parents[1]
    cfg = yaml.safe_load((here / "config.yaml").read_text())
    t = build(cfg, root)
    print(f"length        {t.length:.2f} m at ds {t.ds} ({len(t.s)} stations)")
    print(f"min radius    {1 / np.abs(t.kappa).max():.2f} m")
    print(
        f"corridor      {(t.half_left + t.half_right).min():.2f}..."
        f"{(t.half_left + t.half_right).max():.2f} m"
    )
    print(f"hairpin       {100 * t.hairpin.mean():.0f}% of the lap")
    print(
        f"v_cap         2.5 over {100 * (t.v_cap < 2.55).mean():.0f}%, "
        f"5.2 over {100 * (t.v_cap > 5.15).mean():.0f}%"
    )
    print(f"start station {t.start_station:.2f} m")
    print(f"sdf           {t.sdf.shape} at {t.sdf_res} m")
