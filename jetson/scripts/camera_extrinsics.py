"""Where the ZED sits on the car, and how to check it from depth, IMU and pose.

numpy + PyYAML only, so the Orin's calibrate_camera.py, the Run Lab and the
formula drivers all import this one file.

Frames (REP-103, as vehicle.yaml):

    vehicle   ground under the wheelbase midpoint, x forward, y left, z up
    mount     the ZED's camera_link: its bottom 1/4" mounting hole.  This is
              the point /zed/zed_node/pose reports
    depth     the left lens (left_camera_frame).  Depth is registered to it,
              so every depth pixel back-projects from here

Attitude is roll, then pitch, then yaw, R = Rz(yaw) Ry(pitch) Rx(roll), with
pitch + nose down and roll + left side up -- the convention the drivers'
depth_to_scan levels with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VEHICLE_YAML = (
    Path(__file__).resolve().parent.parent
    / "cfr_arduino_bridge"
    / "config"
    / "vehicle.yaml"
)
MOUNT_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
LENS_KEYS = ("lens_offset_x", "lens_offset_y", "lens_offset_z")


def rotation(roll, pitch, yaw=0.0):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class Mount:
    """camera_link pose in the vehicle frame, plus the factory lens offset."""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    lens: tuple = (0.0, 0.06, 0.015)
    provenance: str = "guess"
    source: str = ""

    def depth_origin(self):
        """Left-lens position in the vehicle frame."""
        return np.array([self.x, self.y, self.z]) + rotation(
            self.roll, self.pitch, self.yaw
        ) @ np.asarray(self.lens, float)

    def as_dict(self):
        d = self.depth_origin()
        return {
            "mount": {k: round(float(getattr(self, k)), 5) for k in MOUNT_KEYS},
            "lens_offset": [round(float(v), 5) for v in self.lens],
            "depth_origin": [round(float(v), 5) for v in d],
            "provenance": self.provenance,
            "source": self.source,
        }


def load_mount(path=VEHICLE_YAML):
    """vehicle.yaml's camera_mount, or None when the file or section is absent."""
    import yaml

    path = Path(path).expanduser()
    if not path.is_file():
        return None
    section = (yaml.safe_load(path.read_text()) or {}).get("camera_mount")
    if not section:
        return None

    def val(key):
        return float(section[key]["value"])

    tags = {section[k].get("provenance", "guess") for k in MOUNT_KEYS}
    return Mount(
        *(val(k) for k in MOUNT_KEYS),
        lens=tuple(val(k) for k in LENS_KEYS),
        provenance="measured" if tags == {"measured"} else "/".join(sorted(tags)),
        source=str(path),
    )


def for_driver(path, trained_x):
    """What a formula driver needs from the mount, and a line saying so.

    `trained_x` is the config's camera.x, the virtual camera the policy was
    trained with at (camera.x, 0), level, facing forward.  Returns
    (height, pitch, scan_origin, message); height and pitch are None and
    scan_origin is None when there is no mount to read.
    """
    mount = load_mount(path)
    if mount is None:
        return None, None, None, f"no camera_mount in {path}: using the config's camera"
    d = mount.depth_origin()
    origin = (float(d[0] - trained_x), float(d[1]), float(mount.yaw))
    return (
        float(d[2]),
        float(mount.pitch),
        origin,
        (
            f"camera mount ({mount.provenance}, {path}): lens at "
            f"x {d[0]:.3f} y {d[1]:+.3f} z {d[2]:.3f} m, pitch {math.degrees(mount.pitch):+.2f} "
            f"roll {math.degrees(mount.roll):+.2f} yaw {math.degrees(mount.yaw):+.2f} deg; "
            f"scan moved by dx {origin[0]:+.3f} dy {origin[1]:+.3f} m onto the trained "
            f"camera at x {trained_x:.3f}"
        ),
    )


# --------------------------------------------------------------- depth points


def back_project(depth, fx, fy, cx, cy, stride=4, min_range=0.3, max_range=6.0):
    """(H, W) optical-axis depth -> (N, 3) points in the lens frame (x fwd)."""
    v = np.arange(stride // 2, depth.shape[0], stride)
    u = np.arange(stride // 2, depth.shape[1], stride)
    z = depth[np.ix_(v, u)]
    a = (cx - u)[None, :] / fx
    b = (cy - v)[:, None] / fy
    pts = np.stack(np.broadcast_arrays(z, z * a, z * b), axis=-1).reshape(-1, 3)
    keep = np.isfinite(pts[:, 0]) & (pts[:, 0] >= min_range) & (pts[:, 0] <= max_range)
    return pts[keep]


def attitude_from_up(up):
    """(roll, pitch) of a frame whose world-up vector, in that frame, is `up`."""
    n = np.asarray(up, float)
    n = n / np.linalg.norm(n)
    return math.atan2(n[1], n[2]), math.asin(max(-1.0, min(1.0, -n[0])))


def up_from_attitude(roll, pitch):
    return np.array(
        [
            -math.sin(pitch),
            math.cos(pitch) * math.sin(roll),
            math.cos(pitch) * math.cos(roll),
        ]
    )


@dataclass
class Floor:
    height: float  # lens above the plane, m
    roll: float
    pitch: float
    inliers: int
    rms: float  # inlier residual, m

    def as_dict(self):
        return {
            "height_m": round(self.height, 4),
            "roll_deg": round(math.degrees(self.roll), 3),
            "pitch_deg": round(math.degrees(self.pitch), 3),
            "inliers": int(self.inliers),
            "rms_mm": round(self.rms * 1000, 2),
        }


def fit_floor(
    points,
    prior_height,
    prior_roll=0.0,
    prior_pitch=0.0,
    gate=0.10,
    tol=0.015,
    iterations=300,
    seed=0,
):
    """RANSAC + least-squares ground plane under a lens at about `prior_height`.

    The gate keeps candidates within `gate` m of where the prior puts the
    floor, so bale faces and the bale tops cannot win the vote.  Points nearer
    than 0.5 m are dropped too: they are the car's own nose on some mounts.
    """
    pts = np.asarray(points, float)
    up0 = up_from_attitude(prior_roll, prior_pitch)
    near = (pts @ up0 + prior_height < gate) & (pts @ up0 + prior_height > -gate)
    near &= np.hypot(pts[:, 0], pts[:, 1]) > 0.5
    cand = pts[near]
    if len(cand) < 50:
        return None
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    for _ in range(iterations):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n @ up0 < 0:
            n = -n
        if n @ up0 < math.cos(math.radians(15)):
            continue
        count = int((np.abs((cand - p[0]) @ n) < tol).sum())
        if count > best_count:
            best, best_count = (n, p[0]), count
    if best is None:
        return None
    n, p0 = best
    inl = cand[np.abs((cand - p0) @ n) < tol]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vt[2] if vt[2] @ up0 > 0 else -vt[2]
    resid = (inl - centroid) @ n
    roll, pitch = attitude_from_up(n)
    return Floor(
        float(-(centroid @ n)), roll, pitch, len(inl), float(np.sqrt(np.mean(resid**2)))
    )


def level(points, roll, pitch, height):
    """Lens-frame points -> heading-aligned frame with z = height above floor."""
    out = np.asarray(points, float) @ rotation(roll, pitch).T
    out[:, 2] += height
    return out


# ------------------------------------------------------------------- targets


def locate_target(
    level_pts, expect_xy, radius=0.45, z_band=(0.05, 0.60), max_width=1.0
):
    """Front-face center of the object nearest `expect_xy` (lens-relative,
    heading-aligned), for a box set square to the car.

    The face is found within `radius` of the expectation, then taken whole:
    clipping it at the search circle would pull its center toward the prior.
    """
    p = level_pts[(level_pts[:, 2] > z_band[0]) & (level_pts[:, 2] < z_band[1])]
    near = p[np.hypot(p[:, 0] - expect_xy[0], p[:, 1] - expect_xy[1]) < radius]
    if len(near) < 20:
        return None
    # The low percentile alone sits ~1.3 sigma of depth noise in front of the
    # face; the median of the slab around it is the face.
    edge = np.percentile(near[:, 0], 10)
    seed_y = float(np.median(near[np.abs(near[:, 0] - edge) < 0.04, 1]))
    slab = p[(np.abs(p[:, 0] - edge) < 0.04) & (np.abs(p[:, 1] - seed_y) < max_width)]
    lo, hi = np.percentile(slab[:, 1], [1, 99])
    return float(np.median(slab[:, 0])), float((lo + hi) / 2), int(len(slab))


def solve_planar(measured, truth, yaw_prior=0.0):
    """Lens (x, y) and yaw from target pairs: truth = R(yaw) measured + t.

    One target fixes the translation only (yaw held at the prior); two or
    more give the yaw as well.  Returns (tx, ty, yaw, rms residual).
    """
    m = np.asarray(measured, float).reshape(-1, 2)
    t = np.asarray(truth, float).reshape(-1, 2)
    if len(m) >= 2:
        mc, tc = m - m.mean(0), t - t.mean(0)
        yaw = math.atan2(
            (mc[:, 0] * tc[:, 1] - mc[:, 1] * tc[:, 0]).sum(),
            (mc[:, 0] * tc[:, 0] + mc[:, 1] * tc[:, 1]).sum(),
        )
    else:
        yaw = yaw_prior
    c, s = math.cos(yaw), math.sin(yaw)
    rot = m @ np.array([[c, s], [-s, c]])
    tx, ty = (t - rot).mean(0)
    rms = float(np.sqrt(np.mean(np.sum((rot + [tx, ty] - t) ** 2, axis=1))))
    return float(tx), float(ty), yaw, rms


# ---------------------------------------------------------------- lever arm


def fit_lever_arm(t, x, y, yaw, min_speed=0.5, max_lat_accel=3.0, window=0.2):
    """How far ahead of the rear axle the pose point sits, from driving.

    The rear axle center is the one point on a car that does not slide
    sideways (no tire slip, which is why the lateral-accel cap), so a point L
    ahead of it moves sideways at omega * L.  A pose frame yawed by d against
    the car adds -d * v to that.  Least squares on
        v_lateral = L * omega - d * v_forward
    over moving samples gives both.  Needs both turn directions in the data:
    on one-way turns L and d are nearly collinear.  Returns None if the track
    does not pin them.
    """
    t, x, y, yaw = (np.asarray(v, float) for v in (t, x, y, yaw))
    if len(t) < 50:
        return None
    grid = np.arange(t[0], t[-1], 0.02)
    yaw_u = np.unwrap(yaw)
    xs, ys, ws = (np.interp(grid, t, v) for v in (x, y, yaw_u))
    k = max(1, int(round(window / 0.02)))
    if len(grid) <= 2 * k + 10:
        return None
    dt = grid[2 * k :] - grid[: -2 * k]
    vx = (xs[2 * k :] - xs[: -2 * k]) / dt
    vy = (ys[2 * k :] - ys[: -2 * k]) / dt
    om = (ws[2 * k :] - ws[: -2 * k]) / dt
    h = ws[k:-k]
    vf = np.cos(h) * vx + np.sin(h) * vy
    vl = -np.sin(h) * vx + np.cos(h) * vy
    # A sample that the source track bridges with a long gap is not a velocity.
    gap = np.interp(grid[k:-k], t[1:], np.diff(t))
    ok = (vf > min_speed) & (np.abs(vf * om) < max_lat_accel) & (gap < 0.2)
    ok &= np.isfinite(vl) & (np.abs(vl) < 2.0) & (np.abs(om) < 4.0)
    if ok.sum() < 100:
        return None
    a = np.column_stack((om[ok], -vf[ok]))
    b = vl[ok]
    # Huber IRLS: the ZED's in-motion pose jumps would own plain least squares.
    w = np.ones(len(b))
    for _ in range(10):
        sol, *_ = np.linalg.lstsq(a * w[:, None], b * w, rcond=None)
        r = b - a @ sol
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = np.sqrt(np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9)))
    left, right = (om[ok] > 0.2).sum(), (om[ok] < -0.2).sum()
    cov = np.linalg.inv(a.T @ (a * (w**2)[:, None])) * s**2
    return {
        "ahead_of_rear_axle_m": round(float(sol[0]), 4),
        "yaw_deg": round(math.degrees(sol[1]), 3),
        "sigma_m": round(float(math.sqrt(cov[0, 0])), 4),
        "sigma_yaw_deg": round(math.degrees(math.sqrt(cov[1, 1])), 3),
        "samples": int(ok.sum()),
        "left_turn_samples": int(left),
        "right_turn_samples": int(right),
        "residual_mps": round(float(s), 4),
    }
