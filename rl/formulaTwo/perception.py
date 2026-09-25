"""ZED 2i depth -> virtual LiDAR, in one file both sides of sim-to-real import.

The pipeline, in the order it runs on the car:

    1. SAMPLE the depth image on a canonical pinhole grid: every 10th column
       and every 5th row of the 640 x 360 image, at pixel centres -- 64
       azimuths.  `sample_depth`.
    2. BACK-PROJECT each pixel with the known mount (0.20 m up, levelled by
       the ZED's IMU), keep the points 0.07-0.33 m above the ground -- bale
       faces, not ground, not sky -- and take the nearest horizontal range per
       column.  `depth_to_scan`.
    3. ENCODE as log-range in [0, 1], invalid columns as -0.25.  `encode`.
    4. STACK the last 4 frames.  `ScanStack`.

In training, step 1 is replaced by `render` (an exact pinhole render of the
2.5-D course: flat ground, bales of one height) followed by `corrupt` (the
ZED's artifacts), and steps 2-4 are THE SAME CODE.  Anything that differs
between the two sides lives in step 1, where it is modelled explicitly.

Why a slice and not the image: the camera sits at 0.20 m and every bale top
is at 0.356 m, so every ray that leaves the camera near the horizon ends on
the first bale it meets.  A depth image of this course has exactly one number
of information per column.  A CNN over 84 x 84 would learn that the hard way,
on a CPU, at a tenth of the sample rate.
"""

from __future__ import annotations

import numpy as np

INVALID = -0.25
_R0 = 0.25


class Camera:
    """The canonical pinhole grid, and what each pixel's ray looks like."""

    def __init__(self, config: dict):
        c = config["camera"]
        wi, hi = int(c["image_width"]), int(c["image_height"])
        cs, rs = int(c["col_stride"]), int(c["row_stride"])
        self.W = wi // cs
        self.H = hi // rs
        half = np.deg2rad(float(c["hfov_deg"])) / 2
        f = (wi / 2) / np.tan(half)
        # Exactly on pixel centres (pixel k spans [k, k+1)), so the car's
        # sampler reads one real pixel per ray rather than rounding between two.
        u = cs * np.arange(self.W) + cs // 2 + 0.5
        v = rs * np.arange(self.H) + rs // 2 + 0.5
        # Camera frame: x forward, y left, z up.  Column u looks along
        # (1, a_u, b_v), so every pixel of a column shares one azimuth.
        self.a = (wi / 2 - u) / f
        b = (hi / 2 - v) / f
        self.rows = np.flatnonzero(np.abs(b) <= float(c["row_slope_limit"]))
        self.b = b[self.rows]
        self.azimuth = np.arctan(self.a)
        self.x = float(c["x"])
        self.z = float(c["z"])
        self.min_range = float(c["min_range"])
        self.max_range = float(c["max_range"])
        self.band = tuple(float(v) for v in c["band"])
        self.scan_max = float(c["scan_max"])
        self.stack = int(c["stack"])
        self.bale_height = float(config["world"]["bale_height"])
        self.width = self.W

    # ------------------------------------------------------------ geometry

    def _rays(self, pitch):
        """Horizontal length and slope of each pixel's ray, nose-down pitch."""
        p = np.asarray(pitch, dtype=float)[:, None, None]
        a = self.a[None, None, :]
        b = self.b[None, :, None]
        fwd = np.cos(p) + b * np.sin(p)
        up = -np.sin(p) + b * np.cos(p)
        hlen = np.sqrt(fwd * fwd + a * a)
        return hlen, up / hlen, fwd

    def render(self, r_h, pitch, height):
        """(B, rows, W) optical-axis depth Z; NaN where the ray sees sky.

        `r_h` (B, W) is the horizontal distance to the first bale along each
        column's azimuth (inf if none).  The rest is exact for a flat ground
        and bales of one height: a descending ray meets the ground first if
        the ground is nearer; a rising ray that clears the first bale's top
        clears every bale behind it.
        """
        hlen, slope, _ = self._rays(pitch)
        h = np.asarray(height, dtype=float)[:, None, None]
        r = r_h[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            r_ground = np.where(slope < 0, h / -slope, np.inf)
            z_bale = h + r * slope
        on_bale = np.isfinite(r) & (z_bale >= 0) & (z_bale <= self.bale_height)
        dist = np.where(
            r_ground < r, r_ground, np.where(on_bale, r, np.nan)
        )
        dist = np.where(np.isfinite(dist), dist, np.nan)
        return dist / hlen

    # ------------------------------------------------------------ the scan

    def depth_to_scan(self, depth, height=None, pitch=None, roll=None):
        """(B, rows, W) depth -> (B, W) nearest in-band horizontal range.

        `height`, `pitch` and `roll` are what the CAR BELIEVES about its
        camera: height above the ground, and attitude against gravity (ROS
        convention: pitch + is nose down, roll + is left side up).  The points
        are LEVELLED by that attitude before the height band is applied.

        Levelling is not optional on the car.  Measured in Gazebo: through the
        20 m chicane the chassis rolls ~5 deg, and an unlevelled scan lost 75%
        of its beams there -- at the image edges a 5 deg roll moves a bale
        point 2 m away by ~0.25 m, straight out of a 0.26 m band -- which
        tripped the depth watchdog and abandoned the run.  Training believed
        level (0, 0), so a levelled scan is the scan the policy learned on.
        Columns with nothing in the band are inf.
        """
        b = depth.shape[0]
        height = np.full(b, self.z) if height is None else np.asarray(height, float)
        pitch = np.zeros(b) if pitch is None else np.asarray(pitch, float)
        roll = np.zeros(b) if roll is None else np.asarray(roll, float)
        a = self.a[None, None, :]
        bb = self.b[None, :, None]
        # Camera-frame point per unit depth: (1, a, b).  Roll about x, then
        # pitch about y.
        cr, sr = np.cos(roll)[:, None, None], np.sin(roll)[:, None, None]
        cp, sp = np.cos(pitch)[:, None, None], np.sin(pitch)[:, None, None]
        y1 = a * cr - bb * sr
        z1 = a * sr + bb * cr
        x2 = cp + z1 * sp
        z2 = -sp + z1 * cp
        with np.errstate(invalid="ignore"):
            r = depth * np.sqrt(x2 * x2 + y1 * y1)
            z = height[:, None, None] + depth * z2
            keep = np.isfinite(depth) & (z >= self.band[0]) & (z <= self.band[1])
        return np.where(keep, r, np.inf).min(axis=1)

    def encode(self, scan):
        """Log range in [0, 1]; invalid columns are INVALID (-0.25).

        Log, because the ranges that matter are the 0.3-1 m to a wall and the
        5-10 m to the end of a straight, and both need resolution.  Invalid is
        its own value, not "far": a bale closer than the ZED's 0.3 m minimum
        is invalid too, and reading that as open road is how a car drives into
        the bale it is touching.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.log(np.maximum(scan, _R0) / _R0) / np.log(self.scan_max / _R0)
        return np.where(np.isfinite(scan), np.clip(e, 0.0, 1.0), INVALID)

    # ------------------------------------------------------------- the car

    def sample_depth(self, image, fx, fy, cx, cy):
        """ZED depth (H0, W0) with its intrinsics -> this grid, as the car does it.

        Nearest pixel at each canonical ray, no filtering, so the noise the
        policy sees on the car is the noise `corrupt` models: one stereo
        sample per grid cell.  Rays outside a narrower real field of view are
        invalid, exactly as `corrupt` makes them.
        """
        h0, w0 = image.shape
        u = np.round(cx - self.a * fx).astype(int)
        v = np.round(cy - self.b * fy).astype(int)
        ok_u = (u >= 0) & (u < w0)
        ok_v = (v >= 0) & (v < h0)
        out = image[np.clip(v, 0, h0 - 1)[:, None], np.clip(u, 0, w0 - 1)[None, :]]
        out = np.where(ok_v[:, None] & ok_u[None, :], out, np.nan).astype(float)
        out[~np.isfinite(out)] = np.nan
        out[(out < self.min_range) | (out > self.max_range)] = np.nan
        return out[None]


def corrupt(depth, cam, p, rng):
    """The ZED's artifacts, on the rendered grid.  `p` holds per-car arrays.

      range noise   sigma = s0 + s2 Z^2 -- stereo disparity error in depth
      edge bleed    at a depth discontinuity, the far pixel takes a value
                    between the two surfaces (flying pixels / fattening)
      dropout       random pixels with no stereo match
      blobs         rectangles with none (glare, shadow, low texture)
      field of view columns a narrower real lens does not see
      range limits  closer than 0.3 m or past 20 m is no depth at all
    """
    b, h, w = depth.shape
    z = depth + rng.standard_normal(depth.shape) * (
        p["noise_s0"][:, None, None] + p["noise_s2"][:, None, None] * depth**2
    )

    left, right = z[..., :-1], z[..., 1:]
    with np.errstate(invalid="ignore"):
        edge = np.abs(left - right) > 0.3
    bleed = edge & (rng.random(edge.shape) < p["edge_bleed"][:, None, None])
    mix = rng.random(edge.shape)
    right_far = right > left
    new_right = np.where(bleed & right_far, left + mix * (right - left), right)
    new_left = np.where(bleed & ~right_far, right + mix * (left - right), left)
    z = z.copy()
    z[..., 1:] = new_right
    z[..., :-1] = np.where(bleed & ~right_far, new_left, z[..., :-1])

    z[rng.random(z.shape) < p["pixel_dropout"][:, None, None]] = np.nan

    n_blob = np.minimum(rng.poisson(p["blob_rate"]), 3)
    if n_blob.any():
        uu = np.arange(w)[None, None, :]
        vv = np.arange(h)[None, :, None]
        for j in range(3):
            on = n_blob > j
            if not on.any():
                break
            cu = rng.uniform(0, w, b)[:, None, None]
            cv = rng.uniform(0, h, b)[:, None, None]
            hu = rng.uniform(1, 6, b)[:, None, None]
            hv = rng.uniform(1, 5, b)[:, None, None]
            m = on[:, None, None] & (np.abs(uu - cu) < hu) & (np.abs(vv - cv) < hv)
            z[m] = np.nan

    outside = np.abs(cam.azimuth)[None, :] > p["hfov_half"][:, None]
    z[np.broadcast_to(outside[:, None, :], z.shape)] = np.nan
    with np.errstate(invalid="ignore"):
        z[(z < cam.min_range) | (z > cam.max_range)] = np.nan
    return z


class ScanStack:
    """The last `stack` encoded scans, newest first, and how old the newest is.

    Updated only when a camera frame ARRIVES (10-15 Hz), not every control
    tick (20 Hz): between frames the policy sees the same stack with a
    growing age, which is what the car will see too.
    """

    def __init__(self, n, stack, width):
        self.frames = np.zeros((n, stack, width), dtype=np.float32)
        self.age = np.zeros(n)

    def reset(self, mask, encoded):
        self.frames[mask] = encoded[:, None, :]
        self.age[mask] = 0.0

    def push(self, mask, encoded, age):
        f = self.frames[mask]
        f[:, 1:] = f[:, :-1]
        f[:, 0] = encoded
        self.frames[mask] = f
        self.age[mask] = age

    def tick(self, dt):
        self.age += dt

    def features(self):
        """(B, stack * width + 1): the stack, then the newest frame's age."""
        n = self.frames.shape[0]
        return np.concatenate(
            [self.frames.reshape(n, -1), np.clip(self.age / 0.2, 0, 3)[:, None]], 1
        )
