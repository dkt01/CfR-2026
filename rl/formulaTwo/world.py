"""The course AS BUILT, which is not the course in the map.

Every episode draws a displacement field per car,

    d(p) = sum_k a_k sin(w_k . p + phi_k),

and the world the plant drives in and the camera sees is the nominal world
pushed through it.  The map the policy's prior and lookahead read is never
touched.  So the difference between where the map says a bale is and where it
actually is -- the thing the depth channel exists to measure -- is a known,
per-episode quantity the reward can score against and the critic can be told.

Two bands, because they mean different things:

  layout  wavelength 8-30 m.  The course was laid out a little differently
          from the drawing.  The TRUE centerline moves with it: the middle of
          the corridor as built is where the car should be.
  jitter  wavelength 1.2-3 m, about a bale.  Each bale sits a few cm off its
          neighbour.  The walls go ragged, the centerline does not move.

Plus a uniform `inflate` for bale scale, applied to the distance field.

Everything is evaluated by mapping a true-world point back into the nominal
frame, q = p - d(q), with one fixed-point iteration (the field's gradient is
under 0.3, so the residual is O(|d| |grad d|^2), well under a millimetre), and
then reading the nominal track's precomputed distance field and Frenet frame.
That keeps the per-car world free: no per-episode distance field is built.
"""

from __future__ import annotations

import numpy as np

from randomization import draw


class World:
    def __init__(self, track, config: dict, n: int, rng: np.random.Generator):
        self.track = track
        self.n = n
        self.rng = rng
        w = config["world"]
        self.cfg = w
        self.k_layout = int(w["layout_modes"])
        self.k_jitter = int(w["jitter_modes"])
        k = self.k_layout + self.k_jitter
        self.amp = np.zeros((n, k, 2))
        self.wav = np.zeros((n, k, 2))
        self.phase = np.zeros((n, k))
        self.inflate = np.zeros(n)
        c = config["collision"]
        hl, hw = float(c["chassis_half_length"]), float(c["chassis_half_width"])
        wx, wy = float(c["wheel_x"]), float(c["wheel_outer_y"])
        r = float(c["wheel_radius"])
        # What touches a bale first: the chassis corners, the outer face of
        # each tyre at its fore and aft edge, and the chassis flank midpoints.
        # The tyres stand 12.5 mm proud of the chassis sides.
        pts = [(sx * hl, sy * hw) for sx in (1, -1) for sy in (1, -1)]
        pts += [(ax * wx + e * r, sy * wy) for ax in (1, -1) for e in (1, -1) for sy in (1, -1)]
        pts += [(0.0, hw), (0.0, -hw)]
        self.body = np.array(pts)
        # The coarse grid the fields are baked onto, covering the distance
        # field's extent.
        self._gres = 0.25
        h, w_ = track.sdf.shape
        self._g0 = track.sdf_origin
        self._gw = int(np.ceil(w_ * track.sdf_res / self._gres)) + 2
        self._gh = int(np.ceil(h * track.sdf_res / self._gres)) + 2
        self._gcells = self._gw * self._gh
        xs = self._g0[0] + np.arange(self._gw) * self._gres
        ys = self._g0[1] + np.arange(self._gh) * self._gres
        mx, my = np.meshgrid(xs, ys)
        self._grid_x, self._grid_y = mx.ravel(), my.ravel()
        self.grid = np.zeros((n, 2, self._gcells, 2), dtype=np.float32)
        self._flat = self.grid.reshape(-1, 2)

    # ------------------------------------------------------------- sampling

    def sample(self, mask, scale=1.0, enabled=True):
        k = int(mask.sum())
        if k == 0:
            return
        w = self.cfg
        u = self.rng.uniform

        def modes(n_modes, amp_spec, wl):
            amp = draw(self.rng, amp_spec, k, scale, enabled)[:, None] / n_modes
            direction = u(-np.pi, np.pi, (k, n_modes))
            a = amp[..., None] * np.stack([np.cos(direction), np.sin(direction)], -1)
            lam = u(wl[0], wl[1], (k, n_modes))
            heading = u(-np.pi, np.pi, (k, n_modes))
            wv = (2 * np.pi / lam)[..., None] * np.stack(
                [np.cos(heading), np.sin(heading)], -1
            )
            return a, wv, u(-np.pi, np.pi, (k, n_modes))

        a1, w1, p1 = modes(self.k_layout, w["layout_amp"], w["layout_wavelength"])
        a2, w2, p2 = modes(self.k_jitter, w["jitter_amp"], w["jitter_wavelength"])
        self.amp[mask] = np.concatenate([a1, a2], 1)
        self.wav[mask] = np.concatenate([w1, w2], 1)
        self.phase[mask] = np.concatenate([p1, p2], 1)
        self.inflate[mask] = draw(self.rng, w["inflate"], k, scale, enabled)
        self._bake(np.flatnonzero(mask))

    # ---------------------------------------------------------------- field

    def _analytic(self, px, py, layout_only, rows):
        kk = self.k_layout if layout_only else self.amp.shape[1]
        amp, wav, ph = self.amp[rows, :kk], self.wav[rows, :kk], self.phase[rows, :kk]
        arg = (
            px[..., None] * wav[:, None, :, 0]
            + py[..., None] * wav[:, None, :, 1]
            + ph[:, None, :]
        )
        s = np.sin(arg)
        return (s * amp[:, None, :, 0]).sum(-1), (s * amp[:, None, :, 1]).sum(-1)

    def _bake(self, rows):
        """Tabulate both fields for these cars on the coarse grid.

        The sum of sines is evaluated ONCE per episode here; every query after
        that is a bilinear lookup.  At 0.25 m against a 1.2 m shortest
        wavelength, interpolation error is under 2 mm per mode.
        """
        gx, gy = self._grid_x, self._grid_y
        px = np.broadcast_to(gx[None, :], (len(rows), gx.size))
        py = np.broadcast_to(gy[None, :], (len(rows), gy.size))
        for kind, layout_only in enumerate((False, True)):
            dx, dy = self._analytic(px, py, layout_only, rows)
            self.grid[rows, kind, :, 0] = dx
            self.grid[rows, kind, :, 1] = dy

    def displacement(self, px, py, layout_only=False, rows=None):
        """d at (B, M) points, each row in its own car's world."""
        rows = np.arange(self.n) if rows is None else np.asarray(rows)
        fx = (px - self._g0[0]) / self._gres
        fy = (py - self._g0[1]) / self._gres
        fx = np.clip(fx, 0, self._gw - 1.001)
        fy = np.clip(fy, 0, self._gh - 1.001)
        ix, iy = fx.astype(np.int32), fy.astype(np.int32)
        ax, ay = fx - ix, fy - iy
        base = (rows[:, None] * 2 + int(layout_only)) * self._gcells + iy * self._gw + ix
        g = self._flat
        w00 = (1 - ax) * (1 - ay)
        w10 = ax * (1 - ay)
        w01 = (1 - ax) * ay
        w11 = ax * ay
        v00, v10 = g[base], g[base + 1]
        v01, v11 = g[base + self._gw], g[base + self._gw + 1]
        out = (
            w00[..., None] * v00
            + w10[..., None] * v10
            + w01[..., None] * v01
            + w11[..., None] * v11
        )
        return out[..., 0], out[..., 1]

    def to_nominal(self, px, py, layout_only=False, rows=None, iterations=2):
        qx, qy = px, py
        for _ in range(iterations):
            dx, dy = self.displacement(qx, qy, layout_only, rows)
            qx, qy = px - dx, py - dy
        return qx, qy

    # ------------------------------------------------------------ geometry

    def clearance(self, px, py, rows=None, iterations=2):
        """(B, M) metres to the nearest bale face in each car's own world."""
        qx, qy = self.to_nominal(px, py, rows=rows, iterations=iterations)
        infl = self.inflate if rows is None else self.inflate[rows]
        return self.track.clearance(qx, qy) - infl[:, None]

    def body_clearance(self, x, y, yaw):
        """(B,) clearance of the car's real footprint: chassis and tyres."""
        ox, oy = self.body[:, 0], self.body[:, 1]
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        px = x[:, None] + c * ox - s * oy
        py = y[:, None] + s * ox + c * oy
        return self.clearance(px, py).min(axis=1)

    def frenet(self, x, y, yaw, hint):
        """Station, lateral and heading error against the centerline AS BUILT.

        The as-built centerline is the nominal one pushed through the LAYOUT
        band only.  Its tangent is rotated by the field's local gradient,
        taken by finite difference along the nominal tangent.
        """
        qx, qy = self.to_nominal(x[:, None], y[:, None], layout_only=True)
        qx, qy = qx[:, 0], qy[:, 0]
        idx, station, lateral = self.track.project(qx, qy, hint)
        t = self.track
        tx, ty = t.tx[idx], t.ty[idx]
        eps = 0.25
        d0x, d0y = self.displacement(qx[:, None], qy[:, None], True)
        d1x, d1y = self.displacement(
            (qx + eps * tx)[:, None], (qy + eps * ty)[:, None], True
        )
        bx = tx * eps + (d1x - d0x)[:, 0]
        by = ty * eps + (d1y - d0y)[:, 0]
        ref = np.arctan2(by, bx)
        psi = np.arctan2(np.sin(yaw - ref), np.cos(yaw - ref))
        return idx, station, lateral, psi

    def map_error(self, x, y):
        """(B, 2) where the as-built world has moved the car's surroundings."""
        dx, dy = self.displacement(x[:, None], y[:, None])
        return np.stack([dx[:, 0], dy[:, 0]], 1)

    # ------------------------------------------------------------- raycast

    def raycast(self, ox, oy, angles, max_range, rows=None, iters=48):
        """(B, R) horizontal distance to the first bale along each ray.

        Sphere tracing in the warped distance field, over LIVE rays only
        (most rays end on the corridor wall within a few steps).  The warp
        stretches distances by at most ~30%, so steps are 0.7 of the local
        clearance; a minimum step bounds the cost of rays grazing along a
        wall, and a ray that ends inside a bale is pulled back by its own
        penetration.  Misses return inf.
        """
        rows = np.arange(self.n) if rows is None else np.asarray(rows)
        b, r = angles.shape
        cos, sin = np.cos(angles).ravel(), np.sin(angles).ravel()
        env = np.repeat(rows, r)
        ox_f, oy_f = np.repeat(ox, r), np.repeat(oy, r)
        t = np.full(b * r, 0.02)
        hit = np.zeros(b * r, dtype=bool)
        live = np.arange(b * r)
        for _ in range(iters):
            if live.size == 0:
                break
            tl = t[live]
            px = ox_f[live] + tl * cos[live]
            py = oy_f[live] + tl * sin[live]
            d = self._clearance_flat(px, py, env[live])
            now = d < 0.004
            tl = np.where(now & (d < 0), tl + d, tl)
            tl = np.where(now, tl, tl + np.maximum(0.7 * d, 0.03))
            t[live] = tl
            hit[live[now]] = True
            live = live[~now & (tl < max_range)]
        return np.where(hit, np.maximum(t, 0.0), np.inf).reshape(b, r)

    def _clearance_flat(self, px, py, env):
        """Clearance at N points, point i in car env[i]'s world (1 iteration)."""
        dx, dy = self.displacement(px[:, None], py[:, None], False, env)
        qx, qy = px - dx[:, 0], py - dy[:, 0]
        return self.track.clearance(qx, qy) - self.inflate[env]
