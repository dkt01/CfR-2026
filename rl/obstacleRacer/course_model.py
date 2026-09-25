#!/usr/bin/env python3
"""The Obstacle Course baked into plan grids the numpy trainer can query fast.

Two geometry sources, because Gazebo has two (see world.py):

  PHYSICS, from the collision primitives, exactly -- every oriented box is cut
  by the vertical line through each cell analytically, so the 24 tilted helix
  segments, the 19% ramp, the 8.5 degree bank, the wedges and the pothole
  board's 18 mm recesses come out as the heights Gazebo's wheels meet.

    support   S_lo/S_hi/S_mu  merged vertical intervals of drivable solid
    obstacle  O_lo/O_hi       merged vertical intervals the chassis must miss

  SENSOR, from the visuals, triangle by triangle -- the hoop's top bar, the
  tunnel roof, the car wash's ribbons and arches and the bump domes are all
  here, and none of them are in the physics.  Per cell:

    E_top/E_n  EXPOSED layers: drivable tops with head room over them (the
               ground under a ramp is not one, the floor under the deck is)
    V_*        vertical intervals of everything the camera renders
    C_*        the same for the car wash's ribbons, which the segmenter
               relabels and lets the car through

  Whether a column blocks is decided at query time, against the surface the
  sensor's ray is following out from under the car (visual_band), because
  that is what the segmenter judges height against: a guard rail at the edge
  of the deck blocks from the deck even though the only layer under it is
  the floor 0.6 m down.

The movable models change per layout, so the grid is baked twice: once for
everything static over the whole course, and once per layout over a WINDOW
covering the bucket section, the hoop corridor and the gap-bale wall, with
the layout's buckets, hoops and bale in it.  Queries read the window when the
point is inside it.

    python3 course_model.py --check    # centerline checks, and PNGs in .cache/
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np

import layouts
import world as world_module

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"

RES = 0.02
KS = 4  # support intervals per cell
KO = 3  # obstacle intervals per cell
KE = 2  # exposed layers per cell
KV = 4  # visual intervals per cell (solid)
KC = 2  # visual intervals per cell (car-wash ribbons)
MARGIN = 0.8

# The segmenter's height rule, cloud_segmentation.hpp.
GROUND_TOLERANCE = 0.05
CLEARANCE = 0.32
# A support top is a layer the car can be on only if this much head room
# stands over it -- the car with its camera is 0.23 m tall.
LAYER_HEADROOM = 0.25
# Tops closer than this are one layer (the pothole board's plies, a pebble on
# the gravel lid).
LAYER_MERGE = 0.08

# Everything the randomizer can move lies inside this, with room for the
# models' own extent (layout yaml bounds, plus the 0.709 m hoop base and a
# bale's length).
WINDOW = (-10.2, -5.4, 3.8, 1.0)  # x0, y0, x1, y1

VIS_NONE, VIS_SOLID, VIS_CARWASH = 0, 1, 2


@dataclass
class Grid:
    x0: float
    y0: float
    nx: int
    ny: int

    def cell_center(self, i, j):
        return self.x0 + (i + 0.5) * RES, self.y0 + (j + 0.5) * RES


# ------------------------------------------------------------------ baking


def _box_vertical_interval(matrix, size, xs, ys):
    """z-interval where the vertical line through each (x, y) is inside the box.

    Exact for any orientation: in the box frame the line is a + t b, and each
    axis bounds t to a slab.  Returns (lo, hi) with lo > hi where it misses.
    """
    rotation = matrix[:3, :3]
    center = matrix[:3, 3]
    half = np.asarray(size) / 2.0
    d = np.stack([xs - center[0], ys - center[1], -np.full_like(xs, center[2])], -1)
    a = d @ rotation  # R^T d, per point
    b = rotation[2, :]  # R^T e_z
    lo = np.full(xs.shape, -np.inf)
    hi = np.full(xs.shape, np.inf)
    for k in range(3):
        if abs(b[k]) < 1e-12:
            outside = np.abs(a[..., k]) > half[k]
            lo = np.where(outside, np.inf, lo)
            hi = np.where(outside, -np.inf, hi)
            continue
        t1 = (-half[k] - a[..., k]) / b[k]
        t2 = (half[k] - a[..., k]) / b[k]
        lo = np.maximum(lo, np.minimum(t1, t2))
        hi = np.minimum(hi, np.maximum(t1, t2))
    return lo, hi


def _cylinder_vertical_interval(matrix, size, xs, ys):
    radius, length = size
    axis = matrix[:3, 2]
    if abs(axis[2]) < 1 - 1e-6:
        raise ValueError("only upright cylinders are modeled")
    cx, cy, cz = matrix[:3, 3]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    lo = np.where(inside, cz - length / 2, np.inf)
    hi = np.where(inside, cz + length / 2, -np.inf)
    return lo, hi


def _footprint_xy(prim):
    matrix = prim.world
    if prim.kind == "box":
        hx, hy, hz = (s / 2 for s in prim.size)
        corners = np.array(
            [
                [sx * hx, sy * hy, sz * hz, 1]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )
    else:
        r, length = prim.size
        corners = np.array(
            [
                [sx * r, sy * r, sz * length / 2, 1]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )
    world_pts = corners @ matrix.T
    return (
        world_pts[:, 0].min(),
        world_pts[:, 1].min(),
        world_pts[:, 0].max(),
        world_pts[:, 1].max(),
    )


@nb.njit(cache=True)
def _insert(lo_arr, hi_arr, mu_arr, n_arr, i, j, lo, hi, mu, merge_gap):
    """Add [lo, hi] to cell (i, j)'s interval set, merging overlaps."""
    k = np.int64(n_arr[j, i])
    cap = np.int64(lo_arr.shape[2])
    # Merge with every interval it touches.
    a = np.int64(0)
    while a < k:
        if lo <= hi_arr[j, i, a] + merge_gap and hi >= lo_arr[j, i, a] - merge_gap:
            lo = min(lo, lo_arr[j, i, a])
            hi = max(hi, hi_arr[j, i, a])
            # Lower of the two frictions: the tire meets the worse surface.
            mu = min(mu, mu_arr[j, i, a])
            k -= 1
            lo_arr[j, i, a] = lo_arr[j, i, k]
            hi_arr[j, i, a] = hi_arr[j, i, k]
            mu_arr[j, i, a] = mu_arr[j, i, k]
            continue
        a += 1
    if k == cap:
        # Full: fold into the interval whose gap to this one is smallest.
        best = np.int64(0)
        best_gap = np.float32(1e9)
        for b in range(k):
            gap = max(lo - hi_arr[j, i, b], lo_arr[j, i, b] - hi)
            if gap < best_gap:
                best_gap = gap
                best = b
        lo_arr[j, i, best] = min(lo, lo_arr[j, i, best])
        hi_arr[j, i, best] = max(hi, hi_arr[j, i, best])
        mu_arr[j, i, best] = min(mu, mu_arr[j, i, best])
    else:
        lo_arr[j, i, k] = lo
        hi_arr[j, i, k] = hi
        mu_arr[j, i, k] = mu
        k += 1
    n_arr[j, i] = k


@nb.njit(cache=True)
def _insert_many(lo_arr, hi_arr, mu_arr, n_arr, ii, jj, los, his, mu, merge_gap):
    for q in range(ii.shape[0]):
        _insert(
            lo_arr, hi_arr, mu_arr, n_arr, ii[q], jj[q], los[q], his[q], mu, merge_gap
        )


@nb.njit(cache=True)
def _sort_cells(lo_arr, hi_arr, mu_arr, n_arr):
    """Order each cell's intervals by top, lowest first."""
    ny, nx, _ = lo_arr.shape
    for j in range(ny):
        for i in range(nx):
            k = int(n_arr[j, i])
            for a in range(1, k):
                b = a
                while b > 0 and hi_arr[j, i, b - 1] > hi_arr[j, i, b]:
                    for arr in (lo_arr, hi_arr, mu_arr):
                        t = arr[j, i, b - 1]
                        arr[j, i, b - 1] = arr[j, i, b]
                        arr[j, i, b] = t
                    b -= 1


@nb.njit(cache=True)
def _clip_triangle_z(tri, x0, y0, x1, y1):
    """z-range of the part of a 3D triangle over [x0,x1]x[y0,y1], or (1,-1)."""
    poly = np.empty((12, 3))
    npoly = 3
    for v in range(3):
        for c in range(3):
            poly[v, c] = tri[v, c]
    out = np.empty((12, 3))
    for plane in range(4):
        axis = 0 if plane < 2 else 1
        bound = (x0, x1, y0, y1)[plane]
        keep_ge = plane == 0 or plane == 2
        nout = 0
        for v in range(npoly):
            a = poly[v]
            b = poly[(v + 1) % npoly]
            da = a[axis] - bound if keep_ge else bound - a[axis]
            db = b[axis] - bound if keep_ge else bound - b[axis]
            if da >= 0:
                out[nout] = a
                nout += 1
            if (da >= 0) != (db >= 0):
                t = da / (da - db)
                out[nout] = a + t * (b - a)
                nout += 1
        npoly = nout
        if npoly == 0:
            return 1.0, -1.0
        for v in range(npoly):
            poly[v] = out[v]
    zlo = 1e9
    zhi = -1e9
    for v in range(npoly):
        zlo = min(zlo, poly[v, 2])
        zhi = max(zhi, poly[v, 2])
    return zlo, zhi


@nb.njit(cache=True)
def _raster_triangles(tris, cls, x0, y0, nx, ny, res, i_off, j_off):
    """Every (cell, zlo, zhi, class) a set of triangles occupies."""
    cap = 1 << 16
    cells = np.empty(cap, np.int64)
    zlos = np.empty(cap)
    zhis = np.empty(cap)
    clss = np.empty(cap, np.uint8)
    n = 0
    for t in range(tris.shape[0]):
        tri = tris[t]
        tx0 = min(tri[0, 0], tri[1, 0], tri[2, 0])
        tx1 = max(tri[0, 0], tri[1, 0], tri[2, 0])
        ty0 = min(tri[0, 1], tri[1, 1], tri[2, 1])
        ty1 = max(tri[0, 1], tri[1, 1], tri[2, 1])
        i0 = max(int(math.floor((tx0 - x0) / res)), i_off)
        i1 = min(int(math.floor((tx1 - x0) / res)), i_off + nx - 1)
        j0 = max(int(math.floor((ty0 - y0) / res)), j_off)
        j1 = min(int(math.floor((ty1 - y0) / res)), j_off + ny - 1)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                cx0 = x0 + i * res
                cy0 = y0 + j * res
                zlo, zhi = _clip_triangle_z(tri, cx0, cy0, cx0 + res, cy0 + res)
                if zlo > zhi:
                    continue
                if n == cells.shape[0]:
                    cells = np.concatenate((cells, np.empty(cells.shape[0], np.int64)))
                    zlos = np.concatenate((zlos, np.empty(zlos.shape[0])))
                    zhis = np.concatenate((zhis, np.empty(zhis.shape[0])))
                    clss = np.concatenate((clss, np.empty(clss.shape[0], np.uint8)))
                cells[n] = (j - j_off) * nx + (i - i_off)
                zlos[n] = zlo
                zhis[n] = zhi
                clss[n] = cls[t]
                n += 1
    return cells[:n], zlos[:n], zhis[:n], clss[:n]


@nb.njit(cache=True)
def _exposed_layers(S_lo, S_hi, S_n, E_top, E_n):
    """Support tops with head room over them: the levels a car can be on."""
    ny, nx, _ = S_lo.shape
    for j in range(ny):
        for i in range(nx):
            k = int(S_n[j, i])
            n = 0
            last = -1e9
            for m in range(k):
                top = S_hi[j, i, m]
                above = S_lo[j, i, m + 1] if m + 1 < k else 1e9
                if above - top < LAYER_HEADROOM:
                    continue
                if n > 0 and top - last < LAYER_MERGE:
                    E_top[j, i, n - 1] = top
                    last = top
                    continue
                if n < E_top.shape[2]:
                    E_top[j, i, n] = top
                    n += 1
                    last = top
            E_n[j, i] = n


@nb.njit(cache=True)
def _visual_intervals(
    cells, zlos, zhis, clss, nx, V_lo, V_hi, V_mu, V_n, C_lo, C_hi, C_mu, C_n
):
    """Merge rasterized triangles into vertical intervals, per class, per cell."""
    for q in range(cells.shape[0]):
        c = cells[q]
        j = c // nx
        i = c % nx
        if clss[q] == VIS_CARWASH:
            _insert(
                C_lo,
                C_hi,
                C_mu,
                C_n,
                i,
                j,
                np.float32(zlos[q]),
                np.float32(zhis[q]),
                np.float32(0.0),
                np.float32(0.02),
            )
        else:
            _insert(
                V_lo,
                V_hi,
                V_mu,
                V_n,
                i,
                j,
                np.float32(zlos[q]),
                np.float32(zhis[q]),
                np.float32(0.0),
                np.float32(0.02),
            )


def bake(prims, grid: Grid, i_off: int, j_off: int, nx: int, ny: int) -> dict:
    """Rasterize `prims` over the cell block [i_off, i_off+nx) x [j_off, j_off+ny)."""
    S_lo = np.zeros((ny, nx, KS), np.float32)
    S_hi = np.zeros((ny, nx, KS), np.float32)
    S_mu = np.ones((ny, nx, KS), np.float32)
    S_n = np.zeros((ny, nx), np.uint8)
    O_lo = np.zeros((ny, nx, KO), np.float32)
    O_hi = np.zeros((ny, nx, KO), np.float32)
    O_mu = np.ones((ny, nx, KO), np.float32)
    O_n = np.zeros((ny, nx), np.uint8)
    # The ground plane, under everything.
    S_lo[:, :, 0] = -0.1
    S_hi[:, :, 0] = 0.0
    S_mu[:, :, 0] = world_module.TIRE_MU
    S_n[:] = 1

    bx0 = grid.x0 + i_off * RES
    by0 = grid.y0 + j_off * RES
    sub = np.array([-0.45, 0.0, 0.45]) * RES
    triangles, classes = [], []
    for prim in prims:
        if prim.source == "visual":
            tris = world_module.world_triangles(prim)
            if len(tris):
                triangles.append(tris)
                classes.append(
                    np.full(
                        len(tris),
                        VIS_CARWASH if prim.vclass == "carwash" else VIS_SOLID,
                        np.uint8,
                    )
                )
            continue
        if prim.kind == "plane":
            continue
        fx0, fy0, fx1, fy1 = _footprint_xy(prim)
        i0 = max(int(math.floor((fx0 - bx0) / RES)) - 1, 0)
        i1 = min(int(math.floor((fx1 - bx0) / RES)) + 1, nx - 1)
        j0 = max(int(math.floor((fy0 - by0) / RES)) - 1, 0)
        j1 = min(int(math.floor((fy1 - by0) / RES)) + 1, ny - 1)
        if i0 > i1 or j0 > j1:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
        cx = bx0 + (ii + 0.5) * RES
        cy = by0 + (jj + 0.5) * RES
        interval = (
            _box_vertical_interval
            if prim.kind == "box"
            else _cylinder_vertical_interval
        )
        if prim.role == "support":
            # Road: sampled at the cell center, so heights are exact there.
            lo, hi = interval(prim.world, prim.size, cx, cy)
            target = (S_lo, S_hi, S_mu, S_n)
        else:
            # Obstacles: the union over a 3x3 sub-sample, so a 34 mm post is
            # never lost between cell centers.
            lo = np.full(cx.shape, np.inf)
            hi = np.full(cx.shape, -np.inf)
            for dx in sub:
                for dy in sub:
                    l2, h2 = interval(prim.world, prim.size, cx + dx, cy + dy)
                    hit = l2 <= h2
                    lo = np.where(hit, np.minimum(lo, l2), lo)
                    hi = np.where(hit, np.maximum(hi, h2), hi)
            target = (O_lo, O_hi, O_mu, O_n)
        hit = lo <= hi
        if not hit.any():
            continue
        _insert_many(
            *target,
            ii[hit].astype(np.int64),
            jj[hit].astype(np.int64),
            lo[hit].astype(np.float32),
            hi[hit].astype(np.float32),
            np.float32(prim.mu),
            np.float32(0.002),
        )
    _sort_cells(S_lo, S_hi, S_mu, S_n)
    _sort_cells(O_lo, O_hi, O_mu, O_n)

    E_top = np.zeros((ny, nx, KE), np.float32)
    E_n = np.zeros((ny, nx), np.uint8)
    _exposed_layers(S_lo, S_hi, S_n, E_top, E_n)
    V_lo = np.zeros((ny, nx, KV), np.float32)
    V_hi = np.zeros((ny, nx, KV), np.float32)
    V_mu = np.zeros((ny, nx, KV), np.float32)
    V_n = np.zeros((ny, nx), np.uint8)
    C_lo = np.zeros((ny, nx, KC), np.float32)
    C_hi = np.zeros((ny, nx, KC), np.float32)
    C_mu = np.zeros((ny, nx, KC), np.float32)
    C_n = np.zeros((ny, nx), np.uint8)
    if triangles:
        tris = np.concatenate(triangles)
        cls = np.concatenate(classes)
        cells, zlos, zhis, clss = _raster_triangles(
            tris, cls, grid.x0, grid.y0, nx, ny, RES, i_off, j_off
        )
        _visual_intervals(
            cells, zlos, zhis, clss, nx, V_lo, V_hi, V_mu, V_n, C_lo, C_hi, C_mu, C_n
        )
    return dict(
        S_lo=S_lo,
        S_hi=S_hi,
        S_mu=S_mu,
        S_n=S_n,
        O_lo=O_lo,
        O_hi=O_hi,
        O_n=O_n,
        E_top=E_top,
        E_n=E_n,
        V_lo=V_lo,
        V_hi=V_hi,
        V_n=V_n,
        C_lo=C_lo,
        C_hi=C_hi,
        C_n=C_n,
    )


FIELDS = (
    "S_lo",
    "S_hi",
    "S_mu",
    "S_n",
    "O_lo",
    "O_hi",
    "O_n",
    "E_top",
    "E_n",
    "V_lo",
    "V_hi",
    "V_n",
    "C_lo",
    "C_hi",
    "C_n",
)


class CourseModel:
    """Static grid plus one window grid per layout, and the queries over them."""

    def __init__(self, seeds, rebuild: bool = False):
        spec = layouts.layout_spec()
        base = world_module.parse()
        static = base.select(dynamic=False)
        xs, ys = [], []
        for prim in static:
            if prim.source == "collision" and prim.kind != "plane":
                fx0, fy0, fx1, fy1 = _footprint_xy(prim)
                xs += [fx0, fx1]
                ys += [fy0, fy1]
        x0 = math.floor((min(xs) - MARGIN) / RES) * RES
        y0 = math.floor((min(ys) - MARGIN) / RES) * RES
        nx = int(math.ceil((max(xs) + MARGIN - x0) / RES))
        ny = int(math.ceil((max(ys) + MARGIN - y0) / RES))
        self.grid = Grid(x0, y0, nx, ny)
        wx0, wy0, wx1, wy1 = WINDOW
        self.wi = int(math.floor((wx0 - x0) / RES))
        self.wj = int(math.floor((wy0 - y0) / RES))
        self.wnx = int(math.ceil((wx1 - wx0) / RES))
        self.wny = int(math.ceil((wy1 - wy0) / RES))
        self.seeds = list(seeds)
        self.layouts = [layouts.load(s) for s in self.seeds]

        CACHE.mkdir(exist_ok=True)
        key = self._cache_key()
        path = CACHE / f"static_{key}.npz"
        if path.exists() and not rebuild:
            self.static = dict(np.load(path))
        else:
            self.static = bake(static, self.grid, 0, 0, nx, ny)
            np.savez_compressed(path, **self.static)
        windows = []
        for seed, layout in zip(self.seeds, self.layouts):
            wpath = CACHE / f"window_{key}_{seed}.npz"
            if wpath.exists() and not rebuild:
                windows.append(dict(np.load(wpath)))
                continue
            laid = world_module.apply_layout(base, layout, spec)
            inside = [p for p in laid.prims if self._near_window(p)]
            win = bake(inside, self.grid, self.wi, self.wj, self.wnx, self.wny)
            np.savez_compressed(wpath, **win)
            windows.append(win)
        self.window = {f: np.stack([w[f] for w in windows]) for f in FIELDS}
        self.spec = spec

    def _cache_key(self) -> str:
        h = hashlib.sha1()
        for path in (
            world_module.WORLD_SDF,
            Path(world_module.__file__),
            Path(__file__),
        ):
            h.update(path.read_bytes())
        for mesh in sorted(world_module.MESH_DIR.glob("*.stl")):
            h.update(mesh.name.encode())
            h.update(str(mesh.stat().st_size).encode())
        return h.hexdigest()[:12]

    def _near_window(self, prim) -> bool:
        if prim.kind == "plane":
            return False
        wx0, wy0, wx1, wy1 = WINDOW
        if prim.source == "visual" and prim.kind == "mesh":
            tris = world_module.world_triangles(prim)
            fx0, fy0 = tris[..., 0].min(), tris[..., 1].min()
            fx1, fy1 = tris[..., 0].max(), tris[..., 1].max()
        else:
            fx0, fy0, fx1, fy1 = _footprint_xy(prim)
        return (
            fx1 >= wx0 - 0.1
            and fx0 <= wx1 + 0.1
            and fy1 >= wy0 - 0.1
            and fy0 <= wy1 + 0.1
        )

    @property
    def G(self):
        g = self.grid
        return np.array(
            [g.x0, g.y0, g.nx, g.ny, self.wi, self.wj, self.wnx, self.wny], np.float64
        )

    def tables(self):
        """(support, obstacles, (layers, solid, wash)): what the kernels read.

        Three small tuples rather than one big one: numba passes a tuple of
        arrays by value, and at thirty-odd arrays that copy cost more than the
        lookup it was carrying.
        """
        s, w, G = self.static, self.window, self.G
        support = (G, s["S_hi"], s["S_mu"], s["S_n"], w["S_hi"], w["S_mu"], w["S_n"])
        obstacles = (G, s["O_lo"], s["O_hi"], s["O_n"], w["O_lo"], w["O_hi"], w["O_n"])
        layers = (G, s["E_top"], s["E_n"], w["E_top"], w["E_n"])
        solid = (G, s["V_lo"], s["V_hi"], s["V_n"], w["V_lo"], w["V_hi"], w["V_n"])
        wash = (G, s["C_lo"], s["C_hi"], s["C_n"], w["C_lo"], w["C_hi"], w["C_n"])
        return support, obstacles, (layers, solid, wash)


# ------------------------------------------------------------------ queries
#
# SUP, OBS and the sensor tuples come from CourseModel.tables(); each starts
# with the grid vector G
# (x0, y0, nx, ny, window i0, window j0, window nx, window ny).  `lay`
# indexes the model's layout list.


@nb.njit(cache=True, inline="always")
def _locate(G, x, y):
    """(in_window, row, col) of the cell holding (x, y); col -1 if off-grid."""
    i = int(math.floor((x - G[0]) / RES))
    j = int(math.floor((y - G[1]) / RES))
    if i < 0 or j < 0 or i >= G[2] or j >= G[3]:
        return False, 0, -1
    wi = i - int(G[4])
    wj = j - int(G[5])
    if 0 <= wi < G[6] and 0 <= wj < G[7]:
        return True, wj, wi
    return False, j, i


@nb.njit(cache=True, inline="always")
def support_below(SUP, lay, x, y, zprobe):
    """(top, mu) of the highest drivable surface at (x, y) at or below zprobe.

    Off the grid is open floor.  A top above zprobe is not road to a wheel at
    that height -- it is a wall to it, or a deck over it.
    """
    win, r, c = _locate(SUP[0], x, y)
    best = -1e9
    best_mu = 1.0
    if c < 0:
        return 0.0, 1.0
    if win:
        hi, mu, n = SUP[4], SUP[5], SUP[6]
        for m in range(n[lay, r, c]):
            top = hi[lay, r, c, m]
            if top <= zprobe and top > best:
                best = top
                best_mu = mu[lay, r, c, m]
    else:
        hi, mu, n = SUP[1], SUP[2], SUP[3]
        for m in range(n[r, c]):
            top = hi[r, c, m]
            if top <= zprobe and top > best:
                best = top
                best_mu = mu[r, c, m]
    if best < -1e8:
        return 0.0, 1.0
    return best, best_mu


# Neighboring cell tops further apart than this are a real edge (a lip, a
# recess wall, a rail) and are not blended into a slope.
SMOOTH_STEP = 0.03


@nb.njit(cache=True, inline="always")
def support_smooth(SUP, lay, x, y, zprobe):
    """support_below, bilinearly blended between cell centers.

    The grid samples a 19% ramp as 3.8 mm stairs; a wheel rolling up stairs
    feeds the dampers a spike at every riser.  Blending the four surrounding
    centers makes a slope a slope, and leaves an edge taller than SMOOTH_STEP
    as sharp as the grid has it.
    """
    G = SUP[0]
    fx = (x - G[0]) / RES - 0.5
    fy = (y - G[1]) / RES - 0.5
    i0 = math.floor(fx)
    j0 = math.floor(fy)
    tx = fx - i0
    ty = fy - j0
    bx = G[0] + (i0 + 0.5) * RES
    by = G[1] + (j0 + 0.5) * RES
    h00, m00 = support_below(SUP, lay, bx, by, zprobe)
    h10, m10 = support_below(SUP, lay, bx + RES, by, zprobe)
    h01, m01 = support_below(SUP, lay, bx, by + RES, zprobe)
    h11, m11 = support_below(SUP, lay, bx + RES, by + RES, zprobe)
    lo = min(min(h00, h10), min(h01, h11))
    hi = max(max(h00, h10), max(h01, h11))
    # The nearest center's surface, for mu and for sharp edges.
    if tx < 0.5:
        near, mu = (h00, m00) if ty < 0.5 else (h01, m01)
    else:
        near, mu = (h10, m10) if ty < 0.5 else (h11, m11)
    if hi - lo > SMOOTH_STEP:
        return near, mu
    h = (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty
    return h, mu


@nb.njit(cache=True, inline="always")
def obstacle_overlap(OBS, lay, x, y, zlo, zhi):
    """Whether any obstacle collision at (x, y) overlaps [zlo, zhi]."""
    win, r, c = _locate(OBS[0], x, y)
    if c < 0:
        return False
    if win:
        lo, hi, n = OBS[4], OBS[5], OBS[6]
        for m in range(n[lay, r, c]):
            if lo[lay, r, c, m] <= zhi and hi[lay, r, c, m] >= zlo:
                return True
    else:
        lo, hi, n = OBS[1], OBS[2], OBS[3]
        for m in range(n[r, c]):
            if lo[r, c, m] <= zhi and hi[r, c, m] >= zlo:
                return True
    return False


@nb.njit(cache=True, inline="always")
def nearest_layer(LAY, lay, x, y, zref):
    """Top of the exposed layer at (x, y) nearest zref, or -1e9 if none."""
    win, r, c = _locate(LAY[0], x, y)
    if c < 0:
        return 0.0
    best = -1e9
    best_d = 1e9
    if win:
        top, n = LAY[3], LAY[4]
        for m in range(n[lay, r, c]):
            d = abs(top[lay, r, c, m] - zref)
            if d < best_d:
                best_d = d
                best = top[lay, r, c, m]
    else:
        top, n = LAY[1], LAY[2]
        for m in range(n[r, c]):
            d = abs(top[r, c, m] - zref)
            if d < best_d:
                best_d = d
                best = top[r, c, m]
    return best


@nb.njit(cache=True, inline="always")
def visual_band(VIS, lay, x, y, ref):
    """(lo, hi) of what the camera would call obstacle at (x, y), or (1, -1).

    The segmenter's rule against the surface `ref` the ray is on: anything
    reaching more than GROUND_TOLERANCE above it, whose lowest part is under
    CLEARANCE above it, blocks.  Below the surface is ground (a lower level
    seen over an edge); starting above the clearance is overhead.
    """
    win, r, c = _locate(VIS[0], x, y)
    lo_out = 1.0
    hi_out = -1.0
    if c < 0:
        return lo_out, hi_out
    if win:
        lo, hi, n = VIS[4], VIS[5], VIS[6]
        for m in range(n[lay, r, c]):
            a = lo[lay, r, c, m]
            b = hi[lay, r, c, m]
            if b > ref + GROUND_TOLERANCE and a < ref + CLEARANCE:
                if hi_out < lo_out:
                    lo_out, hi_out = max(a, ref), b
                else:
                    lo_out, hi_out = min(lo_out, max(a, ref)), max(hi_out, b)
    else:
        lo, hi, n = VIS[1], VIS[2], VIS[3]
        for m in range(n[r, c]):
            a = lo[r, c, m]
            b = hi[r, c, m]
            if b > ref + GROUND_TOLERANCE and a < ref + CLEARANCE:
                if hi_out < lo_out:
                    lo_out, hi_out = max(a, ref), b
                else:
                    lo_out, hi_out = min(lo_out, max(a, ref)), max(hi_out, b)
    return lo_out, hi_out


# ------------------------------------------------------------------ checks


def render(model: CourseModel, lay: int, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import centerline

    g = model.grid
    top = model.static["S_hi"].max(axis=2).copy()
    obst = model.static["O_n"] > 0

    def blocking(d, k=None):
        vlo, vhi, vn = (
            (d["V_lo"], d["V_hi"], d["V_n"])
            if k is None
            else (d["V_lo"][k], d["V_hi"][k], d["V_n"][k])
        )
        top = (d["E_top"] if k is None else d["E_top"][k])[:, :, 0]
        out = np.zeros(vn.shape, bool)
        for m in range(KV):
            live = m < vn
            out |= (
                live
                & (vhi[:, :, m] > top + GROUND_TOLERANCE)
                & (vlo[:, :, m] < top + CLEARANCE)
            )
        return out.astype(float)

    blk = blocking(model.static)
    extent = (g.x0, g.x0 + g.nx * RES, g.y0, g.y0 + g.ny * RES)
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    ax = axes[0]
    ax.imshow(top, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=0.7)
    ax.imshow(
        np.ma.masked_where(~obst, obst),
        origin="lower",
        extent=extent,
        cmap="Reds",
        vmin=0,
        vmax=1,
        alpha=0.9,
    )
    cl = centerline.Centerline(model.layouts[lay])
    ax.plot(cl.points[:, 0], cl.points[:, 1], "w-", lw=0.8)
    ax.set_title(
        f"seed {model.seeds[lay]}: support top (color), obstacles (red), centerline"
    )
    ax = axes[1]
    ax.imshow(blk, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=2)
    ax.set_title("sensor: blocking against the lowest exposed layer")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    seeds = layouts.TRAIN_SEEDS + layouts.HELDOUT_SEEDS
    model = CourseModel(seeds, rebuild=args.rebuild)
    g = model.grid
    print(
        f"grid {g.nx} x {g.ny} at {RES} m from ({g.x0:.2f}, {g.y0:.2f}); window {model.wnx} x {model.wny}"
    )
    if not args.check:
        return 0
    import centerline

    # Every layout's centerline clear of the walls, on this model.  The
    # comparisons against Gazebo itself (surfaces, contact, the sensor) need
    # the running sim: gazebo_check.py.
    failures = centerline.main()
    for lay in range(len(seeds)):
        render(model, lay, CACHE / f"course_{seeds[lay]}.png")
    print(f"renders in {CACHE}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
