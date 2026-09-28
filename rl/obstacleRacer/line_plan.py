"""Plan each layout's line round its own obstacles, as far from them as it can.

The hand-placed line (centerline.py) is one line for every layout, and the
randomizer puts buckets and Wide Section bales on it: 84% of layouts had the
line through a bale, and v7 stalled at 25% through the Wide Section, rocking
where the line it was paid along ran into one.  So from the tunnel exit to the
finish the line is planned per layout on the baked physics grid instead:

* a cell blocks if an obstacle interval stands in the chassis band over the
  cell's floor (course_model's PHYSICS geometry, the same the car hits);
* the path is the cheapest one on that grid where a step costs its length
  times 1 + CLEARANCE_WEIGHT * (1/clearance - 1/CLEARANCE_CAP): it keeps to
  the middle of lanes and gaps, and past CLEARANCE_CAP of room it just takes
  the short way;
* it stays within CORRIDOR of the hand-placed line, except in the two rooms
  whose walls bound them (ROOMS), so no gap in a wall can take it off the
  course's route;
* it goes through ordered via points that keep it on the course's route (the
  potholes lane, the open gap-bale slot, the bucket section's exit) and runs
  square through each hoop's center for HOOP_LEAD either side.

The line is PRIVILEGED: it shapes the reward and never reaches the policy.
Plans are cached in .cache per layout, keyed on the grids and this code.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path

import numba as nb
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"

PLAN_RES = 0.04  # m, twice the baked grid's cell
# The whole baked grid is planned over (it runs MARGIN past the outermost
# wall); only floor-level support counts as ground, so the deck, the ramp and
# the helix are walls to it, and the via points keep it off them anyway.
GROUND_MAX = 0.30  # m: support tops above this are the deck or a wall top
BAND = (0.02, 0.25)  # m over the floor: the chassis, bumper to camera
# The car is 0.35 m across its tires; a path this close to an obstacle
# still clears it.
MIN_CLEARANCE = 0.20
CLEARANCE_CAP = 1.0
CLEARANCE_WEIGHT = 1.5
# Off the hand line by more than this is off the course (the lanes are
# walled well inside it)...
CORRIDOR = 1.5
# ...except in the Wide Section and the bucket section, walled all round,
# where the obstacles stand anywhere (x0, y0, x1, y1).
ROOMS = ((3.0, -9.3, 6.9, -0.6), (-0.66, -4.63, 3.01, -0.88))
SMOOTH_PASSES = 4
SMOOTH_HALF = 4  # points either side in the moving average


def blocked_map(model, lay):
    """(blocked, x0, y0) at PLAN_RES over the whole grid for layout `lay`."""
    import course_model

    g = model.grid
    res = course_model.RES
    i0, j0, i1, j1 = 0, 0, g.nx, g.ny

    def blocked(A):
        k_s = np.arange(A["S_hi"].shape[-1])
        s_ok = k_s < A["S_n"][..., None]
        top = np.where(s_ok & (A["S_hi"] <= GROUND_MAX), A["S_hi"], -1.0).max(-1)
        top = np.maximum(top, 0.0)[..., None]
        k_o = np.arange(A["O_lo"].shape[-1])
        o_ok = k_o < A["O_n"][..., None]
        hit = o_ok & (A["O_lo"] < top + BAND[1]) & (A["O_hi"] > top + BAND[0])
        return hit.any(-1)

    s = model.static
    full = blocked(
        {k: s[k][j0:j1, i0:i1] for k in ("S_hi", "S_n", "O_lo", "O_hi", "O_n")}
    )
    w = model.dense_window(lay)
    wb = blocked(w)
    # Paste the layout's window over the static grid where they overlap.
    wi0, wj0 = model.wi - i0, model.wj - j0
    a0, b0 = max(wi0, 0), max(wj0, 0)
    a1 = min(wi0 + model.wnx, full.shape[1])
    b1 = min(wj0 + model.wny, full.shape[0])
    full[b0:b1, a0:a1] = wb[b0 - wj0 : b1 - wj0, a0 - wi0 : a1 - wi0]
    # Halve the resolution: a coarse cell blocks if any of its four do.
    ny, nx = full.shape[0] // 2 * 2, full.shape[1] // 2 * 2
    f = full[:ny, :nx]
    coarse = f[0::2, 0::2] | f[1::2, 0::2] | f[0::2, 1::2] | f[1::2, 1::2]
    return coarse, g.x0 + i0 * res, g.y0 + j0 * res


def clearance(blocked):
    """Meters from each cell's center to the nearest blocked cell's edge."""
    d = ndimage.distance_transform_edt(~blocked) * PLAN_RES - PLAN_RES / 2
    return np.maximum(d, 0.0)


def step_cost(clear):
    c = np.maximum(clear, 1e-3)
    extra = np.maximum(1.0 / c - 1.0 / CLEARANCE_CAP, 0.0)
    cost = 1.0 + CLEARANCE_WEIGHT * extra
    return np.where(clear >= MIN_CLEARANCE, cost, np.inf)


@nb.njit(cache=True)
def _dijkstra(cost, start, goal):
    """Cheapest 8-connected path from start to goal, as flat cell indices.

    The start and goal cells are always enterable, even if too close to an
    obstacle, so a via point placed in a tight spot still connects.
    """
    ny, nx = cost.shape
    n = ny * nx
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, np.int64)
    done = np.zeros(n, np.bool_)
    dist[start] = 0.0
    heap = [(0.0, start)]
    di = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dj = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        if u == goal:
            break
        uj, ui = u // nx, u % nx
        cu = cost[uj, ui] if u != start else 1.0
        for k in range(8):
            vj, vi = uj + di[k], ui + dj[k]
            if vj < 0 or vi < 0 or vj >= ny or vi >= nx:
                continue
            v = vj * nx + vi
            if done[v]:
                continue
            cv = cost[vj, vi] if v != goal else 1.0
            if not np.isfinite(cv) or not np.isfinite(cu):
                continue
            step = 1.4142135623730951 if di[k] != 0 and dj[k] != 0 else 1.0
            nd = d + step * 0.5 * (cu + cv)
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    path = []
    u = goal
    while u != -1:
        path.append(u)
        if u == start:
            break
        u = prev[u]
    if path[-1] != start:
        return np.empty(0, np.int64)
    out = np.empty(len(path), np.int64)
    for k in range(len(path)):
        out[k] = path[len(path) - 1 - k]
    return out


def off_course(shape, x0, y0, guide):
    """Cells farther than CORRIDOR from the guide polyline, outside ROOMS."""
    ny, nx = shape
    near = np.zeros(shape, bool)
    xy = _densify_xy(guide, PLAN_RES / 2)
    i = ((xy[:, 0] - x0) / PLAN_RES).astype(int)
    j = ((xy[:, 1] - y0) / PLAN_RES).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < nx) & (j < ny)
    near[j[ok], i[ok]] = True
    dist = ndimage.distance_transform_edt(~near) * PLAN_RES
    out = dist > CORRIDOR
    # The grid's edge is open floor, but nothing is out there.
    out[[0, -1], :] = True
    out[:, [0, -1]] = True
    xs = x0 + (np.arange(nx) + 0.5) * PLAN_RES
    ys = y0 + (np.arange(ny) + 0.5) * PLAN_RES
    for rx0, ry0, rx1, ry1 in ROOMS:
        out[np.ix_((ys >= ry0) & (ys <= ry1), (xs >= rx0) & (xs <= rx1))] = False
    return out


def _densify_xy(xy, spacing):
    out = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        t = np.arange(1, k + 1)[:, None] / k
        out.append(a + (b - a) * t)
    return np.concatenate(out)


class Planner:
    """One layout's grid, clearance and costs; plans legs between points."""

    def __init__(self, model, lay, guide):
        self.blocked, self.x0, self.y0 = blocked_map(model, lay)
        self.blocked |= off_course(self.blocked.shape, self.x0, self.y0, guide)
        self.clear = clearance(self.blocked)
        self.cost = step_cost(self.clear)

    def cell(self, p):
        i = int(round((p[0] - self.x0) / PLAN_RES - 0.5))
        j = int(round((p[1] - self.y0) / PLAN_RES - 0.5))
        return j * self.cost.shape[1] + i

    def center(self, flat):
        nx = self.cost.shape[1]
        j, i = np.divmod(flat, nx)
        return np.c_[self.x0 + (i + 0.5) * PLAN_RES, self.y0 + (j + 0.5) * PLAN_RES]

    def clearance_at(self, xy):
        xy = np.atleast_2d(xy)
        i = np.clip(
            ((xy[:, 0] - self.x0) / PLAN_RES).astype(int), 0, self.clear.shape[1] - 1
        )
        j = np.clip(
            ((xy[:, 1] - self.y0) / PLAN_RES).astype(int), 0, self.clear.shape[0] - 1
        )
        return self.clear[j, i]

    def leg(self, a, b):
        cells = _dijkstra(self.cost, self.cell(a), self.cell(b))
        if len(cells) == 0:
            raise RuntimeError(f"no path from {a} to {b}")
        pts = self.center(cells)
        pts[0], pts[-1] = a, b
        return pts


def plan(model, lay, route, guide):
    """Plan the route: (kind, (x, y)) points in driving order.

    kind is "via" (the path must pass through it), or "hoop_in", "hoop",
    "hoop_out": a hoop's lead-in, center and lead-out, joined straight so the
    line runs square through the hoop.  Everything else is planned.
    Returns (smoothed points (N, 2), planner).
    """
    planner = Planner(model, lay, guide)
    pts = [np.asarray([route[0][1]], float)]
    pinned = [np.array([True])]
    for (_, a), (kind_b, b) in zip(route, route[1:]):
        straight = kind_b in ("hoop", "hoop_out")
        seg = np.array([a, b], float) if straight else planner.leg(a, b)
        pts.append(seg[1:])
        mask = np.zeros(len(seg) - 1, bool)
        mask[-1] = True
        pinned.append(mask)
    return _smooth(np.concatenate(pts), np.concatenate(pinned), planner), planner


def _smooth(pts, pinned, planner):
    """Take the grid's stair steps out, never closer to an obstacle than allowed."""
    out = pts.copy()
    kernel = np.ones(2 * SMOOTH_HALF + 1) / (2 * SMOOTH_HALF + 1)
    for _ in range(SMOOTH_PASSES):
        cand = out.copy()
        for c in range(2):
            padded = np.pad(out[:, c], SMOOTH_HALF, mode="edge")
            cand[:, c] = np.convolve(padded, kernel, mode="valid")
        cand[pinned] = out[pinned]
        ok = planner.clearance_at(cand) >= np.minimum(
            planner.clearance_at(out), MIN_CLEARANCE
        )
        out[ok] = cand[ok]
    return out


def resample(xy, spacing):
    """Points every `spacing` meters along the polyline, ends kept."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    xy = xy[keep]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    n = max(2, int(round(arc[-1] / spacing)) + 1)
    s = np.linspace(0.0, arc[-1], n)
    return np.c_[np.interp(s, arc, xy[:, 0]), np.interp(s, arc, xy[:, 1])]


def cache_key(model, layout, route, guide) -> str:
    h = hashlib.sha1()
    h.update(Path(__file__).read_bytes())
    h.update(model._cache_key().encode())
    h.update(json.dumps(layout, sort_keys=True).encode())
    h.update(json.dumps(route).encode())
    h.update(np.ascontiguousarray(guide, np.float64).tobytes())
    return h.hexdigest()[:16]


def planned_xy(model, lay, route, guide, spacing):
    """Cached plan for layout `lay`, resampled every `spacing` meters.

    `guide` is the hand-placed line's (x, y) points: the corridor to stay in.
    """
    layout = model.layouts[lay]
    CACHE.mkdir(exist_ok=True)
    path = (
        CACHE
        / f"line_{layout.get('seed', lay)}_{cache_key(model, layout, route, guide)}.npy"
    )
    if path.exists():
        return np.load(path)
    pts, _ = plan(model, lay, route, guide)
    xy = resample(pts, spacing)
    np.save(path, xy)
    return xy
