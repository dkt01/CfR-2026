"""The Obstacle Course layout draw, free of ROS and Gazebo.

obstacle_randomizer_node draws where the buckets, hoops, wall gap and Wide
Section bales go and then moves the models there.  The draw lives here, on its own, so that
something that is not a ROS node -- rl/obstacleRacer, which trains against
precomputed layouts in a numpy model of the course -- gets exactly the layout
the node would put in Gazebo for the same seed.  Same seed, same numbers: the
node calls these functions, it does not keep a copy of them.

`spec` is the ``obstacle_randomizer.ros__parameters`` mapping from
``config/obstacle_course_layout.yaml``, nested as the YAML nests it.
"""

from __future__ import annotations

import math
import random


class LayoutError(RuntimeError):
    """A layout the spec asks for cannot be drawn."""


def draw(spec: dict, seed: int, bucket_count: int = 0) -> dict:
    """The whole layout for one seed, in the order the node draws it.

    Order matters: all three draws share one generator, so drawing the hoops
    before the buckets would give a different layout for the same seed.
    """
    rng = random.Random(seed)
    buckets = draw_buckets(spec, rng, bucket_count)
    hoops = draw_hoops(spec, rng)
    gap_bale = draw_gap_bale(spec, rng)
    # Last, so every earlier draw is unchanged for a given seed.
    wide_bales = draw_wide_bales(spec, rng, gap_bale)
    return {
        "seed": seed,
        "buckets": buckets,
        "hoops": hoops,
        "gap_bale": gap_bale,
        "wide_bales": wide_bales,
    }


def draw_buckets(spec: dict, rng: random.Random, count: int = 0):
    """Bucket positions honoring the drawing's spacing and clearance.

    The drawing asks for buckets "placed so a path exists around and
    between buckets".  That does not need a separate reachability check:
    3 ft between centers leaves a 0.62 m gap between two 0.29 m buckets,
    and the same clearance off the walls, both of which a 0.30 m car fits
    through.  Keeping the spacing is keeping the path.
    """
    buckets = spec.get("buckets")
    if not buckets:
        return []
    clearance = float(buckets["wall_clearance"])
    spacing = float(buckets["min_spacing"])
    low = [value + clearance for value in buckets["region_min"]]
    high = [value - clearance for value in buckets["region_max"]]
    if low[0] > high[0] or low[1] > high[1]:
        raise LayoutError("bucket clearance leaves no room in the section")

    count_min, count_max = int(buckets["count_min"]), int(buckets["count_max"])
    if count <= 0:
        count = rng.randint(count_min, count_max)
    count = max(count_min, min(count_max, count))

    # Rejection sampling first, restarted rather than relaxed: a greedy
    # fill can paint itself into a corner, and starting over is simpler
    # than backtracking at this size.
    for _ in range(200):
        placed: list[tuple[float, float]] = []
        for _ in range(count * 200):
            if len(placed) == count:
                break
            candidate = (rng.uniform(low[0], high[0]), rng.uniform(low[1], high[1]))
            if all(math.dist(candidate, other) >= spacing for other in placed):
                placed.append(candidate)
        if len(placed) == count:
            return placed

    # The high counts do not fall out of rejection sampling.  Nine buckets
    # 3 ft apart fit the section, but only in very nearly a 3x3 grid, and
    # the chance of stumbling on that by drawing points uniformly is not
    # worth waiting for.  So lay out a grid and jitter it by whatever slack
    # the spacing leaves: still a different layout every time, just not a
    # uniform one, which is the honest trade at the top of the range.
    return grid_layout(rng, low, high, count, spacing)


def grid_layout(rng: random.Random, low, high, count: int, spacing: float):
    span = (high[0] - low[0], high[1] - low[1])
    best = None
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        pitch = tuple(
            span[axis] / (steps - 1) if steps > 1 else float("inf")
            for axis, steps in ((0, columns), (1, rows))
        )
        if min(pitch) < spacing:
            continue
        slack = min(pitch[0] - spacing, pitch[1] - spacing)
        if best is None or slack > best[0]:
            best = (slack, columns, rows, pitch)
    if best is None:
        raise LayoutError(
            f"{count} buckets will not fit {spacing:.2f} m apart in the section"
        )

    _, columns, rows, pitch = best
    cells = [(column, row) for column in range(columns) for row in range(rows)]

    placed = []
    for cell in rng.sample(cells, count):
        position = []
        for axis, (index, steps) in enumerate(((cell[0], columns), (cell[1], rows))):
            if steps == 1:
                # A single row or column is free to sit anywhere: nothing
                # on this axis constrains it.
                position.append(rng.uniform(low[axis], high[axis]))
                continue
            jitter = (pitch[axis] - spacing) / 2
            center = low[axis] + index * pitch[axis]
            offset = rng.uniform(-jitter, jitter)
            position.append(min(high[axis], max(low[axis], center + offset)))
        placed.append(tuple(position))
    return placed


def draw_hoops(spec: dict, rng: random.Random) -> dict[str, tuple[float, float]]:
    """A position for each hoop along the line the drawing puts it on.

    The line spans the full width of the corridor, so the ends are clamped
    by half the hoop's base -- a hoop centered on the very end would stand
    half outside the bales.
    """
    hoops = spec.get("hoops") or {}
    positions = {}
    for hoop in hoops.get("names", []):
        start = list(hoops[hoop]["from"])
        end = list(hoops[hoop]["to"])
        span = math.dist(start, end)
        margin = float(hoops[hoop]["base_length"]) / 2.0
        if span <= 2 * margin:
            positions[hoop] = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            continue
        fraction = rng.uniform(margin / span, 1.0 - margin / span)
        positions[hoop] = (
            start[0] + (end[0] - start[0]) * fraction,
            start[1] + (end[1] - start[1]) * fraction,
        )
    return positions


def draw_gap_bale(spec: dict, rng: random.Random):
    """Which of the four wall bales stands off-course this draw, or None.

    Exactly one is always parked -- the wall is four bale-widths tall and
    the gap it leaves is what the vehicle drives through -- so this picks
    one name rather than a count or a set.
    """
    gap_bales = spec.get("gap_bales")
    if not gap_bales:
        return None
    return rng.choice(list(gap_bales["names"]))


# ---------------------------------------------------------------- wide section
#
# The Wide Section is drawn as "approx 11' x 26' open area with changeable
# boundaries": the bales standing in it (wide_bales.names) are moved between
# runs.  Any yaw, anywhere inside `bounds`, never overlapping a fixed wall or
# each other -- and always leaving a path at least `min_track_width` wide from
# `entry` (where the potholes lane opens into the section) to the bucket
# section's entrance, wherever draw_gap_bale left it.  The width is the
# course's guarantee: 20 in of track through the whole course.
#
# Plain Python on purpose: this runs in the ROS node, where the draw has to
# come out identical to the trainer's for the same seed.

WIDE_TRIES = 400
WIDE_PLACE_TRIES = 200


def _flat_boxes(values) -> list[tuple[float, float, float, float, float]]:
    """(x, y, yaw, length, width) boxes from a flat parameter array."""
    values = [float(v) for v in values]
    if len(values) % 5:
        raise LayoutError("wide_bales.walls must be groups of x, y, yaw, length, width")
    return [tuple(values[i : i + 5]) for i in range(0, len(values), 5)]


def _corners(box) -> list[tuple[float, float]]:
    x, y, yaw, length, width = box
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = length / 2.0, width / 2.0
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))
    ]


def _overlap(a, b) -> bool:
    """Separating-axis test for two boxes; touching does not count."""
    ca, cb = _corners(a), _corners(b)
    for box in (a, b):
        for yaw in (box[2], box[2] + math.pi / 2):
            ax, ay = math.cos(yaw), math.sin(yaw)
            pa = [px * ax + py * ay for px, py in ca]
            pb = [px * ax + py * ay for px, py in cb]
            if max(pa) <= min(pb) + 1e-9 or max(pb) <= min(pa) + 1e-9:
                return False
    return True


def _distance(box, px: float, py: float) -> float:
    """Distance from a point to a box's footprint (0 inside)."""
    x, y, yaw, length, width = box
    c, s = math.cos(yaw), math.sin(yaw)
    lx = c * (px - x) + s * (py - y)
    ly = -s * (px - x) + c * (py - y)
    dx = max(abs(lx) - length / 2.0, 0.0)
    dy = max(abs(ly) - width / 2.0, 0.0)
    return math.hypot(dx, dy)


class _TrackGrid:
    """Cells of path_bounds whose centers are a half track width clear.

    A cell is open when its center is at least `radius` from every wall; a
    path of open cells is a path the track's center line can take with the
    full width free on both sides.  The radius carries half a cell's
    diagonal on top of half the width, so moving between neighboring cell
    centers never cuts closer than the width allows.
    """

    def __init__(self, bounds, step, radius, walls):
        self.x0, self.y0, x1, y1 = bounds
        self.step = step
        self.radius = radius
        self.nx = int(math.ceil((x1 - self.x0) / step))
        self.ny = int(math.ceil((y1 - self.y0) / step))
        self.open = bytearray(b"\x01") * (self.nx * self.ny)
        for wall in walls:
            self.close_near(wall, self.open)

    def cell(self, x: float, y: float):
        i = int((x - self.x0) / self.step)
        j = int((y - self.y0) / self.step)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def close_near(self, box, grid) -> None:
        corners = _corners(box)
        reach = self.radius
        i0 = max(0, int((min(p[0] for p in corners) - reach - self.x0) / self.step))
        i1 = min(
            self.nx - 1, int((max(p[0] for p in corners) + reach - self.x0) / self.step)
        )
        j0 = max(0, int((min(p[1] for p in corners) - reach - self.y0) / self.step))
        j1 = min(
            self.ny - 1, int((max(p[1] for p in corners) + reach - self.y0) / self.step)
        )
        for j in range(j0, j1 + 1):
            cy = self.y0 + (j + 0.5) * self.step
            row = j * self.nx
            for i in range(i0, i1 + 1):
                if (
                    grid[row + i]
                    and _distance(box, self.x0 + (i + 0.5) * self.step, cy) < reach
                ):
                    grid[row + i] = 0

    def connected(self, bales, start, goal) -> bool:
        grid = bytearray(self.open)
        for bale in bales:
            self.close_near(bale, grid)
        a, b = self.cell(*start), self.cell(*goal)
        if a is None or b is None:
            return False
        first, last = a[1] * self.nx + a[0], b[1] * self.nx + b[0]
        if not grid[first] or not grid[last]:
            return False
        grid[first] = 0
        frontier = [first]
        nx, n = self.nx, len(grid)
        while frontier:
            here = frontier.pop()
            if here == last:
                return True
            i = here % nx
            for step in (-nx, nx, -1 if i > 0 else 0, 1 if i < nx - 1 else 0):
                there = here + step
                if step and 0 <= there < n and grid[there]:
                    grid[there] = 0
                    frontier.append(there)
        return False


def wide_walls(spec: dict, gap_bale) -> list[tuple[float, float, float, float, float]]:
    """The fixed walls around the Wide Section, gap bales standing included."""
    wide = spec["wide_bales"]
    walls = _flat_boxes(wide["walls"])
    gap_bales = spec.get("gap_bales") or {}
    length, width = (float(v) for v in wide["size"])
    for name in gap_bales.get("names", []):
        if name == gap_bale:
            continue
        x, y = (float(v) for v in gap_bales[name]["position"])
        walls.append((x, y, float(gap_bales[name]["yaw"]), length, width))
    return walls


def wide_exit(spec: dict, gap_bale) -> tuple[float, float]:
    """Where the path out of the Wide Section ends: in the open entrance slot."""
    gap_bales = spec.get("gap_bales") or {}
    if gap_bale is None or gap_bale not in gap_bales:
        return tuple(float(v) for v in spec["wide_bales"]["exit"])
    x, y = (float(v) for v in gap_bales[gap_bale]["position"])
    return x, y


def wide_track_grid(spec: dict, gap_bale) -> _TrackGrid:
    wide = spec["wide_bales"]
    return _TrackGrid(
        [float(v) for v in wide["path_bounds"]],
        float(wide["grid"]),
        float(wide["min_track_width"]) / 2.0 + float(wide["grid"]) * 0.75,
        wide_walls(spec, gap_bale),
    )


def wide_path_ok(spec: dict, gap_bale, poses: dict) -> bool:
    """Whether `poses` leave the guaranteed track width from entry to exit."""
    wide = spec["wide_bales"]
    length, width = (float(v) for v in wide["size"])
    bales = [(x, y, yaw, length, width) for x, y, yaw in poses.values()]
    grid = wide_track_grid(spec, gap_bale)
    entry = tuple(float(v) for v in wide["entry"])
    return grid.connected(bales, entry, wide_exit(spec, gap_bale))


def draw_wide_bales(spec: dict, rng: random.Random, gap_bale=None):
    """A pose (x, y, yaw) for every Wide Section bale, by name, or {}.

    Rejection sampling, restarted rather than repaired: each bale is drawn
    uniformly in `bounds` at a uniform yaw until it overlaps nothing, and a
    full set that leaves no path of the guaranteed width is thrown away.
    """
    wide = spec.get("wide_bales")
    if not wide:
        return {}
    names = list(wide["names"])
    length, width = (float(v) for v in wide["size"])
    x0, y0, x1, y1 = (float(v) for v in wide["bounds"])
    keep_out = _flat_boxes(wide.get("keep_out", []))
    walls = wide_walls(spec, gap_bale) + keep_out
    grid = wide_track_grid(spec, gap_bale)
    entry = tuple(float(v) for v in wide["entry"])
    goal = wide_exit(spec, gap_bale)
    for _ in range(WIDE_TRIES):
        placed = []
        for _ in names:
            for _ in range(WIDE_PLACE_TRIES):
                box = (
                    rng.uniform(x0, x1),
                    rng.uniform(y0, y1),
                    rng.uniform(0.0, math.pi),
                    length,
                    width,
                )
                inside = all(
                    x0 <= px <= x1 and y0 <= py <= y1 for px, py in _corners(box)
                )
                if inside and not any(_overlap(box, other) for other in walls + placed):
                    placed.append(box)
                    break
            else:
                break
        if len(placed) == len(names) and grid.connected(placed, entry, goal):
            return {name: box[:3] for name, box in zip(names, placed)}
    raise LayoutError(
        f"no Wide Section layout left a {float(wide['min_track_width']):.3f} m path "
        f"in {WIDE_TRIES} tries"
    )


def nominal(spec: dict) -> dict:
    """The layout the drawing shows -- what the node's ~/reset restores."""
    buckets = spec.get("buckets") or {}
    hoops = spec.get("hoops") or {}
    gap_bales = spec.get("gap_bales") or {}
    wide = spec.get("wide_bales") or {}
    return {
        "seed": None,
        "buckets": [
            tuple(buckets["nominal"][f"bucket_{index}"])
            for index in range(int(buckets.get("default_count", 0)))
        ],
        "hoops": {
            hoop: tuple(hoops[hoop]["nominal"]) for hoop in hoops.get("names", [])
        },
        "gap_bale": gap_bales.get("default_gap"),
        "wide_bales": {
            name: tuple(float(v) for v in wide[name]["nominal"])
            for name in wide.get("names", [])
        },
    }
