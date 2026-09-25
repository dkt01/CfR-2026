"""The Obstacle Course layout draw, free of ROS and Gazebo.

obstacle_randomizer_node draws where the buckets, hoops and wall gap go and
then moves the models there.  The draw lives here, on its own, so that
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
    return {"seed": seed, "buckets": buckets, "hoops": hoops, "gap_bale": gap_bale}


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


def nominal(spec: dict) -> dict:
    """The layout the drawing shows -- what the node's ~/reset restores."""
    buckets = spec.get("buckets") or {}
    hoops = spec.get("hoops") or {}
    gap_bales = spec.get("gap_bales") or {}
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
    }
