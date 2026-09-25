"""Course progress for the Obstacle Course: arc length along a centerline.

Arc length `s` along a fixed line through the course is what the reward pays
for.  Progress is "how much further round the lap", going the wrong way is
negative, and because the reward reads only differences in `s` it telescopes
over an episode to `s_end - s_start` -- no amount of weaving farms it, and the
line only has to be roughly right, not a racing line.

`s` is PRIVILEGED: it needs the car's ground-truth pose.  It is used by the
reward and the metrics and never reaches the observation.

The line is hand-placed anchors through the lane, in driving order, with the
helix generated from generate_obstacle_course.py's own constants.  One stretch
depends on the layout: the randomizer opens a different one of four slots in
the bucket section's east wall (gap_bale_0..3), so the line enters the
section through whichever one is open.  `python3 centerline.py` checks every
layout's line against the walls.

Projection is a windowed nearest-point search per car, vectorized over cars
with numba.  The window keeps the start line unambiguous (a lap starts and
ends on the same spot) and stops a car that leaves the lane from being
credited with a stretch of course it went round rather than along.
"""

from __future__ import annotations

import math
import sys

import numba as nb
import numpy as np

import layouts

# Straight-line anchors along the lane, in driving order, as (x, y, z) world
# meters. Derived from generate_obstacle_course.py's rect constants (via its
# to_world half-turn about the start line) and checked against the bale walls
# by this module's __main__. The helical ramp is not listed: it is generated
# between DECK_END and TUNNEL_MOUTH by _helix_points below.
FOOT = 0.3048
INCH = FOOT / 12.0

# Helix, straight out of generate_obstacle_course.py (DXF feet -> world via
# the same half turn). The 270 degree sweep starts at the bridge deck's east
# end and finishes on the helix center's own y, which is the tunnel mouth.
HELIX_CENTRE = (8.299, 1.204)
HELIX_RADIUS = (2.214 + 5.667) / 2 * FOOT
HELIX_START_RAD = math.radians(270.0)
HELIX_SWEEP_RAD = math.radians(270.0)
DECK_HEIGHT = 25 * INCH

DECK_END = (8.299, 0.003, DECK_HEIGHT)
TUNNEL_MOUTH = (7.098, 1.204, 0.0)

# Everything before the helix: the start straight and the ramp up onto the
# deck. RAMP_UP spans x 3.378..6.693 and BRIDGE_DECK 6.693..8.299, both on
# the lane's own y.
BEFORE_HELIX = [
    # The car spawns 0.70 m behind the start line, so s = 0 is where it
    # actually starts, not where the line is. That makes the lap end 0.70 m
    # further along than it began -- which is the point: finishing means
    # crossing the line, having started behind it.
    (-0.70, 0.00, 0.0),
    (0.00, 0.10, 0.0),
    (2.00, 0.05, 0.0),
    (3.38, 0.00, 0.0),  # foot of the ramp
    (5.00, 0.00, DECK_HEIGHT * 0.49),  # on the ramp, 11% grade
    (6.69, 0.00, DECK_HEIGHT),  # ramp meets the deck
    DECK_END,
]

# From the tunnel mouth to the foot of the east corridor, where the route
# into the bucket section starts to depend on the layout.
# The tunnel runs south under the deck; the long corridor at x ~ 7.1 drops to
# the gravel run; the bank turns the car back east through the potholes; the
# corridor at x ~ 4.6 climbs north into the bucket section; the gap in the
# gap-bale wall lets it west into the hoops; hoop_2 turns it north; the car
# wash is the last feature on the start straight.
AFTER_HELIX = [
    TUNNEL_MOUTH,
    (7.10, 0.30, 0.0),  # under the bridge deck
    (7.10, -0.65, 0.0),  # out of the tunnel
    (7.12, -4.00, 0.0),
    (7.15, -7.60, 0.0),
    (7.15, -8.60, 0.0),
    (6.75, -9.10, 0.0),
    (6.65, -9.45, 0.0),
    (6.10, -9.70, 0.0),
    (5.20, -9.85, 0.0),
    (4.00, -9.90, 0.0),
    (3.54, -9.90, 0.0),  # gravel pit, east edge
    (2.30, -9.92, 0.0),
    (1.10, -9.92, 0.0),  # gravel pit, west edge
    (-0.50, -9.90, 0.0),
    # Stay south until past x = -3.43: the wall closing the gravel run's
    # north side reaches that far west, so turning up any earlier drives
    # into it.
    (-2.60, -9.88, 0.0),
    (-3.80, -9.85, 0.0),
    (-4.15, -9.30, 0.0),  # banked turn
    (-4.20, -8.70, 0.0),
    (-3.60, -8.35, 0.0),
    (-2.00, -8.30, 0.0),
    (0.04, -8.30, 0.0),  # potholes, west edge
    (1.20, -8.30, 0.0),
    (2.48, -8.30, 0.0),  # potholes, east edge
    (3.80, -8.15, 0.0),
    (4.55, -7.40, 0.0),
    (4.55, -5.00, 0.0),
]

# Out through the bucket section's west wall, between bale 15 and the
# permanent wall bale below it -- this opening does not move.
BUCKET_EXIT = (-0.88, -1.98, 0.0)
# After hoop_2: north of the stub wall east of it, then back east along the
# start straight through the car wash to the line.
AFTER_HOOPS = [
    (-6.60, 0.75, 0.0),
    (-5.60, 0.45, 0.0),
    (-4.09, 0.28, 0.0),  # car wash, west end
    (-2.87, 0.22, 0.0),
    (-1.65, 0.18, 0.0),  # car wash, east end
    (0.00, 0.10, 0.0),  # finish line, back where it started
]
# How far either side of a hoop the line runs square through it, so the
# line threads the gate rather than cutting across it.
HOOP_LEAD = 0.45


def _after_buckets(layout) -> list[tuple[float, float, float]]:
    """Through the hoops wherever this layout slid them, then home."""
    spec = layouts.layout_spec()["hoops"]
    nominal = {h: spec[h]["nominal"] for h in spec["names"]}
    hoops = layout["hoops"] if layout else nominal
    (x0, y0), (x1, y1), (x2, y2) = hoops["hoop_0"], hoops["hoop_1"], hoops["hoop_2"]
    return [
        BUCKET_EXIT,
        (-2.40, (BUCKET_EXIT[1] + y0) / 2, 0.0),
        # hoop_0 and hoop_1 face along x: the car drives west through them.
        (x0 + HOOP_LEAD, y0, 0.0),
        (x0, y0, 0.0),
        (x0 - HOOP_LEAD, y0, 0.0),
        (x1 + HOOP_LEAD, y1, 0.0),
        (x1, y1, 0.0),
        (x1 - HOOP_LEAD, y1, 0.0),
        (-7.60, -1.60, 0.0),
        # hoop_2 faces along y: the car drives north through it.
        (x2, y2 - 0.9, 0.0),
        (x2, y2 - HOOP_LEAD, 0.0),
        (x2, y2, 0.0),
        (x2, y2 + HOOP_LEAD, 0.0),
        (x2 + 0.1, 0.75, 0.0),
    ] + AFTER_HOOPS


# The four entrance slots are gap_bale_0..3 in obstacle_course_layout.yaml, on
# the bucket section's east wall at this x.
EAST_WALL_X = 3.2374


def _bucket_entry(layout) -> list[tuple[float, float, float]]:
    slot = layout["gap_bale"] if layout else "gap_bale_0"
    y = layouts.layout_spec()["gap_bales"][slot]["position"][1]
    if slot == "gap_bale_0":
        # The drawing's own opening, at the top of the east corridor, which
        # is closed off beyond it: turn west below y ~ -1.75.
        entry = [
            (4.55, -2.60, 0.0),
            (4.35, -2.35, 0.0),
            (4.00, -2.10, 0.0),
            (3.85, -1.75, 0.0),
            (EAST_WALL_X, y - 0.10, 0.0),
            (2.50, y - 0.10, 0.0),
        ]
    else:
        entry = [
            (4.55, y - 0.60, 0.0),
            (4.10, y, 0.0),
            (EAST_WALL_X, y, 0.0),
            (2.40, y, 0.0),
        ]
    # Across the section toward the fixed exit; buckets stand anywhere in
    # it, so this is a progress coordinate, not a path.
    entry.append((1.00, (y - 1.98) / 2.0, 0.0))
    return entry


SAMPLE_SPACING_M = 0.10
# Vertical separation counts for more than horizontal when deciding which part
# of the course a pose belongs to: the bridge deck passes directly over the
# tunnel, so (x, y) alone is ambiguous there and only z separates them.
Z_WEIGHT = 4.0
WINDOW_BACK_M = 3.0
WINDOW_AHEAD_M = 6.0


def _helix_points(count: int = 36):
    cx, cy = HELIX_CENTRE
    points = []
    for index in range(1, count):
        fraction = index / count
        angle = HELIX_START_RAD + HELIX_SWEEP_RAD * fraction
        points.append(
            (
                cx + HELIX_RADIUS * math.cos(angle),
                cy + HELIX_RADIUS * math.sin(angle),
                DECK_HEIGHT * (1.0 - fraction),
            )
        )
    return points


def _densify(anchors, spacing):
    out = [anchors[0]]
    for (x0, y0, z0), (x1, y1, z1) in zip(anchors, anchors[1:]):
        span = math.dist((x0, y0, z0), (x1, y1, z1))
        steps = max(1, int(span / spacing))
        for step in range(1, steps + 1):
            t = step / steps
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t))
    return out


class Centerline:
    """One layout's line: points, cumulative arc length, and section marks."""

    def __init__(self, layout=None, spacing: float = SAMPLE_SPACING_M):
        anchors = (
            BEFORE_HELIX
            + _helix_points()
            + AFTER_HELIX
            + _bucket_entry(layout)
            + _after_buckets(layout)
        )
        self.points = np.asarray(_densify(anchors, spacing))
        steps = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(steps)])
        self.lap_length = float(self.arc[-1])
        self.helix_start_s = float(self.arc[self.nearest_index(DECK_END)])
        self.helix_end_s = float(self.arc[self.nearest_index(TUNNEL_MOUTH)])

    def nearest_index(self, point) -> int:
        return int(np.argmin(np.linalg.norm(self.points - np.asarray(point), axis=1)))

    def index_at(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self.arc, s), 0, len(self.arc) - 1))

    def pose_at(self, s: float):
        """(x, y, z, yaw) at arc length s, yaw aimed 0.75 m down the line.

        Aimed ahead rather than along the local tangent: at a sharp corner the
        tangent points across the turn, which is a bad way to be dealt in.
        """
        i = self.index_at(s)
        x, y, z = self.points[i]
        ahead = self.points[self.index_at(min(s + 0.75, self.lap_length))]
        if (ahead[0] - x) ** 2 + (ahead[1] - y) ** 2 < 1e-9:
            ahead = self.points[min(i + 1, len(self.points) - 1)]
        return float(x), float(y), float(z), math.atan2(ahead[1] - y, ahead[0] - x)


class Centerlines:
    """Every layout's line, padded into arrays the numba projection reads."""

    def __init__(self, layout_list):
        self.lines = [Centerline(layout) for layout in layout_list]
        n = max(len(c.points) for c in self.lines)
        self.points = np.zeros((len(self.lines), n, 3))
        self.arc = np.zeros((len(self.lines), n))
        self.count = np.zeros(len(self.lines), np.int64)
        for k, c in enumerate(self.lines):
            m = len(c.points)
            self.points[k, :m] = c.points
            self.points[k, m:] = c.points[-1]
            self.arc[k, :m] = c.arc
            self.arc[k, m:] = c.arc[-1]
            self.count[k] = m
        self.lap_length = np.array([c.lap_length for c in self.lines])
        self.back = int(WINDOW_BACK_M / SAMPLE_SPACING_M)
        self.ahead = int(WINDOW_AHEAD_M / SAMPLE_SPACING_M)

    def index_at(self, lay, s):
        """Index nearest arc length s on each car's line (vectorized)."""
        out = np.empty(len(lay), np.int64)
        for k, (L, sk) in enumerate(zip(lay, s)):
            out[k] = self.lines[L].index_at(sk)
        return out

    def project(self, lay, index, x, y, z):
        """New (index, arc length, distance off the line) per car."""
        return _project(
            self.points,
            self.arc,
            self.count,
            lay,
            index,
            x,
            y,
            z,
            self.back,
            self.ahead,
        )


@nb.njit(cache=True)
def _project(points, arc, count, lay, index, x, y, z, back, ahead):
    n = index.shape[0]
    out_i = np.empty(n, np.int64)
    out_s = np.empty(n)
    out_d = np.empty(n)
    for k in range(n):
        L = lay[k]
        lo = max(0, index[k] - back)
        hi = min(count[L] - 1, index[k] + ahead)
        best = lo
        best_c = 1e18
        for i in range(lo, hi + 1):
            dx = x[k] - points[L, i, 0]
            dy = y[k] - points[L, i, 1]
            dz = (z[k] - points[L, i, 2]) * Z_WEIGHT
            c = dx * dx + dy * dy + dz * dz
            if c < best_c:
                best_c = c
                best = i
        out_i[k] = best
        out_s[k] = arc[L, best]
        dx = x[k] - points[L, best, 0]
        dy = y[k] - points[L, best, 1]
        out_d[k] = math.sqrt(dx * dx + dy * dy)
    return out_i, out_s, out_d


def _in_bucket_section(x, y) -> bool:
    return -0.66 <= x <= 3.01 and -4.63 <= y <= -0.88


def main() -> int:
    """Check every layout's line clears the walls and runs monotone."""
    import course_model

    seeds = layouts.TRAIN_SEEDS + layouts.HELDOUT_SEEDS
    model = course_model.CourseModel(seeds)
    SUP, OBS, _ = model.tables()
    failures = 0
    for lay, seed in enumerate(seeds):
        line = Centerline(model.layouts[lay])
        worst = (9.0, None)
        for x, y, z in line.points:
            zs, _ = course_model.support_below(SUP, lay, x, y, z + 0.08)
            # A wall at car-body height within 0.10 m of the line fails it.
            # Buckets are excused: they stand anywhere in their section and
            # the line there is a progress coordinate, not a path round them.
            for r in (0.10, 0.20):
                for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                    px, py = x + r * math.cos(a), y + r * math.sin(a)
                    if _in_bucket_section(px, py):
                        continue
                    if course_model.obstacle_overlap(
                        OBS, lay, px, py, zs + 0.05, zs + 0.15
                    ):
                        if r < worst[0]:
                            worst = (r, (round(x, 2), round(y, 2), round(z, 2)))
        ok = worst[1] is None or worst[0] > 0.10
        failures += not ok
        print(
            f"seed {seed}: lap {line.lap_length:6.2f} m, {model.layouts[lay]['gap_bale']}"
            + (
                ""
                if worst[1] is None
                else f", wall within {worst[0]:.2f} m at {worst[1]}"
            )
            + ("" if ok else "  FAIL")
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
