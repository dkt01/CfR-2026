"""Course-progress coordinate for the Obstacle Course.

This exists because `reward.forward_progress` -- displacement projected onto
the car's own heading -- is not a progress measure on a closed course. It
pays the same for driving the lap and for driving it backwards, and it pays
for spinning on the spot. A policy trained against it has no gradient telling
it which way round the course goes.

What replaces it is arc length `s` along a fixed centerline through the
course, so progress is "how much further round the lap are you" and going the
wrong way is negative. Because the reward uses the *difference* in `s`, the
per-step term telescopes over an episode to `s_end - s_start`: the total is
path-independent, so there is no way to farm it by oscillating, and the
centerline only has to be roughly right, not an optimal racing line.

`s` is privileged information: it is derived from the car's ground-truth
pose, which the physical car does not have. That is fine, and deliberate --
this is only ever used to compute the *reward*, which exists solely during
training in simulation. The observation the policy acts on stays
sensor-derived (see `obstacle_env._build_observation`). Training against a
signal the deployed robot cannot see is standard practice; acting on one is
not.

The centerline is a polyline through hand-picked anchors that follow the
lane, in driving order, with the helical ramp generated analytically from
`generate_obstacle_course.py`'s own constants. `python obstacle_course_path.py`
checks every sampled point against the bale walls in the world SDF, so an
anchor that strays into a wall is a failing check rather than a silent
mis-measurement.
"""

from __future__ import annotations

import math

# Straight-line anchors along the lane, in driving order, as (x, y, z) world
# metres. Derived from generate_obstacle_course.py's rect constants (via its
# to_world half-turn about the start line) and checked against the bale walls
# by this module's __main__. The helical ramp is not listed: it is generated
# between DECK_END and TUNNEL_MOUTH by _helix_points below.
FOOT = 0.3048
INCH = FOOT / 12.0

# Helix, straight out of generate_obstacle_course.py (DXF feet -> world via
# the same half turn). The 270 degree sweep starts at the bridge deck's east
# end and finishes on the helix centre's own y, which is the tunnel mouth.
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

# Everything after the helix, from the tunnel mouth round to the finish line.
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
    (4.55, -2.60, 0.0),
    # The corridor up the bucket section's east side is closed off above
    # y ~ -1.75, so the turn west has to be made below that, not at the top.
    (4.35, -2.35, 0.0),
    (4.00, -2.10, 0.0),
    (3.85, -1.75, 0.0),
    (3.40, -1.55, 0.0),
    (2.50, -1.48, 0.0),  # into the bucket section
    (1.20, -1.55, 0.0),
    # The randomizer leaves a different one of the four gap bales out on
    # each draw, so this opening moves up to 2.4 m along the wall. The
    # centerline takes the drawing's own default; a draw that opens a
    # different slot costs a short detour, which `s` absorbs because the
    # reward only ever reads its difference.
    (-0.88, -1.81, 0.0),  # the gap in the gap-bale wall
    (-2.40, -1.88, 0.0),
    (-3.94, -1.95, 0.0),  # hoop_0
    (-5.20, -1.95, 0.0),
    (-6.37, -1.95, 0.0),  # hoop_1
    (-7.60, -1.60, 0.0),
    (-8.20, -0.80, 0.0),
    (-8.13, 0.15, 0.0),  # hoop_2
    # North of the stub wall east of hoop_2 before turning back along the
    # start straight.
    (-8.05, 0.75, 0.0),
    (-6.60, 0.75, 0.0),
    (-5.60, 0.45, 0.0),
    (-4.09, 0.28, 0.0),  # car wash, west end
    (-2.87, 0.22, 0.0),
    (-1.65, 0.18, 0.0),  # car wash, east end
    (0.00, 0.10, 0.0),  # finish line, back where it started
]

SAMPLE_SPACING_M = 0.10


def _helix_points(count: int = 36) -> list[tuple[float, float, float]]:
    """The 270 degree spiral down from the deck to the tunnel mouth.

    Endpoints are dropped: DECK_END and TUNNEL_MOUTH are already the last
    anchor before and the first anchor after, so emitting them again would
    put a zero-length segment in the polyline.
    """
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


def _densify(
    anchors: list[tuple[float, float, float]], spacing: float
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = [anchors[0]]
    for (x0, y0, z0), (x1, y1, z1) in zip(anchors, anchors[1:]):
        span = math.dist((x0, y0, z0), (x1, y1, z1))
        steps = max(1, int(span / spacing))
        for step in range(1, steps + 1):
            t = step / steps
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t))
    return out


def build_centerline(
    spacing: float = SAMPLE_SPACING_M,
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """(points, cumulative arc length) for one lap, start line to start line."""
    anchors = BEFORE_HELIX + _helix_points() + AFTER_HELIX
    points = _densify(anchors, spacing)
    arc = [0.0]
    for previous, current in zip(points, points[1:]):
        arc.append(arc[-1] + math.dist(previous, current))
    return points, arc


# Vertical separation counts for more than horizontal when deciding which
# part of the course a pose belongs to: the bridge deck passes directly over
# the tunnel, so (x, y) alone is genuinely ambiguous there and only z
# separates them.
Z_WEIGHT = 4.0


class CourseProgress:
    """Tracks arc length round the lap, continuously.

    Nearest-point projection is searched in a window around the last match
    rather than globally. That is what makes the start line unambiguous --
    the lap begins and ends on the same spot, so a global search cannot tell
    `s = 0` from `s = lap_length` -- and it keeps the per-step cost constant.
    """

    def __init__(self, spacing: float = SAMPLE_SPACING_M, window_m: float = 6.0):
        self.points, self.arc = build_centerline(spacing)
        self.lap_length = self.arc[-1]
        self.window = max(2, int(window_m / spacing))
        self._index = 0
        # Where the helical ramp runs, so the curriculum can deal starts on
        # the straight ramp and the deck (both raised, both simple) while
        # staying off the spiral, which is 1.2 m in radius with a drop
        # either side and would need the spawn height to be right to within
        # a few centimetres.
        self.helix_start_s = self.arc[self._nearest_index(DECK_END)]
        self.helix_end_s = self.arc[self._nearest_index(TUNNEL_MOUTH)]

    def _nearest_index(self, point: tuple[float, float, float]) -> int:
        return min(
            range(len(self.points)),
            key=lambda i: math.dist(self.points[i], point),
        )

    def reset(self, start_s: float = 0.0) -> float:
        """Rewind to the start line, or to `start_s` metres round the lap."""
        self._index = self.index_at(start_s)
        return self.arc[self._index]

    def index_at(self, s: float) -> int:
        """Index of the centerline sample nearest arc length `s`."""
        s = min(max(s, 0.0), self.lap_length)
        return min(range(len(self.arc)), key=lambda i: abs(self.arc[i] - s))

    def pose_at(self, s: float) -> tuple[float, float, float, float]:
        """(x, y, z, yaw) on the centerline at arc length `s`.

        Yaw is the forward tangent, so a car placed here is already pointing
        the way round the course. Used to start training episodes part-way
        round the lap (see ObstacleCourseEnv's `start_anywhere_prob`): the
        centerline is validated to clear the bale walls by > 0.15 m at every
        ground-level sample, which is what makes it safe to spawn on.
        """
        index = self.index_at(s)
        x, y, z = self.points[index]
        # Aim at a point a little way down the course rather than along the
        # local tangent. At a sharp corner the tangent points across the
        # turn -- measured up to 69 degrees off the direction the car
        # actually has to travel -- which is a bad way to be dealt into one.
        ahead = self.points[self.index_at(min(s + 0.75, self.lap_length))]
        if (ahead[0] - x) ** 2 + (ahead[1] - y) ** 2 < 1e-9:
            ahead = self.points[min(index + 1, len(self.points) - 1)]
        yaw = math.atan2(ahead[1] - y, ahead[0] - x)
        return x, y, z, yaw

    def safe_start_arcs(
        self,
        sdf_path: str,
        spacing: float = 0.25,
        margin: float = 0.15,
        probe_m: float = 1.0,
        finish_margin_m: float = 8.0,
    ) -> list[float]:
        """Arc lengths it is safe to deal a training episode in at.

        Checked, not assumed: the point itself must clear every bale wall by
        the car's half-width, and so must a point `probe_m` straight ahead
        along the spawn heading -- a pose aimed across a tight corner passes
        the first test and fails the second. Elevated samples are excluded
        (see ground_level_span) and so is the run-in to the finish, so a
        dealt start can never be a free lap.

        Only the *static* bale walls are known here; buckets and hoops move
        every layout, so a dealt start can still land on a bucket. That
        costs a truncated episode and nothing else, which is why it is worth
        living with rather than re-checking 10 layouts here.
        """
        boxes = _wall_boxes(sdf_path)
        limit = max(0.0, self.lap_length - finish_margin_m)
        arcs: list[float] = []
        # Per point, not per span: the ground-level part of the course is
        # not one contiguous stretch. Taking a single span after the last
        # elevated sample threw away the start straight (s 0 -> 3.4), which
        # is the run-up to the ramp -- so the ramp was only ever practised
        # by the episodes that started exactly on the line.
        for (x, y, z), s in zip(self.points, self.arc):
            if s > limit:
                break
            if arcs and s - arcs[-1] < spacing:
                continue
            # Raised ground is fine to be dealt onto now that the teleport
            # takes a height -- the ramp and the deck are a straight 11%
            # climb and a flat bridge. The helix is not: a 1.2 m spiral
            # with a drop either side, where being a few centimetres out
            # puts the car over the edge. It still gets practised, by the
            # episodes dealt onto the deck that lead straight into it.
            if self.helix_start_s < s < self.helix_end_s:
                continue
            if z > 0.05 and not s <= self.helix_start_s:
                continue
            _, _, _, yaw = self.pose_at(s)
            probe_x = x + probe_m * math.cos(yaw)
            probe_y = y + probe_m * math.sin(yaw)
            if not any(
                _inside(box, x, y, margin) or _inside(box, probe_x, probe_y, margin)
                for box in boxes
            ):
                arcs.append(s)
        return arcs

    def ground_level_span(self, tolerance: float = 0.05) -> tuple[float, float]:
        """Arc-length range after the last elevated (ramp/deck/helix) sample.

        Spawning is only safe where the centerline sits on the ground: the
        elevated section is a narrow deck with a drop either side, and the
        car would have to be placed on it to within a few centimetres.
        """
        last_elevated = 0
        for index, (_, _, z) in enumerate(self.points):
            if z > tolerance:
                last_elevated = index
        return self.arc[min(last_elevated + 1, len(self.arc) - 1)], self.lap_length

    def _cost(self, index: int, x: float, y: float, z: float) -> float:
        px, py, pz = self.points[index]
        dz = (z - pz) * Z_WEIGHT
        return (x - px) ** 2 + (y - py) ** 2 + dz * dz

    def update(self, x: float, y: float, z: float = 0.0) -> float:
        """Arc length of the closest centerline point, searched forward-biased.

        The window reaches further ahead than behind so a fast car cannot
        outrun it, while still allowing the backwards motion that has to stay
        measurable for wrong-way driving to score negative.
        """
        low = max(0, self._index - self.window // 2)
        high = min(len(self.points) - 1, self._index + self.window)
        best = min(range(low, high + 1), key=lambda i: self._cost(i, x, y, z))
        self._index = best
        return self.arc[best]


def _wall_boxes(sdf_path: str) -> list[tuple[float, float, float, float, float]]:
    """(x, y, yaw, size_x, size_y) for every bale wall box in the world."""
    import xml.etree.ElementTree as ElementTree

    boxes = []
    root = ElementTree.parse(sdf_path).getroot()
    for model in root.iter("model"):
        name = model.get("name") or ""
        if name != "course_bales" and not name.startswith("gap_bale"):
            continue
        pose = model.find("pose")
        mx, my, myaw = 0.0, 0.0, 0.0
        if pose is not None and pose.text:
            values = [float(v) for v in pose.text.split()]
            mx, my, myaw = values[0], values[1], values[5]
        for collision in model.iter("collision"):
            size = collision.find(".//box/size")
            if size is None:
                continue
            cx, cy, cyaw = 0.0, 0.0, 0.0
            child = collision.find("pose")
            if child is not None and child.text:
                values = [float(v) for v in child.text.split()]
                cx, cy, cyaw = values[0], values[1], values[5]
            sx, sy, _ = (float(v) for v in size.text.split())
            cos_yaw, sin_yaw = math.cos(myaw), math.sin(myaw)
            boxes.append(
                (
                    mx + cos_yaw * cx - sin_yaw * cy,
                    my + sin_yaw * cx + cos_yaw * cy,
                    myaw + cyaw,
                    sx,
                    sy,
                )
            )
    return boxes


def wall_boxes(sdf_path: str) -> list[tuple[float, float, float, float, float]]:
    """Public handle on the bale walls, for diagnostics."""
    return _wall_boxes(sdf_path)


def nearest_wall_distance(boxes, x: float, y: float) -> float:
    """Ground-truth distance from (x, y) to the nearest bale wall.

    Privileged and strictly diagnostic -- never an observation and never a
    reward term. It exists to answer one question the car's own scan cannot:
    the scan looks 110 degrees forward, so a wall *alongside* the car is
    invisible to it, and a car wedged against one reports a clear view while
    going nowhere. Comparing this against the scan's own min_clearance says
    whether that is what keeps happening.
    """
    best = float("inf")
    for bx, by, byaw, sx, sy in boxes:
        dx, dy = x - bx, y - by
        cos_yaw, sin_yaw = math.cos(byaw), math.sin(byaw)
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
        # Distance from a point to an axis-aligned rectangle, in its frame.
        gap_x = max(abs(local_x) - sx / 2.0, 0.0)
        gap_y = max(abs(local_y) - sy / 2.0, 0.0)
        best = min(best, math.hypot(gap_x, gap_y))
    return best


def _inside(box, x: float, y: float, margin: float) -> bool:
    bx, by, byaw, sx, sy = box
    dx, dy = x - bx, y - by
    cos_yaw, sin_yaw = math.cos(byaw), math.sin(byaw)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    return abs(local_x) <= sx / 2 + margin and abs(local_y) <= sy / 2 + margin


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdf",
        default=str(repo_root / "jetson/cfr_arduino_bridge/worlds/obstacle_course.sdf"),
    )
    # Half the car's width. A centerline point closer than this to a bale
    # means the anchor either side of it is routed through a wall.
    parser.add_argument("--margin", type=float, default=0.15)
    args = parser.parse_args()

    points, arc = build_centerline()
    print(f"centerline: {len(points)} points, lap length {arc[-1]:.2f} m")

    # The elevated run has no bale walls to check against -- it is bounded by
    # the deck's own rails and the helix's kerbs, which are separate models.
    ground = [(i, p) for i, p in enumerate(points) if p[2] < 0.05]
    print(f"{len(ground)} ground-level points to check against the bale walls")

    boxes = _wall_boxes(args.sdf)
    print(f"{len(boxes)} bale wall boxes in {Path(args.sdf).name}")

    bad = []
    for index, (x, y, _) in ground:
        for box in boxes:
            if _inside(box, x, y, args.margin):
                bad.append((index, x, y, arc[index]))
                break

    if bad:
        print(
            f"\nFAIL: {len(bad)} centerline points are inside a wall (+{args.margin} m):"
        )
        for index, x, y, s in bad[:: max(1, len(bad) // 20)]:
            print(f"  point {index:5d}  ({x:7.2f}, {y:7.2f})  s={s:7.2f}")
        raise SystemExit(1)

    print(f"\nOK: every ground-level point clears the bale walls by > {args.margin} m")

    progress = CourseProgress()
    progress.reset()
    last = 0.0
    regressions = 0
    for x, y, z in progress.points:
        s = progress.update(x, y, z)
        if s < last - 1e-9:
            regressions += 1
        last = s
    print(
        f"tracking its own centerline: final s = {last:.2f} m, {regressions} regressions"
    )
    if regressions or abs(last - progress.lap_length) > 0.5:
        raise SystemExit("FAIL: progress tracking is not monotone along the centerline")
    print("OK: progress is monotone from the start line back to it")
