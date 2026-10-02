#!/usr/bin/env python3
"""Write routes/obstacle.yaml and routes/speed.yaml, Plan Z's route hints.

A route is the course's closed loop as points 0.1 m apart in the world frame
of the simulation (the frame the drawing was laid out in), the pose the car is
parked at, and a section name per point.  In a walled lane the driver slides
it to the middle of the walls it sees and drives that; where the course has
no lane (the Wide Section, the buckets) the route names only the ways out and
a box the search is kept inside.

    python3 make_routes.py            # rewrite both files
    python3 make_routes.py --check    # fail if either file is stale

Sources, so the routes cannot drift from the courses:

    obstacle   rl/obstacleRacer/centerline.py's hand line for the layout the
               drawing shows, labeled with the obstacle-course-regions skill,
               then set down the middle of the course's static walls
               (rl/obstacleRacer/course_model.py) with its bends opened out
               to what the car can turn, and through the middle of each
               hoop's travel (rl/obstacleRacer/layouts.py)
    speed      jetson/cfr_arduino_bridge/config/speed_course_path.json

Needs rl/obstacleRacer's Python environment (numba); the driver itself reads
only the YAML.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = REPO / "jetson" / "cfr_arduino_bridge"
SPACING = 0.10


def start_pose(world):
    """(x, y, yaw) the simulated car is parked at in a course's world."""
    text = (PACKAGE / "worlds" / world).read_text()
    match = re.search(r'<model name="slash">\s*<pose>([^<]+)</pose>', text)
    x, y, _, _, _, yaw = (float(v) for v in match.group(1).split())
    return [round(x, 4), round(y, 4), round(yaw, 4)]


def resample(xy, spacing=SPACING):
    """A closed polyline at even spacing, without its repeated end point."""
    xy = np.asarray(xy, float)
    closed = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(round(arc[-1] / spacing))
    at = np.arange(n) * arc[-1] / n
    return np.c_[np.interp(at, arc, closed[:, 0]), np.interp(at, arc, closed[:, 1])]


# The hand line through a walled lane is roughly placed and has corners; the
# driver follows the route's own shape there, so it is made the line a car can
# drive: down the middle of the walls the drawing gives, corners rounded, and
# bends no tighter than the car turns where there is room to open them out.
LANE_REACH = 1.2  # m either side a wall is looked for
MIN_AIR = 0.33  # m kept between the line and a wall while opening a bend out
TURN_RADIUS = 1.0  # m: full lock is 0.89 with the tires' scrub
# Where the hand line is reworked: the lanes placed by eye, from the tunnel's
# exit to the Wide Section.  Before that it is generated from the drawing's
# own constants (ramp, deck, helix, tunnel) and is already the middle; after
# it the randomizer moves what stands there.
REWORKED = (
    "after_tunnel",
    "narrow_region",
    "after_narrow_region",
    "gravel_pit",
    "after_gravel_pit",
    "banked_turn",
    "after_banked_turn",
    "potholes",
    "after_potholes",
)


def _smooth_loop(xy, passes, fixed):
    for _ in range(passes):
        new = 0.25 * np.roll(xy, 1, 0) + 0.5 * xy + 0.25 * np.roll(xy, -1, 0)
        xy = np.where(fixed[:, None], xy, new)
    return xy


def _normals(xy, span=3):
    tangent = np.roll(xy, -span, 0) - np.roll(xy, span, 0)
    tangent /= np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-9)[:, None]
    return np.c_[-tangent[:, 1], tangent[:, 0]]


def _curvature(xy, span=3):
    before = xy - np.roll(xy, span, 0)
    after = np.roll(xy, -span, 0) - xy
    turn = np.arctan2(
        before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0],
        before[:, 0] * after[:, 0] + before[:, 1] * after[:, 1],
    )
    return turn / (span * SPACING)


def drivable(xy, z, fixed, model):
    """The hand line made drivable against the course's static collisions."""
    import course_model

    SUP, OBS, _ = model.tables()

    def wall(x, y, height):
        top, _ = course_model.support_below(SUP, 0, x, y, height + 0.08)
        return course_model.obstacle_overlap(OBS, 0, x, y, top + 0.05, top + 0.15)

    def air(point, normal, height):
        """Distance to the first wall along `normal`, LANE_REACH if none."""
        for d in np.arange(0.02, LANE_REACH, 0.02):
            if wall(point[0] + d * normal[0], point[1] + d * normal[1], height):
                return d
        return LANE_REACH

    xy = xy.copy()
    for _ in range(4):
        normal = _normals(xy)
        shift = np.zeros(len(xy))
        for i in np.flatnonzero(~fixed):
            left, right = air(xy[i], normal[i], z[i]), air(xy[i], -normal[i], z[i])
            if left < LANE_REACH and right < LANE_REACH:
                shift[i] = 0.5 * (left - right)
            elif left < 0.45:
                shift[i] = left - 0.45
            elif right < 0.45:
                shift[i] = 0.45 - right
        # A shift that changes point to point would put kinks in the line.
        for _ in range(6):
            shift = 0.25 * np.roll(shift, 1) + 0.5 * shift + 0.25 * np.roll(shift, -1)
        xy = xy + np.clip(shift, -0.3, 0.3)[:, None] * normal * ~fixed[:, None]
        xy = _smooth_loop(xy, 6, fixed)
    # Open out the bends that are tighter than the car turns, outward along
    # each point's own normal, while there is air to move into.
    for _ in range(80):
        kappa = _curvature(xy)
        tight = (np.abs(kappa) > 1.0 / TURN_RADIUS) & ~fixed
        if not tight.any():
            break
        push = np.where(tight, -np.sign(kappa), 0.0)
        for _ in range(10):  # carry it into the approach and the exit
            push = 0.25 * np.roll(push, 1) + 0.5 * push + 0.25 * np.roll(push, -1)
        normal = _normals(xy)
        for i in np.flatnonzero((np.abs(push) > 0.02) & ~fixed):
            side = normal[i] * np.sign(push[i])
            if air(xy[i], side, z[i]) > MIN_AIR + 0.02:
                xy[i] = xy[i] + 0.02 * abs(push[i]) * side
        xy = _smooth_loop(xy, 2, fixed)
    return xy


def hoop_lane(xy, section, hoops):
    """The hand line through the hoops, moved to the middle of where each
    hoop can stand.

    It is drawn through the hoops' nominal places, and the randomizer puts
    each anywhere across its lane: the second up to 0.9 m from the line as
    drawn, with the line heading away from it.  From the middle of the lane
    no hoop is more than half its travel off, and the driver eases across
    to whichever side it is.
    """
    xy = xy.copy()
    first, second, third = (hoops[name] for name in hoops["names"])
    lane_y = 0.5 * (first["from"][1] + first["to"][1])
    lane_x = 0.5 * (third["from"][0] + third["to"][0])
    # Where the lane turns north for the third hoop: the line is stretched
    # about this x so that its northward leg is in the middle of that lane.
    turn_x = second["from"][0] - 0.6
    stretch = (lane_x - turn_x) / (third["nominal"][0] - turn_x)
    moved = np.zeros(len(xy), bool)
    for i, name in enumerate(section):
        if name not in ("hoops", "after_hoops"):
            continue
        x, y = xy[i]
        if x > turn_x and y < first["from"][1] and name == "hoops":
            xy[i, 1] = lane_y
            moved[i] = True
        elif x < turn_x:
            xy[i, 0] = turn_x + (x - turn_x) * stretch
            moved[i] = True
    # Round what that left of a corner, over a meter either side.
    near = moved.copy()
    for _ in range(10):
        near |= np.roll(near, 1) | np.roll(near, -1)
    return _smooth_loop(xy, 12, ~near)


def obstacle():
    sys.path.insert(0, str(REPO / "rl" / "obstacleRacer"))
    import centerline
    import course_model
    import env as env_module
    import layouts

    line = centerline.Centerline(None)
    names, labels = env_module.zone_labels([line])
    labels = labels[0]
    # The line starts at the parked car and ends on the timing line; the loop
    # is from its first pass over that line to its end.
    head = line.points[:40, :2]
    first = int(np.argmin(np.linalg.norm(head - line.points[-1, :2], axis=1)))
    xy = line.points[first:-1, :2]
    z = line.points[first:-1, 2]
    section = [names[k] for k in labels[first:-1]]
    section = ["start" if n == "after_car_wash" else n for n in section]
    fixed = np.array([name not in REWORKED for name in section])
    model = course_model.CourseModel(layouts.HELDOUT_SLOT_SEEDS[:1])
    xy = drivable(xy, z, fixed, model)
    xy = hoop_lane(xy, section, layouts.layout_spec()["hoops"])

    # Between the potholes and the bucket section's exit nothing is fixed: the
    # Wide Section's bales, the open entrance slot and the buckets all move.
    # The driver heads for the exit and finds its own way.
    exit_xy = np.asarray(centerline.BUCKET_EXIT[:2])
    first_open = section.index("wide_open_region")
    # The way out of the Wide Section is whichever of the east wall's four
    # slots the randomizer left open: make for just inside any of them.
    gaps = layouts.layout_spec()["gap_bales"]
    slots = [
        [
            round(gaps[name]["position"][0] - 0.45, 3),
            round(gaps[name]["position"][1], 3),
        ]
        for name in gaps["names"]
    ]
    last_open = int(np.argmin(np.linalg.norm(xy - exit_xy, axis=1)))
    # Until it is through a slot the way is inside the Wide Section's own
    # walls: x0, y0, x1, y1 of a box drawn FENCE_OUTSIDE m outside them, so
    # that a course built that far off the drawing still fits inside it.
    # Without it the search, to which unseen ground is free, takes the lane
    # the car came in by, or the back of the zig-zag wall, for the way round.
    fence = [
        round(WIDE_SECTION[0] - FENCE_OUTSIDE, 3),
        round(WIDE_SECTION[1] - FENCE_OUTSIDE, 3),
        round(WIDE_SECTION[2] + FENCE_OUTSIDE, 3),
        round(WIDE_SECTION[3] + FENCE_OUTSIDE, 3),
    ]
    return dict(
        course="obstacle",
        laps=2,
        start_pose=start_pose("obstacle_course.sdf"),
        spacing=SPACING,
        open_regions=[
            dict(first=first_open, last=last_open, stages=[slots], fences=[fence])
        ],
        points=[[round(float(x), 3), round(float(y), 3)] for x, y in xy],
        sections=section,
    )


# The inside faces of the Wide Section's walls, m, world frame: the slotted
# wall on the west, the lane wall on the east, the bale rows north and south
# (read off the course model; the zig-zag wall and the entrance lane open off
# its south-west corner).
WIDE_SECTION = (3.5, -9.3, 6.3, -0.9)
FENCE_OUTSIDE = 0.35


def speed():
    path = json.loads((PACKAGE / "config" / "speed_course_path.json").read_text())
    xy = resample(np.c_[path["x"], path["y"]])
    # The path file runs against the driving direction if the parked car
    # points the other way along it.
    pose = start_pose("speed_course.sdf")
    i = int(np.argmin(np.hypot(xy[:, 0] - pose[0], xy[:, 1] - pose[1])))
    j = (i + 5) % len(xy)
    along = math.atan2(xy[j, 1] - xy[i, 1], xy[j, 0] - xy[i, 0])
    if math.cos(along - pose[2]) < 0:
        xy = xy[::-1]
    return dict(
        course="speed",
        laps=3,
        start_pose=pose,
        spacing=SPACING,
        open_regions=[],
        points=[[round(float(x), 3), round(float(y), 3)] for x, y in xy],
        sections=["track"] * len(xy),
    )


def dump(route):
    """YAML with one point per line and run-length sections, to diff well."""
    runs = []
    for i, name in enumerate(route["sections"]):
        if not runs or runs[-1][0] != name:
            runs.append([name, i])
    head = {
        k: route[k] for k in ("course", "laps", "start_pose", "spacing", "open_regions")
    }
    lines = [
        "# Written by make_routes.py -- do not edit; see its header.",
        yaml.safe_dump(head, default_flow_style=None, sort_keys=False).rstrip(),
        "# [section, first point index]",
        "sections:",
        *[f"  - [{name}, {i}]" for name, i in runs],
        "points:",
        *[f"  - [{x}, {y}]" for x, y in route["points"]],
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = 0
    for build in (obstacle, speed):
        route = build()
        path = HERE / "routes" / f"{route['course']}.yaml"
        text = dump(route)
        if args.check:
            ok = path.exists() and path.read_text() == text
            print(f"{path.name}: {'ok' if ok else 'STALE -- rerun make_routes.py'}")
            stale += not ok
        else:
            path.write_bytes(text.encode())
            print(f"{path.name}: {len(route['points'])} points")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
