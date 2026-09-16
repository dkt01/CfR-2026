#!/usr/bin/env python3
"""Check that the start signal stands where the drawing puts it, in each world.

The drawing places the signal the same way on both courses: three bales down
the wall from the start line -- "approximately 8 ft" -- with the bales moved so
that it stands in line with the inner edge of the bale border.  The generators
derive that placement, and it is easy to break by nudging a number, so this
re-derives all of it from the finished worlds:

* It stands ``start_signal.SIGNAL_DISTANCE`` down the lane from the start line.
* Its board finishes flush with the border's inner edge -- neither reaching
  into the 32 in path nor standing off beyond the wall.
* It stands square across the path, rather than canted towards the point the
  car happens to wait at.
* No bale is left inside the board.  The generators slide the bales it
  displaces along their wall; this is what says they moved far enough.
* It is inside the camera's 110 degree field, and within its range.
* The sight line from the camera, only 8 in off the ground, clears the 14 in
  bale wall it grazes on the way to the arm 32 in up.

    ./scripts/check_signal_sightline.py                  # both worlds
    ./scripts/check_signal_sightline.py worlds/x.sdf     # one

Exits non-zero if a course fails, so it can be run after regenerating.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import start_signal

PACKAGE = Path(__file__).parents[1] / "cfr_arduino_bridge"

BALE_LENGTH = 0.9144
BALE_WIDTH = 0.4572
BALE_HEIGHT = 0.3556

# The ZED sits 0.315 m ahead of the chassis origin and 0.20 m up, and sees
# 110 degrees across.  Its far clip is 20 m.
CAMERA_FORWARD = 0.315
CAMERA_HEIGHT = 0.20
CAMERA_HALF_FOV = math.radians(55.0)
CAMERA_RANGE = 20.0

# The arms turn about a pivot 0.813 m up; the plate is 0.21 m tall, so its
# center is the height to aim the sight line at.
ARM_HEIGHT = 0.813

# How far ahead of the waiting car the start/finish line is.  The obstacle
# course's world origin is the line and its car sits at x = -0.70; the speed
# course has no marker and its generator adopts the same offset, so this one
# number places the line in both.
LINE_AHEAD_OF_CAR = 0.70

# How far the board's inner end may sit from the border's inner edge before
# "in line with" stops being true.  A square board lands on it to within the
# 0.1 mm the SDF rounds poses to, so the millimeters this allows are for the
# wall itself: the obstacle course's bales come from the DXF and sit about
# 2 mm wider than the drawing's nominal 32 in lane.
FLUSH_TOLERANCE = 0.01

# How far off square the board may stand.  Zero, to the same rounding: the
# generators derive the yaw from the path's heading, so anything else means
# one of them has gone back to aiming it at the car.
SQUARE_TOLERANCE = math.radians(0.5)

# How far the signal may sit from the drawing's distance down the lane.
DISTANCE_TOLERANCE = 0.05


def poses(text: str, pattern: str) -> list[list[float]]:
    return [
        [float(value) for value in match.group(1).split()]
        for match in re.finditer(pattern, text)
    ]


def footprint(pose, half_length, half_width):
    cos_yaw, sin_yaw = math.cos(pose[5]), math.sin(pose[5])
    corners = [
        (
            pose[0] + dx * cos_yaw - dy * sin_yaw,
            pose[1] + dx * sin_yaw + dy * cos_yaw,
        )
        for dx in (-half_length, half_length)
        for dy in (-half_width, half_width)
    ]
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return min(xs), max(xs), min(ys), max(ys)


def border_edge(bales, local, along: float, side: float) -> float | None:
    """How far off the lane centerline the bale border's inner edge is.

    Measured beside the signal -- bales within a meter of its position along
    the lane -- and on the signal's side of the centerline, so a course whose
    two walls sit at different widths is measured against the right one.
    """
    edges = []
    for bale in bales:
        corners = [local((x, y)) for x in bale[:2] for y in bale[2:]]
        if max(a for a, _ in corners) < along - 1.0:
            continue
        if min(a for a, _ in corners) > along + 1.0:
            continue
        same_side = [abs(across) for _, across in corners if across * side > 0]
        if same_side:
            edges.append(min(same_side))
    return min(edges) if edges else None


def check(world: Path) -> bool:
    text = world.read_text()

    bale_poses = poses(
        text, r'<visual name="bale_\d+_visual"><pose>([-\d.eE ]+)</pose>'
    )
    bales = [footprint(pose, BALE_LENGTH / 2, BALE_WIDTH / 2) for pose in bale_poses]
    vehicle = poses(text, r'<model name="slash">\s*<pose>([-\d.eE ]+)</pose>')
    signal = poses(
        text,
        r'<model name="start_signal_frame"><static>true</static><pose>([-\d.eE ]+)</pose>',
    )
    if not bales or not vehicle or not signal:
        print(f"{world.name}: no bales, vehicle or signal found")
        return False

    vehicle, signal = vehicle[0], signal[0]
    yaw = vehicle[5]
    forward = (math.cos(yaw), math.sin(yaw))
    left = (-forward[1], forward[0])
    camera = (
        vehicle[0] + CAMERA_FORWARD * forward[0],
        vehicle[1] + CAMERA_FORWARD * forward[1],
    )
    line = (
        vehicle[0] + LINE_AHEAD_OF_CAR * forward[0],
        vehicle[1] + LINE_AHEAD_OF_CAR * forward[1],
    )

    def local(point):
        """A point as (down the lane from the line, left of its centerline)."""
        offset = (point[0] - line[0], point[1] - line[1])
        return (
            offset[0] * forward[0] + offset[1] * forward[1],
            offset[0] * left[0] + offset[1] * left[1],
        )

    failures = []
    frame = (signal[0], signal[1], signal[5])
    frame_size = (start_signal.FRAME_WIDTH, start_signal.FRAME_DEPTH)

    # 1. No bale is left standing inside the board.  The board sits where the
    #    wall was, so this is the check that the generators' slide worked;
    #    the boxes are both tilted, so it is a real box overlap and not the
    #    axis-aligned approximation the sight line below is happy with.
    fouled = [
        index
        for index, pose in enumerate(bale_poses)
        if start_signal.boxes_overlap(
            (pose[0], pose[1], pose[5]), (BALE_LENGTH, BALE_WIDTH), frame, frame_size
        )
    ]
    if fouled:
        failures.append(
            f"the board still stands in bale(s) {', '.join(map(str, fouled))}"
        )

    # 2. It is where the drawing puts it: the right distance down the lane,
    #    with the board's inner end flush against the border's inner edge.
    along, across = local((signal[0], signal[1]))
    if abs(along - start_signal.SIGNAL_DISTANCE) > DISTANCE_TOLERANCE:
        failures.append(
            f"{along / start_signal.FOOT:.2f} ft down the lane, not the "
            f"{start_signal.SIGNAL_DISTANCE / start_signal.FOOT:.2f} ft the "
            "drawing annotates"
        )

    width_axis = (math.cos(signal[5]), math.sin(signal[5]))
    ends = [
        local(
            (
                signal[0] + sign * width_axis[0] * start_signal.FRAME_WIDTH / 2,
                signal[1] + sign * width_axis[1] * start_signal.FRAME_WIDTH / 2,
            )
        )
        for sign in (1, -1)
    ]
    # The end nearer the centerline is the one that has to line up with the
    # border; the other is out past the back of the wall.
    inner = min(abs(end[1]) for end in ends)
    edge = border_edge(bales, local, along, across)
    flush = None if edge is None else inner - edge
    if edge is None:
        failures.append("no bale beside the signal to measure the border against")
    elif flush < -FLUSH_TOLERANCE:
        failures.append(f"the board reaches {-flush * 1000:.0f} mm into the path")
    elif flush > FLUSH_TOLERANCE:
        failures.append(
            f"the board stands {flush * 1000:.0f} mm short of the border's inner edge"
        )

    # 3. It stands square across the path, which is what puts its width
    #    across the lane, its depth along it and its arms' sweep in a plane
    #    facing the car.
    square = (signal[5] - start_signal.yaw_across(yaw) + math.pi) % (
        2 * math.pi
    ) - math.pi
    if abs(square) > SQUARE_TOLERANCE:
        failures.append(
            f"the board is canted {math.degrees(square):+.1f} deg off square "
            "to the path"
        )

    # 4. It is inside the camera's field of view, and within its range.
    bearing = math.atan2(signal[1] - camera[1], signal[0] - camera[0]) - yaw
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
    distance = math.dist(camera, (signal[0], signal[1]))
    if abs(bearing) > CAMERA_HALF_FOV:
        failures.append(
            f"bearing {math.degrees(bearing):+.1f} deg is outside the "
            f"{math.degrees(CAMERA_HALF_FOV):.0f} deg half-field"
        )
    if distance > CAMERA_RANGE:
        failures.append(f"range {distance:.1f} m is beyond the camera's far clip")

    # 5. The sight line to the arm clears the bale walls in between.  Aimed at
    #    the middle of the board, which is the conservative end of the arm's
    #    travel: the arm itself hangs nearer the lane, over less wall.
    margin = None
    for step in range(1, 801):
        fraction = step / 800
        x = camera[0] + (signal[0] - camera[0]) * fraction
        y = camera[1] + (signal[1] - camera[1]) * fraction
        z = CAMERA_HEIGHT + (ARM_HEIGHT - CAMERA_HEIGHT) * fraction
        for bale in bales:
            if bale[0] <= x <= bale[1] and bale[2] <= y <= bale[3]:
                clearance = z - BALE_HEIGHT
                if margin is None or clearance < margin:
                    margin = clearance
    if margin is not None and margin <= 0:
        failures.append(f"sight line passes {-margin * 1000:.0f} mm below a bale top")

    crossing = (
        "crosses no bale"
        if margin is None
        else f"clears bales by {margin * 1000:.0f} mm"
    )
    status = "FAIL" if failures else "ok"
    print(
        f"{world.name:24s} {status:4s}  {along / start_signal.FOOT:.1f} ft down the "
        f"lane, board {'n/a' if flush is None else f'{flush * 1000:+.0f} mm'} off the "
        f"border's edge, {math.degrees(square):+.1f} deg off square, "
        f"bearing {math.degrees(bearing):+.1f} deg, "
        f"range {distance:.2f} m, {crossing}"
    )
    for failure in failures:
        print(f"    - {failure}")
    return not failures


def main() -> int:
    if len(sys.argv) > 1:
        worlds = [Path(argument) for argument in sys.argv[1:]]
    else:
        worlds = sorted((PACKAGE / "worlds").glob("*_course.sdf"))
    return 0 if all([check(world) for world in worlds]) else 1


if __name__ == "__main__":
    sys.exit(main())
