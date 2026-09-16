#!/usr/bin/env python3
"""Check that the car can actually see the start signal, in each world.

The signal's position is chosen by hand in each course generator, subject to
constraints that are easy to state and easy to break by nudging a number: it
has to stand clear of the bales, sit inside the camera's field of view, and be
visible over the bale wall between it and a camera only 8 in off the ground.
Nothing in the world file records that those hold, so this re-derives them
from the geometry.

    ./scripts/check_signal_sightline.py                  # both worlds
    ./scripts/check_signal_sightline.py worlds/x.sdf     # one

Exits non-zero if a course fails, so it can be run after regenerating.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

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
# centre is the height to aim the sight line at.
ARM_HEIGHT = 0.813

# The frame is 0.813 m across and 0.10 m deep.
FRAME_WIDTH = 0.813
FRAME_DEPTH = 0.10


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


def check(world: Path) -> bool:
    text = world.read_text()

    bales = [
        footprint(pose, BALE_LENGTH / 2, BALE_WIDTH / 2)
        for pose in poses(
            text, r'<visual name="bale_\d+_visual"><pose>([-\d.eE ]+)</pose>'
        )
    ]
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
    camera = (
        vehicle[0] + CAMERA_FORWARD * math.cos(yaw),
        vehicle[1] + CAMERA_FORWARD * math.sin(yaw),
    )

    failures = []

    # 1. The frame stands clear of every bale.
    frame = footprint(signal, FRAME_WIDTH / 2, FRAME_DEPTH / 2)
    fouled = [
        bale
        for bale in bales
        if bale[1] > frame[0]
        and bale[0] < frame[1]
        and bale[3] > frame[2]
        and bale[2] < frame[3]
    ]
    if fouled:
        failures.append(f"frame overlaps {len(fouled)} bale(s)")

    # 2. It is inside the camera's field of view, and within its range.
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

    # 3. The sight line to the arm clears the bale walls in between.
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
        "clears no bale"
        if margin is None
        else f"clears bales by {margin * 1000:.0f} mm"
    )
    status = "FAIL" if failures else "ok"
    print(
        f"{world.name:24s} {status:4s}  signal at ({signal[0]:.2f}, {signal[1]:.2f}), "
        f"bearing {math.degrees(bearing):+.1f} deg, range {distance:.2f} m, {crossing}"
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
