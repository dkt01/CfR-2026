#!/usr/bin/env python3
"""Generate the straw-bale layout and start signal in the speed-course SDF.

The rest of ``speed_course.sdf`` is maintained by hand; this rewrites the two
blocks that are derived from something else -- the bales from the site-layout
DXF, and the start signal from the position chosen below.  Both are delimited
in the SDF so the hand written parts around them survive.

Also writes ``config/speed_course_layout.yaml``, so the signal's pose reaches
``obstacle_randomizer_node`` from the same place the world gets it and the two
cannot disagree.

The DXF argument is optional, and the signal does not need it.  It is optional
because ``dxf_speed_bales`` below does not read every revision of the drawing:
it picks the Speed Course out by an x range, and in "site layout 2" the two
courses are separated by y instead, so it selects 82 bales there rather than
202.  Selecting by y instead does find 202 -- but they are a different 202,
laid out up to 39 m away and rotated a quarter turn from the ones committed in
the world, so the committed Speed Course came from a different revision again.
Until that is reconciled, regenerating bales from a drawing that does not match
them would silently replace the course, so the bale layout is only re-derived
when a DXF is actually passed.  The bale *block* is rewritten either way,
because the bale the start signal's board now stands in has to move along the
wall to clear it -- without a DXF it is read out of the world, moved, and
written back, which leaves the other 201 exactly as they were.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import start_signal

INCH = 0.0254
BALE_LENGTH = 36 * INCH
BALE_WIDTH = 18 * INCH
BALE_HEIGHT = 14 * INCH
BALE_Z = BALE_HEIGHT / 2
FOOT = 0.3048
PACKAGE = Path(__file__).parents[1] / "cfr_arduino_bridge"
WORLD_FILE = PACKAGE / "worlds/speed_course.sdf"
LAYOUT_FILE = PACKAGE / "config/speed_course_layout.yaml"

# Where the car waits, matching the hand written <model name="slash"> pose,
# and the direction it drives away in -- the yaw in that same pose.
VEHICLE_START = (20.15, 4.76)
HEADING = math.pi

# The start/finish line, on the lane centreline.  Nothing in this world marks
# it: the bales and the car are all there is, and the drawing only says the car
# starts behind the line.  So it is taken to sit the same 0.70 m ahead of the
# waiting car as the obstacle course's line does, which keeps the two start
# sections identical from the driver's seat.
LINE_AHEAD_OF_CAR = 0.70
START_LINE = (
    VEHICLE_START[0] + LINE_AHEAD_OF_CAR * math.cos(HEADING),
    VEHICLE_START[1] + LINE_AHEAD_OF_CAR * math.sin(HEADING),
)

# Distance from the lane centreline to the inner edge of the bale border on the
# car's left, which is the side the drawing stands the signal on.  Measured off
# the committed wall: its bales sit at y = 4.0673 and are 18 in deep, so their
# inner faces stand at 4.2959, which is 0.464 m from the car's line.
# check_signal_sightline.py re-measures it against the bales in the world.
LANE_EDGE = 0.464

# Where the start signal stands, in world metres: three bales down the wall
# from the start line and in line with the border's inner edge, the same as on
# the obstacle course.  See scripts/start_signal.py.
SIGNAL_POSITION = start_signal.position(START_LINE, HEADING, LANE_EDGE)


def bale_xml(index: int, x: float, y: float, yaw: float) -> str:
    pose = f"{x:.4f} {y:.4f} {BALE_Z:.4f} 0 0 {yaw:.5f}"
    geometry = (
        f"<box><size>{BALE_LENGTH:.4f} {BALE_WIDTH:.4f} {BALE_HEIGHT:.4f}</size></box>"
    )
    material = "<material><diffuse>0.72 0.48 0.12 1</diffuse><specular>0.08 0.05 0.01 1</specular></material>"
    return (
        f'      <collision name="bale_{index}_collision"><pose>{pose}</pose><geometry>{geometry}</geometry></collision>\n'
        f'      <visual name="bale_{index}_visual"><pose>{pose}</pose><geometry>{geometry}</geometry>{material}</visual>'
    )


def dxf_speed_bales(dxf_file: Path) -> list[tuple[float, float, float]]:
    """Extract Speed Course bales, excluding the separate Obstacle Course."""
    lines = dxf_file.read_text(errors="replace").splitlines()
    pairs = [
        (lines[index].strip(), lines[index + 1].strip())
        for index in range(0, len(lines) - 1, 2)
    ]
    entities: list[tuple[str, list[tuple[str, str]]]] = []
    entity: tuple[str, list[tuple[str, str]]] | None = None
    for code, value in pairs:
        if code == "0":
            if entity:
                entities.append(entity)
            entity = (value, [])
        elif entity:
            entity[1].append((code, value))
    if entity:
        entities.append(entity)

    bales: list[tuple[float, float, float]] = []
    for entity_type, tags in entities:
        if (
            entity_type != "HATCH"
            or next((value for code, value in tags if code == "8"), "") != "STRAW_BALES"
        ):
            continue
        boundary_start = next(
            (index for index, (code, _) in enumerate(tags) if code == "91"), None
        )
        if boundary_start is None:
            continue
        boundary = tags[boundary_start + 1 :]
        points: list[tuple[float, float]] = []
        x_value: float | None = None
        end_x: float | None = None
        end_point: tuple[float, float] | None = None
        for code, value in boundary:
            if code == "10":
                x_value = float(value)
            elif code == "20" and x_value is not None:
                points.append((x_value, float(value)))
                x_value = None
            elif code == "11" and end_x is None:
                end_x = float(value)
            elif code == "21" and end_x is not None and end_point is None:
                end_point = (end_x, float(value))
        if not points or end_point is None:
            continue
        min_x, max_x = min(x for x, _ in points), max(x for x, _ in points)
        min_y, max_y = min(y for _, y in points), max(y for _, y in points)
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        # The DXF Speed Course is the 125..174 ft x-coordinate component. The
        # obstacle course uses the separate 100..124 ft component.
        if center_x < 125 or not 50 < center_y < 200:
            continue
        start_x, start_y = points[0]
        dxf_yaw = math.atan2(end_point[1] - start_y, end_point[0] - start_x)
        bales.append((center_x, center_y, dxf_yaw))

    if len(bales) != 202:
        raise RuntimeError(f"Expected 202 Speed Course bales, found {len(bales)}")

    min_course_y = min(y for _, y, _ in bales)
    center_course_x = (min(x for x, _, _ in bales) + max(x for x, _, _ in bales)) / 2
    # DXF Y is the long course axis. Rotate it into Gazebo X and center the
    # narrower DXF X axis around Gazebo Y.
    return [
        ((y - min_course_y) * FOOT, (x - center_course_x) * FOOT, -yaw - math.pi / 2)
        for x, y, yaw in bales
    ]


def world_bales(contents: str) -> list[tuple[float, float, float]]:
    """The bale poses already written into the world, in world metres.

    The bale block is only re-derived from the drawing when a DXF is passed --
    see the module docstring -- but the bales the start signal displaces move
    either way, so the committed ones have to be readable back out.
    """
    pattern = (
        r'<collision name="bale_\d+_collision"><pose>'
        r"(-?[\d.]+) (-?[\d.]+) -?[\d.]+ 0 0 (-?[\d.]+)</pose>"
    )
    return [
        (float(x), float(y), float(yaw)) for x, y, yaw in re.findall(pattern, contents)
    ]


def build_course(bales: list[tuple[float, float, float]]) -> str:
    lines = [
        "    <!-- 135 ft by 47 ft speed course, built from individual 14 x 18 x 36 in straw bales. -->",
        '    <model name="course_bales"><static>true</static><link name="bales">',
    ]
    lines.extend(bale_xml(index, *bale) for index, bale in enumerate(bales))
    lines.append("    </link></model>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Gazebo Speed Course bales and start signal"
    )
    parser.add_argument(
        "dxf_file",
        type=Path,
        nargs="?",
        help=(
            "site-layout DXF; re-derives the straw-bale layout when given. "
            "The start signal, and the bales it displaces, are rewritten "
            "either way."
        ),
    )
    args = parser.parse_args()
    contents = WORLD_FILE.read_text()

    # From the drawing when one is passed, and otherwise straight back out of
    # the world.  The block is rewritten either way, because the bale the
    # signal's board stands in has to move along the wall to clear it.
    bales = (
        dxf_speed_bales(args.dxf_file)
        if args.dxf_file is not None
        else world_bales(contents)
    )
    cleared = start_signal.clear_bales(
        bales, SIGNAL_POSITION, HEADING, (BALE_LENGTH, BALE_WIDTH)
    )
    shifted = sum(1 for before, after in zip(bales, cleared) if before != after)
    if shifted:
        print(f"moved {shifted} bale(s) along the wall to clear the start signal")

    replacement = build_course(cleared) + '\n\n    <model name="slash">'
    contents, replacements = re.subn(
        r"    <!-- (?:44\.7 m by 34\.5 m drawing area|135 ft by 47 ft speed course).*?    <model name=\"slash\">",
        replacement,
        contents,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise RuntimeError("Could not find the existing course-bale block")

    signal = (
        "    <!-- start signal: generated by generate_speed_course.py -->\n"
        + start_signal.models(SIGNAL_POSITION, HEADING)
        + "    <!-- end start signal -->"
    )
    contents, replacements = re.subn(
        r"    <!-- start signal: generated by generate_speed_course\.py -->.*?"
        r"    <!-- end start signal -->",
        lambda _: signal,
        contents,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise RuntimeError("Could not find the existing start-signal block")

    # newline="\n" on every write: the repository is LF throughout, and a
    # Windows dev host would otherwise rewrite the whole world file as CRLF.
    WORLD_FILE.write_text(contents, newline="\n")
    print(f"wrote {WORLD_FILE.relative_to(PACKAGE.parent)}")

    LAYOUT_FILE.write_text(
        start_signal.layout_file(
            "cfr_speed_course",
            SIGNAL_POSITION,
            HEADING,
            "generate_speed_course.py",
        ),
        newline="\n",
    )
    print(f"wrote {LAYOUT_FILE.relative_to(PACKAGE.parent)}")


if __name__ == "__main__":
    main()
