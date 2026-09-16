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
them would silently replace the course, so the bale block is only rewritten
when a DXF is actually passed.
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

# Where the car waits, matching the hand written <model name="slash"> pose.
VEHICLE_START = (20.15, 4.76)

# Where the start signal stands, in world metres.  Outside the outer bale wall
# on the start straight, 4.1 m ahead of the car at a bearing of 20 degrees,
# which clears the 14 in wall between the two by 54 mm.  scripts/start_signal.py
# explains the three constraints; check_signal_sightline.py re-checks them.
SIGNAL_POSITION = (16.00, 3.40)


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


def build_course(dxf_file: Path) -> str:
    bales = dxf_speed_bales(dxf_file)
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
            "site-layout DXF; rewrites the straw-bale block when given. "
            "The start signal is rewritten either way."
        ),
    )
    args = parser.parse_args()
    contents = WORLD_FILE.read_text()

    if args.dxf_file is not None:
        replacement = build_course(args.dxf_file) + '\n\n    <model name="slash">'
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
        + start_signal.models(SIGNAL_POSITION, VEHICLE_START)
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
            VEHICLE_START,
            "generate_speed_course.py",
        ),
        newline="\n",
    )
    print(f"wrote {LAYOUT_FILE.relative_to(PACKAGE.parent)}")


if __name__ == "__main__":
    main()
