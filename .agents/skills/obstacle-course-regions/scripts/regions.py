#!/usr/bin/env python3
"""Look up which Obstacle Course region a name or (x, y) point belongs to.

This imports generate_obstacle_course.py for its rect/circle constants
(CAR_WASH, GRAVEL, TUNNEL, ...) rather than copying coordinates into a
second table. Those constants are read off the course drawing and can move
whenever the DXF is revised; importing them means this stays correct
automatically instead of silently drifting the way a hand-copied table
would. Nothing here calls into the DXF-reading functions, so this works
without the drawing file being present.

Regions are listed in the order the car drives them, matching the tour in
jetson/README.md ("tunnel, narrow path, gravel box, potholes, buckets,
hoops, car wash, and a banked turn...") plus the overpass/helix pair that
precedes the tunnel and the open area that precedes the buckets.

Two regions -- "narrow_region" and "wide_open_region" -- have no dedicated
rect constant in the generator; they are the ordinary lane and the widened
approach to the bucket section, described qualitatively rather than with a
precise box.  See NOTE on each for what that means in practice.

Usage:
    python3 regions.py near 2.1 -9.3       # which region contains this point
    python3 regions.py show gravel_pit     # bounds + source for one region
    python3 regions.py list                # every region, in driving order
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "jetson" / "scripts"))
import generate_obstacle_course as course  # noqa: E402


def rect_world(
    rect: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """A (x0, y0, x1, y1) rect in DXF feet -> (min_x, min_y, max_x, max_y) in world meters."""
    x0, y0, x1, y1 = rect
    ax, ay = course.to_world(x0, y0)
    bx, by = course.to_world(x1, y1)
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def union(
    *rects: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = zip(*rects)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def circle_world(
    centre_ft: tuple[float, float], radius_ft: float
) -> tuple[float, float, float, float]:
    """A DXF-feet centre + radius -> a (min_x, min_y, max_x, max_y) bounding box in world meters."""
    cx, cy = course.to_world(*centre_ft)
    r = radius_ft * course.FOOT
    return (cx - r, cy - r, cx + r, cy + r)


def hoop_lines_world() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    lines = []
    for fixed, travel_from, travel_to, axis, _nominal in course.HOOPS:
        if axis == "y":
            a, b = (
                course.to_world(fixed, travel_from),
                course.to_world(fixed, travel_to),
            )
        else:
            a, b = (
                course.to_world(travel_from, fixed),
                course.to_world(travel_to, fixed),
            )
        lines.append((a, b))
    return lines


def point_segment_distance(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.dist(p, (ax + t * dx, ay + t * dy))


# Two of the three hoops have a FIXED x (their whole line is a single x value),
# so a shared bounding box puts that entire line exactly on the box's edge --
# a point on the line can miss the box on float rounding alone, and the box
# can't tell "near a hoop" from "anywhere in the rectangle between them"
# either way. Checking distance to the nearest hoop's actual travel line is
# both more robust and more honest about what "in hoops" means. Half the
# hoop's own base width, plus a little slack for where a car's centre could
# sit while straddling it.
HOOP_LINE_TOLERANCE = course.HOOP_BASE_LENGTH / 2 + 0.1


def near_hoop_line(x: float, y: float) -> bool:
    return any(
        point_segment_distance((x, y), a, b) <= HOOP_LINE_TOLERANCE
        for a, b in hoop_lines_world()
    )


# The helix is an annulus (inner/outer radius) swept through 270 degrees, not
# a filled disc -- its own hollow axis and the 90 degree wedge the sweep
# skips (where the tunnel sits, per build_helix()'s comment) are both outside
# it. A plain bounding-box or bounding-circle check would wrongly claim
# both. World-frame angles are DXF angles rotated by +180 degrees, the same
# half turn to_world() applies to positions -- see build_helix() in
# generate_obstacle_course.py, which this mirrors.
def in_helix(x: float, y: float) -> bool:
    cx, cy = course.to_world(*course.HELIX_CENTRE)
    inner = course.HELIX_INNER_R * course.FOOT
    outer = course.HELIX_OUTER_R * course.FOOT
    radius = math.dist((x, y), (cx, cy))
    if not (inner <= radius <= outer):
        return False
    start = math.radians(course.HELIX_START_DEG) + math.pi
    sweep = math.radians(course.HELIX_SWEEP_DEG)
    angle = (math.atan2(y - cy, x - cx) - start) % (2 * math.pi)
    return angle <= sweep


@dataclass
class Region:
    name: str
    order: int
    aliases: list[str]
    source: str
    bounds: (
        tuple[float, float, float, float] | None
    )  # (min_x, min_y, max_x, max_y), world meters -- for display only when shape_test is set
    note: str = ""
    approximate: bool = False
    shape_test: Callable[[float, float], bool] | None = (
        None  # overrides the bounds rectangle when the true shape isn't a box
    )

    def contains(self, x: float, y: float) -> bool:
        if self.shape_test is not None:
            return self.shape_test(x, y)
        if self.bounds is None:
            return False
        min_x, min_y, max_x, max_y = self.bounds
        return min_x <= x <= max_x and min_y <= y <= max_y


def _build_regions() -> list[Region]:
    hoop_pts = [pt for line in hoop_lines_world() for pt in line]
    hoop_xs = [p[0] for p in hoop_pts]
    hoop_ys = [p[1] for p in hoop_pts]

    return [
        Region(
            name="overpass_ramp",
            order=1,
            aliases=["bridge", "bridge deck", "ramp up", "overpass"],
            source="RAMP_UP + BRIDGE_DECK (generate_obstacle_course.py)",
            bounds=union(rect_world(course.RAMP_UP), rect_world(course.BRIDGE_DECK)),
            note="The straight ramp up plus the flat deck it leads to, 25 in "
            "above the floor. Called an overpass because the deck passes "
            "over the tunnel below it, not because the drawing names it that.",
        ),
        Region(
            name="helical_ramp",
            order=2,
            aliases=["helix", "spiral ramp", "ramp down"],
            source="HELIX_CENTRE / HELIX_INNER_R / HELIX_OUTER_R / HELIX_START_DEG / HELIX_SWEEP_DEG (generate_obstacle_course.py)",
            bounds=circle_world(course.HELIX_CENTRE, course.HELIX_OUTER_R),
            note="270 degree spiral down from the bridge deck to the tunnel "
            "mouth. Containment checks the real annulus (inner/outer radius) "
            "and the swept angle range, not just a bounding box -- the "
            "displayed bounds are still the outer radius's box, for a quick "
            "look, but `near`/`classify` use the exact shape.",
            approximate=True,
            shape_test=in_helix,
        ),
        Region(
            name="tunnel",
            order=3,
            aliases=["tunnel"],
            source="TUNNEL (generate_obstacle_course.py)",
            bounds=rect_world(course.TUNNEL),
            note="Closed on top and sides; carries the bridge deck above it.",
        ),
        Region(
            name="narrow_region",
            order=4,
            aliases=["narrow path", "narrow section"],
            source="no dedicated constant -- see note",
            bounds=None,
            note="jetson/README.md lists this between the tunnel and the "
            "gravel box, but generate_obstacle_course.py has no rect for "
            "it -- it's the ordinary LANE_WIDTH-wide (32 in) lane connecting "
            "the two, not a distinct built feature the way the other "
            "regions are. There's no bounding box to give; treat a point "
            "here as 'between tunnel and gravel_pit, not in either.'",
            approximate=True,
        ),
        Region(
            name="gravel_pit",
            order=5,
            aliases=["gravel", "gravel box", "pea gravel"],
            source="GRAVEL (generate_obstacle_course.py)",
            bounds=rect_world(course.GRAVEL),
            note="Low-friction lid plus scattered pebbles; GRAVEL_ENTRY_RAMP "
            "and GRAVEL_EXIT_RAMP are the approach/exit ramps, not part of "
            "this box.",
        ),
        Region(
            name="banked_turn",
            order=6,
            aliases=["bank", "banked turn", "bank turn"],
            source="BANK (generate_obstacle_course.py)",
            bounds=rect_world(course.BANK),
            note="8.5 degree banked turn, 48 in by 128 in.",
        ),
        Region(
            name="potholes",
            order=7,
            aliases=["pothole", "pothole board", "potholes"],
            source="POTHOLE (generate_obstacle_course.py)",
            bounds=rect_world(course.POTHOLE),
            note="Plywood board with raised bumps; POTHOLE_ENTRY_RAMP and "
            "POTHOLE_EXIT_RAMP are the approach/exit ramps, not part of "
            "this box.",
        ),
        Region(
            name="wide_open_region",
            order=8,
            aliases=["wide area", "open area", "car wash approach"],
            source="no dedicated constant -- see note",
            bounds=None,
            note="The open floor around the car wash, between the main loop "
            "and the bucket section's entrance -- the same 'wide area' "
            "obstacle_randomizer_node's gap-bale wall separates from the "
            "bucket section (see jetson/README.md's 'bucket section's "
            "entrance' note). No rect constant defines its edges, so "
            "there's no bounding box to give here either.",
            approximate=True,
        ),
        Region(
            name="buckets",
            order=9,
            aliases=["bucket section", "bucket region", "buckets"],
            source="BUCKET_REGION (generate_obstacle_course.py)",
            bounds=rect_world(course.BUCKET_REGION),
            note="Randomized count and placement; obstacle_randomizer_node "
            "owns the live layout, this is just the region it draws within.",
        ),
        Region(
            name="hoops",
            order=10,
            aliases=["hoop", "hoops"],
            source="HOOPS (generate_obstacle_course.py)",
            bounds=(min(hoop_xs), min(hoop_ys), max(hoop_xs), max(hoop_ys)),
            note=f"Containment checks distance to the nearest of the three "
            f"hoops' actual travel lines (within {HOOP_LINE_TOLERANCE:.3f} m), "
            "not a shared bounding box -- two of the three hoops have a "
            "fixed x, so their whole line would otherwise sit exactly on "
            "the box's edge. The displayed bounds are still the box around "
            "all three lines, for a quick look; `near`/`classify` use the "
            "per-line distance.",
            approximate=True,
            shape_test=near_hoop_line,
        ),
        Region(
            name="car_wash",
            order=11,
            aliases=["car wash", "carwash"],
            source="CAR_WASH (generate_obstacle_course.py)",
            bounds=rect_world(course.CAR_WASH),
            note="Five ribboned arches; collision is on the uprights only, "
            "the ribbons themselves are visual-only.",
        ),
    ]


REGIONS: list[Region] = _build_regions()
BY_NAME: dict[str, Region] = {}
for _region in REGIONS:
    BY_NAME[_region.name] = _region
    for _alias in _region.aliases:
        BY_NAME.setdefault(_alias.lower(), _region)


def find(name: str) -> Region | None:
    return BY_NAME.get(name.strip().lower())


def classify(x: float, y: float) -> list[Region]:
    """Every region (there should usually be zero or one) containing (x, y)."""
    return [region for region in REGIONS if region.contains(x, y)]


def _format_region(region: Region) -> str:
    lines = [
        f"{region.order:2d}. {region.name}  (aliases: {', '.join(region.aliases)})"
    ]
    if region.bounds:
        min_x, min_y, max_x, max_y = region.bounds
        tag = " (approximate)" if region.approximate else ""
        lines.append(
            f"    bounds{tag}: x [{min_x:.3f}, {max_x:.3f}], y [{min_y:.3f}, {max_y:.3f}] (world meters)"
        )
    else:
        lines.append("    bounds: none -- descriptive region, see note")
    lines.append(f"    source: {region.source}")
    if region.note:
        lines.append(f"    note: {region.note}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list every region in driving order")

    show = sub.add_parser("show", help="show one region's bounds and source")
    show.add_argument(
        "name", help="region name or alias, e.g. 'gravel_pit' or 'gravel box'"
    )

    near = sub.add_parser("near", help="which region(s) contain a world-meter point")
    near.add_argument("x", type=float)
    near.add_argument("y", type=float)

    args = parser.parse_args()

    if args.command == "list":
        for region in REGIONS:
            print(_format_region(region))
        return

    if args.command == "show":
        region = find(args.name)
        if region is None:
            print(
                f"no region matches '{args.name}' -- run 'list' to see names and aliases",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(_format_region(region))
        return

    if args.command == "near":
        hits = classify(args.x, args.y)
        if not hits:
            print(
                f"({args.x}, {args.y}) is not inside any region's bounds "
                "(it may be in one of the two descriptive regions -- "
                "narrow_region or wide_open_region -- which have none)"
            )
            return
        for region in hits:
            print(_format_region(region))


if __name__ == "__main__":
    main()
