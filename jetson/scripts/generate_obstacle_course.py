#!/usr/bin/env python3
"""Generate the Gazebo Obstacle Course world from its DXF layout.

Counterpart to ``generate_speed_course.py``.  That one only has to place straw
bales; this one also has to place eleven sections of obstacle, so it writes
the whole world rather than patching a block inside an existing file.

Everything the car drives on comes from the DXF, which is the drawing the
bale walls were laid out from.  Everything the car drives *past* is a mesh
tessellated out of the CAD by ``step_to_stl.py``.  Where the two disagree --
the helical ramp radius differs by about 5% between them -- the DXF wins,
because the bales around it were drawn to the DXF.

Collision is primitives throughout.  The meshes are decimated visuals and
would make a poor driving surface, and the car has to be able to climb the
ramp, corner on the bank and cross the gravel without catching on a triangle
edge.

Two outputs:

* ``worlds/obstacle_course.sdf`` -- the world, carrying the two ``cfr:sensors``
  markers that ``simulation.launch.py`` fills in for ``sensors:=true``.
* ``config/obstacle_course_layout.yaml`` -- the bounds and spacing rules for
  the variable elements, so ``obstacle_randomizer_node`` does not carry a
  second copy of the geometry.

Usage:
    ./generate_obstacle_course.py "2026 course designs ... site layout 2.dxf"
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import start_signal

FOOT = 0.3048
INCH = 0.0254

PACKAGE = Path(__file__).parents[1] / "cfr_arduino_bridge"
MESH_URI = "model://cfr_arduino_bridge/meshes"

# Straw bales are nominally 36 x 18 x 14 in, matching generate_speed_course.py.
# The DXF hatches measure about 34 in along their length because the outline is
# drawn with a wavy straw edge that cuts inside the corners.
BALE_LENGTH = 36 * INCH
BALE_WIDTH = 18 * INCH
BALE_HEIGHT = 14 * INCH

# The start/finish line, and the centerline of the 32 in wide start lane, in
# DXF feet.  The world origin sits here with +x pointing the way the car
# drives away from the line, so a path goal reads the same as it would on the
# real course.
START_X = 97.515
START_Y = 109.1045

# Course features, as (x0, y0, x1, y1) rectangles in DXF feet, read off the
# PATH_ELEMENTS and OBJECTS layers.
CAR_WASH = (102.932, 107.105, 110.932, 111.105)
POTHOLE = (89.394, 134.091, 97.394, 138.091)
POTHOLE_ENTRY_RAMP = (97.393, 135.424, 100.060, 138.091)
POTHOLE_EXIT_RAMP = (86.728, 134.090, 89.395, 138.090)
GRAVEL = (85.897, 139.614, 93.897, 143.614)
GRAVEL_ENTRY_RAMP = (83.230, 140.281, 85.897, 142.947)
GRAVEL_EXIT_RAMP = (93.897, 139.614, 96.564, 143.614)
BANK = (108.729, 134.518, 112.729, 145.185)
TUNNEL = (72.896, 107.765, 75.562, 112.765)
RAMP_UP = (75.557, 107.770, 86.432, 110.436)
BRIDGE_DECK = (70.286, 107.770, 75.557, 110.436)

# The helical ramp down.  The DXF draws it as a 270 degree bulged polyline;
# these are the center and the two radii that polyline resolves to, and the
# deck height it descends from.  4 ft centerline radius and 41.5 in width in
# the drawing's own annotation, 11% grade, all three of which this reproduces.
HELIX_CENTRE = (70.287, 105.153)
HELIX_INNER_R = 2.214
HELIX_OUTER_R = 5.667
HELIX_START_DEG = 90.0
HELIX_SWEEP_DEG = 270.0
HELIX_SEGMENTS = 24

DECK_HEIGHT = 25 * INCH
DECK_THICKNESS = 0.013
LANE_WIDTH = 32 * INCH
RAIL_HEIGHT = 6 * INCH

# Car wash arch row: the first arch sits this far from the base plate's center
# and they repeat at this pitch, both measured off the CAD assembly.  The row
# is not centered on the plate, so the offset cannot be derived from the pitch.
CAR_WASH_FIRST_ARCH = -0.9495
CAR_WASH_ARCH_PITCH = 0.4572

# Variable elements.  Buckets sit anywhere inside their section; hoops slide
# along the dashed lines the drawing puts them on.
BUCKET_REGION = (87.644, 111.986, 99.654, 124.291)
BUCKET_MIN_SPACING = 3.0
BUCKET_WALL_CLEARANCE = 2.5
BUCKET_DEFAULT_COUNT = 6
BUCKET_MAX = 9
BUCKET_MIN = 2
# Somewhere off the course to stand the buckets a draw does not use.  Below
# y = 100 ft nothing in this world exists -- that is where the Speed Course
# starts, and it is a different world file -- so a row here is clear of every
# bale whatever its x.
BUCKET_PARKING = (100.0, 97.5)
BUCKET_PARKING_PITCH = 0.5

BUCKET_NOMINAL = [
    (90.433, 121.786),
    (96.600, 121.786),
    (95.433, 114.494),
    (90.420, 115.454),
    (97.100, 117.077),
    (93.517, 121.786),
]

# (fixed coordinate, travel from, travel to, axis, nominal position).  "y"
# means the hoop slides along DXF y with its uprights spanning x, and the
# other way round for "x".
HOOPS = [
    (110.443, 112.618, 118.618, "y", 114.7205),
    (118.415, 112.618, 118.618, "y", 116.9945),
    (108.616, 122.406, 128.406, "x", 124.2045),
]
HOOP_BASE_LENGTH = 0.709

# The direction the car drives away from the start line.  to_world() turns
# the drawing so that this is +x, which is what makes a path goal read the
# same here as it does on the real course.
LANE_HEADING = 0.0

# Where the start signal stands, in world meters.  The drawing places it the
# same way on both courses -- three bales down the wall from the start line,
# in line with the inner edge of the bale border -- so this comes out of
# start_signal.position() rather than being chosen here.  The world origin is
# the start line, so the line is (0, 0) with LANE_HEADING, and the border's
# inner edge is half the 32 in lane to the left.  It stands square across the
# lane rather than aimed at the waiting car -- see start_signal.yaw_across.
SIGNAL_POSITION = start_signal.position((0.0, 0.0), LANE_HEADING, LANE_WIDTH / 2)

# Pea gravel: a surface the tires slip on, plus something for them to climb
# over.  Gazebo has no granular physics, so the box gets a low friction lid
# and a scatter of pebbles rather than a bed of stones.
# Pothole board and bumps, from the CAD: 1.5 in of plywood with 0.75 in
# bumps standing on it.
POTHOLE_BOARD_HEIGHT = 1.5 * INCH
POTHOLE_BUMP_HEIGHT = 0.75 * INCH

GRAVEL_FRICTION = 0.35
GRAVEL_SURFACE_Z = 2 * INCH
GRAVEL_PEBBLES = 90
GRAVEL_SEED = 20260901

VEHICLE_START = (-0.70, 0.0, 0.12)

# Rendered sensors are off by default, because gz-sim-sensors needs a render
# context and the headless container may not have one.  They cannot be a
# launch argument on their own -- Gazebo ignores its default server config
# once a world declares plugins -- so the world carries these two markers and
# simulation.launch.py substitutes them when sensors:=true.  Markers rather
# than a second generated world: the Speed Course's world is hand maintained,
# and a derived copy of a hand maintained file goes stale the first time
# somebody edits one and not the other.
SYSTEM_MARKER = "<!-- cfr:sensors-system -->"
CAMERA_MARKER = "<!-- cfr:sensors-camera -->"


def dxf_entities(dxf_file: Path):
    """(type, [(code, value)]) for every entity in the file."""
    lines = dxf_file.read_text(errors="replace").splitlines()
    pairs = [
        (lines[index].strip(), lines[index + 1].strip())
        for index in range(0, len(lines) - 1, 2)
    ]
    entities = []
    entity = None
    for code, value in pairs:
        if code == "0":
            if entity:
                entities.append(entity)
            entity = (value, [])
        elif entity:
            entity[1].append((code, value))
    if entity:
        entities.append(entity)
    return entities


def dxf_obstacle_bales(dxf_file: Path) -> list[tuple[float, float, float]]:
    """Straw bale centers and headings, in DXF feet and radians.

    The two courses share one drawing.  The Obstacle Course is the half above
    y = 100 ft; below that is the Speed Course, which
    ``generate_speed_course.py`` owns.
    """
    bales = []
    for entity_type, tags in dxf_entities(dxf_file):
        if entity_type != "HATCH":
            continue
        if next((value for code, value in tags if code == "8"), "") != "STRAW_BALES":
            continue
        boundary_start = next(
            (index for index, (code, _) in enumerate(tags) if code == "91"), None
        )
        if boundary_start is None:
            continue

        points: list[tuple[float, float]] = []
        x_value = end_x = None
        end_point = None
        for code, value in tags[boundary_start + 1 :]:
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

        centre_x = (min(x for x, _ in points) + max(x for x, _ in points)) / 2
        centre_y = (min(y for _, y in points) + max(y for _, y in points)) / 2
        if centre_y < 100:
            continue
        start_x, start_y = points[0]
        yaw = math.atan2(end_point[1] - start_y, end_point[0] - start_x)
        bales.append((centre_x, centre_y, yaw))

    if len(bales) != 121:
        raise RuntimeError(f"Expected 121 Obstacle Course bales, found {len(bales)}")
    return bales


def to_world(x_ft: float, y_ft: float) -> tuple[float, float]:
    """DXF feet to world meters.

    A half turn about the start line, so that the direction the car leaves the
    line -- which is -x in the drawing -- becomes +x in the world.  A mirror
    would line up just as well and silently flip every heading, so this stays
    a rotation.
    """
    return (START_X - x_ft) * FOOT, (START_Y - y_ft) * FOOT


def rect_centre(rect) -> tuple[float, float]:
    x0, y0, x1, y1 = rect
    return to_world((x0 + x1) / 2, (y0 + y1) / 2)


def rect_size(rect) -> tuple[float, float]:
    x0, y0, x1, y1 = rect
    return abs(x1 - x0) * FOOT, abs(y1 - y0) * FOOT


def box(
    name: str,
    pose: tuple,
    size: tuple,
    colour: str,
    collide: bool = True,
    visual: bool = True,
    friction: float | None = None,
) -> str:
    """One box on an existing link, as a collision, a visual, or both.

    Anything an obstacle mesh already draws passes `visual=False`.  A
    primitive standing in for the mesh's shape is there to be collided with,
    and drawing it as well puts an untextured box in front of the mesh -- on
    the start signal that was enough to hide the arms completely.
    """
    pose_text = " ".join(f"{value:.4f}" for value in pose)
    size_text = " ".join(f"{value:.4f}" for value in size)
    geometry = f"<box><size>{size_text}</size></box>"
    out = ""
    if collide:
        surface = ""
        if friction is not None:
            surface = (
                f"<surface><friction><ode><mu>{friction}</mu><mu2>{friction}</mu2></ode>"
                f"<bullet><friction>{friction}</friction><friction2>{friction}</friction2>"
                "</bullet></friction></surface>"
            )
        out += (
            f'      <collision name="{name}_collision"><pose>{pose_text}</pose>'
            f"<geometry>{geometry}</geometry>{surface}</collision>\n"
        )
    if visual:
        out += (
            f'      <visual name="{name}_visual"><pose>{pose_text}</pose>'
            f"<geometry>{geometry}</geometry>"
            f"<material><diffuse>{colour}</diffuse></material></visual>\n"
        )
    return out


def cylinder(
    name: str,
    pose: tuple,
    radius: float,
    length: float,
    colour: str,
    collide: bool = True,
    visual: bool = True,
) -> str:
    pose_text = " ".join(f"{value:.4f}" for value in pose)
    geometry = f"<cylinder><radius>{radius:.4f}</radius><length>{length:.4f}</length></cylinder>"
    out = ""
    if collide:
        out += (
            f'      <collision name="{name}_collision"><pose>{pose_text}</pose>'
            f"<geometry>{geometry}</geometry></collision>\n"
        )
    if visual:
        out += (
            f'      <visual name="{name}_visual"><pose>{pose_text}</pose>'
            f"<geometry>{geometry}</geometry>"
            f"<material><diffuse>{colour}</diffuse></material></visual>\n"
        )
    return out


def mesh_visual(name: str, pose: tuple, uri: str, colour: str) -> str:
    pose_text = " ".join(f"{value:.4f}" for value in pose)
    return (
        f'      <visual name="{name}_visual"><pose>{pose_text}</pose>'
        f"<geometry><mesh><uri>{uri}</uri></mesh></geometry>"
        f"<material><diffuse>{colour}</diffuse></material></visual>\n"
    )


def static_model(name: str, body: str) -> str:
    return (
        f'    <model name="{name}"><static>true</static><link name="link">\n'
        f"{body}"
        "    </link></model>\n"
    )


PLYWOOD = "0.62 0.48 0.31 1"
# The start signal's frame is painted sky blue on the course; only its two
# arms are the red and green the car is looking for.
SKY_BLUE = "0.53 0.81 0.92 1"
STRAW = "0.72 0.48 0.12 1"
PVC = "0.88 0.88 0.90 1"
GRAVEL_GREY = "0.44 0.43 0.40 1"
FOIL = "0.74 0.76 0.78 1"


def build_bales(bales) -> str:
    # Placed in world meters before anything is written out, because the one
    # or two bales the start signal's board stands in have to move along the
    # wall to clear it, and that is a world-space measurement against the
    # board's footprint.  See start_signal.clear_bales.
    placed = [(*to_world(x_ft, y_ft), yaw + math.pi) for x_ft, y_ft, yaw in bales]
    cleared = start_signal.clear_bales(
        placed, SIGNAL_POSITION, LANE_HEADING, (BALE_LENGTH, BALE_WIDTH)
    )
    shifted = sum(1 for before, after in zip(placed, cleared) if before != after)
    if shifted:
        print(f"moved {shifted} bale(s) along the wall to clear the start signal")

    body = ""
    for index, (x, y, yaw) in enumerate(cleared):
        body += box(
            f"bale_{index}",
            (x, y, BALE_HEIGHT / 2, 0, 0, yaw),
            (BALE_LENGTH, BALE_WIDTH, BALE_HEIGHT),
            STRAW,
        )
    return (
        "    <!-- 65 ft by 48 ft obstacle course, walled with 14 x 18 x 36 in"
        " straw bales. -->\n" + static_model("course_bales", body)
    )


def build_car_wash() -> str:
    """Base and arches as CAD, uprights as collision, ribbons as visuals only.

    The five arches are one mesh instanced five times -- they are identical on
    the course, and baking five copies into one STL costs five times the
    bytes and five times the triangles for no difference on screen.

    The ribbons are the point of the section -- the car has to push through
    them -- but they are streamer weight.  Giving them collision would have
    Gazebo resolving forty contacts against a vehicle they could not deflect,
    so they are drawn and not felt.
    """
    x, y = rect_centre(CAR_WASH)
    width, depth = rect_size(CAR_WASH)
    body = mesh_visual("base", (x, y, 0, 0, 0, 0), f"{MESH_URI}/car_wash_base.stl", PVC)
    # The base plate is a 13 mm lip the car drives over.
    body += box(
        "base", (x, y, 0.0065, 0, 0, 0), (width, depth, 0.013), PVC, visual=False
    )

    arch_span = 1.151
    for arch in range(5):
        arch_x = x + CAR_WASH_FIRST_ARCH + CAR_WASH_ARCH_PITCH * arch
        body += mesh_visual(
            f"arch_{arch}",
            (arch_x, y, 0, 0, 0, 0),
            f"{MESH_URI}/car_wash_arch.stl",
            PVC,
        )
        for side in (-1, 1):
            body += cylinder(
                f"upright_{arch}_{side}",
                (arch_x, y + side * arch_span / 2, 0.21, 0, 0, 0),
                0.017,
                0.42,
                PVC,
                visual=False,
            )
        for row in range(8):
            strip_y = y + (row - 3.5) * 0.127
            colour = "0.90 0.25 0.20 1" if row % 2 else "0.20 0.45 0.85 1"
            body += box(
                f"strip_{arch}_{row}",
                (arch_x, strip_y, 0.3275, 0, 0, 0),
                (0.037, 0.051, 0.425),
                colour,
                collide=False,
            )
    return (
        "    <!-- Car wash: ribbons are visual only, they cannot deflect the car. -->\n"
        + static_model("car_wash", body)
    )


def build_gravel(random) -> str:
    """The gravel box: low friction lid plus a scatter of pebbles.

    Wheel slip and an uneven surface are what this section tests, and neither
    falls out of a flat plate.  The lid's friction is two orders below the
    course floor's, and the pebbles give the suspension something to find.
    """
    x, y = rect_centre(GRAVEL)
    width, depth = rect_size(GRAVEL)
    body = mesh_visual("box", (x, y, 0, 0, 0, 0), f"{MESH_URI}/gravel_box.stl", PLYWOOD)
    body += box(
        "base", (x, y, 0.0065, 0, 0, 0), (width, depth, 0.013), PLYWOOD, visual=False
    )
    for side, offset in (("n", depth / 2), ("s", -depth / 2)):
        body += box(
            f"rail_{side}",
            (x, y + offset, 0.032, 0, 0, 0),
            (width, 0.038, 0.038),
            PLYWOOD,
            visual=False,
        )
    body += box(
        "surface",
        (x, y, GRAVEL_SURFACE_Z - 0.019, 0, 0, 0),
        (width - 0.076, depth - 0.076, 0.038),
        GRAVEL_GREY,
        visual=False,
        friction=GRAVEL_FRICTION,
    )
    for pebble in range(GRAVEL_PEBBLES):
        px = x + random.uniform(-width / 2 + 0.1, width / 2 - 0.1)
        py = y + random.uniform(-depth / 2 + 0.1, depth / 2 - 0.1)
        size = random.uniform(0.03, 0.055)
        height = random.uniform(0.008, 0.019)
        body += box(
            f"pebble_{pebble}",
            (px, py, GRAVEL_SURFACE_Z + height / 2, 0, 0, random.uniform(0, math.pi)),
            (size, size * random.uniform(0.6, 1.0), height),
            GRAVEL_GREY,
            friction=GRAVEL_FRICTION,
        )
    return (
        "    <!-- Gravel: low friction lid and loose pebbles stand in for pea gravel. -->\n"
        + static_model("gravel_box", body)
    )


def build_potholes(circles) -> str:
    x, y = rect_centre(POTHOLE)
    body = mesh_visual(
        "board", (x, y, 0, 0, 0, 0), f"{MESH_URI}/pothole_board.stl", PLYWOOD
    )
    width, depth = rect_size(POTHOLE)
    # The board is 1.5 in of plywood and the bumps another 0.75 in on top of
    # it, both straight off the CAD -- and the approach ramps are built to
    # reach 1.5 in, so the collision has to be the same height as the mesh or
    # the car drives up the ramp and drops through the board.
    body += box(
        "base",
        (x, y, POTHOLE_BOARD_HEIGHT / 2, 0, 0, 0),
        (width, depth, POTHOLE_BOARD_HEIGHT),
        PLYWOOD,
        visual=False,
    )
    for index, (bump_x, bump_y) in enumerate(circles):
        bx, by = to_world(bump_x, bump_y)
        # The mesh already sits at its own height above the board, so it is
        # placed on the floor rather than on the board.
        body += mesh_visual(
            f"bump_{index}",
            (bx, by, 0, 0, 0, 0),
            f"{MESH_URI}/pothole_bump.stl",
            PLYWOOD,
        )
        body += cylinder(
            f"bump_{index}_body",
            (bx, by, POTHOLE_BOARD_HEIGHT + POTHOLE_BUMP_HEIGHT / 2, 0, 0, 0),
            0.076,
            POTHOLE_BUMP_HEIGHT,
            PLYWOOD,
            visual=False,
        )
    return (
        "    <!-- Potholes: bumps stand proud, holes are cut into the board mesh. -->\n"
        + static_model("pothole_section", body)
    )


# Which way each ramp mesh climbs in its own frame, measured off the CAD.
# They are not all drawn the same way round, so the generator cannot assume
# one and turn the odd one out by eye.
RAMP_MESH_CLIMB = {
    "ramp_up.stl": 1,
    "ramp_pothole_entry.stl": 1,
    "ramp_narrowing.stl": -1,
}


def build_ramp(name: str, rect, mesh: str, rise: float, platform) -> str:
    """One approach ramp, climbing towards the raised section it serves.

    Which way that is cannot be assumed: a section's entry and exit ramps sit
    on opposite sides of it, so the one that climbs towards +x on the way in
    has to climb towards -x on the way out.  The direction is taken from where
    the raised section actually is, and both the wedge and the mesh follow it.

    Pitch runs the other way to the climb: SDF applies it about +y, so a
    negative pitch is what lifts the +x end.
    """
    x, y = rect_centre(rect)
    width, depth = rect_size(rect)
    slope = math.atan2(rise, width)
    climb = 1 if rect_centre(platform)[0] > x else -1
    yaw = 0.0 if RAMP_MESH_CLIMB[mesh] == climb else math.pi
    body = mesh_visual(name, (x, y, 0, 0, 0, yaw), f"{MESH_URI}/{mesh}", PLYWOOD)
    body += box(
        f"{name}_wedge",
        (x, y, rise / 2 - 0.02, 0, -climb * slope, 0),
        (width / math.cos(slope), depth, 0.04),
        PLYWOOD,
        visual=False,
    )
    return static_model(name, body)


def build_bridge() -> str:
    """Straight ramp up, flat bridge deck, and the tunnel that carries it."""
    body = ""

    low_x, _ = to_world(RAMP_UP[2], 0)
    high_x, _ = to_world(RAMP_UP[0], 0)
    run = high_x - low_x
    slope = math.atan2(DECK_HEIGHT, run)
    body += box(
        "ramp",
        (
            (low_x + high_x) / 2,
            to_world(0, (RAMP_UP[1] + RAMP_UP[3]) / 2)[1],
            DECK_HEIGHT / 2 - 0.05,
            0,
            -slope,
            0,
        ),
        (run / math.cos(slope), LANE_WIDTH, 0.10),
        PLYWOOD,
    )
    for side in (-1, 1):
        body += box(
            f"ramp_rail_{side}",
            (
                (low_x + high_x) / 2,
                to_world(0, (RAMP_UP[1] + RAMP_UP[3]) / 2)[1]
                + side * (LANE_WIDTH + 0.05) / 2,
                DECK_HEIGHT / 2 + RAIL_HEIGHT / 2,
                0,
                -slope,
                0,
            ),
            (run / math.cos(slope), 0.05, RAIL_HEIGHT),
            PLYWOOD,
        )

    deck_x, deck_y = rect_centre(BRIDGE_DECK)
    deck_width, _ = rect_size(BRIDGE_DECK)
    # Thin, because the tunnel below it is 24.5 in tall and the deck surface
    # is at 25 in -- the tunnel is what holds the bridge up, on the course and
    # here.
    body += box(
        "deck",
        (deck_x, deck_y, DECK_HEIGHT - DECK_THICKNESS / 2, 0, 0, 0),
        (deck_width, LANE_WIDTH, DECK_THICKNESS),
        PLYWOOD,
    )
    for side in (-1, 1):
        body += box(
            f"deck_rail_{side}",
            (
                deck_x,
                deck_y + side * (LANE_WIDTH + 0.05) / 2,
                DECK_HEIGHT + RAIL_HEIGHT / 2,
                0,
                0,
                0,
            ),
            (deck_width, 0.05, RAIL_HEIGHT),
            PLYWOOD,
        )
    return (
        "    <!-- 20% ramp up to a flat bridge deck, 25 in above the floor. -->\n"
        + static_model("bridge", body)
    )


def build_tunnel() -> str:
    x, y = rect_centre(TUNNEL)
    # The tunnel runs along y, so its length is the rectangle's y extent --
    # taking the x extent instead walls only the first half of it and leaves
    # the car free to drive out of the side.
    _, length = rect_size(TUNNEL)
    body = mesh_visual(
        "tunnel", (x, y, 0, 0, 0, math.pi / 2), f"{MESH_URI}/tunnel.stl", FOIL
    )
    height = 24.5 * INCH
    for side in (-1, 1):
        body += box(
            f"wall_{side}",
            (x + side * (LANE_WIDTH + 0.05) / 2, y, height / 2, 0, 0, 0),
            (0.05, length, height),
            FOIL,
            visual=False,
        )
    return (
        "    <!-- Tunnel: closed on top and sides, and it carries the bridge deck. -->\n"
        + static_model("tunnel", body)
    )


def build_helix() -> str:
    """The 270 degree ramp down, as tilted box segments.

    The DXF draws this as a polyline with a 270 degree bulge; resolving the
    bulge gives the center and the two radii above.  Segmenting it is what
    makes it drivable -- a decimated mesh of the CAD helix would catch the
    wheels on every triangle edge, and a single box cannot be a spiral.

    It is one link in a continuous path: the bridge deck feeds the top of the
    ramp, and the bottom of the ramp feeds the tunnel that runs under the
    bridge.  Both joins fall out of the drawing's own geometry -- the helix
    center sits one mean radius from the end of the deck, on the lane
    centerline, and a quarter turn short of a full circle from there is the
    tunnel's centerline -- but only if the turn runs counter-clockwise.
    """
    centre_x, centre_y = to_world(*HELIX_CENTRE)
    inner = HELIX_INNER_R * FOOT
    outer = HELIX_OUTER_R * FOOT
    mean_r = (inner + outer) / 2
    width = outer - inner
    sweep = math.radians(HELIX_SWEEP_DEG)
    # The drawing's 90 degrees, turned by the half turn the world takes about
    # the start line.  That lands the top of the ramp on the lane centerline
    # at the end of the bridge deck, which is what it has to join.
    start = math.radians(HELIX_START_DEG) + math.pi
    grade = DECK_HEIGHT / (mean_r * sweep)
    slope = math.atan(grade)

    body = ""
    for index in range(HELIX_SEGMENTS):
        fraction = (index + 0.5) / HELIX_SEGMENTS
        # Counter-clockwise, seen from above.  That is the direction that
        # makes the ramp a path: it leaves the bridge deck heading the way the
        # car was already going, and 270 degrees later it comes out on the
        # tunnel's centerline pointing into the mouth.  Clockwise puts the
        # entry backwards and the exit 8 ft the wrong side of the tunnel.
        angle = start + sweep * fraction
        height = DECK_HEIGHT * (1.0 - fraction)
        chord = 2 * outer * math.sin(sweep / (2 * HELIX_SEGMENTS)) * 1.04
        x = centre_x + mean_r * math.cos(angle)
        y = centre_y + mean_r * math.sin(angle)
        # Each segment points the way the car drives, along the turn.
        yaw = angle + math.pi / 2
        # Pitch is positive here where the straight ramp's is negative: yaw
        # points each segment along the way the car drives, and on the helix
        # that is downhill.  Getting this backwards tilts every slab against
        # the descent and turns the spiral into a sawtooth.
        body += box(
            f"deck_{index}",
            (x, y, height - 0.05, 0, slope, yaw),
            (chord, width, 0.10),
            PLYWOOD,
        )
        for name, radius in (("inner", inner - 0.025), ("outer", outer + 0.025)):
            body += box(
                f"{name}_rail_{index}",
                (
                    centre_x + radius * math.cos(angle),
                    centre_y + radius * math.sin(angle),
                    height + RAIL_HEIGHT / 2,
                    0,
                    slope,
                    yaw,
                ),
                (chord, 0.05, RAIL_HEIGHT),
                PLYWOOD,
            )
    return (
        "    <!-- Helical ramp down: 4 ft centerline radius, 41.5 in wide, 11% grade. -->\n"
        + static_model("helix", body)
    )


def build_bank() -> str:
    x, y = rect_centre(BANK)
    width, length = rect_size(BANK)
    body = mesh_visual(
        "bank", (x, y, 0, 0, 0, math.pi), f"{MESH_URI}/bank.stl", PLYWOOD
    )
    # 8.5 degrees, high side outboard of the turn.
    tilt = math.radians(8.5)
    body += box(
        "surface",
        (x, y, width * math.tan(tilt) / 2 - 0.02, 0, 0, 0),
        (width, length, 0.04),
        PLYWOOD,
        visual=False,
    )
    return "    <!-- 8.5 degree banked turn, 48 in by 128 in. -->\n" + static_model(
        "bank", body
    )


def parking_spot(index: int) -> tuple[float, float]:
    """Where bucket `index` stands when a draw leaves it out."""
    x, y = to_world(*BUCKET_PARKING)
    return x - BUCKET_PARKING_PITCH * index, y


def build_buckets() -> str:
    """One model per bucket so the randomizer can move them independently."""
    out = "    <!-- Buckets: moved at runtime by obstacle_randomizer_node. -->\n"
    for index in range(BUCKET_MAX):
        if index < len(BUCKET_NOMINAL):
            x, y = to_world(*BUCKET_NOMINAL[index])
        else:
            # Spares stand off the course until a draw calls for them.  Nine
            # exist because the rules allow up to nine; a static model cannot
            # be spawned on demand, so they are all here from the start.
            x, y = parking_spot(index)
        body = mesh_visual("bucket", (0, 0, 0, 0, 0, 0), f"{MESH_URI}/bucket.stl", PVC)
        body += cylinder("body", (0, 0, 0.19, 0, 0, 0), 0.145, 0.38, PVC, visual=False)
        out += (
            f'    <model name="bucket_{index}"><static>true</static>'
            f"<pose>{x:.4f} {y:.4f} 0 0 0 0</pose>\n"
            f'      <link name="link">\n{body}      </link>\n'
            "    </model>\n"
        )
    return out


def build_hoops() -> str:
    out = "    <!-- Hoops: slide along the dashed lines in the drawing. -->\n"
    for index, (fixed, _, _, axis, nominal) in enumerate(HOOPS):
        if axis == "y":
            x, y = to_world(fixed, nominal)
            yaw = math.pi / 2
        else:
            x, y = to_world(nominal, fixed)
            yaw = 0.0
        body = mesh_visual("hoop", (0, 0, 0, 0, 0, 0), f"{MESH_URI}/hoop.stl", PVC)
        body += box(
            "base", (0, 0, 0.0065, 0, 0, 0), (0.709, 0.405, 0.013), PVC, visual=False
        )
        for side in (-1, 1):
            body += cylinder(
                f"upright_{side}",
                (side * 0.292, 0, 0.26, 0, 0, 0),
                0.017,
                0.52,
                PVC,
                visual=False,
            )
        out += (
            f'    <model name="hoop_{index}"><static>true</static>'
            f"<pose>{x:.4f} {y:.4f} 0 0 0 {yaw:.5f}</pose>\n"
            f'      <link name="link">\n{body}      </link>\n'
            "    </model>\n"
        )
    return out


def build_start_signal() -> str:
    return start_signal.models(SIGNAL_POSITION, LANE_HEADING)


def build_vehicle() -> str:
    x, y, z = VEHICLE_START
    wheels = ""
    for name, wx, wy in (
        ("front_left", 0.162, 0.145),
        ("front_right", 0.162, -0.145),
        ("rear_left", -0.162, 0.145),
        ("rear_right", -0.162, -0.145),
    ):
        wheels += (
            f'      <link name="{name}_wheel"><pose>{wx} {wy} 0.055 -1.5708 0 0</pose>'
            "<inertial><mass>0.12</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy>"
            "<izz>0.001</izz></inertia></inertial>"
            '<collision name="collision"><geometry><cylinder><radius>0.055</radius>'
            "<length>0.035</length></cylinder></geometry><surface><friction>"
            "<ode><mu>50</mu><mu2>1</mu2><fdir1>0 0 1</fdir1></ode>"
            "<bullet><friction>1</friction><friction2>1</friction2>"
            "<rolling_friction>0.001</rolling_friction></bullet></friction></surface></collision>"
            '<visual name="visual"><geometry><cylinder><radius>0.055</radius>'
            "<length>0.035</length></cylinder></geometry>"
            "<material><diffuse>0.04 0.04 0.04 1</diffuse></material></visual></link>\n"
        )

    return f"""    <model name="slash">
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="chassis">
        <inertial><mass>3.5</mass><inertia><ixx>0.08</ixx><iyy>0.12</iyy><izz>0.16</izz></inertia></inertial>
        <collision name="collision"><pose>0 0 0.10 0 0 0</pose><geometry><box><size>0.55 0.30 0.12</size></box></geometry></collision>
        <visual name="body"><pose>0 0 0.10 0 0 0</pose><geometry><box><size>0.55 0.30 0.12</size></box></geometry><material><diffuse>0.85 0.08 0.04 1</diffuse></material></visual>
        <visual name="forward_direction_marker"><pose>0.025 0 0.16 0 1.5708 0</pose><geometry><cone><radius>0.12</radius><length>0.35</length></cone></geometry><material><diffuse>0.05 1 0.08 1</diffuse><emissive>0.02 0.45 0.04 1</emissive></material></visual>
        <visual name="zed2i_mount"><pose>0.25 0 0.155 0 0 0</pose><geometry><box><size>0.06 0.12 0.05</size></box></geometry><material><diffuse>0.20 0.23 0.26 1</diffuse></material></visual>
        <visual name="zed2i_housing"><pose>0.295 0 0.20 0 0 0</pose><geometry><box><size>0.03025 0.17525 0.04310</size></box></geometry><material><diffuse>0.22 0.31 0.38 1</diffuse><emissive>0.01 0.03 0.05 1</emissive><specular>0.45 0.45 0.45 1</specular></material></visual>
        <visual name="zed2i_left_lens"><pose>0.315 0.06 0.20 0 1.5708 0</pose><geometry><cylinder><radius>0.015</radius><length>0.006</length></cylinder></geometry><material><diffuse>0.08 0.55 0.85 1</diffuse><emissive>0.02 0.16 0.28 1</emissive></material></visual>
        <visual name="zed2i_right_lens"><pose>0.315 -0.06 0.20 0 1.5708 0</pose><geometry><cylinder><radius>0.015</radius><length>0.006</length></cylinder></geometry><material><diffuse>0.08 0.55 0.85 1</diffuse><emissive>0.02 0.16 0.28 1</emissive></material></visual>
          {CAMERA_MARKER}
      </link>
      <link name="front_left_steering"><pose>0.162 0.145 0.055 0 0 0</pose><inertial><mass>0.05</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial></link>
      <link name="front_right_steering"><pose>0.162 -0.145 0.055 0 0 0</pose><inertial><mass>0.05</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial></link>
{wheels}      <joint name="front_left_steering_joint" type="revolute"><parent>chassis</parent><child>front_left_steering</child><axis><xyz>0 0 1</xyz><limit><lower>-0.40</lower><upper>0.40</upper><effort>1000000</effort></limit></axis></joint>
      <joint name="front_right_steering_joint" type="revolute"><parent>chassis</parent><child>front_right_steering</child><axis><xyz>0 0 1</xyz><limit><lower>-0.40</lower><upper>0.40</upper><effort>1000000</effort></limit></axis></joint>
      <joint name="front_left_wheel_joint" type="revolute"><parent>front_left_steering</parent><child>front_left_wheel</child><axis><xyz>0 0 1</xyz><limit><lower>-1000000</lower><upper>1000000</upper><effort>1000000</effort></limit></axis></joint>
      <joint name="front_right_wheel_joint" type="revolute"><parent>front_right_steering</parent><child>front_right_wheel</child><axis><xyz>0 0 1</xyz><limit><lower>-1000000</lower><upper>1000000</upper><effort>1000000</effort></limit></axis></joint>
      <joint name="rear_left_wheel_joint" type="revolute"><parent>chassis</parent><child>rear_left_wheel</child><axis><xyz>0 0 1</xyz><limit><lower>-1000000</lower><upper>1000000</upper><effort>1000000</effort></limit></axis></joint>
      <joint name="rear_right_wheel_joint" type="revolute"><parent>chassis</parent><child>rear_right_wheel</child><axis><xyz>0 0 1</xyz><limit><lower>-1000000</lower><upper>1000000</upper><effort>1000000</effort></limit></axis></joint>
      <plugin filename="gz-sim-ackermann-steering-system" name="gz::sim::systems::AckermannSteering">
        <topic>/sim/cmd_vel</topic><odom_topic>/model/slash/odometry</odom_topic>
        <left_joint>front_left_wheel_joint</left_joint><left_joint>rear_left_wheel_joint</left_joint>
        <right_joint>front_right_wheel_joint</right_joint><right_joint>rear_right_wheel_joint</right_joint>
        <left_steering_joint>front_left_steering_joint</left_steering_joint><right_steering_joint>front_right_steering_joint</right_steering_joint>
        <wheel_base>0.324</wheel_base><wheel_separation>0.290</wheel_separation><wheel_radius>0.055</wheel_radius><steering_limit>0.40</steering_limit>
      </plugin>
      <!-- Ground truth pose, bridged to /zed/zed_node/pose.  It stands in for
           the ZED's map frame topic, which the SDK corrects on loop closure;
           the ackermann odometry above drifts and is never corrected, exactly
           as the real camera's ~/odom is not.  lap_counter reads this one. -->
      <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
        <publish_model_pose>true</publish_model_pose><publish_link_pose>false</publish_link_pose>
        <publish_collision_pose>false</publish_collision_pose><publish_visual_pose>false</publish_visual_pose>
        <publish_nested_model_pose>false</publish_nested_model_pose><use_pose_vector_msg>false</use_pose_vector_msg>
        <update_frequency>30</update_frequency>
      </plugin>
    </model>
"""


def build_world(dxf_file: Path) -> str:
    # Fixed seed: the pebble scatter is part of the world, not something the
    # randomizer re-rolls, so it has to come out the same on every generate.
    pebbles = random.Random(GRAVEL_SEED)
    bales = dxf_obstacle_bales(dxf_file)
    pothole_bumps = dxf_pothole_bumps(dxf_file)

    ground_x, ground_y = to_world(97.5, 124.0)
    return (
        '<?xml version="1.0"?>\n'
        '<sdf version="1.10">\n'
        f'  <world name="cfr_obstacle_course">\n'
        '    <physics name="1ms" type="ignored">\n'
        "      <max_step_size>0.001</max_step_size>\n"
        "      <real_time_factor>1.0</real_time_factor>\n"
        "    </physics>\n"
        '    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>\n'
        '    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>\n'
        '    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>\n'
        f"    {SYSTEM_MARKER}\n"
        "\n"
        '    <light type="directional" name="sun">\n'
        "      <pose>0 0 20 0 0 0</pose>\n"
        "      <cast_shadows>true</cast_shadows>\n"
        "      <diffuse>0.8 0.8 0.8 1</diffuse>\n"
        "      <specular>0.2 0.2 0.2 1</specular>\n"
        "      <direction>-0.4 0.2 -0.9</direction>\n"
        "    </light>\n"
        "\n"
        f'    <model name="ground"><static>true</static><pose>{ground_x:.3f} {ground_y:.3f} 0 0 0 0</pose><link name="link">\n'
        '      <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>30 24</size></plane></geometry><surface><friction><ode><mu>50</mu></ode><bullet><friction>1</friction><rolling_friction>0.001</rolling_friction></bullet></friction></surface></collision>\n'
        '      <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>30 24</size></plane></geometry><material><diffuse>0.16 0.25 0.13 1</diffuse></material></visual>\n'
        "    </link></model>\n"
        "\n"
        + build_bales(bales)
        + "\n"
        + build_bridge()
        + build_tunnel()
        + build_helix()
        + build_car_wash()
        + build_gravel(pebbles)
        + build_potholes(pothole_bumps)
        + build_ramp(
            "pothole_entry_ramp",
            POTHOLE_ENTRY_RAMP,
            "ramp_pothole_entry.stl",
            1.5 * INCH,
            POTHOLE,
        )
        + build_ramp(
            "pothole_exit_ramp",
            POTHOLE_EXIT_RAMP,
            "ramp_pothole_entry.stl",
            1.5 * INCH,
            POTHOLE,
        )
        + build_ramp(
            "gravel_entry_ramp", GRAVEL_ENTRY_RAMP, "ramp_up.stl", 2 * INCH, GRAVEL
        )
        + build_ramp(
            "gravel_exit_ramp", GRAVEL_EXIT_RAMP, "ramp_narrowing.stl", 2 * INCH, GRAVEL
        )
        + build_bank()
        + "\n"
        + build_buckets()
        + build_hoops()
        + build_start_signal()
        + "\n"
        + build_vehicle()
        + "  </world>\n"
        "</sdf>\n"
    )


def dxf_pothole_bumps(dxf_file: Path) -> list[tuple[float, float]]:
    """Bump centers on the OBJECTS layer, in DXF feet.

    Bumps and holes are both drawn as a pair of concentric circles inside the
    pothole board, and both are the 6 in feature the drawing annotates.  What
    separates them is the second circle: a bump is drawn with a wider base
    ring around it, a hole with a narrower one inside it.  So the 7.5 in class
    is the bumps, and the 4.25 in class is the holes -- which are cut into the
    board mesh rather than placed.

    The drawing has 16 bumps where the CAD assembly has 14, so the holes in
    the board mesh do not line up one for one with the 12 the drawing shows.
    Both are annotated "arbitrarily placed", so neither count is load bearing,
    and the drawing is the newer of the two.
    """
    bumps = []
    for entity_type, tags in dxf_entities(dxf_file):
        if entity_type != "CIRCLE":
            continue
        if next((value for code, value in tags if code == "8"), "") != "OBJECTS":
            continue
        radius = float(next(value for code, value in tags if code == "40"))
        if abs(radius - 0.3125) > 1e-6:
            continue
        x = float(next(value for code, value in tags if code == "10"))
        y = float(next(value for code, value in tags if code == "20"))
        bumps.append((x, y))
    if not 8 <= len(bumps) <= 24:
        raise RuntimeError(f"Implausible pothole bump count: {len(bumps)}")
    return bumps


def build_layout_yaml() -> str:
    """Randomization bounds, in world meters, for obstacle_randomizer_node.

    Shaped as a ROS 2 parameter file, which means nested maps of scalars and
    arrays only.  A list of hoops would not load, so the hoops are named in
    `hoops.names` and each gets its own map keyed by that name.
    """
    x0, y0 = to_world(BUCKET_REGION[0], BUCKET_REGION[1])
    x1, y1 = to_world(BUCKET_REGION[2], BUCKET_REGION[3])
    parking_x, parking_y = parking_spot(0)

    nominal = ""
    for index, bucket in enumerate(BUCKET_NOMINAL):
        world_x, world_y = to_world(*bucket)
        nominal += f"          bucket_{index}: [{world_x:.4f}, {world_y:.4f}]" + "\n"

    signal = start_signal.layout_block(SIGNAL_POSITION, LANE_HEADING)
    names = ", ".join(f"hoop_{index}" for index in range(len(HOOPS)))
    hoops = ""
    for index, (fixed, travel_from, travel_to, axis, nominal_at) in enumerate(HOOPS):
        if axis == "y":
            a_x, a_y = to_world(fixed, travel_from)
            b_x, b_y = to_world(fixed, travel_to)
            nominal_x, nominal_y = to_world(fixed, nominal_at)
            yaw = math.pi / 2
        else:
            a_x, a_y = to_world(travel_from, fixed)
            b_x, b_y = to_world(travel_to, fixed)
            nominal_x, nominal_y = to_world(nominal_at, fixed)
            yaw = 0.0
        hoops += "\n".join(
            [
                f"      hoop_{index}:",
                f"        yaw: {yaw:.5f}",
                f"        from: [{a_x:.4f}, {a_y:.4f}]",
                f"        to: [{b_x:.4f}, {b_y:.4f}]",
                f"        nominal: [{nominal_x:.4f}, {nominal_y:.4f}]",
                f"        base_length: {HOOP_BASE_LENGTH}",
                "",
            ]
        )

    return f"""# Generated by generate_obstacle_course.py -- do not edit by hand.
#
# Bounds for the elements the course varies between runs, in world meters.
# The randomizer reads them from here so that it and the world cannot drift
# apart when the DXF changes.
obstacle_randomizer:
  ros__parameters:
    world: cfr_obstacle_course
    # 0 draws a count in [count_min, count_max]; anything else is used as is.
    bucket_count: 0
    # -1 draws a fresh seed on every randomize call.
    seed: -1
    buckets:
      # The section the buckets may stand in.  wall_clearance shrinks it by
      # the gap the drawing keeps between a bucket and the bale walls; every
      # bucket in the drawing sits exactly that far off the nearest wall.
      region_min: [{min(x0, x1):.4f}, {min(y0, y1):.4f}]
      region_max: [{max(x0, x1):.4f}, {max(y0, y1):.4f}]
      wall_clearance: {BUCKET_WALL_CLEARANCE * FOOT:.4f}
      min_spacing: {BUCKET_MIN_SPACING * FOOT:.4f}
      count_min: {BUCKET_MIN}
      count_max: {BUCKET_MAX}
      default_count: {BUCKET_DEFAULT_COUNT}
      # Buckets a draw leaves out stand here, off the course, spaced along -x.
      parking: [{parking_x:.4f}, {parking_y:.4f}]
      parking_pitch: {BUCKET_PARKING_PITCH}
      # Where the drawing itself puts them; the reset service restores these.
      nominal:
{nominal}    hoops:
      names: [{names}]
{hoops}{signal}"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Gazebo Obstacle Course from its DXF layout"
    )
    parser.add_argument(
        "dxf_file",
        type=Path,
        help="site-layout DXF containing Speed and Obstacle courses",
    )
    args = parser.parse_args()

    worlds = PACKAGE / "worlds"
    config = PACKAGE / "config"
    worlds.mkdir(parents=True, exist_ok=True)

    world = worlds / "obstacle_course.sdf"
    # newline="\n" on every write: the repository is LF throughout, and a
    # Windows dev host would otherwise turn the shebang of anything generated
    # into CRLF and the SDF into a file git sees as wholly rewritten.
    world.write_text(build_world(args.dxf_file), newline="\n")
    print(f"wrote {world.relative_to(PACKAGE.parent)}")

    layout = config / "obstacle_course_layout.yaml"
    layout.write_text(build_layout_yaml(), newline="\n")
    print(f"wrote {layout.relative_to(PACKAGE.parent)}")


if __name__ == "__main__":
    main()
