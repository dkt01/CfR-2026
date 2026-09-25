"""Where the point-cloud segmentation fixtures are captured, and what they hold.

Shared by the two ends of the fixture pipeline: `jetson/scripts/
capture_segmentation_fixtures.py` renders one fixture per scenario in Gazebo,
and `test_cloud_segmentation.py` scores `cloud_segmentation` against them.
Keeping the table in one file is what stops the two drifting: a scenario the
tests expect but nobody captured is a failing test, not a silent skip.

A scenario is a *car* pose -- where the chassis sits and which way it points
-- not a camera pose. The capture tool drops the car onto the course's
driving surfaces (ramp, deck, helix, bank, pothole board, gravel) and works
out the pitch and roll from where its four wheels land, so "inclined on the
ramp" or "one tire in a pothole" is a statement about the course rather than
an angle somebody guessed. `wheels` overrides that for the pothole cases.

Part labels
-----------
Every visual in the capture world carries one of the PART_* ids below, and a
segmentation camera rendered alongside the ZED reports them per pixel. They
name what a surface *is*; which class the segmenter should give it also
depends on how the camera sees it (a ramp's top is drivable, its side is a
wall), and that mapping lives in the test, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Unlabeled pixels, which in practice are only sky: every visual in the
# capture world is given one of the ids below.
PART_NONE = 0
PART_FLOOR = 1
PART_BALE = 2
PART_BUCKET = 3
PART_HOOP = 4
PART_CARWASH_ARCH = 5
PART_CARWASH_RIBBON = 6
PART_CARWASH_BASE = 7
# The overpass ramp's slab and the bridge deck.
PART_RAMP = 8
# Every 6 in guard rail: ramp, deck and both sides of the helix.
PART_RAIL = 9
PART_HELIX = 10
PART_TUNNEL = 11
PART_BANK = 12
PART_POTHOLE_BOARD = 13
PART_POTHOLE_BUMP = 14
# The gravel tray and its pebbles.
PART_GRAVEL = 15
# The 1.5-2 in wedges onto the pothole board and the gravel tray.
PART_SMALL_RAMP = 16
PART_SIGNAL = 17

PART_NAMES = {
    PART_NONE: "none",
    PART_FLOOR: "floor",
    PART_BALE: "bale",
    PART_BUCKET: "bucket",
    PART_HOOP: "hoop",
    PART_CARWASH_ARCH: "carwash_arch",
    PART_CARWASH_RIBBON: "carwash_ribbon",
    PART_CARWASH_BASE: "carwash_base",
    PART_RAMP: "ramp",
    PART_RAIL: "rail",
    PART_HELIX: "helix",
    PART_TUNNEL: "tunnel",
    PART_BANK: "bank",
    PART_POTHOLE_BOARD: "pothole_board",
    PART_POTHOLE_BUMP: "pothole_bump",
    PART_GRAVEL: "gravel",
    PART_SMALL_RAMP: "small_ramp",
    PART_SIGNAL: "signal",
}

# (model name, visual name prefix or None for every visual) -> part id. First
# match wins, so the specific entries for a model come before its catch-all.
VISUAL_PARTS = [
    ("ground", None, PART_FLOOR),
    ("course_bales", None, PART_BALE),
    ("gap_bale_", None, PART_BALE),
    ("wide_bale_", None, PART_BALE),
    ("bucket_", None, PART_BUCKET),
    ("hoop_", None, PART_HOOP),
    ("car_wash", "strip_", PART_CARWASH_RIBBON),
    ("car_wash", "base", PART_CARWASH_BASE),
    ("car_wash", None, PART_CARWASH_ARCH),
    ("bridge", "ramp_rail", PART_RAIL),
    ("bridge", "deck_rail", PART_RAIL),
    ("bridge", None, PART_RAMP),
    ("helix", "inner_rail", PART_RAIL),
    ("helix", "outer_rail", PART_RAIL),
    ("helix", None, PART_HELIX),
    ("tunnel", None, PART_TUNNEL),
    ("bank", None, PART_BANK),
    ("pothole_section", "bump_", PART_POTHOLE_BUMP),
    ("pothole_section", None, PART_POTHOLE_BOARD),
    ("pothole_entry_ramp", None, PART_SMALL_RAMP),
    ("pothole_exit_ramp", None, PART_SMALL_RAMP),
    ("gravel_entry_ramp", None, PART_SMALL_RAMP),
    ("gravel_exit_ramp", None, PART_SMALL_RAMP),
    ("gravel_box", None, PART_GRAVEL),
    ("start_signal_", None, PART_SIGNAL),
]


def part_for_visual(model: str, visual: str) -> int | None:
    """The part id a visual renders as, or None if the table has no entry."""
    for model_prefix, visual_prefix, part in VISUAL_PARTS:
        if not model.startswith(model_prefix):
            continue
        if visual_prefix is None or visual.startswith(visual_prefix):
            return part
    return None


@dataclass(frozen=True)
class Scenario:
    """One captured view.

    `x`, `y` are the chassis origin in world meters and `yaw` its heading in
    degrees. `wheels` pins a wheel's contact height (meters, world z) where
    the surface model would put it somewhere else -- a tire down a pothole
    recess, or up on a bump. `moves` repositions movable models (buckets,
    gap bales, hoops) for this capture only, as (x, y, yaw degrees).
    `expect` names the parts this view exists to test; the test asserts each
    is in view and classified as the scenario says.
    """

    name: str
    course: str
    x: float
    y: float
    yaw: float
    note: str
    wheels: dict = field(default_factory=dict)
    moves: dict = field(default_factory=dict)
    expect: tuple = ()
    # The car is up on the overpass or the helix. Where two drivable
    # surfaces overlap in plan -- the deck over the tunnel -- the capture
    # tool otherwise stands the car on the lower one.
    elevated: bool = False


# Wheel contact points in the chassis frame, from the vehicle model: axles at
# +-0.162 m, wheels at +-0.145 m, chassis origin on the ground plane between
# them. The camera sits 0.315 m ahead of that origin and 0.20 m above it.
WHEEL_CONTACTS = {
    "front_left": (0.162, 0.145),
    "front_right": (0.162, -0.145),
    "rear_left": (-0.162, 0.145),
    "rear_right": (-0.162, -0.145),
}
CAMERA_OFFSET = (0.315, 0.0, 0.20)

# A pothole recess is dished 18 mm into the 1.5 in board, down to 0.020 m;
# a bump lifts a tire 0.75 in above the board.
HOLE_FLOOR = 0.020
BUMP_TOP = 0.0572

# Pothole recess and bump centers used below, world meters: recesses from
# generate_obstacle_course.pothole_full_height_cells, bumps from the world's
# bump visuals.
HOLE_A = (1.867, -8.225)
HOLE_B = (0.951, -8.073)
BUMP_A = (1.1038, -8.5302)


def _car_origin_for_wheel(wheel: str, point: tuple, yaw_deg: float) -> tuple:
    """Chassis origin that puts `wheel`'s contact patch on `point`."""
    import math

    ox, oy = WHEEL_CONTACTS[wheel]
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return point[0] - (c * ox - s * oy), point[1] - (s * ox + c * oy)


_HOLE_CAR = _car_origin_for_wheel("front_left", HOLE_A, 0.0)
_BUMP_CAR = _car_origin_for_wheel("rear_right", BUMP_A, 0.0)
_DIAG_CAR = _car_origin_for_wheel("front_right", HOLE_B, 20.0)

SCENARIOS = [
    # ---------------------------------------------------------------- speed
    # The Speed Course is driven toward -x along its start straight; the
    # centerline in rl/bale_follower/course_path.json runs the other way.
    Scenario(
        "speed_start_grid",
        "speed",
        20.15,
        4.76,
        180.0,
        "the car's own start pose, start signal 3.1 m ahead and right",
        expect=("signal",),
    ),
    Scenario(
        "speed_signal_close",
        "speed",
        18.30,
        4.76,
        180.0,
        "start signal board 1.3 m ahead, filling the right of the frame",
        expect=("signal",),
    ),
    Scenario(
        "speed_signal_abeam",
        "speed",
        17.55,
        4.80,
        200.0,
        "passing the signal, turned toward it",
        expect=("signal",),
    ),
    Scenario(
        "speed_signal_over_wall",
        "speed",
        14.5,
        6.15,
        0.0,
        "return straight: the 1.2 m signal stands above the bale wall",
        expect=("signal",),
    ),
    Scenario(
        "speed_straight",
        "speed",
        26.5,
        4.75,
        180.0,
        "long straight, walls converging to the vanishing point",
    ),
    Scenario(
        "speed_straight_hugging_wall",
        "speed",
        24.0,
        4.97,
        180.0,
        "0.22 m off center toward the dividing wall",
    ),
    Scenario(
        "speed_facing_wall",
        "speed",
        24.0,
        4.75,
        90.0,
        "turned square to the dividing wall, 0.4 m of floor before it",
    ),
    Scenario(
        "speed_oblique_wall",
        "speed",
        24.0,
        4.75,
        140.0,
        "40 degrees off the straight, wall crossing the frame diagonally",
    ),
    Scenario(
        "speed_sweeper",
        "speed",
        8.58,
        3.96,
        198.0,
        "gentle left-hand sweeper",
    ),
    Scenario(
        "speed_hairpin_entry",
        "speed",
        4.64,
        -4.38,
        276.0,
        "tightest hairpin, apex wall close on the inside",
    ),
    Scenario(
        "speed_hairpin_apex",
        "speed",
        2.56,
        -6.16,
        176.0,
        "hairpin apex, bale end-caps straight ahead",
    ),
    Scenario(
        "speed_chicane",
        "speed",
        36.05,
        -4.73,
        106.0,
        "far-end hairpin, walls on three sides",
    ),
    Scenario(
        "speed_return_straight",
        "speed",
        21.23,
        6.15,
        0.0,
        "return straight, outer wall and dividing wall both close",
    ),
    # ------------------------------------------------------------- obstacle
    # Driving order: start straight, overpass ramp and deck, helix down,
    # tunnel, corridor south, gravel, bank, potholes, bucket section, gap
    # wall, hoops, car wash, back to the line.
    Scenario(
        "obs_start_grid",
        "obstacle",
        -0.70,
        0.0,
        0.0,
        "start pose: signal ahead-left, overpass ramp beyond it",
        expect=("signal", "ramp"),
    ),
    Scenario(
        "obs_signal_close",
        "obstacle",
        1.30,
        0.0,
        15.0,
        "turned toward the signal board, 1 m away",
        expect=("signal",),
    ),
    Scenario(
        "obs_ramp_approach",
        "obstacle",
        2.40,
        0.0,
        0.0,
        "ramp foot 1 m ahead, surface climbing out of the floor",
        expect=("ramp", "rail"),
    ),
    Scenario(
        "obs_ramp_straddle",
        "obstacle",
        3.43,
        0.0,
        0.0,
        "front wheels on the ramp, rear on the floor: nose up, half pitch",
        expect=("ramp", "rail"),
    ),
    Scenario(
        "obs_ramp_mid",
        "obstacle",
        5.00,
        0.0,
        0.0,
        "halfway up the 19% ramp, fully inclined",
        elevated=True,
        expect=("ramp", "rail"),
    ),
    Scenario(
        "obs_ramp_crest",
        "obstacle",
        6.60,
        0.0,
        0.0,
        "cresting onto the deck: nose leveling over the break",
        elevated=True,
        expect=("ramp",),
    ),
    Scenario(
        "obs_deck",
        "obstacle",
        7.60,
        0.0,
        0.0,
        "on the deck 0.64 m up, helix turning away ahead, floor far below",
        elevated=True,
        expect=("helix", "rail"),
    ),
    Scenario(
        "obs_helix_top",
        "obstacle",
        None,
        None,
        None,
        "helix, 10% of the way down",
        elevated=True,
        expect=("helix", "rail"),
    ),
    Scenario(
        "obs_helix_mid",
        "obstacle",
        None,
        None,
        None,
        "helix, halfway down, banked view of the spiral",
        elevated=True,
        expect=("helix", "rail"),
    ),
    Scenario(
        "obs_helix_bottom",
        "obstacle",
        None,
        None,
        None,
        "helix, 85% down, tunnel mouth coming into view",
        elevated=True,
        expect=("helix", "tunnel"),
    ),
    Scenario(
        "obs_tunnel_mouth",
        "obstacle",
        7.10,
        1.75,
        -90.0,
        "last of the helix, tunnel mouth filling the frame",
        expect=("tunnel",),
    ),
    Scenario(
        "obs_in_tunnel",
        "obstacle",
        7.10,
        0.55,
        -90.0,
        "inside the tunnel: walls both sides, roof overhead",
        expect=("tunnel",),
    ),
    Scenario(
        "obs_tunnel_exit",
        "obstacle",
        7.10,
        -0.95,
        -90.0,
        "just out of the tunnel, narrow corridor south",
    ),
    Scenario(
        "obs_narrow_corridor",
        "obstacle",
        7.12,
        -4.0,
        -90.0,
        "narrow corridor between tunnel and gravel",
    ),
    Scenario(
        "obs_gravel_approach",
        "obstacle",
        5.10,
        -9.90,
        180.0,
        "gravel entry ramp 0.75 m ahead, pebbles beyond",
        expect=("gravel", "small_ramp"),
    ),
    Scenario(
        "obs_gravel_on_ramp",
        "obstacle",
        3.95,
        -9.90,
        180.0,
        "on the gravel entry wedge",
        expect=("gravel",),
    ),
    Scenario(
        "obs_in_gravel",
        "obstacle",
        2.30,
        -9.92,
        180.0,
        "in the middle of the gravel pit",
        expect=("gravel",),
    ),
    Scenario(
        "obs_gravel_exit",
        "obstacle",
        0.70,
        -9.90,
        180.0,
        "on the gravel exit wedge, dropping back to the floor",
    ),
    Scenario(
        "obs_bank_approach",
        "obstacle",
        -2.80,
        -9.87,
        175.0,
        "banked turn ahead, its 0.56 m outer wall across the frame",
        expect=("bank",),
    ),
    Scenario(
        "obs_bank_entry",
        "obstacle",
        -4.00,
        -9.45,
        120.0,
        "climbing onto the bank, rolled toward its low edge",
        expect=("bank",),
    ),
    Scenario(
        "obs_bank_mid",
        "obstacle",
        -4.20,
        -8.90,
        85.0,
        "mid-bank, heading along it, rolled 9 degrees",
        expect=("bank",),
    ),
    Scenario(
        "obs_bank_exit",
        "obstacle",
        -3.75,
        -8.40,
        20.0,
        "leaving the bank toward the potholes",
    ),
    Scenario(
        "obs_pothole_approach",
        "obstacle",
        -1.60,
        -8.40,
        0.0,
        "pothole entry wedge ahead, board, recesses and bumps beyond",
        expect=("pothole_board", "pothole_bump"),
    ),
    Scenario(
        "obs_pothole_on_board",
        "obstacle",
        0.50,
        -8.25,
        0.0,
        "on the board among recesses and bumps",
        expect=("pothole_board", "pothole_bump"),
    ),
    Scenario(
        "obs_pothole_tire_in_recess",
        "obstacle",
        _HOLE_CAR[0],
        _HOLE_CAR[1],
        0.0,
        "front-left tire down a recess: nose-down and rolled left",
        wheels={"front_left": HOLE_FLOOR},
        expect=("pothole_board",),
    ),
    Scenario(
        "obs_pothole_tire_on_bump",
        "obstacle",
        _BUMP_CAR[0],
        _BUMP_CAR[1],
        0.0,
        "rear-right tire up on a bump",
        wheels={"rear_right": BUMP_TOP},
        expect=("pothole_board",),
    ),
    Scenario(
        "obs_pothole_diagonal",
        "obstacle",
        _DIAG_CAR[0],
        _DIAG_CAR[1],
        20.0,
        "front-right in a recess, rear-left on a bump: worst twist",
        wheels={"front_right": HOLE_FLOOR, "rear_left": BUMP_TOP},
        expect=("pothole_board",),
    ),
    Scenario(
        "obs_pothole_exit",
        "obstacle",
        2.85,
        -8.30,
        0.0,
        "on the exit wedge, corridor turn ahead",
    ),
    Scenario(
        "obs_corridor_north",
        "obstacle",
        4.55,
        -5.0,
        90.0,
        "corridor up the bucket section's east side",
    ),
    Scenario(
        "obs_bucket_entry",
        "obstacle",
        3.40,
        -1.55,
        180.0,
        "entering the bucket section, buckets scattered ahead",
        expect=("bucket",),
    ),
    Scenario(
        "obs_buckets_mid",
        "obstacle",
        1.45,
        -2.60,
        200.0,
        "among the buckets",
        expect=("bucket",),
    ),
    Scenario(
        "obs_bucket_close",
        "obstacle",
        3.11,
        -1.935,
        180.0,
        "bucket 0.5 m from the camera, dead ahead",
        expect=("bucket",),
    ),
    Scenario(
        "obs_gap_wall",
        "obstacle",
        0.40,
        -2.10,
        175.0,
        "gap-bale wall ahead with its one-bale opening",
        expect=("bale",),
    ),
    Scenario(
        "obs_through_gap",
        "obstacle",
        -0.45,
        -1.80,
        180.0,
        "in the gap, hoop 0 ahead",
        expect=("hoop",),
    ),
    Scenario(
        "obs_hoop0_far",
        "obstacle",
        -0.95,
        -1.71,
        180.0,
        "hoop 0 square on, 3 m",
        expect=("hoop",),
    ),
    Scenario(
        "obs_hoop0_mid",
        "obstacle",
        -2.40,
        -1.71,
        180.0,
        "hoop 0 square on, 1.2 m from the camera",
        expect=("hoop",),
    ),
    Scenario(
        "obs_hoop0_close",
        "obstacle",
        -3.25,
        -1.71,
        180.0,
        "hoop 0, 0.4 m: its top has left the frame",
        expect=("hoop",),
    ),
    Scenario(
        "obs_hoop0_oblique",
        "obstacle",
        -2.50,
        -1.10,
        205.0,
        "hoop 0 from 35 degrees off its axis",
        expect=("hoop",),
    ),
    Scenario(
        "obs_hoop0_edge_on",
        "obstacle",
        -3.94,
        -0.60,
        -90.0,
        "hoop 0 edge-on: its two uprights in line",
        expect=("hoop",),
    ),
    Scenario(
        "obs_hoops_in_line",
        "obstacle",
        -1.80,
        -1.95,
        183.0,
        "hoop 0 ahead with hoop 1 visible through and beyond it",
        expect=("hoop",),
    ),
    Scenario(
        "obs_hoop_bucket_behind",
        "obstacle",
        -2.40,
        -1.71,
        180.0,
        "bucket parked 0.6 m past hoop 0, seen through the hoop",
        moves={"bucket_6": (-4.55, -1.71, 0.0)},
        expect=("hoop", "bucket"),
    ),
    Scenario(
        "obs_hoop1_approach",
        "obstacle",
        -4.90,
        -2.40,
        180.0,
        "hoop 1 square on, 1.3 m",
        expect=("hoop",),
    ),
    Scenario(
        "obs_hoop2_approach",
        "obstacle",
        -8.135,
        -1.25,
        90.0,
        "hoop 2 after the turn north, 1.1 m",
        expect=("hoop",),
    ),
    Scenario(
        "obs_carwash_far",
        "obstacle",
        -6.10,
        0.25,
        -5.0,
        "car wash ahead, 2 m to the first arch",
        expect=("carwash_arch", "carwash_ribbon"),
    ),
    Scenario(
        "obs_carwash_near",
        "obstacle",
        -4.60,
        0.0,
        0.0,
        "0.45 m from the first curtain of ribbons",
        expect=("carwash_arch", "carwash_ribbon"),
    ),
    Scenario(
        "obs_carwash_inside",
        "obstacle",
        -2.95,
        0.0,
        0.0,
        "between arches, ribbons in every direction",
        expect=("carwash_ribbon",),
    ),
    Scenario(
        "obs_carwash_oblique",
        "obstacle",
        -5.10,
        -0.55,
        30.0,
        "car wash from 30 degrees off its axis",
        expect=("carwash_arch", "carwash_ribbon"),
    ),
    Scenario(
        "obs_carwash_bucket_behind",
        "obstacle",
        -4.60,
        0.0,
        0.0,
        "bucket parked just past the car wash's last arch",
        moves={"bucket_6": (-1.30, 0.10, 0.0)},
        expect=("carwash_ribbon", "bucket"),
    ),
    Scenario(
        "obs_carwash_exit",
        "obstacle",
        -1.40,
        0.05,
        0.0,
        "out of the car wash: start line, signal and ramp ahead",
        expect=("signal",),
    ),
]

# The helix scenarios are placed by how far down the spiral they are rather
# than by (x, y), because the useful numbers -- how far along, which way the
# tangent points -- are awkward to read off a plan. The capture tool resolves
# these from the generator's helix constants.
HELIX_FRACTIONS = {
    "obs_helix_top": 0.10,
    "obs_helix_mid": 0.50,
    "obs_helix_bottom": 0.85,
}

BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
