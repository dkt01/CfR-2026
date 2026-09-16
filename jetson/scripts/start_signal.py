"""The autonomous start signal, shared by both course generators.

Both courses start the same way: the car waits behind the line and begins
driving when the signal turns from red to green.  So both worlds carry the
same two models and the same randomiser block, and they live here rather than
in one generator with a copy in the other.

The signal is a fixed frame and an arm pair on a revolute joint.  The two arms
are 90 degrees apart about that joint, so whichever lies horizontal stands out
past the sky blue board and is the one the car sees; a quarter turn swaps
them.  It starts at 0, which is the angle the red arm is modelled at, so a
freshly loaded world always shows red.

The joint is what makes the turn a turn.  The arm sweeps at 90 degrees a
second, like the one on the course, and a detector has to cope with it part
way round -- so the motion has to happen at Gazebo's rate rather than in
whatever steps something outside can push in.  Stepping the model's pose
through ``set_pose`` was the obvious alternative and it does not work: each
call is a ``gz service`` subprocess costing about 340 ms, so a one second
sweep fits three poses and arrives as a stutter.  Here the node publishes a
joint setpoint on a bridged topic instead, which costs nothing, and the
position controller follows it.

The drawing now places the signal, and places it the same way on both
courses: three bales down the wall from the start line -- annotated
"approximately 8 ft from start line" -- with the bales moved so that the
signal stands in line with the inner edge of the bale border.  So
``position`` below derives it from each course's start line and lane edge
rather than each generator choosing a spot.

Standing it at the lane edge is what makes the rest of this simple.  The board
is 32 in wide and its arms sweep in its plane, so it stands square across the
lane and faces back up it -- see ``yaw_across``, which is derived from the
path's heading and not from where the car happens to wait.  That puts the
board's width across the lane and its 4 in of depth along it.  Centred on the
lane edge, half the board would be in the path, so it sits half a width
outboard: inner end flush with the edge, body where the wall was -- hence the
bales moving.  ``clear_bales`` slides the one bale it displaces along the wall
until it clears, and the board's 4 in of depth stands in the gap.

Two constraints are left, and both are slack now:

* Inside the camera's 110 degree field.  At 8 ft down a 32 in lane the arm is
  about 11 degrees off the lane axis, against a 55 degree half-field.
* The sight line from the camera, only 8 in off the ground, has to clear the
  14 in bale wall it grazes on the way.  It used to pass 24 mm over the bales,
  because the signal was out beyond the wall and the ray had to climb the
  whole way; from inside the lane it clears them by a third of a metre.

``scripts/check_signal_sightline.py`` re-checks both against the generated
worlds, along with the placement itself.
"""

from __future__ import annotations

import math

MESH_URI = "model://cfr_arduino_bridge/meshes"

FOOT = 0.3048
INCH = 0.0254

# Height of the pivot the arms turn about, from the CAD.
PIVOT_HEIGHT = 0.813

# The sky blue board, from the CAD: 32 in wide, 4 in deep, 48 in tall.  Its
# width is the reason the signal stands off the lane edge rather than on it --
# see the module docstring -- and check_signal_sightline.py measures its
# footprint against the bales.
FRAME_WIDTH = 0.813
FRAME_DEPTH = 0.10
FRAME_HEIGHT = 1.219

# How much further than just-touching a displaced bale is slid.  The minimum
# leaves it exactly against the board, and the 0.1 mm rounding the SDF writes
# poses at can put it a hair back inside; 5 mm of daylight also reads as a
# gap rather than as a bale leaning on the signal.
BALE_CLEARANCE = 0.005

# How far down the lane the signal stands, measured from the start line.  The
# drawing says three bales along the wall and annotates it "approximately
# 8 ft"; the walls are laid with the bales overlapping slightly, which puts
# three of them at about 8.6 ft, so the annotated figure is the one used and
# the two agree to within half a bale.
SIGNAL_DISTANCE = 8 * FOOT

# Joint angles the two arms lie horizontal at.  Red is the rest position, so
# the world loads showing red without anything having to command it.
RED_ANGLE = 0.0
GREEN_ANGLE = math.pi / 2

# Gazebo topic the joint setpoint arrives on.  simulation.launch.py bridges it
# to std_msgs/Float64 under the same name, so the randomiser can publish to it
# as an ordinary ROS topic.
COMMAND_TOPIC = "/start_signal/arm"

# The frame is painted sky blue on the course; only the arms carry the red and
# green the car is looking for.
SKY_BLUE = "0.53 0.81 0.92 1"
RED = "0.85 0.09 0.07 1"
GREEN = "0.10 0.70 0.20 1"


def yaw_across(heading: float) -> float:
    """Yaw that stands the signal square across a path running along `heading`.

    The frame mesh faces along its own -y, so turning that to look back down
    the path -- at a car driving away along `heading` -- is a quarter turn
    short of the heading itself.

    Square across the path, and deliberately not aimed at the point the car
    waits at.  The two are not the same thing: the board stands off to the
    side of a 32 in lane, so a car on the centreline is a good 16 degrees off
    the perpendicular, and aiming at it would cant the board, its arms and its
    footprint by that much.  The sign on the course stands square to the path,
    and a signal 8 ft away is read the same either way.
    """
    return heading - math.pi / 2


def position(
    line: tuple[float, float], heading: float, lane_edge: float
) -> tuple[float, float]:
    """Where the signal stands, from the start line and the lane's inner edge.

    `line` is a point on the start/finish line, `heading` the direction the car
    drives away from it, and `lane_edge` the distance from the lane centreline
    to the inner edge of the bale border on the car's left -- which is the side
    the drawing puts the signal on, on both courses.

    The board is centred half a width outboard of that edge, so its inner end
    finishes flush with the edge and none of it reaches into the path.
    """
    forward = (math.cos(heading), math.sin(heading))
    left = (-forward[1], forward[0])
    offset = lane_edge + FRAME_WIDTH / 2
    return (
        line[0] + forward[0] * SIGNAL_DISTANCE + left[0] * offset,
        line[1] + forward[1] * SIGNAL_DISTANCE + left[1] * offset,
    )


def box_axes(yaw: float):
    """A box's own length and width directions, in world coordinates."""
    return (math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))


def box_span(pose: tuple, size: tuple, axis: tuple) -> tuple[float, float]:
    """A box's shadow on `axis`, as (low, high)."""
    middle = pose[0] * axis[0] + pose[1] * axis[1]
    half = sum(
        abs(direction[0] * axis[0] + direction[1] * axis[1]) * extent / 2
        for direction, extent in zip(box_axes(pose[2]), size)
    )
    return middle - half, middle + half


def slide_clear(
    bale: tuple, bale_size: tuple, frame: tuple, frame_size: tuple
) -> float:
    """How far to slide `bale` along its own length to get it out of `frame`.

    Signed, in metres, and zero when the two are already apart.  Both boxes are
    (x, y, yaw) with (length, width): a bale can only move along the wall it is
    part of, so this is the separating-axis overlap divided by how much of that
    axis the slide direction covers, minimised over the four axes.
    """
    slide = box_axes(bale[2])[0]
    best = None
    for yaw in (bale[2], frame[2]):
        for axis in box_axes(yaw):
            low, high = box_span(bale, bale_size, axis)
            frame_low, frame_high = box_span(frame, frame_size, axis)
            if low >= frame_high or frame_low >= high:
                return 0.0  # apart on this axis, so not touching at all
            covered = slide[0] * axis[0] + slide[1] * axis[1]
            if abs(covered) < 1e-9:
                continue  # sliding does not move the box along this axis
            for gap in (frame_high - low, frame_low - high):
                candidate = gap / covered
                if best is None or abs(candidate) < abs(best):
                    best = candidate
    return best or 0.0


def boxes_overlap(a: tuple, a_size: tuple, b: tuple, b_size: tuple) -> bool:
    """Whether two (x, y, yaw) boxes with (length, width) share any ground.

    Expressed through `slide_clear`, which returns zero exactly when the two
    are apart: a box's own length axis is always one of the axes tested, so an
    overlap always comes back as a real slide.
    """
    return slide_clear(a, a_size, b, b_size) != 0.0


def clear_bales(
    bales: list[tuple[float, float, float]],
    signal: tuple[float, float],
    heading: float,
    bale_size: tuple[float, float],
) -> list[tuple[float, float, float]]:
    """`bales` with any the signal's board stands in slid along their wall.

    The drawing moves the bales so the signal can stand in line with the inner
    edge of the border; the DXF this repository has predates that, so the move
    happens here.  Each bale in the way slides along its own length -- the
    direction its wall runs -- by the least that gets it clear, which leaves
    the wall continuous and the board's 4 in of depth standing in the gap.

    Sliding a bale into its neighbour is not checked for: the walls are laid
    with the bales already overlapping by an inch or two, and
    check_signal_sightline.py is what confirms the result clears the signal.
    """
    frame = (signal[0], signal[1], yaw_across(heading))
    frame_size = (FRAME_WIDTH, FRAME_DEPTH)
    moved = []
    for x, y, yaw in bales:
        slide = slide_clear((x, y, yaw), bale_size, frame, frame_size)
        if slide:
            slide += math.copysign(BALE_CLEARANCE, slide)
        moved.append((x + math.cos(yaw) * slide, y + math.sin(yaw) * slide, yaw))
    return moved


def mesh_visual(name: str, pose: tuple, stl: str, colour: str, indent: str) -> str:
    pose_text = " ".join(f"{value:.4f}" for value in pose)
    return (
        f'{indent}<visual name="{name}_visual"><pose>{pose_text}</pose>'
        f"<geometry><mesh><uri>{MESH_URI}/{stl}</uri></mesh></geometry>"
        f"<material><diffuse>{colour}</diffuse></material></visual>\n"
    )


def models(position: tuple[float, float], heading: float) -> str:
    """SDF for the frame and the jointed arm pair, square across the path."""
    x, y = position
    yaw = yaw_across(heading)

    frame = mesh_visual(
        "frame", (0, 0, 0, 0, 0, 0), "start_signal_frame.stl", SKY_BLUE, " " * 8
    )
    # Collision only: drawing this box as well would put it in front of the
    # mesh and hide the arms completely.
    frame += (
        '        <collision name="post_collision">'
        f"<pose>0 0 {FRAME_HEIGHT / 2:.2f} 0 0 0</pose><geometry><box>"
        f"<size>{FRAME_WIDTH:.3f} {FRAME_DEPTH:.2f} {FRAME_HEIGHT:.3f}</size>"
        "</box></geometry></collision>\n"
    )

    arms = mesh_visual(
        "red", (0, 0, 0, 0, 0, 0), "start_signal_arm_red.stl", RED, " " * 10
    )
    arms += mesh_visual(
        "green",
        (0, 0, 0, 0, GREEN_ANGLE, 0),
        "start_signal_arm_green.stl",
        GREEN,
        " " * 10,
    )

    return (
        "    <!-- Start signal.  The arms turn on a revolute joint, commanded\n"
        f"         over {COMMAND_TOPIC}; see scripts/start_signal.py. -->\n"
        f'    <model name="start_signal_frame"><static>true</static>'
        f"<pose>{x:.4f} {y:.4f} 0 0 0 {yaw:.5f}</pose>\n"
        f'      <link name="link">\n{frame}      </link>\n'
        "    </model>\n"
        f'    <model name="start_signal_arms">\n'
        f"      <pose>{x:.4f} {y:.4f} {PIVOT_HEIGHT:.4f} 0 0 {yaw:.5f}</pose>\n"
        # Not static, because a static model has no joints to turn -- pinned
        # to the world instead, which holds it just as still.
        '      <link name="pivot">\n'
        "        <inertial><mass>0.1</mass><inertia><ixx>0.001</ixx>"
        "<iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>\n"
        "      </link>\n"
        '      <link name="arms">\n'
        # Gravity off: this is a prop on a driven joint, and a 0.8 m arm left
        # to hang would droop between the world loading and the first command.
        "        <gravity>false</gravity>\n"
        "        <inertial><mass>0.5</mass><inertia><ixx>0.002</ixx>"
        "<iyy>0.03</iyy><izz>0.03</izz></inertia></inertial>\n"
        f"{arms}"
        "      </link>\n"
        '      <joint name="world_joint" type="fixed">'
        "<parent>world</parent><child>pivot</child></joint>\n"
        f'      <joint name="arm_joint" type="revolute">\n'
        "        <parent>pivot</parent><child>arms</child>\n"
        "        <axis><xyz>0 1 0</xyz>\n"
        f"          <limit><lower>{RED_ANGLE:.5f}</lower><upper>{GREEN_ANGLE:.5f}</upper>"
        "<effort>50</effort><velocity>5</velocity></limit>\n"
        "        </axis>\n"
        "      </joint>\n"
        # A force PID, deliberately, and not use_velocity_commands: in
        # velocity mode the joint runs past the setpoint to whichever limit
        # it is heading for and jams there for good, so the arm turns red to
        # green once and never comes back.  The gains are gentle because the
        # arm is light and the setpoint arrives as a ramp, not a step.
        '      <plugin filename="gz-sim-joint-position-controller-system"'
        ' name="gz::sim::systems::JointPositionController">\n'
        "        <joint_name>arm_joint</joint_name>\n"
        f"        <topic>{COMMAND_TOPIC}</topic>\n"
        "        <p_gain>12</p_gain>\n"
        "        <i_gain>0.2</i_gain>\n"
        "        <d_gain>2.0</d_gain>\n"
        "      </plugin>\n"
        "    </model>\n"
    )


def layout_block(
    position: tuple[float, float], heading: float, indent: str = "    "
) -> str:
    """The `start_signal:` parameters for obstacle_randomizer_node."""
    return (
        f"{indent}start_signal:\n"
        f"{indent}  # Joint setpoint topic, bridged from Gazebo by\n"
        f"{indent}  # simulation.launch.py.  The randomiser ramps the setpoint\n"
        f"{indent}  # across it rather than commanding the far end, so the arm\n"
        f"{indent}  # turns at a stated rate instead of as fast as it can.\n"
        f"{indent}  topic: {COMMAND_TOPIC}\n"
        f"{indent}  red_angle: {RED_ANGLE:.5f}\n"
        f"{indent}  green_angle: {GREEN_ANGLE:.5f}\n"
        f"{indent}  # Where the signal stands, for reference; the arm is moved\n"
        f"{indent}  # through the joint, so nothing here is commanded as a pose.\n"
        f"{indent}  pose: [{position[0]:.4f}, {position[1]:.4f}, {PIVOT_HEIGHT:.4f}]\n"
        f"{indent}  yaw: {yaw_across(heading):.5f}\n"
    )


def layout_file(
    world: str,
    position: tuple[float, float],
    heading: float,
    generator: str,
) -> str:
    """A whole parameter file for a course whose signal is its only variable."""
    return (
        f"# Generated by {generator} -- do not edit by hand.\n"
        "#\n"
        "# The Speed Course varies nothing between runs except the start signal,\n"
        "# so obstacle_randomizer_node runs here with only its signal services\n"
        "# live; randomize and reset have nothing to move and say so.\n"
        "obstacle_randomizer:\n"
        "  ros__parameters:\n"
        f"    world: {world}\n" + layout_block(position, heading)
    )
