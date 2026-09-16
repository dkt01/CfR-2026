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

The drawing places no signal -- only the CAD has one -- so each course picks
its own spot, subject to the same three constraints:

* Outside the bale walls, since the lane itself is solid bale either side.
* Inside the camera's 110 degree field rather than off the edge of it.
* Placed so the sight line from the camera, only 8 in off the ground, clears
  the 14 in bale wall between the two.  This is the binding one, and it wants
  the signal *down* the lane rather than out to the side: moving it sideways
  buys field of view but costs height at the wall, because the ray has to have
  finished climbing by the time it gets there.

``scripts/check_signal_sightline.py`` re-checks all three against the
generated worlds.
"""

from __future__ import annotations

import math

MESH_URI = "model://cfr_arduino_bridge/meshes"

# Height of the pivot the arms turn about, from the CAD.
PIVOT_HEIGHT = 0.813

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


def yaw_towards(position: tuple[float, float], target: tuple[float, float]) -> float:
    """Yaw that turns the signal's face towards `target`.

    The frame mesh faces along its own -y, so turning that to point down a
    bearing is a quarter turn past the bearing itself.
    """
    bearing = math.atan2(target[1] - position[1], target[0] - position[0])
    return bearing + math.pi / 2


def mesh_visual(name: str, pose: tuple, stl: str, colour: str, indent: str) -> str:
    pose_text = " ".join(f"{value:.4f}" for value in pose)
    return (
        f'{indent}<visual name="{name}_visual"><pose>{pose_text}</pose>'
        f"<geometry><mesh><uri>{MESH_URI}/{stl}</uri></mesh></geometry>"
        f"<material><diffuse>{colour}</diffuse></material></visual>\n"
    )


def models(position: tuple[float, float], target: tuple[float, float]) -> str:
    """SDF for the frame and the jointed arm pair, facing `target`."""
    x, y = position
    yaw = yaw_towards(position, target)

    frame = mesh_visual(
        "frame", (0, 0, 0, 0, 0, 0), "start_signal_frame.stl", SKY_BLUE, " " * 8
    )
    # Collision only: drawing this box as well would put it in front of the
    # mesh and hide the arms completely.
    frame += (
        '        <collision name="post_collision"><pose>0 0 0.61 0 0 0</pose>'
        "<geometry><box><size>0.813 0.10 1.219</size></box></geometry></collision>\n"
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
    position: tuple[float, float], target: tuple[float, float], indent: str = "    "
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
        f"{indent}  yaw: {yaw_towards(position, target):.5f}\n"
    )


def layout_file(
    world: str,
    position: tuple[float, float],
    target: tuple[float, float],
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
        f"    world: {world}\n" + layout_block(position, target)
    )
