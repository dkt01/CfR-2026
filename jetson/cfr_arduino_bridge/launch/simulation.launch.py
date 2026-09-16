"""Shared Gazebo Harmonic bringup behind the two per-course launch files.

Prefer `speed_course.launch.py` or `obstacle_course.launch.py`, which name a
world and its randomizer layout.  This file is what they both include, and is
still usable directly with `world:=`.
"""

import tempfile
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Both worlds carry these two markers, one at world level and one inside the
# vehicle's chassis link.  Gazebo ignores its own default server config as
# soon as a world declares plugins of its own, so the sensors system has to be
# written into the world file -- there is no launch argument for it.  Filling
# the markers in here beats keeping a second copy of each world: the Speed
# Course's world is maintained by hand, and a derived copy of a hand
# maintained file goes stale the first time somebody edits one and not the
# other.
SYSTEM_MARKER = "<!-- cfr:sensors-system -->"
CAMERA_MARKER = "<!-- cfr:sensors-camera -->"

SENSORS_SYSTEM = """<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>"""

# One rgbd_camera rather than a color and a depth sensor: they would share a
# calibration anyway, and Gazebo publishes image, depth_image, points and
# camera_info off this one.  simulation.launch.py's bridge renames them to the
# ZED's topics below.
SENSORS_CAMERA = """<sensor name="zed2i" type="rgbd_camera">
            <pose>0.315 0 0.20 0 0 0</pose>
            <always_on>1</always_on>
            <update_rate>15</update_rate>
            <topic>/zed/gz/rgbd</topic>
            <camera>
              <horizontal_fov>1.91986</horizontal_fov>
              <image><width>640</width><height>360</height><format>R8G8B8</format></image>
              <clip><near>0.2</near><far>20</far></clip>
            </camera>
          </sensor>"""


def resolve_world(context, *_args, **_kwargs):
    """Hand Gazebo the world, with the rendered sensors switched on or not.

    Without `sensors:=true` the world is used exactly as it sits in the
    package, markers and all -- an XML comment costs nothing.  With it, the
    markers are replaced and the result written beside the other simulation
    scratch files, because Gazebo takes a path and not a string.
    """
    world = Path(context.perform_substitution(LaunchConfiguration("world")))
    if context.perform_substitution(LaunchConfiguration("sensors")).lower() not in (
        "true",
        "1",
    ):
        return [world]

    text = world.read_text()
    for marker, replacement in (
        (SYSTEM_MARKER, SENSORS_SYSTEM),
        (CAMERA_MARKER, SENSORS_CAMERA),
    ):
        if marker not in text:
            raise RuntimeError(
                f"{world.name} has no {marker}, so sensors:=true cannot add the "
                "rendered ZED to it"
            )
        text = text.replace(marker, replacement, 1)

    scratch = Path(tempfile.gettempdir()) / "cfr_sim"
    scratch.mkdir(parents=True, exist_ok=True)
    rendered = scratch / world.name
    rendered.write_text(text)
    return [rendered]


def generate_launch_description():
    package_share = FindPackageShare("cfr_arduino_bridge")
    params_file = PathJoinSubstitution([package_share, "config", "arduino_bridge.yaml"])
    world_file = PathJoinSubstitution([package_share, "worlds", "speed_course.sdf"])

    params_arg = DeclareLaunchArgument("params_file", default_value=params_file)
    world_arg = DeclareLaunchArgument("world", default_value=world_file)
    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="false",
        description="Start Gazebo GUI; requires an authorized host display",
    )
    websocket_arg = DeclareLaunchArgument(
        "websocket",
        default_value="false",
        description="Start gzweb-compatible WebSocket server on port 9002",
    )
    sensors_arg = DeclareLaunchArgument(
        "sensors",
        default_value="false",
        description=(
            "Render the ZED RGB-D pair.  Needs a render context: a GPU, or "
            "LIBGL_ALWAYS_SOFTWARE=1 for llvmpipe at reduced frame rate"
        ),
    )
    world_name_arg = DeclareLaunchArgument(
        "world_name",
        default_value="cfr_speed_course",
        # The <world name> inside the SDF, which is not derivable from its
        # path.  teleport_api and the randomizer both address Gazebo services
        # under it.
        description="Name of the world inside the SDF",
    )
    randomizer_arg = DeclareLaunchArgument(
        "randomizer",
        default_value="false",
        description="Start obstacle_randomizer_node; needs layout_file",
    )
    layout_arg = DeclareLaunchArgument(
        "layout_file",
        default_value=PathJoinSubstitution(
            [package_share, "config", "obstacle_course_layout.yaml"]
        ),
        description="Bounds for the course's variable elements",
    )

    # Mesh URIs in the worlds are model://cfr_arduino_bridge/meshes/..., which
    # Gazebo resolves by looking for a directory called cfr_arduino_bridge on
    # this path -- so it wants the share root, one level above the package's
    # own share directory.
    resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        [
            PathJoinSubstitution([package_share, ".."]),
            ":",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )

    def gazebo_actions(context, *args, **kwargs):
        world = str(resolve_world(context)[0])
        server = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
                )
            ),
            launch_arguments={"gz_args": f"-r -s -v 3 {world}"}.items(),
        )
        gui = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
                )
            ),
            launch_arguments={"gz_args": f"-g -v 3 {world}"}.items(),
            condition=IfCondition(LaunchConfiguration("gui")),
        )
        return [server, gui]

    gazebo = OpaqueFunction(function=gazebo_actions)
    websocket_server = ExecuteProcess(
        cmd=[
            "gz",
            "launch",
            "-v",
            "3",
            PathJoinSubstitution([package_share, "worlds", "websocket.gzlaunch"]),
        ],
        additional_env={
            "GZ_CONFIG_PATH": [
                "/opt/ros_ws/install/share/gz:",
                EnvironmentVariable("GZ_CONFIG_PATH", default_value=""),
            ],
            "LD_LIBRARY_PATH": [
                "/opt/ros_ws/install/lib:",
                EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
            ],
        },
        condition=IfCondition(LaunchConfiguration("websocket")),
        output="screen",
    )
    teleport_api = Node(
        package="cfr_arduino_bridge",
        executable="teleport_api.py",
        name="teleport_api",
        output="screen",
        additional_env={"CFR_SIM_WORLD": LaunchConfiguration("world_name")},
    )

    randomizer = Node(
        package="cfr_arduino_bridge",
        executable="obstacle_randomizer_node.py",
        name="obstacle_randomizer",
        output="screen",
        parameters=[LaunchConfiguration("layout_file"), {"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("randomizer")),
    )

    command_bridge = Node(
        package="cfr_arduino_bridge",
        executable="sim_vehicle_node",
        name="sim_vehicle",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": True}],
        remappings=[
            ("~/drive_cmd", "/drive_cmd"),
            ("~/status", "/arduino_bridge/status"),
            ("cmd_vel", "/sim/cmd_vel"),
        ],
    )

    # The camera topics are bridged under the names Gazebo's rgbd_camera
    # sensor actually publishes and renamed to the ZED's on the ROS side.
    # Color and depth come from one sensor, so they share a calibration and
    # there is only one camera_info to bridge.
    gazebo_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/sim/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/model/slash/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/zed/gz/rgbd/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/zed/gz/rgbd/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/zed/gz/rgbd/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/zed/gz/rgbd/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            # Setpoint for the start signal's arm joint.  The randomizer ramps
            # this to turn the arm at a stated rate; going through the bridge
            # rather than the gz CLI is what makes a smooth sweep affordable.
            "/start_signal/arm@std_msgs/msg/Float64]gz.msgs.Double",
        ],
        remappings=[
            ("/model/slash/odometry", "/zed/zed_node/odom"),
            ("/zed/gz/rgbd/image", "/zed/zed_node/left/image_rect_color"),
            (
                "/zed/gz/rgbd/camera_info",
                "/zed/zed_node/left/image_rect_color/camera_info",
            ),
            ("/zed/gz/rgbd/depth_image", "/zed/zed_node/depth/depth_registered"),
            ("/zed/gz/rgbd/points", "/zed/zed_node/point_cloud/cloud_registered"),
        ],
    )

    # Only with the camera rendered, since it has nothing to read otherwise.
    # The arm turns over about a second and the detector wants a couple of
    # frames of it, so llvmpipe's ~5 Hz is enough; see the README.
    start_signal_detector = Node(
        package="cfr_arduino_bridge",
        executable="start_signal_detector_node.py",
        name="start_signal_detector",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": True}],
        remappings=[("image", "/zed/zed_node/left/image_rect_color")],
        condition=IfCondition(LaunchConfiguration("sensors")),
    )

    cmd_vel_to_drive = Node(
        package="cfr_arduino_bridge",
        executable="cmd_vel_to_drive_node",
        name="cmd_vel_to_drive",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": True}],
        remappings=[("cmd_vel", "/cmd_vel"), ("drive_cmd", "/drive_cmd")],
    )

    path_follower = Node(
        package="cfr_arduino_bridge",
        executable="path_follower_node",
        name="path_follower",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": True}],
        remappings=[("~/odom", "/zed/zed_node/odom"), ("cmd_vel", "/cmd_vel")],
    )

    return LaunchDescription(
        [
            params_arg,
            world_arg,
            gui_arg,
            websocket_arg,
            sensors_arg,
            world_name_arg,
            randomizer_arg,
            layout_arg,
            resource_path,
            gazebo,
            websocket_server,
            teleport_api,
            randomizer,
            command_bridge,
            gazebo_bridge,
            start_signal_detector,
            cmd_vel_to_drive,
            path_follower,
        ]
    )
