"""Shared Gazebo Harmonic bringup behind the two per-course launch files.

Prefer `speed_course.launch.py` or `obstacle_course.launch.py`, which name a
world and its randomizer layout.  This file is what they both include, and is
still usable directly with `world:=`.
"""

import sys
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
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensors_world import resolve_world  # noqa: E402

# Both worlds carry these two markers, one at world level and one inside the
# vehicle's chassis link.  Gazebo ignores its own default server config as
# soon as a world declares plugins of its own, so the sensors system has to be
# written into the world file -- there is no launch argument for it.  Filling
# the markers in here beats keeping a second copy of each world: the Speed
# Course's world is maintained by hand, and a derived copy of a hand
# maintained file goes stale the first time somebody edits one and not the


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
    laps_arg = DeclareLaunchArgument(
        "laps",
        default_value="3",
        description="Laps before lap_counter latches ~/done; 3 speed, 2 obstacle",
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

    # Only meaningful alongside the randomizer, which is what moves the
    # hoops this watches; the Speed Course shares this launch file and its
    # layout carries no hoops.names, so hoop_monitor_node just reports an
    # always-clear status there.
    hoop_monitor = Node(
        package="cfr_arduino_bridge",
        executable="hoop_monitor_node.py",
        name="hoop_monitor",
        output="screen",
        parameters=[LaunchConfiguration("layout_file"), {"use_sim_time": True}],
        remappings=[
            ("pose", "/zed/zed_node/pose"),
            ("hoop_layout", "/obstacle_randomizer/hoop_layout"),
        ],
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
            # Ground truth, standing in for the ZED's map-frame pose: the
            # ackermann plugin's odometry above drifts and is never
            # corrected, exactly as the real camera's ~/odom is not.
            "/model/slash/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
            # Same ground-truth pose training.launch.py bridges, for anything
            # (run_policy.py, path_racer.py's tf pose source) built against
            # that topic instead of /zed/zed_node/pose.
            "/world/cfr_speed_course/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
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
            ("/model/slash/pose", "/zed/zed_node/pose"),
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

    lap_counter = Node(
        package="cfr_arduino_bridge",
        executable="lap_counter_node.py",
        name="lap_counter",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"use_sim_time": True},
            # Typed, because a launch argument arrives as the string "3" and
            # the node declares this one as an int.
            {
                "target_laps": ParameterValue(
                    LaunchConfiguration("laps"), value_type=int
                )
            },
        ],
        remappings=[
            ("pose", "/zed/zed_node/pose"),
            ("status", "/arduino_bridge/status"),
            ("go", "/start_signal_detector/go"),
        ],
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
            laps_arg,
            resource_path,
            gazebo,
            websocket_server,
            teleport_api,
            randomizer,
            hoop_monitor,
            command_bridge,
            gazebo_bridge,
            start_signal_detector,
            cmd_vel_to_drive,
            path_follower,
            lap_counter,
        ]
    )
