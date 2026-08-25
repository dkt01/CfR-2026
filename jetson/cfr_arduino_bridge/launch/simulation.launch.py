"""Run autonomy stack against Gazebo Harmonic speed-course simulation."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
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

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={
            "gz_args": ["-r -s -v 3 ", LaunchConfiguration("world")]
        }.items(),
    )
    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={
            "gz_args": ["-g -v 3 ", LaunchConfiguration("world")]
        }.items(),
        condition=IfCondition(LaunchConfiguration("gui")),
    )
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

    gazebo_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/sim/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/model/slash/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/zed/zed_node/left/image_rect_color@sensor_msgs/msg/Image[gz.msgs.Image",
            "/zed/zed_node/left/image_rect_color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/zed/zed_node/depth/depth_registered@sensor_msgs/msg/Image[gz.msgs.Image",
            "/zed/zed_node/depth/depth_registered/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/zed/zed_node/depth/depth_registered/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        remappings=[
            ("/model/slash/odometry", "/zed/zed_node/odom"),
            (
                "/zed/zed_node/depth/depth_registered/points",
                "/zed/zed_node/point_cloud/cloud_registered",
            ),
        ],
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
            gazebo,
            gazebo_gui,
            websocket_server,
            teleport_api,
            command_bridge,
            gazebo_bridge,
            cmd_vel_to_drive,
            path_follower,
        ]
    )
