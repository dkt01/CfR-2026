"""Simulation stack for RL training.

Same as simulation.launch.py minus path_follower_node (which publishes zeros
on /cmd_vel while idle and would fight the policy for control) and minus the
camera bridges, plus a ground-truth pose bridge the RL environment needs.

Gazebo runs freely rather than being stepped by the environment: driving it
through WorldControl `multi_step` triggers heap corruption in the server
(`malloc(): unaligned fastbin chunk detected`), which leaves the process
alive but its service threads dead. The environment paces itself against the
wall clock instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("cfr_arduino_bridge")
    params_file = PathJoinSubstitution([package_share, "config", "arduino_bridge.yaml"])
    world_file = PathJoinSubstitution([package_share, "worlds", "speed_course.sdf"])

    params_arg = DeclareLaunchArgument("params_file", default_value=params_file)
    world_arg = DeclareLaunchArgument("world", default_value=world_file)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={"gz_args": ["-r -s -v 3 ", LaunchConfiguration("world")]}.items(),
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
            # Ground-truth world pose. The Ackermann plugin's odometry is
            # dead-reckoned from wheel rotation, so it ignores teleports --
            # useless for RL episode resets. This topic carries the real pose.
            "/world/cfr_speed_course/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        remappings=[("/model/slash/odometry", "/zed/zed_node/odom")],
    )

    cmd_vel_to_drive = Node(
        package="cfr_arduino_bridge",
        executable="cmd_vel_to_drive_node",
        name="cmd_vel_to_drive",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {"use_sim_time": True}],
        remappings=[("cmd_vel", "/cmd_vel"), ("drive_cmd", "/drive_cmd")],
    )

    return LaunchDescription(
        [
            params_arg,
            world_arg,
            gazebo,
            teleport_api,
            command_bridge,
            gazebo_bridge,
            cmd_vel_to_drive,
        ]
    )
