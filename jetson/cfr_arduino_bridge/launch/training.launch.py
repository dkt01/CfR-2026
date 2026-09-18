"""Simulation stack for RL training.

Same as simulation.launch.py minus path_follower_node (which publishes zeros
on /cmd_vel while idle and would fight the policy for control), plus a
ground-truth pose bridge the RL environment needs.

`sensors:=true` adds the rendered ZED and bridges its point cloud, so the
environment can build observations through the same cloud_scan path the robot
uses instead of ray-casting known bale geometry. It costs real-time factor
(measured 1.00 -> 0.63), so it is off by default.

Gazebo runs freely rather than being stepped by the environment: driving it
through WorldControl `multi_step` triggers heap corruption in the server
(`malloc(): unaligned fastbin chunk detected`), which leaves the process
alive but its service threads dead. The environment paces itself against the
wall clock instead.
"""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensors_world import resolve_world  # noqa: E402


def generate_launch_description():
    package_share = FindPackageShare("cfr_arduino_bridge")
    params_file = PathJoinSubstitution([package_share, "config", "arduino_bridge.yaml"])
    world_file = PathJoinSubstitution([package_share, "worlds", "speed_course.sdf"])

    params_arg = DeclareLaunchArgument("params_file", default_value=params_file)
    world_arg = DeclareLaunchArgument("world", default_value=world_file)
    # model:// mesh URIs resolve against this. Without it the world loads
    # straight out of the package share and finds its meshes by luck; once
    # sensors:=true rewrites the world into /tmp, that luck runs out and the
    # world fails to load entirely. Wants the share root, one level up.
    resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        [
            PathJoinSubstitution([package_share, ".."]),
            ":",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )

    sensors_arg = DeclareLaunchArgument(
        "sensors",
        default_value="false",
        description="render the ZED and bridge its point cloud (costs ~40% RTF)",
    )

    def gazebo_actions(context):
        world = str(resolve_world(context)[0])
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
                    )
                ),
                launch_arguments={"gz_args": f"-r -s -v 3 {world}"}.items(),
            )
        ]

    gazebo = OpaqueFunction(function=gazebo_actions)

    points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        condition=IfCondition(LaunchConfiguration("sensors")),
        arguments=[
            "/zed/gz/rgbd/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        remappings=[
            ("/zed/gz/rgbd/points", "/zed/zed_node/point_cloud/cloud_registered"),
        ],
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
            sensors_arg,
            resource_path,
            gazebo,
            points_bridge,
            teleport_api,
            command_bridge,
            gazebo_bridge,
            cmd_vel_to_drive,
        ]
    )
