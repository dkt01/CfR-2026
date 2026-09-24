"""Simulation stack for RL training.

Same as simulation.launch.py minus path_follower_node (which publishes zeros
on /cmd_vel while idle and would fight the policy for control), plus a
ground-truth pose bridge the RL environment needs.

`sensors:=true` adds the rendered ZED and bridges its point cloud, so the
environment can build observations through the same cloud_scan path the robot
uses instead of ray-casting known bale geometry. It costs real-time factor
(measured 1.00 -> 0.63), so it is off by default. The cloud passes through
zed_cloud_noise_node on its way to the ZED topic, exactly as it does in
simulation.launch.py, so a policy trains on the same stereo-like cloud it is
evaluated and deployed against; `cloud_noise:=false` bridges Gazebo's perfect
cloud straight through instead, as runs before that did.

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
    PythonExpression,
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

    cloud_noise_arg = DeclareLaunchArgument(
        "cloud_noise",
        default_value="true",
        description="with sensors:=true, add the stereo noise simulation.launch.py adds",
    )
    noisy = PythonExpression(
        [
            "'",
            LaunchConfiguration("sensors"),
            "'.lower() in ('true', '1') and '",
            LaunchConfiguration("cloud_noise"),
            "'.lower() in ('true', '1')",
        ]
    )
    clean = PythonExpression(
        [
            "'",
            LaunchConfiguration("sensors"),
            "'.lower() in ('true', '1') and '",
            LaunchConfiguration("cloud_noise"),
            "'.lower() not in ('true', '1')",
        ]
    )
    # With noise, the raw cloud keeps its Gazebo name and zed_cloud_noise
    # publishes the ZED's; without, the bridge renames it straight across.
    points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        condition=IfCondition(noisy),
        arguments=[
            "/zed/gz/rgbd/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
    )
    zed_cloud_noise = Node(
        package="cfr_arduino_bridge",
        executable="zed_cloud_noise_node",
        name="zed_cloud_noise",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(noisy),
    )
    clean_points_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        condition=IfCondition(clean),
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
            cloud_noise_arg,
            resource_path,
            gazebo,
            points_bridge,
            zed_cloud_noise,
            clean_points_bridge,
            teleport_api,
            command_bridge,
            gazebo_bridge,
            cmd_vel_to_drive,
        ]
    )
