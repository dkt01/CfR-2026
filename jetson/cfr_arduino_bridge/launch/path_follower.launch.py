"""Launch the L-shape / waypoint path follower.

Run alongside arduino_bridge.launch.py:
    ros2 launch cfr_arduino_bridge arduino_bridge.launch.py
    ros2 launch cfr_arduino_bridge path_follower.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution(
        [FindPackageShare("cfr_arduino_bridge"), "config", "arduino_bridge.yaml"]
    )

    odom_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="/zed/zed_node/odom",
        description="Odometry source for pose feedback (position + heading)",
    )
    cmd_vel_arg = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="Twist topic published by the path follower",
    )
    params_arg = DeclareLaunchArgument(
        "params_file", default_value=params_file, description="Parameter file"
    )

    path_follower = Node(
        package="cfr_arduino_bridge",
        executable="path_follower_node",
        name="path_follower",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
        remappings=[
            ("~/odom", LaunchConfiguration("odom_topic")),
            ("cmd_vel", LaunchConfiguration("cmd_vel_topic")),
        ],
    )

    return LaunchDescription([odom_arg, cmd_vel_arg, params_arg, path_follower])
