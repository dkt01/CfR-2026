"""Bring up the Jetson side of the onboard serial link."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution(
        [FindPackageShare("cfr_arduino_bridge"), "config", "arduino_bridge.yaml"]
    )

    device_arg = DeclareLaunchArgument(
        "device", default_value="/dev/ttyACM0", description="Arduino USB serial device"
    )
    params_arg = DeclareLaunchArgument(
        "params_file", default_value=params_file, description="Parameter file for both nodes"
    )
    cmd_vel_arg = DeclareLaunchArgument(
        "use_cmd_vel",
        default_value="true",
        description="Also run cmd_vel_to_drive_node to translate geometry_msgs/Twist",
    )

    bridge = Node(
        package="cfr_arduino_bridge",
        executable="arduino_bridge_node",
        name="arduino_bridge",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"device": LaunchConfiguration("device")},
        ],
        remappings=[("~/drive_cmd", "/drive_cmd")],
    )

    cmd_vel_to_drive = Node(
        package="cfr_arduino_bridge",
        executable="cmd_vel_to_drive_node",
        name="cmd_vel_to_drive",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
        remappings=[("cmd_vel", "/cmd_vel"), ("drive_cmd", "/drive_cmd")],
        condition=IfCondition(LaunchConfiguration("use_cmd_vel")),
    )

    return LaunchDescription([device_arg, params_arg, cmd_vel_arg, bridge, cmd_vel_to_drive])
