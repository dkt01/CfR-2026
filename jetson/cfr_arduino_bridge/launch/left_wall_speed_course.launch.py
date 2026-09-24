"""Run the three-lap speed course with a simple left-wall follower."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("cfr_arduino_bridge")
    return LaunchDescription(
        [
            DeclareLaunchArgument("websocket", default_value="false"),
            DeclareLaunchArgument("auto_start_signal", default_value="true"),
            SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([share, "launch", "speed_course.launch.py"])
                ),
                launch_arguments={
                    "sensors": "true",
                    "websocket": LaunchConfiguration("websocket"),
                    "path_follower": "false",
                }.items(),
            ),
            Node(
                package="cfr_arduino_bridge",
                executable="left_wall_follower_node",
                name="left_wall_follower",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "auto_start_signal": ParameterValue(
                            LaunchConfiguration("auto_start_signal"), value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
