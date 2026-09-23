"""Speed-course wall follower with browser transport and manual signal control."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("cfr_arduino_bridge")
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [share, "launch", "left_wall_speed_course.launch.py"]
                    )
                ),
                launch_arguments={
                    "websocket": "true",
                    "auto_start_signal": "false",
                }.items(),
            ),
        ]
    )
