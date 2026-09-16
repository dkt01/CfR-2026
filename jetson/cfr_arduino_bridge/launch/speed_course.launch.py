"""Run the autonomy stack against the Gazebo Speed Course.

    ros2 launch cfr_arduino_bridge speed_course.launch.py

The Speed Course has no buckets or hoops, but it starts on the same visual
signal the Obstacle Course does, so the randomiser runs here too -- with a
layout that carries only the start signal.  See obstacle_course.launch.py for
the other course.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("cfr_arduino_bridge")

    passthrough = [
        DeclareLaunchArgument(
            "sensors",
            default_value="false",
            description="Render the ZED RGB-D pair; needs a render context",
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="false",
            description="Start Gazebo GUI; requires an authorized host display",
        ),
        DeclareLaunchArgument(
            "websocket",
            default_value="false",
            description="Start gzweb-compatible WebSocket server on port 9002",
        ),
    ]

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "simulation.launch.py"])
        ),
        launch_arguments={
            "world": PathJoinSubstitution(
                [package_share, "worlds", "speed_course.sdf"]
            ),
            "gui": LaunchConfiguration("gui"),
            "websocket": LaunchConfiguration("websocket"),
            "sensors": LaunchConfiguration("sensors"),
            "world_name": "cfr_speed_course",
            "randomizer": "true",
            "layout_file": PathJoinSubstitution(
                [package_share, "config", "speed_course_layout.yaml"]
            ),
        }.items(),
    )

    return LaunchDescription([*passthrough, simulation])
