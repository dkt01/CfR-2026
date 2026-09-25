"""Run the autonomy stack against the Gazebo Obstacle Course.

    ros2 launch cfr_arduino_bridge obstacle_course.launch.py

Adds the randomizer and its layout to the shared simulation bringup.  See
speed_course.launch.py for the other course.
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
        DeclareLaunchArgument(
            "path_follower",
            default_value="true",
            description="Start the DrivePath action controller",
        ),
        DeclareLaunchArgument(
            "laps",
            default_value="1",
            description="Laps before lap_counter latches ~/done; one run is one lap",
        ),
        DeclareLaunchArgument(
            "cmd_vel_to_drive",
            default_value="true",
            description=(
                "Start cmd_vel_to_drive_node. A driver that publishes "
                "/drive_cmd itself (rl/obstacleRacer) must turn this off"
            ),
        ),
    ]

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "simulation.launch.py"])
        ),
        launch_arguments={
            "world": PathJoinSubstitution(
                [package_share, "worlds", "obstacle_course.sdf"]
            ),
            "gui": LaunchConfiguration("gui"),
            "websocket": LaunchConfiguration("websocket"),
            "sensors": LaunchConfiguration("sensors"),
            "path_follower": LaunchConfiguration("path_follower"),
            "cmd_vel_to_drive": LaunchConfiguration("cmd_vel_to_drive"),
            "world_name": "cfr_obstacle_course",
            "randomizer": "true",
            "laps": LaunchConfiguration("laps"),
            "layout_file": PathJoinSubstitution(
                [package_share, "config", "obstacle_course_layout.yaml"]
            ),
        }.items(),
    )

    return LaunchDescription([*passthrough, simulation])
