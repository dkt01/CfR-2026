"""Watch every hoop on the Obstacle Course, and say whether the car threaded it.

Run alongside the bridge and the ZED, which is what publishes the pose this
reads:

    ~/software/scripts/launch.sh                                # bridge + ZED
    ros2 launch cfr_arduino_bridge hoop_monitor.launch.py

In the Gazebo Obstacle Course this comes up already, alongside the
randomizer -- see simulation.launch.py. This file is for the car, or for a
bench run with the layout file supplied directly.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    layout_file = PathJoinSubstitution(
        [
            FindPackageShare("cfr_arduino_bridge"),
            "config",
            "obstacle_course_layout.yaml",
        ]
    )

    layout_arg = DeclareLaunchArgument(
        "layout_file",
        default_value=layout_file,
        description="Hoop names/yaw/base_length -- the same file the randomizer draws from",
    )
    pose_arg = DeclareLaunchArgument(
        "pose_topic",
        default_value="/zed/zed_node/pose",
        description="Map-frame pose; the loop-closed one, not ~/odom",
    )

    monitor = Node(
        package="cfr_arduino_bridge",
        executable="hoop_monitor_node.py",
        name="hoop_monitor",
        output="screen",
        parameters=[LaunchConfiguration("layout_file")],
        remappings=[
            ("pose", LaunchConfiguration("pose_topic")),
            ("hoop_layout", "/obstacle_randomizer/hoop_layout"),
        ],
    )

    return LaunchDescription([layout_arg, pose_arg, monitor])
