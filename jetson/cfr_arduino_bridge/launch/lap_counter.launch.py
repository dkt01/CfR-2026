"""Count laps of the start/finish line on the car.

Run alongside the bridge and the ZED, which is what publishes the pose this
reads:

    ~/software/scripts/launch.sh                                # bridge + ZED
    ros2 launch cfr_arduino_bridge lap_counter.launch.py

In the Gazebo courses the counter comes up already, with the right lap target
for the course, so this file is for the car -- or for a bench run with no
start signal detector and no Arduino:

    ros2 launch cfr_arduino_bridge lap_counter.launch.py laps:=2
    ros2 launch cfr_arduino_bridge lap_counter.launch.py free_run:=true

`free_run:=true` arms on the first pose instead of the start signal and counts
without waiting for the Arduino to report AUTO_ACTIVE, which is what makes the
counter usable from path_tui with nothing else running.

The pose defaults to the ZED's map-frame topic rather than its odometry: the
SDK applies loop closure to that one and never to `~/odom`, and three laps of
the speed course is about 300 m of travel back to the same spot.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    NotSubstitution,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution(
        [FindPackageShare("cfr_arduino_bridge"), "config", "arduino_bridge.yaml"]
    )

    params_arg = DeclareLaunchArgument(
        "params_file", default_value=params_file, description="Parameter file"
    )
    laps_arg = DeclareLaunchArgument(
        "laps",
        default_value="3",
        description="Laps to run: 3 for the speed course, 2 for the obstacle course",
    )
    pose_arg = DeclareLaunchArgument(
        "pose_topic",
        default_value="/zed/zed_node/pose",
        description="Map-frame pose; the loop-closed one, not ~/odom",
    )
    free_run_arg = DeclareLaunchArgument(
        "free_run",
        default_value="false",
        description="Arm on the first pose and ignore run mode, for the bench",
    )

    counter = Node(
        package="cfr_arduino_bridge",
        executable="lap_counter_node.py",
        name="lap_counter",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            # Typed, because a launch argument arrives as a string and the
            # node declares these as an int and a pair of bools.
            {
                "target_laps": ParameterValue(
                    LaunchConfiguration("laps"), value_type=int
                ),
                "require_go": ParameterValue(
                    NotSubstitution(LaunchConfiguration("free_run")), value_type=bool
                ),
                "require_auto_active": ParameterValue(
                    NotSubstitution(LaunchConfiguration("free_run")), value_type=bool
                ),
            },
        ],
        remappings=[
            ("pose", LaunchConfiguration("pose_topic")),
            ("status", "/arduino_bridge/status"),
            ("go", "/start_signal_detector/go"),
        ],
    )

    return LaunchDescription([params_arg, laps_arg, pose_arg, free_run_arg, counter])
