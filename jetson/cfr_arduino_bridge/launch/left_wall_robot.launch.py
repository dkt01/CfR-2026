"""Physical speed-course stack: Arduino, ZED, wall follower and three laps.

    ros2 launch cfr_arduino_bridge left_wall_robot.launch.py device:=/dev/ttyACM0

The follower reads the ZED cloud. It does not load the simulated SDF path.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("cfr_arduino_bridge")
    config = PathJoinSubstitution([share, "config", "arduino_bridge.yaml"])
    zed_config = PathJoinSubstitution([share, "config", "cfr_zed2i.yaml"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("device", default_value="/dev/ttyACM0"),
            DeclareLaunchArgument(
                "cloud_topic",
                default_value="/zed/zed_node/point_cloud/cloud_registered",
            ),
            DeclareLaunchArgument(
                "image_topic", default_value="/zed/zed_node/rgb/color/rect/image"
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([share, "launch", "arduino_bridge.launch.py"])
                ),
                launch_arguments={"device": LaunchConfiguration("device")}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("zed_wrapper"),
                            "launch",
                            "zed_camera.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "camera_model": "zed2i",
                    "ros_params_override_path": zed_config,
                }.items(),
            ),
            Node(
                package="cfr_arduino_bridge",
                executable="start_signal_detector_node.py",
                name="start_signal_detector",
                output="screen",
                parameters=[config],
                remappings=[("image", LaunchConfiguration("image_topic"))],
            ),
            Node(
                package="cfr_arduino_bridge",
                executable="lap_counter_node.py",
                name="lap_counter",
                output="screen",
                parameters=[config, {"target_laps": 3}],
                remappings=[
                    ("pose", "/zed/zed_node/pose"),
                    ("status", "/arduino_bridge/status"),
                    ("go", "/start_signal_detector/go"),
                ],
            ),
            Node(
                package="cfr_arduino_bridge",
                executable="left_wall_follower_node",
                name="left_wall_follower",
                output="screen",
                parameters=[
                    {
                        "source": "cloud",
                        "cloud_topic": LaunchConfiguration("cloud_topic"),
                        "auto_start_signal": False,
                    }
                ],
            ),
        ]
    )
