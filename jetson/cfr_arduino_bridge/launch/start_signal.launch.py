"""Watch the camera for the start signal, and latch it for the driver.

Run alongside the ZED, which is what publishes the image this reads:

    ~/software/scripts/launch.sh                                  # bridge + ZED
    ros2 launch cfr_arduino_bridge start_signal.launch.py

In the Gazebo courses the detector comes up with `sensors:=true` already, so
this file is for the car -- or for pointing the detector at a different camera
topic or a bag:

    ros2 launch cfr_arduino_bridge start_signal.launch.py debug:=true
    ros2 launch cfr_arduino_bridge start_signal.launch.py image_topic:=/other

`debug:=true` publishes an annotated copy of each frame on
`/start_signal_detector/debug_image`, which is how the colour thresholds get
tuned against the real signal.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution(
        [FindPackageShare("cfr_arduino_bridge"), "config", "arduino_bridge.yaml"]
    )

    image_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/zed/zed_node/left/image_rect_color",
        description="Colour image to look for the signal in",
    )
    params_arg = DeclareLaunchArgument(
        "params_file", default_value=params_file, description="Parameter file"
    )
    debug_arg = DeclareLaunchArgument(
        "debug",
        default_value="false",
        description="Publish an annotated frame on ~/debug_image for tuning",
    )

    detector = Node(
        package="cfr_arduino_bridge",
        executable="start_signal_detector_node.py",
        name="start_signal_detector",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            # Typed, because a launch argument arrives as the string
            # "false" and the node declares this one as a bool.
            {
                "debug_image": ParameterValue(
                    LaunchConfiguration("debug"), value_type=bool
                )
            },
        ],
        remappings=[("image", LaunchConfiguration("image_topic"))],
    )

    return LaunchDescription([image_arg, params_arg, debug_arg, detector])
