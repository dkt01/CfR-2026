"""Live ZED exposure/ROI tuning bridge, for the Run Lab's Calibration page.

    ~/software/scripts/launch.sh --no-cmd-vel      # ZED + bridge, no drivetrain
    ros2 launch cfr_arduino_bridge zed_live_tuning.launch.py

Starts only rosbridge (so a browser can `ros2 param set` the running
/zed/zed_node live) and web_video_server (so it can show the rectified image
and the ROI mask as it tunes). CAMERA-ONLY: this never subscribes to or
publishes drive_cmd, so unlike launch.sh it carries none of the
actuator-arming risk -- it can be started and stopped freely while someone is
tuning exposure at the bench, with no E-Stop cycle needed because nothing
here can move the car.

rosbridge_suite and web_video_server are not part of any other launch file in
this repo; install them once with
`sudo apt install ros-jazzy-rosbridge-suite ros-jazzy-web-video-server`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rosbridge_port_arg = DeclareLaunchArgument(
        "rosbridge_port", default_value="9090", description="rosbridge websocket port"
    )
    video_port_arg = DeclareLaunchArgument(
        "video_port", default_value="8080", description="web_video_server HTTP port"
    )

    rosbridge = Node(
        package="rosbridge_server",
        executable="rosbridge_websocket",
        name="rosbridge_websocket",
        output="screen",
        parameters=[{"port": LaunchConfiguration("rosbridge_port")}],
    )
    video_server = Node(
        package="web_video_server",
        executable="web_video_server",
        name="web_video_server",
        output="screen",
        parameters=[{"port": LaunchConfiguration("video_port")}],
    )

    return LaunchDescription(
        [rosbridge_port_arg, video_port_arg, rosbridge, video_server]
    )
