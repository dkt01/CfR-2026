"""The obstacle racer in Gazebo: the course, the start signal, the lap counter
and the driver, in one launch.

    ros2 launch rl/obstacleRacer/obstacle_racer_sim.launch.py policy:=runs/v4/policy.npz

Brings up obstacle_course.launch.py with the ZED rendered (the policy sees
nothing else), path_follower and cmd_vel_to_drive off (each would be a second
publisher on /drive_cmd), and lap_counter set to `laps`.  With the camera on,
that launch also runs start_signal_detector on the rendered image, so the
driver starts exactly as it does on the car: it holds still until
/start_signal_detector/go latches, and takes the throttle off when
/lap_counter/done does.

The simulated signal starts on red.  Turn it green with

    ros2 service call /obstacle_randomizer/start_signal std_srvs/srv/SetBool "{data: true}"

or pass green_after:=<s> to have this launch do it that many seconds after
start.  The detector needs to have seen red first, so leave Gazebo time to
render (llvmpipe takes a while).  To skip the signal altogether:

    ros2 service call /obstacle_racer/manual_start std_srvs/srv/SetBool "{data: true}"

validate.sh is the scripted check; this is the file for watching a run.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

HERE = Path(__file__).resolve().parent


def generate_launch_description():
    args = [
        DeclareLaunchArgument("policy", default_value=str(HERE / "runs/v1/policy.npz")),
        DeclareLaunchArgument("config", default_value=str(HERE / "config.yaml")),
        DeclareLaunchArgument(
            "driver", default_value="policy", description="policy | prior"
        ),
        DeclareLaunchArgument("prior_speed", default_value="1.0"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument(
            "laps",
            default_value="1",
            description="Laps before lap_counter latches ~/done; one run is one lap",
        ),
        DeclareLaunchArgument(
            "green_after",
            default_value="0",
            description="s after launch to turn the signal green; 0 leaves it to you",
        ),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("websocket", default_value="false"),
    ]

    course = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("cfr_arduino_bridge"),
                    "launch",
                    "obstacle_course.launch.py",
                ]
            )
        ),
        launch_arguments={
            "sensors": "true",
            "path_follower": "false",
            "cmd_vel_to_drive": "false",
            "laps": LaunchConfiguration("laps"),
            "gui": LaunchConfiguration("gui"),
            "websocket": LaunchConfiguration("websocket"),
        }.items(),
    )

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(HERE / "obstacle_racer.launch.py")),
        launch_arguments={
            "policy": LaunchConfiguration("policy"),
            "config": LaunchConfiguration("config"),
            "driver": LaunchConfiguration("driver"),
            "prior_speed": LaunchConfiguration("prior_speed"),
            "speed_scale": LaunchConfiguration("speed_scale"),
            "use_sim_time": "true",
        }.items(),
    )

    return LaunchDescription(
        [*args, course, driver, OpaqueFunction(function=green_light)]
    )


def green_light(context, *args, **kwargs):
    """The randomizer's start_signal service, green_after seconds in."""
    delay = float(LaunchConfiguration("green_after").perform(context))
    if delay <= 0:
        return []
    call = ExecuteProcess(
        cmd=[
            "ros2",
            "service",
            "call",
            "/obstacle_randomizer/start_signal",
            "std_srvs/srv/SetBool",
            "{data: true}",
        ],
        output="screen",
    )
    return [TimerAction(period=delay, actions=[call])]
