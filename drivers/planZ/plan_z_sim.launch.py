"""Plan Z in Gazebo: a course, its start signal and lap counter, and the driver.

    ros2 launch drivers/planZ/plan_z_sim.launch.py course:=obstacle green_after:=60
    ros2 launch drivers/planZ/plan_z_sim.launch.py course:=speed laps:=3

Brings up the course with the ZED rendered (the driver sees nothing else),
path_follower and cmd_vel_to_drive off (each would be a second publisher on
/drive_cmd), and lap_counter set to `laps`.  The driver starts as it does on
the car: it holds still until /start_signal_detector/go latches, and takes
the throttle off when /lap_counter/done does.

The simulated signal starts on red.  Turn it green with

    ros2 service call /obstacle_randomizer/start_signal std_srvs/srv/SetBool "{data: true}"

or pass green_after:=<s>.  To skip the signal:

    ros2 service call /obstacle_racer/manual_start std_srvs/srv/SetBool "{data: true}"

(/formula_one/manual_start on the Speed Course).  validate.sh is the
scripted check; this is the file for watching a run.

The simulated car can be put out of true, to see what the driver makes of
it: camera_rpy_deg:="0 2 -3" turns the simulated ZED on its mount (roll,
pitch, yaw), and params_file:=<a copy of arduino_bridge.yaml> with shifted
steering_angle_points gives it a steering bias.  validate.sh does both.
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
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

HERE = Path(__file__).resolve().parent
LAPS = {"obstacle": "2", "speed": "3"}


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "course", default_value="obstacle", description="obstacle | speed"
        ),
        DeclareLaunchArgument("config", default_value=str(HERE / "config.yaml")),
        DeclareLaunchArgument("knobs", default_value=""),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument(
            "laps", default_value="", description="default: 2 obstacle, 3 speed"
        ),
        DeclareLaunchArgument(
            "green_after",
            default_value="0",
            description="s after launch to turn the signal green; 0 leaves it to you",
        ),
        DeclareLaunchArgument("camera_rpy_deg", default_value="0 0 0"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("websocket", default_value="false"),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=stack)])


def stack(context, *args, **kwargs):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    course = value("course")
    share = FindPackageShare("cfr_arduino_bridge").perform(context)
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(f"{share}/launch/{course}_course.launch.py"),
        launch_arguments={
            "sensors": "true",
            "path_follower": "false",
            "cmd_vel_to_drive": "false",
            "laps": value("laps") or LAPS[course],
            "camera_rpy_deg": value("camera_rpy_deg"),
            "gui": value("gui"),
            "websocket": value("websocket"),
        }.items(),
    )
    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(HERE / "plan_z.launch.py")),
        launch_arguments={
            "course": course,
            "config": value("config"),
            "knobs": value("knobs"),
            "speed_scale": value("speed_scale"),
            "use_sim_time": "true",
        }.items(),
    )
    actions = [sim, driver]
    delay = float(value("green_after"))
    if delay > 0:
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
        actions.append(TimerAction(period=delay, actions=[call]))
    return actions
