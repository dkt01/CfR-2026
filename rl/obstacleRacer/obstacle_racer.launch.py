"""Run the obstacle racer against whatever stack is already up.

    ros2 launch rl/obstacleRacer/obstacle_racer.launch.py policy:=runs/v1/policy.npz

Brings up only the driver.  It assumes something else is publishing the ZED
cloud and pose and consuming /drive_cmd: `obstacle_course.launch.py
sensors:=true cmd_vel_to_drive:=false path_follower:=false` in Gazebo, or
`arduino_bridge.launch.py` with the ZED on the car.  validate.sh starts the
simulator too.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

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
        DeclareLaunchArgument("use_sim_time", default_value="true"),
    ]
    driver = ExecuteProcess(
        cmd=[
            "python3",
            str(HERE / "obstacle_racer_node.py"),
            "--ros-args",
            "-p",
            ["policy:=", LaunchConfiguration("policy")],
            "-p",
            ["config:=", LaunchConfiguration("config")],
            "-p",
            ["driver:=", LaunchConfiguration("driver")],
            "-p",
            ["prior_speed:=", LaunchConfiguration("prior_speed")],
            "-p",
            ["speed_scale:=", LaunchConfiguration("speed_scale")],
            "-p",
            ["use_sim_time:=", LaunchConfiguration("use_sim_time")],
        ],
        output="screen",
    )
    return LaunchDescription(args + [driver])
