"""Run the trained policy against whatever stack is already up.

    ros2 launch rl/formulaOne/formula_one.launch.py policy:=runs/v1/policy.npz

Brings up only the driver and, optionally, RViz.  It assumes something else is
already publishing /zed/zed_node/pose and consuming /drive_cmd -- either
`speed_course.launch.py` in Gazebo or `arduino_bridge.launch.py` on the car.
`validate.sh` is the wrapper that starts the simulator too.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent


def generate_launch_description():
    args = [
        DeclareLaunchArgument("policy", default_value=str(HERE / "runs/v1/policy.npz")),
        DeclareLaunchArgument("config", default_value=str(HERE / "config.yaml")),
        DeclareLaunchArgument("driver", default_value="policy",
                              description="policy | baseline"),
        DeclareLaunchArgument("laps", default_value="0", description="0 takes it from config"),
        DeclareLaunchArgument("anchor", default_value="signal",
                              description="signal latches the start pose; world trusts the frame"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
    ]

    driver = ExecuteProcess(
        cmd=["python3", str(HERE / "formula_one_node.py"),
             "--ros-args",
             "-p", ["policy:=", LaunchConfiguration("policy")],
             "-p", ["config:=", LaunchConfiguration("config")],
             "-p", ["driver:=", LaunchConfiguration("driver")],
             "-p", ["laps:=", LaunchConfiguration("laps")],
             "-p", ["anchor:=", LaunchConfiguration("anchor")],
             "-p", ["speed_scale:=", LaunchConfiguration("speed_scale")],
             "-p", ["use_sim_time:=", LaunchConfiguration("use_sim_time")]],
        output="screen",
    )

    rviz = Node(
        package="rviz2", executable="rviz2", name="formula_one_rviz",
        arguments=["-d", str(HERE / "rviz/formula_one.rviz")],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="log",
    )
    return LaunchDescription([*args, driver, rviz])
