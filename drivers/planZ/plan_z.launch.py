"""Run Plan Z against whatever stack is already up.

    ros2 launch drivers/planZ/plan_z.launch.py course:=obstacle
    ros2 launch drivers/planZ/plan_z.launch.py course:=speed knobs:="speed_scale=0.5"

Brings up only the driver.  It assumes something else is publishing the ZED
cloud and pose and consuming /drive_cmd: `obstacle_course.launch.py
sensors:=true cmd_vel_to_drive:=false path_follower:=false` (or
speed_course.launch.py) in Gazebo, or `launch.sh --no-cmd-vel` on the car.
plan_z_sim.launch.py starts the simulator too; plan_z_car.launch.py adds what
a run on the car needs.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

HERE = Path(__file__).resolve().parent
# The ROS name per course: what record_run.py, the Run Lab and
# rl/obstacleRacer/gazebo_check.py already look for.
NODE_NAME = {"obstacle": "obstacle_racer", "speed": "formula_one"}


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "course", default_value="obstacle", description="obstacle | speed"
        ),
        DeclareLaunchArgument("config", default_value=str(HERE / "config.yaml")),
        DeclareLaunchArgument(
            "knobs",
            default_value="",
            description='config.yaml overrides, "name=value ..."',
        ),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument(
            "node_name",
            default_value="",
            description="default: the course's usual driver name",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=driver)])


def driver(context, *args, **kwargs):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    course = value("course")
    name = value("node_name") or NODE_NAME.get(course, "plan_z")
    knobs = f"speed_scale={value('speed_scale')} {value('knobs')}".strip()
    return [
        ExecuteProcess(
            cmd=[
                "python3",
                str(HERE / "plan_z_node.py"),
                "--ros-args",
                "-r",
                f"__node:={name}",
                "-p",
                f"course:={course}",
                "-p",
                f"config:={value('config')}",
                "-p",
                f"knobs:={knobs}",
                "-p",
                f"use_sim_time:={value('use_sim_time')}",
            ],
            output="screen",
        )
    ]
