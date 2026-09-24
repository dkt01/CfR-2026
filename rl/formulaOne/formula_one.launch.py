"""Run the trained policy against whatever stack is already up.

    ros2 launch rl/formulaOne/formula_one.launch.py policy:=runs/v1/policy.npz

Brings up only the driver and, optionally, RViz.  It assumes something else is
already publishing /zed/zed_node/pose and consuming /drive_cmd -- either
`speed_course.launch.py` in Gazebo or `arduino_bridge.launch.py` on the car.
`validate.sh` is the wrapper that starts the simulator too.

Recording: record:=auto (the default) starts jetson/scripts/record_run.py
beside the driver whenever use_sim_time is false -- i.e. on the car -- so a
real run is never driven unrecorded by accident.  record:=true records in
simulation too; record:=false never does.  record_args passes extra flags,
e.g. record_args:="--svo --map".  The run lands in ~/cfr_runs; pull it with
the Run Lab (web/run-lab) or jetson/scripts/sync_runs.sh.
"""

import shlex

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent


def generate_launch_description():
    args = [
        DeclareLaunchArgument("policy", default_value=str(HERE / "runs/v1/policy.npz")),
        DeclareLaunchArgument("config", default_value=str(HERE / "config.yaml")),
        DeclareLaunchArgument(
            "driver", default_value="policy", description="policy | baseline"
        ),
        DeclareLaunchArgument(
            "laps", default_value="0", description="0 takes it from config"
        ),
        DeclareLaunchArgument(
            "anchor",
            default_value="signal",
            description="signal latches the start pose; world trusts the frame",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument(
            "record",
            default_value="auto",
            description="auto records when use_sim_time is false; true | false",
        ),
        DeclareLaunchArgument("record_label", default_value=""),
        DeclareLaunchArgument(
            "record_args",
            default_value="",
            description='extra record_run.py flags, e.g. "--svo --map"',
        ),
    ]

    driver = ExecuteProcess(
        cmd=[
            "python3",
            str(HERE / "formula_one_node.py"),
            "--ros-args",
            "-p",
            ["policy:=", LaunchConfiguration("policy")],
            "-p",
            ["config:=", LaunchConfiguration("config")],
            "-p",
            ["driver:=", LaunchConfiguration("driver")],
            "-p",
            ["laps:=", LaunchConfiguration("laps")],
            "-p",
            ["anchor:=", LaunchConfiguration("anchor")],
            "-p",
            ["speed_scale:=", LaunchConfiguration("speed_scale")],
            "-p",
            ["use_sim_time:=", LaunchConfiguration("use_sim_time")],
        ],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="formula_one_rviz",
        arguments=["-d", str(HERE / "rviz/formula_one.rviz")],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="log",
    )
    return LaunchDescription([*args, driver, rviz, OpaqueFunction(function=recorder)])


def recorder(context, *args, **kwargs):
    """record_run.py, when record resolves true.

    Long signal timeouts on purpose: on Ctrl-C the recorder still has to save
    the ZED area memory, restore the cloud rate, close the bag and copy the
    logs.  The launch default escalates to SIGKILL after 5 s, which would
    leave a bag with no finished metadata.
    """

    def value(name):
        return LaunchConfiguration(name).perform(context)

    mode = value("record").lower()
    sim = value("use_sim_time").lower() in ("true", "1")
    if mode == "false" or (mode == "auto" and sim):
        return []
    driver = value("driver")
    label = value("record_label") or (
        "baseline" if driver == "baseline" else Path(value("policy")).parent.name
    )
    script = HERE.parents[1] / "jetson" / "scripts" / "record_run.py"
    cmd = [
        "python3",
        str(script),
        "--label",
        f"f1_{label}",
        "--driver",
        driver,
        "--speed-scale",
        value("speed_scale"),
        "--config",
        value("config"),
    ]
    if driver != "baseline":
        cmd += ["--policy", value("policy")]
    cmd += shlex.split(value("record_args"))
    return [
        ExecuteProcess(
            cmd=cmd,
            output="screen",
            sigterm_timeout="45",
            sigkill_timeout="60",
        )
    ]
