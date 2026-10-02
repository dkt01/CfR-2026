"""Launch formulaThree with its checkpoint config; retain /formula_one telemetry for Run Lab."""

import shlex
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
# A synced Orin tree (jetson/scripts/syncSoftware.sh) has the chosen policy
# and its config at the top level; a checkout defaults to runs/f3_v1/.
DEPLOYED = HERE / "policy.npz"
POLICY_DIR = HERE if DEPLOYED.exists() else HERE / "runs/f3_v1"
SIM_CAMERA_INFO = "/zed/zed_node/left/image_rect_color/camera_info"
CAR_CAMERA_INFO = "/zed/zed_node/depth/camera_info"


def generate_launch_description():
    args = [
        DeclareLaunchArgument("policy", default_value=str(POLICY_DIR / "policy.npz")),
        DeclareLaunchArgument(
            "config",
            default_value="",
            description="Default: config beside policy; source config for baseline",
        ),
        DeclareLaunchArgument(
            "driver", default_value="policy", description="policy | baseline"
        ),
        DeclareLaunchArgument(
            "laps", default_value="0", description="0 takes it from config (3)"
        ),
        DeclareLaunchArgument(
            "anchor",
            default_value="signal",
            description="signal latches the start pose; world trusts the frame",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("speed_scale", default_value="1.0"),
        DeclareLaunchArgument("node_name", default_value="formula_one"),
        DeclareLaunchArgument(
            "depth_topic", default_value="/zed/zed_node/depth/depth_registered"
        ),
        DeclareLaunchArgument("camera_info_topic", default_value="auto"),
        DeclareLaunchArgument(
            "depth_fallback",
            default_value="stop",
            description="stop (coast to rest, run over) | map (race on the map alone)",
        ),
        DeclareLaunchArgument(
            "depth_qos",
            default_value="auto",
            description="auto matches the depth publisher; reliable | best_effort",
        ),
        DeclareLaunchArgument(
            "depth_hold_after",
            default_value="0.3",
            description="s without depth before the network is taken off the wheel",
        ),
        DeclareLaunchArgument(
            "depth_timeout",
            default_value="1.0",
            description="s without depth before it is LOST and depth_fallback runs",
        ),
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

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="formula_three_rviz",
        arguments=["-d", str(HERE / "rviz/formula_three.rviz")],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="log",
    )
    return LaunchDescription(
        [
            *args,
            OpaqueFunction(function=driver),
            rviz,
            OpaqueFunction(function=recorder),
        ]
    )


def driver(context, *args, **kwargs):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    sim = value("use_sim_time").lower() in ("true", "1")
    info = value("camera_info_topic")
    if info == "auto":
        info = SIM_CAMERA_INFO if sim else CAR_CAMERA_INFO
    params = {
        "policy": value("policy"),
        "config": value("config")
        or str(
            HERE / "config.yaml"
            if value("driver") == "baseline"
            else Path(value("policy")).with_name("config.yaml")
        ),
        "driver": value("driver"),
        "laps": value("laps"),
        "anchor": value("anchor"),
        "speed_scale": value("speed_scale"),
        "use_sim_time": value("use_sim_time"),
        "depth_topic": value("depth_topic"),
        "camera_info_topic": info,
        "depth_fallback": value("depth_fallback"),
        "depth_qos": value("depth_qos"),
        "depth_hold_after": value("depth_hold_after"),
        "depth_timeout": value("depth_timeout"),
    }
    cmd = ["python3", str(HERE / "formula_three_node.py"), "--ros-args"]
    cmd += ["-r", f"__node:={value('node_name')}"]
    for key, val in params.items():
        cmd += ["-p", f"{key}:={val}"]
    return [ExecuteProcess(cmd=cmd, output="screen")]


def recorder(context, *args, **kwargs):
    """record_run.py, when record resolves true -- see formula_one.launch.py."""

    def value(name):
        return LaunchConfiguration(name).perform(context)

    mode = value("record").lower()
    sim = value("use_sim_time").lower() in ("true", "1")
    if mode == "false" or (mode == "auto" and sim):
        return []
    drv = value("driver")
    label = value("record_label") or (
        "baseline" if drv == "baseline" else Path(value("policy")).parent.name
    )
    script = HERE.parents[1] / "jetson" / "scripts" / "record_run.py"
    cmd = [
        "python3",
        str(script),
        "--label",
        f"f3_{label}",
        "--driver",
        drv,
        "--speed-scale",
        value("speed_scale"),
        "--config",
        value("config")
        or str(
            HERE / "config.yaml"
            if drv == "baseline"
            else Path(value("policy")).with_name("config.yaml")
        ),
    ]
    if drv != "baseline":
        cmd += ["--policy", value("policy")]
    cmd += shlex.split(value("record_args"))
    return [
        ExecuteProcess(
            cmd=cmd, output="screen", sigterm_timeout="45", sigkill_timeout="60"
        )
    ]
