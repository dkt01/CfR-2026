"""The obstacle racer on the car: start signal, lap counter, driver, recorder.

LAUNCHING THIS ARMS THE ACTUATORS.  Run it only at the course with the E-Stop
remote in hand, never unattended or in the background.

Two terminals on the Orin.  First the Arduino bridge and the ZED, with
cmd_vel_to_drive off (it would be a second publisher on /drive_cmd):

    ~/software/scripts/launch.sh --no-cmd-vel

Then this:

    ros2 launch <dir>/obstacle_racer_car.launch.py policy:=<dir>/policy.npz \\
        config:=<dir>/config.yaml speed_scale:=0.3

It adds, beside the bridge and camera:

    start_signal_detector   on the ZED's color image; latches /start_signal_detector/go
    lap_counter             `laps` laps (1: one run is one lap); latches /lap_counter/done
    obstacle_racer_node     drives on go, takes the throttle off on done
    record_run.py           the run, into ~/cfr_runs (record:=false to skip)

The car will not move until the detector sees red turn green.  To release it
by hand instead:

    ros2 service call /obstacle_racer/manual_start std_srvs/srv/SetBool "{data: true}"

and `{data: false}` stops it without killing the node.

The recorder is told --cloud-hz 0.  By default it turns the ZED cloud down to
1 Hz for the run, which is harmless for formulaOne (it never reads the cloud)
but would starve this driver: it steers from the cloud, and holds zero speed
whenever the cloud is more than cloud_timeout (0.5 s) old.  So the cloud is
left at its configured rate and not recorded; the driver's telemetry, which
carries every observation it acted on, is.
"""

import shlex
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

HERE = Path(__file__).resolve().parent


def generate_launch_description():
    share = FindPackageShare("cfr_arduino_bridge")
    args = [
        DeclareLaunchArgument("policy", default_value=str(HERE / "policy.npz")),
        DeclareLaunchArgument("config", default_value=str(HERE / "config.yaml")),
        DeclareLaunchArgument(
            "driver", default_value="policy", description="policy | prior"
        ),
        DeclareLaunchArgument("prior_speed", default_value="1.0"),
        DeclareLaunchArgument(
            "speed_scale",
            default_value="1.0",
            description="Multiplies every speed command; work up from 0.3",
        ),
        DeclareLaunchArgument(
            "laps",
            default_value="1",
            description="Laps before lap_counter latches ~/done",
        ),
        DeclareLaunchArgument(
            "image_topic",
            # The ZED wrapper's name for it on the car; Gazebo bridges the
            # simulated camera as /zed/zed_node/left/image_rect_color.
            default_value="/zed/zed_node/rgb/color/rect/image",
            description="Color image the start signal detector watches",
        ),
        DeclareLaunchArgument(
            "signal_debug",
            default_value="false",
            description="Annotated frames on /start_signal_detector/debug_image",
        ),
        DeclareLaunchArgument(
            "record", default_value="true", description="true | false"
        ),
        DeclareLaunchArgument("record_label", default_value=""),
        DeclareLaunchArgument(
            "record_args",
            default_value="",
            description='extra record_run.py flags, e.g. "--svo"',
        ),
    ]

    start_signal = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "start_signal.launch.py"])
        ),
        launch_arguments={
            "image_topic": LaunchConfiguration("image_topic"),
            "debug": LaunchConfiguration("signal_debug"),
        }.items(),
    )
    # Arms on the start signal and counts only under AUTO_ACTIVE (free_run
    # false), exactly as at a race.
    lap_counter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "lap_counter.launch.py"])
        ),
        launch_arguments={
            "laps": LaunchConfiguration("laps"),
            "free_run": "false",
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
            "use_sim_time": "false",
        }.items(),
    )
    return LaunchDescription(
        [*args, start_signal, lap_counter, driver, OpaqueFunction(function=recorder)]
    )


def recorder(context, *args, **kwargs):
    """record_run.py, as formula_one.launch.py starts it, minus the cloud.

    Long signal timeouts on purpose: on Ctrl-C the recorder still has to save
    the ZED area memory, close the bag and copy the logs.
    """

    def value(name):
        return LaunchConfiguration(name).perform(context)

    if value("record").lower() not in ("true", "1"):
        return []
    driver = value("driver")
    label = value("record_label") or (
        "prior" if driver == "prior" else Path(value("policy")).parent.name
    )
    script = HERE.parents[1] / "jetson" / "scripts" / "record_run.py"
    cmd = [
        "python3",
        str(script),
        "--label",
        f"or_{label}",
        "--driver",
        driver,
        "--speed-scale",
        value("speed_scale"),
        "--config",
        value("config"),
        "--cloud-hz",
        "0",
        "--extra-topic",
        "/obstacle_racer/telemetry",
    ]
    if driver != "prior":
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
