"""Plan Z on the car: start signal, lap counter, driver, recorder.

LAUNCHING THIS ARMS THE ACTUATORS.  Run it only at the course with the E-Stop
remote in hand, never unattended or in the background.
jetson/scripts/launchPlanZ.sh is the way to start it: it checks the stack
and asks for GO first.

Two terminals on the Orin.  First the Arduino bridge and the ZED, with
cmd_vel_to_drive off (it would be a second publisher on /drive_cmd):

    ~/software/scripts/launch.sh --no-cmd-vel

Then this, through the wrapper:

    ~/software/scripts/launchPlanZ.sh --course obstacle

It adds, beside the bridge and camera:

    start_signal_detector   on the ZED's color image; latches /start_signal_detector/go
    lap_counter             `laps` laps (2 Obstacle Course, 3 Speed Course),
                            with the lateral gate the Obstacle Course needs
    plan_z_node             drives on go or Manual Start, throttle off on done
    record_run.py           the run, into ~/cfr_runs (record:=false to skip)

The car will not move until the detector sees red turn green, or the
Arduino's Manual Start bit goes from 0 to 1.  From a terminal:

    ros2 service call /obstacle_racer/manual_start std_srvs/srv/SetBool "{data: true}"

(/formula_one/manual_start on the Speed Course), and `{data: false}` stops
it without killing the node.

The recorder is told --cloud-hz 0: by default it turns the ZED cloud down to
1 Hz for the run, and this driver steers from the cloud.

pose_forward is set to 0.315 here: the ZED's pose is the camera's, that far
ahead of the chassis center, and what the driver remembers of the walls it
has passed swings about the wrong point if that is left at Gazebo's 0.
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
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

HERE = Path(__file__).resolve().parent
LAPS = {"obstacle": "2", "speed": "3"}
# lap_counter's cross-track gate, m.  On the Obstacle Course the pothole lane
# crosses the plane of the start line 8.3 m to the side, heading the same way,
# and without the gate the first lap is counted there.
LATERAL_GATE = {"obstacle": "3.0", "speed": "0.0"}
TELEMETRY = {"obstacle": "/obstacle_racer/telemetry", "speed": "/formula_one/telemetry"}


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "course", default_value="obstacle", description="obstacle | speed"
        ),
        DeclareLaunchArgument("config", default_value=str(HERE / "config.yaml")),
        DeclareLaunchArgument("knobs", default_value=""),
        DeclareLaunchArgument(
            "speed_scale", default_value="1.0", description="work up from 0.3"
        ),
        DeclareLaunchArgument(
            "laps", default_value="", description="default: 2 obstacle, 3 speed"
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/zed/zed_node/rgb/color/rect/image",
            description="Color image the start signal detector watches",
        ),
        DeclareLaunchArgument("signal_debug", default_value="false"),
        DeclareLaunchArgument(
            "record", default_value="true", description="true | false"
        ),
        DeclareLaunchArgument("record_label", default_value=""),
        DeclareLaunchArgument("record_args", default_value=""),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=stack)])


def stack(context, *args, **kwargs):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    course = value("course")
    share = FindPackageShare("cfr_arduino_bridge").perform(context)
    start_signal = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(f"{share}/launch/start_signal.launch.py"),
        launch_arguments={
            "image_topic": value("image_topic"),
            "debug": value("signal_debug"),
        }.items(),
    )
    # Arms on the start signal and counts only under AUTO_ACTIVE, as at a race.
    lap_counter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(f"{share}/launch/lap_counter.launch.py"),
        launch_arguments={
            "laps": value("laps") or LAPS[course],
            "free_run": "false",
            "lateral_gate": LATERAL_GATE[course],
        }.items(),
    )
    knobs = f"pose_forward=0.315 {value('knobs')}".strip()
    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(HERE / "plan_z.launch.py")),
        launch_arguments={
            "course": course,
            "config": value("config"),
            "knobs": knobs,
            "speed_scale": value("speed_scale"),
            "use_sim_time": "false",
        }.items(),
    )
    actions = [start_signal, lap_counter, driver]
    if value("record").lower() in ("true", "1"):
        script = HERE.parents[1] / "jetson" / "scripts" / "record_run.py"
        cmd = [
            "python3",
            str(script),
            "--label",
            f"pz_{value('record_label') or course}",
            "--driver",
            "plan_z",
            "--speed-scale",
            value("speed_scale"),
            "--config",
            value("config"),
            "--cloud-hz",
            "0",
            "--extra-topic",
            TELEMETRY[course],
        ]
        cmd += shlex.split(value("record_args"))
        # Long signal timeouts on purpose: on Ctrl-C the recorder still has
        # to save the ZED area memory, close the bag and copy the logs.
        actions.append(
            ExecuteProcess(
                cmd=cmd, output="screen", sigterm_timeout="45", sigkill_timeout="60"
            )
        )
    return actions
