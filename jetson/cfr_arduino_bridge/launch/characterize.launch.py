"""Run one characterization profile and record it locally on the Jetson.

    ros2 launch cfr_arduino_bridge characterize.launch.py profile:=coastdown

That is the whole interface, on purpose.  Testing happens away from a network,
on a laptop that may lose Wi-Fi to the car mid-run, so a session that needs
several terminals and a remembered set of parameter overrides is a session that
goes wrong in the field.  One command, one profile name.

This launch owns the run directory.  It is picked here, before anything starts,
so the Arduino serial traces can be pointed INTO it - those are opened once at
node start and truncated, so they cannot be redirected later without restarting
the bridge and losing the Arduino handshake.

See docs/characterization.md for the procedure and docs/field-card.md for the
one-page printable version.
"""

import os
from datetime import datetime, timezone

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_profile(share, name):
    """Accept either a bare profile name or a path to a profile file."""
    if os.path.sep in name or name.endswith(".yaml"):
        path = os.path.abspath(os.path.expanduser(name))
    else:
        path = os.path.join(share, "config", "profiles", f"{name}.yaml")
    if not os.path.isfile(path):
        available = sorted(
            os.path.splitext(entry)[0]
            for entry in os.listdir(os.path.join(share, "config", "profiles"))
            if entry.endswith(".yaml")
        )
        raise RuntimeError(
            f"no profile at {path}. Available profiles: {', '.join(available)}"
        )
    return path


def _launch_setup(context, *args, **kwargs):
    share = get_package_share_directory("cfr_arduino_bridge")
    profile_name = LaunchConfiguration("profile").perform(context)
    profile_path = _resolve_profile(share, profile_name)

    run_root = os.path.expanduser(LaunchConfiguration("run_root").perform(context))
    label = LaunchConfiguration("label").perform(context)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    basename = os.path.splitext(os.path.basename(profile_path))[0]
    suffix = f"_{label}" if label else ""
    run_dir = os.path.join(run_root, f"{stamp}_{basename}{suffix}")
    os.makedirs(run_dir, exist_ok=True)

    params_file = os.path.join(share, "config", "arduino_bridge.yaml")
    slew = LaunchConfiguration("speed_slew_rate").perform(context)

    bridge_overrides = {
        "device": LaunchConfiguration("device").perform(context),
        # Traces land in the run directory rather than /tmp, where the next run
        # would truncate them.  The host timestamp is what lets the Arduino's
        # own millis()-stamped lines be joined to telemetry.csv and the bag.
        "tx_trace_path": os.path.join(run_dir, "arduino_tx.log"),
        "rx_trace_path": os.path.join(run_dir, "arduino_rx.log"),
        "trace_timestamps": True,
        "max_speed": float(LaunchConfiguration("max_speed").perform(context)),
    }
    if slew:
        # Raised for step-response work: the production 2.0 m/s^2 takes 1.6 s to
        # reach 3.2 m/s and would swamp a plant time constant near 0.5 s.  Left
        # alone for tune_profile, which scores the loop as it will be flown.
        bridge_overrides["speed_slew_rate"] = float(slew)

    bridge = Node(
        package="cfr_arduino_bridge",
        executable="arduino_bridge_node",
        name="arduino_bridge",
        output="screen",
        parameters=[params_file, bridge_overrides],
        remappings=[("~/drive_cmd", "/drive_cmd")],
    )

    zed = ExecuteProcess(
        cmd=[
            "ros2",
            "launch",
            "zed_wrapper",
            "zed_camera.launch.py",
            f"camera_model:={LaunchConfiguration('zed_model').perform(context)}",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_zed")),
    )

    runner = Node(
        package="cfr_arduino_bridge",
        executable="maneuver_runner_node.py",
        name="maneuver_runner",
        output="screen",
        parameters=[
            {
                "profile": profile_path,
                "run_dir": run_dir,
                "operator": LaunchConfiguration("operator").perform(context),
                "surface": LaunchConfiguration("surface").perform(context),
                "notes": LaunchConfiguration("notes").perform(context),
                "gain_overrides": LaunchConfiguration("gains").perform(context),
                "record_bag": LaunchConfiguration("record_bag").perform(context).lower()
                == "true",
                "require_estop_cycle": LaunchConfiguration("require_estop_cycle")
                .perform(context)
                .lower()
                == "true",
            }
        ],
        remappings=[
            ("drive_cmd", "/drive_cmd"),
            ("status", "/arduino_bridge/status"),
            ("odom", LaunchConfiguration("odom_topic").perform(context)),
        ],
    )

    banner = ExecuteProcess(
        cmd=["echo", f"[characterize] profile={basename}  run_dir={run_dir}"],
        output="screen",
    )

    # The runner finishing is the end of the session.  Tearing the bridge down
    # with it matches launch.sh's fate sharing, and avoids leaving a live
    # actuator link attached to a node that is no longer driving it.
    shutdown_with_runner = RegisterEventHandler(
        OnProcessExit(target_action=runner, on_exit=[EmitEvent(event=Shutdown())])
    )

    return [banner, bridge, zed, runner, shutdown_with_runner]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile",
                description="Profile name (e.g. coastdown) or a path to a profile YAML",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="/dev/ttyACM0",
                description="Arduino USB serial device",
            ),
            DeclareLaunchArgument(
                "run_root",
                default_value="~/cfr_runs",
                description="Where run directories are created, on the Jetson",
            ),
            DeclareLaunchArgument(
                "label",
                default_value="",
                description="Optional suffix on the run directory, e.g. kp24",
            ),
            DeclareLaunchArgument(
                "gains",
                default_value="",
                description='Gain overrides, e.g. "speed_kp=24.0 speed_ki=10.0"',
            ),
            DeclareLaunchArgument(
                "speed_slew_rate",
                default_value="",
                description="Override the bridge slew rate; raise it for step response work",
            ),
            DeclareLaunchArgument(
                "max_speed",
                default_value="5.0",
                description="Bridge speed clamp for the session, m/s",
            ),
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/zed/zed_node/odom",
                description="Odometry the runner uses for distance and aborts",
            ),
            DeclareLaunchArgument(
                "use_zed", default_value="true", description="Start the ZED camera node"
            ),
            DeclareLaunchArgument(
                "zed_model", default_value="zed2i", description="ZED camera model"
            ),
            DeclareLaunchArgument(
                "record_bag",
                default_value="true",
                description="Record a rosbag alongside telemetry.csv",
            ),
            DeclareLaunchArgument(
                "require_estop_cycle",
                default_value="true",
                description="Require E-Stop asserted then cleared before moving. "
                "Only set false on blocks, never on the ground.",
            ),
            DeclareLaunchArgument(
                "operator", default_value="", description="Recorded in metadata"
            ),
            DeclareLaunchArgument(
                "surface", default_value="asphalt", description="Recorded in metadata"
            ),
            DeclareLaunchArgument(
                "notes", default_value="", description="Recorded in metadata"
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
