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
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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

    use_sim = LaunchConfiguration("use_sim").perform(context).lower() in ("true", "1")
    # The real Arduino has its own node name; the Gazebo stand-in is a
    # different node ("sim_vehicle", see simulation.launch.py), so the runner
    # has to be told which one hosts the (no-op, in sim) speed_* gains
    # parameter service.
    bridge_node_name = "/sim_vehicle" if use_sim else "/arduino_bridge"

    use_zed = LaunchConfiguration("use_zed").perform(context).lower() in ("true", "1")

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
    )

    # Gazebo stand-in for the Arduino + ZED, so the characterization procedure
    # itself - arming, the safety envelope, gains handshake, CSV/bag output -
    # can be rehearsed before it ever runs against the real car.  The
    # simulated vehicle is an ideal, instant-response model (see
    # sim_vehicle_node.cpp): this exercises the software, not the plant, and
    # produces no data that belongs in vehicle.yaml.
    world_name = LaunchConfiguration("world").perform(context)
    world_path = (
        world_name
        if os.path.isabs(world_name)
        else os.path.join(share, "worlds", world_name)
    )
    resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        [
            PathJoinSubstitution([FindPackageShare("cfr_arduino_bridge"), ".."]),
            ":",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )
    gui = LaunchConfiguration("gui").perform(context).lower() in ("true", "1")
    gz_flags = "-r -v 3" if gui else "-r -s -v 3"
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={"gz_args": f"{gz_flags} {world_path}"}.items(),
    )
    sim_vehicle = Node(
        package="cfr_arduino_bridge",
        executable="sim_vehicle_node",
        name="sim_vehicle",
        output="screen",
        parameters=[params_file, {"use_sim_time": True}, bridge_overrides],
        remappings=[
            ("~/drive_cmd", "/drive_cmd"),
            ("~/status", "/arduino_bridge/status"),
            # Gazebo's AckermannSteering plugin subscribes here (see the world
            # file); missing this remap leaves the vehicle receiving nothing
            # and looking stationary with no error anywhere.
            ("cmd_vel", "/sim/cmd_vel"),
        ],
    )
    gazebo_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            # ROS->GZ: without this, sim_vehicle_node's Twist never reaches
            # Gazebo's transport - the AckermannSteering plugin subscribes via
            # gz transport, not ROS, and the two only meet through this bridge.
            "/sim/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/model/slash/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/model/slash/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        remappings=[
            ("/model/slash/odometry", "/zed/zed_node/odom"),
            ("/model/slash/pose", "/zed/zed_node/pose"),
        ],
        parameters=[{"use_sim_time": True}],
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
                "bridge_node": bridge_node_name,
                "use_sim_time": use_sim,
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

    actions = [banner]
    if use_sim:
        actions += [resource_path, gazebo, sim_vehicle, gazebo_bridge]
    else:
        actions.append(bridge)
        if use_zed:
            actions.append(zed)
    actions += [runner, shutdown_with_runner]
    return actions


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
            DeclareLaunchArgument(
                "use_sim",
                default_value="false",
                description="Run against Gazebo (sim_vehicle_node) instead of the real "
                "Arduino and ZED. Rehearses the procedure, not the plant - the sim "
                "vehicle is an ideal instant-response model, so runs against it produce "
                "no data for vehicle.yaml. The sim never reports E-Stop asserted, so "
                "pass require_estop_cycle:=false alongside this.",
            ),
            DeclareLaunchArgument(
                "world",
                default_value="speed_course.sdf",
                description="Gazebo world (bare filename under worlds/, or an absolute "
                "path); only used with use_sim:=true",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="Show the Gazebo GUI; only used with use_sim:=true and "
                "needs an authorized display",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
