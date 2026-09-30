"""FormulaSubZero driver through the FormulaTwo vehicle and telemetry boundary."""
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

HERE = Path(__file__).resolve().parent
PYTHON = os.environ.get("FORMULA_SUB_ZERO_PYTHON", str(HERE / ".venv/bin/python"))


def generate_launch_description():
    defaults = {"config": str(HERE / "config.yaml"), "python": PYTHON, "laps": "3",
                "rviz": "false", "use_sim_time": "false", "speed_scale": "0.3",
                "record": "auto", "record_label": "mpc", "anchor": "signal", "driver": "mpc",
                "pose_is_camera": "true"}
    return LaunchDescription([
        *(DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()),
        # Gazebo already starts a detector. The bridge/ZED car launch does not.
        Node(
            package="cfr_arduino_bridge",
            executable="start_signal_detector_node.py",
            name="start_signal_detector",
            output="screen",
            parameters=[PathJoinSubstitution([
                FindPackageShare("cfr_arduino_bridge"), "config", "arduino_bridge.yaml"
            ])],
            remappings=[("image", "/zed/zed_node/rgb/color/rect/image")],
            condition=UnlessCondition(LaunchConfiguration("use_sim_time")),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(HERE.parent / "formulaTwo/formula_two.launch.py")),
            launch_arguments={**{k: LaunchConfiguration(k) for k in defaults},
                              "node_name": "formula_one"}.items(),
        ),
    ])
