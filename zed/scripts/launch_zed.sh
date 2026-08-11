#!/usr/bin/env bash

set -eo pipefail

ROS2_WS="$HOME/ros2_ws"
ROSBOARD_DIR="$HOME/rosboard"

source /opt/ros/jazzy/setup.bash
source "$ROS2_WS/install/setup.bash"

cleanup() {
    kill "$ZED_PID" "$ROSBOARD_PID" 2>/dev/null || true
    wait "$ZED_PID" "$ROSBOARD_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i &
ZED_PID=$!

(cd "$ROSBOARD_DIR" && ./run) &
ROSBOARD_PID=$!

echo "ZED camera node (PID $ZED_PID) and rosboard (PID $ROSBOARD_PID) running."
echo "Visualize over port 8888"

wait -n "$ZED_PID" "$ROSBOARD_PID"
