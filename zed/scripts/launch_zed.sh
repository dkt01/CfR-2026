#!/usr/bin/env bash

set -eo pipefail

ROS2_WS="$HOME/ros2_ws"
ROSBOARD_DIR="$HOME/rosboard"
# Pins positional tracking, so loop closure is on by decision rather than by
# whichever wrapper revision happens to be built.  lap_counter reads the map
# frame pose this produces; see config/cfr_zed2i.yaml.
ZED_PARAMS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../config" && pwd)/cfr_zed2i.yaml"

source /opt/ros/jazzy/setup.bash
source "$ROS2_WS/install/setup.bash"

cleanup() {
    kill "$ZED_PID" "$ROSBOARD_PID" 2>/dev/null || true
    wait "$ZED_PID" "$ROSBOARD_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i \
    ros_params_override_path:="$ZED_PARAMS" &
ZED_PID=$!

(cd "$ROSBOARD_DIR" && ./run) &
ROSBOARD_PID=$!

echo "ZED camera node (PID $ZED_PID) and rosboard (PID $ROSBOARD_PID) running."
echo "Visualize over port 8888"

wait -n "$ZED_PID" "$ROSBOARD_PID"
