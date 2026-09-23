#!/usr/bin/env bash
# Bring up the sim, run path_racer.py for a fixed wall-clock window, tear
# down. Standalone (not _sim_stack.sh) because path_racer.py runs forever
# via rclpy.spin and needs to be killed by wall clock, not by exiting.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DURATION="${1:-240}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR}"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/install/setup.bash"
source "$SCRIPT_DIR/.venv/bin/activate"

setsid nohup ros2 launch cfr_arduino_bridge training.launch.py sensors:=false \
    > "$LOG_DIR/racer_sim.log" 2>&1 &
SIM_PID=$!
trap 'kill -9 $SIM_PID 2>/dev/null; ps -eo pid,cmd | grep -E "gz sim|ros2 launch cfr_arduino_bridge|parameter_bridge|sim_vehicle_node|cmd_vel_to_drive|teleport_api" | grep -v grep | awk "{print \$1}" | xargs -r kill -9' EXIT

for _ in $(seq 1 60); do
    (exec 3<>/dev/tcp/localhost/9003) 2>/dev/null && { exec 3>&-; ros2 topic list 2>/dev/null | grep -q dynamic_pose && break; }
    sleep 2
done

timeout "$DURATION" python "$SCRIPT_DIR/path_racer.py" > "$LOG_DIR/path_racer_check.log" 2>&1 || true
echo "=== path_racer output ==="
cat "$LOG_DIR/path_racer_check.log"
