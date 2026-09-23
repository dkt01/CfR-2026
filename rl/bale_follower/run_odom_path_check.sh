#!/usr/bin/env bash
# Smoke-test path_racer.py's --pose-msg odom code path against a synthetic
# ground-truth Odometry republisher, standing in for the QuestNav republisher
# that does not exist in this repo yet. Proves the CODE PATH, not the
# hardware -- see odom_republisher.py's docstring.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DURATION="${1:-80}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR}"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/install/setup.bash"
source "$SCRIPT_DIR/.venv/bin/activate"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    ps -eo pid,cmd | grep -E "gz sim|ros2 launch cfr_arduino_bridge|parameter_bridge|sim_vehicle_node|cmd_vel_to_drive|teleport_api|odom_republisher" \
        | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
}
trap cleanup EXIT

setsid nohup ros2 launch cfr_arduino_bridge training.launch.py sensors:=false \
    > "$LOG_DIR/odom_sim.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 1 60); do
    (exec 3<>/dev/tcp/localhost/9003) 2>/dev/null && { exec 3>&-; ros2 topic list 2>/dev/null | grep -q dynamic_pose && break; }
    sleep 2
done

python "$SCRIPT_DIR/odom_republisher.py" > "$LOG_DIR/odom_republisher.log" 2>&1 &
PIDS+=($!)
sleep 2

timeout "$DURATION" python "$SCRIPT_DIR/path_racer.py" \
    --pose-msg odom --pose-topic /sim_ground_truth/odom \
    > "$LOG_DIR/odom_path_racer.log" 2>&1 || true

echo "=== --pose-msg odom run ==="
grep -E "LAP|waiting for pose|flipped" "$LOG_DIR/odom_path_racer.log" || cat "$LOG_DIR/odom_path_racer.log"
