#!/usr/bin/env bash
# Race path_racer.py against the training stack, but count laps through the
# REAL lap_counter node (free-run: require_go/require_auto_active off) rather
# than path_racer's own arc-length log -- the same message a competition run
# reports, driven off ground-truth pose via pose_republisher.py so the full
# simulation.launch.py stack (whose path_follower_node would fight path_racer
# for /cmd_vel) is not needed.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DURATION="${1:-260}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR}"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/install/setup.bash"
source "$SCRIPT_DIR/.venv/bin/activate"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    ps -eo pid,cmd | grep -E "gz sim|ros2 launch cfr_arduino_bridge|parameter_bridge|sim_vehicle_node|cmd_vel_to_drive|teleport_api|lap_counter_node|pose_republisher" \
        | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
}
trap cleanup EXIT

setsid nohup ros2 launch cfr_arduino_bridge training.launch.py sensors:=false \
    > "$LOG_DIR/official_sim.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 1 60); do
    (exec 3<>/dev/tcp/localhost/9003) 2>/dev/null && { exec 3>&-; ros2 topic list 2>/dev/null | grep -q dynamic_pose && break; }
    sleep 2
done

python "$SCRIPT_DIR/pose_republisher.py" > "$LOG_DIR/pose_republisher.log" 2>&1 &
PIDS+=($!)

ros2 run cfr_arduino_bridge lap_counter_node.py --ros-args \
    -p use_sim_time:=true \
    -p target_laps:=7 \
    -p require_go:=false \
    -p require_auto_active:=false \
    -r pose:=/sim_ground_truth/pose \
    -r status:=/arduino_bridge/status \
    > "$LOG_DIR/official_lap_counter.log" 2>&1 &
PIDS+=($!)

sleep 3   # let the counter subscribe and arm before the car starts moving

python "$SCRIPT_DIR/lap_count_listener.py" > "$LOG_DIR/official_lap_count.log" 2>&1 &
PIDS+=($!)

timeout "$DURATION" python "$SCRIPT_DIR/path_racer.py" > "$LOG_DIR/official_path_racer.log" 2>&1 || true

echo "=== official lap_counter log ==="
cat "$LOG_DIR/official_lap_count.log"
echo "=== path_racer's own log, for comparison ==="
grep "LAP" "$LOG_DIR/official_path_racer.log" || true
