#!/usr/bin/env bash
# Collect teacher samples across as many simulator deaths as it takes.
#
#   CFR_SENSORS=1 ./pretrain_resilient.sh 20000
#
# The ZED render pipeline takes the sim down every 15-30 minutes, so this
# restarts the stack and keeps collecting; samples accumulate in the .npz.
# The supervised fit runs afterwards and needs no simulator.

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${1:?usage: pretrain_resilient.sh <samples> [extra pretrain_lap.py args]}"
shift || true
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR}"
SAMPLES="${SAMPLES:-$SCRIPT_DIR/checkpoints_lap0/teacher.npz}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

kill_sim() {
    ps -eo pid,cmd \
      | grep -E "gz sim|ros2 launch cfr_arduino_bridge|parameter_bridge|sim_vehicle_node|cmd_vel_to_drive|teleport_api" \
      | grep -v grep | awk '{print $1}' \
      | while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done
    sleep 3
}

start_sim() {
    kill_sim
    ros2 daemon stop >/dev/null 2>&1 || true
    setsid nohup ros2 launch cfr_arduino_bridge training.launch.py \
        sensors:="$([ -n "${CFR_SENSORS}" ] && echo true || echo false)" \
        > "$LOG_DIR/pretrain_sim.log" 2>&1 &
    local deadline=$((SECONDS + 240))
    while [ $SECONDS -lt $deadline ]; do
        if (exec 3<>/dev/tcp/localhost/9003) 2>/dev/null; then
            exec 3>&-
            if ros2 topic list 2>/dev/null | grep -q dynamic_pose; then
                if [ -z "${CFR_SENSORS}" ] || timeout 20 ros2 topic echo --once \
                     /zed/zed_node/point_cloud/cloud_registered >/dev/null 2>&1; then
                    return 0
                fi
            fi
        fi
        sleep 2
    done
    return 1
}

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$REPO_ROOT/install/setup.bash"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"

have() { python - "$SAMPLES" <<'PY'
import sys, pathlib, numpy as np
path = pathlib.Path(sys.argv[1])
print(len(np.load(path)["observations"]) if path.exists() else 0)
PY
}

attempt=0
while [ "$(have)" -lt "$TARGET" ]; do
    attempt=$((attempt + 1))
    log "collection attempt $attempt: $(have)/$TARGET samples so far"
    start_sim || { log "simulation failed to start; retrying"; continue; }
    set +e
    python "$SCRIPT_DIR/pretrain_lap.py" --collect-only --steps "$TARGET" \
        --samples "$SAMPLES" "$@" >> "$LOG_DIR/pretrain.log" 2>&1
    set -e
    log "attempt $attempt ended with $(have)/$TARGET samples"
done

kill_sim
log "collected $(have) samples; fitting (no simulator needed)"
python "$SCRIPT_DIR/pretrain_lap.py" --fit --samples "$SAMPLES" "$@" \
    2>&1 | tee -a "$LOG_DIR/pretrain.log"
