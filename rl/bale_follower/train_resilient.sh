#!/usr/bin/env bash
# Long training runs that survive the simulator dying under them.
#
# Gazebo's pose stream stops after roughly five hours of continuous training;
# the v4 run died that way at 139k of 250k steps and had to be resumed by
# hand. This wrapper runs training in chunks: when a chunk exits early it
# tears the whole simulation stack down, brings up a fresh one, and resumes
# from the newest checkpoint, until the cumulative step target is reached.
#
#   ./train_resilient.sh 250000 checkpoints_v5
#   ./train_resilient.sh 250000 checkpoints_v5 --resume-from checkpoints_v4/best_model.zip
#
# Progress is tracked cumulatively across chunks because SB3 restarts its own
# step counter on every resume -- the per-chunk checkpoint names cannot be
# compared across restarts.

# No -u: ROS 2's setup.bash trips over unbound variables under nounset.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOTAL="${1:?usage: train_resilient.sh <total_timesteps> <checkpoint_dir> [extra train.py args]}"
CKPT_DIR="${2:?usage: train_resilient.sh <total_timesteps> <checkpoint_dir> [extra train.py args]}"
shift 2
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR}"
RUN_LOG="$LOG_DIR/train_resilient.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

kill_sim() {
    # Match the bare `gz sim` server too, not just the ros2 launch wrapper:
    # orphaned servers survived earlier cleanups and silently corrupted a day
    # of measurements by publishing a second car onto the same pose topic.
    ps -eo pid,cmd \
        | grep -E "gz sim|ros2 launch cfr_arduino_bridge|parameter_bridge|sim_vehicle_node|cmd_vel_to_drive|teleport_api" \
        | grep -v grep | awk '{print $1}' \
        | while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done
    sleep 3
}

start_sim() {
    kill_sim
    # CFR_SENSORS=1 renders the ZED and bridges its point cloud, which the
    # environment needs when config.yaml sets scan_source: cloud. This wrapper
    # relaunches the sim on every chunk, so the flag has to be repeated here --
    # setting it only where the stack is first brought up silently drops the
    # camera on the first restart, and the run dies mid-chunk.
    setsid nohup ros2 launch cfr_arduino_bridge training.launch.py \
        sensors:="$([ -n "${CFR_SENSORS}" ] && echo true || echo false)" \
        > "$LOG_DIR/resilient_sim.log" 2>&1 &
    for _ in $(seq 1 60); do
        if (exec 3<>/dev/tcp/localhost/9003) 2>/dev/null; then
            exec 3>&-
            if ros2 topic list 2>/dev/null | grep -q dynamic_pose; then
                [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: pose ok"
                # Count the gz binary only: a single logical server also shows
                # up as a `/bin/sh -c ruby ... gz sim` wrapper, so matching the
                # whole command line reports 2-3 for one healthy server.
                local servers
                servers=$(ps -eo cmd | grep "^gz sim -r -s" -c || true)
                if [ "$servers" -ne 1 ]; then
                    log "WARNING: $servers gz servers running, expected 1 -- duplicate"
                    log "servers publish onto one pose topic and silently corrupt training"
                fi
                if [ -n "${CFR_SENSORS}" ]; then
                    # Wait for an actual message, not a publisher count:
                    # `ros2 topic info` starts a fresh node per call and
                    # reports what it discovered in a short window, so it
                    # under-reports a live publisher often enough to burn the
                    # whole retry budget (observed). Receiving one cloud
                    # proves the render pipeline and the bridge both work.
                    if ! timeout 15 ros2 topic echo --once \
                        /zed/zed_node/point_cloud/cloud_registered >/dev/null 2>&1; then
                        # sleep before continuing: a bare `continue` skips the
                        # loop's sleep and spends the whole retry budget in a
                        # few hundred milliseconds, long before the render
                        # pipeline has produced its first cloud.
                        [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: no cloud publisher yet"
                        sleep 2
                        continue
                    fi
                fi
                [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: all checks passed"
                return 0
            else
                [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: no dynamic_pose yet"
            fi
        else
            [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: teleport API not up"
        fi
        sleep 2
    done
    return 1
}

newest_checkpoint() {
    ls -t "$CKPT_DIR"/bale_follower_*_steps.zip 2>/dev/null | head -1
}

steps_in() {  # steps recorded in a checkpoint filename
    basename "$1" | sed -E 's/.*_([0-9]+)_steps\.zip/\1/'
}

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$REPO_ROOT/install/setup.bash"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"
mkdir -p "$CKPT_DIR"

log "target $TOTAL steps into $CKPT_DIR"
completed=0
attempt=0
resume_args=("$@")

while [ "$completed" -lt "$TOTAL" ]; do
    attempt=$((attempt + 1))
    remaining=$((TOTAL - completed))
    log "chunk $attempt: $remaining steps remaining"

    start_sim || { log "simulation failed to start; retrying"; sleep 10; continue; }

    before=$(newest_checkpoint || true)
    before_steps=0
    [ -n "$before" ] && before_steps=$(steps_in "$before")

    set +e
    python "$SCRIPT_DIR/train.py" --total-timesteps "$remaining" \
        --checkpoint-dir "$CKPT_DIR" "${resume_args[@]}" \
        >> "$LOG_DIR/train_chunk_${attempt}.log" 2>&1
    status=$?
    set -e

    after=$(newest_checkpoint || true)
    after_steps=0
    [ -n "$after" ] && after_steps=$(steps_in "$after")
    # A resumed chunk restarts SB3's counter, so progress is the newest
    # checkpoint's own step count, not a difference against the previous run.
    if [ -n "$before" ] && [ "$after_steps" -gt "$before_steps" ] && [ "$attempt" -eq 1 ]; then
        gained=$((after_steps - before_steps))
    else
        gained=$after_steps
    fi
    completed=$((completed + gained))

    if [ "$status" -eq 0 ]; then
        log "chunk $attempt finished cleanly (+$gained steps, $completed/$TOTAL)"
        break
    fi

    log "chunk $attempt died with status $status (+$gained steps, $completed/$TOTAL); see train_chunk_${attempt}.log"
    latest=$(newest_checkpoint || true)
    if [ -z "$latest" ]; then
        log "no checkpoint to resume from; aborting"
        kill_sim
        exit 1
    fi
    log "resuming from $latest"
    resume_args=(--resume-from "$latest")
    sleep 5
done

log "done: $completed steps over $attempt chunk(s)"
log "best deterministic checkpoint: $CKPT_DIR/best_model.zip"
kill_sim
