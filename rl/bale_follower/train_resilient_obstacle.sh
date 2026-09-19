#!/usr/bin/env bash
# Long training runs on the Obstacle Course that survive the simulator dying
# under them. Sibling to train_resilient.sh (see that file for the full
# rationale -- Gazebo's pose stream dies after roughly five hours of
# continuous training) but pointed at obstacle_course.launch.py and
# train_obstacle.py instead of the Speed Course's training.launch.py/train.py:
#
# - ObstacleCourseEnv perceives entirely through the ZED's simulated point
#   cloud, so `sensors:=true` is not optional the way CFR_SENSORS is for the
#   Speed Course -- it is always on here.
# - Readiness is checked on /zed/zed_node/pose and the point cloud topic
#   instead of grepping for dynamic_pose: ObstacleCourseEnv reads pose from
#   the per-model /zed/zed_node/pose topic, not the world's dynamic_pose
#   bridge (see the obstacle-course-dynamic-pose-bridge-bug project note).
# - obstacle_course.launch.py also starts the randomizer and hoop_monitor,
#   which train_obstacle.py needs for layout cycling and hoop pass/fail.
# - `autonomy:=false` is mandatory, and was missing until now. That launch
#   file brings up the full stack including path_follower_node, which
#   publishes zeros on /cmd_vel while idle -- the same topic the RL env
#   drives. Two publishers on one topic meant the vehicle saw the policy's
#   commands interleaved with path_follower's zeros and achieved about a
#   tenth of what was asked (measured: commanded 1.47 m/s, achieved 0.15).
#   training.launch.py exists for exactly this reason on the Speed Course
#   ("minus path_follower_node ... would fight the policy for control");
#   the Obstacle Course cannot use it because it needs the randomizer and
#   hoop_monitor, so simulation.launch.py now takes an `autonomy` argument.
#
#   ./train_resilient_obstacle.sh 200000 checkpoints_obstacle_v1
#   ./train_resilient_obstacle.sh 200000 checkpoints_obstacle_v1 --resume-from checkpoints_obstacle_v1/best_model.zip

# No -u: ROS 2's setup.bash trips over unbound variables under nounset.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOTAL="${1:?usage: train_resilient_obstacle.sh <total_timesteps> <checkpoint_dir> [extra train_obstacle.py args]}"
CKPT_DIR="${2:?usage: train_resilient_obstacle.sh <total_timesteps> <checkpoint_dir> [extra train_obstacle.py args]}"
shift 2
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR}"
RUN_LOG="$LOG_DIR/train_resilient_obstacle.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

kill_sim() {
    ps -eo pid,cmd \
        | grep -E "gz sim|ros2 launch cfr_arduino_bridge|parameter_bridge|sim_vehicle_node|cmd_vel_to_drive|teleport_api|path_follower_node|lap_counter|start_signal|obstacle_randomizer_node|hoop_monitor_node" \
        | grep -v grep | awk '{print $1}' \
        | while read -r pid; do kill -9 "$pid" 2>/dev/null || true; done
    sleep 3
}

start_sim() {
    kill_sim
    ros2 daemon stop >/dev/null 2>&1 || true
    setsid nohup ros2 launch cfr_arduino_bridge obstacle_course.launch.py \
        sensors:=true autonomy:=false \
        > "$LOG_DIR/resilient_sim.log" 2>&1 &
    for _ in $(seq 1 60); do
        if (exec 3<>/dev/tcp/localhost/9003) 2>/dev/null; then
            exec 3>&-
            if ros2 topic list 2>/dev/null | grep -q "/zed/zed_node/pose"; then
                [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: pose ok"
                local servers
                servers=$(ps -eo cmd | grep "^gz sim -r -s" -c || true)
                if [ "$servers" -ne 1 ]; then
                    log "WARNING: $servers gz servers running, expected 1 -- duplicate"
                    log "servers publish onto one pose topic and silently corrupt training"
                fi
                # Wait for an actual cloud message, not a publisher count: see
                # train_resilient.sh's identical wait for why.
                if ! timeout 15 ros2 topic echo --once \
                    /zed/zed_node/point_cloud/cloud_registered >/dev/null 2>&1; then
                    [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: no cloud publisher yet"
                    sleep 2
                    continue
                fi
                [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: all checks passed"
                return 0
            else
                [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: no /zed/zed_node/pose yet"
            fi
        else
            [ -n "${CFR_START_DEBUG}" ] && log "  start_sim: teleport API not up"
        fi
        sleep 2
    done
    return 1
}

newest_checkpoint() {
    ls -t "$CKPT_DIR"/obstacle_course_*_steps.zip 2>/dev/null | head -1
}

steps_in() {
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
    python "$SCRIPT_DIR/train_obstacle.py" --total-timesteps "$remaining" \
        --checkpoint-dir "$CKPT_DIR" "${resume_args[@]}" \
        >> "$LOG_DIR/train_obstacle_chunk_${attempt}.log" 2>&1
    status=$?
    set -e

    after=$(newest_checkpoint || true)
    after_steps=0
    [ -n "$after" ] && after_steps=$(steps_in "$after")
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

    log "chunk $attempt died with status $status (+$gained steps, $completed/$TOTAL); see train_obstacle_chunk_${attempt}.log"
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
