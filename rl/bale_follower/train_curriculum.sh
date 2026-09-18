#!/usr/bin/env bash
# Speed curriculum: train at a survivable speed first, then raise the ceiling.
#
#   ./train_curriculum.sh                 # all three stages, ZED point cloud
#   CFR_STAGES="2 3" ./train_curriculum.sh   # resume partway
#
# An untrained policy emits actions ~N(0, 1), so mean action 0 maps to the
# middle of [-reverse_speed, max_speed]. With max_speed 6.5 that is 3.0 m/s
# from the first step, well past the 2.78 m/s the 1.31 m hairpins physically
# allow, so the car crashes before it has learned to steer, banks the
# collision penalty, and settles into not moving. Starting at 2.5 puts the
# initial speed at 1.0 m/s, which the course tolerates, and each later stage
# starts from a policy that can already drive.
#
# Each stage rewrites max_speed in config.yaml and resumes from the previous
# stage's best checkpoint. Raising max_speed rescales the action (action[0]
# spans [-reverse_speed, max_speed]) and the speed observation, so the policy
# needs a few thousand steps to re-anchor its throttle at each boundary --
# the cost of doing this in stages rather than continuously.
#
# CFR_SENSORS=1 is set for every stage: config.yaml uses scan_source: cloud,
# which needs the rendered ZED and its bridged point cloud.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
STAGES="${CFR_STAGES:-1 2 3}"

# stage: max_speed, steps, checkpoint dir
stage_speed()  { case "$1" in 1) echo 2.5;; 2) echo 4.0;; 3) echo 6.5;; esac; }
stage_steps()  { case "$1" in 1) echo 150000;; 2) echo 150000;; 3) echo 250000;; esac; }
stage_dir()    { echo "$SCRIPT_DIR/checkpoints_v9_stage$1"; }

set_max_speed() {
    python3 - "$CONFIG" "$1" <<'PY'
import re, sys
path, speed = sys.argv[1], sys.argv[2]
s = open(path).read()
s = re.sub(r"^  max_speed: [\d.]+$", f"  max_speed: {speed}", s, count=1, flags=re.M)
open(path, "w").write(s)
PY
}

log() { echo "[$(date '+%H:%M:%S')] curriculum: $*"; }

previous_best=""
for stage in $STAGES; do
    speed=$(stage_speed "$stage")
    steps=$(stage_steps "$stage")
    dir=$(stage_dir "$stage")

    set_max_speed "$speed"
    log "stage $stage -> max_speed $speed, $steps steps, into $(basename "$dir")"

    resume=()
    if [ -n "$previous_best" ] && [ -f "$previous_best" ]; then
        resume=(--resume-from "$previous_best")
        log "  resuming from $(basename "$(dirname "$previous_best")")/$(basename "$previous_best")"
    fi

    CFR_SENSORS=1 "$SCRIPT_DIR/train_resilient.sh" "$steps" "$dir" "${resume[@]}"

    if [ -f "$dir/best_model.zip" ]; then
        previous_best="$dir/best_model.zip"
        log "stage $stage done; best kept at $previous_best"
    else
        log "stage $stage produced no best_model.zip; stopping"
        exit 1
    fi
done

log "curriculum complete; final policy: $previous_best"
