#!/usr/bin/env bash
# One-command lap-time training: brings the headless Gazebo stack up, runs
# PPO against it, tears it down again. Arguments pass through to train_lap.py.
#
#   ./launch_lap_training.sh --max-speed 3.0 --total-timesteps 150000 \
#       --checkpoint-dir checkpoints_lap1
#
# config_lap.yaml sets scan_source: cloud, so the ZED has to be rendered --
# export CFR_SENSORS=1 before running, or the env aborts after 200 steps
# rather than silently training on the analytic scan.
# CFR_USE_RUNNING_SIM=1 reuses a sim you started yourself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_sim_stack.sh"

echo "starting PPO lap training"
python "$SCRIPT_DIR/train_lap.py" "$@"
