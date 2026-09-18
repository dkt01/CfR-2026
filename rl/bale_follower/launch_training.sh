#!/usr/bin/env bash
# One-command RL training: brings up the headless Gazebo stack, waits until
# the teleport API and ground-truth pose bridge are ready, runs PPO training,
# and tears the stack down again on exit (clean or not).
#
#   ./launch_training.sh                          # config.yaml defaults
#   ./launch_training.sh --total-timesteps 100000
#   ./launch_training.sh --resume-from checkpoints/final_model.zip
#
# All arguments are passed through to train.py. Max speed, traction, the
# simulated-ZED observation model, and reward shaping live in config.yaml.
# Set CFR_USE_RUNNING_SIM=1 to train against a sim you started yourself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_sim_stack.sh"

echo "starting PPO training"
python "$SCRIPT_DIR/train.py" "$@"
