#!/usr/bin/env bash
# One-command policy evaluation: brings up the headless Gazebo stack, runs N
# deterministic episodes of a trained checkpoint through the CasADi command
# smoother, prints/saves metrics (distance, speed, collision rate, steering
# jerk), and tears the stack down on exit.
#
#   ./test_policy.sh                                        # final_model, 5 episodes
#   ./test_policy.sh --episodes 10 --max-speed 2.0
#   ./test_policy.sh --no-smoother                          # raw-policy A/B baseline
#   ./test_policy.sh --checkpoint checkpoints/bale_follower_4000_steps.zip
#
# All arguments are passed through to evaluate.py. Set CFR_USE_RUNNING_SIM=1
# to evaluate against a sim you started yourself (e.g. the web-viewer stack).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_sim_stack.sh"

echo "starting evaluation"
python "$SCRIPT_DIR/evaluate.py" "$@"
