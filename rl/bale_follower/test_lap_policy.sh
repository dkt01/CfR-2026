#!/usr/bin/env bash
# Evaluate a lap-time checkpoint against a fresh simulation stack.
#
#   ./test_lap_policy.sh --checkpoint checkpoints_lap2/best_model.zip --episodes 5
#   CFR_USE_RUNNING_SIM=1 ./test_lap_policy.sh --checkpoint ... --from-start
#
# Export CFR_SENSORS=1 for checkpoints trained with scan_source: cloud.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_sim_stack.sh"

python "$SCRIPT_DIR/evaluate_lap.py" "$@"
