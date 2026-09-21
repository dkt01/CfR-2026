#!/usr/bin/env bash
# Behaviour-clone the scripted driver into a policy, against a live sim.
#   CFR_SENSORS=1 ./pretrain_lap.sh --steps 20000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_sim_stack.sh"
python "$SCRIPT_DIR/pretrain_lap.py" "$@"
