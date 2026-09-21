#!/usr/bin/env bash
# Scripted-driver shakedown of the lap environment against a real simulation.
#
#   ./lap_live_check.sh --laps 2
#   CFR_SENSORS=1 ./lap_live_check.sh --laps 2 --scan-source cloud

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_sim_stack.sh"

python "$SCRIPT_DIR/lap_live_check.py" "$@"
