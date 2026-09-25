#!/usr/bin/env bash
#
# Launch the formulaOne driver on the car (the v12 policy unless another was
# synced), at a third of full speed by default.
#
#   ~/software/scripts/launch.sh --no-cmd-vel      # terminal 1: bridge + ZED
#   ~/software/scripts/launchFormulaOne.sh         # terminal 2: this
#   ~/software/scripts/launchFormulaOne.sh -s 0.5 --laps 1
#   ~/software/scripts/launchFormulaOne.sh --baseline
#   ~/software/scripts/launchFormulaOne.sh -n      # print the command only
#
# Checks the bridge is up, nothing else publishes /drive_cmd and the pose is
# live, then asks for GO with the E-stop in hand.  See rl/formulaOne/DEPLOY.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ~/software/scripts -> ~/software/formulaOne, as syncSoftware.sh lays it out.
DRIVER_NAME=formulaOne
DRIVER_DIR="${FORMULA_ONE_DIR:-$(dirname "${SCRIPT_DIR}")/formulaOne}"
LAUNCH_FILE="${DRIVER_DIR}/formula_one.launch.py"
NEEDS_DEPTH=false

# shellcheck source=_formula_launch.sh
source "${SCRIPT_DIR}/_formula_launch.sh"
formula_launch "$@"
