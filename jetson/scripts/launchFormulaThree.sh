#!/usr/bin/env bash
#
# Launch the formulaThree driver on the car (map + ZED depth, three laps), at a
# third of full speed by default.
#
#   ~/software/scripts/launch.sh --no-cmd-vel      # terminal 1: bridge + ZED
#   ~/software/scripts/launchFormulaThree.sh         # terminal 2: this
#   ~/software/scripts/launchFormulaThree.sh -s 0.5 --depth-fallback map
#   ~/software/scripts/launchFormulaThree.sh -n      # print the command only
#
# Checks formulaOne's list plus the depth image (the policy will not move
# without it) and the ZED IMU (the depth scan is levelled on it), then asks
# for GO with the E-stop in hand.  See rl/formulaThree/README.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_NAME=formulaThree
DRIVER_DIR="${FORMULA_THREE_DIR:-$(dirname "${SCRIPT_DIR}")/formulaThree}"
LAUNCH_FILE="${DRIVER_DIR}/formula_three.launch.py"
NEEDS_DEPTH=true

# shellcheck source=_formula_launch.sh
source "${SCRIPT_DIR}/_formula_launch.sh"
formula_launch "$@"
