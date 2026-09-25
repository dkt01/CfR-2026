#!/usr/bin/env bash
#
# Launch the formulaTwo driver on the car (map + ZED depth, three laps), at a
# third of full speed by default.
#
#   ~/software/scripts/launch.sh --no-cmd-vel      # terminal 1: bridge + ZED
#   ~/software/scripts/launchFormulaTwo.sh         # terminal 2: this
#   ~/software/scripts/launchFormulaTwo.sh -s 0.5 --depth-fallback map
#   ~/software/scripts/launchFormulaTwo.sh -n      # print the command only
#
# Checks formulaOne's list plus the depth image (the policy will not move
# without it) and the ZED IMU (the depth scan is levelled on it), then asks
# for GO with the E-stop in hand.  See rl/formulaTwo/README.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_NAME=formulaTwo
DRIVER_DIR="${FORMULA_TWO_DIR:-$(dirname "${SCRIPT_DIR}")/formulaTwo}"
LAUNCH_FILE="${DRIVER_DIR}/formula_two.launch.py"
NEEDS_DEPTH=true

# shellcheck source=_formula_launch.sh
source "${SCRIPT_DIR}/_formula_launch.sh"
formula_launch "$@"
