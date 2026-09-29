#!/usr/bin/env bash
# FormulaSubZero on the car. Start launch.sh --no-cmd-vel in terminal 1.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_NAME=FormulaSubZero
DRIVER_DIR="${FORMULA_SUB_ZERO_DIR:-$(dirname "${SCRIPT_DIR}")/formulaSubZero}"
LAUNCH_FILE="${DRIVER_DIR}/formula_sub_zero.launch.py"
DRIVER_KIND=mpc
NEEDS_DEPTH=true
FSZ_PYTHON="${FORMULA_SUB_ZERO_PYTHON:-${DRIVER_DIR}/.venv/bin/python}"
if [[ ! -x "${FSZ_PYTHON}" ]]; then
  echo "error: CasADi environment missing; run ${DRIVER_DIR}/setup.sh on the Orin" >&2
  exit 1
fi
# shellcheck source=_formula_launch.sh
source "${SCRIPT_DIR}/_formula_launch.sh"
formula_launch "$@"
