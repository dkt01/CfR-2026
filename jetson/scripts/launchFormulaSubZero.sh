#!/usr/bin/env bash
# FormulaSubZero on the car. Start launch.sh --no-cmd-vel in terminal 1.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_NAME=FormulaSubZero
DRIVER_DIR="${FORMULA_SUB_ZERO_DIR:-$(dirname "${SCRIPT_DIR}")/formulaSubZero}"
LAUNCH_FILE="${DRIVER_DIR}/formula_sub_zero.launch.py"
DRIVER_KIND=mpc
NEEDS_DEPTH=true
FSZ_PYTHON="${FORMULA_SUB_ZERO_PYTHON:-${DRIVER_DIR}/.venv/bin/python}"
if [[ ! -x "${FSZ_PYTHON}" && -z "${FORMULA_SUB_ZERO_PYTHON:-}" ]]; then
  FSZ_PYTHON=python3
fi
if ! command -v "${FSZ_PYTHON}" >/dev/null 2>&1; then
  echo "error: FormulaSubZero Python is unavailable: ${FSZ_PYTHON}" >&2
  exit 1
fi
export FORMULA_SUB_ZERO_PYTHON="${FSZ_PYTHON}"
# shellcheck source=_formula_launch.sh
source "${SCRIPT_DIR}/_formula_launch.sh"
formula_launch "$@"
