#!/usr/bin/env bash
# Gazebo continuous-motion, contact, and grazing verdict.
set -euo pipefail
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-84}"
export GZ_PARTITION="${GZ_PARTITION:-formula_sub_zero}"
export FORMULA_VALIDATE_LOG_DIR="${FORMULA_VALIDATE_LOG_DIR:-/tmp/formula_sub_zero_${ROS_DOMAIN_ID}}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${FORMULA_SUB_ZERO_PYTHON:-${here}/.venv/bin/python}"
if [[ ! -x "${python}" && -z "${FORMULA_SUB_ZERO_PYTHON:-}" ]]; then
  python=python3
fi
if ! command -v "${python}" >/dev/null 2>&1 ||
   ! "${python}" -c 'import numpy, yaml' >/dev/null 2>&1; then
  echo "FormulaSubZero Python lacks NumPy or PyYAML; run ${here}/setup.sh" >&2
  exit 1
fi
# Corridor mode uses depth and steering geometry; map_mpc still needs IPOPT.
if grep -Eq '^[[:space:]]*navigation:[[:space:]]*map_mpc' "${here}/config.yaml"; then
  if ! "${python}" - <<'PYTHON'
import casadi as ca
opti = ca.Opti()
x = opti.variable()
opti.minimize((x - 1) ** 2)
opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})
assert abs(float(opti.solve().value(x)) - 1) < 1e-6
PYTHON
  then
    echo "CasADi/IPOPT environment unusable; run ${here}/setup.sh" >&2
    exit 1
  fi
fi
if [[ $# -eq 0 ]]; then
  set -- --check 60
fi
exec "${here}/../formulaTwo/validate.sh" --mpc --continuous --speed-scale 1.0 --config "${here}/config.yaml" --python "${python}" "$@"
