#!/usr/bin/env bash
# Gazebo ground-truth lap, contact, and grazing verdict.
set -euo pipefail
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-84}"
export GZ_PARTITION="${GZ_PARTITION:-formula_sub_zero}"
export FORMULA_VALIDATE_LOG_DIR="${FORMULA_VALIDATE_LOG_DIR:-/tmp/formula_sub_zero_${ROS_DOMAIN_ID}}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${FORMULA_SUB_ZERO_PYTHON:-${here}/.venv/bin/python}"
if [[ ! -x "${python}" ]]; then
  echo "CasADi environment missing; run ${here}/setup.sh" >&2
  exit 1
fi
# Check the selected environment before starting Gazebo or ROS processes.
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
if [[ $# -eq 0 ]]; then
  set -- --check 420
fi
exec "${here}/../formulaTwo/validate.sh" --mpc --config "${here}/config.yaml" --python "${python}" "$@"
