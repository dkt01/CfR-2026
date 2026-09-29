#!/usr/bin/env bash
# Gazebo ground-truth lap, contact, and grazing verdict.
set -euo pipefail
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-84}"
export GZ_PARTITION="${GZ_PARTITION:-formula_sub_zero}"
export FORMULA_VALIDATE_LOG_DIR="${FORMULA_VALIDATE_LOG_DIR:-/tmp/formula_sub_zero_${ROS_DOMAIN_ID}}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${FORMULA_SUB_ZERO_PYTHON:-${here}/.venv/bin/python}"
if [[ ! -x "${python}" ]]; then
  python="${here}/../bale_follower/.venv/bin/python"
fi
if [[ ! -x "${python}" ]]; then
  echo "CasADi environment missing; run ${here}/setup.sh" >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  set -- --check 420
fi
exec "${here}/../formulaTwo/validate.sh" --mpc --config "${here}/config.yaml" --python "${python}" "$@"
