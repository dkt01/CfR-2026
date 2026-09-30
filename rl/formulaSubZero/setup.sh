#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ $# -gt 1 ]]; then
  echo "usage: setup.sh [casadi-arm64.whl]" >&2
  exit 2
fi
python3 -m venv --system-site-packages .venv
if [[ $# -eq 1 ]]; then
  .venv/bin/python -m pip install --no-index --no-deps "$1"
else
  .venv/bin/python -m pip install 'casadi>=3.8.1,<4'
fi
# rclpy is supplied by ROS, whose site-packages are added by setup.bash.
set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u
.venv/bin/python - <<'PYTHON'
import casadi as ca
import rclpy
opti = ca.Opti()
x = opti.variable()
opti.minimize((x - 1) ** 2)
opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})
assert abs(float(opti.solve().value(x)) - 1) < 1e-6
print("CasADi", ca.__version__, "IPOPT and ROS Python ready")
PYTHON
