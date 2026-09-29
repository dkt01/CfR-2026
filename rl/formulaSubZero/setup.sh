#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'casadi>=3.8,<4'
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
