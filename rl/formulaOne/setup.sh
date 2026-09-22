#!/usr/bin/env bash
# One-time training environment.  Nothing here is needed on the car.
#
# The venv is deliberately plain rather than --system-site-packages: training
# has no business importing rclpy, and the deployment path (numpy + whatever
# ROS already ships) stays provably free of torch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=${VENV:-.venv}
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel >/dev/null
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch
"$VENV/bin/pip" install -r requirements.txt

echo
"$VENV/bin/python" selftest.py
echo
echo "ready.  next:  ./train.sh --dir runs/v1"
