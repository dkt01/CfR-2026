#!/usr/bin/env bash
# Start the Run Lab: pull runs off the Orin, analyse them, replay them.
#
#   web/run-lab/run.sh              # http://localhost:8765
#   web/run-lab/run.sh --port 9000
#   ORIN_HOST=tejam@orin.local web/run-lab/run.sh     # over Wi-Fi instead of USB-C
#   CFR_RUNS_LOCAL=/data/runs web/run-lab/run.sh      # runs somewhere other than <repo>/runs
#
# First run sets up its own venv (.venv, not the ROS Python) and builds the
# UI; later runs start in a second.  Replays in Gazebo / RViz additionally
# need ROS 2 Jazzy and a built workspace (`colcon build` at the repo root) --
# everything else works without ROS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PORT=8765
HOST=127.0.0.1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --lan)  HOST=0.0.0.0; shift;;   # reachable from other machines: no auth, so trusted networks only
    --rebuild) rm -rf frontend/dist; shift;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done

# The ROS environment puts its own site-packages on PYTHONPATH; the Run Lab
# must not import them (it reads bags with `rosbags`, not rclpy).
unset PYTHONPATH

if [[ ! -x .venv/bin/python ]]; then
  echo "setting up the Python environment (once)..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

if [[ ! -f frontend/dist/index.html ]]; then
  command -v npm >/dev/null || { echo "npm is needed once to build the UI"; exit 1; }
  echo "building the UI (once)..."
  (cd frontend && npm ci --no-audit --no-fund && npx vite build)
fi

echo "Run Lab on http://localhost:${PORT}/"
cd server
exec ../.venv/bin/python -m uvicorn app:app --host "$HOST" --port "$PORT" --log-level warning
