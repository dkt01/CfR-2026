#!/usr/bin/env bash
# Self-test, train, export and score in one go, so the artefact that gets
# validated is always the artefact that was just trained.
#
#   ./train.sh --dir runs/f2_v1
#   ./train.sh --dir runs/f2_v2 --resume runs/f2_v1/best_model.zip
#
# Uses formulaOne's venv unless VENV says otherwise: same numpy, torch and
# stable-baselines3, nothing new to install.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=${VENV:-../formulaOne/.venv}
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV (run ../formulaOne/setup.sh)"; exit 1; }
PY="$VENV/bin/python"

DIR=runs/f2_v1
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DIR="$2"; ARGS+=(--dir "$2"); shift 2;;
    *) ARGS+=("$1"); shift;;
  esac
done

"$PY" selftest.py
"$PY" train.py "${ARGS[@]}"
"$PY" export_policy.py "$DIR/best_model.zip" -o "$DIR/policy.npz"
"$PY" evaluate.py "$DIR/policy.npz" --plot "$DIR/report.png"
