#!/usr/bin/env bash
# Train, export and score in one go, so the artefact that gets validated is
# always the artefact that was just trained.
#
#   ./train.sh --dir runs/v1
#   ./train.sh --dir runs/v2 --resume runs/v1/best_model.zip
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=${VENV:-.venv}
[[ -x "$VENV/bin/python" ]] || { echo "run ./setup.sh first"; exit 1; }
PY="$VENV/bin/python"

DIR=runs/v1
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

echo
echo "policy:  $DIR/policy.npz"
echo "report:  $DIR/report.png"
echo "drive it in Gazebo:  ./validate.sh --policy $DIR/policy.npz"
