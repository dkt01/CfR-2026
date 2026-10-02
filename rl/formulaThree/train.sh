#!/usr/bin/env bash
# Self-test, train, export and score in one go, so the artifact that gets
# validated is always the artifact that was just trained.
#
#   ./train.sh --dir runs/f3_v1
#   ./train.sh --dir runs/f3_v2 --resume runs/f3_v1/best_model.zip
#
# VENV may point to the existing formulaOne training environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -z "${VENV:-}" ]]; then
  VENV=.venv
  [[ -x "$VENV/bin/python" ]] || VENV=../formulaOne/.venv
fi
[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV (run ./setup.sh)"; exit 1; }
PY="$VENV/bin/python"

DIR=runs/f3_v1
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) DIR="$2"; ARGS+=(--dir "$2"); shift 2;;
    --dir=*) DIR="${1#--dir=}"; ARGS+=("$1"); shift;;
    --help|-h) exec "$PY" train.py --help;;
    *) ARGS+=("$1"); shift;;
  esac
done

"$PY" selftest.py
"$PY" reward_probe.py
"$PY" train.py "${ARGS[@]}"
"$PY" export_policy.py "$DIR/best_model.zip" -o "$DIR/policy.npz"
"$PY" evaluate.py "$DIR/policy.npz" --plot "$DIR/report.png"
