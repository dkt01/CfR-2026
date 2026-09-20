#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
viewer_dir="$repo_root/web/gzweb-viewer"

if ! command -v npm >/dev/null; then
  echo "npm is required for the browser viewer" >&2
  exit 1
fi
if [[ ! -d "$viewer_dir/node_modules" ]]; then
  echo "Install viewer dependencies first: cd $viewer_dir && npm ci" >&2
  exit 1
fi

(cd "$viewer_dir" && npm run dev -- --port 5173) &
viewer_pid=$!
cleanup() {
  kill "$viewer_pid" 2>/dev/null || true
  wait "$viewer_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Viewer: http://localhost:5173/"
ros2 launch cfr_arduino_bridge left_wall_web.launch.py
