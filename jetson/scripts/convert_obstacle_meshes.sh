#!/usr/bin/env bash
#
# Regenerate cfr_arduino_bridge/meshes/*.stl from the obstacle CAD.
#
# Only needed when the CAD changes -- the STLs are committed, so building and
# running the simulation does not need Docker or any CAD tooling.  The
# conversion itself does: OpenCASCADE bindings that pip-install cleanly want a
# Python neither the Windows dev host nor the Orin necessarily has, so it runs
# in the small image built from step_to_stl.Dockerfile.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JETSON_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE="cfr-step-to-stl"

CAD_FILE="${1:-}"
if [[ -z "$CAD_FILE" || "$CAD_FILE" == "-h" || "$CAD_FILE" == "--help" ]]; then
    cat <<EOF
Usage: $(basename "$0") <All Obstacles.step>

Tessellates the assembled obstacle CAD into the STL visuals the obstacle
course world references, writing them to
  ${JETSON_DIR}/cfr_arduino_bridge/meshes/

Pass the *assembled* export, not the per-part directory: the individual STEP
files carry no assembly relationships, so the pieces of a car wash or a start
signal cannot be put back together from them.
EOF
    exit 1
fi

if [[ ! -f "$CAD_FILE" ]]; then
    echo "error: no such file: $CAD_FILE" >&2
    exit 1
fi

CAD_DIR="$(cd "$(dirname "$CAD_FILE")" && pwd)"
CAD_NAME="$(basename "$CAD_FILE")"

echo "Building $IMAGE ..."
docker build -q -f "$SCRIPT_DIR/step_to_stl.Dockerfile" -t "$IMAGE" "$SCRIPT_DIR" >/dev/null

echo "Converting $CAD_NAME ..."
# MSYS_NO_PATHCONV keeps Git Bash on Windows from rewriting the container-side
# paths into Windows ones before docker ever sees them.
MSYS_NO_PATHCONV=1 docker run --rm \
    -v "${CAD_DIR}:/cad:ro" \
    -v "${JETSON_DIR}:/jetson" \
    "$IMAGE" /jetson/scripts/step_to_stl.py \
    "/cad/${CAD_NAME}" /jetson/cfr_arduino_bridge/meshes
