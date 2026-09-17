#!/usr/bin/env bash
#
# Run the jetson/ colcon test suite exactly as CI does (.github/workflows/ci.yml's
# build-and-test job), in a throwaway `ros:jazzy-ros-base` container -- so this
# doesn't need to be reconstructed by hand each time (rosdep install, the
# read-only mount so build.sh can't be tempted to write into the repo, ROS2_WS
# pointed outside it, etc).
#
# Usage:
#   test.sh [--no-test] [package ...]
#
# With no packages, builds/tests everything under jetson/, same as CI.
# --no-test builds only (skips colcon test), useful as a quick compile check.

set -euo pipefail

IMAGE="ros:jazzy-ros-base"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"

RUN_TESTS=true
PACKAGES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-test) RUN_TESTS=false; shift ;;
        *) PACKAGES+=("$1"); shift ;;
    esac
done

build_flag=""
[ "$RUN_TESTS" = true ] && build_flag="--test"

log() { echo "[test.sh] $*"; }

log "running in a throwaway $IMAGE container (repo mounted read-only)"

# The repo mounts :ro because build.sh builds into ROS2_WS (default
# $HOME/ros2_ws, i.e. /root/ros2_ws in this image) -- a read-only mount is
# cheap insurance that nothing here writes build artifacts back into the
# working tree.
docker run --rm -v "$REPO_DIR:/repo:ro" "$IMAGE" bash -c "
    set -e
    apt-get update -qq
    rosdep update
    rosdep install --from-paths /repo/jetson --ignore-src --rosdistro jazzy --reinstall -r -y
    /repo/jetson/scripts/build.sh $build_flag ${PACKAGES[*]:-}
"
