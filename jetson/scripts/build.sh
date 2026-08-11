#!/usr/bin/env bash
#
# Build the nodes on the Orin.
#
# Works from wherever the sources live (~/software after a sync, or the repo's
# jetson/ directory), since the source location is resolved from this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"

BUILD_TYPE=Release
RUN_TESTS=false
CLEAN=false
PACKAGES=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [package ...]

Builds the packages in ${SOURCE_DIR} into ${ROS2_WS}.

Options:
  -t, --test     Run the test suites after building
  -c, --clean    Remove this package's build/install artifacts first
  -g, --debug    Build with -DCMAKE_BUILD_TYPE=Debug instead of Release
  -h, --help     This message

With no package arguments, everything under the source directory is built.

Examples:
  $(basename "$0")
  $(basename "$0") --test
  $(basename "$0") --clean cfr_arduino_bridge
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t | --test)
            RUN_TESTS=true
            shift
            ;;
        -c | --clean)
            CLEAN=true
            shift
            ;;
        -g | --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
        *)
            PACKAGES+=("$1")
            shift
            ;;
    esac
done

if [[ ! -d "$SOURCE_DIR/cfr_arduino_bridge" ]]; then
    echo "error: $SOURCE_DIR does not contain the jetson packages" >&2
    exit 1
fi

if [[ ! -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ]]; then
    echo "error: ROS 2 $ROS_DISTRO_NAME not found at /opt/ros/$ROS_DISTRO_NAME" >&2
    exit 1
fi

# ROS's setup.bash references unset variables internally, so nounset has to
# be relaxed just for the sourcing.
set +u
# shellcheck disable=SC1090
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
set -u

mkdir -p "$ROS2_WS"
cd "$ROS2_WS"

if [[ "$CLEAN" == true ]]; then
    for package in cfr_interfaces cfr_arduino_bridge; do
        echo "removing build/$package and install/$package"
        rm -rf "build/$package" "install/$package"
    done
fi

build_args=(--base-paths "$SOURCE_DIR" --cmake-args "-DCMAKE_BUILD_TYPE=$BUILD_TYPE")
test_args=(--base-paths "$SOURCE_DIR")
if [[ ${#PACKAGES[@]} -gt 0 ]]; then
    build_args+=(--packages-up-to "${PACKAGES[@]}")
    test_args+=(--packages-select "${PACKAGES[@]}")
fi

echo "building $SOURCE_DIR ($BUILD_TYPE) into $ROS2_WS"
colcon build "${build_args[@]}"

if [[ "$RUN_TESTS" == true ]]; then
    echo
    echo "running tests"
    # colcon test exits non-zero on failure; keep going so test-result can
    # print which case failed before this script exits.
    test_status=0
    colcon test "${test_args[@]}" || test_status=$?
    colcon test-result --verbose || true
    if [[ $test_status -ne 0 ]]; then
        echo "tests FAILED" >&2
        exit $test_status
    fi
    echo "tests passed"
fi

echo
echo "build complete.  to run:"
echo "  $SCRIPT_DIR/launch.sh"
