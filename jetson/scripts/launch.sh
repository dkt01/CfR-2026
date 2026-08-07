#!/usr/bin/env bash
#
# Onboard bringup on the Orin: the Arduino bridge plus the ZED camera.
#
# The two are started together and share a fate -- if either exits, the other
# is torn down, so the car is never left with a live actuator link and a dead
# perception stack (or the reverse).
#
# Checks the things that actually go wrong in the pits -- missing device, no
# dialout membership, ModemManager holding the port -- before starting ROS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
ROSBOARD_DIR="${ROSBOARD_DIR:-$HOME/rosboard}"

DEVICE="${ARDUINO_DEVICE:-/dev/ttyACM0}"
CAMERA_MODEL="${ZED_CAMERA_MODEL:-zed2i}"
USE_CMD_VEL=true
USE_ZED=true
USE_BRIDGE=true
USE_ROSBOARD=false
USE_FAKE_ARDUINO=false
SKIP_CHECKS=false
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [name:=value ...]

Brings up the onboard stack from $ROS2_WS:
  arduino_bridge_node + cmd_vel_to_drive_node   (actuator link)
  zed_wrapper zed_camera.launch.py              (perception)

Options:
  -d, --device DEV     Arduino serial device (env ARDUINO_DEVICE, default: $DEVICE)
  -m, --camera MODEL   ZED model (env ZED_CAMERA_MODEL, default: $CAMERA_MODEL)
      --no-zed         Do not start the ZED camera node
      --no-bridge      Do not start the Arduino bridge
      --no-cmd-vel     Do not start cmd_vel_to_drive_node
      --rosboard       Also start rosboard from $ROSBOARD_DIR
      --fake-arduino   No Arduino attached: run fake_arduino.py instead and point the
                      bridge at it.  Simulates the auto-arm handshake only -- no real
                      actuators, no real E-Stop.  See fake_arduino.py --help.
      --skip-checks    Skip the preflight checks
  -h, --help           This message

Any remaining name:=value arguments are passed through to the bridge launch.

Examples:
  $(basename "$0")
  $(basename "$0") --no-zed --device /dev/ttyACM1
  $(basename "$0") --rosboard
  $(basename "$0") --fake-arduino --no-zed
  $(basename "$0") max_throttle:=0.15
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d | --device)
            DEVICE="$2"
            shift 2
            ;;
        -m | --camera)
            CAMERA_MODEL="$2"
            shift 2
            ;;
        --no-zed)
            USE_ZED=false
            shift
            ;;
        --no-bridge)
            USE_BRIDGE=false
            shift
            ;;
        --no-cmd-vel)
            USE_CMD_VEL=false
            shift
            ;;
        --rosboard)
            USE_ROSBOARD=true
            shift
            ;;
        --fake-arduino)
            USE_FAKE_ARDUINO=true
            shift
            ;;
        --skip-checks)
            SKIP_CHECKS=true
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
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$USE_BRIDGE" != true && "$USE_ZED" != true ]]; then
    echo "error: nothing to launch, --no-bridge and --no-zed are both set" >&2
    exit 2
fi

# Set up cleanup before anything gets backgrounded (fake_arduino.py, below),
# so it's torn down even if a later check exits.
PIDS=()

cleanup() {
    trap - EXIT INT TERM
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

if [[ ! -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ]]; then
    echo "error: ROS 2 $ROS_DISTRO_NAME not found at /opt/ros/$ROS_DISTRO_NAME" >&2
    exit 1
fi

if [[ ! -f "$ROS2_WS/install/setup.bash" ]]; then
    echo "error: $ROS2_WS is not built.  run $SCRIPT_DIR/build.sh first" >&2
    exit 1
fi

if [[ "$USE_BRIDGE" == true && "$USE_FAKE_ARDUINO" == true ]]; then
    echo "WARNING: --fake-arduino -- no real actuators, no real E-Stop.  bench/dev only."
    "$SCRIPT_DIR/fake_arduino.py" --link /tmp/fake_arduino &
    PIDS+=("$!")
    DEVICE=/tmp/fake_arduino
    sleep 0.5  # let it create the symlink before the bridge tries to open it
elif [[ "$SKIP_CHECKS" != true && "$USE_BRIDGE" == true ]]; then
    if [[ ! -e "$DEVICE" ]]; then
        echo "error: $DEVICE does not exist.  is the Arduino plugged in?" >&2
        echo "       available: $(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | tr '\n' ' ')" >&2
        echo "       no hardware to test with?  try --fake-arduino" >&2
        exit 1
    fi

    if [[ ! -r "$DEVICE" || ! -w "$DEVICE" ]]; then
        echo "error: no read/write access to $DEVICE" >&2
        echo "       sudo usermod -aG dialout $USER   # then log out and back in" >&2
        exit 1
    fi

    # ModemManager probes freshly enumerated ttyACM devices and corrupts the
    # first second of traffic, which looks exactly like a flaky Arduino.
    if systemctl is-active --quiet ModemManager 2>/dev/null; then
        echo "warning: ModemManager is running and will probe $DEVICE" >&2
        echo "         sudo systemctl disable --now ModemManager" >&2
    fi
fi

# ROS's setup.bash scripts reference unset variables internally, so nounset
# has to be relaxed just for the sourcing.
set +u
# shellcheck disable=SC1090
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
# shellcheck disable=SC1091
source "$ROS2_WS/install/setup.bash"
set -u

if [[ "$USE_ZED" == true ]] && ! ros2 pkg prefix zed_wrapper >/dev/null 2>&1; then
    echo "error: zed_wrapper not found in $ROS2_WS.  see ../../zed/README.md," >&2
    echo "       or pass --no-zed to run the actuator link on its own" >&2
    exit 1
fi

if [[ "$USE_ROSBOARD" == true && ! -x "$ROSBOARD_DIR/run" ]]; then
    echo "error: rosboard not found at $ROSBOARD_DIR" >&2
    exit 1
fi

# The bridge comes up first: it starts feeding the Arduino neutral commands
# immediately, so the car sits in a known state while the camera initializes.
if [[ "$USE_BRIDGE" == true ]]; then
    ros2 launch cfr_arduino_bridge arduino_bridge.launch.py \
        "device:=$DEVICE" \
        "use_cmd_vel:=$USE_CMD_VEL" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} &
    BRIDGE_PID=$!
    PIDS+=("$BRIDGE_PID")
    echo "arduino bridge (PID $BRIDGE_PID) on $DEVICE, cmd_vel translation: $USE_CMD_VEL"
fi

if [[ "$USE_ZED" == true ]]; then
    ros2 launch zed_wrapper zed_camera.launch.py "camera_model:=$CAMERA_MODEL" &
    ZED_PID=$!
    PIDS+=("$ZED_PID")
    echo "zed camera (PID $ZED_PID) model $CAMERA_MODEL"
fi

if [[ "$USE_ROSBOARD" == true ]]; then
    (cd "$ROSBOARD_DIR" && ./run) &
    ROSBOARD_PID=$!
    PIDS+=("$ROSBOARD_PID")
    echo "rosboard (PID $ROSBOARD_PID) on port 8888"
fi

echo
echo "E-STOP: keep the offboard remote in hand and the wheels clear"
echo "ctrl-c stops everything"

# Fail fast: if any one of them dies, tear the rest down rather than driving on
# with half a stack.
wait -n ${PIDS[@]+"${PIDS[@]}"} || true
echo
echo "a node exited, shutting down the rest"
