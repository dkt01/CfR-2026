#!/usr/bin/env bash
#
# Turnkey path-following bench session: brings up the actuator link, ZED
# camera, and path_follower_node in the background, then hands the terminal
# to the interactive segment-builder TUI. Everything started here stops
# together when the TUI exits or the session is interrupted -- same
# fate-sharing launch.sh already uses for the bridge and ZED.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
LOG_DIR="${LOG_DIR:-/tmp/cfr_path_following}"

DEVICE="${ARDUINO_DEVICE:-/dev/ttyACM0}"
ACTION_NAME="/path_follower/drive_path"
SKIP_CHECKS=false
NO_STACK=false
FAKE_ARDUINO=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Starts the actuator link + ZED (via launch.sh) and path_follower_node in the
background, then runs path_tui.py in the foreground. Quitting the TUI (or
Ctrl-C) tears down everything this script started.

Options:
  -d, --device DEV   Arduino serial device (default: $DEVICE)
      --action NAME  DrivePath action name (default: $ACTION_NAME)
      --no-stack     Do not start the bridge/ZED; assume launch.sh is already
                      running in another terminal
      --fake-arduino No Arduino attached: pass --fake-arduino through to
                      launch.sh.  No real actuators, no real E-Stop -- see
                      launch.sh --help / fake_arduino.py --help.  ZED still
                      starts normally; path_follower_node needs its odometry
                      regardless of whether the Arduino is real.
      --skip-checks  Skip launch.sh's preflight checks (device, dialout, ModemManager)
  -h, --help         This message

Background process output goes to $LOG_DIR/*.log rather than the terminal, so
the TUI's screen stays clean.

Examples:
  $(basename "$0")
  $(basename "$0") --device /dev/ttyACM1
  $(basename "$0") --no-stack      # bridge/ZED already running elsewhere
  $(basename "$0") --fake-arduino  # no Arduino attached
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d | --device)
            DEVICE="$2"
            shift 2
            ;;
        --action)
            ACTION_NAME="$2"
            shift 2
            ;;
        --no-stack)
            NO_STACK=true
            shift
            ;;
        --fake-arduino)
            FAKE_ARDUINO=true
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
        *)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$FAKE_ARDUINO" == true && "$NO_STACK" == true ]]; then
    echo "error: --fake-arduino has no effect with --no-stack (launch.sh is never started here)" >&2
    exit 2
fi

if [[ ! -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash" ]]; then
    echo "error: ROS 2 $ROS_DISTRO_NAME not found at /opt/ros/$ROS_DISTRO_NAME" >&2
    exit 1
fi

if [[ ! -f "$ROS2_WS/install/setup.bash" ]]; then
    echo "error: $ROS2_WS is not built.  run $SCRIPT_DIR/build.sh first" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

PIDS=()

cleanup() {
    trap - EXIT INT TERM
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        kill -- "-$pid" 2>/dev/null || true
    done
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

if [[ "$NO_STACK" != true ]]; then
    echo "starting actuator link + ZED (log: $LOG_DIR/stack.log)"
    stack_args=(--device "$DEVICE")
    # Restrict the actuator converter to commands from this path follower.
    # Other nodes publishing on /cmd_vel must not arm autonomous control.
    stack_args+=("cmd_vel_topic:=/path_follower/cmd_vel")
    if [[ "$SKIP_CHECKS" == true ]]; then
        stack_args+=(--skip-checks)
    fi
    if [[ "$FAKE_ARDUINO" == true ]]; then
        stack_args+=(--fake-arduino)
    fi
    setsid "$SCRIPT_DIR/launch.sh" "${stack_args[@]}" >"$LOG_DIR/stack.log" 2>&1 &
    STACK_PID=$!
    PIDS+=("$STACK_PID")

    # launch.sh's own preflight checks fail fast (missing device, zed_wrapper
    # not built, etc.).  It runs in the background, so under set -e a quick
    # death here would otherwise go unnoticed and this script would carry on
    # starting path_follower_node and the TUI with no actuator link or
    # odometry behind them.
    sleep 2
    if ! kill -0 "$STACK_PID" 2>/dev/null; then
        echo "error: launch.sh exited immediately, see $LOG_DIR/stack.log:" >&2
        tail -n 20 "$LOG_DIR/stack.log" >&2
        exit 1
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

echo "starting path_follower_node (log: $LOG_DIR/path_follower.log)"
setsid ros2 launch cfr_arduino_bridge path_follower.launch.py \
    "cmd_vel_topic:=/path_follower/cmd_vel" >"$LOG_DIR/path_follower.log" 2>&1 &
PIDS+=("$!")

echo "launching TUI -- q or Ctrl-C there stops everything started here"
sleep 1

"$SCRIPT_DIR/path_tui.py" --action "$ACTION_NAME"
