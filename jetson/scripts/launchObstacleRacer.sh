#!/usr/bin/env bash
#
# Launch the obstacleRacer driver on the car (Obstacle Course, one lap), at a
# third of full speed by default.
#
#   ~/software/scripts/launch.sh --no-cmd-vel      # terminal 1: bridge + ZED
#   ~/software/scripts/launchObstacleRacer.sh      # terminal 2: this
#   ~/software/scripts/launchObstacleRacer.sh -s 0.5 --laps 2
#   ~/software/scripts/launchObstacleRacer.sh --prior --prior-speed 0.8
#   ~/software/scripts/launchObstacleRacer.sh -n   # print the command only
#
# Checks the bridge is up, nothing else publishes /drive_cmd, and the pose and
# the ZED point cloud are live (the racer steers from the cloud and holds zero
# speed without it), then asks for GO with the E-stop in hand.  See
# rl/obstacleRacer/obstacle_racer_car.launch.py.
#
# The driver ARMS THE ACTUATORS.  Nothing here runs it without a person
# confirming, at the car, that the E-stop is in their hand -- unless they
# pass --yes, which is for them to decide, not a script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ~/software/scripts -> ~/software/obstacleRacer, as syncSoftware.sh lays it out.
DRIVER_DIR="${OBSTACLE_RACER_DIR:-$(dirname "${SCRIPT_DIR}")/obstacleRacer}"
LAUNCH_FILE="${DRIVER_DIR}/obstacle_racer_car.launch.py"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
CLOUD_TOPIC=/zed/zed_node/point_cloud/cloud_registered

policy="${DRIVER_DIR}/policy.npz"
config="${DRIVER_DIR}/config.yaml"
laps=0 record=true record_args="" driver=policy prior_speed=""
signal_debug=false skip_checks=false yes=false dry_run=false extra=()
SPEED_SCALE="${SPEED_SCALE:-0.3}"
# syncSoftware.sh writes the run name beside the policy it copied.
LABEL=""
[[ -f "${DRIVER_DIR}/POLICY_RUN" ]] && LABEL="$(tr -d '[:space:]' <"${DRIVER_DIR}/POLICY_RUN")"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [name:=value ...]

Launches the obstacleRacer driver on the car from ${DRIVER_DIR}.
Start the bridge and the ZED first, WITHOUT cmd_vel_to_drive:
  ~/software/scripts/launch.sh --no-cmd-vel

Options:
  -s, --speed-scale X   multiply every speed command (default: ${SPEED_SCALE})
  -l, --laps N          laps before stopping (default: the launch file's, 1)
      --label NAME      recording label, prefixed or_ (default: the synced
                        run, ${LABEL:-none}; else the policy's directory)
      --policy FILE     policy .npz (default: ${DRIVER_DIR}/policy.npz)
      --config FILE     its config.yaml (default: ${DRIVER_DIR}/config.yaml)
      --prior           scripted prior driver instead of the policy
      --prior-speed V   the prior's speed, m/s (default: the launch file's, 1.0)
      --signal-debug    annotated frames on /start_signal_detector/debug_image
      --no-record       do not record the run
      --record-args "…" extra record_run.py flags, e.g. "--svo"
      --skip-checks     skip the preflight checks
  -y, --yes             do not ask for the E-stop confirmation
  -n, --dry-run         print the launch command and exit
  -h, --help            this message

Any name:=value arguments are passed through to the launch file.

Then release the car with the real start signal, or:
  ros2 service call /obstacle_racer/manual_start std_srvs/srv/SetBool "{data: true}"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s | --speed-scale) SPEED_SCALE="$2"; shift 2 ;;
    -l | --laps) laps="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --policy) policy="$(realpath "$2")"; shift 2 ;;
    --config) config="$(realpath "$2")"; shift 2 ;;
    --prior) driver=prior; shift ;;
    --prior-speed) prior_speed="$2"; shift 2 ;;
    --signal-debug) signal_debug=true; shift ;;
    --no-record) record=false; shift ;;
    --record-args) record_args="$2"; shift 2 ;;
    --skip-checks) skip_checks=true; shift ;;
    -y | --yes) yes=true; shift ;;
    -n | --dry-run) dry_run=true; shift ;;
    -h | --help) usage; exit 0 ;;
    *:=*) extra+=("$1"); shift ;;
    *) echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

# --- files
[[ -f "${LAUNCH_FILE}" ]] || { echo "error: no ${LAUNCH_FILE} -- run jetson/scripts/syncSoftware.sh" >&2; exit 1; }
[[ -f "${config}" ]] || { echo "error: no config at ${config}" >&2; exit 1; }
if [[ "${driver}" == policy && ! -f "${policy}" ]]; then
  echo "error: no policy at ${policy} -- sync one (syncSoftware.sh --racer-policy RUN), or pass --prior" >&2
  exit 1
fi

args=(
  config:="${config}" driver:="${driver}"
  speed_scale:="${SPEED_SCALE}" record:="${record}"
  signal_debug:="${signal_debug}"
)
[[ "${driver}" == policy ]] && args+=(policy:="${policy}")
[[ "${laps}" != 0 ]] && args+=(laps:="${laps}")
[[ -n "${prior_speed}" ]] && args+=(prior_speed:="${prior_speed}")
[[ -n "${LABEL}" ]] && args+=(record_label:="${LABEL}")
[[ -n "${record_args}" ]] && args+=(record_args:="${record_args}")
args+=("${extra[@]}")

if [[ "${dry_run}" == true ]]; then
  printf 'ros2 launch %s' "${LAUNCH_FILE}"
  printf ' %q' "${args[@]}"
  echo
  exit 0
fi

# ROS's setup scripts read unset variables.
set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
# shellcheck disable=SC1091
source "${ROS2_WS}/install/setup.bash" || { echo "error: no workspace at ${ROS2_WS} -- syncSoftware.sh --build" >&2; exit 1; }
set -u

if [[ "${skip_checks}" != true ]]; then
  echo "preflight..."
  fail=false
  # --no-daemon: the daemon keeps reporting nodes that have died.
  nodes="$(timeout 15 ros2 node list --no-daemon 2>/dev/null || true)"
  if ! grep -q arduino_bridge <<<"${nodes}"; then
    echo "  FAIL  arduino_bridge is not running -- start ~/software/scripts/launch.sh --no-cmd-vel"
    fail=true
  else
    echo "  ok    arduino_bridge is running"
  fi
  if grep -q -E '(formula_(one|two)|obstacle_racer)$' <<<"${nodes}"; then
    echo "  FAIL  an RL driver is already running -- one driver at a time"
    fail=true
  fi
  # A second /drive_cmd publisher (cmd_vel_to_drive republishes on a timer)
  # means the Arduino acts on whichever arrived last.  `|| true`: under
  # pipefail a failed probe would otherwise end the script here, silently,
  # halfway through the checklist.
  pubs="$(timeout 10 ros2 topic info /drive_cmd 2>/dev/null | sed -n 's/^Publisher count: *//p' || true)"
  if [[ -n "${pubs}" && "${pubs}" != 0 ]]; then
    echo "  FAIL  /drive_cmd already has ${pubs} publisher(s) -- relaunch with launch.sh --no-cmd-vel"
    fail=true
  else
    echo "  ok    nothing else publishes /drive_cmd"
  fi
  if timeout 8 ros2 topic echo /zed/zed_node/pose --once --field header >/dev/null 2>&1; then
    echo "  ok    /zed/zed_node/pose is live"
  else
    echo "  FAIL  no /zed/zed_node/pose -- is the ZED up?"
    fail=true
  fi
  # Both drivers read the cloud: the prior steers from it as the policy does.
  if timeout 8 ros2 topic echo "${CLOUD_TOPIC}" --once --field header >/dev/null 2>&1; then
    echo "  ok    ${CLOUD_TOPIC} is live"
  else
    echo "  FAIL  no ${CLOUD_TOPIC} -- the racer holds zero speed without it"
    fail=true
  fi
  if [[ "${fail}" == true ]]; then
    echo "preflight failed -- fix the above, or --skip-checks if you know better"
    exit 1
  fi
fi

echo
echo "  driver       obstacleRacer (${driver})"
[[ "${driver}" == policy ]] && echo "  policy       ${policy}${LABEL:+  (run ${LABEL})}"
[[ "${driver}" == prior ]] && echo "  prior speed  ${prior_speed:-launch default} m/s"
echo "  config       ${config}"
echo "  speed scale  ${SPEED_SCALE}"
echo "  laps         $([[ "${laps}" == 0 ]] && echo 'launch default (1)' || echo "${laps}")"
echo "  record       ${record}${LABEL:+ as or_${LABEL}}"
echo
echo "  THIS ARMS THE ACTUATORS.  The car has no brakes.  Have the E-stop IN"
echo "  YOUR HAND and the course clear."
if [[ "${yes}" != true ]]; then
  read -r -p "  Type GO to launch: " answer </dev/tty
  [[ "${answer}" == GO ]] || { echo "not launched."; exit 1; }
fi
echo
echo "waiting for the start signal once it is up; release by hand with:"
echo "  ros2 service call /obstacle_racer/manual_start std_srvs/srv/SetBool \"{data: true}\""
echo "stop without killing the node with {data: false}"
echo
exec ros2 launch "${LAUNCH_FILE}" "${args[@]}"
