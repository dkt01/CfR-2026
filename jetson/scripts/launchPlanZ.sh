#!/usr/bin/env bash
#
# Launch Plan Z, the backup driver that is not a learned policy, on the car:
# either course, at a third of full speed by default.
#
#   ~/software/scripts/launch.sh --no-cmd-vel            # terminal 1: bridge + ZED
#   ~/software/scripts/launchPlanZ.sh --course obstacle  # terminal 2: this
#   ~/software/scripts/launchPlanZ.sh --course speed -s 0.5
#   ~/software/scripts/launchPlanZ.sh --course speed steer_trim=0.02 camera_yaw_deg=-2
#   ~/software/scripts/launchPlanZ.sh --course obstacle -n   # print the command only
#
# Checks the bridge is up, nothing else publishes /drive_cmd, and the pose and
# the ZED point cloud are live (Plan Z steers from the cloud and holds zero
# speed without it), then asks for GO with the E-stop in hand.  See
# drivers/planZ/README.md for the knobs and plan_z_car.launch.py for what is
# started.
#
# The driver ARMS THE ACTUATORS.  Nothing here runs it without a person
# confirming, at the car, that the E-stop is in their hand -- unless they
# pass --yes, which is for them to decide, not a script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ~/software/scripts -> ~/software/planZ, as syncSoftware.sh lays it out.
DRIVER_DIR="${PLAN_Z_DIR:-$(dirname "${SCRIPT_DIR}")/planZ}"
LAUNCH_FILE="${DRIVER_DIR}/plan_z_car.launch.py"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
CLOUD_TOPIC=/zed/zed_node/point_cloud/cloud_registered

course="" config="${DRIVER_DIR}/config.yaml"
laps=0 record=true record_args="" label=""
signal_debug=false skip_checks=false yes=false dry_run=false extra=() knobs=()
SPEED_SCALE="${SPEED_SCALE:-0.3}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --course speed|obstacle [options] [knob=value ...] [name:=value ...]

Launches Plan Z on the car from ${DRIVER_DIR}.
Start the bridge and the ZED first, WITHOUT cmd_vel_to_drive:
  ~/software/scripts/launch.sh --no-cmd-vel

Options:
  -c, --course NAME     speed (3 laps) or obstacle (2 laps); required
  -s, --speed-scale X   multiply every speed command (default: ${SPEED_SCALE})
  -l, --laps N          laps before stopping (default: the course's)
      --label NAME      recording label, prefixed pz_ (default: the course)
      --config FILE     knobs file (default: ${DRIVER_DIR}/config.yaml)
      --signal-debug    annotated frames on /start_signal_detector/debug_image
      --no-record       do not record the run
      --record-args "…" extra record_run.py flags, e.g. "--svo"
      --skip-checks     skip the preflight checks
  -y, --yes             do not ask for the E-stop confirmation
  -n, --dry-run         print the launch command and exit
  -h, --help            this message

knob=value sets any knob of config.yaml for this run by its name, e.g.
  steer_trim=0.02  camera_yaw_deg=-2  body_margin=0.05  sections.helical_ramp=0.8
(the table of which knob answers which failure is in drivers/planZ/README.md).
name:=value arguments are passed through to the launch file.

Then release the car with the real start signal, or:
  ros2 service call /obstacle_racer/manual_start std_srvs/srv/SetBool "{data: true}"
(/formula_one/manual_start on the Speed Course).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c | --course) course="$2"; shift 2 ;;
    -s | --speed-scale) SPEED_SCALE="$2"; shift 2 ;;
    -l | --laps) laps="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    --config) config="$(realpath "$2")"; shift 2 ;;
    --signal-debug) signal_debug=true; shift ;;
    --no-record) record=false; shift ;;
    --record-args) record_args="$2"; shift 2 ;;
    --skip-checks) skip_checks=true; shift ;;
    -y | --yes) yes=true; shift ;;
    -n | --dry-run) dry_run=true; shift ;;
    -h | --help) usage; exit 0 ;;
    *:=*) extra+=("$1"); shift ;;
    *=*) knobs+=("$1"); shift ;;
    *) echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

case "${course}" in
  speed) node=formula_one; default_laps=3 ;;
  obstacle) node=obstacle_racer; default_laps=2 ;;
  *) echo "error: --course speed|obstacle is required" >&2; usage >&2; exit 2 ;;
esac

# --- files
[[ -f "${LAUNCH_FILE}" ]] || { echo "error: no ${LAUNCH_FILE} -- run jetson/scripts/syncSoftware.sh" >&2; exit 1; }
[[ -f "${config}" ]] || { echo "error: no config at ${config}" >&2; exit 1; }
[[ -f "${DRIVER_DIR}/routes/${course}.yaml" ]] || { echo "error: no route at ${DRIVER_DIR}/routes/${course}.yaml" >&2; exit 1; }

args=(
  course:="${course}" config:="${config}"
  speed_scale:="${SPEED_SCALE}" record:="${record}"
  signal_debug:="${signal_debug}"
)
[[ "${laps}" != 0 ]] && args+=(laps:="${laps}")
[[ ${#knobs[@]} -gt 0 ]] && args+=(knobs:="${knobs[*]}")
[[ -n "${label}" ]] && args+=(record_label:="${label}")
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

# A misspelled knob would otherwise be found by the node, after GO.
python3 - "${DRIVER_DIR}" "${config}" "${knobs[@]}" <<'PY' || exit 1
import sys

sys.path.insert(0, sys.argv[1])
from route import load_config

try:
    load_config(sys.argv[2], dict(pair.split("=", 1) for pair in sys.argv[3:]))
except (KeyError, ValueError) as error:
    sys.exit(f"error: {error}")
PY

if [[ "${skip_checks}" != true ]]; then
  echo "preflight..."
  fail=false
  # The bridge by its status topic, not by `ros2 node list --no-daemon`:
  # that takes one discovery snapshot and returns before the graph is known.
  # A message on /arduino_bridge/status is proof it is up and talking to the
  # Arduino.
  if timeout 8 ros2 topic echo /arduino_bridge/status --once --field mode >/dev/null 2>&1; then
    echo "  ok    arduino_bridge is running (/arduino_bridge/status is live)"
  else
    echo "  FAIL  no /arduino_bridge/status -- start ~/software/scripts/launch.sh --no-cmd-vel"
    fail=true
  fi
  # Best effort only, for the same reason: the /drive_cmd publisher check
  # below is what actually catches a second driver.  --no-daemon because the
  # daemon keeps reporting nodes that have died.
  nodes="$(timeout 15 ros2 node list --no-daemon 2>/dev/null || true)"
  if grep -q -E '(formula_(one|two)|obstacle_racer)$' <<<"${nodes}"; then
    echo "  FAIL  a driver is already running -- one driver at a time"
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
  if timeout 8 ros2 topic echo "${CLOUD_TOPIC}" --once --field header >/dev/null 2>&1; then
    echo "  ok    ${CLOUD_TOPIC} is live"
  else
    echo "  FAIL  no ${CLOUD_TOPIC} -- Plan Z holds zero speed without it"
    fail=true
  fi
  if [[ "${fail}" == true ]]; then
    echo "preflight failed -- fix the above, or --skip-checks if you know better"
    exit 1
  fi
fi

echo
echo "  driver       Plan Z"
echo "  course       ${course}"
echo "  config       ${config}"
echo "  knobs        ${knobs[*]:-(as in the config)}"
echo "  speed scale  ${SPEED_SCALE}"
echo "  laps         $([[ "${laps}" == 0 ]] && echo "${default_laps}" || echo "${laps}")"
echo "  record       ${record} as pz_${label:-${course}}"
echo
echo "  Place the car on the start mark, square to the lane: the route is"
echo "  laid out from where it stands when it is released."
echo
echo "  THIS ARMS THE ACTUATORS.  The car has no brakes.  Have the E-stop IN"
echo "  YOUR HAND and the course clear."
if [[ "${yes}" != true ]]; then
  read -r -p "  Type GO to launch: " answer </dev/tty
  [[ "${answer}" == GO ]] || { echo "not launched."; exit 1; }
fi
echo
echo "waiting for the start signal once it is up; release by hand with:"
echo "  ros2 service call /${node}/manual_start std_srvs/srv/SetBool \"{data: true}\""
echo "stop without killing the node with {data: false}"
echo
exec ros2 launch "${LAUNCH_FILE}" "${args[@]}"
