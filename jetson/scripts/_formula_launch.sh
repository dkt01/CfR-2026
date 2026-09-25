# shellcheck shell=bash
#
# Shared by launchFormulaOne.sh and launchFormulaTwo.sh: launch one RL driver
# on the car, after checking the things that make a run go wrong in the pits.
# Sourced, not run.  The wrappers set DRIVER_NAME, DRIVER_DIR, LAUNCH_FILE,
# NEEDS_DEPTH, then call formula_launch "$@".
#
# The driver ARMS THE ACTUATORS.  Nothing here runs it without a person
# confirming, at the car, that the E-stop is in their hand -- unless they
# pass --yes, which is for them to decide, not a script.

formula_usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [name:=value ...]

Launches the ${DRIVER_NAME} driver on the car from ${DRIVER_DIR}.
Start the bridge and the ZED first, WITHOUT cmd_vel_to_drive:
  ~/software/scripts/launch.sh --no-cmd-vel

Options:
  -s, --speed-scale X   multiply every speed command (default: ${SPEED_SCALE})
  -l, --laps N          laps before stopping (default: the policy's config)
      --label NAME      recording label (default: the synced run, ${LABEL:-none})
      --policy FILE     policy .npz (default: ${DRIVER_DIR}/policy.npz)
      --config FILE     its config.yaml (default: ${DRIVER_DIR}/config.yaml)
      --baseline        scripted driver instead of the policy
      --no-record       do not record the run
      --record-args "…" extra record_run.py flags, e.g. "--svo"
EOF
  if [[ "${NEEDS_DEPTH}" == true ]]; then
    cat <<EOF
      --depth-fallback stop|map
                        when depth is lost: coast to rest (default) or race
                        on the map alone
EOF
  fi
  cat <<EOF
      --skip-checks     skip the preflight checks
  -y, --yes             do not ask for the E-stop confirmation
  -n, --dry-run         print the launch command and exit
  -h, --help            this message

Any name:=value arguments are passed through to the launch file.

Then release the car with the real start signal, or:
  ros2 service call /formula_one/manual_start std_srvs/srv/SetBool "{data: true}"
EOF
}

formula_launch() {
  set -euo pipefail
  ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
  local policy="${DRIVER_DIR}/policy.npz" config="${DRIVER_DIR}/config.yaml"
  local laps=0 record=auto record_args="" driver=policy fallback=stop
  local skip_checks=false yes=false dry_run=false extra=()
  SPEED_SCALE="${SPEED_SCALE:-0.3}"
  # syncSoftware.sh writes the run name beside the policy it copied.
  LABEL=""
  [[ -f "${DRIVER_DIR}/POLICY_RUN" ]] && LABEL="$(tr -d '[:space:]' <"${DRIVER_DIR}/POLICY_RUN")"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -s | --speed-scale) SPEED_SCALE="$2"; shift 2 ;;
      -l | --laps) laps="$2"; shift 2 ;;
      --label) LABEL="$2"; shift 2 ;;
      --policy) policy="$(realpath "$2")"; shift 2 ;;
      --config) config="$(realpath "$2")"; shift 2 ;;
      --baseline) driver=baseline; shift ;;
      --no-record) record=false; shift ;;
      --record-args) record_args="$2"; shift 2 ;;
      --depth-fallback) fallback="$2"; shift 2 ;;
      --skip-checks) skip_checks=true; shift ;;
      -y | --yes) yes=true; shift ;;
      -n | --dry-run) dry_run=true; shift ;;
      -h | --help) formula_usage; exit 0 ;;
      *:=*) extra+=("$1"); shift ;;
      *) echo "error: unknown option '$1'" >&2; formula_usage >&2; exit 2 ;;
    esac
  done

  # --- files
  [[ -f "${LAUNCH_FILE}" ]] || { echo "error: no ${LAUNCH_FILE} -- run jetson/scripts/syncSoftware.sh" >&2; exit 1; }
  [[ -f "${config}" ]] || { echo "error: no config at ${config}" >&2; exit 1; }
  if [[ "${driver}" == policy && ! -f "${policy}" ]]; then
    echo "error: no policy at ${policy} -- sync one, or pass --baseline" >&2
    exit 1
  fi
  if [[ "${fallback}" != stop && "${fallback}" != map ]]; then
    echo "error: --depth-fallback is stop or map, not '${fallback}'" >&2
    exit 2
  fi

  local args=(
    use_sim_time:=false rviz:=false
    config:="${config}" driver:="${driver}" laps:="${laps}"
    speed_scale:="${SPEED_SCALE}" record:="${record}"
  )
  [[ "${driver}" == policy ]] && args+=(policy:="${policy}")
  [[ -n "${LABEL}" ]] && args+=(record_label:="${LABEL}")
  [[ -n "${record_args}" ]] && args+=(record_args:="${record_args}")
  [[ "${NEEDS_DEPTH}" == true ]] && args+=(depth_fallback:="${fallback}")
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
    local fail=false nodes pubs
    # --no-daemon: the daemon keeps reporting nodes that have died.
    nodes="$(timeout 15 ros2 node list --no-daemon 2>/dev/null || true)"
    if ! grep -q arduino_bridge <<<"${nodes}"; then
      echo "  FAIL  arduino_bridge is not running -- start ~/software/scripts/launch.sh --no-cmd-vel"
      fail=true
    else
      echo "  ok    arduino_bridge is running"
    fi
    if grep -q -E 'formula_(one|two)$' <<<"${nodes}"; then
      echo "  FAIL  a formula driver is already running -- one driver at a time"
      fail=true
    fi
    # A second /drive_cmd publisher (cmd_vel_to_drive republishes on a timer)
    # means the Arduino acts on whichever arrived last: DEPLOY.md, section 5.
    # `|| true`: under pipefail a failed probe would otherwise end the script
    # here, silently, halfway through the checklist.
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
      echo "  FAIL  no /zed/zed_node/pose -- is the ZED up with its race config?"
      fail=true
    fi
    if [[ "${NEEDS_DEPTH}" == true && "${driver}" == policy ]]; then
      if timeout 8 ros2 topic echo /zed/zed_node/depth/depth_registered --once --field header >/dev/null 2>&1; then
        echo "  ok    /zed/zed_node/depth/depth_registered is live"
      else
        echo "  FAIL  no depth image -- the formulaTwo policy will not move without it"
        fail=true
      fi
      # Not fatal: without the IMU the scan is levelled on the pose, which the
      # ZED flattens in two_d_mode, so a rolling car loses beams.
      if timeout 5 ros2 topic echo /zed/zed_node/imu/data --once --field header >/dev/null 2>&1; then
        echo "  ok    /zed/zed_node/imu/data is live (the depth scan is levelled on it)"
      else
        echo "  WARN  no /zed/zed_node/imu/data -- the depth scan will not be levelled"
      fi
    fi
    if [[ "${fail}" == true ]]; then
      echo "preflight failed -- fix the above, or --skip-checks if you know better"
      exit 1
    fi
  fi

  echo
  echo "  driver       ${DRIVER_NAME} (${driver})"
  [[ "${driver}" == policy ]] && echo "  policy       ${policy}${LABEL:+  (run ${LABEL})}"
  echo "  config       ${config}"
  echo "  speed scale  ${SPEED_SCALE}"
  echo "  laps         $([[ "${laps}" == 0 ]] && echo 'from config' || echo "${laps}")"
  echo "  record       ${record}${LABEL:+ as ${LABEL}}"
  [[ "${NEEDS_DEPTH}" == true ]] && echo "  depth lost   ${fallback}"
  echo
  echo "  THIS ARMS THE ACTUATORS.  The car has no brakes: from 5.2 m/s it coasts"
  echo "  ~15 m.  Have the E-stop IN YOUR HAND and the run-off clear."
  if [[ "${yes}" != true ]]; then
    local answer
    read -r -p "  Type GO to launch: " answer </dev/tty
    [[ "${answer}" == GO ]] || { echo "not launched."; exit 1; }
  fi
  echo
  echo "waiting for the start signal once it is up; release by hand with:"
  echo "  ros2 service call /formula_one/manual_start std_srvs/srv/SetBool \"{data: true}\""
  echo
  exec ros2 launch "${LAUNCH_FILE}" "${args[@]}"
}
