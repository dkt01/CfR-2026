#!/usr/bin/env bash
# Native Gazebo/RViz validation using the installed ROS 2 Jazzy workspace.
# --policy runs/f3_v1/policy.npz --check [300] writes an independent JSON verdict.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$(cd ../.. && pwd)

POLICY="$PWD/runs/f3_v1/policy.npz"
LAPS=""
GUI=false; RVIZ=true; DRIVER=policy; LOOPBACK=false; SPEED=1.0; WEB=true
MANUAL_START=false
FALLBACK=stop
CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)  POLICY="$(realpath "$2")"; shift 2;;
    --baseline) DRIVER=baseline; shift;;
    --speed-scale) SPEED="$2"; shift 2;;
    --loopback) LOOPBACK=true; WEB=false; shift;;
    --laps)    LAPS="$2"; shift 2;;
    --gui)     GUI=true; shift;;
    --no-rviz) RVIZ=false; shift;;
    --web)     WEB=true; shift;;
    --no-web)  WEB=false; shift;;
    --depth-fallback) FALLBACK="$2"; shift 2;;
    --manual-start) MANUAL_START=true; shift;;
    --sensors) shift;;  # always on; accepted for formulaOne muscle memory
    --no-sensors)
      echo "formulaThree drives on the rendered ZED's depth: without sensors:=true"
      echo "there is no depth image and the car will not move.  Not supported."
      exit 1;;
    --check)
      CHECK=300; RVIZ=false; WEB=false; shift
      if [[ "${1:-}" =~ ^[0-9]+$ ]]; then CHECK="$1"; shift; fi;;
    --help|-h) echo "Usage: $0 [--policy FILE|--baseline] [--check [SECONDS]] [--laps N] [--no-rviz] [--no-web] [--loopback]"; exit 0;;
    *) echo "unknown argument: $1"; exit 1;;
  esac
done
[[ "$DRIVER" == "baseline" || -f "$POLICY" ]] || \
  { echo "no policy at $POLICY -- run ./train.sh, or pass --baseline"; exit 1; }
# A policy drives with the config it was trained under: the run directory's,
# when there is one beside it.
CONFIG="$PWD/config.yaml"
if [[ "$DRIVER" == policy ]]; then
  CONFIG="$(dirname "$POLICY")/config.yaml"
  [[ -f "$CONFIG" ]] || { echo "missing checkpoint config: $CONFIG"; exit 1; }
fi
if [[ -z "$LAPS" ]]; then
  LAPS=$(python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["env"]["laps"])' "$CONFIG")
fi

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-83}
export GZ_PARTITION=${GZ_PARTITION:-formula_three}
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  GZ_PARTITION=$GZ_PARTITION"
echo "policy $POLICY"
echo "config $CONFIG"

WS=${ROS2_WS:-$REPO}
[[ -f "$WS/install/setup.bash" ]] || { echo "build the workspace first: colcon build"; exit 1; }
set +u
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
set -u

# The install tree is what ros2 launch reads; a stale one accepts launch
# arguments and ignores them.
LAUNCH_SHARE="$WS/install/cfr_arduino_bridge/share/cfr_arduino_bridge/launch"
if [[ "$LOOPBACK" == "false" ]]; then
  if ! grep -q cmd_vel_to_drive "$LAUNCH_SHARE/speed_course.launch.py" 2>/dev/null || \
     ! cmp -s "$REPO/jetson/cfr_arduino_bridge/launch/sensors_world.py" "$LAUNCH_SHARE/sensors_world.py"; then
    echo "The installed launch files are older than the source."
    echo "Rebuild first:  (cd $REPO && colcon build --packages-select cfr_arduino_bridge)"
    exit 1
  fi
fi

# Every child leads its own process group and cleanup signals the GROUP:
# `ros2 launch` forks Gazebo, the bridge, sim_vehicle, lap_counter and the
# randomizer, and orphans keep driving the same Gazebo silently.
PIDS=()
spawn() { setsid "$@" & PIDS+=($!); }
cleanup() {
  echo
  echo "stopping..."
  for pid in "${PIDS[@]:-}"; do kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true; done
  sleep 3
  for pid in "${PIDS[@]:-}"; do kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true; done
  sleep 1
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --no-daemon: the ros2 daemon keeps reporting nodes after they have died.
# A run that just finished can still be shutting down (its cleanup sends
# SIGINT, waits 3 s, then SIGKILL), so give a stale stack a few seconds to
# vanish before refusing -- back-to-back runs otherwise race it.
running=true
for _ in 1 2 3; do
  if timeout 12 ros2 node list --no-daemon 2>/dev/null | grep -q -E 'sim_vehicle|ros_loopback'; then
    sleep 5
  else
    running=false; break
  fi
done
if [[ "$running" == "true" ]]; then
  echo "A simulation is already running on ROS_DOMAIN_ID=$ROS_DOMAIN_ID."
  echo "Stop it first, or use a different domain:  ROS_DOMAIN_ID=84 $0 ..."
  exit 1
fi

pose_publishers() {
  timeout 10 ros2 topic info /zed/zed_node/pose 2>/dev/null | sed -n 's/^Publisher count: *//p' || true
}
# TWO GAZEBO SERVERS ON ONE DOMAIN put two cars on the one pose topic.  Seen
# 2026-09-26: an orphaned server's upside-down car interleaved with the live
# one, the driver latched its start anchor on the wrong car, and the verdict
# was garbage.  Nothing may publish the pose before we start anything.
pubs=$(pose_publishers)
if [[ -n "$pubs" && "$pubs" != 0 ]]; then
  echo "Something already publishes /zed/zed_node/pose on ROS_DOMAIN_ID=$ROS_DOMAIN_ID ($pubs publisher(s))."
  echo "An orphaned simulator?  Check:  pgrep -af 'gz sim'  -- or use another domain."
  exit 1
fi

SIM_LOG=/tmp/formula_three_sim.log
if [[ "$LOOPBACK" == "true" ]]; then
  echo "starting the loopback car with a rendered depth camera (no Gazebo)..."
  # The loopback publishes /start_signal_detector/go itself, go_after seconds
  # after it starts -- its stand-in for the signal turning green.  Late
  # enough that the driver is up and waiting for it.
  spawn python3 ros_loopback.py --ros-args -p config:="$CONFIG" -p go_after:=25.0 >"$SIM_LOG" 2>&1
  SIM_TIME=false
else
  echo "starting the Speed Course (laps=$LAPS, rendered ZED on)..."
  # path_follower and cmd_vel_to_drive BOTH off: each publishes DriveCommand
  # on a timer, and sim_vehicle acts on whichever arrived last.
  spawn ros2 launch cfr_arduino_bridge speed_course.launch.py \
       path_follower:=false cmd_vel_to_drive:=false \
       laps:="$LAPS" gui:="$GUI" sensors:=true websocket:="$WEB" \
       >"$SIM_LOG" 2>&1
  SIM_TIME=true
fi

if [[ "$WEB" == "true" ]]; then
  echo -n "waiting for the viewer websocket on :9002"
  bound=false
  for _ in $(seq 40); do
    if (exec 3<>/dev/tcp/127.0.0.1/9002) 2>/dev/null; then
      exec 3<&- 3>&-; bound=true; echo " ok"; break
    fi
    echo -n "."; sleep 1
  done
  if [[ "$bound" != "true" ]]; then
    echo
    echo "  WARNING: nothing bound :9002 -- see $SIM_LOG (needs ros-jazzy-gz-launch-vendor)."
  else
    echo "  browser viewer: http://localhost:5173   (npm run dev in web/gzweb-viewer)"
  fi
fi

wait_topic() {  # topic, label, seconds
  echo -n "waiting for $2"
  for _ in $(seq "$3"); do
    if timeout 5 ros2 topic echo "$1" --once --field header >/dev/null 2>&1; then echo " ok"; return 0; fi
    echo -n "."; sleep 1
  done
  echo
  echo "  no $2 on $1 after $3 s -- see $SIM_LOG"
  return 1
}
wait_topic /zed/zed_node/pose "the pose stream" 60 || exit 1
pubs=$(pose_publishers)
if [[ "$pubs" != 1 ]]; then
  echo "/zed/zed_node/pose has ${pubs:-?} publishers, not 1 -- two simulators on one domain.  Aborting."
  exit 1
fi
if [[ "$DRIVER" == "policy" ]]; then
  # The rendered camera takes longer to come up than the pose, and the driver
  # will (correctly) refuse to move until it does.  Say so here rather than
  # leave a car sitting on the line looking like a policy that will not go.
  wait_topic /zed/zed_node/depth/depth_registered "the depth image" 90 || exit 1
fi

MONITOR_OUT=/tmp/formula_three_monitor.json
rm -f "$MONITOR_OUT"
spawn python3 run_monitor.py --out "$MONITOR_OUT" --config "$CONFIG" \
      --ros-args -p use_sim_time:="$SIM_TIME" >/tmp/formula_three_monitor.log 2>&1

DRIVER_LOG=/tmp/formula_three_driver.log
LAUNCH_ARGS=(policy:="$POLICY" config:="$CONFIG" driver:="$DRIVER" laps:="$LAPS"
             rviz:="$RVIZ" speed_scale:="$SPEED" use_sim_time:="$SIM_TIME"
             depth_fallback:="$FALLBACK")
if [[ "$CHECK" != "0" ]]; then
  spawn ros2 launch "$PWD/formula_three.launch.py" "${LAUNCH_ARGS[@]}" >"$DRIVER_LOG" 2>&1
else
  # To the terminal, not through `| tee`: a pipeline would run spawn in a
  # subshell, its PID would never reach PIDS, and cleanup would orphan the
  # driver -- the exact trap the process-group handling above exists for.
  spawn ros2 launch "$PWD/formula_three.launch.py" "${LAUNCH_ARGS[@]}"
fi
sleep 6

wait_for_go() {  # seconds
  # Wait on the DRIVER having received go -- it logs "GREEN -- anchoring and
  # going" from its /start_signal_detector/go callback and from nothing else.
  # That is what actually releases the car, and it sidesteps `ros2 topic echo`,
  # which on a latched topic here reported "could not determine the type" or
  # never returned, even with the type and transient_local given.
  echo -n "waiting for the driver to see the start signal"
  for _ in $(seq "$1"); do
    if grep -q "GREEN -- anchoring and going" "$DRIVER_LOG" 2>/dev/null; then
      echo " GREEN"; return 0
    fi
    echo -n "."; sleep 1
  done
  echo
  return 1
}

if [[ "$MANUAL_START" == "true" ]]; then
  # The bypass: release through the node's service, no signal involved.
  echo -n "releasing the car by service (--manual-start, no start signal)"
  released=false
  for _ in $(seq 30); do
    if timeout 10 ros2 service call /formula_one/manual_start std_srvs/srv/SetBool \
         "{data: true}" 2>/dev/null | grep -q "success=True"; then
      released=true; echo " ok"; break
    fi
    echo -n "."; sleep 1
  done
  [[ "$released" == "true" ]] || { echo; echo "could not reach /formula_one/manual_start -- the driver never came up (see $DRIVER_LOG)."; exit 1; }
elif [[ "$LOOPBACK" == "true" ]]; then
  wait_for_go 60 || { echo "the loopback never published go (see $SIM_LOG)"; exit 1; }
elif [[ "$CHECK" != "0" ]]; then
  # Unattended: turn the signal green ourselves, through the same service the
  # web viewer's button uses.  The arm sweeps, the rendered camera sees it,
  # the detector latches go -- the whole chain the car relies on.
  echo -n "turning the start signal green"
  for _ in $(seq 30); do
    if timeout 10 ros2 service call /obstacle_randomizer/start_signal std_srvs/srv/SetBool \
         "{data: true}" 2>/dev/null | grep -q "success=True"; then
      echo " ok"; break
    fi
    echo -n "."; sleep 1
  done
  wait_for_go 60 || {
    echo "the signal was asked to go green but /start_signal_detector/go never latched."
    echo "Check the detector sees the rendered signal:  ros2 topic echo /start_signal_detector/state"
    exit 1
  }
else
  echo
  echo "The car is waiting for the START SIGNAL.  Turn it green with the web viewer's"
  echo "'Set signal: Go', or:"
  echo "  ROS_DOMAIN_ID=$ROS_DOMAIN_ID ros2 service call /obstacle_randomizer/start_signal std_srvs/srv/SetBool \"{data: true}\""
fi

verdict() {
  python3 validation.py "$MONITOR_OUT" "$DRIVER_LOG" "$LAPS" "$CONFIG"
}

if [[ "$CHECK" != "0" ]]; then
  echo
  echo "running to a verdict (up to ${CHECK}s).  driver log: $DRIVER_LOG"
  deadline=$((SECONDS + CHECK))
  while (( SECONDS < deadline )); do
    grep -q "STOPPED" "$DRIVER_LOG" 2>/dev/null && break
    sleep 2
  done
  sleep 2  # one more monitor write
  echo
  if verdict; then
    echo
    where=$([[ "$LOOPBACK" == "true" ]] && echo "on the loopback car" || echo "in Gazebo")
    echo "PASS -- $LAPS laps and a stop $where, clearance margin held, speed limits met, depth never lost."
    exit 0
  fi
  echo
  echo "FAIL.  Where it got to:"
  grep -E "station|DEPTH|depth late|STOPPED" "$DRIVER_LOG" | tail -8 | sed 's/^/  /' || true
  exit 1
fi

echo
echo "running.  Ctrl-C to stop.   simulator log: $SIM_LOG   monitor: $MONITOR_OUT"
wait
