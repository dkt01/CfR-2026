#!/usr/bin/env bash
# Watch the formulaTwo policy drive the real Gazebo course, with RViz up.
#
#   ./validate.sh                                    # bestModel/f2_v2_40M
#   ./validate.sh --policy runs/f2_v2/policy.npz
#   ./validate.sh --check 300                        # unattended PASS/FAIL
#   ./validate.sh --gui                              # Gazebo's own window too
#   ./validate.sh --no-web                           # skip the :9002 viewer server
#   ./validate.sh --baseline                         # scripted driver, no checkpoint
#   ./validate.sh --loopback                         # no Gazebo: plumbing only
#   ./validate.sh --depth-fallback map               # race on if depth is lost
#   ./validate.sh --manual-start                     # skip the signal, release by service
#
# THE CAR WAITS FOR THE START SIGNAL.  Interactively nothing releases it but
# the signal turning green -- the web viewer's "Set signal: Go", or
#   ros2 service call /obstacle_randomizer/start_signal std_srvs/srv/SetBool "{data: true}"
# -- which the rendered camera sees and start_signal_detector latches onto
# /start_signal_detector/go.  --check turns the signal green itself through
# that same service, so the unattended run exercises the real trigger chain.
# Only --manual-start bypasses it (/formula_one/manual_start).
#
# formulaOne's validate.sh, with three differences:
#
#   * THE RENDERED ZED IS NOT OPTIONAL.  sensors:=true is what creates the
#     rgbd camera, and without its depth image the policy will not move
#     (by design -- see formula_two_node.py).  formulaOne's --check turns the
#     sensors off for speed; this one cannot, and --no-sensors is refused.
#   * PASS IS STRICTER.  The driver logs STOPPED for a run it abandoned too,
#     so PASS needs FINISHED on all three laps, no DEPTH LOST, and no contact
#     -- measured by run_monitor.py from the ground-truth pose and the real
#     chassis and tyre footprint, not from the driver's own map clearance.
#   * ITS OWN ROS DOMAIN (80) AND GZ PARTITION, so it cannot interleave with
#     a formulaOne simulation on the same machine.
#
# This is the check that matters before the car: the offline evaluator scores
# the policy against the model it trained on; this scores it against Gazebo's
# contacts, tyres and rendered depth, and the real nodes on the real topics.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$(cd ../.. && pwd)

POLICY="$PWD/bestModel/f2_v2_40M/policy.npz"
LAPS=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['env']['laps'])")
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
      echo "formulaTwo drives on the rendered ZED's depth: without sensors:=true"
      echo "there is no depth image and the car will not move.  Not supported."
      exit 1;;
    --check)   CHECK="${2:-300}"; RVIZ=false; WEB=false; shift 2;;
    *) echo "unknown argument: $1"; exit 1;;
  esac
done
[[ "$DRIVER" == "baseline" || -f "$POLICY" ]] || \
  { echo "no policy at $POLICY -- run ./train.sh, or pass --baseline"; exit 1; }
# A policy drives with the config it was trained under: the run directory's,
# when there is one beside it.
CONFIG="$PWD/config.yaml"
[[ -f "$(dirname "$POLICY")/config.yaml" ]] && CONFIG="$(dirname "$POLICY")/config.yaml"

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-80}
export GZ_PARTITION=${GZ_PARTITION:-formula_two}
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  GZ_PARTITION=$GZ_PARTITION"
echo "policy $POLICY"
echo "config $CONFIG"

[[ -f "$REPO/install/setup.bash" ]] || { echo "build the workspace first: colcon build"; exit 1; }
set +u
source /opt/ros/jazzy/setup.bash
source "$REPO/install/setup.bash"
set -u

# The install tree is what ros2 launch reads; a stale one accepts launch
# arguments and ignores them.
LAUNCH_SHARE="$REPO/install/cfr_arduino_bridge/share/cfr_arduino_bridge/launch"
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
trap cleanup EXIT INT TERM

# --no-daemon: the ros2 daemon keeps reporting nodes after they have died.
if timeout 12 ros2 node list --no-daemon 2>/dev/null | grep -q -E 'sim_vehicle|ros_loopback'; then
  echo "A simulation is already running on ROS_DOMAIN_ID=$ROS_DOMAIN_ID."
  echo "Stop it first, or use a different domain:  ROS_DOMAIN_ID=81 $0 ..."
  exit 1
fi

SIM_LOG=/tmp/formula_two_sim.log
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
if [[ "$DRIVER" == "policy" ]]; then
  # The rendered camera takes longer to come up than the pose, and the driver
  # will (correctly) refuse to move until it does.  Say so here rather than
  # leave a car sitting on the line looking like a policy that will not go.
  wait_topic /zed/zed_node/depth/depth_registered "the depth image" 90 || exit 1
fi

MONITOR_OUT=/tmp/formula_two_monitor.json
rm -f "$MONITOR_OUT"
spawn python3 run_monitor.py --out "$MONITOR_OUT" --config "$CONFIG" \
      --ros-args -p use_sim_time:="$SIM_TIME" >/tmp/formula_two_monitor.log 2>&1

DRIVER_LOG=/tmp/formula_two_driver.log
LAUNCH_ARGS=(policy:="$POLICY" config:="$CONFIG" driver:="$DRIVER" laps:="$LAPS"
             rviz:="$RVIZ" speed_scale:="$SPEED" use_sim_time:="$SIM_TIME"
             depth_fallback:="$FALLBACK")
if [[ "$CHECK" != "0" ]]; then
  spawn ros2 launch "$PWD/formula_two.launch.py" "${LAUNCH_ARGS[@]}" >"$DRIVER_LOG" 2>&1
else
  # To the terminal, not through `| tee`: a pipeline would run spawn in a
  # subshell, its PID would never reach PIDS, and cleanup would orphan the
  # driver -- the exact trap the process-group handling above exists for.
  spawn ros2 launch "$PWD/formula_two.launch.py" "${LAUNCH_ARGS[@]}"
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
    if ros2 service call /formula_one/manual_start std_srvs/srv/SetBool \
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
    if ros2 service call /obstacle_randomizer/start_signal std_srvs/srv/SetBool \
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
  python3 - "$MONITOR_OUT" "$DRIVER_LOG" "$LAPS" <<'EOF'
import json, re, sys
mon_path, log_path, laps = sys.argv[1], sys.argv[2], int(sys.argv[3])
log = open(log_path, errors="replace").read()
try:
    m = json.load(open(mon_path))
except Exception:
    m = {}
finished = re.search(rf"FINISHED {laps} laps in ([\d.]+) s", log)
stopped = "STOPPED" in log
lost = "DEPTH LOST" in log or "ABANDONED" in log
lap_times = re.findall(r"lap (\d+) of \d+\s+([\d.]+) s", log)
clr = m.get("min_clearance")
contact = m.get("contact_samples", 0)
print(f"  laps           {' / '.join(t for _, t in lap_times) or 'none'}")
print(f"  race           {finished.group(1) + ' s' if finished else 'not finished'}")
print(f"  min clearance  {clr:+.3f} m at station {m.get('min_clearance_station', 0):.1f} m (chassis + tyres, ground truth)" if clr is not None else "  min clearance  no pose samples")
print(f"  grazing        {m.get('grazing_samples', 0)} of {m.get('samples', 0)} pose samples inside the graze band")
print(f"  depth          {m.get('depth_holds', 0)} hold(s), {'LOST' if lost else 'never lost'}")
ct, lt = m.get("first_contact_time"), m.get("depth_lost_time")
if ct is not None:
    order = ("contact FIRST, then depth lost -- a driving failure" if lt is not None and ct <= lt
             else "depth lost first, then contact" if lt is not None else "contact without depth loss")
    print(f"  contact        first at station {m.get('first_contact_station', 0):.1f} m: {order}")
if m.get("max_abs_roll_deg", 0) > 30:
    print(f"  ROLLED OVER    max |roll| {m['max_abs_roll_deg']:.0f} deg -- tripped by a bale")
ok = bool(finished) and stopped and not lost and clr is not None and contact == 0
sys.exit(0 if ok else 1)
EOF
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
    echo "PASS -- $LAPS laps and a stop $where, no contact, depth never lost."
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
