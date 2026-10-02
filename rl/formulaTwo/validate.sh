#!/usr/bin/env bash
# Watch the formulaTwo policy drive the real Gazebo course, with RViz up.
#
#   ./validate.sh                                    # bestModel/f2_v3_59M
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

POLICY="$PWD/bestModel/f2_v3_59M/policy.npz"
LAPS=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['env']['laps'])")
GUI=false; RVIZ=true; DRIVER=policy; LOOPBACK=false; SPEED=1.0; WEB=true
MANUAL_START=false
CONTINUOUS=false
FALLBACK=stop
CONFIG_OVERRIDE=""
PYTHON=python3
CHECK=0
START_ALONG=0.0; START_LATERAL=0.0; START_HEADING=0.0; WORLD_SCALE=0.0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)  POLICY="$(realpath "$2")"; shift 2;;
    --baseline) DRIVER=baseline; shift;;
    --mpc) DRIVER=mpc; shift;;
    --config) CONFIG_OVERRIDE="$(realpath "$2")"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    --speed-scale) SPEED="$2"; shift 2;;
    --loopback) LOOPBACK=true; WEB=false; shift;;
    --start-along) START_ALONG="$2"; shift 2;;
    --start-lateral) START_LATERAL="$2"; shift 2;;
    --start-heading-deg) START_HEADING="$2"; shift 2;;
    --world-scale) WORLD_SCALE="$2"; shift 2;;
    --laps)    LAPS="$2"; shift 2;;
    --gui)     GUI=true; shift;;
    --no-rviz) RVIZ=false; shift;;
    --web)     WEB=true; shift;;
    --no-web)  WEB=false; shift;;
    --depth-fallback) FALLBACK="$2"; shift 2;;
    --manual-start) MANUAL_START=true; shift;;
    --continuous) CONTINUOUS=true; shift;;
    --sensors) shift;;  # always on; accepted for formulaOne muscle memory
    --no-sensors)
      echo "formulaTwo drives on the rendered ZED's depth: without sensors:=true"
      echo "there is no depth image and the car will not move.  Not supported."
      exit 1;;
    --check)   CHECK="${2:-300}"; RVIZ=false; WEB=false; shift 2;;
    *) echo "unknown argument: $1"; exit 1;;
  esac
done
for name in START_ALONG START_LATERAL START_HEADING WORLD_SCALE; do
  if [[ "${!name}" =~ ^-?[0-9]+$ ]]; then
    printf -v "$name" '%s.0' "${!name}"
  fi
done
[[ "$LOOPBACK" == true || "$WORLD_SCALE" == 0.0 ]] || \
  { echo "--world-scale requires --loopback" >&2; exit 2; }
[[ "$DRIVER" != "policy" || -f "$POLICY" ]] || \
  { echo "no policy at $POLICY -- run ./train.sh, or pass --baseline"; exit 1; }
# A policy drives with the config it was trained under: the run directory's,
# when there is one beside it.
CONFIG="$PWD/config.yaml"
[[ "$DRIVER" == "policy" && -f "$(dirname "$POLICY")/config.yaml" ]] && CONFIG="$(dirname "$POLICY")/config.yaml"
[[ -n "$CONFIG_OVERRIDE" ]] && CONFIG="$CONFIG_OVERRIDE"

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-80}
export GZ_PARTITION=${GZ_PARTITION:-formula_two}
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  GZ_PARTITION=$GZ_PARTITION"
echo "driver $DRIVER"
if [[ "$DRIVER" == policy ]]; then echo "policy $POLICY"; fi
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
  echo "Stop it first, or use a different domain:  ROS_DOMAIN_ID=81 $0 ..."
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

LOG_DIR="${FORMULA_VALIDATE_LOG_DIR:-/tmp/formula_two_${ROS_DOMAIN_ID}}"
mkdir -p "$LOG_DIR"
SIM_LOG="$LOG_DIR/sim.log"
if [[ "$LOOPBACK" == "true" ]]; then
  echo "starting the loopback car with a rendered depth camera (no Gazebo)..."
  # The loopback publishes /start_signal_detector/go itself, go_after seconds
  # after it starts -- its stand-in for the signal turning green.  Late
  # enough that the driver is up and waiting for it.
  spawn python3 ros_loopback.py --ros-args -p config:="$CONFIG" -p go_after:=25.0 \
       -p start_along:="$START_ALONG" -p start_lateral:="$START_LATERAL" \
       -p start_heading_deg:="$START_HEADING" -p world_scale:="$WORLD_SCALE" \
       >"$SIM_LOG" 2>&1
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
SIM_PID=${PIDS[-1]}

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

startup_failure() {
  ! kill -0 "$SIM_PID" 2>/dev/null ||
    grep -Eq '\[ERROR\] \[(gazebo-1|sim_vehicle_node-[0-9]+)\]: process has died' "$SIM_LOG"
}

report_startup_failure() {
  if ! kill -0 "$SIM_PID" 2>/dev/null; then
    echo "  simulator launch exited before the sensor stream started."
  else
    echo "  Gazebo or the simulated vehicle exited before the sensor stream started."
  fi
  grep -E '\[ERROR\]|Segmentation fault|Traceback|Exception' "$SIM_LOG" | tail -8 || true
  echo "  simulator log: $SIM_LOG"
}

wait_topic() {  # topic, label, wall-clock seconds
  local deadline=$((SECONDS + $3))
  echo -n "waiting for $2"
  while (( SECONDS < deadline )); do
    if startup_failure; then echo; report_startup_failure; return 1; fi
    if timeout 3 ros2 topic echo "$1" --once --field header >/dev/null 2>&1; then
      echo " ok"; return 0
    fi
    echo -n "."
    sleep 1
  done
  echo
  echo "  no $2 on $1 after $3 s (wall clock)."
  if startup_failure; then
    report_startup_failure
  else
    echo "  simulator log: $SIM_LOG"
    tail -15 "$SIM_LOG"
  fi
  return 1
}
wait_topic /zed/zed_node/pose "the pose stream" 60 || exit 1
pubs=$(pose_publishers)
if [[ "$pubs" != 1 ]]; then
  echo "/zed/zed_node/pose has ${pubs:-?} publishers, not 1 -- two simulators on one domain.  Aborting."
  exit 1
fi
if [[ "$DRIVER" == "policy" || "$DRIVER" == "mpc" ]]; then
  # The rendered camera takes longer to come up than the pose, and the driver
  # will (correctly) refuse to move until it does.  Say so here rather than
  # leave a car sitting on the line looking like a policy that will not go.
  wait_topic /zed/zed_node/depth/depth_registered "the depth image" 90 || exit 1
fi

MONITOR_OUT="$LOG_DIR/monitor.json"
rm -f "$MONITOR_OUT"
spawn python3 run_monitor.py --out "$MONITOR_OUT" --config "$CONFIG" \
      --world-scale "$WORLD_SCALE" \
      --ros-args -p use_sim_time:="$SIM_TIME" >"$LOG_DIR/monitor.log" 2>&1

DRIVER_LOG="$LOG_DIR/driver.log"
LAUNCH_ARGS=(policy:="$POLICY" config:="$CONFIG" driver:="$DRIVER" laps:="$LAPS"
             rviz:="$RVIZ" speed_scale:="$SPEED" use_sim_time:="$SIM_TIME"
             depth_fallback:="$FALLBACK" python:="$PYTHON" record:=false
             pose_is_camera:=false)
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
  python3 - "$MONITOR_OUT" "$DRIVER_LOG" "$LAPS" "$DRIVER" "$CONTINUOUS" <<'EOF'
import csv, json, math, re, sys
from pathlib import Path
mon_path, log_path, laps, driver, continuous = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "true"
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
if m.get("pose_jumps", 0) or m.get("clock_reversals", 0):
    print(f"  CONTAMINATED   {m.get('pose_jumps', 0)} pose jumps, {m.get('clock_reversals', 0)} clock reversals -- two simulators; this run proves nothing")
if continuous:
    trace = Path(mon_path).with_suffix(".trace.csv")
    rows = list(csv.DictReader(trace.open())) if trace.exists() else []
    total = 0.0
    recent = 0.0
    recent3 = 0.0
    end = float(rows[-1]["t"]) if rows else 0.0
    for before, after in zip(rows, rows[1:]):
        step = math.hypot(float(after["x"]) - float(before["x"]),
                          float(after["y"]) - float(before["y"]))
        if step < 1.0:
            total += step
            if float(after["t"]) >= end - 10.0:
                recent += step
            if float(after["t"]) >= end - 3.0:
                recent3 += step
    print(f"  travel         {total:.1f} m total, {recent:.1f} m in last 10 s")
    ok = (total >= 10.0 and recent >= 2.0 and recent3 >= 0.5 and not lost and
          "pose is stale" not in log and
          clr is not None and contact == 0 and m.get("grazing_samples", 0) == 0 and
          not m.get("pose_jumps", 0) and not m.get("clock_reversals", 0))
    sys.exit(0 if ok else 1)
ok = (bool(finished) and stopped and "TIMED OUT" not in log and not lost
      and clr is not None and contact == 0
      and not m.get("pose_jumps", 0) and not m.get("clock_reversals", 0)
      and (driver != "mpc" or m.get("grazing_samples", 0) == 0))
sys.exit(0 if ok else 1)
EOF
}

if [[ "$CHECK" != "0" ]]; then
  echo
  echo "running to a verdict (up to ${CHECK}s).  driver log: $DRIVER_LOG"
  deadline=$((SECONDS + CHECK))
  while (( SECONDS < deadline )); do
    if [[ "$CONTINUOUS" != true ]] && grep -q "STOPPED" "$DRIVER_LOG" 2>/dev/null; then break; fi
    sleep 2
  done
  sleep 2  # one more monitor write
  echo
  if verdict; then
    echo
    where=$([[ "$LOOPBACK" == "true" ]] && echo "on the loopback car" || echo "in Gazebo")
    if [[ "$CONTINUOUS" == true ]]; then
      echo "PASS -- continuous corridor driving $where, moving at end, no contact or grazing, depth never lost."
    else
      echo "PASS -- $LAPS laps and a stop $where, no contact$([[ "$DRIVER" == mpc ]] && echo ", no grazing"), depth never lost."
    fi
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
