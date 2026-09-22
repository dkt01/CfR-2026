#!/usr/bin/env bash
# Watch a policy drive the real Gazebo course, with RViz up.
#
#   ./validate.sh                                   # runs/v1/policy.npz
#   ./validate.sh --policy runs/v2/policy.npz
#   ./validate.sh --gui                             # Gazebo's own window too
#   ./validate.sh --no-sensors                      # skip the rendered ZED
#   ./validate.sh --no-web                          # skip the :9002 viewer server
#
# The rendered ZED is ON by default: it is what publishes the point cloud that
# both rviz2 and the web viewer draw, and a viewer with nothing to show is a
# worse default than a slower sim.  It also means the run starts on the real
# visual signal rather than the manual service.
#   ./validate.sh --baseline                        # scripted driver, no checkpoint
#   ./validate.sh --loopback --baseline             # no Gazebo either: plumbing only
#   ./validate.sh --check 240 --policy runs/v1/policy.npz   # unattended PASS/FAIL
#
# Without --sensors the ZED is not rendered, so there is no start signal to
# see and the run is triggered through the node's ~/manual_start service.  The
# policy does not know the difference: it is the same latch either way.
#
# This is the check that matters before the car.  The offline evaluator scores
# the policy against the model it was trained on; this scores it against
# Gazebo's contacts, Gazebo's tire friction and the real nodes on the real
# topics, none of which the trainer has ever seen.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$(cd ../.. && pwd)

POLICY="$PWD/runs/v1/policy.npz"
LAPS=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['env']['laps'])")
GUI=false; SENSORS=true; RVIZ=true; DRIVER=policy; LOOPBACK=false; SPEED=1.0
# ON by default, for the same reason SENSORS is: the browser viewer connects to
# ws://<host>:9002, that server is `gz launch websocket.gzlaunch`, and
# simulation.launch.py only starts it when websocket:=true.  This script used
# not to pass it at all, so the viewer sat there saying it was not connected
# while everything it needed was running except the one process it talks to.
WEB=true
CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)  POLICY="$(realpath "$2")"; shift 2;;
    --baseline) DRIVER=baseline; shift;;
    --speed-scale) SPEED="$2"; shift 2;;
    --loopback) LOOPBACK=true; shift;;
    --laps)    LAPS="$2"; shift 2;;
    --gui)     GUI=true; shift;;
    --sensors) SENSORS=true; shift;;
    --no-sensors) SENSORS=false; shift;;
    --no-rviz) RVIZ=false; shift;;
    --web)     WEB=true; shift;;
    --no-web)  WEB=false; shift;;
    # Unattended: no window, no viewer, run to a verdict and exit non-zero if
    # the car did not get round.  This is the acceptance test -- the offline
    # evaluator scores the policy against the model it was trained on, and a
    # model can be wrong in a way that only Gazebo's contacts and tires show.
    --check)   CHECK="${2:-240}"; RVIZ=false; SENSORS=false; WEB=false; shift 2;;
    *) echo "unknown argument: $1"; exit 1;;
  esac
done
[[ "$DRIVER" == "baseline" || -f "$POLICY" ]] || \
  { echo "no policy at $POLICY -- run ./train.sh, or pass --baseline"; exit 1; }

# Its own domain by default.  Another simulator on this machine publishes the
# same topics, and two stacks that can see each other interleave silently
# rather than failing -- see the note in node_selftest.py.
# gz transport needs the same treatment as ROS: two Gazebo servers that can
# see each other publish onto one pose topic and quietly corrupt the run.
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
export GZ_PARTITION=${GZ_PARTITION:-formula_one}
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  GZ_PARTITION=$GZ_PARTITION"

# ROS's setup scripts read unset variables, so nounset has to come off while
# they are sourced and go straight back on afterwards.
[[ -f "$REPO/install/setup.bash" ]] || { echo "build the workspace first: colcon build"; exit 1; }
set +u
source /opt/ros/jazzy/setup.bash
source "$REPO/install/setup.bash"
set -u

# The install tree is what ros2 launch actually reads, and it is easy for it
# to be older than the source -- which shows up as a launch argument being
# accepted and then quietly ignored.
LAUNCH_SHARE="$REPO/install/cfr_arduino_bridge/share/cfr_arduino_bridge/launch"
if [[ "$LOOPBACK" == "false" ]] && \
   ! grep -q cmd_vel_to_drive "$LAUNCH_SHARE/speed_course.launch.py" 2>/dev/null; then
  echo "The installed launch files are older than the source."
  echo "Rebuild first:  (cd $REPO && colcon build --packages-select cfr_arduino_bridge)"
  exit 1
fi

# Every child is started with setsid so it leads its own process group, and
# cleanup signals the GROUP (kill -- -PGID), not just the leader.
#
# `ros2 launch` forks a Gazebo server, a bridge, sim_vehicle, lap_counter and
# the randomizer.  Killing the launch process alone orphans all of them, and
# they keep running: a second sim_vehicle then consumes /drive_cmd and drives
# /sim/cmd_vel alongside the first, each with its own internal speed state,
# into one Gazebo.  Nothing errors -- the car simply behaves as though the
# policy were worse than it is.  That cost a wrong conclusion once already.
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

# Refuse to start on top of a stack that is already up: a second one does not
# fail, it interleaves.
# --no-daemon: the ros2 daemon caches discovery and keeps reporting nodes for
# a while after their processes are gone, which would block a legitimate start.
if timeout 12 ros2 node list --no-daemon 2>/dev/null | grep -q sim_vehicle; then
  echo "A simulation is already running on ROS_DOMAIN_ID=$ROS_DOMAIN_ID."
  echo "Stop it first, or use a different domain:  ROS_DOMAIN_ID=78 $0 ..."
  exit 1
fi

if [[ "$LOOPBACK" == "true" ]]; then
  echo "starting the loopback car (no Gazebo)..."
  spawn python3 ros_loopback.py >/tmp/formula_one_sim.log 2>&1
  SIM_TIME=false
else
  echo "starting the Speed Course (laps=$LAPS, sensors=$SENSORS)..."
  # path_follower and cmd_vel_to_drive BOTH have to be off.  Each of them
  # publishes DriveCommand on a timer regardless of whether anything is
  # driving them, and sim_vehicle_node acts on whichever command arrived
  # last -- so leaving either running means the car is steered by a mixture
  # of the policy and a stream of neutral commands, at no obvious point of
  # failure.  It just drives badly.
  spawn ros2 launch cfr_arduino_bridge speed_course.launch.py \
       path_follower:=false cmd_vel_to_drive:=false \
       laps:="$LAPS" gui:="$GUI" sensors:="$SENSORS" websocket:="$WEB" \
       >/tmp/formula_one_sim.log 2>&1
  SIM_TIME=true
fi

# Check the port actually BOUND, not just that the argument was passed.  A
# launch argument that is accepted and then does nothing is this project's
# most reliable way to lose an hour -- see the note above about the install
# tree going stale.
if [[ "$WEB" == "true" && "$LOOPBACK" == "false" ]]; then
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
    echo "  WARNING: nothing bound :9002, so the browser viewer will say it is"
    echo "  not connected.  Check /tmp/formula_one_sim.log for gz launch errors"
    echo "  (the websocket server needs ros-jazzy-gz-launch-vendor)."
  else
    echo "  browser viewer: http://localhost:5173   (npm run dev in web/gzweb-viewer)"
  fi
fi

echo -n "waiting for the pose stream"
for _ in $(seq 60); do
  if ros2 topic echo /zed/zed_node/pose --once >/dev/null 2>&1; then echo " ok"; break; fi
  echo -n "."; sleep 1
done

DRIVER_LOG=/tmp/formula_one_driver.log
if [[ "$CHECK" != "0" ]]; then
  spawn ros2 launch "$PWD/formula_one.launch.py" \
       policy:="$POLICY" driver:="$DRIVER" laps:="$LAPS" rviz:="$RVIZ" \
       speed_scale:="$SPEED" use_sim_time:="$SIM_TIME" >"$DRIVER_LOG" 2>&1
else
  spawn ros2 launch "$PWD/formula_one.launch.py" \
       policy:="$POLICY" driver:="$DRIVER" laps:="$LAPS" rviz:="$RVIZ" \
       speed_scale:="$SPEED" use_sim_time:="$SIM_TIME"
fi
sleep 6

if [[ "$SENSORS" == "false" && "$LOOPBACK" == "false" ]]; then
  # WAIT for the service rather than sleeping at it.  A fixed sleep followed
  # by a call whose output goes to /dev/null fails silently when the driver
  # takes a second longer than expected to come up, and the only symptom is a
  # car that sits on the line for the whole run while the log says, quite
  # correctly, "waiting for the start signal".
  echo -n "releasing the car (no rendered signal without --sensors)"
  released=false
  for _ in $(seq 30); do
    if ros2 service call /formula_one/manual_start std_srvs/srv/SetBool \
         "{data: true}" 2>/dev/null | grep -q "success=True"; then
      released=true; echo " ok"; break
    fi
    echo -n "."; sleep 1
  done
  if [[ "$released" != "true" ]]; then
    echo
    echo "could not reach /formula_one/manual_start -- the driver never came up."
    exit 1
  fi
fi

if [[ "$CHECK" != "0" ]]; then
  echo
  echo "running to a verdict (up to ${CHECK}s).  driver log: $DRIVER_LOG"
  deadline=$((SECONDS + CHECK))
  while (( SECONDS < deadline )); do
    if grep -q "STOPPED" "$DRIVER_LOG" 2>/dev/null; then
      echo
      grep -E "lap [0-9]|FINISHED|STOPPED" "$DRIVER_LOG" | sed 's/^/  /'
      echo
      echo "PASS -- two laps and a stop, in Gazebo."
      exit 0
    fi
    sleep 2
  done
  echo
  grep -E "lap [0-9]|FINISHED|STOPPED" "$DRIVER_LOG" | sed 's/^/  /' || true
  echo
  echo "FAIL -- no stop within ${CHECK}s.  Where it got to:"
  grep -E "station" "$DRIVER_LOG" | tail -8 | sed 's/^/  /' || true
  exit 1
fi

echo
echo "running.  Ctrl-C to stop.   simulator log: /tmp/formula_one_sim.log"
wait
