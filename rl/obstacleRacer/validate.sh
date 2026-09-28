#!/usr/bin/env bash
# The obstacle racer in Gazebo: bring up the course, the real segmenter and
# the real node, and run gazebo_check.py against them.
#
#   ./validate.sh                                  # validate runs/v1/policy.npz
#   ./validate.sh --policy runs/v2/policy.npz --starts 5 --timeout 150
#   ./validate.sh --prior                          # the steering prior alone
#   ./validate.sh --seg-gap [--poses 150]          # sensor model vs segmenter
#   ./validate.sh --surfaces                       # plant vs Gazebo's body motion
#   ./validate.sh --flat-surfaces                  # steering on open floor
#   ./validate.sh --helix-surfaces                 # short passes on the helix
#   ./validate.sh --cadence                        # raw and processed cloud timing
#
# Run inside the sim container (sim-launch skill) with the workspace built:
#   docker exec -it <container> bash -lc 'cd /repo/rl/obstacleRacer && ./validate.sh'
#
# The ZED is always rendered: the policy sees nothing else.  path_follower and
# cmd_vel_to_drive are both off -- each publishes DriveCommand on a timer, and
# sim_vehicle_node acts on whichever command arrived last.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$(cd ../.. && pwd)

POLICY="$PWD/runs/v1/policy.npz"
CONFIG="$PWD/config.yaml"
MODE=validate
DRIVER=policy
STARTS=5
SEEDS=""
TIMEOUT=""
POSES=""
HELIX_POSES=""
MONITOR_CLOUD=""
CADENCE_SECONDS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICY="$(realpath "$2")"; shift 2;;
    --config) CONFIG="$(realpath "$2")"; shift 2;;
    --prior) DRIVER=prior; shift;;
    --starts) STARTS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --poses) POSES="$2"; shift 2;;
    --helix-poses) HELIX_POSES="$2"; shift 2;;
    --monitor-cloud) MONITOR_CLOUD=1; shift;;
    --seconds) CADENCE_SECONDS="$2"; shift 2;;
    --cadence) MODE=cadence; shift;;
    --seg-gap) MODE=seg-gap; shift;;
    --surfaces) MODE=surfaces; shift;;
    --helix-surfaces) MODE=helix-surfaces; shift;;
    --flat-surfaces) MODE=flat-surfaces; shift;;
    *) echo "unknown argument: $1"; exit 1;;
  esac
done
if [[ "$MODE" == "validate" && "$DRIVER" == "policy" && ! -f "$POLICY" ]]; then
  echo "no policy at $POLICY -- export one: python3 export_policy.py runs/v1/best_model.zip -o runs/v1/policy.npz"
  exit 1
fi

# Its own ROS domain and gz partition: two stacks that can see each other
# interleave silently rather than failing.
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-78}
export GZ_PARTITION=${GZ_PARTITION:-obstacle_racer}
# jetson/scripts/build.sh builds into $ROS2_WS (default ~/ros2_ws), not the repo.
WS="${ROS2_WS:-$HOME/ros2_ws}"
[[ -f "$WS/install/setup.bash" ]] || { echo "build the workspace first: jetson/scripts/build.sh"; exit 1; }
set +u
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
set -u

# The checker runs the numpy course model beside the real stack.
python3 -c "import numba, scipy" 2>/dev/null || pip3 install --break-system-packages -q numba scipy

PIDS=()
spawn() { setsid "$@" & PIDS+=($!); }
cleanup() {
  for pid in "${PIDS[@]:-}"; do kill -INT -- "-$pid" 2>/dev/null || true; done
  sleep 3
  for pid in "${PIDS[@]:-}"; do kill -9 -- "-$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

if timeout 12 ros2 node list --no-daemon 2>/dev/null | grep -q sim_vehicle; then
  echo "A simulation is already running on ROS_DOMAIN_ID=$ROS_DOMAIN_ID; stop it or pick another domain."
  exit 1
fi

echo "starting the Obstacle Course (sensors on, 1 lap)..."
spawn ros2 launch cfr_arduino_bridge obstacle_course.launch.py \
  sensors:=true path_follower:=false cmd_vel_to_drive:=false laps:=1 \
  >/tmp/obstacle_racer_sim.log 2>&1

echo -n "waiting for the pose stream"
for _ in $(seq 90); do
  if ros2 topic echo /zed/zed_node/pose --once >/dev/null 2>&1; then echo " ok"; break; fi
  echo -n "."; sleep 1
done

if [[ "$MODE" == "validate" ]]; then
  spawn ros2 launch "$PWD/obstacle_racer.launch.py" policy:="$POLICY" config:="$CONFIG" driver:="$DRIVER" \
    >/tmp/obstacle_racer_driver.log 2>&1
  echo -n "waiting for the driver"
  for _ in $(seq 60); do
    if ros2 service list 2>/dev/null | grep -q /obstacle_racer/manual_start; then echo " ok"; break; fi
    echo -n "."; sleep 1
  done
  python3 gazebo_check.py validate --starts "$STARTS" ${SEEDS:+--seeds "$SEEDS"} ${TIMEOUT:+--timeout "$TIMEOUT"} ${MONITOR_CLOUD:+--monitor-cloud}
elif [[ "$MODE" == "seg-gap" ]]; then
  python3 gazebo_check.py seg-gap ${POSES:+--poses "$POSES"} ${SEEDS:+--seeds "$SEEDS"} ${HELIX_POSES:+--helix-poses "$HELIX_POSES"}
elif [[ "$MODE" == "cadence" ]]; then
  python3 gazebo_check.py cadence ${CADENCE_SECONDS:+--seconds "$CADENCE_SECONDS"}
else
  if [[ "$MODE" == "helix-surfaces" ]]; then
    python3 gazebo_check.py surfaces --helix-only
  elif [[ "$MODE" == "flat-surfaces" ]]; then
    python3 gazebo_check.py surfaces --flat-only
  else
    python3 gazebo_check.py "$MODE"
  fi
fi
