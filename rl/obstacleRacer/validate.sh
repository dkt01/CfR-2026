#!/usr/bin/env bash
# The obstacle racer in Gazebo: bring up the course, the real segmenter and
# the real node, and run gazebo_check.py against them.
#
#   ./validate.sh                                  # validate runs/v1/policy.npz
#   ./validate.sh --policy runs/v2/policy.npz --starts 5 --timeout 150
#   ./validate.sh --prior                          # the steering prior alone
#   ./validate.sh --seg-gap [--poses 150]          # sensor model vs segmenter
#   ./validate.sh --surfaces                       # plant vs Gazebo's body motion
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
MODE=validate
DRIVER=policy
STARTS=5
SEEDS=""
TIMEOUT=""
POSES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy) POLICY="$(realpath "$2")"; shift 2;;
    --prior) DRIVER=prior; shift;;
    --starts) STARTS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --poses) POSES="$2"; shift 2;;
    --seg-gap) MODE=seg-gap; shift;;
    --surfaces) MODE=surfaces; shift;;
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
  spawn ros2 launch "$PWD/obstacle_racer.launch.py" policy:="$POLICY" driver:="$DRIVER" \
    >/tmp/obstacle_racer_driver.log 2>&1
  echo -n "waiting for the driver"
  for _ in $(seq 60); do
    if ros2 service list 2>/dev/null | grep -q /obstacle_racer/manual_start; then echo " ok"; break; fi
    echo -n "."; sleep 1
  done
  python3 gazebo_check.py validate --starts "$STARTS" ${SEEDS:+--seeds "$SEEDS"} ${TIMEOUT:+--timeout "$TIMEOUT"}
elif [[ "$MODE" == "seg-gap" ]]; then
  python3 gazebo_check.py seg-gap ${POSES:+--poses "$POSES"}
else
  python3 gazebo_check.py "$MODE"
fi
