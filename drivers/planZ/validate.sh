#!/usr/bin/env bash
# Plan Z in Gazebo: bring up a course with the ZED rendered and the real
# driver node, release the car, and say how the run went (gazebo_watch.py).
#
#   ./validate.sh --course speed                    # 3 laps and a stop
#   ./validate.sh --course obstacle --seed 208      # one layout, 2 laps
#   ./validate.sh --course obstacle --laps 1 --seed 201
#   ./validate.sh --course speed --steer-bias 0.035 --camera-yaw -3
#   ./validate.sh --course speed --knobs "speed_scale=0.8 steer_trim=0.02"
#   ./validate.sh --matrix speed                    # nominal + every fault
#   ./validate.sh --matrix obstacle
#
# The faults are put into the SIMULATED CAR, with the driver's knobs left
# alone, so what is tested is the driver against a car that really is off:
#
#   --steer-bias RAD   added to every angle of sim_vehicle's steering table
#   --steer-gain K     ... and the table scaled by this
#   --camera-yaw/-pitch/-roll DEG   the simulated ZED turned on its mount
#
# and two that Gazebo cannot be given (its pose is ground truth, its course
# is the drawing) are put into what the driver is told:
#
#   --pose-drift M_PER_M --yaw-drift DEG_PER_M   the pose creeping off
#   --route-offset "X Y YAW_DEG"                 the course built off the drawing
#
# Run inside the sim container with the workspace built:
#   docker exec -it <container> bash -lc 'cd /repo/drivers/planZ && ./validate.sh --course speed'
#
# Results land in runs/ as JSON, one per run, with the pose trace.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$(cd ../.. && pwd)

COURSE=speed LAPS="" SEED="" KNOBS="" TIMEOUT="" RTF=1.0 LABEL="" MATRIX=""
BIAS=0 GAIN=1 CAM_YAW=0 CAM_PITCH=0 CAM_ROLL=0 DRIFT=0 YAW_DRIFT=0 OFFSET="0 0 0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --course) COURSE="$2"; shift 2;;
    --laps) LAPS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --knobs) KNOBS="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --rtf) RTF="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --steer-bias) BIAS="$2"; shift 2;;
    --steer-gain) GAIN="$2"; shift 2;;
    --camera-yaw) CAM_YAW="$2"; shift 2;;
    --camera-pitch) CAM_PITCH="$2"; shift 2;;
    --camera-roll) CAM_ROLL="$2"; shift 2;;
    --pose-drift) DRIFT="$2"; shift 2;;
    --yaw-drift) YAW_DRIFT="$2"; shift 2;;
    --route-offset) OFFSET="$2"; shift 2;;
    --matrix) MATRIX="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 1;;
  esac
done

if [[ -n "$MATRIX" ]]; then
  # Nominal, then each fault at the size the car is expected to show, then
  # the larger ones that find where it breaks.  One line per run at the end.
  cases=(
    "nominal|"
    "steer_bias_-0.035|--steer-bias -0.035"
    "steer_bias_+0.035|--steer-bias 0.035"
    "steer_gain_0.82|--steer-gain 0.82"
    "steer_gain_1.10|--steer-gain 1.10"
    "camera_yaw_-3|--camera-yaw -3"
    "camera_yaw_+3|--camera-yaw 3"
    "camera_pitch_-2|--camera-pitch -2"
    "camera_pitch_+2|--camera-pitch 2"
    "camera_roll_-2|--camera-roll -2"
    "camera_roll_+2|--camera-roll 2"
    "bias_+0.035_yaw_-3|--steer-bias 0.035 --camera-yaw -3"
    "bias_-0.035_yaw_+3|--steer-bias -0.035 --camera-yaw 3"
    "pose_drift_1pct|--pose-drift 0.01 --yaw-drift 0.05"
    "route_offset|--route-offset '0.3 0.3 3'"
    "steer_bias_-0.07|--steer-bias -0.07"
    "steer_bias_+0.07|--steer-bias 0.07"
    "camera_yaw_-5|--camera-yaw -5"
    "camera_yaw_+5|--camera-yaw 5"
  )
  mkdir -p runs
  summary="runs/matrix_${MATRIX}_$(date +%Y%m%d_%H%M%S).txt"
  seeds=("")
  [[ "$MATRIX" == obstacle ]] && seeds=(201 202 208 218)
  for entry in "${cases[@]}"; do
    name="${entry%%|*}"; flags="${entry#*|}"
    for seed in "${seeds[@]}"; do
      label="${name}${seed:+_seed$seed}"
      # Faults on one layout only on the Obstacle Course (a run is two laps
      # and six minutes of wall clock); nominal on all four.
      [[ "$name" != nominal && -n "$seed" && "$seed" != 208 ]] && continue
      echo "=== $label"
      set +e
      eval "\"$0\" --course \"$MATRIX\" --label \"$label\" ${seed:+--seed $seed} ${LAPS:+--laps $LAPS} ${KNOBS:+--knobs \"$KNOBS\"} --rtf $RTF $flags" \
        | tee /tmp/plan_z_case.log | grep -E "lap [0-9]|PASS|FAIL" || true
      set -e
      grep -E '^\{' /tmp/plan_z_case.log | tail -1 >>"$summary" || echo "{\"label\": \"$label\", \"outcome\": \"no result\"}" >>"$summary"
    done
  done
  echo
  python3 - "$summary" <<'PY'
import json, sys
print(f"{'case':30s} {'clean':5s} {'outcome':9s} laps  {'lap times':24s} closest  contact  hoops")
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f"{r.get('label',''):30s} {str(r.get('clean','-')):5s} {r.get('outcome',''):9s} {r.get('laps','-'):>3}   "
          f"{str(r.get('lap_times','')):24s} {r.get('closest','-'):>6}   {r.get('contact_samples','-'):>5}   {r.get('hoops','-')}")
PY
  echo "results: $summary"
  exit 0
fi

[[ "$COURSE" == speed || "$COURSE" == obstacle ]] || { echo "--course speed|obstacle"; exit 1; }
[[ -n "$LAPS" ]] || LAPS=$([[ "$COURSE" == speed ]] && echo 3 || echo 2)
[[ -n "$TIMEOUT" ]] || TIMEOUT=$([[ "$COURSE" == speed ]] && echo 300 || echo 400)

# Its own ROS domain and gz partition: two stacks that can see each other
# interleave silently rather than failing.
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-79}
export GZ_PARTITION=${GZ_PARTITION:-plan_z}
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
WS="${ROS2_WS:-$HOME/ros2_ws}"
[[ -f "$WS/install/setup.bash" ]] || { echo "build the workspace first: jetson/scripts/build.sh"; exit 1; }
set +u
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
set -u

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

SHARE="$WS/install/cfr_arduino_bridge/share/cfr_arduino_bridge"
# build.sh copies the world into the install tree; write it fresh from the
# source with this run's real-time factor (1.0 restores the stock world).
sed "s|<real_time_factor>[^<]*</real_time_factor>|<real_time_factor>$RTF</real_time_factor>|" \
  "$REPO/jetson/cfr_arduino_bridge/worlds/${COURSE}_course.sdf" >"$SHARE/worlds/${COURSE}_course.sdf"
grep -q camera_rpy_deg "$SHARE/launch/simulation.launch.py" || \
  { echo "The installed launch files are older than the source: rebuild (jetson/scripts/build.sh)."; exit 1; }

# The simulated car's steering table, shifted and scaled.
PARAMS=/tmp/plan_z_params.yaml
python3 - "$REPO/jetson/cfr_arduino_bridge/config/arduino_bridge.yaml" "$PARAMS" "$BIAS" "$GAIN" <<'PY'
import sys, yaml
src, dst, bias, gain = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
cfg = yaml.safe_load(open(src))
car = cfg["sim_vehicle"]["ros__parameters"]
car["steering_angle_points"] = [round(a * gain + bias, 5) for a in car["steering_angle_points"]]
yaml.safe_dump(cfg, open(dst, "w"))
PY

read -r OX OY OYAW <<<"$OFFSET"
ALL_KNOBS="inject_pose_drift=$DRIFT inject_yaw_drift_deg=$YAW_DRIFT route_offset_x=$OX route_offset_y=$OY route_offset_yaw_deg=$OYAW $KNOBS"

echo "starting the ${COURSE} course (laps $LAPS, rtf $RTF, steer bias $BIAS gain $GAIN, camera rpy $CAM_ROLL $CAM_PITCH $CAM_YAW)..."
spawn ros2 launch "$PWD/plan_z_sim.launch.py" course:="$COURSE" laps:="$LAPS" \
  knobs:="$ALL_KNOBS" camera_rpy_deg:="$CAM_ROLL $CAM_PITCH $CAM_YAW" params_file:="$PARAMS" \
  >/tmp/plan_z_sim.log 2>&1

if [[ -n "$SEED" && "$COURSE" == obstacle ]]; then
  echo -n "waiting for the randomizer"
  for _ in $(seq 120); do
    if ros2 service list 2>/dev/null | grep -q /obstacle_randomizer/randomize; then echo " ok"; break; fi
    echo -n "."; sleep 1
  done
  # The service can be listed a moment before the node answers for its
  # parameters.
  for _ in $(seq 20); do
    ros2 param set /obstacle_randomizer seed "$SEED" 2>/dev/null | grep -q successful && break
    sleep 1
  done
  ros2 service call /obstacle_randomizer/randomize std_srvs/srv/Trigger >/dev/null
  echo "layout seed $SEED"
fi

mkdir -p runs
OUT="runs/${COURSE}_${LABEL:-run}_$(date +%Y%m%d_%H%M%S).json"
python3 gazebo_watch.py --course "$COURSE" --timeout "$TIMEOUT" --out "$OUT" --label "${LABEL:-run}"
status=$?
echo "trace: $PWD/$OUT   logs: /tmp/plan_z_sim.log"
exit $status
