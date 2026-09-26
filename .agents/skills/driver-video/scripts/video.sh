#!/usr/bin/env bash
# Make a video of one driver on one course, under Gazebo physics or as a
# replay of its numpy training sim.  Runs on the host; the sim runs in a
# Docker container of its own (cfr-video), separate from sim-launch's cfr-sim.
#
#   video.sh gazebo --course obstacle --policy rl/obstacleRacer/runs/v6/policy.npz --seed 104
#   video.sh replay --course obstacle --policy rl/obstacleRacer/runs/v6/best_model.zip --seed 104
#   video.sh gazebo --course speed    --policy rl/formulaOne/bestModel/v12/policy.npz
#   video.sh replay --course speed    --policy rl/formulaOne/bestModel/v12/policy.npz --pick any
#   video.sh stop                     # stop the container (it keeps its build)
#
# Options:
#   --policy PATH     policy.npz, or an SB3 .zip (exported to .npz first)
#   --config PATH     the policy's config.yaml (default: beside the policy)
#   --seed N          obstacle layout seed (default 104)
#   --out FILE.mp4    default: <policy dir>/video/<name>.mp4
#   --label TEXT      driver name in the title (default: from the policy path)
#   --timeout S       gazebo: sim seconds after the start before giving up (180)
#   --rtf R           Gazebo real-time factor cap (0.1; raise it on an idle machine)
#   --episodes N      replay: numpy starts to try (obstacle 24, speed 1)
#   --pick P          replay: fastest | first | median | any  (any = longest run if none finish)
#   --track NPZ       replay: reuse a rollout.py track instead of rolling out again
#   --limit N         replay: render only the first N frames (a quick look)
#   --keep            leave the container running afterwards
#   --rebuild         rebuild the container's ROS workspace from the repo first
set -euo pipefail
# Git Bash rewrites /paths in arguments to Windows programs; docker needs
# container paths left alone, the host python needs them converted.
dk() { MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker "$@"; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
CONTAINER=cfr-video
IMAGE=unfrobotics/docker-ros2-jazzy-gz-rviz2:latest
log() { echo "[video] $*"; }
die() { echo "[video] ERROR: $*" >&2; exit 1; }
in_box() { dk exec "$CONTAINER" bash -c "source /opt/ros/jazzy/setup.bash; [ -f ~/ros2_ws/install/setup.bash ] && source ~/ros2_ws/install/setup.bash; $1"; }
put() { dk exec -i "$CONTAINER" bash -c "cat > '$2'" < "$1"; }  # docker cp trips on Windows paths

MODE=${1:-}; shift || true
[[ "$MODE" == "stop" ]] && { dk stop "$CONTAINER" >/dev/null 2>&1 || true; log "stopped $CONTAINER"; exit 0; }
[[ "$MODE" == "gazebo" || "$MODE" == "replay" ]] || die "first argument: gazebo | replay | stop"

COURSE=""; POLICY=""; CONFIG=""; SEED=104; OUT=""; LABEL=""; TIMEOUT=180; RTF=0.1
EPISODES=""; PICK=fastest; TRACK=""; LIMIT=0; KEEP=false; REBUILD=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --course) COURSE=$2; shift 2;;
    --policy) POLICY=$2; shift 2;;
    --config) CONFIG=$2; shift 2;;
    --seed) SEED=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --label) LABEL=$2; shift 2;;
    --timeout) TIMEOUT=$2; shift 2;;
    --rtf) RTF=$2; shift 2;;
    --episodes) EPISODES=$2; shift 2;;
    --pick) PICK=$2; shift 2;;
    --track) TRACK=$2; shift 2;;
    --limit) LIMIT=$2; shift 2;;
    --keep) KEEP=true; shift;;
    --rebuild) REBUILD=true; shift;;
    *) die "unknown argument: $1";;
  esac
done
[[ "$COURSE" == "obstacle" || "$COURSE" == "speed" ]] || die "--course obstacle | speed"
[[ -n "$POLICY" || -n "$TRACK" ]] || die "--policy is required"
[[ "$MODE" == replay || -n "$POLICY" ]] || die "gazebo mode needs --policy"

# Paths may be relative to the caller or to the repo; runs/ is gitignored, so
# from a worktree they usually live in the main checkout -- try that too.
MAIN=""
if [[ -f "$REPO/.git" ]]; then  # a linked worktree: .git names <main>/.git/worktrees/<name>
  MAIN=$(sed -n 's|^gitdir: *\(.*\)/\.git/worktrees/.*|\1|p' "$REPO/.git")
fi
resolve() {
  local p=$1
  for c in "$p" "$REPO/$p" "${MAIN:-$REPO}/$p"; do [[ -e "$c" ]] && { (cd "$(dirname "$c")" && echo "$(pwd)/$(basename "$c")"); return; }; done
  die "not found: $p"
}
# Host python: rl/obstacleRacer's venv has numpy, numba, torch and SB3, enough
# for both courses' numpy sims and both export scripts.
if [[ -z "${PYTHON:-}" ]]; then
  for v in "$REPO" "${MAIN:-$REPO}"; do
    for b in Scripts/python.exe bin/python; do
      [[ -x "$v/rl/obstacleRacer/.venv/$b" ]] && { PYTHON="$v/rl/obstacleRacer/.venv/$b"; break 2; }
    done
  done
fi
[[ -n "${PYTHON:-}" ]] || die "no rl/obstacleRacer/.venv; set PYTHON to a python with numpy, numba, pyyaml (and torch + stable-baselines3 for .zip policies)"

RLDIR=$([[ "$COURSE" == obstacle ]] && echo obstacleRacer || echo formulaOne)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/driver-video.XXXXXX")
if [[ -n "$POLICY" ]]; then
  POLICY=$(resolve "$POLICY")
  CONFIG=$(resolve "${CONFIG:-$(dirname "$POLICY")/config.yaml}")
  if [[ "$POLICY" == *.zip ]]; then
    log "exporting $POLICY -> policy.npz"
    "$PYTHON" "$REPO/rl/$RLDIR/export_policy.py" "$POLICY" -o "$WORK/policy.npz" | tail -2
    NPZ="$WORK/policy.npz"
  else
    NPZ="$POLICY"
  fi
  NAME=$(basename "$(dirname "$POLICY")")_$(basename "${POLICY%.*}")
  LABEL=${LABEL:-"$RLDIR $(basename "$(dirname "$POLICY")")"}
else
  TRACK=$(resolve "$TRACK")
  NAME=$(basename "${TRACK%.*}")
  LABEL=${LABEL:-"$RLDIR"}
fi
[[ "$COURSE" == obstacle ]] && TAG="${NAME}_seed${SEED}_${MODE}" || TAG="${NAME}_${MODE}"
if [[ -z "$OUT" ]]; then
  OUT="$(dirname "${POLICY:-$TRACK}")/video/$TAG.mp4"
fi
mkdir -p "$(dirname "$OUT")"
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"

# ---------------------------------------------------------------- numpy rollout
if [[ "$MODE" == replay && -z "$TRACK" ]]; then
  log "rolling out in the numpy sim"
  set +e
  "$PYTHON" "$HERE/rollout.py" --course "$COURSE" --policy "$NPZ" --config "$CONFIG" --seed "$SEED" \
    ${EPISODES:+--episodes "$EPISODES"} --pick "$PICK" --out "$WORK/track.npz"
  rc=$?
  set -e
  [[ $rc -eq 0 ]] || die "no usable episode (see above); try --pick any, more --episodes, another --seed, or gazebo mode"
  TRACK="$WORK/track.npz"
  cp "$TRACK" "${OUT%.mp4}_track.npz"; cp "$TRACK.json" "${OUT%.mp4}_track.npz.json"
fi

# ---------------------------------------------------------------- container
want=$(cd "$REPO" && (pwd -W 2>/dev/null || pwd))
mount=$(dk inspect -f '{{index .Config.Labels "cfr.repo"}}' "$CONTAINER" 2>/dev/null || true)
if [[ -n "$mount" && "$mount" != "$want" ]] || { [[ -z "$mount" ]] && dk inspect "$CONTAINER" >/dev/null 2>&1; }; then
  log "$CONTAINER mounts ${mount:-an unknown checkout}, not $want: recreating it"
  dk rm -f "$CONTAINER" >/dev/null
  mount=""
fi
if [[ -z "$mount" ]]; then
  log "creating $CONTAINER"
  dk run -d --name "$CONTAINER" --label "cfr.repo=$want" -v "$want:/repo" "$IMAGE" sleep infinity >/dev/null
  REBUILD=true
fi
# A restart is the one reliable way to clear a previous launch: killing
# `ros2 launch` orphans gz sim and the nodes.
dk restart "$CONTAINER" >/dev/null
dk exec "$CONTAINER" bash -c "command -v ffmpeg >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null)"
if [[ "$REBUILD" == true ]] || ! dk exec "$CONTAINER" test -f /root/ros2_ws/install/setup.bash; then
  log "building the ROS workspace (a few minutes)"
  in_box "cd /repo && ./jetson/scripts/build.sh 2>&1 | tail -3"
fi
dk exec "$CONTAINER" bash -c "rm -rf /work && mkdir -p /work/out"
put "$HERE/capture.py" /work/capture.py
[[ -n "${NPZ:-}" ]] && put "$NPZ" /work/policy.npz && put "$CONFIG" /work/config.yaml
[[ -n "$TRACK" ]] && put "$TRACK" /work/track.npz && put "$TRACK.json" /work/track.npz.json

# ---------------------------------------------------------------- the stack
WORLD=$([[ "$COURSE" == obstacle ]] && echo cfr_obstacle_course || echo cfr_speed_course)
LAUNCH=$([[ "$COURSE" == obstacle ]] && echo obstacle_course.launch.py || echo speed_course.launch.py)
# Obstacle + gazebo renders the ZED for the driver (sensors:=true fills the
# world's sensors marker itself); every other case runs sensors:=false and the
# film cameras need the Sensors system patched in.
SENSORS_PLUGIN=--sensors-plugin
[[ "$MODE" == gazebo && "$COURSE" == obstacle ]] && SENSORS_PLUGIN=""
in_box "python3 /work/capture.py patch-world --course $COURSE --rtf $RTF $SENSORS_PLUGIN"
bg() { dk exec -d "$CONTAINER" bash -c "source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && export LIBGL_ALWAYS_SOFTWARE=1 && $1 > $2 2>&1"; }
if [[ "$MODE" == replay ]]; then
  bg "ros2 launch cfr_arduino_bridge $LAUNCH sensors:=false path_follower:=false cmd_vel_to_drive:=false" /work/sim.log
elif [[ "$COURSE" == obstacle ]]; then
  bg "ros2 launch /repo/rl/obstacleRacer/obstacle_racer_sim.launch.py policy:=/work/policy.npz config:=/work/config.yaml laps:=1" /work/sim.log
else
  LAPS=$(in_box "python3 -c \"import yaml;print(yaml.safe_load(open('/work/config.yaml'))['env']['laps'])\"")
  bg "ros2 launch cfr_arduino_bridge $LAUNCH sensors:=false path_follower:=false cmd_vel_to_drive:=false laps:=$LAPS" /work/sim.log
  bg "ros2 launch /repo/rl/formulaOne/formula_one.launch.py policy:=/work/policy.npz config:=/work/config.yaml rviz:=false record:=false use_sim_time:=true" /work/driver.log
fi
W="/world/$WORLD"
bg "ros2 run ros_gz_bridge parameter_bridge /video/chase@sensor_msgs/msg/Image[gz.msgs.Image /video/zed@sensor_msgs/msg/Image[gz.msgs.Image /video/map@sensor_msgs/msg/Image[gz.msgs.Image $W/control@ros_gz_interfaces/srv/ControlWorld $W/set_pose@ros_gz_interfaces/srv/SetEntityPose $W/create@ros_gz_interfaces/srv/SpawnEntity $W/remove@ros_gz_interfaces/srv/DeleteEntity --ros-args -p use_sim_time:=true" /work/bridge.log
log "waiting for the stack"
READY="$W/set_pose"
[[ "$MODE" == gazebo && "$COURSE" == obstacle ]] && READY=/obstacle_racer/manual_start
[[ "$MODE" == gazebo && "$COURSE" == speed ]] && READY=/formula_one/manual_start
in_box "for i in \$(seq 120); do ros2 service list 2>/dev/null | grep -q '$READY' && exit 0; sleep 2; done; echo 'stack did not come up; tail of /work/sim.log:'; tail -20 /work/sim.log; exit 1" \
  || die "stack failed"

# ---------------------------------------------------------------- film
SEEDTXT=""
[[ "$COURSE" == obstacle ]] && SEEDTXT="layout seed $SEED | "
if [[ "$MODE" == gazebo ]]; then
  SUB="Gazebo physics | ${SEEDTXT}sim-time playback"
  EXTRA="--seed $SEED --timeout $TIMEOUT"
  if [[ "$COURSE" == obstacle ]]; then
    EXTRA="$EXTRA --zed-topic /zed/zed_node/left/image_rect_color"
  else
    EXTRA="$EXTRA --laps $LAPS"
  fi
  NOTE=""
else
  SUB="${SEEDTXT}physics: numpy training sim, replayed in Gazebo's renderer"
  EXTRA="--track /work/track.npz --limit $LIMIT"
  NOTE="not Gazebo physics"
fi
TITLE="$LABEL on the $([[ $COURSE == obstacle ]] && echo Obstacle || echo Speed) Course"
log "filming ($MODE): $TITLE"
TEXT=$(printf '%q ' --title "$TITLE" --subtitle "$SUB" --note "$NOTE")  # any apostrophes survive
in_box "set -o pipefail; cd /work && python3 capture.py $MODE --course $COURSE --out /work/out $TEXT $EXTRA 2>&1 | grep -av -i 'warn'" \
  || die "capture failed; logs in the container: /work/sim.log /work/bridge.log"
in_box "cd /work/out && ffmpeg -y -loglevel error -f concat -safe 0 -i list.txt -vf fps=30,format=yuv420p -c:v libx264 -crf 20 -preset slow -movflags +faststart video.mp4 && ffprobe -v error -show_entries format=duration -of csv=p=0 video.mp4"
dk exec "$CONTAINER" cat /work/out/video.mp4 > "$OUT"
dk exec "$CONTAINER" cat /work/out/summary.json > "${OUT%.mp4}_summary.json"
log "summary: $(tr -d '\n' < "${OUT%.mp4}_summary.json")"
[[ "$KEEP" == true ]] || dk stop "$CONTAINER" >/dev/null
log "video: $OUT"
