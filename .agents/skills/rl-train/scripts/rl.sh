#!/usr/bin/env bash
#
# Run, watch and stop RL training for the bale-following policy inside the
# Docker sim image, so a multi-hour run can be started from a Windows host
# without reconstructing the container, venv, workspace build and launch
# arguments each time.
#
# Usage:
#   rl.sh setup                                   # container + venv + workspace build
#   rl.sh start [--course speed|obstacle] [--steps N] [--dir NAME]
#               [--resume-from PATH] [--sensors|--no-sensors] [--curriculum] [--force]
#   rl.sh status [--dir NAME]
#   rl.sh logs [--lines N]
#   rl.sh eval --checkpoint PATH [--episodes N]
#   rl.sh stop
#
# This drives the repo's own wrappers (train_resilient.sh, test_policy.sh)
# rather than reimplementing them: their chunked auto-resume exists because
# Gazebo's pose stream dies after roughly five hours, and that logic is worth
# more than a tidier invocation would be.

set -euo pipefail

# Git Bash (MSYS) rewrites arguments that look like Unix paths into Windows
# paths before the process sees them, which mangles both the container-side
# paths in these docker exec strings and any URL passed to pip: the torch
# index URL arrives as `C:\Program Files\...` and pip is handed two commands
# that do not exist. Turning conversion off is the standard remedy for
# driving Docker from Git Bash and is harmless elsewhere.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

CONTAINER=cfr-rl
IMAGE="unfrobotics/docker-ros2-jazzy-gz-rviz2:latest"
VENV_VOLUME=cfr-rl-venv
RL_IN_CONTAINER=/repo/rl/bale_follower
# A separate ROS domain and Gazebo partition from the cfr-sim viewer container.
# Two Gazebo servers reachable from each other publish onto one pose topic and
# corrupt each other's measurements silently -- the exact failure train_resilient.sh
# warns about, except across containers where its pgrep count would not see it.
DOMAIN_ID=42
GZ_PART=cfr_rl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
RL_DIR="$REPO_DIR/rl/bale_follower"

PYTHON_BIN="$(command -v python3 || command -v python)"

log() { echo "[rl.sh] $*"; }
die() { echo "[rl.sh] error: $*" >&2; exit 1; }

in_container() { docker exec "$CONTAINER" bash -lc "$1"; }
in_container_quiet() { docker exec "$CONTAINER" bash -lc "$1" >/dev/null 2>&1; }

# ---- container ------------------------------------------------------------

ensure_container() {
    docker info >/dev/null 2>&1 || die "Docker is not running (docker info failed). Start Docker Desktop first."
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; then
        return
    fi
    if docker inspect "$CONTAINER" >/dev/null 2>&1; then
        log "starting existing container '$CONTAINER'"
        docker start "$CONTAINER" >/dev/null
        return
    fi
    log "creating container '$CONTAINER' from $IMAGE"
    # The venv is a named volume mounted over rl/bale_follower/.venv rather
    # than a directory in the bind mount: pip unpacks tens of thousands of
    # files (torch alone is most of a gigabyte) and doing that onto the
    # Windows filesystem through the mount is slow enough to matter, both
    # when installing and on every import afterwards. The volume also
    # survives `docker rm`, so a recreated container does not reinstall.
    docker run -d --name "$CONTAINER" \
        -v "$REPO_DIR:/repo" \
        -v "$VENV_VOLUME:$RL_IN_CONTAINER/.venv" \
        -e ROS_DOMAIN_ID="$DOMAIN_ID" \
        -e GZ_PARTITION="$GZ_PART" \
        "$IMAGE" sleep infinity >/dev/null
}

# ---- setup ----------------------------------------------------------------

setup() {
    local gpu=false
    while [ $# -gt 0 ]; do
        case "$1" in
            --gpu) gpu=true; shift ;;
            *) die "unknown setup option '$1'" ;;
        esac
    done

    ensure_container

    if ! in_container_quiet "dpkg -s python3-venv python3-pip"; then
        log "installing python3-venv and python3-pip (the sim image ships neither)"
        in_container "sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip" >/dev/null
    fi

    if ! in_container_quiet "test -x $RL_IN_CONTAINER/.venv/bin/python"; then
        # --system-site-packages is not optional: rclpy lives in
        # /opt/ros/jazzy and is reached through the ROS setup script, and an
        # isolated venv hides it from the env at import time.
        log "creating venv (--system-site-packages, for rclpy)"
        in_container "python3 -m venv --system-site-packages $RL_IN_CONTAINER/.venv"
    fi

    if ! in_container_quiet "$RL_IN_CONTAINER/.venv/bin/python -c 'import stable_baselines3'"; then
        if [ "$gpu" = false ]; then
            # PyPI's default linux torch wheel is the CUDA build: ~2.5 GB of
            # download for a container with no GPU. Training is paced by the
            # simulator's wall clock anyway (control_hz steps per second), so
            # the CPU wheel costs nothing in throughput here.
            log "installing torch (CPU wheel) -- several minutes"
            in_container "$RL_IN_CONTAINER/.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu"
        fi
        log "installing the rest of requirements.txt"
        in_container "$RL_IN_CONTAINER/.venv/bin/pip install -q -r $RL_IN_CONTAINER/requirements.txt"
    fi

    # ROS2_WS=/repo, not the image's default ~/ros2_ws: _sim_stack.sh and
    # train_resilient.sh both source $REPO_ROOT/install/setup.bash and refuse
    # to run without it. Building where they look keeps those wrappers usable
    # unmodified. install/, build/ and log/ are already gitignored.
    log "building the workspace into /repo (colcon)"
    in_container "source /opt/ros/jazzy/setup.bash && cd /repo && ROS2_WS=/repo ./jetson/scripts/build.sh" 2>&1 | sed 's/^/  /'

    log "setup complete"
    in_container "$RL_IN_CONTAINER/.venv/bin/python -c 'import stable_baselines3, torch; print(\"stable_baselines3\", stable_baselines3.__version__, \"torch\", torch.__version__)'" | sed 's/^/  /'
}

# ---- start ----------------------------------------------------------------

config_scan_source() {
    sed -n 's/^\s*scan_source:\s*\([a-z]*\).*/\1/p' "$RL_DIR/config.yaml" | head -1
}

training_running() {
    # Bracket the first character so pgrep's own command line cannot match.
    docker exec "$CONTAINER" bash -c "pgrep -f '[t]rain_resilient.sh|[t]rain.py|[t]rain_curriculum.sh' >/dev/null 2>&1"
}

start() {
    local course=speed steps=200000 dir="" resume="" sensors="" curriculum=false force=false
    while [ $# -gt 0 ]; do
        case "$1" in
            --course) course="$2"; shift 2 ;;
            --steps) steps="$2"; shift 2 ;;
            --dir) dir="$2"; shift 2 ;;
            --resume-from) resume="$2"; shift 2 ;;
            --sensors) sensors=true; shift ;;
            --no-sensors) sensors=false; shift ;;
            --curriculum) curriculum=true; shift ;;
            --force) force=true; shift ;;
            *) die "unknown start option '$1'" ;;
        esac
    done
    [ "$course" = speed ] || [ "$course" = obstacle ] || die "--course must be speed or obstacle"

    if ! "$PYTHON_BIN" "$SCRIPT_DIR/course_preflight.py" --course "$course" --repo "$REPO_DIR"; then
        [ "$force" = true ] || die "preflight failed; fix the checks above or pass --force to start anyway"
        log "preflight failed but --force was given; starting"
    fi

    ensure_container
    in_container_quiet "test -x $RL_IN_CONTAINER/.venv/bin/python" || die "no venv in the container; run 'rl.sh setup' first"
    in_container_quiet "test -f /repo/install/setup.bash" || die "workspace not built at /repo/install; run 'rl.sh setup' first"
    if training_running; then
        die "training is already running in '$CONTAINER' -- 'rl.sh status' to see it, 'rl.sh stop' to end it"
    fi

    # config.yaml's scan_source decides whether the ZED has to be rendered.
    # Getting this wrong is quiet and expensive: with scan_source: cloud and
    # no sensors, the env waits for a point cloud nothing publishes.
    local scan; scan="$(config_scan_source)"
    if [ -z "$sensors" ]; then
        if [ "$scan" = cloud ]; then sensors=true; else sensors=false; fi
        log "config.yaml scan_source: ${scan:-unset} -> sensors=$sensors"
    fi

    [ -n "$dir" ] || dir="checkpoints_$(date +%Y%m%d_%H%M)"
    local resume_args=""
    [ -n "$resume" ] && resume_args="--resume-from $resume"

    local env_prefix="LOG_DIR=$RL_IN_CONTAINER"
    [ "$sensors" = true ] && env_prefix="$env_prefix CFR_SENSORS=1"

    local command
    if [ "$curriculum" = true ]; then
        # train_curriculum.sh rewrites config.yaml's max_speed in place at each
        # stage and names its own checkpoint directories, so --steps and --dir
        # do not apply to it.
        log "starting the speed curriculum (this rewrites env.max_speed in config.yaml at each stage)"
        command="cd $RL_IN_CONTAINER && $env_prefix setsid bash train_curriculum.sh"
    else
        log "starting: $steps steps into $dir (course $course, sensors $sensors)"
        command="cd $RL_IN_CONTAINER && $env_prefix setsid bash train_resilient.sh $steps $dir $resume_args"
    fi

    docker exec -d "$CONTAINER" bash -lc "$command > $RL_IN_CONTAINER/rl_run.log 2>&1"

    sleep 5
    if training_running; then
        log "running. Logs: rl/bale_follower/rl_run.log, train_resilient.log, train_chunk_*.log"
        log "progress: rl.sh status --dir $dir"
        # Wall-clock expectation, so nobody waits on a run that is on schedule:
        # the env paces itself at control_hz steps per second against the wall
        # clock, and rendering the ZED drops the real-time factor to ~0.63.
        local hours
        hours="$("$PYTHON_BIN" -c "print(f'{$steps / 10.0 / 3600.0 / (0.63 if '$sensors' == 'true' else 1.0):.1f}')")"
        log "expect roughly ${hours} h of wall clock for $steps steps, plus eval overhead"
    else
        echo "[rl.sh] training exited immediately. Last log lines:" >&2
        in_container "tail -n 30 $RL_IN_CONTAINER/rl_run.log" >&2 || true
        exit 1
    fi
}

# ---- status ---------------------------------------------------------------

status() {
    local dir=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --dir) dir="$2"; shift 2 ;;
            *) die "unknown status option '$1'" ;;
        esac
    done

    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
        log "container '$CONTAINER': not running"
        return 0
    fi

    if training_running; then
        log "training: running"
    else
        log "training: not running (finished, stopped, or never started)"
    fi

    # One healthy server shows up once as the gz binary; more than one means
    # two worlds are publishing onto the same pose topic.
    local servers
    servers="$(docker exec "$CONTAINER" bash -c "ps -eo cmd | grep -c '^gz sim -r -s' || true" | tr -d '\r')"
    log "gz servers in container: ${servers:-0}$([ "${servers:-0}" -gt 1 ] && echo '  <-- duplicates corrupt the run' || true)"

    if [ -z "$dir" ]; then
        dir="$(ls -td "$RL_DIR"/checkpoints* 2>/dev/null | head -1 || true)"
        [ -n "$dir" ] || { log "no checkpoint directory found under rl/bale_follower"; return 0; }
    else
        dir="$RL_DIR/$dir"
    fi

    echo
    "$PYTHON_BIN" "$SCRIPT_DIR/progress.py" --dir "$dir" --logs "$RL_DIR"
}

# ---- logs -----------------------------------------------------------------

logs() {
    local lines=40
    while [ $# -gt 0 ]; do
        case "$1" in
            --lines) lines="$2"; shift 2 ;;
            *) die "unknown logs option '$1'" ;;
        esac
    done
    in_container "tail -n $lines \$(ls -t $RL_IN_CONTAINER/train_chunk_*.log 2>/dev/null | head -1) 2>/dev/null || tail -n $lines $RL_IN_CONTAINER/rl_run.log"
}

# ---- eval -----------------------------------------------------------------

evaluate() {
    local checkpoint="" episodes=5 extra=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --checkpoint) checkpoint="$2"; shift 2 ;;
            --episodes) episodes="$2"; shift 2 ;;
            *) extra+=("$1"); shift ;;
        esac
    done
    [ -n "$checkpoint" ] || die "--checkpoint is required"
    ensure_container
    if training_running; then
        die "training is running; its Gazebo server owns the topics and test_policy.sh refuses to start a second one. Stop the run first, or evaluate a copy of the checkpoint elsewhere."
    fi
    local scan; scan="$(config_scan_source)"
    local prefix=""
    [ "$scan" = cloud ] && prefix="CFR_SENSORS=1 "
    docker exec "$CONTAINER" bash -lc \
        "cd $RL_IN_CONTAINER && ${prefix}bash test_policy.sh --checkpoint $checkpoint --episodes $episodes ${extra[*]:-}"
}

# ---- stop -----------------------------------------------------------------

stop() {
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
        log "container '$CONTAINER' is not running; nothing to stop"
        return 0
    fi
    log "stopping training (best_model.zip and the step checkpoints are already on disk)"
    # SIGINT first: train_resilient.sh's own EXIT trap tears the Gazebo stack
    # down, and train.py's finally-block closes the env. Killing outright
    # leaves the rclpy executor and the gz server behind, and a stale server
    # poisons the next run's topics.
    docker exec "$CONTAINER" bash -c "pkill -INT -f '[t]rain_resilient.sh|[t]rain_curriculum.sh|[t]rain.py'" || true
    sleep 8
    docker exec "$CONTAINER" bash -c "pkill -KILL -f '[t]rain_resilient.sh|[t]rain_curriculum.sh|[t]rain.py'" 2>/dev/null || true
    docker exec "$CONTAINER" bash -c "pkill -f 'gz sim'" 2>/dev/null || true
    docker exec "$CONTAINER" bash -c "pkill -f 'ros2 launch cfr_arduino_bridge'" 2>/dev/null || true
    log "stopped"
}

# ---- dispatch -------------------------------------------------------------

command="${1:-}"
[ $# -gt 0 ] && shift || true
case "$command" in
    setup) setup "$@" ;;
    start) start "$@" ;;
    status) status "$@" ;;
    logs) logs "$@" ;;
    eval) evaluate "$@" ;;
    stop) stop "$@" ;;
    *)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
