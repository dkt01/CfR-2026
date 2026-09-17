#!/usr/bin/env bash
#
# Start/stop the CfR-2026 Gazebo simulation (in the unfrobotics Docker image)
# together with the gzweb browser viewer, so both come up with one call
# instead of the ~8 manual steps (start container, install gz-launch-vendor,
# build, launch with websocket:=true, verify the ports actually bound, start
# the vite dev server, confirm it's listening).
#
# Usage:
#   sim.sh start --course speed|obstacle [--gui] [--sensors] [--laps N] [--no-viewer]
#   sim.sh stop
#   sim.sh status
#
# Idempotent: calling `start` again while things are already up just makes
# sure both halves (sim + viewer) are running and prints the URL -- it does
# not relaunch or restart what's already there. That's on purpose: the sim
# holds state (randomized buckets/hoops, vehicle pose) that a casual rerun
# should not throw away.

set -euo pipefail

CONTAINER=cfr-sim
IMAGE="unfrobotics/docker-ros2-jazzy-gz-rviz2:latest"
VIEWER_PORT=5173
WS_PORT=9002
TELEPORT_PORT=9003
DEV_LOG=/tmp/cfr_sim_dev.log
DEV_PID_FILE=/tmp/cfr_sim_dev.pid

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
VIEWER_DIR="$REPO_DIR/web/gzweb-viewer"

log() { echo "[sim.sh] $*"; }

# ---- container lifecycle -----------------------------------------------

container_exists() {
    docker inspect "$CONTAINER" >/dev/null 2>&1
}

container_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]
}

ensure_container() {
    if ! docker info >/dev/null 2>&1; then
        echo "Docker doesn't seem to be running (docker info failed). Start Docker Desktop first." >&2
        exit 1
    fi
    if container_running; then
        log "container '$CONTAINER' already running"
        return
    fi
    if container_exists; then
        log "starting existing container '$CONTAINER'"
        docker start "$CONTAINER" >/dev/null
        return
    fi
    log "creating container '$CONTAINER' from $IMAGE"
    # The repo is mounted read-write: build.sh builds into the container's
    # own ~/ros2_ws, never into the mount, but the launch files, worlds and
    # configs are read live from /repo so an edit on the host takes effect
    # on the next launch without rebuilding the image.
    docker run -d --name "$CONTAINER" \
        -v "$REPO_DIR:/repo" \
        -p "$WS_PORT:$WS_PORT" -p "$TELEPORT_PORT:$TELEPORT_PORT" \
        "$IMAGE" sleep infinity >/dev/null
}

ensure_gz_launch_vendor() {
    if docker exec "$CONTAINER" bash -lc "dpkg -s ros-jazzy-gz-launch-vendor" >/dev/null 2>&1; then
        return
    fi
    # Without this, `gz launch` isn't a command: websocket:=true's process
    # exits 255 at startup and the rest of the launch comes up looking
    # normal, so the only symptom is the viewer stuck on "Connecting to
    # simulation" -- worth installing proactively rather than diagnosing.
    log "installing ros-jazzy-gz-launch-vendor (needed for websocket:=true)"
    docker exec "$CONTAINER" bash -lc "sudo apt-get update -qq && sudo apt-get install -y -qq ros-jazzy-gz-launch-vendor" >/dev/null
}

build_workspace() {
    log "building workspace"
    docker exec "$CONTAINER" bash -lc \
        "source /opt/ros/jazzy/setup.bash && cd /repo && ./jetson/scripts/build.sh" \
        2>&1 | sed 's/^/  /'
}

sim_launch_running() {
    # The bracket around the first letter keeps this pattern from matching
    # its own invoking command line -- pgrep -f matches full cmdlines, and
    # without it the literal search string here would match itself, always
    # reporting "running" even against a freshly restarted container.
    docker exec "$CONTAINER" bash -c "pgrep -f '[r]os2 launch cfr_arduino_bridge' >/dev/null 2>&1"
}

# A host-side `docker port` check is not enough: docker-proxy binds the
# published port on the host even when nothing inside the container is
# actually listening, so a plain TCP connect can succeed against a dead
# server. Read the container's own /proc/net/tcp instead.
port_listening_in_container() {
    local port="$1"
    docker exec "$CONTAINER" python3 -c "
import sys
port = $port
with open('/proc/net/tcp') as f:
    lines = f.readlines()[1:]
listening = any(int(l.split()[1].split(':')[1], 16) == port for l in lines)
sys.exit(0 if listening else 1)
"
}

wait_for_ports() {
    local tries=30
    while [ "$tries" -gt 0 ]; do
        if port_listening_in_container "$WS_PORT" && port_listening_in_container "$TELEPORT_PORT"; then
            return 0
        fi
        sleep 1
        tries=$((tries - 1))
    done
    return 1
}

launch_sim() {
    local course="$1" gui="$2" sensors="$3" laps="$4"
    local launch_file="${course}_course.launch.py"
    local extra_args=()
    [ -n "$laps" ] && extra_args+=("laps:=$laps")

    log "launching $launch_file (gui:=$gui sensors:=$sensors websocket:=true ${extra_args[*]:-})"
    docker exec -d "$CONTAINER" bash -lc \
        "source /opt/ros/jazzy/setup.bash && source /root/ros2_ws/install/setup.bash && \
         ros2 launch cfr_arduino_bridge $launch_file websocket:=true gui:=$gui sensors:=$sensors ${extra_args[*]:-} \
         > /tmp/sim_launch.log 2>&1"

    if ! wait_for_ports; then
        echo "Ports $WS_PORT/$TELEPORT_PORT never came up. Recent launch log:" >&2
        docker exec "$CONTAINER" bash -c "tail -n 40 /tmp/sim_launch.log" >&2 || true
        exit 1
    fi
    log "simulation up: websocket on $WS_PORT, teleport bridge on $TELEPORT_PORT"
}

# ---- browser viewer -------------------------------------------------------

viewer_running() {
    [ -f "$DEV_PID_FILE" ] && kill -0 "$(cat "$DEV_PID_FILE")" 2>/dev/null
}

start_viewer() {
    if viewer_running; then
        log "viewer already running (pid $(cat "$DEV_PID_FILE"))"
        return
    fi
    log "starting gzweb viewer dev server on :$VIEWER_PORT"
    (
        cd "$VIEWER_DIR"
        nohup npm run dev -- --port "$VIEWER_PORT" > "$DEV_LOG" 2>&1 &
        echo $! > "$DEV_PID_FILE"
    )
    local tries=15
    while [ "$tries" -gt 0 ]; do
        grep -q "ready in" "$DEV_LOG" 2>/dev/null && return 0
        sleep 1
        tries=$((tries - 1))
    done
    echo "Viewer dev server didn't report ready. Recent log:" >&2
    tail -n 20 "$DEV_LOG" >&2 || true
    exit 1
}

stop_viewer() {
    if [ -f "$DEV_PID_FILE" ]; then
        local pid
        pid="$(cat "$DEV_PID_FILE")"
        # `npm run dev` forks vite as a child; killing the tracked pid's
        # process group takes both, a plain `kill $pid` can leave vite
        # running as an orphan that keeps the port bound.
        kill -- -"$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null \
            || kill "$pid" 2>/dev/null \
            || true
        rm -f "$DEV_PID_FILE"
        log "stopped viewer dev server"
    else
        log "no tracked viewer process (pid file missing) -- leaving any :$VIEWER_PORT listener alone"
    fi
}

# ---- subcommands ----------------------------------------------------------

cmd_start() {
    local course="" gui=false sensors=false laps="" with_viewer=true
    while [ $# -gt 0 ]; do
        case "$1" in
            --course) course="$2"; shift 2 ;;
            --gui) gui=true; shift ;;
            --sensors) sensors=true; shift ;;
            --laps) laps="$2"; shift 2 ;;
            --no-viewer) with_viewer=false; shift ;;
            *) echo "unknown argument: $1" >&2; exit 1 ;;
        esac
    done
    case "$course" in
        speed|obstacle) ;;
        *) echo "pass --course speed|obstacle" >&2; exit 1 ;;
    esac

    ensure_container
    ensure_gz_launch_vendor

    if sim_launch_running; then
        log "a simulation launch is already running in '$CONTAINER' -- leaving it as-is (use 'sim.sh stop' first to relaunch with different arguments)"
    else
        build_workspace
        launch_sim "$course" "$gui" "$sensors" "$laps"
    fi

    if [ "$with_viewer" = true ]; then
        start_viewer
        local url="http://localhost:$VIEWER_PORT/"
        [ "$course" = "obstacle" ] && url="${url}?course=obstacle"
        log "viewer: $url"
    fi
}

cmd_stop() {
    stop_viewer
    if container_running; then
        # `pkill -f "ros2 launch"` orphans gz sim and the ROS nodes instead
        # of stopping them; `docker restart` is the clean way to kill
        # everything the launch started and leave the container ready for
        # the next `sim.sh start`.
        log "restarting container '$CONTAINER' to stop the simulation"
        docker restart "$CONTAINER" >/dev/null
    else
        log "container not running, nothing to stop"
    fi
}

cmd_status() {
    if ! container_exists; then
        echo "container: does not exist"
        return
    fi
    if ! container_running; then
        echo "container: stopped"
        return
    fi
    echo "container: running"
    if sim_launch_running; then
        echo "simulation: running"
        for p in "$WS_PORT" "$TELEPORT_PORT"; do
            if port_listening_in_container "$p"; then
                echo "  port $p: listening"
            else
                echo "  port $p: NOT listening"
            fi
        done
    else
        echo "simulation: not running"
    fi
    if viewer_running; then
        echo "viewer: running (pid $(cat "$DEV_PID_FILE")) -- http://localhost:$VIEWER_PORT/"
    else
        echo "viewer: not running"
    fi
}

case "${1:-}" in
    start) shift; cmd_start "$@" ;;
    stop) shift; cmd_stop "$@" ;;
    status) shift; cmd_status "$@" ;;
    *)
        echo "usage: sim.sh start --course speed|obstacle [--gui] [--sensors] [--laps N] [--no-viewer]" >&2
        echo "       sim.sh stop" >&2
        echo "       sim.sh status" >&2
        exit 1
        ;;
esac
