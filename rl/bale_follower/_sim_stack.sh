#!/usr/bin/env bash
# Shared plumbing for launch_training.sh and test_policy.sh: bring up the
# Gazebo training stack, wait until it is actually ready, and guarantee
# teardown. Not meant to be run directly -- source it.
#
# Readiness is not "the launch printed something": the env dies at startup if
# the teleport API is not answering or the ground-truth pose bridge is not up,
# so this waits for both explicitly before handing control to Python.

# No -u: ROS 2's setup.bash trips over unbound variables under nounset.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCH_LOG="${LAUNCH_LOG:-$SCRIPT_DIR/sim_stack.log}"
SIM_PID=""

die() { echo "error: $*" >&2; exit 1; }

setup_environment() {
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    [ -f "$REPO_ROOT/install/setup.bash" ] || die "workspace not built: run 'colcon build' in $REPO_ROOT first"
    # shellcheck disable=SC1091
    source "$REPO_ROOT/install/setup.bash"
    [ -f "$SCRIPT_DIR/.venv/bin/activate" ] || die "no venv: see rl/bale_follower/README.md Setup section"
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/bin/activate"
}

start_sim() {
    # README: several Gazebo servers on the same topics produce poses that
    # jump between worlds. Refuse to stack a second one.
    if pgrep -f "gz sim" > /dev/null; then
        die "a Gazebo server is already running (pgrep -f 'gz sim'); stop it or unset CFR_USE_RUNNING_SIM"
    fi

    echo "starting simulation stack (log: $LAUNCH_LOG)"
    # CFR_SENSORS=1 renders the ZED so the env can build observations from
    # its point cloud (config.yaml scan_source: cloud). Costs real-time
    # factor, so it is opt-in.
    setsid ros2 launch cfr_arduino_bridge training.launch.py \
        sensors:="$([ -n "${CFR_SENSORS}" ] && echo true || echo false)" > "$LAUNCH_LOG" 2>&1 &
    SIM_PID=$!
    trap stop_sim EXIT INT TERM

    echo -n "waiting for teleport API and pose bridge"
    # Rendering the ZED adds world rewriting, ogre2 startup and the first
    # render pass before anything answers, which overruns a 60 s budget on a
    # busy machine.
    local wait_s=60
    [ -n "${CFR_SENSORS}" ] && wait_s=180
    local deadline=$((SECONDS + wait_s))
    while [ $SECONDS -lt $deadline ]; do
        if (exec 3<>/dev/tcp/localhost/9003) 2>/dev/null; then
            exec 3>&-
            if ros2 topic list 2>/dev/null | grep -q "dynamic_pose/info"; then
                echo " ready"
                return 0
            fi
        fi
        kill -0 "$SIM_PID" 2>/dev/null || { echo; die "launch exited early; see $LAUNCH_LOG"; }
        echo -n "."
        sleep 1
    done
    echo
    die "simulation not ready after ${wait_s} s; see $LAUNCH_LOG"
}

stop_sim() {
    trap - EXIT INT TERM
    if [ -n "$SIM_PID" ] && kill -0 "$SIM_PID" 2>/dev/null; then
        echo "stopping simulation stack"
        # ros2 launch forwards SIGINT to its children for a clean shutdown;
        # negative pid signals the whole setsid process group.
        kill -INT -- "-$SIM_PID" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "$SIM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$SIM_PID" 2>/dev/null || true
    fi
    # gz sim server sometimes survives its parent, and a stale one poisons
    # the next run's topics.
    pkill -f "gz sim" 2>/dev/null || true
}

setup_environment
if [ "${CFR_USE_RUNNING_SIM:-0}" = "1" ]; then
    echo "CFR_USE_RUNNING_SIM=1: reusing the already-running simulation"
else
    start_sim
fi
