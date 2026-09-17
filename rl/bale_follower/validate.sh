#!/usr/bin/env bash
# Watch the trained policy drive: opens three terminal windows that together
# replace the manual three-terminal dance from the README --
#
#   1. the full simulation stack (simulation.launch.py, cameras + viewer bridges)
#   2. the glue simulation.launch.py lacks: the ground-truth pose bridge the
#      policy needs, plus disabling path_follower's idle zero-publishing so it
#      stops fighting the policy for /cmd_vel
#   3. the policy itself, at the demo speed cap
#
#   ./validate.sh                                   # RL policy, models/rl_straight_boost
#   DEMO_PROGRAM=path_racer.py ./validate.sh        # planned-line MPC tracker instead
#   ./validate.sh --straight-speed 5.0              # extra args pass to the chosen driver
#   DEMO_CHECKPOINT=checkpoints_v6/best_model.zip ./validate.sh
#
# 1.5 m/s is the measured best deterministic configuration for the v3
# checkpoint (see REPORT.md) -- the policy's raw mean action floors the
# throttle, and the cap is what keeps the demo on the track.
#
# Close the terminal windows (or Ctrl-C in them) to stop; window 1 owns the
# Gazebo server, so closing it takes the whole demo down.

# No -u: ROS 2's setup.bash trips over unbound variables under nounset.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORLD=cfr_speed_course
POSE_TOPIC="/world/$WORLD/dynamic_pose/info"
# models/ rather than checkpoints_v7/: checkpoint directories are training
# scratch and get overwritten by the next run into the same name. This is the
# kept copy.
CHECKPOINT="${DEMO_CHECKPOINT:-$SCRIPT_DIR/models/rl_straight_boost/best_model.zip}"
# The RL policy is the default. It is slower round a lap than path_racer.py
# (~40 s vs ~32 s) but measurably better driving: 0.092 m mean cross-track
# against the tracker's 0.127-0.147, never outside the 0.325 m corridor
# margin, and it does it on a 6 m forward scan with no prior map. It also
# carries its own speed envelope -- 3.9 m/s on the straights, 2.0 through the
# hairpins -- so the "RL is the slow fallback" tradeoff no longer holds.
# Set DEMO_PROGRAM=path_racer.py for the planned-line tracker.
PROGRAM="${DEMO_PROGRAM:-run_policy.py}"

die() { echo "error: $*" >&2; exit 1; }

source_ros() {
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    # shellcheck disable=SC1091
    source "$REPO_ROOT/install/setup.bash"
}

wait_for_pose_topic() {
    echo -n "waiting for $POSE_TOPIC"
    for _ in $(seq 1 60); do
        if ros2 topic list 2>/dev/null | grep -q "dynamic_pose/info"; then
            echo " ready"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo
    die "simulation did not come up within 60 s"
}

case "${1:-}" in
_sim)
    source_ros
    # websocket:=true starts the gzweb WebSocket server on port 9002 -- the
    # browser viewer shows "Simulation disconnected" without it.
    exec ros2 launch cfr_arduino_bridge simulation.launch.py websocket:=true
    ;;
_glue)
    source_ros
    # The pose bridge must exist before waiting on the ROS side of the topic,
    # since ros2 topic list only shows bridged topics.
    echo "starting ground-truth pose bridge"
    ros2 run ros_gz_bridge parameter_bridge \
        "$POSE_TOPIC@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V" &
    BRIDGE_PID=$!
    trap 'kill $BRIDGE_PID 2>/dev/null' EXIT
    wait_for_pose_topic
    echo "disabling path_follower idle publishing (retrying until the node is up)"
    for _ in $(seq 1 30); do
        if ros2 param set /path_follower keep_auto_active_when_idle false 2>/dev/null; then
            echo "path_follower silenced; leave this window open (it owns the pose bridge)"
            break
        fi
        sleep 2
    done
    wait $BRIDGE_PID
    ;;
_policy)
    shift
    source_ros
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/bin/activate"
    wait_for_pose_topic
    if [ "$PROGRAM" = "path_racer.py" ]; then
        echo "starting path racer (planned line from course_path.json)"
        exec python "$SCRIPT_DIR/path_racer.py" "$@"
    fi
    # No --max-speed override: v6 carries its trained cap (2.0 m/s) in its
    # metadata. The old 1.5 override existed because v3's deterministic mean
    # action floored the throttle -- a symptom of the broken vehicle model,
    # fixed at the source.
    echo "starting policy: $CHECKPOINT"
    exec python "$SCRIPT_DIR/run_policy.py" --checkpoint "$CHECKPOINT" "$@"
    ;;
""|-*)
    [ -f "$REPO_ROOT/install/setup.bash" ] || die "workspace not built: run 'colcon build' in $REPO_ROOT"
    [ -f "$SCRIPT_DIR/.venv/bin/activate" ] || die "no venv: see README.md Setup"
    [ -f "$CHECKPOINT" ] || die "checkpoint not found: $CHECKPOINT"
    # Absolute path, and re-exported inside the spawned command below:
    # gnome-terminal windows get their environment from the terminal server
    # process, not from this shell, so the variable would otherwise be lost.
    CHECKPOINT="$(readlink -f "$CHECKPOINT")"
    if pgrep -f "gz sim" > /dev/null; then
        die "a Gazebo server is already running; stop it first (duplicate servers corrupt each other's topics)"
    fi

    TERM_CMD=""
    for candidate in gnome-terminal x-terminal-emulator konsole xterm; do
        command -v "$candidate" > /dev/null && { TERM_CMD=$candidate; break; }
    done
    [ -n "$TERM_CMD" ] || die "no terminal emulator found (tried gnome-terminal, x-terminal-emulator, konsole, xterm)"

    open_terminal() { # title, role, extra args...
        local title=$1; shift
        # env -u: snap-app environments (VS Code's integrated terminal) leak
        # GTK_PATH/LD_LIBRARY_PATH pointing into the snap, which crashes
        # gnome-terminal with a GLIBC symbol lookup error. The child shells
        # rebuild what they need by sourcing ROS themselves.
        local clean_env=(env -u GTK_PATH -u GDK_PIXBUF_MODULE_FILE -u LD_LIBRARY_PATH)
        local inner="DEMO_CHECKPOINT=\"$CHECKPOINT\" DEMO_PROGRAM=\"$PROGRAM\" \"$SCRIPT_DIR/validate.sh\" $*"
        case "$TERM_CMD" in
        gnome-terminal)
            "${clean_env[@]}" gnome-terminal --title="$title" -- bash -c "$inner; echo; echo '[exited -- press enter to close]'; read" ;;
        konsole)
            "${clean_env[@]}" konsole --title "$title" -e bash -c "$inner; read" & ;;
        *)
            "${clean_env[@]}" "$TERM_CMD" -T "$title" -e bash -c "$inner; read" & ;;
        esac
    }

    echo "opening 3 terminals: simulation, glue (pose bridge + param), policy"
    open_terminal "1: simulation stack" _sim
    sleep 2
    open_terminal "2: pose bridge + path_follower param" _glue
    open_terminal "3: RL policy" _policy "$@"
    echo
    echo "watch it at http://localhost:5173 (start it with: cd web/gzweb-viewer && npm run dev)"
    echo "stop everything by closing window 1 (it owns the Gazebo server)"
    ;;
*)
    die "unknown argument '$1' (internal roles are _sim/_glue/_policy; extra run_policy.py args only work after './validate.sh')"
    ;;
esac
