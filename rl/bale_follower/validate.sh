#!/usr/bin/env bash
# Watch the car drive: opens three terminal windows that together replace the
# manual three-terminal dance from the README --
#
#   1. the full simulation stack (simulation.launch.py, cameras + viewer bridges)
#   2. the glue simulation.launch.py lacks: the ground-truth pose bridge the
#      driver needs, plus disabling path_follower's idle zero-publishing so it
#      stops fighting the driver for /cmd_vel
#   3. the driver itself
#
#   ./validate.sh                                   # medley_racer.py (default)
#   DEMO_PROGRAM=path_racer.py ./validate.sh        # planned-line MPC tracker
#   DEMO_PROGRAM=run_policy.py ./validate.sh        # the trained RL policy
#   ./validate.sh --debug                           # extra args pass to the chosen driver
#   DEMO_RVIZ=1 ./validate.sh                       # + rviz2 (car, bales, ZED cloud);
#                                                   #   also turns on sensors:=true, which the
#                                                   #   cloud needs. Prefix LIBGL_ALWAYS_SOFTWARE=1
#                                                   #   if this machine has no GPU.
#   DEMO_CHECKPOINT=checkpoints_v6/best_model.zip DEMO_PROGRAM=run_policy.py ./validate.sh
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
# medley_racer.py is the default: strict corridor following + pure pursuit +
# a CasADi speed/steering coupling, replanned from a live scan every tick with
# no offline plan and no trained checkpoint to load. It needs neither
# course_path.json (which path_racer.py depends on, and which does not
# regenerate reproducibly -- see its --traction help) nor a policy zip.
#
# The other two remain available:
#   DEMO_PROGRAM=path_racer.py   planned racing line from course_path.json
#   DEMO_PROGRAM=run_policy.py   the trained RL policy, needs DEMO_CHECKPOINT
PROGRAM="${DEMO_PROGRAM:-medley_racer.py}"

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

# _policy used to start publishing /cmd_vel as soon as the pose topic
# existed, with nothing checking whether _glue's own silencing of
# path_follower (config/arduino_bridge.yaml ships
# keep_auto_active_when_idle: true, so it publishes idle zero-Twists on
# /cmd_vel by default) had actually landed yet -- the two run in separate
# terminals with no synchronisation beyond a `sleep 2` between windows 1 and
# 2. A driver that starts commanding low speeds right into that window reads
# as "something is pulling the robot backwards": path_follower's competing
# zero and the driver's own command interleave on the same topic. Waiting
# for the parameter to actually read back false removes the race instead of
# hoping the window-open order covers it.
wait_for_path_follower_silenced() {
    echo -n "waiting for path_follower to be silenced"
    for _ in $(seq 1 60); do
        # Match loosely ("false" anywhere in the output) rather than the
        # exact "Boolean value is: False" the CLI prints today -- that
        # wording is a `ros2 param get` implementation detail, not a
        # contract, and a false negative here just means the driver never
        # starts rather than starting into the race this exists to avoid.
        if ros2 param get /path_follower keep_auto_active_when_idle 2>/dev/null | grep -qi false; then
            echo " ready"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo
    die "path_follower never reported keep_auto_active_when_idle=false -- is window 2 (the glue terminal) still up?"
}

case "${1:-}" in
_sim)
    source_ros
    # websocket:=true starts the gzweb WebSocket server on port 9002 -- the
    # browser viewer shows "Simulation disconnected" without it.
    #
    # sensors:=true only when RViz is up: it is what renders the ZED, and
    # without it /zed/zed_node/point_cloud/cloud_registered has no publisher
    # at all, so an RViz point cloud display would just sit empty. It costs a
    # render context (GPU, or LIBGL_ALWAYS_SOFTWARE=1 for llvmpipe at a few
    # Hz), which is why it is not on by default.
    sim_args=(websocket:=true)
    [ -n "${DEMO_RVIZ:-}" ] && sim_args+=(sensors:=true)
    exec ros2 launch cfr_arduino_bridge simulation.launch.py "${sim_args[@]}"
    ;;
_rviz)
    source_ros
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/bin/activate"
    wait_for_pose_topic
    # The bridge supplies what RViz cannot get on its own here: a TF tree
    # (the bridged Pose_V carries empty frame ids -- ros_gz#172/#410), a
    # chassis marker (there is no URDF in this repo, so RobotModel has
    # nothing to load), and the cloud's own frame, read off the first cloud
    # that arrives rather than guessed.
    python "$SCRIPT_DIR/rviz_bridge.py" &
    BRIDGE_PID=$!
    trap 'kill $BRIDGE_PID 2>/dev/null' EXIT
    exec rviz2 -d "$SCRIPT_DIR/sim.rviz"
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
    wait_for_path_follower_silenced
    if [ "$PROGRAM" = "path_racer.py" ]; then
        echo "starting path racer (planned line from course_path.json)"
        exec python "$SCRIPT_DIR/path_racer.py" "$@"
    fi
    if [ "$PROGRAM" = "medley_racer.py" ]; then
        # No --checkpoint: this one has no offline plan or trained policy to
        # load, only live follow-the-gap + pure pursuit + CasADi -- see its
        # own docstring. Silently falling through to run_policy.py here (as
        # this dispatch used to do for any PROGRAM other than path_racer.py)
        # is what made an earlier medley_racer.py tuning pass look like it
        # had no effect: the RL policy was running instead, unlabelled.
        echo "starting medley racer (follow-the-gap + pure pursuit + CasADi, no offline plan)"
        exec python "$SCRIPT_DIR/medley_racer.py" "$@"
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
    [ -f "$SCRIPT_DIR/$PROGRAM" ] || die "no such driver: $SCRIPT_DIR/$PROGRAM (set DEMO_PROGRAM to one of medley_racer.py, path_racer.py, run_policy.py)"
    # Only run_policy.py loads a checkpoint; demanding one for the others
    # would fail a default run over a file it never opens.
    if [ "$PROGRAM" = "run_policy.py" ]; then
        [ -f "$CHECKPOINT" ] || die "checkpoint not found: $CHECKPOINT"
        # Absolute path, and re-exported inside the spawned command below:
        # gnome-terminal windows get their environment from the terminal
        # server process, not from this shell, so it would otherwise be lost.
        CHECKPOINT="$(readlink -f "$CHECKPOINT")"
    fi
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
        local inner="DEMO_CHECKPOINT=\"$CHECKPOINT\" DEMO_PROGRAM=\"$PROGRAM\" DEMO_RVIZ=\"${DEMO_RVIZ:-}\" LIBGL_ALWAYS_SOFTWARE=\"${LIBGL_ALWAYS_SOFTWARE:-}\" \"$SCRIPT_DIR/validate.sh\" $*"
        case "$TERM_CMD" in
        gnome-terminal)
            "${clean_env[@]}" gnome-terminal --title="$title" -- bash -c "$inner; echo; echo '[exited -- press enter to close]'; read" ;;
        konsole)
            "${clean_env[@]}" konsole --title "$title" -e bash -c "$inner; read" & ;;
        *)
            "${clean_env[@]}" "$TERM_CMD" -T "$title" -e bash -c "$inner; read" & ;;
        esac
    }

    if [ -n "${DEMO_RVIZ:-}" ]; then
        echo "opening 4 terminals: simulation (sensors:=true for the ZED), glue, $PROGRAM, rviz2"
    else
        echo "opening 3 terminals: simulation, glue (pose bridge + param), $PROGRAM"
    fi
    open_terminal "1: simulation stack" _sim
    sleep 2
    open_terminal "2: pose bridge + path_follower param" _glue
    open_terminal "3: $PROGRAM" _policy "$@"
    [ -n "${DEMO_RVIZ:-}" ] && open_terminal "4: rviz2" _rviz
    echo
    echo "watch it at http://localhost:5173 (start it with: cd web/gzweb-viewer && npm run dev)"
    echo "stop everything by closing window 1 (it owns the Gazebo server)"
    ;;
*)
    die "unknown argument '$1' (internal roles are _sim/_glue/_policy; extra driver args only work after './validate.sh')"
    ;;
esac
