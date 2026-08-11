#!/usr/bin/env bash
#
# Stop ROS software started on the Jetson, including orphaned launchers.

set -euo pipefail

readonly SELF_PID=$$
readonly PARENT_PID=$PPID
readonly PROCESS_PATTERN='(^|/)(ros2|_ros2_daemon)( |$)|/opt/ros/|/ros2_ws/install/|ros2 launch|launch(PathFollowingTUI)?[.]sh|launch_zed[.]sh|arduino_bridge_node|cmd_vel_to_drive_node|path_follower_node|zed_wrapper|rosboard|path_tui[.]py|fake_arduino[.]py'

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage: $(basename "$0")

Stops ROS launchers and nodes started on this Jetson. Sends SIGTERM first,
then SIGKILL after five seconds if a process remains.
EOF
    exit 0
fi

find_ros_pids() {
    ps -eo pid=,args= | awk -v pattern="$PROCESS_PATTERN" \
        -v self="$SELF_PID" -v parent="$PARENT_PID" \
        '$0 !~ /awk -v pattern=/ && $0 !~ /ps -eo/ &&
         $0 ~ pattern && $1 != self && $1 != parent { print $1 }'
}

stop_pids() {
    local signal="$1"
    shift
    local pid
    for pid in "$@"; do
        kill "-$signal" "$pid" 2>/dev/null || true
    done
}

mapfile -t ROS_PIDS < <(find_ros_pids)

if [[ ${#ROS_PIDS[@]} -eq 0 ]]; then
    echo "no ROS software found"
    exit 0
fi

echo "stopping ROS processes: ${ROS_PIDS[*]}"
stop_pids TERM "${ROS_PIDS[@]}"

for _ in {1..20}; do
    mapfile -t ROS_PIDS < <(find_ros_pids)
    [[ ${#ROS_PIDS[@]} -eq 0 ]] && echo "ROS software stopped" && exit 0
    sleep 0.25
done

echo "warning: forcing remaining ROS processes: ${ROS_PIDS[*]}" >&2
stop_pids KILL "${ROS_PIDS[@]}"
echo "ROS software stopped"
