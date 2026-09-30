#!/usr/bin/env bash
# Bring up the Obstacle Course for gazebo_env.py: the course with the ZED
# rendered, no driver, and a bridge for the world's ControlWorld service so
# the trainer can step it.  Runs in the background inside a sim container;
# gazebo_vec.py calls it after a `docker restart`.
#
#   gazebo_stack.sh [--rtf 1.0]     # returns once the pose and service are up
#
# The world is stepped paused (multi_step per control period), so the
# real-time factor only caps how fast a step may run; sim_vehicle_node and
# the cloud pipeline run in wall time and need that slack.
set -euo pipefail
# CFR_REPO when run from a copy (gazebo_vec.py strips CRLF into /tmp first).
REPO=${CFR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RTF=1.0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rtf) RTF="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 1;;
  esac
done
WS="${ROS2_WS:-$HOME/ros2_ws}"
set +u
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
set -u
export LIBGL_ALWAYS_SOFTWARE=1

WORLD="$WS/install/cfr_arduino_bridge/share/cfr_arduino_bridge/worlds/obstacle_course.sdf"
sed "s|<real_time_factor>[^<]*</real_time_factor>|<real_time_factor>$RTF</real_time_factor>|" \
  "$REPO/jetson/cfr_arduino_bridge/worlds/obstacle_course.sdf" >"$WORLD"

setsid ros2 launch cfr_arduino_bridge obstacle_course.launch.py \
  sensors:=true path_follower:=false cmd_vel_to_drive:=false laps:=1 \
  >/tmp/gazebo_stack.log 2>&1 &
setsid ros2 run ros_gz_bridge parameter_bridge \
  /world/cfr_obstacle_course/control@ros_gz_interfaces/srv/ControlWorld \
  >/tmp/gazebo_bridge.log 2>&1 &

for _ in $(seq 120); do
  if timeout 5 ros2 topic echo /zed/zed_node/pose --once >/dev/null 2>&1 &&
    ros2 service list 2>/dev/null | grep -q /obstacle_randomizer/randomize &&
    ros2 service list 2>/dev/null | grep -q /world/cfr_obstacle_course/control; then
    # Nodes training never reads, each costing a core share: the C++
    # segmenter (gazebo_env segments the cloud itself, as the driver does),
    # the start light detector, and the lap and hoop judges (env.py judges).
    for node in cloud_segmentation_node start_signal_detector_node \
      lap_counter_node hoop_monitor_node; do
      pkill -f "lib/cfr_arduino_bridge/$node" || true
    done
    echo "stack up"
    exit 0
  fi
  sleep 2
done
echo "stack did not come up; tail of /tmp/gazebo_stack.log:"
tail -20 /tmp/gazebo_stack.log
exit 1
