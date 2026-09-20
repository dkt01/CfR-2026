#!/usr/bin/env bash
# Run the corridor validation against a simulator that is already up.
#
# validate_corridor.py imports ObstacleCourseEnv, which imports rclpy, so it
# needs the same three-layer environment train_resilient_obstacle.sh sets up
# before it runs anything: ROS itself, this workspace's overlay (for the
# cfr_arduino_bridge messages and services), and the training venv last so
# its numpy/torch win over the system ones.
#
#   docker exec cfr-rl-obstacle /repo/rl/bale_follower/validate_corridor.sh --samples 14
#
# Start the simulator first -- any of the obstacle-course launches will do:
#
#   ros2 launch cfr_arduino_bridge obstacle_course.launch.py sensors:=true autonomy:=false
#
# sensors:=true is not optional: the corridor estimate is computed from the
# ZED's point cloud, and without rendering there is no cloud to compute from.

# No -u: ROS 2's setup.bash trips over unbound variables under nounset.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "$REPO_ROOT/install/setup.bash"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"

exec python "$SCRIPT_DIR/validate_corridor.py" "$@"
