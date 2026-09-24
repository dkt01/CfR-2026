#!/usr/bin/env bash
#
# Put ROS 2 and the workspace into the current shell:
#
#     source ~/cfr/jetson/scripts/setEnv.sh
#
# It has to be SOURCED, not run: a script that is run gets its own shell, and
# the environment it sets up disappears when that shell exits.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "error: source this script rather than running it:" >&2
    echo "       source $0" >&2
    exit 1
fi

source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
