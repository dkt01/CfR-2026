#!/usr/bin/env bash
#
# Calibrate where the ZED sits on the car (height, pitch, roll; x, y, yaw with
# targets) from a parked capture, and optionally apply it to this car.
# Nothing here arms the actuators: it needs the ZED only.
#
#   ~/software/scripts/launch.sh --no-bridge          # terminal 1: ZED only
#   ~/software/scripts/launchCalibration.sh           # terminal 2: floor + IMU
#   ~/software/scripts/launchCalibration.sh --target 2.0,0 --target 3.0,0.6
#   ~/software/scripts/launchCalibration.sh --tape 0.47,0,0.19 --from rear-axle
#
# See calibrate_camera.py for what each input measures.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
VEHICLE="$(dirname "${SCRIPT_DIR}")/cfr_arduino_bridge/config/vehicle.yaml"
APPLY=ask
YES=false
SKIP_CHECKS=false
PASS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

  --target X,Y        box front-face center, vehicle frame (repeatable; two or
                      more also measure yaw)
  --tape X,Y,Z        mounting hole measured by hand
  --from midpoint|rear-axle
                      what --target/--tape x is measured from (default midpoint)
  --frames N          depth frames to average (default 30)
  --label NAME        appended to the run directory name
  --apply / --no-apply
                      write the result into this car's vehicle.yaml without
                      asking / never (default: ask)
  --skip-checks       skip the topic checks
  -y, --yes           do not wait for Enter before capturing
  -h, --help          this message

Setup: the car on flat, level floor (check with a level), 0.5-4 m of open
floor ahead, nobody touching it.  Targets are boxes at least 0.3 m tall, faces
square to the car, at least 1 m apart sideways; give the center of each front
face.  The vehicle frame is the ground under the WHEELBASE MIDPOINT, x forward,
y left; tape from the rear axle and pass --from rear-axle if that is easier.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target | --tape | --from | --frames | --label) PASS+=("$1" "$2"); shift 2 ;;
    --apply) APPLY=yes; shift ;;
    --no-apply) APPLY=no; shift ;;
    --skip-checks) SKIP_CHECKS=true; shift ;;
    -y | --yes) YES=true; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

# ROS's setup scripts read unset variables.
set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
# shellcheck disable=SC1091
source "${ROS2_WS}/install/setup.bash" 2>/dev/null || true
set -u

if [[ "${SKIP_CHECKS}" != true ]]; then
  echo "preflight..."
  fail=false
  for topic in /zed/zed_node/depth/depth_registered /zed/zed_node/depth/camera_info /zed/zed_node/imu/data; do
    if timeout 8 ros2 topic echo "${topic}" --once --field header >/dev/null 2>&1; then
      echo "  ok    ${topic}"
    else
      echo "  FAIL  no ${topic}"
      fail=true
    fi
  done
  if [[ "${fail}" == true ]]; then
    echo "start the ZED first: ${SCRIPT_DIR}/launch.sh --no-bridge"
    exit 1
  fi
fi

echo
echo "  Car on flat, level floor, open floor 0.5-4 m ahead, targets placed,"
echo "  hands off.  The capture takes ~3 s."
if [[ "${YES}" != true ]]; then
  read -r -p "  Press Enter to capture: " _ </dev/tty
fi

out="$(mktemp)"
trap 'rm -f "${out}"' EXIT
python3 "${SCRIPT_DIR}/calibrate_camera.py" --vehicle "${VEHICLE}" "${PASS[@]}" | tee "${out}"
run_dir="$(sed -n 's/^run directory: //p' "${out}")"
[[ -n "${run_dir}" ]] || exit 1

if [[ "${APPLY}" == ask ]]; then
  read -r -p "Apply to this car's vehicle.yaml (${VEHICLE})? [y/N] " answer </dev/tty
  [[ "${answer}" == [yY]* ]] && APPLY=yes || APPLY=no
fi
if [[ "${APPLY}" == yes ]]; then
  python3 "${SCRIPT_DIR}/apply_vehicle_patch.py" "${run_dir}" --vehicle "${VEHICLE}"
  echo "Drivers read it at startup: relaunch any that are running."
fi

cat <<EOF

To keep it (syncSoftware.sh overwrites this car's copy), on the laptop:
  jetson/scripts/sync_runs.sh
  jetson/scripts/apply_vehicle_patch.py runs/$(basename "${run_dir}")
  jetson/scripts/propagate_camera.py       # Gazebo's ZED and SubZero's pose offset
then commit vehicle.yaml.  Re-run this afterwards: every row should read ok.
EOF
