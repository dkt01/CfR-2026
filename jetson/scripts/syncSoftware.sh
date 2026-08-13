#!/usr/bin/env bash
#
# Sync the Jetson ROS 2 packages from a development host to the Orin.
#
# Source only: the host is x86_64 and the Orin is aarch64, so build artifacts
# are never transferred.  Use --build to compile on the Orin after syncing.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_DIR="$(dirname "${SCRIPT_DIR}")"

REMOTE_HOST="${ORIN_HOST:-tejam@192.168.55.1}"
REMOTE_DIR="${ORIN_DIR:-~/software}"
REMOTE_WS="${ORIN_WS:-~/ros2_ws}"
ROS_DISTRO_NAME="${ORIN_ROS_DISTRO:-jazzy}"

DRY_RUN=false
DELETE=false
DO_BUILD=false
DO_TEST=false

# Build artifacts and editor droppings never belong on the robot.
readonly EXCLUDES=(
  '.git/'
  'build/'
  'install/'
  'log/'
  'logs/'
  'bin/'
  'lib/'
  '__pycache__/'
  '*.pyc'
  '*.swp'
  '*~'
  '.DS_Store'
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Syncs $(basename "${SOURCE_DIR}")/ to ${REMOTE_HOST}:${REMOTE_DIR}

Options:
  -H, --host HOST   Orin ssh host or alias   (env ORIN_HOST, default: tejam@192.168.55.1)
  -d, --dir DIR     Destination directory    (env ORIN_DIR, default: ~/software)
  -w, --ws DIR      colcon workspace on Orin (env ORIN_WS, default: ~/ros2_ws)
  -n, --dry-run     Show what would transfer without changing anything
      --delete      Remove files on the Orin that no longer exist locally
  -b, --build       Run colcon build on the Orin after syncing
  -t, --test        Run colcon test on the Orin after building (implies --build)
  -h, --help        This message

Examples:
  $(basename "$0") --dry-run
  $(basename "$0") --host orin.local --build
  ORIN_HOST=tejam@192.168.55.1 $(basename "$0") --build --test
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -H | --host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    -d | --dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    -w | --ws)
      REMOTE_WS="$2"
      shift 2
      ;;
    -n | --dry-run)
      DRY_RUN=true
      shift
      ;;
    --delete)
      DELETE=true
      shift
      ;;
    -b | --build)
      DO_BUILD=true
      shift
      ;;
    -t | --test)
      DO_BUILD=true
      DO_TEST=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

HAS_RSYNC=false
if command -v rsync >/dev/null; then
  HAS_RSYNC=true
elif ! command -v scp >/dev/null; then
  echo "error: neither rsync nor scp is installed on this host" >&2
  exit 1
fi

if [[ "${HAS_RSYNC}" != true && "${DELETE}" == true ]]; then
  echo "error: --delete requires rsync; scp fallback cannot remove remote files" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}/cfr_arduino_bridge" ]]; then
  echo "error: ${SOURCE_DIR} does not look like the jetson source directory" >&2
  exit 1
fi

rsync_args=(--archive --compress --human-readable --itemize-changes)
for pattern in "${EXCLUDES[@]}"; do
  rsync_args+=(--exclude "${pattern}")
done

if [[ "${DRY_RUN}" == true ]]; then
  rsync_args+=(--dry-run)
  echo "== dry run, nothing will be written =="
fi

if [[ "${DELETE}" == true ]]; then
  rsync_args+=(--delete)
  echo "== --delete: files under ${REMOTE_DIR} with no local counterpart will be removed =="
fi

echo "syncing ${SOURCE_DIR}/ -> ${REMOTE_HOST}:${REMOTE_DIR}/"

# Trailing slashes matter: copy the contents of jetson/, not the directory.
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"
if [[ "${HAS_RSYNC}" == true ]]; then
  rsync "${rsync_args[@]}" "${SOURCE_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
else
  echo "rsync unavailable; using scp fallback"
  while IFS= read -r source_file; do
    relative_file="${source_file#"${SOURCE_DIR}/"}"
    remote_file="${REMOTE_DIR}/${relative_file}"
    if [[ "${DRY_RUN}" == true ]]; then
      printf 'would copy %s -> %s:%s\n' "${source_file}" "${REMOTE_HOST}" "${remote_file}"
    else
      remote_directory="${REMOTE_DIR}/$(dirname "${relative_file}")"
      ssh "${REMOTE_HOST}" "mkdir -p ${remote_directory}"
      scp "${source_file}" "${REMOTE_HOST}:${remote_file}"
    fi
  done < <(find "${SOURCE_DIR}" -type f \
    ! -path "${SOURCE_DIR}/.git/*" \
    ! -path "${SOURCE_DIR}/build/*" \
    ! -path "${SOURCE_DIR}/install/*" \
    ! -path "${SOURCE_DIR}/log/*" \
    ! -path "${SOURCE_DIR}/logs/*" \
    ! -path "${SOURCE_DIR}/bin/*" \
    ! -path "${SOURCE_DIR}/lib/*" \
    ! -path '*/__pycache__/*' \
    ! -name '*.pyc' \
    ! -name '*.swp' \
    ! -name '*~' \
    ! -name '.DS_Store' \
    -print)
fi

if [[ "${DRY_RUN}" == true ]]; then
  echo "dry run complete"
  exit 0
fi

echo "sync complete"

if [[ "${DO_BUILD}" != true ]]; then
  echo
  echo "to build on the Orin:"
  echo "  ssh ${REMOTE_HOST} 'source /opt/ros/${ROS_DISTRO_NAME}/setup.bash &&" \
    "cd ${REMOTE_WS} && colcon build --base-paths ${REMOTE_DIR}'"
  exit 0
fi

echo
echo "building on ${REMOTE_HOST} in ${REMOTE_WS}"
ssh "${REMOTE_HOST}" "bash -lc '
  set -eo pipefail
  source /opt/ros/${ROS_DISTRO_NAME}/setup.bash
  set -u
  mkdir -p ${REMOTE_WS}
  cd ${REMOTE_WS}
  # The Jetson clock can lag files synced from the development host.  A clean
  # package build prevents Make from retaining an older installed binary when
  # source timestamps appear to be in the future.
  rm -rf build/cfr_interfaces install/cfr_interfaces build/cfr_arduino_bridge install/cfr_arduino_bridge
  colcon build --base-paths ${REMOTE_DIR} --cmake-args -DCMAKE_BUILD_TYPE=Release
'"
echo "build complete"

if [[ "${DO_TEST}" == true ]]; then
  echo
  echo "testing on ${REMOTE_HOST}"
  ssh "${REMOTE_HOST}" "bash -lc '
    set -eo pipefail
    source /opt/ros/${ROS_DISTRO_NAME}/setup.bash
    set -u
    cd ${REMOTE_WS}
    colcon test --base-paths ${REMOTE_DIR} --packages-select cfr_arduino_bridge
    colcon test-result --verbose
  '"
  echo "tests complete"
fi

echo
echo "on the Orin:"
echo "  source ${REMOTE_WS}/install/setup.bash"
echo "  ros2 launch cfr_arduino_bridge arduino_bridge.launch.py"
