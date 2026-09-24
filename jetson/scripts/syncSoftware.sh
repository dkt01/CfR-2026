#!/usr/bin/env bash
#
# Sync the Jetson ROS 2 packages from a development host to the Orin, and with
# them the formulaOne driver and the one policy it races with.
#
# Source only: the host is x86_64 and the Orin is aarch64, so build artifacts
# are never transferred.  Use --build to compile on the Orin after syncing.
#
# Orin layout this produces (default --dir ~/software):
#
#   ~/software/                 jetson/ (the ROS packages and scripts)
#   ~/software/formulaOne/      rl/formulaOne/ code, plus the chosen policy's
#                               policy.npz and config.yaml at its top level
#   ~/jetson -> ~/software      formulaOne finds the course files and
#                               record_run.py under <two dirs up>/jetson/,
#                               as it does in the repo

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_DIR="$(dirname "${SCRIPT_DIR}")"
readonly F1_DIR="$(dirname "${SOURCE_DIR}")/rl/formulaOne"

REMOTE_HOST="${ORIN_HOST:-tejam@192.168.55.1}"
REMOTE_DIR="${ORIN_DIR:-~/software}"
REMOTE_WS="${ORIN_WS:-~/ros2_ws}"
ROS_DISTRO_NAME="${ORIN_ROS_DISTRO:-jazzy}"
# The policy that races.  Taken from rl/formulaOne/bestModel/<run>/, which is
# committed, or failing that from rl/formulaOne/runs/<run>/.
F1_RUN="${F1_RUN:-v12}"
SYNC_F1=true

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

# The scp fallback below has no rsync-style --exclude, so it has to prune the
# same list itself. Derive its find(1) predicate from EXCLUDES rather than
# hand-maintaining a second copy: rsync's directory patterns match at any
# depth, so a lone top-level "! -path SOURCE_DIR/build/*" would miss a build/
# nested under a package and silently sync it, unlike the rsync path.
FIND_PRUNE_ARGS=()
for pattern in "${EXCLUDES[@]}"; do
  if [[ "${pattern}" == */ ]]; then
    FIND_PRUNE_ARGS+=(! -path "*/${pattern%/}/*")
  else
    FIND_PRUNE_ARGS+=(! -name "${pattern}")
  fi
done
readonly FIND_PRUNE_ARGS

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Syncs $(basename "${SOURCE_DIR}")/ to ${REMOTE_HOST}:${REMOTE_DIR}

Options:
  -H, --host HOST   Orin ssh host or alias   (env ORIN_HOST, default: tejam@192.168.55.1)
  -d, --dir DIR     Destination directory    (env ORIN_DIR, default: ~/software)
  -w, --ws DIR      colcon workspace on Orin (env ORIN_WS, default: ~/ros2_ws)
  -p, --policy RUN  formulaOne policy to deploy (env F1_RUN, default: v12)
      --no-f1       Sync jetson/ only, not formulaOne or its policy
  -n, --dry-run    Show what would transfer without changing anything
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
    -p | --policy)
      F1_RUN="$2"
      shift 2
      ;;
    --no-f1)
      SYNC_F1=false
      shift
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

# The Orin has no fixed IP: it's reached over a direct USB-gadget link
# (192.168.55.1) or over Wi-Fi/LAN (something like 192.168.0.167) depending
# on how it's plugged in, and both addresses front the same host key. Without
# accept-new, connecting from a new address makes every ssh/scp/rsync call
# below fail outright with "Host key verification failed" -- and worse, the
# BatchMode probe just below reports that as "no public-key auth" and walks
# the user into an unnecessary password prompt that would not have fixed
# anything. accept-new still errors out if the host key ever actually
# *changes*, so this doesn't disable verification, just first-contact TOFU.
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

# ssh, scp, and rsync each open their own connection, so a host that only
# accepts a password (no key in ssh-agent) would otherwise ask for it once
# per remote command below. ControlMaster connection sharing would fix that
# on Linux/macOS, but it does not work over Git for Windows' bundled OpenSSH
# (session multiplexing fails there), which this script must also support.
# So instead: probe whether public-key auth alone gets in, and if not, read
# the password once here and hand it to every remote command via sshpass.
SSH_CMD=(ssh "${SSH_OPTS[@]}")
SCP_CMD=(scp "${SSH_OPTS[@]}")
RSYNC_RSH="ssh ${SSH_OPTS[*]}"
echo "checking SSH access to ${REMOTE_HOST}..."
if ! ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_HOST}" true >/dev/null 2>&1; then
  # The `&&` chain keeps this compatible with `set -e`: if sshpass is missing,
  # or there is no controlling terminal to prompt on (e.g. run from cron), the
  # read is skipped rather than aborting the script on a failed redirect.
  # Keep stderr visible: Bash writes read -p's prompt there.
  if command -v sshpass >/dev/null && read -rs -p "Password for ${REMOTE_HOST}: " SSH_PASSWORD </dev/tty; then
    echo
    export SSHPASS="${SSH_PASSWORD}"
    unset SSH_PASSWORD
    SSH_CMD=(sshpass -e ssh "${SSH_OPTS[@]}")
    SCP_CMD=(sshpass -e scp "${SSH_OPTS[@]}")
    RSYNC_RSH="sshpass -e ssh ${SSH_OPTS[*]}"
  else
    echo "note: public-key login to ${REMOTE_HOST} isn't set up (or no" >&2
    echo "      terminal is available to ask for the password once), so" >&2
    echo "      ssh/scp/rsync will each prompt separately. Install sshpass" >&2
    echo "      and run this interactively, or run ssh-copy-id" >&2
    echo "      ${REMOTE_HOST}, to be asked only once." >&2
  fi
fi

# The Orin has no RTC, so its clock resets to some fixed build date on every
# power cycle and only free-runs from there. rsync/scp compare timestamps to
# decide what changed, and a build clean-rebuilds two packages specifically
# because of clock skew (see the comment below) -- so fix the clock first.
LOCAL_EPOCH="$(date +%s)"
if [[ "${DRY_RUN}" == true ]]; then
  echo "would sync the Orin's clock to $(date -d "@${LOCAL_EPOCH}" 2>/dev/null || date -r "${LOCAL_EPOCH}")"
elif "${SSH_CMD[@]}" "${REMOTE_HOST}" "sudo -n date --set=@${LOCAL_EPOCH}" >/dev/null 2>&1; then
  echo "synced the Orin's clock to this host's time"
else
  echo "warning: couldn't set the Orin's clock (needs passwordless sudo for" >&2
  echo "         'date' on the Orin); if timestamps still look wrong, run:" >&2
  echo "           ssh ${REMOTE_HOST} 'sudo date --set=@${LOCAL_EPOCH}'" >&2
fi

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

# Resolve the policy before anything is transferred, so a typo in --policy
# fails here rather than after the ROS packages have already gone across.
if [[ "${SYNC_F1}" == true ]]; then
  F1_POLICY_DIR=""
  for candidate in "${F1_DIR}/bestModel/${F1_RUN}" "${F1_DIR}/runs/${F1_RUN}"; do
    if [[ -f "${candidate}/policy.npz" && -f "${candidate}/config.yaml" ]]; then
      F1_POLICY_DIR="${candidate}"
      break
    fi
  done
  if [[ -z "${F1_POLICY_DIR}" ]]; then
    echo "error: no policy.npz + config.yaml for '${F1_RUN}' under" >&2
    echo "       ${F1_DIR}/bestModel/ or ${F1_DIR}/runs/" >&2
    exit 1
  fi
  echo "formulaOne policy: ${F1_POLICY_DIR#"$(dirname "$(dirname "${F1_DIR}")")"/}"
fi

# What gets synced is exactly one NUL-delimited list of paths relative to
# the local tree, consumed identically by both the rsync and scp-fallback paths
# below, so they can never again disagree on what to exclude (see the git
# history of this file for the bug that caused). git already knows how to
# apply .gitignore, including nested ignore files like a tool's own
# .pytest_cache/.gitignore, so prefer asking it over re-deriving that logic;
# fall back to the hardcoded EXCLUDES list only when git isn't available.
FILE_LIST="$(mktemp)"
trap 'rm -f "${FILE_LIST}"' EXIT

list_files() {
  local tree="$1"
  if command -v git >/dev/null && git -C "${tree}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${tree}" ls-files -z --cached --others --exclude-standard -- .
  else
    echo "note: ${tree} isn't a git checkout (or git isn't installed), so" >&2
    echo "      .gitignore can't be consulted; falling back to this script's" >&2
    echo "      own hardcoded ignore list, which may sync extra cache/build" >&2
    echo "      files that .gitignore would otherwise catch." >&2
    find "${tree}" -type f "${FIND_PRUNE_ARGS[@]}" -printf '%P\0'
  fi
}

# Copies the files named in FILE_LIST from LOCAL_TREE to REMOTE_TREE.
sync_tree() {
  local tree="$1" remote_tree="$2"
  local rsync_args=(--archive --compress --human-readable --itemize-changes -e "${RSYNC_RSH}" --files-from="${FILE_LIST}" --from0)

  if [[ "${DRY_RUN}" == true ]]; then
    rsync_args+=(--dry-run)
  fi

  if [[ "${DELETE}" == true ]]; then
    rsync_args+=(--delete)
    echo "== --delete: files under ${remote_tree} with no local counterpart will be removed =="
  fi

  echo "syncing ${tree}/ -> ${REMOTE_HOST}:${remote_tree}/"

  # Trailing slashes matter: copy the contents of the tree, not the directory.
  if [[ "${DRY_RUN}" != true ]]; then
    "${SSH_CMD[@]}" "${REMOTE_HOST}" "mkdir -p ${remote_tree}"
  fi
  if [[ "${HAS_RSYNC}" == true ]]; then
    rsync "${rsync_args[@]}" "${tree}/" "${REMOTE_HOST}:${remote_tree}/"
  else
    echo "rsync unavailable; using scp fallback"
    local relative_file
    while IFS= read -r -d '' relative_file; do
      copy_file "${tree}/${relative_file}" "${remote_tree}/${relative_file}"
    done < "${FILE_LIST}"
  fi
}

# One file, to an exact remote path.  Used by the scp fallback and for the
# policy files, which land under a different name than they have locally.
copy_file() {
  local source_file="$1" remote_file="$2"
  if [[ "${DRY_RUN}" == true ]]; then
    printf 'would copy %s -> %s:%s\n' "${source_file}" "${REMOTE_HOST}" "${remote_file}"
    return
  fi
  # </dev/null: ssh/scp otherwise inherit the caller's stdin (FILE_LIST in
  # the scp fallback loop) and drain it, so only the first file would transfer.
  "${SSH_CMD[@]}" "${REMOTE_HOST}" "mkdir -p $(dirname "${remote_file}")" </dev/null
  if [[ "${HAS_RSYNC}" == true ]]; then
    rsync --compress --times --itemize-changes -e "${RSYNC_RSH}" "${source_file}" "${REMOTE_HOST}:${remote_file}" </dev/null
  else
    "${SCP_CMD[@]}" "${source_file}" "${REMOTE_HOST}:${remote_file}" </dev/null
  fi
}

if [[ "${DRY_RUN}" == true ]]; then
  echo "== dry run, nothing will be written =="
fi

list_files "${SOURCE_DIR}" >"${FILE_LIST}"
sync_tree "${SOURCE_DIR}" "${REMOTE_DIR}"

if [[ "${SYNC_F1}" == true ]]; then
  F1_REMOTE="${REMOTE_DIR}/formulaOne"
  # The driver's code, less what does not belong on the car: the top-level
  # config.yaml (the policy's own config replaces it just below -- a policy
  # must drive with the config it was trained under) and the other saved
  # models in bestModel/.
  list_files "${F1_DIR}" | grep -z -v -E '^(config\.yaml|bestModel/.*)$' >"${FILE_LIST}"
  sync_tree "${F1_DIR}" "${F1_REMOTE}"

  echo "policy ${F1_RUN} -> ${REMOTE_HOST}:${F1_REMOTE}/"
  copy_file "${F1_POLICY_DIR}/policy.npz" "${F1_REMOTE}/policy.npz"
  copy_file "${F1_POLICY_DIR}/config.yaml" "${F1_REMOTE}/config.yaml"

  # The track cache saves the Orin building a distance field over every bale
  # on first use.  Keyed by a hash of the world and track settings, so a stale
  # entry is ignored rather than used.  rsync only: it is ~40 MB.
  if [[ -d "${F1_DIR}/.cache" && "${HAS_RSYNC}" == true ]]; then
    echo "track cache -> ${REMOTE_HOST}:${F1_REMOTE}/.cache/"
    cache_args=(--archive --compress --human-readable -e "${RSYNC_RSH}")
    [[ "${DRY_RUN}" == true ]] && cache_args+=(--dry-run)
    rsync "${cache_args[@]}" "${F1_DIR}/.cache/" "${REMOTE_HOST}:${F1_REMOTE}/.cache/"
  fi

  # formulaOne resolves the course files and record_run.py as
  # <two dirs above itself>/jetson/..., which is the repo's layout.  Here that
  # is ~/jetson, so point it at the synced jetson/ tree.  Never replaces a
  # real directory of that name.
  link_parent="$(dirname "${REMOTE_DIR}")"
  if [[ "${DRY_RUN}" == true ]]; then
    echo "would link ${link_parent}/jetson -> ${REMOTE_DIR}"
  else
    "${SSH_CMD[@]}" "${REMOTE_HOST}" "
      link=${link_parent}/jetson
      if [ -e \"\$link\" ] && [ ! -L \"\$link\" ]; then
        echo \"warning: \$link exists and is not a symlink; formulaOne will not find the course\" >&2
      else
        ln -sfn ${REMOTE_DIR} \"\$link\" && echo \"linked \$link -> ${REMOTE_DIR}\"
      fi
    "
  fi
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
"${SSH_CMD[@]}" "${REMOTE_HOST}" "bash -lc '
  set -eo pipefail
  source /opt/ros/${ROS_DISTRO_NAME}/setup.bash
  set -u
  mkdir -p ${REMOTE_WS}
  cd ${REMOTE_WS}
  # Belt and suspenders alongside the clock sync above: if that did not set
  # the time (no passwordless sudo), stale timestamps could still make Make
  # retain an older installed binary, so force these two packages to rebuild.
  rm -rf build/cfr_interfaces install/cfr_interfaces build/cfr_arduino_bridge install/cfr_arduino_bridge
  colcon build --base-paths ${REMOTE_DIR} --cmake-args -DCMAKE_BUILD_TYPE=Release
'"
echo "build complete"

if [[ "${DO_TEST}" == true ]]; then
  echo
  echo "testing on ${REMOTE_HOST}"
  "${SSH_CMD[@]}" "${REMOTE_HOST}" "bash -lc '
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
