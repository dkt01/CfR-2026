#!/usr/bin/env bash
#
# Pull characterization run directories off the Orin.
#
# Runs are recorded locally on the car because Wi-Fi drops out at range; this is
# how they get back to a laptop afterwards, when there is a network again.
# Mirrors syncSoftware.sh's host/dir conventions and its rsync-optional
# behaviour, so it works from Windows git bash too.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${ORIN_HOST:-orin.local}"
REMOTE_DIR="${ORIN_RUNS:-~/cfr_runs}"
LOCAL_DIR="${CFR_RUNS_LOCAL:-${SCRIPT_DIR}/../../runs}"
DELETE_REMOTE=0
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: sync_runs.sh [options]

  --host HOST     Orin hostname (default $ORIN_HOST, else orin.local)
  --remote DIR    Remote run directory (default $ORIN_RUNS, else ~/cfr_runs)
  --local DIR     Local destination (default $CFR_RUNS_LOCAL, else <repo>/runs)
  --list          List runs on the Orin and exit
  --purge         Delete runs from the Orin after a verified copy
  --dry-run       Show what would transfer
  -h, --help      This message

Runs are never deleted from the Orin unless --purge is given, and --purge only
removes a run after rsync reports the copy succeeded. A characterization run is
one trip to a car park; losing one to a careless flag is not worth the disk.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --remote) REMOTE_DIR="$2"; shift 2 ;;
        --local) LOCAL_DIR="$2"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        --purge) DELETE_REMOTE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "${LIST_ONLY:-}" ]]; then
    echo "runs on ${HOST}:${REMOTE_DIR}"
    ssh "$HOST" "ls -1 ${REMOTE_DIR} 2>/dev/null || echo '(none)'"
    exit 0
fi

mkdir -p "$LOCAL_DIR"
echo "pulling ${HOST}:${REMOTE_DIR}/ -> ${LOCAL_DIR}/"

if command -v rsync >/dev/null 2>&1; then
    RSYNC_ARGS=(-az --partial --info=progress2)
    [[ $DRY_RUN -eq 1 ]] && RSYNC_ARGS+=(--dry-run)
    rsync "${RSYNC_ARGS[@]}" "${HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
else
    # syncSoftware.sh carries the same fallback: a typical Windows git bash has
    # ssh and tar but no rsync.
    echo "rsync not found, falling back to tar over ssh"
    if [[ $DRY_RUN -eq 1 ]]; then
        ssh "$HOST" "ls -1 ${REMOTE_DIR}"
        exit 0
    fi
    ssh "$HOST" "tar -C ${REMOTE_DIR} -czf - ." | tar -C "$LOCAL_DIR" -xzf -
fi

if [[ $DRY_RUN -eq 1 ]]; then
    exit 0
fi

echo
echo "local runs:"
ls -1 "$LOCAL_DIR"

if [[ $DELETE_REMOTE -eq 1 ]]; then
    echo
    read -r -p "delete these runs from ${HOST}? [y/N] " reply
    if [[ "$reply" == "y" || "$reply" == "Y" ]]; then
        ssh "$HOST" "rm -rf ${REMOTE_DIR:?}/*"
        echo "removed from the Orin"
    else
        echo "left alone"
    fi
fi

cat <<NEXT

Analyse a run with:
  ./scripts/analyze_run.py ${LOCAL_DIR}/<run> --mass <kg>
NEXT
