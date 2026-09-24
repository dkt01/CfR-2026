#!/usr/bin/env bash
#
# Sync jetson/ to the Orin and build it there, chaining what's otherwise two
# separate manual steps (remembering syncSoftware.sh's flags, then knowing to
# follow up with --build/--test). Deliberately does NOT launch anything on
# the car -- see SKILL.md for why that step stays manual/confirmed.
#
# Usage:
#   deploy.sh sync [--host HOST] [--test] [--dry-run]
#   deploy.sh launch-cmd [launch.sh args...]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
SYNC_SCRIPT="$REPO_DIR/jetson/scripts/syncSoftware.sh"

log() { echo "[deploy.sh] $*"; }

cmd_sync() {
    local host="" do_test=false extra=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --host) host="$2"; shift 2 ;;
            --test) do_test=true; shift ;;
            --dry-run) extra+=(--dry-run); shift ;;
            *) echo "unknown argument: $1" >&2; exit 1 ;;
        esac
    done

    # --host is easy to pass as a bare hostname/IP (the sync target, e.g.
    # "192.168.0.167" copied from a ping/nmap result) without the "user@"
    # SSH needs. That used to fail the reachability check below while
    # blaming power/network -- ssh was actually trying to auth as whatever
    # the local shell's username is. Default the user the same way
    # syncSoftware.sh's own ORIN_HOST does.
    local default_user="${ORIN_HOST:-tejam@192.168.55.1}"
    default_user="${default_user%%@*}"
    if [ -n "$host" ] && [[ "$host" != *@* ]]; then
        log "no user in --host '$host' -- assuming '$default_user' (pass user@host to override)"
        host="${default_user}@${host}"
    fi

    local args=(--build)
    [ "$do_test" = true ] && args=(--test)
    [ -n "$host" ] && args+=(--host "$host")
    args+=("${extra[@]}")

    # syncSoftware.sh's own ssh calls have no connect timeout, so without
    # this check an unreachable Orin (not on this network, USB-Ethernet not
    # plugged in) hangs for however long the OS's default TCP connect
    # timeout is -- tens of seconds to a couple minutes -- before failing.
    local check_host="${host:-${ORIN_HOST:-tejam@192.168.55.1}}"
    log "checking Orin reachability ($check_host)"
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$check_host" true 2>/dev/null; then
        echo "[deploy.sh] can't reach $check_host over ssh -- either it's not powered on/connected (USB-Ethernet or Wi-Fi), or the user/host is wrong (expected user@host, e.g. tejam@192.168.0.167; got '$check_host')" >&2
        exit 1
    fi

    log "syncing and building on the Orin (${args[*]})"
    "$SYNC_SCRIPT" "${args[@]}"
}

cmd_launch_cmd() {
    local host="${ORIN_HOST:-tejam@192.168.55.1}"
    log "this only prints the command -- run it yourself once you're at the bench"
    log "with the E-stop remote in hand:"
    echo
    echo "  ssh -t $host './software/scripts/launch.sh $*'"
}

case "${1:-}" in
    sync) shift; cmd_sync "$@" ;;
    launch-cmd) shift; cmd_launch_cmd "$@" ;;
    *)
        echo "usage: deploy.sh sync [--host HOST] [--test] [--dry-run]" >&2
        echo "       deploy.sh launch-cmd [launch.sh args...]" >&2
        exit 1
        ;;
esac
