#!/usr/bin/env bash
# Live view of an obstacle-course training run.
#
# Training writes SB3's stdout tables to train_obstacle_chunk_N.log; there is
# no tensorboard. This pulls the most recent chunk log apart into the two
# things worth watching: the deterministic eval history (how far round the
# course a greedy policy actually gets, always from the start line) and the
# latest rollout table.
#
#   ./monitor_training.sh            refreshing single screen, 30 s
#   ./monitor_training.sh -n 10      refreshing single screen, 10 s
#   ./monitor_training.sh --follow   scrolling stream of tables as they land
#   ./monitor_training.sh --once     print once and exit (for scripts/logs)
#
# From outside the container:
#   docker exec -it cfr-rl-obstacle /repo/rl/bale_follower/monitor_training.sh
set -uo pipefail

LOG_GLOB="${LOG_GLOB:-/repo/rl/bale_follower/train_obstacle_chunk_*.log}"
INTERVAL=30
MODE=watch

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--interval) INTERVAL="$2"; shift 2 ;;
        -f|--follow)   MODE=follow; shift ;;
        -1|--once)     MODE=once; shift ;;
        -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Newest by mtime, not by number: train_resilient_obstacle.sh restarts chunk
# numbering on a fresh run, so the highest-numbered log can be an old one.
latest_log() {
    # shellcheck disable=SC2086  # deliberate glob expansion
    ls -t $LOG_GLOB 2>/dev/null | head -1
}

# The run's own start marker. Each resumed chunk appends to these logs, and
# older runs' output can still be above, so anchor on the last one.
render() {
    local log
    log="$(latest_log)"
    if [ -z "$log" ]; then
        echo "no training log matching $LOG_GLOB"
        return
    fi

    echo "=== $log"
    local checkpoints
    checkpoints="$(ls -t /repo/rl/bale_follower/checkpoints_obstacle_*/ -d 2>/dev/null | head -1)"
    [ -n "$checkpoints" ] && echo "=== $checkpoints"
    echo

    echo "--- deterministic evals (from the start line, greedy) ---"
    grep -h 'deterministic eval' "$log" | tail -12
    echo

    echo "--- latest rollout ---"
    # Tables are fenced by dashed rules. Keep the last *complete* block, so a
    # half-written table being flushed right now does not show up truncated.
    awk '
        /^-{10}/    { if (buf != "") last = buf; buf = ""; next }
        /^\|/       { buf = buf $0 "\n" }
        END         { printf "%s", (buf != "" ? buf : last) }
    ' "$log"
}

case "$MODE" in
    once)
        render
        ;;
    follow)
        log="$(latest_log)"
        [ -z "$log" ] && { echo "no training log matching $LOG_GLOB" >&2; exit 1; }
        echo "following $log (Ctrl-C to stop)"
        tail -n 200 -f "$log" \
            | grep --line-buffered -E '^\||^-{10}|deterministic eval'
        ;;
    watch)
        if command -v watch > /dev/null 2>&1 && [ -t 1 ]; then
            export LOG_GLOB
            exec watch -n "$INTERVAL" -t "$0" --once
        fi
        # No tty (piped, or watch missing): fall back to a plain loop.
        while true; do
            render
            sleep "$INTERVAL"
        done
        ;;
esac
