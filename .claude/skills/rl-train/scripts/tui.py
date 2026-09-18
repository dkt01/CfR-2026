#!/usr/bin/env python3
"""Full-screen, auto-refreshing dashboard for an RL training run.

`progress.py` answers "is this run still learning" once and exits; this
redraws that same picture every few seconds plus two things `progress.py`
does not show: a step-count gauge against the run's own target (from
`train_resilient.log`'s "target N steps" line and the checkpoint mtimes,
not a wall-clock guess), and a trend line across every run this project has
ever kept a `best_model.json` for -- `checkpoints_*/` (in progress or
abandoned) and `models/*/` (promoted) -- so a plateau in the current run
reads as "the ceiling" or "worse than what we already have" rather than in
isolation.

    python3 tui.py --dir rl/bale_follower/checkpoints_v10
    python3 tui.py --dir ... --interval 10
    python3 tui.py --dir ... --once        # one frame, no loop (for logging)

Ctrl+C to exit. Reuses progress.py's eval-parsing and verdict logic directly
rather than re-implementing it, so the two never disagree.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress  # noqa: E402  (needs sys.path set first)

CONTAINER = "cfr-rl"
CLEAR = "\x1b[2J\x1b[H"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
SPARK_LEVELS = "▁▂▃▄▅▆▇█"


def color(text, code):
    return f"{code}{text}{RESET}"


def sparkline(values):
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    return "".join(SPARK_LEVELS[min(7, int((v - lo) / span * 7))] for v in values)


def bar(fraction, width=30):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {fraction * 100:5.1f}%"


def docker_state(container):
    """Live process/server state, read straight from docker -- not a log guess."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"container": None, "training": None, "gz_servers": None}
    if result.returncode != 0 or result.stdout.strip() != "true":
        return {"container": False, "training": False, "gz_servers": None}
    try:
        training = (
            subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "bash",
                    "-c",
                    "pgrep -f '[t]rain_resilient.sh|[t]rain.py|[t]rain_curriculum.sh' "
                    ">/dev/null 2>&1",
                ],
                timeout=5,
            ).returncode
            == 0
        )
        gz_raw = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "bash",
                "-c",
                "ps -eo cmd | grep -c '^gz sim -r -s' || true",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        gz_servers = int(gz_raw) if gz_raw.isdigit() else None
    except (OSError, subprocess.TimeoutExpired):
        training, gz_servers = None, None
    return {"container": True, "training": training, "gz_servers": gz_servers}


def step_rate(checkpoint_dir):
    """Steps/second inferred from the two most recent step checkpoints on disk."""
    files = sorted(
        checkpoint_dir.glob("bale_follower_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(files) < 2:
        return None
    a, b = files[-2], files[-1]
    match_a, match_b = (
        progress.STEPS_RE.search(a.name),
        progress.STEPS_RE.search(b.name),
    )
    if not match_a or not match_b:
        return None
    dt = b.stat().st_mtime - a.stat().st_mtime
    if dt <= 0:
        return None
    return (int(match_b.group(1)) - int(match_a.group(1))) / dt


def run_target_steps(log_dir):
    """The cumulative step count train_resilient.sh is running toward.

    Logged once at start ("target N steps into DIR"); unlike CHUNK_RE's
    per-chunk "/target" it exists from the first second, before any chunk
    has finished or died.
    """
    resilient_log = log_dir / "train_resilient.log"
    if not resilient_log.exists():
        return None
    for line in resilient_log.read_text(errors="replace").splitlines():
        if "target" in line and "steps into" in line:
            try:
                return int(line.split("target", 1)[1].split("steps", 1)[0].strip())
            except ValueError:
                continue
    return None


def current_cumulative_steps(log_dir, state):
    """Best current estimate of total steps taken, on the same scale as the target.

    Checkpoints save far more often than deterministic evals (every ~1000
    steps vs. every ~8000), so the newest checkpoint's in-chunk step count
    plus that chunk's starting offset tracks closer to "right now" than the
    last eval line does.
    """
    offsets, _ = progress.chunk_offsets(log_dir / "train_resilient.log")
    chunk_logs = sorted(
        log_dir.glob("train_chunk_*.log"),
        key=lambda p: int(progress.re.search(r"(\d+)", p.name).group(1)),
    )
    current_chunk = 1
    if chunk_logs:
        match = progress.re.search(r"(\d+)", chunk_logs[-1].name)
        if match:
            current_chunk = int(match.group(1))
    base = offsets.get(current_chunk, max(offsets.values(), default=0))
    in_chunk = state.get("newest_in_chunk_steps") or 0
    return base + in_chunk


def discover_runs(root):
    """Every best_model.json this project has kept, oldest first.

    checkpoints_*/ covers in-progress and abandoned runs; models/*/ covers
    ones promoted as a named milestone. A run only appears once it has
    produced at least one deterministic eval.
    """
    runs = []
    for pattern in ("checkpoints*/best_model.json", "models/*/best_model.json"):
        for path in root.glob(pattern):
            try:
                data = progress.json.loads(path.read_text())
                mtime = path.stat().st_mtime
            except (OSError, progress.json.JSONDecodeError):
                continue
            runs.append(
                {
                    "name": path.parent.name,
                    "path": path,
                    "mtime": mtime,
                    "distance_m": data.get("deterministic_distance_m"),
                    "at_timesteps": data.get("at_timesteps"),
                    "total_timesteps": data.get("total_timesteps"),
                    "max_speed": data.get("env", {}).get("max_speed"),
                }
            )
    runs.sort(key=lambda item: item["mtime"])
    return runs


def format_age(seconds):
    if seconds is None:
        return "?"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds / 3600:.1f}h"


def render(checkpoint_dir, log_dir, root, container, patience, min_evals, threshold):
    lines = []
    now = time.strftime("%H:%M:%S")
    lines.append(f"{BOLD}RL training dashboard{RESET}  {DIM}{now}{RESET}")
    lines.append(f"run: {checkpoint_dir.name}")
    lines.append("")

    docker = docker_state(container)
    if docker["container"] is None:
        status_line = f"{YELLOW}docker unreachable{RESET}"
    elif not docker["container"]:
        status_line = f"{DIM}container '{container}' not running{RESET}"
    else:
        train_txt = (
            color("running", GREEN)
            if docker["training"]
            else color("not running", RED)
            if docker["training"] is False
            else color("unknown", YELLOW)
        )
        gz = docker["gz_servers"]
        gz_txt = f"{gz} gz server(s)"
        if isinstance(gz, int) and gz > 1:
            gz_txt = color(f"{gz} gz servers -- duplicates corrupt the run", RED)
        status_line = f"training: {train_txt}   {gz_txt}"
    lines.append(status_line)
    lines.append("")

    evals, notes = progress.collect_evals(log_dir)
    state = progress.checkpoint_state(checkpoint_dir)
    target = run_target_steps(log_dir)
    current = current_cumulative_steps(log_dir, state)
    rate = step_rate(checkpoint_dir)

    if target:
        lines.append(f"progress  {bar(current / target)}  ({current}/{target} steps)")
        if rate and current < target:
            eta_s = int((target - current) / rate)
            lines.append(
                f"          {DIM}~{rate * 60:.0f} steps/min, "
                f"ETA {format_age(eta_s)}{RESET}"
            )
    else:
        lines.append(
            f"progress  {DIM}no target logged yet ({current} steps so far){RESET}"
        )
    lines.append("")

    if evals:
        recent = evals[-20:]
        spark = sparkline([item["distance_m"] for item in recent])
        lines.append(
            f"deterministic eval distance (last {len(recent)}): {color(spark, GREEN)}"
        )
        lines.append(
            f"  latest {evals[-1]['distance_m']:.1f} m @ {evals[-1]['steps']} steps"
        )
        if state.get("best_distance_m") is not None:
            lines.append(
                f"  best   {state['best_distance_m']:.1f} m @ "
                f"{state.get('best_at_timesteps')} in-chunk steps "
                f"({format_age(state.get('best_age_s'))} ago)"
            )
        slope = progress.slope_per_10k(evals)
        if slope is not None:
            slope_color = GREEN if slope > 0 else RED
            lines.append(
                f"  slope  {color(f'{slope:+.2f} m / 10k steps', slope_color)}"
            )
        result = progress.verdict(evals, patience, min_evals, threshold)
        verdict_color = {
            "improving": GREEN,
            "too-early": DIM,
            "plateau": YELLOW,
            "regressing": RED,
        }.get(result["verdict"], RESET)
        lines.append(
            f"  verdict {color(result['verdict'].upper(), verdict_color)} -- {result['reason']}"
        )
    else:
        lines.append(
            f"{DIM}no deterministic eval lines yet "
            f"(~8k steps, roughly 15 min at real-time factor 1.0){RESET}"
        )
    for note in notes:
        lines.append(f"  {color('restart: ' + note, YELLOW)}")
    lines.append("")

    runs = discover_runs(root)
    lines.append(f"{BOLD}trend across runs{RESET} ({len(runs)} with a best_model.json)")
    if runs:
        distances = [r["distance_m"] for r in runs if r["distance_m"] is not None]
        if len(distances) > 1:
            lines.append(f"  {color(sparkline(distances), GREEN)}")
        for run in runs[-8:]:
            marker = " <- current" if run["path"].parent == checkpoint_dir else ""
            dist = (
                f"{run['distance_m']:.1f} m" if run["distance_m"] is not None else "?"
            )
            lines.append(
                f"  {time.strftime('%m-%d', time.localtime(run['mtime']))}  "
                f"{run['name']:<28} {dist:>10}  "
                f"@{run.get('at_timesteps') or '?'} steps  "
                f"max_speed {run.get('max_speed')}{marker}"
            )
        if len(distances) > 1:
            delta = distances[-1] - distances[0]
            trend_color = GREEN if delta > 0 else RED if delta < 0 else DIM
            lines.append(
                f"  {color(f'{delta:+.1f} m first-to-last across {len(distances)} runs', trend_color)}"
            )
    else:
        lines.append(f"  {DIM}no runs with a saved best_model.json yet{RESET}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir", required=True, help="checkpoint directory of the run to watch"
    )
    parser.add_argument(
        "--logs",
        default=None,
        help="directory holding train_resilient.log and train_chunk_*.log "
        "(defaults to the checkpoint directory's parent)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="directory to scan for cross-run history "
        "(defaults to the checkpoint directory's parent)",
    )
    parser.add_argument("--container", default=CONTAINER)
    parser.add_argument(
        "--interval", type=float, default=5.0, help="seconds between redraws"
    )
    parser.add_argument(
        "--once", action="store_true", help="render a single frame and exit"
    )
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-evals", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()

    checkpoint_dir = Path(args.dir).resolve()
    log_dir = Path(args.logs).resolve() if args.logs else checkpoint_dir.parent
    root = Path(args.root).resolve() if args.root else checkpoint_dir.parent

    try:
        while True:
            frame = render(
                checkpoint_dir,
                log_dir,
                root,
                args.container,
                args.patience,
                args.min_evals,
                args.threshold,
            )
            if args.once:
                print(frame)
                return
            print(
                CLEAR
                + frame
                + f"\n\n{DIM}refreshing every {args.interval:.0f}s -- Ctrl+C to exit{RESET}"
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
