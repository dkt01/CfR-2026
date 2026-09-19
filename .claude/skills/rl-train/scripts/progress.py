#!/usr/bin/env python3
"""Read a training run's logs and checkpoints and say whether it is still improving.

Training is paced against the wall clock at `control_hz` steps per second, so a
200k-step run costs five and a half hours at best. The question that matters
while it burns is not "is it alive" but "is it still learning" -- and the only
metric that answers that honestly is the deterministic evaluation distance
train.py prints between rollouts, because that is what deployment runs. The
stochastic training reward can climb while the mean action is degenerate (the
v3 run did exactly that), so this deliberately does not treat ep_rew_mean as
the progress signal; it reports it only as context.

    python3 progress.py --dir rl/bale_follower/checkpoints_v10
    python3 progress.py --dir ... --json

Always exits 0: the verdict is in the output, not the exit code, because
stopping a run early is a decision for a human.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

EVAL_RE = re.compile(
    r"\[deterministic eval\] (\d+) steps: mean ([\d.]+) m over (\d+) episodes"
)
CHUNK_RE = re.compile(
    r"chunk (\d+) (?:finished cleanly|died with status (\d+)) "
    r"\(\+(\d+) steps, (\d+)/(\d+)\)"
)
TARGET_RE = re.compile(r"target \d+ steps into (\S+)")
REW_RE = re.compile(r"\|\s+ep_rew_mean\s+\|\s+([-\d.e+]+)\s+\|")
STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")


def log_names(checkpoint_dir):
    """(resilient_log, chunk_glob, run_log) filenames for this run's course.

    rl/bale_follower holds every course's logs together, so the checkpoint
    dir -- not a --course flag -- is what tells us whether train_resilient.sh
    or train_resilient_obstacle.sh wrote them.
    """
    is_obstacle = "obstacle" in checkpoint_dir.name.lower() or any(
        checkpoint_dir.glob("obstacle_course_*_steps.zip")
    )
    if is_obstacle:
        return (
            "train_resilient_obstacle.log",
            "train_obstacle_chunk_*.log",
            "rl_run_obstacle.log",
        )
    return "train_resilient.log", "train_chunk_*.log", "rl_run.log"


def chunk_offsets(resilient_log, run_name):
    """Cumulative step offset at the start of each chunk, plus restart notes.

    SB3 restarts its own step counter on every resume, so a chunk's eval lines
    report in-chunk steps. train_resilient.sh logs the cumulative total after
    each chunk, and that is what makes the per-chunk numbers comparable.

    train_resilient.log and train_chunk_*.log are shared filenames across every
    run in the checkpoint parent dir -- a later run (or a restart of the same
    run with a different --total-timesteps) reuses "train_chunk_1.log" etc. So
    only lines logged after THIS run's own "target N steps into <run_name>"
    marker count; anything before it belongs to a previous run/attempt and
    must not leak into this run's offsets or restart notes.
    """
    offsets = {1: 0}
    notes = []
    if not resilient_log.exists():
        return offsets, notes
    lines = resilient_log.read_text(errors="replace").splitlines()
    start = 0
    for index, line in enumerate(lines):
        match = TARGET_RE.search(line)
        if match and match.group(1) == run_name:
            start = index
    for line in lines[start:]:
        match = CHUNK_RE.search(line)
        if not match:
            continue
        chunk, status, gained, cumulative, _target = match.groups()
        offsets[int(chunk) + 1] = int(cumulative)
        if status:
            notes.append(
                f"chunk {chunk} died with status {status} after +{gained} steps"
            )
    return offsets, notes


def collect_evals(log_dir, checkpoint_dir):
    resilient_log, chunk_glob, run_log = log_names(checkpoint_dir)
    offsets, notes = chunk_offsets(log_dir / resilient_log, checkpoint_dir.name)
    evals = []
    chunk_logs = sorted(
        (
            p
            for p in log_dir.glob(chunk_glob)
            if int(re.search(r"(\d+)", p.name).group(1)) in offsets
        ),
        key=lambda p: int(re.search(r"(\d+)", p.name).group(1)),
    )
    if not chunk_logs:
        chunk_logs = [p for p in (log_dir / run_log,) if p.exists()]
    for path in chunk_logs:
        match = re.search(r"chunk_(\d+)", path.name)
        chunk = int(match.group(1)) if match else 1
        offset = offsets.get(chunk, 0)
        for line in path.read_text(errors="replace").splitlines():
            found = EVAL_RE.search(line)
            if found:
                in_chunk, distance, episodes = found.groups()
                evals.append(
                    {
                        "chunk": chunk,
                        "steps": offset + int(in_chunk),
                        "distance_m": float(distance),
                        "episodes": int(episodes),
                    }
                )
    return evals, notes


def recent_rewards(log_dir, checkpoint_dir, valid_chunks, count=3):
    _, chunk_glob, run_log = log_names(checkpoint_dir)
    values = []
    logs = sorted(
        p
        for p in log_dir.glob(chunk_glob)
        if int(re.search(r"(\d+)", p.name).group(1)) in valid_chunks
    ) or list(log_dir.glob(run_log))
    for path in logs:
        values.extend(
            float(value) for value in REW_RE.findall(path.read_text(errors="replace"))
        )
    return values[-count:]


def slope_per_10k(evals, window=8):
    """Least-squares metres gained per 10k steps over the recent evals."""
    sample = evals[-window:]
    if len(sample) < 3:
        return None
    xs = [item["steps"] / 10000.0 for item in sample]
    ys = [item["distance_m"] for item in sample]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def verdict(evals, patience, min_evals, threshold):
    if len(evals) < min_evals:
        return {
            "verdict": "too-early",
            "reason": f"{len(evals)} deterministic evals so far; "
            f"need {min_evals} before a trend means anything",
        }

    distances = [item["distance_m"] for item in evals]
    best_index = max(range(len(distances)), key=lambda i: distances[i])
    since_best = len(distances) - 1 - best_index
    steps_since_best = evals[-1]["steps"] - evals[best_index]["steps"]

    window = max(3, patience // 2 + 1)
    recent = distances[-window:]
    previous = distances[-2 * window : -window]
    gain = None
    if previous:
        previous_mean = sum(previous) / len(previous)
        recent_mean = sum(recent) / len(recent)
        gain = (recent_mean - previous_mean) / max(previous_mean, 1e-6)

    common = {"evals_since_best": since_best, "steps_since_best": steps_since_best}
    if gain is not None and gain < -0.10:
        return {
            "verdict": "regressing",
            "reason": f"last {window} evals average {gain * 100:+.0f}% against the "
            f"{window} before them; best was {distances[best_index]:.1f} m at "
            f"{evals[best_index]['steps']} steps",
            **common,
        }
    if since_best >= patience and (gain is None or gain < threshold):
        drift = "unknown" if gain is None else f"{abs(gain) * 100:.1f}%"
        return {
            "verdict": "plateau",
            "reason": f"no new best in {since_best} evals ({steps_since_best} steps); "
            f"recent average within {drift} of the previous window",
            **common,
        }
    return {
        "verdict": "improving",
        "reason": f"best {distances[best_index]:.1f} m at "
        f"{evals[best_index]['steps']} steps, {since_best} evals ago",
        **common,
    }


def checkpoint_state(checkpoint_dir):
    state = {"dir": str(checkpoint_dir), "exists": checkpoint_dir.is_dir()}
    if not state["exists"]:
        return state
    steps_files = sorted(
        checkpoint_dir.glob("*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    if steps_files:
        newest = steps_files[-1]
        match = STEPS_RE.search(newest.name)
        state["newest_checkpoint"] = newest.name
        state["newest_in_chunk_steps"] = int(match.group(1)) if match else None
        state["age_s"] = round(time.time() - newest.stat().st_mtime)
    best_json = checkpoint_dir / "best_model.json"
    if best_json.exists():
        try:
            best = json.loads(best_json.read_text())
        except (json.JSONDecodeError, OSError):
            return state
        state["best_distance_m"] = best.get("deterministic_distance_m")
        state["best_at_timesteps"] = best.get("at_timesteps")
        state["best_max_speed"] = best.get("env", {}).get("max_speed")
        state["best_age_s"] = round(time.time() - best_json.stat().st_mtime)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="checkpoint directory")
    parser.add_argument(
        "--logs",
        default=None,
        help="directory holding train_resilient.log and train_chunk_*.log "
        "(defaults to the checkpoint directory's parent, which is where "
        "train_resilient.sh writes them)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="evals without a new best before calling plateau",
    )
    parser.add_argument("--min-evals", type=int, default=4)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="relative gain over the previous window that still counts as progress",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.dir).resolve()
    log_dir = Path(args.logs).resolve() if args.logs else checkpoint_dir.parent

    evals, notes = collect_evals(log_dir, checkpoint_dir)
    resilient_log_name, _, _ = log_names(checkpoint_dir)
    valid_chunks, _ = chunk_offsets(log_dir / resilient_log_name, checkpoint_dir.name)
    state = checkpoint_state(checkpoint_dir)
    result = {
        "checkpoints": state,
        "evals": evals,
        "restarts": notes,
        "recent_ep_rew_mean": recent_rewards(log_dir, checkpoint_dir, valid_chunks),
        "slope_m_per_10k_steps": slope_per_10k(evals),
    }
    result.update(verdict(evals, args.patience, args.min_evals, args.threshold))

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"checkpoints: {state.get('dir')}")
    if state.get("newest_checkpoint"):
        print(
            f"  newest      {state['newest_checkpoint']} "
            f"({state['age_s'] // 60} min ago)"
        )
    if state.get("best_distance_m") is not None:
        print(
            f"  best_model  {state['best_distance_m']:.1f} m at "
            f"{state['best_at_timesteps']} in-chunk steps "
            f"(max_speed {state.get('best_max_speed')}, "
            f"{state.get('best_age_s', 0) // 60} min ago)"
        )
    print()
    if evals:
        print("deterministic eval (mean distance over the callback's episodes):")
        previous = None
        for item in evals[-12:]:
            delta = (
                "" if previous is None else f"  {item['distance_m'] - previous:+.1f}"
            )
            print(f"  {item['steps']:>8} steps   {item['distance_m']:6.1f} m{delta}")
            previous = item["distance_m"]
    else:
        print("no deterministic eval lines yet -- the callback runs every 16 rollouts")
        print("(~8k steps, roughly 15 min of wall clock at real-time factor 1.0)")
    if result["slope_m_per_10k_steps"] is not None:
        print(f"\nrecent slope: {result['slope_m_per_10k_steps']:+.2f} m per 10k steps")
    if result["recent_ep_rew_mean"]:
        print(f"ep_rew_mean (context only): {result['recent_ep_rew_mean']}")
    for note in notes:
        print(f"restart: {note}")
    print(f"\nverdict: {result['verdict'].upper()} -- {result['reason']}")


if __name__ == "__main__":
    main()
