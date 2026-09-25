"""tui.py's view of an rl/obstacleRacer run, read from its history.json.

The obstacle racer trains in numpy, on the host or anywhere, with no Gazebo,
no Docker container and no chunk logs, and it writes every evaluation as a
record in <run>/history.json (see rl/obstacleRacer/train.py).  So this reads
that file directly instead of parsing log lines, and checks liveness from the
file's age and from the trainer's pid rather than from `docker exec`.

The verdict is progress.py's, fed the held-out layouts' mean progress in
meters from the start box: the four layouts the policy never trains on are
the honest measure of whether it is still learning to drive.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import progress


def is_racer_run(run_dir: Path) -> bool:
    path = run_dir / "history.json"
    if not path.exists():
        return False
    try:
        records = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return bool(records) and "heldout" in records[0]


def _alive(pid):
    if not pid:
        return None
    try:
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return code.value == 259  # STILL_ACTIVE
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def render(run_dir: Path, ui, patience, min_evals, threshold) -> str:
    """One frame. `ui` is tui.py's module, for its colors and drawing helpers."""
    records = json.loads((run_dir / "history.json").read_text())
    last = records[-1]
    lines = []
    lines.append(
        f"{ui.BOLD}Obstacle racer training{ui.RESET}  {ui.DIM}{time.strftime('%H:%M:%S')}{ui.RESET}"
    )
    lines.append(f"run: {run_dir.name}")
    lines.append("")

    age = time.time() - (run_dir / "history.json").stat().st_mtime
    alive = _alive(last.get("pid"))
    finished = (
        any(r.get("steps") == r.get("target") for r in records[-1:])
        and len(records) > 1
    )
    if alive:
        state = ui.color("running", ui.GREEN)
    elif alive is False:
        state = ui.color(
            "finished" if finished else "not running", ui.DIM if finished else ui.RED
        )
    else:
        state = ui.color("unknown", ui.YELLOW)
    lines.append(f"trainer: {state}   last eval {ui.format_age(int(age))} ago")

    steps, target = last["steps"], last.get("target") or last["steps"]
    lines.append(f"steps   {ui.bar(steps / max(target, 1))}  {steps:,} / {target:,}")
    if len(records) >= 2 and records[-1]["wall"] > records[0]["wall"]:
        rate = (records[-1]["steps"] - records[0]["steps"]) / (
            records[-1]["wall"] - records[0]["wall"]
        )
        eta = (target - steps) / rate if rate > 0 else None
        lines.append(
            f"rate    {rate:,.0f} steps/s   eta {ui.format_age(int(eta)) if eta else '?'}"
        )
    lines.append("")

    def series(key, part):
        return [r[part][key] for r in records if r.get(part)]

    def fmt_lap(value):
        return f"{value:5.1f} s" if value else "   -   "

    for part, title in (("heldout", "held-out"), ("train", "training")):
        rec = last[part]
        finish = series("finish", part)
        prog = series("progress", part)
        lines.append(
            f"{ui.BOLD}{title:9s}{ui.RESET} finish {100 * rec['finish']:5.1f}%  "
            f"lap {fmt_lap(rec['lap_time'])} (best {fmt_lap(rec['best_lap'])})  "
            f"progress {100 * rec['progress']:5.1f}% ({rec['progress_m']:.1f} m)  "
            f"hoops {rec['hoops']:.2f}/3"
        )
        lines.append(f"          finish   {ui.sparkline(finish)}")
        lines.append(f"          progress {ui.sparkline(prog)}")
    laps = [r["heldout"]["lap_time"] for r in records if r["heldout"].get("lap_time")]
    if laps:
        lines.append(
            f"          held-out lap time {ui.sparkline([-x for x in laps])} (higher bar = faster)"
        )
    lines.append("")

    lines.append(f"{ui.BOLD}how held-out runs end{ui.RESET}")
    for cause, count in list(last["heldout"]["ends"].items())[:8]:
        lines.append(f"  {count:4d}  {cause}")
    lines.append(
        "  by layout: "
        + "  ".join(
            f"{seed}:{100 * v:.0f}%"
            for seed, v in last["heldout"]["finish_by_seed"].items()
        )
    )
    rollout = last.get("rollout") or {}
    if rollout.get("ep_rew_mean") is not None:
        lines.append(
            f"  training rollouts since last eval: ep_rew_mean {rollout['ep_rew_mean']:.1f}, "
            f"{rollout['outcomes']}"
        )
    lines.append("")

    # Per obstacle: held-out clear rate when dealt just before it, and how
    # often training met it and failed there since the last eval.
    sections = last["heldout"].get("sections") or {}
    zones = rollout.get("zones") or {}
    if sections:
        lines.append(
            f"{ui.BOLD}obstacles{ui.RESET}        held-out clear   trend        "
            "training met / failed"
        )
        for name, rate in sections.items():
            trend = [
                r["heldout"]["sections"].get(name, 0.0)
                for r in records
                if r["heldout"].get("sections")
            ]
            met, failed = zones.get(name, [0, 0])
            tint = ui.GREEN if rate >= 0.8 else ui.YELLOW if rate >= 0.4 else ui.RED
            lines.append(
                f"  {name:15s} {ui.color(f'{100 * rate:5.0f}%', tint)}          "
                f"{ui.sparkline(trend[-12:]):12s} {met:7d} / {failed:<6d}"
            )
        lines.append("")

    att = last["train"].get("attitude_deg") or {}
    shown = [
        z
        for z in (
            "overpass_ramp",
            "helical_ramp",
            "banked_turn",
            "potholes",
            "gravel_pit",
        )
        if z in att
    ]
    if shown:
        lines.append(
            f"{ui.BOLD}max body attitude by section (training layouts){ui.RESET}"
        )
        lines.append(
            "  "
            + "  ".join(
                f"{z}: pitch {att[z][0]:.1f} roll {att[z][1]:.1f}" for z in shown
            )
        )
        lines.append("")

    evals = [
        {"steps": r["steps"], "distance_m": r["heldout"]["progress_m"]} for r in records
    ]
    v = progress.verdict(evals, patience, min_evals, threshold)
    tint = {"improving": ui.GREEN, "plateau": ui.YELLOW, "regressing": ui.RED}.get(
        v["verdict"], ui.DIM
    )
    lines.append(f"verdict: {ui.color(v['verdict'], tint)} -- {v['reason']}")
    return "\n".join(lines)
