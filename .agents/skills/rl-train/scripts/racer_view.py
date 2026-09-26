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
    # `wall` restarts at 0 in each trainer process, so a resumed run's rate
    # must come only from the current process's records, not from step 0.
    seg = records[-1:]
    for r in reversed(records[:-1]):
        if r.get("pid") != seg[0].get("pid") or r["wall"] >= seg[0]["wall"]:
            break
        seg.insert(0, r)
    if len(seg) >= 2 and seg[-1]["wall"] > seg[0]["wall"]:
        rate = (seg[-1]["steps"] - seg[0]["steps"]) / (seg[-1]["wall"] - seg[0]["wall"])
        eta = (target - steps) / rate if rate > 0 else None
        lines.append(
            f"rate    {rate:,.0f} steps/s   eta {ui.format_age(int(eta)) if eta else '?'}"
        )
    lines.append("")

    def series(key, part):
        return [r[part][key] for r in records if r.get(part)]

    def series_opt(key, part):
        return [r[part][key] for r in records if r.get(part) and key in r[part]]

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
        # Runs before speed was logged have no avg_speed; show nothing for them.
        if rec.get("avg_speed") is not None:
            avg = series_opt("avg_speed", part)
            lines.append(
                f"          speed    {ui.sparkline(avg)}  avg {rec['avg_speed']:.2f} m/s "
                f"along the course, top {rec['top_speed']:.2f} m/s (mean per run)"
            )
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
    if rollout.get("stuck_starts"):
        trend = [
            r["rollout"]["stuck_recovered"]
            for r in records
            if (r.get("rollout") or {}).get("stuck_starts")
        ]
        lines.append(
            f"  stuck starts backed out and drove on: "
            f"{100 * rollout['stuck_recovered']:.0f}% of {rollout['stuck_starts']}  "
            f"{ui.sparkline(trend[-12:])}"
        )
    lines.append("")

    # Per obstacle: held-out clear rate when dealt just before it, and how
    # often training met it and failed there since the last eval.
    sections = last["heldout"].get("sections") or {}
    zones = rollout.get("zones") or {}
    practice = rollout.get("practice") or {}
    if sections:
        # Widths come from the longest name, so wide_open_region fits too.
        w = max(len("obstacles") - 2, *(len(n) for n in sections))
        lines.append(
            f"{ui.BOLD}{'obstacles':<{w + 2}}{ui.RESET} {'held-out clear':>14}  "
            f"{'trend':<12}  {'training met':>12} / {'failed':<6}  {'practice':>8}"
        )
        for row, (name, rate) in enumerate(sections.items()):
            trend = [
                r["heldout"]["sections"].get(name, 0.0)
                for r in records
                if r["heldout"].get("sections")
            ]
            met, failed = zones.get(name, [0, 0])
            tint = ui.GREEN if rate >= 0.8 else ui.YELLOW if rate >= 0.4 else ui.RED
            pct = f"{100 * rate:.0f}%"
            prac = f"{100 * practice[name]:.0f}%" if name in practice else ""
            # Alternate gray and white so neighboring rows' bars stay apart.
            spark = f"{ui.sparkline(trend[-12:], 0.0, 1.0):<12}"
            spark = ui.color(spark, ui.GRAY if row % 2 else ui.WHITE)
            lines.append(
                f"  {name:<{w}} {ui.color(f'{pct:>14}', tint)}  "
                f"{spark}  {met:>12} / {failed:<6}  "
                f"{prac:>8}"
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
