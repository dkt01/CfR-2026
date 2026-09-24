#!/usr/bin/env python3
"""Print the part of a processed run that answers "how did it go".

    python summarize_run.py runs/<run>                 # verdicts, KPIs, laps, events
    python summarize_run.py runs/<run> --sections      # also the per-section table
    python summarize_run.py runs/<run> --logs WARN     # also /rosout at WARN and above
    python summarize_run.py runs/<run1> runs/<run2>    # several runs, one after another

Reads <run>/analysis/summary.json (and logs.json for --logs), which
web/run-lab/server/analyze.py writes.  Standard library only, so it runs from
any Python -- no Run Lab venv needed just to read results.

summary.json is ~hundreds of KB of nested stats; dumping it into context to
find the verdict wastes most of it.  This prints the same things the Run
Lab's Overview page leads with, each with the timeline second it happened at
so a follow-up can jump into series.json or Rerun at that moment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "FATAL": 50}


def fmt(value, unit=""):
    if value is None:
        return "-"
    if isinstance(value, float):
        value = f"{value:.3g}" if abs(value) < 1000 else f"{value:.0f}"
    return f"{value} {unit}".strip()


def at(t):
    return f"@{t:7.2f}s" if isinstance(t, (int, float)) else " " * 9


def summarize(run: Path, sections: bool, log_level: str | None, max_events: int):
    analysis = run / "analysis"
    path = analysis / "summary.json"
    if not path.exists():
        print(f"== {run.name}: not processed (no {path})")
        print(
            f"   process it: <run-lab venv python> web/run-lab/server/analyze.py {run}"
        )
        return 1
    s = json.loads(path.read_text())
    meta = s.get("meta", {})
    md = meta.get("metadata") or {}

    print(f"== {meta.get('run', run.name)}")
    bits = [
        f"kind={md.get('kind', '?')}",
        f"driver={meta.get('driver') or md.get('driver') or '?'}",
        f"sim={meta.get('simulation')}",
        f"speed_scale={md.get('speed_scale') or '-'}",
        f"analysis v{meta.get('version')}",
    ]
    if md.get("notes"):
        bits.append(f"notes={md['notes']!r}")
    print("   " + "  ".join(bits))
    frame = s.get("frame") or {}
    print(
        f"   track frame: {frame.get('method')}  fit_rms={fmt(frame.get('fit_rms_m'), 'm')}"
        f"  anchors={frame.get('anchors')}"
    )

    print("\n  KPIs")
    for k in s.get("kpis", []):
        tone = f"  [{k['tone']}]" if k.get("tone") else ""
        print(
            f"    {k.get('label', k.get('key')):<18} {fmt(k.get('value'), k.get('unit', ''))}{tone}"
        )

    verdicts = s.get("verdicts") or {}
    for heading, key in (("Worked", "worked"), ("Failed", "failed")):
        items = verdicts.get(key) or []
        print(f"\n  {heading} ({len(items)})")
        for v in items:
            print(f"    {at(v.get('t'))}  {v.get('title')}")
            if v.get("detail"):
                print(f"               {v['detail']}")

    laps = s.get("laps") or {}
    if laps.get("laps"):
        print(
            f"\n  Laps  {laps.get('completed')}/{laps.get('target') or '?'}"
            f"  finished={laps.get('finished')}  race_time={fmt(laps.get('race_time'), 's')}"
        )
        for lap in laps["laps"]:
            print(
                f"    lap {lap.get('lap')}: {fmt(lap.get('time'), 's'):>8}"
                f"  max {fmt(lap.get('max_speed'), 'm/s')}"
                f"  min_clear {fmt(lap.get('min_clearance'), 'm')}"
                f"  mean|cte| {fmt(lap.get('mean_abs_cte'), 'm')}"
            )

    events = s.get("events") or []
    if events:
        # An E-stop chattering or a link flapping produces dozens of identical
        # events back to back; one line with a count and time span reads better.
        runs = []
        for e in events:
            key = (e.get("kind"), e.get("severity"), e.get("text"))
            if runs and runs[-1][0] == key:
                runs[-1][2] = e
                runs[-1][3] += 1
            else:
                runs.append([key, e, e, 1])
        shown = runs[:max_events]
        more = f", first {max_events} groups" if len(runs) > max_events else ""
        print(f"\n  Events ({len(events)}{more})")
        for _key, first, last, count in shown:
            st = f" st={first['station']}m" if first.get("station") is not None else ""
            rep = f"  x{count} until {last.get('t')}s" if count > 1 else ""
            print(
                f"    {at(first.get('t'))}  {first.get('severity', ''):<5} "
                f"{first.get('kind', ''):<12} {first.get('text', '')}{st}{rep}"
            )

    if sections:
        rows = s.get("sections") or []
        print(f"\n  Sections ({len(rows)})")
        for row in rows:
            a = row.get("all") or {}
            name = row.get("name") or row.get("kind") or row.get("id") or "?"
            print(
                f"    {name:<22} min_clear {fmt(a.get('min_clearance'), 'm'):>8}"
                f"  max|cte| {fmt(a.get('max_abs_cte'), 'm'):>8}"
                f"  speed {fmt(a.get('min_speed'))}-{fmt(a.get('max_speed'), 'm/s')}"
                f"  over_cap {fmt(a.get('max_over_cap'), 'm/s')}"
            )

    if log_level:
        logs_path = analysis / "logs.json"
        floor = LEVELS.get(log_level.upper(), 30)
        logs = json.loads(logs_path.read_text()) if logs_path.exists() else []
        hits = [line for line in logs if _level(line.get("level")) >= floor]
        print(f"\n  /rosout >= {log_level.upper()} ({len(hits)})")
        for line in hits[:200]:
            print(
                f"    {at(line.get('t'))}  {line.get('node', '')}: {line.get('msg', '')}"
            )
    print()
    return 0


def _level(value):
    if isinstance(value, (int, float)):
        return int(value)
    return LEVELS.get(str(value).upper(), 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--sections", action="store_true")
    ap.add_argument("--logs", metavar="LEVEL", help="DEBUG|INFO|WARN|ERROR|FATAL")
    ap.add_argument("--max-events", type=int, default=40)
    a = ap.parse_args()
    worst = 0
    for run in a.runs:
        worst = max(worst, summarize(run, a.sections, a.logs, a.max_events))
    return worst


if __name__ == "__main__":
    sys.exit(main())
