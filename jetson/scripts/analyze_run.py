#!/usr/bin/env python3
"""Analyse a characterization run directory.

    ./scripts/analyze_run.py ~/cfr_runs/20260915T141203Z_coastdown --mass 4.7

Writes report.md and vehicle_patch.yaml next to the run, plus any SVG plots.
Needs nothing but a stock Python 3 - no numpy, no ROS, no network - because the
laptop this gets run on may have none of them when it matters.

vehicle_patch.yaml is the point of the exercise: it is the set of measured
values this run justifies, ready to be folded into config/vehicle.yaml with
apply_vehicle_patch.py, which stamps each one with the run that produced it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from characterize import analyze as analysis  # noqa: E402
from characterize import runs as run_loader  # noqa: E402


def format_report(run, result):
    battery = run.battery_span()
    tach = run.tach_health()

    lines = [
        f"# {run.profile}",
        "",
        f"- run: `{run.path}`",
        f"- started: {run.metadata.get('started_utc', 'unknown')}",
        f"- result: **{run.result}**"
        + (
            f" - {run.metadata['result_detail']}"
            if run.metadata.get("result_detail")
            else ""
        ),
        f"- surface: {run.metadata.get('surface', 'unknown')}",
        f"- git: `{(run.metadata.get('git_sha') or 'unknown')[:12]}`",
        f"- telemetry rows: {len(run.rows)}",
    ]
    if run.metadata.get("gain_overrides"):
        lines.append(f"- gain overrides: `{run.metadata['gain_overrides']}`")
    if battery:
        if "mv_start" in battery:
            lines.append(
                f"- pack: {battery['mv_start'] / 1000:.2f} V -> "
                f"{battery['mv_end'] / 1000:.2f} V "
                f"(sagged to {battery['mv_min'] / 1000:.2f} V under load)"
            )
        elif "level_start" in battery:
            lines.append(
                f"- pack level: {battery['level_start']} -> {battery['level_end']} of 255"
            )
    if tach:
        lines.append(
            f"- tachometer: {tach['merged_pulses']} merged pulses, "
            f"{tach['rejected_edges']} rejected edges, "
            f"{tach['frame_loss_fraction'] * 100:.2f}% of Jetson frames rejected"
        )
    if run.skipped_lines:
        lines.append(
            f"- {run.skipped_lines} unparseable serial lines (truncated frames are normal)"
        )

    if run.aborted:
        lines += [
            "",
            "> **This run did not complete.** Anything below is fitted to a partial "
            "profile and should be treated as indicative, not as a measurement.",
        ]

    lines += ["", "## Findings", ""]
    lines += (
        ["```", *result["summary"], "```"]
        if result["summary"]
        else ["(nothing to report)"]
    )

    if result.get("plots"):
        lines += ["", "## Plots", ""]
        lines += [
            f"- [{os.path.basename(plot)}]({os.path.basename(plot)})"
            for plot in result["plots"]
            if plot
        ]

    if result["vehicle"]:
        lines += ["", "## Values for vehicle.yaml", ""]
        lines.append("```")
        for key, value in sorted(result["vehicle"].items()):
            lines.append(
                f"{key}: {value!r}"
                if isinstance(value, list)
                else f"{key}: {value:.6g}"
                if isinstance(value, float)
                else f"{key}: {value}"
            )
        lines.append("```")
        lines += [
            "",
            "Apply with:",
            "",
            "```bash",
            f"./scripts/apply_vehicle_patch.py {run.path}",
            "```",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir", help="run directory produced by characterize.launch.py"
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=None,
        help="vehicle mass in kg, required by the coastdown fit (A1)",
    )
    parser.add_argument(
        "--speed-source",
        choices=("auto", "odometry", "wheel_rpm"),
        default="auto",
        help=(
            "ground-speed channel for the longitudinal fits. auto (default) "
            "uses ZED odometry unless the segment contains physically "
            "impossible speeds, in which case it falls back to the tachometer"
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="write files without printing"
    )
    args = parser.parse_args(argv)

    try:
        run = run_loader.load(args.run_dir)
    except (FileNotFoundError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    try:
        result = analysis.analyze(
            run, {"mass": args.mass, "speed_source": args.speed_source}
        )
    except KeyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"error: {run.profile}: {error}", file=sys.stderr)
        return 1

    report = format_report(run, result)
    report_path = os.path.join(run.path, "report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    if result["vehicle"]:
        import yaml

        patch_path = os.path.join(run.path, "vehicle_patch.yaml")
        with open(patch_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {"run": run.path, "profile": run.profile, "values": result["vehicle"]},
                handle,
                sort_keys=True,
                default_flow_style=False,
            )

    if not args.quiet:
        print(report)
        print(f"written: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
