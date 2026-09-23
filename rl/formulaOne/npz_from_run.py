#!/usr/bin/env python3
"""Turn a car characterization run into the .npz traces fit_yaw_lag.py reads.

    python3 npz_from_run.py ~/cfr_runs/20260925T…_step_steer_fine --out /tmp/car
    python3 fit_yaw_lag.py /tmp/car/*.npz

`fit_yaw_lag.py` fits `plant.yaw_response_tau` by replaying a recorded command
schedule through plant.py and scoring it against a measured yaw rate.  Until
now the only thing that could produce those traces was `measure_step_steer.py`,
which drives Gazebo -- so the 0.34 s currently in config.yaml is fitted to ten
traces on ONE SIMULATED CAR.  This reads the same traces off the real one, so
the same fit, scored the same way, can be run against the car that will race.

WHICH CHANNEL
-------------
A run directory now carries three yaw sources and they are not interchangeable:

  odom   `~/odom`, raw visual odometry.  Never corrected, therefore continuous,
         therefore safe to differentiate.  THE DEFAULT, and the one that makes
         the answer comparable with the Gazebo fit, which used pose from a
         simulator that has no loop closure to jump on.
  pose   `~/pose`, the map-frame pose.  The SDK rewrites it when it closes a
         loop, and a closure inside a 1.4 s step lands as a step in yaw that a
         derivative reads as the car snapping sideways.  Cross-check only.
  imu    gyroscope z, a DIRECT yaw-rate measurement at the IMU's own rate, with
         nothing differentiated and nothing visual in the path.  The highest
         resolution available and the best answer to "when did the car start
         rotating"; it is integrated here only so it can go through the same
         savgol pipeline as the other two rather than a second one.

Gyro bias is removed using the pre-step window, where the car is going
straight: over one step it is a constant offset, and a constant offset in yaw
RATE is a ramp in yaw that the fit would read as a slow turn.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# How much straight-running baseline to keep ahead of each step.  It has to
# cover the command dead time (~0.19 s measured) with room to see the yaw rate
# still at zero through it, or the fit has nothing to anchor t=0 against.
PRE_S = 0.6


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def step_windows(rows, prefix):
    """(label, onset t_ros, hold duration) for every step whose label matches.

    Onset comes from telemetry.csv rather than from the sensor files because
    that is where the COMMAND changes: the runner publishes the new command and
    writes the row in the same tick, so this is the moment the step was issued,
    to within one control period.  Timing it off a camera sample instead would
    fold the camera's period into the measured dead time.
    """
    out, current, start, last = [], None, None, None
    for row in rows:
        if row.get("phase") != "running":
            continue
        label = row.get("step_label") or ""
        t = float(row["t_ros"])
        if label != current:
            if current and current.startswith(prefix):
                out.append((current, start, last - start))
            current, start = label, t
        last = t
    if current and current.startswith(prefix):
        out.append((current, start, last - start))
    return out


def yaw_series(run, source):
    """(t_ros, yaw) for the chosen channel, unwrapped, at its native rate."""
    if source == "imu":
        rows = read_csv(run / "imu.csv")
        t = np.array([float(r["t_ros"]) for r in rows])
        wz = np.array([float(r["wz"]) for r in rows])
        return t, wz, "rate"
    rows = [r for r in read_csv(run / "pose.csv") if r["source"] == source]
    if not rows:
        raise SystemExit(f"no rows with source={source!r} in {run / 'pose.csv'}")
    t = np.array([float(r["t_ros"]) for r in rows])
    yaw = np.unwrap(np.array([float(r["yaw"]) for r in rows]))
    return t, yaw, "yaw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "run", type=Path, help="a run directory from characterize.launch.py"
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="directory for the .npz files"
    )
    ap.add_argument("--source", choices=("odom", "pose", "imu"), default="odom")
    ap.add_argument("--prefix", default="step_", help="step labels to export")
    args = ap.parse_args()

    for name in ("telemetry.csv", "pose.csv" if args.source != "imu" else "imu.csv"):
        if not (args.run / name).is_file():
            raise SystemExit(
                f"{args.run / name} is missing. Runs recorded before 2026-09-22 "
                f"have only the 50 Hz resampled telemetry.csv, which cannot "
                f"resolve a transient this short -- that is what produced the "
                f"1.76 s artifact. Re-run the profile."
            )

    telemetry = read_csv(args.run / "telemetry.csv")
    steps = step_windows(telemetry, args.prefix)
    if not steps:
        raise SystemExit(f"no steps labelled {args.prefix!r} in {args.run}")

    t_all, values, kind = yaw_series(args.run, args.source)
    args.out.mkdir(parents=True, exist_ok=True)

    # One file per step.  fit_yaw_lag.load() keys traces by command within a
    # file, so two repeats of the same command would collide in one.
    written, seen = [], defaultdict(int)
    for label, onset, hold in steps:
        window = (t_all >= onset - PRE_S) & (t_all <= onset + hold)
        t = t_all[window] - onset
        if len(t) < 12:
            print(f"  {label:20s} only {len(t)} samples, skipped")
            continue
        block = values[window]
        if kind == "rate":
            pre = block[t < 0.0]
            if len(pre):
                block = block - float(np.mean(pre))  # gyro bias, see module docstring
            yaw = np.concatenate(
                [[0.0], np.cumsum(np.diff(t) * (block[1:] + block[:-1]) / 2)]
            )
        else:
            yaw = block
        yaw = yaw - float(np.interp(0.0, t, yaw))

        # The command the step actually held, and the speed it held it at, both
        # read back out of the run rather than out of the profile: a step that
        # was slew-limited or clipped did not command what the YAML said.
        rows = [
            r
            for r in telemetry
            if r.get("step_label") == label and r.get("cmd_steering")
        ]
        command = float(rows[-1]["cmd_steering"])
        speeds = [float(r["speed"]) for r in rows if r.get("speed")]
        speed = float(np.median([s for s in speeds if s > 0.2] or [0.0]))
        if speed <= 0.0:
            print(f"  {label:20s} no usable speed channel, skipped")
            continue

        seen[label] += 1
        path = args.out / f"{args.run.name}_{label}.npz"
        np.savez(
            path,
            speed=np.array(speed),
            pre=np.array(PRE_S),
            post=np.array(float(hold)),
            **{f"t_{command:+.2f}": t, f"yaw_{command:+.2f}": yaw},
        )
        rate = len(t) / (t[-1] - t[0])
        written.append(path)
        print(
            f"  {label:20s} cmd {command:+.2f}  v {speed:.2f} m/s  "
            f"{len(t):4d} samples over {t[-1] - t[0]:.2f} s ({rate:.0f} Hz)  "
            f"-> {path.name}"
        )

    if not written:
        raise SystemExit("nothing written")
    print(f"\n  {len(written)} traces from {args.source}.")
    print(f"  python3 fit_yaw_lag.py {args.out}/*.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
