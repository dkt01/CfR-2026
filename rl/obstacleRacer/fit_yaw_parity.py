#!/usr/bin/env python3
"""Fit Numba yaw lag to the open-floor passes saved by gazebo_check.py.

    python3 fit_yaw_parity.py runs/gazebo/surfaces_YYYYMMDD_HHMMSS.json

This is a diagnostic fit to Gazebo, not a measurement of the real car. Keep
the passed trace and test the result on separate courses before changing the
training model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize_scalar

import course_model
import layouts
import plant as P

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", type=Path)
    ap.add_argument("--check", type=Path, help="separate helix surface trace")
    args = ap.parse_args()
    rows = [
        r
        for r in json.loads(args.traces.read_text())
        if r["name"] == "flat steering control"
    ]
    if len(rows) != 2:
        raise SystemExit("need the 1 and 2 m/s open-floor steering passes")
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    cfg["randomize"]["enabled"] = False
    model = course_model.CourseModel([layouts.TRAIN_SEEDS[0]])
    pl = P.Plant(cfg, model, 1, np.random.default_rng(0))
    hz = float(cfg["env"]["control_hz"])
    steps = int(cfg["env"]["substeps"])

    def error(row, tau):
        first = row["gazebo"]
        pl.reset(
            np.arange(1),
            0,
            np.array([first["x"][0]]),
            np.array([first["y"][0]]),
            np.array([max(0.0, first["z"][0] - 0.02)]),
            np.array([first["yaw"][0]]),
            np.zeros(1),
        )
        pl.params[0, P.P_YAW_TAU] = tau
        n = len(row["model"]["t"])
        yaw = np.empty(n)
        for i in range(n):
            pl.step(np.array([0.55]), np.array([row["speed"]]), steps)
            yaw[i] = pl.state[0, P.S_YAW]
        t = np.arange(1, n + 1) / hz
        target = np.interp(t, first["t"], np.unwrap(first["yaw"]))
        return float(np.sqrt(np.mean(np.degrees(np.unwrap(yaw) - target) ** 2)))

    nominal = float(cfg["plant"]["yaw_response_tau"])
    combined_fit = None
    for label, selected in [
        ("1 m/s", rows[:1]),
        ("2 m/s", rows[1:]),
        ("combined", rows),
    ]:

        def objective(tau):
            return float(np.mean([error(r, tau) ** 2 for r in selected]))

        fit = minimize_scalar(objective, bounds=(0.0, 0.6), method="bounded")
        if label == "combined":
            combined_fit = fit.x
        print(
            f"{label}: nominal tau {nominal:.3f} s -> "
            f"{math.sqrt(objective(nominal)):.2f} deg RMS; "
            f"fit {fit.x:.3f} s -> {math.sqrt(fit.fun):.2f} deg RMS"
        )
    if args.check:
        check = [
            r
            for r in json.loads(args.check.read_text())
            if r["name"].startswith("helix ")
        ]
        for speed in (1.0, 2.0):
            selected = [r for r in check if r["speed"] == speed]

            def rms(tau):
                return math.sqrt(np.mean([error(r, tau) ** 2 for r in selected]))

            print(
                f"helix {speed:.0f} m/s: nominal {rms(nominal):.2f} deg RMS; "
                f"open-floor fit {rms(combined_fit):.2f} deg RMS"
            )


if __name__ == "__main__":
    main()
