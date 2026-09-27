#!/usr/bin/env python3
"""Fit plant.py's steering chain to a Gazebo run, from what the car was told.

    python3 fit_gazebo.py /path/run.trace.csv /path/run.cmd.csv

run_monitor.py records, on the simulation clock, the pose (trace.csv) and the
DriveCommand and tachometer (cmd.csv) of a validate.sh run.  This replays the
recorded steering commands, at the MEASURED speed, through plant.py's chain
-- dead time, servo lag, the command->angle table, scrub/understeer, chassis
yaw lag -- and compares the yaw rate it produces with the yaw rate Gazebo's
car actually had.

Yaw rate, not position: it does not integrate, so an open-loop replay over a
whole lap cannot drift, and any disagreement is the model's.

Every "car" in the batched Plant is one candidate parameter set, so a grid of
thousands is a single vectorised replay.  Printed: the nominal model's error,
the best fit, and the phase and gain of both at the weave frequency -- the
number that decides whether a closed loop rings.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plant import Plant  # noqa: E402

HERE = Path(__file__).resolve().parent
DT = 0.005


def load(trace_path, cmd_path, cut_roll=8.0):
    tr = np.genfromtxt(trace_path, delimiter=",", names=True)
    cm = np.genfromtxt(cmd_path, delimiter=",", names=True, dtype=None, encoding=None)
    kind = cm["kind"]
    cmd = np.stack([cm["stamp"][kind == "cmd"], cm["a"][kind == "cmd"]], 1)
    cmd = cmd[np.argsort(cmd[:, 0])]
    t = tr["stamp"]
    order = np.argsort(t)
    t, x, y, yaw, roll = (tr[k][order] for k in ("stamp", "x", "y", "yaw", "roll"))
    keep = np.r_[True, np.diff(t) > 1e-4]
    t, x, y, yaw, roll = t[keep], x[keep], y[keep], yaw[keep], roll[keep]
    # Up to the first sign of a bale strike: after that the car is not
    # driving, it is being pushed around.
    bad = np.flatnonzero(np.abs(roll) > cut_roll)
    if len(bad):
        stop = bad[0] - 30
        t, x, y, yaw = t[:stop], x[:stop], y[:stop], yaw[:stop]
    return t, x, y, np.unwrap(yaw), cmd


def resample(t, x, y, yaw, cmd):
    """Everything on one 200 Hz grid over the stretch where the car moves."""
    v_pose = np.hypot(np.gradient(x, t), np.gradient(y, t))
    moving = np.flatnonzero(v_pose > 0.5)
    t0, t1 = t[moving[0]], t[moving[-1]]
    g = np.arange(t0 - 1.0, t1, DT)
    # Speed and yaw rate from the pose, smoothed over ~0.1 s: the pose is 30 Hz.
    k = 5
    ker = np.ones(k) / k
    xs, ys = np.convolve(x, ker, "same"), np.convolve(y, ker, "same")
    v = np.hypot(np.gradient(xs, t), np.gradient(ys, t))
    r = np.convolve(np.gradient(yaw, t), ker, "same")
    speed = np.interp(g, t, v)
    rate = np.interp(g, t, r)
    # Steering: zero-order hold of the latest command.
    idx = np.clip(np.searchsorted(cmd[:, 0], g, side="right") - 1, 0, len(cmd) - 1)
    steer = np.where(g >= cmd[0, 0], cmd[idx, 1], 0.0)
    return g, steer, speed, rate


def replay(cfg, params, steer, speed):
    """(N, T) yaw rate for N parameter sets under one command/speed history."""
    n = len(params["dead_time"])
    p = Plant(
        {**cfg, "randomize": {**cfg["randomize"], "enabled": False}},
        n,
        np.random.default_rng(0),
    )
    z = np.zeros(n)
    p.reset(np.ones(n, bool), z, z, z, np.full(n, speed[0]))
    for name, v in params.items():
        getattr(p, name)[:] = v
    out = np.empty((n, len(steer)))
    dt = np.full(n, DT)
    for i in range(len(steer)):
        p.speed[:] = speed[i]
        p.target[:] = speed[i]
        p.motor_target[:] = speed[i]
        out[:, i] = p.substep(np.full(n, steer[i]), np.full(n, speed[i]), dt)
    return out


def score(sim, rate, speed, lo=1.5, hi=99.0):
    """Relative RMS yaw-rate error over samples with lo < v < hi."""
    m = (speed > lo) & (speed < hi)
    err = sim[:, m] - rate[m]
    return np.sqrt((err**2).mean(1)) / np.sqrt((rate[m] ** 2).mean())


def weave_response(sig, steer, speed, f):
    """Gain and phase (deg) of steer -> yaw rate at frequency f, fast straights."""
    m = speed > 3.5
    t = np.arange(len(steer)) * DT
    ref = np.exp(-2j * np.pi * f * t)
    s_in = np.sum(steer[m] * ref[m])
    s_out = np.sum(sig[m] * ref[m])
    h = s_out / s_in
    return abs(h), np.degrees(np.angle(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("cmd")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--weave-hz", type=float, default=0.4)
    ap.add_argument("--min-speed", type=float, default=1.5)
    ap.add_argument("--max-speed", type=float, default=99.0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    t, x, y, yaw, cmd = load(args.trace, args.cmd)
    g, steer, speed, rate = resample(t, x, y, yaw, cmd)
    print(
        f"{len(g) * DT:.1f} s of driving, {np.mean(speed > 3.5) * 100:.0f}% above 3.5 m/s"
    )

    pl = cfg["plant"]
    nominal = dict(
        dead_time=[float(pl["command_dead_time"])],
        steer_tau=[float(pl["steering_tau"])],
        yaw_tau=[float(pl["yaw_response_tau"])],
        tire_scrub=[float(pl["tire_scrub"])],
        understeer=[float(pl["understeer_gradient"])],
    )
    nom = replay(cfg, {k: np.array(v) for k, v in nominal.items()}, steer, speed)
    e_nom = score(nom, rate, speed, args.min_speed, args.max_speed)[0]

    grid = dict(
        dead_time=[0.03, 0.06, 0.10, 0.15, 0.19, 0.25, 0.32],
        steer_tau=[0.005, 0.02, 0.05, 0.10, 0.16],
        yaw_tau=[0.25, 0.34, 0.45, 0.6, 0.75, 0.9, 1.1],
        tire_scrub=[0.95, 1.10, 1.25, 1.45],
        understeer=[0.0, 0.007, 0.02, 0.04],
    )
    combos = list(itertools.product(*grid.values()))
    keys = list(grid)
    best = (np.inf, None)
    errs = np.empty(len(combos))
    for start in range(0, len(combos), 600):
        chunk = np.array(combos[start : start + 600])
        params = {k: chunk[:, i] for i, k in enumerate(keys)}
        e = score(
            replay(cfg, params, steer, speed),
            rate,
            speed,
            args.min_speed,
            args.max_speed,
        )
        errs[start : start + len(chunk)] = e
        j = int(np.argmin(e))
        if e[j] < best[0]:
            best = (e[j], dict(zip(keys, chunk[j])))
    print(
        f"\nnominal plant.py   relative yaw-rate error {e_nom:.3f}   { ({k: v[0] for k, v in nominal.items()}) }"
    )
    print(
        f"best of {len(combos)} fits  relative yaw-rate error {best[0]:.3f}   {best[1]}"
    )
    # How sharply is each parameter pinned?  Error of the best fit with that
    # one parameter moved, the others held.
    order = np.argsort(errs)[:10]
    print("\nten best:")
    for i in order:
        print(
            "  "
            + "  ".join(f"{k} {v:.3f}" for k, v in zip(keys, combos[i]))
            + f"   err {errs[i]:.3f}"
        )

    fit = replay(cfg, {k: np.array([v]) for k, v in best[1].items()}, steer, speed)
    for name, sig in (("gazebo", rate), ("nominal", nom[0]), ("fitted", fit[0])):
        gn, ph = weave_response(sig, steer, speed, args.weave_hz)
        print(
            f"  {name:8s} steer->yaw-rate at {args.weave_hz} Hz, v>3.5: gain {gn:.2f}  phase {ph:+.0f} deg"
        )


if __name__ == "__main__":
    main()
