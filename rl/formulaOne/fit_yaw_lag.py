#!/usr/bin/env python3
"""Fit plant.py's yaw-response lag to Gazebo step-steer traces.

    python3 measure_step_steer.py --speed 2.5 --save /tmp/steer_2p5.npz
    python3 fit_yaw_lag.py /tmp/steer_*.npz

`measure_turn_radius.py` established what the car does once it is SETTLED in a
corner: a uniform ~1.10x wider radius than a kinematic bicycle, which
`tire_scrub` now carries.  `measure_step_steer.py` then showed that matching
the settled radius is not the same as matching the way the car GETS there --
after a step of the steering command Gazebo takes 0.2-0.4 s longer than the
model to reach half its steady yaw rate.  That is the entry to a hairpin, and
it is where both the scripted baseline and two trained policies beached.

This file turns that observation into one number instead of an adjective.  It
replays each recorded command schedule through plant.py at a grid of
`yaw_response_tau` values and reports the one that minimises RMS yaw-rate
error against Gazebo across every trace at once -- so the value in config.yaml
is fitted to the measurement rather than reasoned toward from a crash.

Two alternative explanations are scored alongside it, because a fit that is
never compared to anything is just a number with a decimal point:

    tau = 0          the model as it was, no lag at all
    steering_tau     the same delay blamed on the servo instead of the chassis

If the servo explanation fitted as well, the servo would be the better place
to put it -- it is already in the model.  It does not, and the reason is in
the shape: a lag on the ANGLE is bent by tan() and by the steering limit, so
it cannot fit large and small commands with one constant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import savgol_filter

from plant import Plant

HERE = Path(__file__).resolve().parent


def smoothed_rate(t, yaw, dt_grid=0.01, win_s=0.15, poly=3):
    """Same Savitzky-Golay derivative measure_step_steer.py reports with."""
    t_grid = np.arange(t[0], t[-1], dt_grid)
    yaw_grid = np.interp(t_grid, t, yaw)
    win = int(win_s / dt_grid)
    win += 1 - (win % 2)
    win = max(win, poly + 2 + (poly + 2) % 2 + 1)
    return t_grid, savgol_filter(yaw_grid, win, poly, deriv=1, delta=dt_grid)


def model_trace(cfg, speed, command, t_pre, t_post, yaw_tau=None, steering_tau=None):
    """Replay one step-steer schedule through plant.py.

    `yaw_tau` / `steering_tau` override config.yaml for this trace only, which
    is what makes the sweep below a sweep rather than a series of edits.
    """
    plant_cfg = dict(cfg["plant"])
    if yaw_tau is not None:
        plant_cfg["yaw_response_tau"] = float(yaw_tau)
    if steering_tau is not None:
        plant_cfg["steering_tau"] = float(steering_tau)
    flat = {
        **cfg,
        "plant": plant_cfg,
        "randomize": {**cfg["randomize"], "enabled": False},
    }
    plant = Plant(flat, 1, np.random.default_rng(0))
    one = np.array([True])
    plant.reset(
        one, np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([speed])
    )
    dt = plant.dt_sub
    n_pre, n_post = int(round(t_pre / dt)), int(round(t_post / dt))
    t = np.empty(n_pre + n_post)
    rate = np.empty_like(t)
    for i in range(n_pre):
        r = plant.substep(np.array([0.0]), np.array([speed]), np.array([dt]))
        t[i], rate[i] = (i + 1) * dt - t_pre, r[0]
    for i in range(n_post):
        r = plant.substep(np.array([command]), np.array([speed]), np.array([dt]))
        t[n_pre + i], rate[n_pre + i] = (i + 1) * dt, r[0]
    return t, rate


def load(paths):
    """Every usable (speed, command, t_grid, gazebo_rate, pre, post) trace."""
    out = []
    for path in paths:
        d = np.load(path)
        speed = float(d["speed"])
        pre, post = float(d["pre"]), float(d["post"])
        for key in d.files:
            if not key.startswith("t_"):
                continue
            cmd = float(key[2:])
            t, yaw = d[key], d["yaw_" + key[2:]]
            tg, rate = smoothed_rate(t, yaw)
            steady = float(np.mean(rate[tg > post - 0.4]))
            # A run where the car never turned, or turned the wrong way, is a
            # run where the teleport or the settle failed -- not data about
            # the lag.  Scoring it would drag the fit toward whatever value
            # best reproduces a car sitting still.
            if abs(steady) < 0.2 or np.sign(steady) != np.sign(cmd):
                print(
                    f"  skipping {Path(path).name} cmd {cmd:+.2f}: "
                    f"steady yaw rate {steady:+.3f} rad/s is not a turn"
                )
                continue
            out.append(
                dict(
                    path=Path(path).name,
                    speed=speed,
                    cmd=cmd,
                    t=tg,
                    rate=rate,
                    pre=pre,
                    post=post,
                    steady=steady,
                )
            )
    return out


def rms(traces, cfg, **override):
    """Mean RELATIVE yaw-rate error over every trace, and the per-trace list.

    Scored from the step to the end of the window -- the transient IS the
    measurement, so it is not down-weighted against the settled tail -- and
    divided by each trace's own steady yaw rate.

    The division is what stops one trace buying the fit.  A step to full lock
    reaches 3.8 rad/s where a hairpin-sized step reaches 0.9; in absolute
    rad/s the big one carries four times the weight of the small one, so an
    unnormalised fit is really a fit to the largest steering angle in the set
    -- which is the one steering angle this car never uses in the corridor.
    """
    errs = []
    for tr in traces:
        tm, rm = model_trace(
            cfg, tr["speed"], tr["cmd"], tr["pre"], tr["post"], **override
        )
        on_grid = np.interp(tr["t"], tm, rm)
        window = (tr["t"] >= 0.0) & (tr["t"] <= tr["post"])
        err = np.sqrt(np.mean((on_grid[window] - tr["rate"][window]) ** 2))
        errs.append(float(err / abs(tr["steady"])))
    return float(np.mean(errs)), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "traces", type=Path, nargs="+", help="npz from measure_step_steer.py --save"
    )
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument(
        "--write",
        action="store_true",
        help="write the fitted tau back into config.yaml",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    traces = load(args.traces)
    if not traces:
        sys.exit("no usable traces")
    print(f"\n  {len(traces)} traces, {sorted({t['speed'] for t in traces})} m/s\n")

    grid = np.arange(0.0, 0.81, 0.01)
    scores = [rms(traces, cfg, yaw_tau=tau)[0] for tau in grid]
    best_i = int(np.argmin(scores))
    best_tau = float(grid[best_i])

    # Per-trace fits as well.  One aggregate number hides whether the traces
    # agree; if the per-trace optima are scattered, a single first-order lag
    # is the wrong SHAPE and no value of it is the answer.
    per_tau = []
    for tr in traces:
        one = [rms([tr], cfg, yaw_tau=tau)[0] for tau in grid]
        per_tau.append(float(grid[int(np.argmin(one))]))

    print("  yaw_response_tau sweep (mean relative yaw-rate error)")
    for tau, sc in zip(grid, scores):
        if abs(round(tau * 100) % 5) < 1e-9:
            mark = "  <-- best" if abs(tau - best_tau) < 1e-9 else ""
            print(f"    tau {tau:.2f} s   {sc:.4f}{mark}")

    # The alternative explanation, scored the same way.
    s_grid = np.arange(0.0, 0.81, 0.02)
    s_scores = [rms(traces, cfg, yaw_tau=0.0, steering_tau=st)[0] for st in s_grid]
    s_best = float(s_grid[int(np.argmin(s_scores))])

    print("\n  --- fit ---")
    print(f"  no lag at all        (tau 0)            err {scores[0]:.4f}")
    print(
        f"  lag on the SERVO     (steering_tau {s_best:.2f})  err {min(s_scores):.4f}"
    )
    print(
        f"  lag on the CHASSIS   (yaw_tau {best_tau:.2f})      "
        f"err {min(scores):.4f}   <-- the model plant.py now has"
    )
    improve = 100 * (1 - min(scores) / scores[0])
    print(f"\n  chassis lag cuts the yaw-rate error by {improve:.0f}% against no lag.")

    _, per = rms(traces, cfg, yaw_tau=best_tau)
    _, per0 = rms(traces, cfg, yaw_tau=0.0)
    print(
        f"\n  {'trace':>22} {'steady':>8} {'err tau=0':>10} {'err fitted':>11}"
        f" {'own best tau':>13}"
    )
    for tr, e0, e1, tau in zip(traces, per0, per, per_tau):
        print(
            f"  {tr['speed']:5.1f} m/s cmd {tr['cmd']:+.2f}   "
            f"{tr['steady']:+7.3f} {e0:10.3f} {e1:11.3f} {tau:13.2f}"
        )
    print(
        f"\n  per-trace tau: median {np.median(per_tau):.2f} s, "
        f"spread {np.min(per_tau):.2f}-{np.max(per_tau):.2f} s"
    )

    if args.write:
        text = args.config.read_text()
        import re

        new = re.sub(
            r"(  yaw_response_tau: )[0-9.]+(\s*#[^\n]*)?",
            rf"\g<1>{best_tau:.2f}"
            f"    # s, FITTED to {len(traces)} Gazebo step-steer traces",
            text,
            count=1,
        )
        args.config.write_text(new)
        print(f"\n  wrote yaw_response_tau: {best_tau:.2f} to {args.config}")
    else:
        print(
            f"\n  set  plant.yaw_response_tau: {best_tau:.2f}  (or re-run with --write)"
        )


if __name__ == "__main__":
    main()
