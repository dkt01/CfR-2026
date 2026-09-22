#!/usr/bin/env python3
"""Score a policy offline, and draw what it did.

    python3 evaluate.py runs/v1/policy.npz
    python3 evaluate.py --baseline                 # the scripted floor
    python3 evaluate.py runs/v1/policy.npz --plot report.png

One line per requirement, printed in the order they were asked for:

    field finish %   fraction of 256 runs that complete two laps AND come to
                     a stop, with the dead time, drag, steering trim, slew,
                     chassis lag and localisation all redrawn.  This is the
                     one that predicts the day.  Note it counts STOPPED, not
                     the line crossed: a car still doing 5 m/s at the end of
                     lap two has not finished a run on a course with no
                     run-off.
    two-lap time     seconds of RACE, start line to finish line.  The coast
                     down afterwards is not the lap time and is not counted
                     in it.
    min clearance    metres from the body to the nearest bale, worst over
                     every run.  Negative is a hit; under `graze_margin` is
                     the graze the rules do not allow.
    cross-track      mean and worst distance off the centerline while racing.
    jerk             rms change in the steering command between ticks -- the
                     sign-changing part, which is what "jittery" means.
    lap gain         seconds the best lap took off the slowest one.
    stop             metres past the finish line the car came to rest.
    max overspeed    m/s above the cap, worst over every run.  The command is
                     capped structurally, so anything here is the car
                     ARRIVING too fast -- a braking failure, not a throttle
                     one.

A policy that does not beat the scripted baseline on the first of those has
not earned a run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import track as track_mod
from baseline import BaselineDriver, feasible_profile
from env import FormulaOneEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def driver_for(args, track, cfg, n):
    if args.baseline:
        d = BaselineDriver(track, cfg)
        d.reset(n)

        return (lambda env, obs: env.scripted_action(d)), "scripted baseline"

    from policy import NumpyPolicy

    net = NumpyPolicy.load(args.policy)
    return (lambda env, obs: net.act(obs)), str(args.policy)


def run(env, act, episodes):
    obs = env.reset()
    out = []
    for _ in range(40000):
        obs, _, terminated, truncated, info = env.step(act(env, obs))
        for i in np.flatnonzero(terminated | truncated):
            out.append(info[i])
        if len(out) >= episodes:
            break
    return out[:episodes]


def table(name, rows, graze_margin):
    finished = [r for r in rows if r["stopped"]]
    times = (np.array([r["race_time"] for r in finished])
             if finished else np.array([np.nan]))
    clear = np.array([r["min_clearance"] for r in rows])
    over = np.array([r["max_overspeed"] for r in rows])
    cte = np.array([r["mean_cte"] for r in rows])
    max_cte = np.array([r["max_cte"] for r in rows])
    jerk = np.array([r["steer_jerk_rms"] for r in rows])
    gains = np.array(
        [r["last_lap"] - r["best_lap"] for r in finished
         if r["last_lap"] > 0 and np.isfinite(r["best_lap"])] or [np.nan])
    stops = (np.array([r["stop_distance"] for r in finished])
             if finished else np.array([np.nan]))
    grazed = float(np.mean(clear < graze_margin))
    print(
        f"  {name:<9} {100*len(finished)/len(rows):5.1f}%   "
        f"{np.nanmean(times):6.2f} +/- {np.nanstd(times):4.2f} s   "
        f"{clear.min():+.3f} m   {np.mean(cte):.3f}/{max_cte.max():.3f} m   "
        f"{np.mean(jerk):.4f}   {np.nanmean(gains):+5.2f} s   "
        f"{np.nanmean(stops):5.1f} m   {over.max():+.2f} m/s   "
        f"grazed {100*grazed:4.1f}%   crashed {sum(r['crashed'] for r in rows)}/{len(rows)}"
    )
    return dict(finish=len(finished) / len(rows), time=float(np.nanmean(times)),
                cte=float(np.mean(cte)), jerk=float(np.mean(jerk)))


def trace(cfg, track, act, seed=99):
    """One deterministic lap, logged, for the plot."""
    env = FormulaOneEnv(cfg, track, 1, seed, deterministic=True)
    env.random_start = False
    obs = env.reset()
    rows = []
    for _ in range(6000):
        s = env.snapshot()
        rows.append((s["station"][0], s["speed"][0], s["v_cap"][0],
                     s["clearance"][0], s["x"][0], s["y"][0], s["elapsed"][0],
                     env.frame["lateral"][0], env.last_steer[0]))
        obs, _, term, trunc, _ = env.step(act(env, obs))
        if term[0] or trunc[0]:
            break
    return np.array(rows)


def plot(path, track, log, cfg, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    station, speed, cap, clear, x, y, elapsed, lateral, steer = log.T
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(track.x, track.y, color="0.85", lw=8, solid_capstyle="round", zorder=1)
    hp = track.hairpin
    ax.scatter(track.x[hp], track.y[hp], s=3, color="0.6", zorder=2, label="hairpin (2.5 m/s)")
    sc = ax.scatter(x, y, c=speed, cmap="RdYlGn", vmin=2.0, vmax=5.4, s=7, zorder=3)
    plt.colorbar(sc, ax=ax, label="m/s", fraction=0.025)
    ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{label} -- {elapsed[-1]:.2f} s for {cfg['env']['laps']} laps, "
                 f"min clearance {clear.min():.3f} m")

    ax = fig.add_subplot(gs[1, :])
    order = np.argsort(station)
    ax.plot(track.s, track.v_cap, color="crimson", lw=1.4, label="rule cap")
    ax.plot(track.s, feasible_profile(track, cfg), color="steelblue", lw=1.0,
            ls="--", label="coast-feasible ideal")
    ax.scatter(station, speed, s=4, color="black", label="driven")
    ax.set_xlabel("station (m)"); ax.set_ylabel("m/s"); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=3); ax.set_xlim(0, track.length)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(station, clear, lw=0.9, color="darkgreen")
    ax.axhline(cfg["reward"]["graze_margin"], color="orange", ls="--", label="graze")
    ax.axhline(0.0, color="crimson", ls="-", label="contact")
    ax.set_xlabel("station (m)"); ax.set_ylabel("clearance (m)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.set_xlim(0, track.length)

    ax = fig.add_subplot(gs[2, 1])
    # The two requirements that do not show up in a line on a map: how far
    # off the centerline it ran, and how much the steering shook getting
    # there.
    ax.plot(station, lateral, lw=0.9, color="navy", label="cross-track")
    ax.plot(station, steer, lw=0.7, color="darkorange", alpha=0.8,
            label="steering cmd")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("station (m)"); ax.set_ylabel("m  /  command")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    ax.set_title(f"cte {np.abs(lateral).mean():.3f} m mean, "
                 f"{np.abs(lateral).max():.3f} m worst", fontsize=8)

    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"\n  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", type=Path)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()
    if not args.baseline and args.policy is None:
        ap.error("give a policy .npz, or --baseline")

    cfg = yaml.safe_load(args.config.read_text())
    tr = track_mod.build(cfg, ROOT)

    print(f"\n  {'':<9} {'finish':>6}   {'race time':>14}   {'clear':>8}   "
          f"{'cte mean/max':>13}   {'jerk':>6}   {'gain':>7}   {'stop':>6}   "
          f"{'overspeed':>9}")
    results = {}
    for name, deterministic in (("nominal", True), ("field", False)):
        env = FormulaOneEnv(cfg, tr, 64, seed=4242, deterministic=deterministic)
        env.random_start = False
        act, label = driver_for(args, tr, cfg, 64)
        results[name] = table(
            name, run(env, act, args.episodes if not deterministic else 64),
            float(cfg["reward"]["graze_margin"]))

    if args.plot:
        act, label = driver_for(args, tr, cfg, 1)
        plot(args.plot, tr, trace(cfg, tr, act), cfg, label)
    print()


if __name__ == "__main__":
    main()
