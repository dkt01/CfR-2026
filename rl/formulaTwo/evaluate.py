#!/usr/bin/env python3
"""Score a policy offline against the as-built, seen-through-a-ZED course.

    python3 evaluate.py runs/f2_v1/policy.npz
    python3 evaluate.py ../formulaOne/bestModel/v12/policy.npz   # v12, map only
    python3 evaluate.py --baseline
    python3 evaluate.py runs/f2_v1/policy.npz --plot report.png

A policy narrower than the actor observation (v12 is 35 wide) is fed the
first columns only -- the same map features it was trained on.

    finish     THREE laps and at rest, from the grid.
    lap        race time / 3, standing start included.  v12 is ~32 s/lap in
               Gazebo and 35.85 s/lap in this model (when it finishes).
    --dr       field at graded randomisation strengths as well.  At full
               strength even the scripted prior at 1.4 m/s finishes 0%, so
               the graded rows are where policies actually differ.
    clear      body (chassis + tyres) to bale, worst and 10th percentile.
    cte, gate  against the centerline as built; gate is the reward's
               tracking factor (1 = on the line, pointing along it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import track as track_mod
from baseline import BaselineDriver
from env import FormulaTwoEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def driver(args, track, cfg):
    if args.baseline:
        d = BaselineDriver(track, cfg)
        return (lambda env, obs: env.scripted_action(d)), "scripted baseline"
    from policy import NumpyPolicy

    net = NumpyPolicy.load(args.policy)
    w = net.obs_dim
    return (lambda env, obs: net.act(obs[:, :w])), f"{args.policy} ({w} inputs)"


def run(env, act, episodes):
    """First episode of each car; `episodes` > env.n runs more batches."""
    out = []
    while len(out) < episodes:
        obs = env.reset()
        first = [None] * env.n
        for _ in range(12000):
            obs, _, terminated, truncated, info = env.step(act(env, obs))
            for i in np.flatnonzero(terminated | truncated):
                if first[i] is None:
                    first[i] = info[i]
            if all(f is not None for f in first):
                break
        out += [f for f in first if f is not None]
    return out[:episodes]


def row(name, rows):
    done = [r for r in rows if r["stopped"]]
    laps = np.array([r["race_time"] / 3 for r in done]) if done else np.array([np.nan])
    clear = np.array([r["min_clearance"] for r in rows])
    print(
        f"  {name:<8} {100 * len(done) / len(rows):5.1f}%  "
        f"{np.nanmean(laps):6.2f} +/- {np.nanstd(laps):4.2f} s/lap  "
        f"clear {clear.min():+.3f} / p10 {np.percentile(clear, 10):+.3f}  "
        f"cte {np.mean([r['mean_cte'] for r in rows]):.3f}  "
        f"gate {np.mean([r['mean_gate'] for r in rows]):.2f}  "
        f"jerk {np.mean([r['steer_jerk_rms'] for r in rows]):.4f}  "
        f"crash {sum(r['crashed'] for r in rows)}/{len(rows)}  "
        f"dist {np.median([r['distance'] for r in rows]):.0f} m"
    )
    if done:
        lt = np.array([r["lap_times"] for r in done])
        print(
            f"           laps {' / '.join(f'{v:.2f}' for v in lt.mean(0))} s   "
            f"stop +{np.mean([r['stop_distance'] for r in done]):.1f} m"
        )


def plot(path, track, cfg, act, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = FormulaTwoEnv(cfg, track, 1, 99, deterministic=True)
    env.random_start = False
    obs = env.reset()
    log = []
    for _ in range(12000):
        s = env.snapshot()
        log.append([s[k][0] for k in ("x", "y", "speed", "station", "clearance", "lateral")])
        obs, _, te, tr, _ = env.step(act(env, obs))
        if te[0] or tr[0]:
            break
    x, y, v, st, cl, lat = np.array(log).T
    fig, ax = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw=dict(height_ratios=[2, 1, 1]))
    ax[0].plot(track.x, track.y, color="0.85", lw=8, zorder=1)
    sc = ax[0].scatter(x, y, c=v, cmap="RdYlGn", vmin=2, vmax=5.4, s=5, zorder=3)
    plt.colorbar(sc, ax=ax[0], label="m/s", fraction=0.025)
    ax[0].set_aspect("equal")
    ax[0].set_title(f"{label} -- {len(log) / 20:.1f} s, min clearance {cl.min():.3f} m")
    t = np.arange(len(v)) / 20
    ax[1].plot(t, v, lw=0.8, label="speed")
    ax[1].set_ylabel("m/s")
    ax[2].plot(t, cl, lw=0.8, color="darkgreen", label="clearance")
    ax[2].plot(t, lat, lw=0.8, color="navy", label="cross-track")
    ax[2].axhline(cfg["reward"]["graze_margin"], color="orange", ls="--")
    ax[2].axhline(0, color="crimson")
    ax[2].set_xlabel("s")
    ax[2].legend()
    for a in ax[1:]:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", type=Path)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--plot", type=Path)
    ap.add_argument("--dr", type=str, default="0.25,0.5,0.75")
    args = ap.parse_args()
    if not args.baseline and args.policy is None:
        ap.error("give a policy .npz, or --baseline")
    cfg = yaml.safe_load(args.config.read_text())
    tr = track_mod.build(cfg, ROOT)
    act, label = driver(args, tr, cfg)
    print(f"\n  {label}")
    runs = [("nominal", True, 32, 1.0)]
    runs += [(f"dr {s}", False, 64, float(s)) for s in args.dr.split(",") if s]
    runs += [("field", False, args.episodes, 1.0)]
    for name, det, n, scale in runs:
        env = FormulaTwoEnv(cfg, tr, min(n, 128), seed=4242, deterministic=det)
        env.random_start = False
        env.dr_scale = scale
        row(name, run(env, act, n))
    if args.plot:
        plot(args.plot, tr, cfg, act, label)
    print()


if __name__ == "__main__":
    main()
