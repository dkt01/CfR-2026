#!/usr/bin/env python3
"""Evaluate clean completions, lap time and body clearance on held-out seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import track as track_mod
from baseline import BaselineDriver
from env import FormulaThreeEnv
from metrics import is_clean, json_ready

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def driver(args, track, cfg):
    if args.baseline:
        d = BaselineDriver(track, cfg)
        return (lambda env, obs: env.scripted_action(d)), "scripted baseline"
    from policy import NumpyPolicy

    net = NumpyPolicy.load(args.policy)
    net.check_config(cfg)
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
        if any(f is None for f in first):
            raise RuntimeError("Evaluation did not reach an outcome for every car")
        out += first
    return out[:episodes]


def row(name, rows):
    done = [r for r in rows if is_clean(r)]
    laps = np.array([r["lap_time"] for r in done]) if done else np.array([np.nan])
    lap_mean = float(laps.mean()) if done else float("nan")
    lap_std = float(laps.std()) if done else float("nan")
    clear = np.array([r["min_clearance"] for r in rows])
    print(
        f"  {name:<8} {100 * len(done) / len(rows):5.1f}%  "
        f"{lap_mean:6.2f} +/- {lap_std:4.2f} s/lap  "
        f"clear {clear.min():+.3f} / p10 {np.percentile(clear, 10):+.3f}  "
        f"cte {np.mean([r['mean_cte'] for r in rows]):.3f}  "
        f"gate {np.mean([r['mean_gate'] for r in rows]):.2f}  "
        f"jerk {np.mean([r['steer_jerk_rms'] for r in rows]):.4f}  "
        f"crash {sum(r['crashed'] for r in rows)}/{len(rows)}  "
        f"recovered {sum(r['recovered'] for r in rows)}/{len(rows)}  "
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

    env = FormulaThreeEnv(cfg, track, 1, 99, deterministic=True)
    env.random_start = False
    obs = env.reset()
    log = []
    for _ in range(12000):
        s = env.snapshot()
        log.append(
            [s[k][0] for k in ("x", "y", "speed", "station", "clearance", "lateral")]
        )
        obs, _, te, tr, _ = env.step(act(env, obs))
        if te[0] or tr[0]:
            break
    x, y, v, st, cl, lat = np.array(log).T
    fig, ax = plt.subplots(
        3, 1, figsize=(14, 11), gridspec_kw=dict(height_ratios=[2, 1, 1])
    )
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
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--plot", type=Path)
    ap.add_argument("--dr", type=str, default="0.25,0.5,0.75")
    ap.add_argument("--json", type=Path, help="Per-episode results for comparison")
    args = ap.parse_args()
    if not args.baseline and args.policy is None:
        ap.error("give a policy .npz, or --baseline")
    if args.episodes < 1:
        ap.error("--episodes must be positive")
    config_path = args.config or (
        HERE / "config.yaml" if args.baseline else args.policy.with_name("config.yaml")
    )
    cfg = yaml.safe_load(config_path.read_text())
    tr = track_mod.build(cfg, ROOT)
    act, label = driver(args, tr, cfg)
    print(f"\n  {label}")
    runs = [("nominal", True, min(8, args.episodes), 1.0)]
    runs += [
        (f"dr {s}", False, min(64, args.episodes), float(s))
        for s in args.dr.split(",")
        if s
    ]
    runs += [("field", False, args.episodes, 1.0)]
    report = {}
    for name, det, n, scale in runs:
        env = FormulaThreeEnv(cfg, tr, min(n, 128), seed=4242, deterministic=det)
        env.random_start = False
        env.dr_scale = scale
        records = run(env, act, n)
        row(name, records)
        report[name] = [
            {k: v for k, v in r.items() if k != "terminal_observation"} for r in records
        ]
    if args.json:
        args.json.write_text(json.dumps(json_ready(report), indent=2, allow_nan=False))
    if args.plot:
        plot(args.plot, tr, cfg, act, label)
    print()


if __name__ == "__main__":
    main()
