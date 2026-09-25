#!/usr/bin/env python3
"""Replay a policy from the start box and show where and how its runs end.

    python3 diagnose.py runs/v1/best_model.zip            # held-out layouts
    python3 diagnose.py runs/v1/best_model.zip --train    # training layouts
    python3 diagnose.py --prior                           # the prior alone

For every episode it prints the outcome, and for the last 2 s before the end
a trace of position, arc length, speed (true and commanded), steering (prior
and total), pitch, roll and clearance, plus a PNG of the tracks over the
course render.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import course_model
import layouts
import observation as O
import plant as P
from env import CLEARANCE_RINGS
from train import EvalEnv

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", type=Path)
    ap.add_argument(
        "--prior", action="store_true", help="drive the prior alone at a fixed throttle"
    )
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument(
        "--tail", type=float, default=2.0, help="seconds of trace before each end"
    )
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--png", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seeds = layouts.TRAIN_SEEDS + layouts.HELDOUT_SEEDS
    model = course_model.CourseModel(seeds)
    ids = (
        np.arange(len(layouts.TRAIN_SEEDS))
        if args.train
        else np.arange(len(layouts.TRAIN_SEEDS), len(seeds))
    )
    env = EvalEnv(
        cfg, model, args.per * len(ids), ids, seed=10_002, start_box_only=True
    )
    env.layout_ids_cycle = np.tile(ids, args.per)

    if args.prior:
        predict = lambda obs: env.scripted_action()  # noqa: E731
    else:
        from stable_baselines3 import PPO

        policy = PPO.load(args.model, device="cpu")
        predict = lambda obs: policy.predict(obs, deterministic=True)[0]  # noqa: E731

    obs = env.reset()
    n = env.n
    tracks = [[] for _ in range(n)]
    done = np.zeros(n, bool)
    results = [None] * n
    steps = int(cfg["env"]["episode_s"] * cfg["env"]["control_hz"]) + 5
    for _ in range(steps):
        a = predict(obs)
        prior = env.prior.copy()
        steer, vcmd = O.action_to_command(np.clip(a, -1, 1), prior, cfg)
        obs, _, term, trunc, info = env.step(a)
        st = env.plant.state
        clear = P.body_clearance(env.plant.OBS, env.plant.lay, st, CLEARANCE_RINGS)
        for i in range(n):
            if done[i]:
                continue
            if info[i]:
                pose = info[i]["pose"]
                tracks[i].append(
                    (
                        pose[0],
                        pose[1],
                        pose[2],
                        info[i]["s_end"],
                        pose[6],
                        vcmd[i],
                        prior[i],
                        steer[i],
                        np.degrees(pose[4]),
                        np.degrees(pose[5]),
                        np.nan,
                    )
                )
                results[i] = info[i]
                done[i] = True
            else:
                tracks[i].append(
                    (
                        st[i, P.S_X],
                        st[i, P.S_Y],
                        st[i, P.S_Z],
                        env.s[i],
                        st[i, P.S_V],
                        vcmd[i],
                        prior[i],
                        steer[i],
                        np.degrees(st[i, P.S_PITCH]),
                        np.degrees(st[i, P.S_ROLL]),
                        clear[i],
                    )
                )
        if done.all():
            break

    tail = int(args.tail * cfg["env"]["control_hz"])
    for i, r in enumerate(results):
        if r is None:
            continue
        print(
            f"\n#{i} seed {r['seed']} {r['outcome']}@{r['zone']}  s {r['s_end']:.1f}/{r['lap_length']:.1f} m "
            f"in {r['time']:.1f} s  hoops {r['hoops']}  min clear {r['min_clearance']:.3f}"
        )
        print(
            "     x      y      z      s     v   vcmd  prior steer  pitch  roll  clear"
        )
        for row in tracks[i][-tail:]:
            print(" ".join(f"{v:6.2f}" for v in row))

    speeds = np.concatenate([np.asarray(t)[:, 4] for t in tracks if t])
    print(
        f"\nmean speed {speeds.mean():.2f} m/s, 90th pct {np.percentile(speeds, 90):.2f}"
    )

    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        for line in env.lines.lines[ids[0] : ids[0] + 1]:
            ax.plot(line.points[:, 0], line.points[:, 1], "k--", lw=0.5)
        for i, t in enumerate(tracks):
            t = np.asarray(t)
            sc = ax.scatter(
                t[:, 0], t[:, 1], c=t[:, 4], s=2, cmap="viridis", vmin=0, vmax=3.5
            )
            ax.plot(t[-1, 0], t[-1, 1], "rx")
        fig.colorbar(sc, label="speed m/s")
        ax.set_aspect("equal")
        fig.savefig(args.png, dpi=120)
        print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
