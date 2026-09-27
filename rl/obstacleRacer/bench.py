#!/usr/bin/env python3
"""Environment throughput, where the step's time goes, and a same-result check.

    python3 bench.py                       # steps/s and the split, 256 cars
    python3 bench.py --save base.npz       # record obs/reward/done for --check
    python3 bench.py --check base.npz      # same seed, same results?
    python3 bench.py --policy runs/v5/best_model.zip   # cars driven by a policy

Random actions drive most cars into a wall; --policy gives the states, and
so the ray lengths, that training actually meets.

A speedup that is meant to change nothing is checked with --save before it
and --check after.  Both run numba on one thread: its per-thread random
generators make a parallel run's noise depend on how cars were scheduled.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="cars")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--check", type=Path)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--policy", type=Path, help="SB3 zip to drive with (deterministic)")
    args = ap.parse_args()
    exact = args.save or args.check
    if exact:
        os.environ["NUMBA_NUM_THREADS"] = "1"

    import numpy as np
    import yaml

    import course_model
    import env as env_module
    import layouts

    cfg = yaml.safe_load(args.config.read_text())
    model = course_model.CourseModel(layouts.TRAIN_SEEDS + layouts.HELDOUT_SEEDS)
    env = env_module.ObstacleEnv(
        cfg, model, args.n, np.arange(len(layouts.TRAIN_SEEDS)), seed=0
    )

    spent = defaultdict(float)

    def timed(owner, name, label):
        fn = getattr(owner, name)

        def wrapper(*a, **kw):
            t = time.perf_counter()
            out = fn(*a, **kw)
            spent[label] += time.perf_counter() - t
            return out

        setattr(owner, name, wrapper)

    timed(env.sensor, "read", "sensor")
    timed(env.plant, "step", "plant")
    timed(env_module.P, "body_clearance", "clearance")
    timed(env_module.O, "prior_steer", "prior")

    if args.policy:
        from stable_baselines3 import PPO

        policy = PPO.load(args.policy, device="cpu")

        def act(obs, k):
            return policy.predict(obs, deterministic=True)[0]
    else:
        rng = np.random.default_rng(1)
        actions = rng.uniform(-1.0, 1.0, (args.warmup + args.steps, args.n, 2))

        def act(obs, k):
            return actions[k]

    obs = env.reset()
    record = {"obs": [obs.copy()], "rew": [], "done": []}
    for k in range(args.warmup):
        obs, rew, term, trunc, _ = env.step(act(obs, k))
        record["obs"].append(obs.copy())
        record["rew"].append(rew)
        record["done"].append(term | trunc)
    spent.clear()
    wall = 0.0
    for k in range(args.warmup, args.warmup + args.steps):
        a = act(obs, k)
        t0 = time.perf_counter()
        obs, rew, term, trunc, _ = env.step(a)
        wall += time.perf_counter() - t0
        if exact:
            record["obs"].append(obs.copy())
            record["rew"].append(rew)
            record["done"].append(term | trunc)

    ms = 1e3 * wall / args.steps
    threads = "1 thread (exact mode)" if exact else f"{os.cpu_count()} cpus"
    print(f"{args.n} cars, {args.steps} steps, {threads}")
    print(f"  {args.n * args.steps / wall:8.0f} car-steps/s   {ms:6.1f} ms/step")
    rest = wall - sum(spent.values())
    for label, t in sorted(spent.items(), key=lambda kv: -kv[1]) + [("rest", rest)]:
        print(f"  {label:10s} {1e3 * t / args.steps:6.1f} ms  {100 * t / wall:4.0f}%")

    if exact:
        rec = {k: np.stack(v) for k, v in record.items()}
        if args.save:
            np.savez_compressed(args.save, **rec)
            print(f"saved {args.save}")
        if args.check:
            ref = np.load(args.check)
            ok = True
            for k in rec:
                if ref[k].shape != rec[k].shape:
                    print(f"  {k}: shape {rec[k].shape} vs {ref[k].shape}")
                    ok = False
                    continue
                diff = np.abs(rec[k].astype(np.float64) - ref[k].astype(np.float64))
                bad = np.argwhere(diff > 1e-9)
                if len(bad):
                    ok = False
                    print(
                        f"  {k}: {len(bad)} values differ, first at {tuple(bad[0])}, "
                        f"max {diff.max():.3g}"
                    )
            print("MATCH" if ok else "DIFFERENT")
            sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
