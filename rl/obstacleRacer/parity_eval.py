#!/usr/bin/env python3
"""Evaluate an exported policy in Numba on the same layouts Gazebo validates.

    python3 parity_eval.py runs/v8/policy.npz --seeds 201,202,208,218 --starts 5

The random start offsets are independent of gazebo_check.py's draws. Compare
rates over enough starts, not individual trajectories. The exported .npz is
used here so the Numba and Gazebo drivers execute the same policy weights.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import numba as nb
import yaml

import course_model
import plant as P
from env import ObstacleEnv
from policy import NumpyPolicy

HERE = Path(__file__).resolve().parent


class EvalEnv(ObstacleEnv):
    layout_ids_cycle = None
    _next = 0

    def reset(self):
        self._next = 0
        return super().reset()

    def _pick_layouts(self, k):
        ids = self.layout_ids_cycle
        if ids is None:
            return super()._pick_layouts(k)
        pick = ids[(self._next + np.arange(k)) % len(ids)]
        self._next += k
        return pick


def summarize(records):
    finishes = [r for r in records if r["outcome"] == "finish"]
    by_seed = {}
    for record in records:
        by_seed.setdefault(record["seed"], []).append(record["outcome"] == "finish")
    return dict(
        n=len(records),
        finish=len(finishes) / len(records),
        finish_by_seed={
            str(seed): float(np.mean(values))
            for seed, values in sorted(by_seed.items())
        },
        ends=dict(
            Counter(
                f"{r['outcome']}@{r['zone']}"
                for r in records
                if r["outcome"] != "finish"
            )
        ),
    )


class ExportedDriver:
    def __init__(self, path: Path, n: int):
        self.policy = NumpyPolicy.load(path)
        self.policy.reset(n)

    def __call__(self, obs):
        return self.policy.act(obs)

    def ended(self, mask):
        if self.policy.h is not None:
            self.policy.h[mask] = 0.0
            self.policy.c[mask] = 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("policy", type=Path)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--seeds", default="201,202,208,218")
    ap.add_argument("--starts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=10_002)
    ap.add_argument(
        "--threads",
        type=int,
        default=1,
        help="one thread keeps Numba's sensor noise reproducible",
    )
    args = ap.parse_args()

    nb.set_num_threads(args.threads)
    seeds = [int(s) for s in args.seeds.split(",")]
    cfg = yaml.safe_load(args.config.read_text())
    policy = NumpyPolicy.load(args.policy)
    from observation import obs_dim

    if policy.obs_dim != obs_dim(cfg):
        raise SystemExit("policy observation width does not match config")
    model = course_model.CourseModel(seeds)
    env = EvalEnv(
        cfg,
        model,
        args.starts * len(seeds),
        np.arange(len(seeds)),
        seed=args.seed,
        start_box_only=True,
    )
    env.layout_ids_cycle = np.tile(np.arange(len(seeds)), args.starts)
    obs = env.reset()
    starts = env.plant.state[:, [P.S_X, P.S_Y, P.S_Z, P.S_YAW]].copy()
    driver = ExportedDriver(args.policy, env.n)
    complete = np.zeros(env.n, bool)
    records = []
    limit = int(cfg["env"]["episode_s"] * cfg["env"]["control_hz"]) + 5
    for _ in range(limit):
        obs, _, terminated, truncated, info = env.step(driver(obs))
        driver.ended(terminated | truncated)
        for i, result in enumerate(info):
            if result and not complete[i]:
                result["start_pose"] = starts[i].tolist()
                records.append(result)
                complete[i] = True
        if complete.all():
            break
    if not complete.all():
        raise SystemExit(f"{(~complete).sum()} episodes did not end")
    summary = summarize(records)
    out = HERE / "runs" / "gazebo" / f"numba_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(
                seeds=seeds,
                starts=args.starts,
                config=str(args.config),
                yaw_response_tau=cfg["plant"]["yaw_response_tau"],
                summary=summary,
                episodes=records,
            ),
            indent=1,
            default=lambda value: (
                value.tolist() if hasattr(value, "tolist") else str(value)
            ),
        )
    )
    print(f"{summary['n']} Numba starts: {100 * summary['finish']:.1f}% finish")
    print(f"by seed: {summary['finish_by_seed']}")
    print(f"endings: {summary['ends']}")
    print(f"results: {out}")


if __name__ == "__main__":
    main()
