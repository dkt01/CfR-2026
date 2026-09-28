#!/usr/bin/env python3
"""Time PPO updates on one real rollout without changing a training run.

    .venv/Scripts/python.exe bench_ppo.py runs/v7/best_model.zip

Uses 14 cached layouts to keep startup short, but the configured car count,
rollout length, policy and PPO minibatches. Each timed update starts from the
same weights, optimizer state and rollout. No checkpoint is written.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import course_model
import layouts
import ppo_policy
from env import ObstacleEnv
from train import sb3_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tcfg = cfg["train"]
    seeds = layouts.TRAIN_SEEDS[:10] + layouts.HELDOUT_SEEDS
    course = course_model.CourseModel(seeds)
    env = sb3_adapter(
        ObstacleEnv(cfg, course, int(tcfg["n_envs"]), np.arange(10), seed=0),
        float(tcfg["reward_scale"]),
    )
    model = ppo_policy.load(args.checkpoint, env=env, device="cpu")
    model.verbose = 0
    model.learn(
        total_timesteps=int(tcfg["n_envs"]) * int(tcfg["n_steps"]),
        reset_num_timesteps=True,
    )
    policy_state = copy.deepcopy(model.policy.state_dict())
    optimizer_state = copy.deepcopy(model.policy.optimizer.state_dict())

    # Interleave settings so background load affects both measurements.
    for threads, mkldnn in ((4, True), (2, False), (2, False), (4, True)):
        model.policy.load_state_dict(policy_state)
        model.policy.optimizer.load_state_dict(optimizer_state)
        torch.set_num_threads(threads)
        torch.backends.mkldnn.enabled = mkldnn
        start = time.perf_counter()
        model.train()
        print(
            f"threads={threads} mkldnn={mkldnn} "
            f"ppo_update={time.perf_counter() - start:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
