#!/usr/bin/env python3
"""Evaluate a trained bale-following checkpoint and report metrics.

Runs N deterministic episodes against a live simulation (started by
test_policy.sh or `ros2 launch cfr_arduino_bridge training.launch.py`),
routing the policy's raw commands through the CasADi smoother the same way
deployment would, and prints/saves per-episode and aggregate metrics:
distance covered, mean speed, collisions, minimum wall clearance, and
steering-command jerk (the thing the smoother exists to reduce).

    python evaluate.py --checkpoint checkpoints/final_model.zip --episodes 5

`--no-smoother` evaluates the raw policy for an A/B comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from casadi_smoother import CommandSmoother, smoother_from_metadata
from env import MAX_STEERING_ANGLE, WHEELBASE, BaleFollowerEnv
from reward import RewardConfig
from zed_sim import ZedSimConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"


def run_episode(
    env: BaleFollowerEnv, model: PPO, smoother: CommandSmoother | None,
    deterministic: bool = True,
) -> dict:
    observation, _ = env.reset()
    if smoother is not None:
        smoother.reset()

    distance = 0.0
    speeds: list[float] = []
    clearances: list[float] = []
    steer_jerk: list[float] = []
    reward_total = 0.0
    prev_delta = 0.0
    collided = stuck = False
    steps = 0

    while True:
        action, _ = model.predict(observation, deterministic=deterministic)
        speed, steer_fraction = env.decode_action(action)
        delta = steer_fraction * MAX_STEERING_ANGLE
        if smoother is not None:
            speed, delta = smoother.smooth(speed, delta)
            # Re-encode so env.step applies exactly the smoothed pair (its
            # traction clamp is a no-op on an already-feasible command).
            action = env.encode_action(speed, delta / MAX_STEERING_ANGLE)
        observation, reward, terminated, truncated, info = env.step(action)

        distance += max(0.0, info["progress_distance"])
        speeds.append(info["speed"])
        clearances.append(info["min_clearance"])
        steer_jerk.append(abs(delta - prev_delta))
        prev_delta = delta
        reward_total += reward
        steps += 1
        if terminated or truncated:
            collided = info["collided"]
            stuck = info["stuck"]
            break

    return {
        "steps": steps,
        "duration_s": steps / env.control_hz,
        "distance_m": round(distance, 2),
        "mean_speed": round(statistics.fmean(speeds), 2) if speeds else 0.0,
        "min_clearance": round(min(clearances), 3) if clearances else None,
        "mean_steer_jerk_rad": round(statistics.fmean(steer_jerk), 4) if steer_jerk else 0.0,
        "reward": round(reward_total, 1),
        "collided": collided,
        "stuck": stuck,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "checkpoints/final_model.zip"),
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    parser.add_argument("--teleport-url", default="http://localhost:9003/api/sim/teleport")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample the policy like training does instead of taking "
                             "the Gaussian mean; diagnoses train/eval gaps")
    # Default matches deployment: run_policy.py publishes raw commands for a
    # policy trained against the env's actuator limits, so evaluating through
    # the smoother would measure a configuration nobody runs.
    parser.add_argument("--smoother", action="store_true",
                        help="route commands through the CasADi smoother (A/B baseline, or "
                             "for checkpoints trained without env-side actuator limits)")
    parser.add_argument("--no-smoother", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-speed", type=float, default=None,
                        help="override the trained max speed cap")
    parser.add_argument("--traction", type=float, default=None,
                        help="override the trained friction coefficient")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="write results JSON here")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.exists():
        raise SystemExit(f"missing {metadata_path} (written by train.py alongside the checkpoint)")
    with open(metadata_path) as handle:
        metadata = json.load(handle)

    env_config = dict(metadata["env"])
    if args.max_speed is not None:
        env_config["max_speed"] = args.max_speed
    if args.traction is not None:
        env_config["traction"] = args.traction

    env = BaleFollowerEnv(
        sdf_path=args.sdf_path,
        teleport_url=args.teleport_url,
        reward_config=RewardConfig(**metadata["reward"]),
        zed_config=ZedSimConfig(**metadata.get("zed_sim", {})),
        **env_config,
    )
    model = PPO.load(str(checkpoint))
    smoother = None if (args.no_smoother or not args.smoother) else smoother_from_metadata(
        metadata, env.control_hz, env.traction, env.max_speed, WHEELBASE, MAX_STEERING_ANGLE
    )

    episodes = []
    try:
        for index in range(args.episodes):
            env.reset(seed=args.seed + index)  # seeds the start-pose RNG
            result = run_episode(env, model, smoother, deterministic=not args.stochastic)
            episodes.append(result)
            print(f"episode {index + 1}/{args.episodes}: "
                  f"{result['distance_m']:.1f} m in {result['duration_s']:.0f} s, "
                  f"mean {result['mean_speed']:.2f} m/s, "
                  f"{'COLLIDED' if result['collided'] else 'stuck' if result['stuck'] else 'clean'}")
    finally:
        env.close()

    clean = [e for e in episodes if not e["collided"]]
    summary = {
        "checkpoint": str(checkpoint),
        "smoother": smoother is not None,
        "deterministic": not args.stochastic,
        "max_speed": env_config["max_speed"],
        "traction": env_config.get("traction", 0.6),
        "episodes": len(episodes),
        "collision_rate": round(sum(e["collided"] for e in episodes) / len(episodes), 2),
        "mean_distance_m": round(statistics.fmean(e["distance_m"] for e in episodes), 2),
        "mean_speed": round(statistics.fmean(e["mean_speed"] for e in episodes), 2),
        "mean_steer_jerk_rad": round(statistics.fmean(e["mean_steer_jerk_rad"] for e in episodes), 4),
        "clean_episode_mean_distance_m": (
            round(statistics.fmean(e["distance_m"] for e in clean), 2) if clean else None
        ),
        "per_episode": episodes,
    }

    print("\n=== summary ===")
    for key, value in summary.items():
        if key != "per_episode":
            print(f"{key:32s} {value}")

    output = Path(args.output) if args.output else Path(
        f"eval_{checkpoint.stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
