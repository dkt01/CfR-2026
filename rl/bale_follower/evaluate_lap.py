#!/usr/bin/env python3
"""Measure a lap-time checkpoint: pace, contact, smoothness, recoveries.

    ./test_lap_policy.sh --checkpoint checkpoints_lap2/best_model.zip --episodes 5

Reports what the objective is actually about -- lap times -- alongside the
two things that can make a fast number meaningless: whether the car touched
anything, and whether the steering was executable. The reward configuration
comes from the checkpoint's metadata, not from config_lap.yaml, so a
checkpoint is always scored under the shaping it was trained with.

No CasADi smoother option, unlike `evaluate.py`. A policy whose action IS the
servo's rate cannot produce a command the servo cannot execute, so there is
nothing left for a smoother to project -- and v6 measured that putting one in
front of a policy trained against the limits costs 24% of the distance.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml
from stable_baselines3 import PPO

from lap_env import LapRacerEnv
from lap_reward import LapRewardConfig
from train_lap import DEFAULT_CONFIG, DEFAULT_SDF, run_episode
from zed_sim import ZedSimConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    parser.add_argument("--teleport-url",
                        default="http://localhost:9003/api/sim/teleport")
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--from-start", action="store_true",
                        help="start every episode at the SDF spawn pose, as a "
                             "competition run does, instead of anywhere on the loop")
    parser.add_argument("--output", default=None, help="write metrics JSON here")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    metadata_path = checkpoint.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        print(f"checkpoint metadata: {metadata_path.name} "
              f"(objective {metadata.get('objective', 'unknown')})")
    else:
        metadata = yaml.safe_load(Path(args.config).read_text())
        print(f"no metadata beside {checkpoint.name}; falling back to "
              f"{Path(args.config).name}")

    env_config = dict(metadata["env"])
    if args.max_speed is not None:
        env_config["max_speed"] = args.max_speed
    if args.from_start:
        env_config["randomize_start"] = False
        env_config["wedged_start_prob"] = 0.0

    env = LapRacerEnv(
        sdf_path=args.sdf_path,
        teleport_url=args.teleport_url,
        lap_reward_config=LapRewardConfig(**metadata["reward"]),
        zed_config=ZedSimConfig(**metadata.get("zed_sim", {})),
        **env_config,
    )
    model = PPO.load(str(checkpoint))

    try:
        episodes = [run_episode(env, model) for _ in range(args.episodes)]
    finally:
        env.close()

    for index, episode in enumerate(episodes, start=1):
        print(f"  episode {index}: {episode['laps']} lap(s) "
              f"{episode['lap_times']}, {episode['s_progress']} m in "
              f"{episode['elapsed_s']} s, mean {episode['mean_speed']} m/s, "
              f"clearance {episode['min_clearance']} m, "
              f"{episode['recoveries']} recoveries"
              + (", COLLIDED" if episode["collided"] else "")
              + (", stuck" if episode["stuck"] else ""))

    laps = [e["best_lap_s"] for e in episodes if e["best_lap_s"]]
    summary = {
        "checkpoint": str(checkpoint),
        "episodes": args.episodes,
        "laps_completed": sum(e["laps"] for e in episodes),
        "best_lap_s": min(laps) if laps else None,
        "mean_lap_s": round(statistics.fmean(
            t for e in episodes for t in e["lap_times"]), 2) if laps else None,
        "mean_s_progress_m": round(
            statistics.fmean(e["s_progress"] for e in episodes), 1),
        "mean_speed": round(statistics.fmean(e["mean_speed"] for e in episodes), 2),
        "min_clearance_m": min(e["min_clearance"] for e in episodes),
        "mean_steer_rate_rad_s": round(
            statistics.fmean(e["mean_steer_rate"] for e in episodes), 3),
        "collision_rate": sum(e["collided"] for e in episodes) / args.episodes,
        "stuck_rate": sum(e["stuck"] for e in episodes) / args.episodes,
        "recoveries_per_episode": round(
            statistics.fmean(e["recoveries"] for e in episodes), 2),
        "per_episode": episodes,
    }
    print("\n" + json.dumps({k: v for k, v in summary.items()
                             if k != "per_episode"}, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
