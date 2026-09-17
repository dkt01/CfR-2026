#!/usr/bin/env python3
"""PPO training entrypoint for the bale-following policy.

Expects a paused Gazebo simulation already running:

    ros2 launch cfr_arduino_bridge training.launch.py

Training drives simulation time itself via WorldControl multi_step, so this
runs as fast as the physics allows rather than at real-time factor 1.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env import BaleFollowerEnv
from reward import RewardConfig
from zed_sim import ZedSimConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def build_env(config: dict, sdf_path: str, teleport_url: str) -> Monitor:
    env_config = config["env"]
    env = BaleFollowerEnv(
        sdf_path=sdf_path,
        teleport_url=teleport_url,
        reward_config=RewardConfig(**config["reward"]),
        zed_config=ZedSimConfig(**config.get("zed_sim", {})),
        **env_config,
    )
    return Monitor(env)


class DeterministicEvalCallback(BaseCallback):
    """Periodically evaluate the *deterministic* policy and keep the best.

    The v3 run showed why this exists: stochastic training curves climbed
    steadily while the deterministic mean action was degenerate (throttle
    floored, crashing in seconds). Training-curve reward selects for the
    sampled policy; deployment runs the mean. This callback runs a few
    deterministic episodes between rollouts and saves `best_model.zip`
    whenever mean distance improves, so checkpoint selection tracks the
    deployment metric.

    Runs on the same single env between rollouts (one Gazebo instance is all
    there is), then resets it and resyncs the collector's cached observation
    so PPO's next rollout starts from a coherent state.
    """

    def __init__(self, raw_env: BaleFollowerEnv, metadata: dict, save_dir: Path,
                 every_rollouts: int = 16, episodes: int = 3) -> None:
        super().__init__()
        self.raw_env = raw_env
        self.metadata = metadata
        self.save_dir = save_dir
        self.every_rollouts = every_rollouts
        self.episodes = episodes
        self.rollouts = 0
        # Carry the previous best across a resumed run, or an early mediocre
        # eval overwrites a better checkpoint from before the restart.
        self.best_distance = -float("inf")
        best_json = save_dir / "best_model.json"
        if best_json.exists():
            with open(best_json) as handle:
                self.best_distance = json.load(handle).get(
                    "deterministic_distance_m", -float("inf")
                )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts += 1
        if self.rollouts % self.every_rollouts != 0:
            return

        distances = []
        for _ in range(self.episodes):
            observation, _ = self.raw_env.reset()
            distance = 0.0
            while True:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = self.raw_env.step(action)
                distance += max(0.0, info["progress_distance"])
                if terminated or truncated:
                    break
            distances.append(distance)

        mean_distance = sum(distances) / len(distances)
        self.logger.record("eval/deterministic_distance_m", mean_distance)
        print(f"[deterministic eval] {self.num_timesteps} steps: "
              f"mean {mean_distance:.1f} m over {self.episodes} episodes "
              f"(best {max(self.best_distance, mean_distance):.1f})")
        if mean_distance > self.best_distance:
            self.best_distance = mean_distance
            best_path = self.save_dir / "best_model"
            self.model.save(str(best_path))
            with open(best_path.with_suffix(".json"), "w") as handle:
                json.dump({**self.metadata, "deterministic_distance_m": mean_distance,
                           "at_timesteps": self.num_timesteps}, handle, indent=2)

        # Our episodes drove the sim; hand PPO a fresh episode and matching
        # cached observation or its next rollout pairs stale obs with the
        # teleported car.
        reset_obs = self.model.env.reset()
        self.model._last_obs = reset_obs
        self.model._last_episode_starts = np.ones((self.model.env.num_envs,), dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    parser.add_argument("--teleport-url", default="http://localhost:9003/api/sim/teleport")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    training_config = config["training"]
    total_timesteps = args.total_timesteps or training_config.pop("total_timesteps")
    training_config.pop("total_timesteps", None)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(config, args.sdf_path, args.teleport_url)

    # Always close the env: leaving the background rclpy executor running on
    # an abnormal exit takes the interpreter down hard, which has been enough
    # to wedge the Gazebo server's service layer for the next run.
    try:
        if args.resume_from:
            model = PPO.load(args.resume_from, env=env)
        else:
            model = PPO("MlpPolicy", env, verbose=1, **training_config)

        # run_policy.py validates against this so a checkpoint can never be
        # run with mismatched observation geometry.
        # The steering slew the policy trained under has to travel with the
        # checkpoint: a policy trained without it fails the moment a
        # rate-limited controller is placed in front of it.
        metadata = {
            "env": config["env"],
            "reward": config["reward"],
            "zed_sim": config.get("zed_sim", {}),
            "smoother": config.get("smoother", {}),
            "total_timesteps": total_timesteps,
        }

        callbacks = [
            CheckpointCallback(
                save_freq=max(1000, training_config.get("n_steps", 512)),
                save_path=str(checkpoint_dir),
                name_prefix="bale_follower",
            ),
            DeterministicEvalCallback(
                raw_env=env.unwrapped, metadata=metadata, save_dir=checkpoint_dir
            ),
        ]
        model.learn(total_timesteps=total_timesteps, callback=callbacks)

        final_path = checkpoint_dir / "final_model"
        model.save(str(final_path))
        with open(checkpoint_dir / "final_model.json", "w") as handle:
            json.dump(metadata, handle, indent=2)

        print(f"saved {final_path}.zip and metadata")
    finally:
        env.close()


if __name__ == "__main__":
    main()
