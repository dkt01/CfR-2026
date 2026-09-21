#!/usr/bin/env python3
"""PPO training for the lap-time policy.

    ./launch_lap_training.sh --max-speed 3.0 --total-timesteps 150000 \
        --checkpoint-dir checkpoints_lap1
    ./launch_lap_training.sh --resume-from checkpoints_lap1/best_model.zip \
        --checkpoint-dir checkpoints_lap2

Two stages, deliberately. A fresh policy that meets 4.5 m/s learns to brake
with its exploration noise rather than its mean action -- the v3 failure,
which cost a 100k-step run -- so the first stage caps the speed low enough
that the mean has to do the work, and the second lifts the cap onto a policy
that already knows the corridor.

Differences from `train.py` that matter:

* **Reward normalisation.** Returns here run to ~2000 (three laps plus
  bonuses) against ~200 for the corridor objective, and a value function
  chasing that range is slow to settle. `VecNormalize(norm_obs=False,
  norm_reward=True)` rescales the advantage signal only -- the policy still
  sees raw observations, so nothing about deployment changes and no
  normalisation statistics have to travel with the checkpoint.
* **Selection on pace, not distance.** `train.py` keeps the checkpoint that
  drove furthest. Distance saturates the moment a policy can complete the
  lap target, and every later improvement is in the clock.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from lap_env import LapRacerEnv
from lap_reward import LapRewardConfig
from zed_sim import ZedSimConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_lap.yaml"

# What the policy's inputs and outputs mean. Stamped into every checkpoint so
# a runtime cannot pair a policy with the wrong observation or, worse, feed
# its steering output in as an angle when it is a rate.
OBSERVATION_LAYOUT = (
    "[scan_t (num_lidar_bins, /lidar_max_range)] + "
    "[scan_t-1 .. scan_t-scan_history] + "
    "[speed (signed, (v+reverse_speed)/(max_speed+reverse_speed)), "
    "yaw_rate ((w+3)/6), steering_angle ((delta/0.40+1)/2), recovering (0|1)]"
)
ACTION_LAYOUT = (
    "[throttle -> target speed in [-reverse_speed, max_speed]; "
    "steering RATE as a fraction of max_steering_rate, integrated by the env]"
)


def build_env(config: dict, sdf_path: str, teleport_url: str,
              max_speed: float | None = None) -> LapRacerEnv:
    env_config = dict(config["env"])
    if max_speed is not None:
        env_config["max_speed"] = max_speed
    return LapRacerEnv(
        sdf_path=sdf_path,
        teleport_url=teleport_url,
        lap_reward_config=LapRewardConfig(**config["reward"]),
        zed_config=ZedSimConfig(**config.get("zed_sim", {})),
        **env_config,
    )


def episode_score(episode: dict, time_budget: float, lap_target: int) -> float:
    """One number to rank checkpoints by, in metres.

    Arc covered, less 60 m for a collision (over half a lap: no amount of
    pace buys one back), plus -- for an episode that completed its laps --
    the time it left on the clock at 2 m/s, which makes a lap 5 s quicker
    worth about 10 m of extra distance.
    """
    score = episode["s_progress"]
    if episode["collided"]:
        score -= 60.0
    # Time left on the clock is paid ONLY to an episode that actually
    # finished its laps. Paying it unconditionally ranks a policy by how
    # SOON its episode ended, whichever way it ended: measured, a crash at
    # the first hairpin (22 m, 12 s) scored 238 while a crash after a full
    # lap (110 m, 60 s) scored 230, so every real improvement was rejected
    # in favour of dying sooner. An entire 200k-step run selected on that.
    if episode["laps"] >= lap_target:
        score += 2.0 * max(0.0, time_budget - episode["elapsed_s"])
    return score


def run_episode(env: LapRacerEnv, model: PPO) -> dict:
    observation, _ = env.reset()
    speeds, clearances, rates = [], [], []
    recoveries, steps = 0, 0
    was_recovering = False
    info: dict = {}
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        speeds.append(info["speed"])
        clearances.append(info["min_clearance"])
        rates.append(info["steer_rate"])
        if info["recovering"] and not was_recovering:
            recoveries += 1
        was_recovering = info["recovering"]
        steps += 1
        if terminated or truncated:
            break
    lap_times = info.get("lap_times", [])
    return {
        "steps": steps,
        "elapsed_s": round(env._episode_time, 1),
        "laps": info.get("laps", 0),
        "lap_times": [round(t, 2) for t in lap_times],
        "best_lap_s": round(min(lap_times), 2) if lap_times else None,
        "s_progress": round(info.get("s_progress", 0.0), 1),
        "mean_speed": round(statistics.fmean(speeds), 2) if speeds else 0.0,
        "min_clearance": round(min(clearances), 3) if clearances else None,
        "mean_steer_rate": round(statistics.fmean(rates), 3) if rates else 0.0,
        "recoveries": recoveries,
        "collided": bool(info.get("collided", False)),
        "stuck": bool(info.get("stuck", False)),
    }


class LapEvalCallback(BaseCallback):
    """Deterministic evaluation, keeping the best checkpoint by pace."""

    def __init__(self, raw_env: LapRacerEnv, metadata: dict, save_dir: Path,
                 every_rollouts: int = 8, episodes: int = 3) -> None:
        super().__init__()
        self.raw_env = raw_env
        self.metadata = metadata
        self.save_dir = save_dir
        self.every_rollouts = every_rollouts
        self.episodes = episodes
        self.rollouts = 0
        self.best_score = -float("inf")
        best_json = save_dir / "best_model.json"
        if best_json.exists():
            with open(best_json) as handle:
                self.best_score = json.load(handle).get("score", -float("inf"))

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollouts += 1
        if self.rollouts % self.every_rollouts != 0:
            return

        budget = self.raw_env.episode_time_limit_s
        episodes = [run_episode(self.raw_env, self.model) for _ in range(self.episodes)]
        score = statistics.fmean(
            episode_score(e, budget, self.raw_env.max_laps) for e in episodes)
        laps = [e["laps"] for e in episodes]
        best_laps = [e["best_lap_s"] for e in episodes if e["best_lap_s"]]
        collisions = sum(e["collided"] for e in episodes)

        self.logger.record("eval/score", score)
        self.logger.record("eval/laps", statistics.fmean(laps))
        self.logger.record("eval/s_progress_m",
                           statistics.fmean(e["s_progress"] for e in episodes))
        self.logger.record("eval/collision_rate", collisions / len(episodes))
        self.logger.record("eval/mean_speed",
                           statistics.fmean(e["mean_speed"] for e in episodes))
        self.logger.record("eval/recoveries",
                           statistics.fmean(e["recoveries"] for e in episodes))
        if best_laps:
            self.logger.record("eval/best_lap_s", min(best_laps))
        print(
            f"[lap eval] {self.num_timesteps} steps: score {score:.0f} "
            f"(best {max(self.best_score, score):.0f}), laps {laps}, "
            f"best lap {min(best_laps) if best_laps else '-'} s, "
            f"collisions {collisions}/{len(episodes)}",
            flush=True,
        )

        if score > self.best_score:
            self.best_score = score
            path = self.save_dir / "best_model"
            self.model.save(str(path))
            normaliser = self.model.get_vec_normalize_env()
            if normaliser is not None:
                normaliser.save(str(self.save_dir / "vecnormalize.pkl"))
            with open(path.with_suffix(".json"), "w") as handle:
                json.dump({**self.metadata, "score": score,
                           "at_timesteps": self.num_timesteps,
                           "episodes": episodes}, handle, indent=2)

        reset_obs = self.model.env.reset()
        self.model._last_obs = reset_obs
        self.model._last_episode_starts = np.ones(
            (self.model.env.num_envs,), dtype=bool
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    parser.add_argument("--teleport-url",
                        default="http://localhost:9003/api/sim/teleport")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--max-speed", type=float, default=None,
                        help="override env.max_speed, for the speed curriculum")
    parser.add_argument("--checkpoint-dir",
                        default=str(Path(__file__).resolve().parent / "checkpoints_lap"))
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="override training.learning_rate. Fine-tuning a cloned policy "
             "wants a smaller one than training from scratch: the value "
             "function starts untrained, so the first rollouts produce noisy "
             "advantages, and at 3e-4 those can undo the clone before the "
             "critic is worth listening to.")
    parser.add_argument(
        "--ent-coef", type=float, default=None,
        help="override training.ent_coef. The clone is saved with a small "
             "action std on purpose; an entropy bonus inflates it back.")
    parser.add_argument(
        "--throttle-bias", type=float, default=0.5,
        help="initial bias on the throttle output of a FRESH policy, in "
             "action units. The throttle maps [-1, 1] to "
             "[-reverse_speed, max_speed], so an untrained mean of 0 asks "
             "for 0.4 m/s and PPO has to learn to move before it can learn "
             "to drive; 0.5 starts it at ~1.2 m/s instead. Changes where "
             "exploration starts, not what is being optimised.")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    training_config = dict(config["training"])
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    if args.ent_coef is not None:
        training_config["ent_coef"] = args.ent_coef
    total_timesteps = args.total_timesteps or training_config.pop("total_timesteps")
    training_config.pop("total_timesteps", None)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_env = build_env(config, args.sdf_path, args.teleport_url, args.max_speed)
    dummy_env = DummyVecEnv([lambda: Monitor(raw_env)])
    # The running reward scale has to survive a resume: the sim dies every
    # few hours and `train_resilient.sh` restarts the chunk, and a fresh
    # normaliser would hand the value function a step change in target scale
    # at exactly the point it had settled.
    stats_path = checkpoint_dir / "vecnormalize.pkl"
    if args.resume_from and stats_path.exists():
        vec_env = VecNormalize.load(str(stats_path), dummy_env)
        vec_env.training = True
        print(f"resumed reward normalisation from {stats_path}")
    else:
        vec_env = VecNormalize(
            dummy_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=50.0,
            gamma=training_config.get("gamma", 0.99),
        )

    try:
        if args.resume_from:
            # verbose travels with the checkpoint, and a behaviour-cloned one
            # is saved with verbose=0 -- which silently removed every rollout
            # table from a 200k-step run's log.
            model = PPO.load(args.resume_from, env=vec_env, verbose=1)
            # PPO.load restores the hyperparameters the checkpoint was saved
            # with, so a command-line override has to be re-applied on top.
            if args.learning_rate is not None:
                model.learning_rate = args.learning_rate
                model.lr_schedule = lambda _progress: args.learning_rate
            if args.ent_coef is not None:
                model.ent_coef = args.ent_coef
            print(f"resumed {args.resume_from} at lr {model.lr_schedule(1.0):.1e}, "
                  f"ent_coef {model.ent_coef}, action std "
                  f"{float(model.policy.log_std.exp().mean()):.2f}", flush=True)
        else:
            model = PPO("MlpPolicy", vec_env, verbose=1, **training_config)
            if args.throttle_bias:
                with torch.no_grad():
                    model.policy.action_net.bias[0] += args.throttle_bias
                speed = raw_env.decode_action(
                    np.array([args.throttle_bias, 0.0]))[0]
                print(f"initial throttle bias {args.throttle_bias:+.2f} "
                      f"-> untrained mean asks for {speed:.2f} m/s", flush=True)

        metadata = {
            "objective": "lap_time",
            "env": {**config["env"],
                    **({"max_speed": args.max_speed} if args.max_speed else {})},
            "reward": config["reward"],
            "zed_sim": config.get("zed_sim", {}),
            "observation": OBSERVATION_LAYOUT,
            "action": ACTION_LAYOUT,
            "course_path": "course_path.json",
            "total_timesteps": total_timesteps,
            "resumed_from": args.resume_from,
            "throttle_bias": args.throttle_bias if not args.resume_from else None,
        }
        with open(checkpoint_dir / "run_metadata.json", "w") as handle:
            json.dump(metadata, handle, indent=2)

        callbacks = [
            CheckpointCallback(
                save_freq=max(2000, training_config.get("n_steps", 2048)),
                save_path=str(checkpoint_dir),
                name_prefix="lap_racer",
            ),
            LapEvalCallback(raw_env=raw_env, metadata=metadata,
                            save_dir=checkpoint_dir),
        ]
        model.learn(total_timesteps=total_timesteps, callback=callbacks)

        final_path = checkpoint_dir / "final_model"
        model.save(str(final_path))
        vec_env.save(str(stats_path))
        with open(checkpoint_dir / "final_model.json", "w") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"saved {final_path}.zip and metadata")
    finally:
        raw_env.close()


if __name__ == "__main__":
    main()
