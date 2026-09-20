#!/usr/bin/env python3
"""PPO training entrypoint for the Obstacle Course driving policy.

Sibling to train.py -- see that file for the shared mechanics (checkpointing,
the deterministic-eval callback, always closing the env). This one wires up
ObstacleCourseEnv/ObstacleRewardConfig instead of the Speed Course's, and
needs a different launch:

    ros2 launch cfr_arduino_bridge obstacle_course.launch.py \
        sensors:=true autonomy:=false
    python train_obstacle.py

`sensors:=true` is not optional here: ObstacleCourseEnv perceives entirely
through the ZED's simulated point cloud (see obstacle_env.py's module
docstring for why), and that topic publishes nothing without it.

`autonomy:=false` is not optional either: without it that launch starts
path_follower_node, which publishes zeros on the same command topic this
environment drives, and the car ends up doing about a tenth of what the
policy asks for. See train_resilient_obstacle.sh.
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

from obstacle_env import ObstacleCourseEnv
from obstacle_reward import ObstacleRewardConfig
from zed_sim import ZedSimConfig

# Named stretches of the lap, by arc length, so "where does it stop" is a
# readable answer instead of a number. Boundaries follow the course's own
# features (see obstacle_course_path.py): the ramp and deck run to the
# helix at 9.1 m, the helix to 14.8, then the tunnel and the long south
# corridor, the gravel and the bank, the potholes and the climb north, and
# finally the buckets and hoops back to the line.
ZONES = (
    ("start_ramp_deck", 0.0, 9.1),
    ("helix", 9.1, 14.8),
    ("tunnel_south", 14.8, 26.0),
    ("gravel_bank", 26.0, 40.0),
    ("potholes_north", 40.0, 52.0),
    ("buckets_hoops", 52.0, 75.0),
)
ZONE_NAMES = tuple(name for name, _, _ in ZONES)


def zone_of(course_s: float) -> str:
    for name, low, high in ZONES:
        if low <= course_s < high:
            return name
    return ZONES[-1][0]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/obstacle_course.sdf"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_obstacle.yaml"


def build_env(config: dict, sdf_path: str, teleport_url: str) -> Monitor:
    env_config = config["env"]
    env = ObstacleCourseEnv(
        sdf_path=sdf_path,
        teleport_url=teleport_url,
        reward_config=ObstacleRewardConfig(**config["reward"]),
        zed_config=ZedSimConfig(**config.get("zed_sim", {})),
        **env_config,
    )
    return Monitor(env)


class EpisodeOutcomeCallback(BaseCallback):
    """Log *why* episodes end, not just how long they lasted.

    `ep_len_mean` and `ep_rew_mean` alone cannot tell a policy that is
    learning to drive from one that is learning to quit -- two runs died
    that way, and both were diagnosable only after the fact by reasoning
    about what the numbers had to mean. These counters make the failure
    modes legible while the run is going:

    - `outcome/collision_rate` climbing towards 1.0: the car is crashing,
      not driving.
    - `outcome/stuck_rate` climbing towards 1.0 with `ep_len_mean` pinned
      at the stuck window: the quit trap. This is the one to watch.
    - `outcome/timeout_rate` at 1.0 with `outcome/course_advance_mean`
      near zero: the opposite failure -- the car has learned that sitting
      still is safer than trying, which a big collision_penalty can cause
      once the per-step floor is zero.
    - `outcome/course_advance_mean` is the number that should go up. It is
      metres gained *from wherever the episode was dealt in*, so it stays
      comparable across dealt starts, unlike raw course_s.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset_counts()

    def reset_counts(self) -> None:
        self.episodes = 0
        self.collisions = 0
        self.stuck = 0
        self.hoop_misses = 0
        self.laps = 0
        self.advances: list[float] = []
        self.steps = 0
        self.speed_sum = 0.0
        self.abs_speed_sum = 0.0
        self.clearance_sum = 0.0
        self.cmd_speed_sum = 0.0
        self.dt_sum = 0.0
        self.reward_parts = dict.fromkeys(
            ("r_progress", "r_proximity", "r_touch", "r_smoothness"), 0.0
        )
        self.steer_abs_sum = 0.0
        self.yaw_abs_sum = 0.0
        self.true_gap_sum = 0.0
        self.wedged_steps = 0
        # Where round the course episodes end. "The car stalls after 3 m"
        # is a different problem depending on whether it stalls in the same
        # place every time (one feature is blocking it) or all over the
        # course (it cannot drive at all), and the mean alone cannot tell
        # those apart. See ZONES.
        self.zone_ends = dict.fromkeys(ZONE_NAMES, 0)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            # Every step, not just terminal ones: "the car is barely
            # advancing round the course" has two very different causes --
            # it is not moving, or it is moving and not getting anywhere --
            # and only the raw speed separates them.
            if "speed" in info:
                self.steps += 1
                self.speed_sum += float(info["speed"])
                self.abs_speed_sum += abs(float(info["speed"]))
                self.clearance_sum += float(info.get("min_clearance", 0.0))
                self.cmd_speed_sum += float(info.get("cmd_speed", 0.0))
                self.dt_sum += float(info.get("dt", 0.0))
                for part in self.reward_parts:
                    self.reward_parts[part] += float(info.get(part, 0.0))
                self.steer_abs_sum += abs(float(info.get("steer", 0.0)))
                self.yaw_abs_sum += abs(float(info.get("yaw_rate", 0.0)))
                true_gap = float(info.get("true_wall_gap", 99.0))
                self.true_gap_sum += true_gap
                # Touching a wall the forward scan says is not there: the
                # car's half-width is 0.15 m, so under 0.2 m is scraping.
                if true_gap < 0.2 and float(info.get("min_clearance", 0.0)) > 0.5:
                    self.wedged_steps += 1
            if not done:
                continue
            self.episodes += 1
            self.collisions += bool(info.get("collided"))
            self.stuck += bool(info.get("stuck"))
            self.hoop_misses += bool(info.get("hoop_missed"))
            self.laps += bool(info.get("lap_completed"))
            self.advances.append(float(info.get("course_advance", 0.0)))
            self.zone_ends[zone_of(float(info.get("course_s", 0.0)))] += 1
        return True

    def _on_rollout_end(self) -> None:
        if not self.episodes:
            return
        done = self.episodes
        ended = self.collisions + self.stuck + self.hoop_misses + self.laps
        self.logger.record("outcome/collision_rate", self.collisions / done)
        self.logger.record("outcome/stuck_rate", self.stuck / done)
        self.logger.record("outcome/hoop_miss_rate", self.hoop_misses / done)
        self.logger.record("outcome/lap_rate", self.laps / done)
        # Whatever is left ran out the clock.
        self.logger.record("outcome/timeout_rate", max(0, done - ended) / done)
        self.logger.record(
            "outcome/course_advance_mean", sum(self.advances) / len(self.advances)
        )
        self.logger.record("outcome/course_advance_max", max(self.advances))
        if self.steps:
            # speed_mean near zero with abs_speed_mean well above it means
            # the car is driving back and forth, not sitting still.
            self.logger.record("outcome/speed_mean", self.speed_sum / self.steps)
            self.logger.record(
                "outcome/abs_speed_mean", self.abs_speed_sum / self.steps
            )
            self.logger.record(
                "outcome/clearance_mean", self.clearance_sum / self.steps
            )
            self.logger.record(
                "outcome/cmd_speed_mean", self.cmd_speed_sum / self.steps
            )
            self.logger.record("outcome/dt_mean", self.dt_sum / self.steps)
            # Per episode, so they add up to roughly ep_rew_mean and can be
            # read against each other directly.
            for part, total in self.reward_parts.items():
                self.logger.record(f"reward/{part}", total / done)
            self.logger.record(
                "outcome/steer_abs_mean", self.steer_abs_sum / self.steps
            )
            self.logger.record("outcome/yaw_abs_mean", self.yaw_abs_sum / self.steps)
            self.logger.record("outcome/true_gap_mean", self.true_gap_sum / self.steps)
            # The share of steps spent scraping a wall the scan reports as
            # clear. If this is large, the car is being stopped by something
            # it has no way to perceive, and no amount of training fixes it.
            self.logger.record(
                "outcome/wedged_unseen_frac", self.wedged_steps / self.steps
            )
        for name, count in self.zone_ends.items():
            self.logger.record(f"zone_end/{name}", count / done)
        self.reset_counts()


class DeterministicEvalCallback(BaseCallback):
    """Periodically evaluate the *deterministic* policy and keep the best.

    Same rationale as train.py's -- see there. Tracks how far round the
    course the policy gets per deterministic episode as the selection
    metric: under a fixed episode cap, a policy that reliably covers more
    of the lap is also the faster one, and distance is a steadier signal
    this early than raw reward, which a single missed hoop can swing by
    hundreds of points.

    Always evaluates from the start line, never a dealt start -- see the
    `start_s` option in the episode loop below.
    """

    def __init__(
        self,
        raw_env: ObstacleCourseEnv,
        metadata: dict,
        save_dir: Path,
        # Halved when n_steps went 512 -> 2048, so evaluation still lands
        # about every 16k steps instead of every 33k.
        every_rollouts: int = 8,
        episodes: int = 3,
    ) -> None:
        super().__init__()
        self.raw_env = raw_env
        self.metadata = metadata
        self.save_dir = save_dir
        self.every_rollouts = every_rollouts
        self.episodes = episodes
        self.rollouts = 0
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
        hoops_missed = 0
        laps = 0
        for _ in range(self.episodes):
            # Always from the start line, never a dealt start: training
            # deals episodes in all round the lap (start_anywhere_prob), and
            # an eval that inherited that would measure the draw, not the
            # policy.
            observation, _ = self.raw_env.reset(options={"start_s": 0.0})
            reached = 0.0
            while True:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = self.raw_env.step(action)
                # How far round the course it got, not how far it drove:
                # summing displacement rewarded a policy for wandering.
                reached = max(reached, info["course_s"])
                if info.get("hoop_missed"):
                    hoops_missed += 1
                if info.get("lap_completed"):
                    laps += 1
                if terminated or truncated:
                    break
            distances.append(reached)

        mean_distance = sum(distances) / len(distances)
        self.logger.record("eval/deterministic_distance_m", mean_distance)
        self.logger.record("eval/hoops_missed", hoops_missed)
        self.logger.record("eval/laps_completed", laps)
        print(
            f"[deterministic eval] {self.num_timesteps} steps: "
            f"reached {mean_distance:.1f} m round the course over "
            f"{self.episodes} episodes "
            f"(best {max(self.best_distance, mean_distance):.1f}), "
            f"{laps} lap(s), {hoops_missed} hoop miss(es)"
        )
        if mean_distance > self.best_distance:
            self.best_distance = mean_distance
            best_path = self.save_dir / "best_model"
            self.model.save(str(best_path))
            with open(best_path.with_suffix(".json"), "w") as handle:
                json.dump(
                    {
                        **self.metadata,
                        "deterministic_distance_m": mean_distance,
                        "at_timesteps": self.num_timesteps,
                    },
                    handle,
                    indent=2,
                )

        reset_obs = self.model.env.reset()
        self.model._last_obs = reset_obs
        self.model._last_episode_starts = np.ones(
            (self.model.env.num_envs,), dtype=bool
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    parser.add_argument(
        "--teleport-url", default="http://localhost:9003/api/sim/teleport"
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(Path(__file__).resolve().parent / "checkpoints_obstacle"),
    )
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

    try:
        if args.resume_from:
            model = PPO.load(args.resume_from, env=env)
        else:
            model = PPO("MlpPolicy", env, verbose=1, **training_config)

        metadata = {
            "env": config["env"],
            "reward": config["reward"],
            "zed_sim": config.get("zed_sim", {}),
            "total_timesteps": total_timesteps,
        }

        callbacks = [
            CheckpointCallback(
                save_freq=max(1000, training_config.get("n_steps", 512)),
                save_path=str(checkpoint_dir),
                name_prefix="obstacle_course",
            ),
            EpisodeOutcomeCallback(),
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
