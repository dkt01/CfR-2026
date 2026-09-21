#!/usr/bin/env python3
"""Clone the scripted driver into the policy, then let PPO race it.

Measured on this course: a car that does not steer touches a bale after a
median of 2.56 m, even starting exactly on the racing line pointed exactly
the right way. The serpentine has no straight to coast down -- its
"straight-ish" 83% still bends 43 degrees over 2.5 m of travel.

That is a bootstrapping trap for PPO. It cannot steer, so episodes end after
15 steps, so it never collects enough experience about what steering does to
learn to steer. Three runs from scratch confirmed it: the policy reliably
learns the throttle (whose effect on reward is immediate) and never learns
the steering (whose effect arrives a second later, after the episode has
already ended).

But a competent driver already exists -- `lap_driver.PursuitDriver`, which
laps in 47 s -- so the task is not to discover driving, it is to beat 47 s.
This clones the driver's actions into the policy first, and hands PPO a
policy that can already get round. Standard teacher/student with a
privileged teacher: the teacher reads the planned line and the true pose,
the student sees only the ZED-shaped scan and its own proprioception, so
what is learned is deployable even though what taught it was not.

    ./pretrain_resilient.sh 20000          # collect, surviving sim deaths
    python pretrain_lap.py --fit           # fit, needing no simulator at all
    ./launch_lap_training.sh --resume-from checkpoints_lap0/pretrained.zip \
        --max-speed 2.0 --checkpoint-dir checkpoints_lap1

Collection and fitting are separate phases because the simulator dies every
15-30 minutes with the ZED rendering (the 5 h figure in REPORT.md predates
`CFR_SENSORS`), and losing a fit because the thing that produced its data
has already gone is pointless. Samples are appended to an .npz as they are
collected, so a death costs the last few hundred of them and nothing else.

A tenth of the teacher's steps are deliberately perturbed. Without them the
student only ever sees the states a good driver visits, and the first time
PPO's own noise puts it 20 cm off the line it is somewhere it has no idea
about -- the classic imitation-learning failure. The teacher's correction
from the perturbed state is exactly the data that fixes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from lap_driver import PursuitDriver
from lap_env import LapRacerEnv
from lap_reward import LapRewardConfig
from train_lap import (ACTION_LAYOUT, DEFAULT_CONFIG, DEFAULT_SDF,
                       OBSERVATION_LAYOUT, build_env)
from zed_sim import ZedSimConfig


def load_samples(path: Path):
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["observations"], data["actions"]


def save_samples(path: Path, observations, actions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observations=np.asarray(observations, np.float32),
                        actions=np.asarray(actions, np.float32))


def collect(env: LapRacerEnv, driver: PursuitDriver, steps: int,
            perturb_prob: float, rng: np.random.Generator,
            samples_path: Path, existing=(None, None),
            crash_dropout: int = 10):
    """Teacher rollouts: the student's observation, the teacher's action.

    The last `crash_dropout` samples before a collision are thrown away: they
    are the ones that drove into the bale, and cloning them teaches the
    student to do the same. Pure pursuit has no recovery behaviour, so it
    does crash -- out of the wedged starts especially -- and those episodes
    are still worth their earlier, good samples.
    """
    observations = list(existing[0]) if existing[0] is not None else []
    actions = list(existing[1]) if existing[1] is not None else []
    episode_start = len(observations)
    observation, _ = env.reset()
    episodes, collisions, dropped = 1, 0, 0
    while len(observations) < steps:
        pose = env._prev_pose
        action = driver.action(pose.x, pose.y, pose.yaw)
        # Record what the teacher would do HERE, then sometimes do something
        # else -- so the next state is off the teacher's own distribution and
        # the sample after it is a correction.
        observations.append(observation)
        actions.append(action.copy())
        if rng.random() < perturb_prob:
            action = np.clip(action + rng.normal(0.0, 0.5, size=2), -1.0, 1.0
                             ).astype(np.float32)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if info["collided"]:
                collisions += 1
                keep = max(episode_start, len(observations) - crash_dropout)
                dropped += len(observations) - keep
                del observations[keep:]
                del actions[keep:]
            episodes += 1
            episode_start = len(observations)
            observation, _ = env.reset()
        if len(observations) and len(observations) % 1000 == 0:
            save_samples(samples_path, observations, actions)
            print(f"  {len(observations)}/{steps} samples, {episodes} episodes, "
                  f"{collisions} teacher collisions, {dropped} dropped",
                  flush=True)
    save_samples(samples_path, observations, actions)
    return (np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.float32), episodes, collisions)


def clone(model: PPO, observations: np.ndarray, actions: np.ndarray,
          epochs: int, batch_size: int, learning_rate: float) -> float:
    """Regress the policy's MEAN action onto the teacher's."""
    device = model.policy.device
    x = torch.as_tensor(observations, device=device)
    y = torch.as_tensor(actions, device=device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    count = len(x)
    loss_value = float("nan")
    for epoch in range(epochs):
        order = torch.randperm(count, device=device)
        total = 0.0
        for start in range(0, count, batch_size):
            batch = order[start:start + batch_size]
            distribution = model.policy.get_distribution(x[batch])
            mean = distribution.distribution.mean
            loss = torch.nn.functional.mse_loss(mean, y[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            total += loss.item() * len(batch)
        loss_value = total / count
        print(f"  epoch {epoch + 1}/{epochs}: action MSE {loss_value:.5f}",
              flush=True)
    return loss_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    parser.add_argument("--teleport-url",
                        default="http://localhost:9003/api/sim/teleport")
    parser.add_argument("--steps", type=int, default=20000,
                        help="teacher samples to collect (20k ~ 17 min at 20 Hz)")
    parser.add_argument("--max-speed", type=float, default=2.0)
    parser.add_argument("--plan-scale", type=float, default=0.85,
                        help="fraction of the plan's min-time speed the "
                             "teacher asks for; the student is meant to beat "
                             "this, not match it")
    parser.add_argument("--perturb-prob", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--exploration-std", type=float, default=0.3,
                        help="log_std the cloned policy is saved with. The "
                             "PPO default of 1.0 samples the whole action "
                             "range and would trample the cloned behaviour "
                             "before it earned anything.")
    parser.add_argument("--out", default="checkpoints_lap0/pretrained.zip")
    parser.add_argument("--samples", default="checkpoints_lap0/teacher.npz",
                        help="where teacher samples accumulate across sim deaths")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--fit", action="store_true",
                        help="fit the saved samples; needs no simulator")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    samples_path = Path(args.samples)

    if args.fit:
        observations, actions = load_samples(samples_path)
        if observations is None:
            raise SystemExit(f"no samples at {samples_path}; collect first")
        print(f"fitting {len(observations)} teacher samples from {samples_path}")
        fit(args, config, observations, actions)
        return

    env = build_env(config, args.sdf_path, args.teleport_url, args.max_speed)
    driver = PursuitDriver(env, target_speed=args.max_speed,
                           plan_scale=args.plan_scale)
    rng = np.random.default_rng()
    existing = load_samples(samples_path)
    if existing[0] is not None:
        print(f"resuming from {len(existing[0])} samples already collected")

    try:
        print(f"collecting to {args.steps} teacher samples at "
              f"{args.plan_scale:.2f}x the planned speed", flush=True)
        observations, actions, episodes, collisions = collect(
            env, driver, args.steps, args.perturb_prob, rng, samples_path,
            existing)
        print(f"have {len(observations)} samples over {episodes} episodes "
              f"({collisions} teacher collisions)", flush=True)
    finally:
        env.close()

    if not args.collect_only:
        fit(args, config, observations, actions)


def fit(args, config: dict, observations: np.ndarray,
        actions: np.ndarray) -> None:
    """Supervised phase. Builds the policy against a spaces-only stub env, so
    a dead simulator cannot cost a fit."""
    import gymnasium

    class SpacesOnly(gymnasium.Env):
        """Carries the real observation and action spaces and nothing else."""

        observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(observations.shape[1],), dtype=np.float32)
        action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(actions.shape[1],), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(self.observation_space.shape, np.float32), {}

        def step(self, action):
            return (np.zeros(self.observation_space.shape, np.float32), 0.0,
                    False, False, {})

    training_config = dict(config["training"])
    training_config.pop("total_timesteps", None)
    model = PPO("MlpPolicy", DummyVecEnv([lambda: Monitor(SpacesOnly())]),
                verbose=0, **training_config)
    loss = clone(model, observations, actions, args.epochs,
                 args.batch_size, args.learning_rate)
    with torch.no_grad():
        model.policy.log_std.fill_(float(np.log(args.exploration_std)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out.with_suffix("")))
    metadata = {
        "objective": "lap_time",
        "pretrained": "behaviour cloning from lap_driver.PursuitDriver",
        "teacher_plan_scale": args.plan_scale,
        "samples": int(len(observations)),
        "action_mse": loss,
        "exploration_std": args.exploration_std,
        "env": {**config["env"], "max_speed": args.max_speed},
        "reward": config["reward"],
        "zed_sim": config.get("zed_sim", {}),
        "observation": OBSERVATION_LAYOUT,
        "action": ACTION_LAYOUT,
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved {out} (action MSE {loss:.5f})")


if __name__ == "__main__":
    main()
