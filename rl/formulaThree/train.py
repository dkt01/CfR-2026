#!/usr/bin/env python3
"""Asymmetric PPO with sensor-only actor and clean-finish-first checkpoint selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from metrics import is_clean, selection_key, json_ready

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------- policy


def make_policy_class():
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import (
        BaseFeaturesExtractor,
        FlattenExtractor,
    )

    class ActorSlice(BaseFeaturesExtractor):
        """The actor's view: privileged columns zeroed, width unchanged.

        Zeroed rather than sliced so both extractors share one width, which
        SB3's MlpExtractor requires.  A zero input gives its weights a zero
        gradient, so those columns stay exactly as initialized and the export
        drops them.
        """

        def __init__(self, observation_space, actor_dim):
            super().__init__(observation_space, int(observation_space.shape[0]))
            mask = torch.zeros(int(observation_space.shape[0]))
            mask[:actor_dim] = 1.0
            self.register_buffer("mask", mask)
            self.actor_dim = actor_dim

        def forward(self, obs):
            return obs * self.mask

    class AsymmetricPolicy(ActorCriticPolicy):
        def __init__(self, *args, actor_dim=None, **kwargs):
            kwargs.update(
                share_features_extractor=False,
                features_extractor_class=ActorSlice,
                features_extractor_kwargs=dict(actor_dim=actor_dim),
            )
            super().__init__(*args, **kwargs)
            self.vf_features_extractor = FlattenExtractor(self.observation_space)

        def actor_parameters(self):
            return (
                list(self.mlp_extractor.policy_net.parameters())
                + list(self.action_net.parameters())
                + [self.log_std]
            )

    return AsymmetricPolicy


# ------------------------------------------------------------------------ env


def sb3_adapter(env):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import VecEnv

    class Adapter(VecEnv):
        def __init__(self, inner):
            self.inner = inner
            super().__init__(
                inner.n,
                gym.spaces.Box(-10.0, 10.0, (inner.obs_dim,), np.float32),
                gym.spaces.Box(-1.0, 1.0, (inner.act_dim,), np.float32),
            )

        def reset(self):
            return self.inner.reset()

        def step_async(self, actions):
            self._actions = actions

        def step_wait(self):
            obs, reward, terminated, truncated, info = self.inner.step(self._actions)
            for i in np.flatnonzero(truncated):
                info[i]["TimeLimit.truncated"] = True
            return obs, reward, terminated | truncated, info

        def close(self):
            self.inner.close()

        def get_attr(self, name, indices=None):
            return [getattr(self.inner, name)] * self.num_envs

        def set_attr(self, name, value, indices=None):
            self.inner.set(name, value)

        def env_method(self, name, *a, indices=None, **kw):
            raise NotImplementedError

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * self.num_envs

    return Adapter(env)


# ----------------------------------------------------------------- evaluation


def rollout(env, predict, episodes, max_steps=8000):
    """The FIRST episode of each of `episodes` cars (env.n must be >= it).

    Not the first N terminations: a car that crashes early restarts and
    would be counted again before the slow finishers are in, which
    understates every finish rate.
    """
    obs = env.reset()
    out = [None] * env.n
    for _ in range(max_steps):
        obs, _, terminated, truncated, info = env.step(predict(obs))
        for i in np.flatnonzero(terminated | truncated):
            if out[i] is None:
                out[i] = info[i]
        if all(o is not None for o in out[:episodes]):
            break
    if any(o is None for o in out[:episodes]):
        raise RuntimeError("Evaluation did not terminate every requested episode")
    return out[:episodes]


def summarise(records):
    if not records:
        return dict(ret=-1e9, finish=0.0, n=0)
    done = [r for r in records if is_clean(r)]
    times = [r["race_time"] for r in done]
    clear = np.array([r["min_clearance"] for r in records])
    return dict(
        ret=float(np.mean([r["episode"]["r"] for r in records])),
        finish=len(done) / len(records),
        crashed=float(np.mean([r["crashed"] for r in records])),
        time=float(np.mean(times)) if times else float("nan"),
        lap=float(np.mean([r["lap_time"] for r in done])) if times else float("nan"),
        best_lap=(
            float(np.nanmean([r["best_lap"] for r in done])) if done else float("nan")
        ),
        distance=float(np.mean([r["distance"] for r in records])),
        clearance=float(clear.min()),
        clearance_p10=float(np.percentile(clear, 10)),
        cte=float(np.mean([r["mean_cte"] for r in records])),
        max_cte=float(np.max([r["max_cte"] for r in records])),
        gate=float(np.mean([r["mean_gate"] for r in records])),
        jerk=float(np.mean([r["steer_jerk_rms"] for r in records])),
        overspeed=float(np.max([r["max_overspeed"] for r in records])),
        stop=(
            float(np.mean([r["stop_distance"] for r in done])) if done else float("nan")
        ),
        n=len(records),
    )


def lr_schedule(tcfg):
    lr0 = float(tcfg["learning_rate"])
    final = lr0 * float(tcfg.get("lr_final_frac", 1.0))
    if final >= lr0:
        return lr0
    return lambda remaining: final + (lr0 - final) * max(remaining, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=HERE / "runs/f3_v1")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=128)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument(
        "--reset-std",
        type=str,
        default=None,
        help="steer,throttle log-std to restart exploration at on --resume",
    )
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="override train.learning_rate (a fine-tune wants less than a fresh run)",
    )
    ap.add_argument(
        "--dr-start",
        type=float,
        default=None,
        help="override dr_curriculum.start (a resume continues a ramp)",
    )
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from vec import ParallelEnv

    torch.set_num_threads(4)
    cfg = yaml.safe_load(args.config.read_text())
    tcfg = cfg["train"]
    if args.learning_rate is not None:
        tcfg["learning_rate"] = args.learning_rate
    total = int(tcfg["total_steps"]) if args.steps is None else args.steps
    if (
        total < 1
        or args.workers < 1
        or args.eval_episodes < 1
        or (args.eval_every is not None and args.eval_every < 1)
    ):
        ap.error("steps, workers and eval-episodes must be positive")
    args.dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("schema") != "formulaThree-v1":
        raise ValueError("formulaThree requires schema formulaThree-v1")
    if tcfg.get("init_from") or tcfg.get("critic_warmup_steps"):
        raise ValueError(
            "Train formulaThree from scratch: its speed action has different semantics"
        )
    if args.resume:
        previous = yaml.safe_load(args.resume.with_name("config.yaml").read_text())
        if previous.get("schema") != cfg["schema"]:
            raise ValueError("Cannot resume a different racer/action schema")
        for section in ("env", "camera", "plant", "track", "collision"):
            if previous[section] != cfg[section]:
                raise ValueError(f"Resume changes {section}; start a new run instead")
    if args.steps is not None:
        tcfg["total_steps"] = total
    if args.dr_start is not None:
        tcfg["dr_curriculum"]["start"] = args.dr_start
    (args.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    n_envs = int(tcfg["n_envs"])
    train_inner = ParallelEnv(cfg, n_envs, args.workers, seed=int(tcfg["seed"]))
    train_env = sb3_adapter(train_inner)
    n_eval = args.eval_episodes
    nominal = ParallelEnv(
        cfg,
        min(32, n_eval),
        min(2, args.workers),
        seed=10_001,
        deterministic=True,
        random_start=False,
    )
    field = ParallelEnv(
        cfg, n_eval, min(4, args.workers), seed=10_002, random_start=False
    )
    # Half-strength evaluation diagnoses the sim-to-real gap.
    half = ParallelEnv(
        cfg, min(64, n_eval), min(2, args.workers), seed=10_003, random_start=False
    )
    half.set("dr_scale", 0.5)
    actor_dim = train_inner.actor_dim
    print(
        f"obs {train_inner.obs_dim} = actor {actor_dim} "
        f"(map {train_inner.map_dim} + depth {actor_dim - train_inner.map_dim}) "
        f"+ privileged {train_inner.obs_dim - actor_dim}"
    )

    Policy = make_policy_class()
    policy_kwargs = dict(
        actor_dim=actor_dim,
        net_arch=dict(pi=list(tcfg["pi_arch"]), vf=list(tcfg["vf_arch"])),
        log_std_init=float(tcfg["log_std_init"]),
    )
    if args.resume:
        model = PPO.load(
            args.resume,
            env=train_env,
            device="cpu",
            custom_objects=dict(
                policy_class=Policy,
                learning_rate=lr_schedule(tcfg),
                **{
                    name: tcfg[name]
                    for name in (
                        "n_steps",
                        "batch_size",
                        "n_epochs",
                        "gamma",
                        "gae_lambda",
                        "clip_range",
                        "ent_coef",
                    )
                },
            ),
        )
        print(f"resumed from {args.resume}")
        if args.reset_std:
            import torch

            with torch.no_grad():
                model.policy.log_std.copy_(
                    torch.tensor([float(v) for v in args.reset_std.split(",")])
                )
            print(f"  log_std reset to {args.reset_std}")
    else:
        model = PPO(
            Policy,
            train_env,
            n_steps=int(tcfg["n_steps"]),
            batch_size=int(tcfg["batch_size"]),
            n_epochs=int(tcfg["n_epochs"]),
            gamma=float(tcfg["gamma"]),
            gae_lambda=float(tcfg["gae_lambda"]),
            clip_range=float(tcfg["clip_range"]),
            ent_coef=float(tcfg["ent_coef"]),
            learning_rate=lr_schedule(tcfg),
            verbose=1,
            seed=int(tcfg["seed"]),
            device="cpu",
            policy_kwargs=policy_kwargs,
        )

    history = []
    best = [None]
    started = time.time()

    def evaluate_now(label):
        def predict(obs):
            return model.predict(obs, deterministic=True)[0]

        t0 = time.time()
        for e, seed in ((nominal, 10_001), (field, 10_002), (half, 10_003)):
            e.reseed(seed)
        nom = summarise(rollout(nominal, predict, nominal.n))
        fld = summarise(rollout(field, predict, n_eval))
        hlf = summarise(rollout(half, predict, half.n))
        history.append(
            dict(
                steps=int(model.num_timesteps),
                wall=time.time() - started,
                dr_scale=float(curriculum.scale),
                nominal=nom,
                field=fld,
                field_half=hlf,
            )
        )
        (args.dir / "history.json").write_text(
            json.dumps(json_ready(history), indent=1, allow_nan=False)
        )
        print(
            f"  [{label}] nominal {100 * nom['finish']:3.0f}% {nom['lap']:5.2f}s/lap "
            f"clr {nom['clearance']:+.3f} cte {nom['cte']:.3f} | "
            f"field {100 * fld['finish']:3.0f}% {fld['lap']:5.2f}s/lap "
            f"crash {100 * fld['crashed']:3.0f}% clr p10 {fld['clearance_p10']:+.3f} "
            f"cte {fld['cte']:.3f} gate {fld['gate']:.2f} | "
            f"half {100 * hlf['finish']:3.0f}% {hlf['lap']:5.2f}s ret {hlf['ret']:7.1f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        rank = selection_key(nom, fld)
        if best[0] is None or rank > best[0]:
            best[0] = rank
            model.save(args.dir / "best_model")
            print(f"      new best -> {args.dir / 'best_model.zip'}", flush=True)
        model.save(args.dir / "last_model")
        model.save(args.dir / f"step_{model.num_timesteps // 1000}k")

    class Curriculum(BaseCallback):
        """Randomisation ramp, critic warm-up, periodic evaluation."""

        def __init__(self, every):
            super().__init__()
            dc = tcfg.get("dr_curriculum") or {}
            self.start = float(
                args.dr_start if args.dr_start is not None else dc.get("start", 1.0)
            )
            self.ramp = float(dc.get("ramp", 0.0))
            self.every = every
            self.next_at = None
            self.scale = 1.0
            self.first = None

        def _set_scale(self):
            done = (self.num_timesteps - self.first) / max(total, 1)
            frac = 1.0 if self.ramp <= 0 else min(done / self.ramp, 1.0)
            scale = self.start + (1.0 - self.start) * frac
            if abs(scale - self.scale) > 0.01 or self.next_at is None:
                self.scale = scale
                train_inner.set("dr_scale", scale)

        def _on_training_start(self):
            self.first = self.num_timesteps
            self.scale = -1.0
            self._set_scale()
            self.next_at = self.num_timesteps + self.every

        def _on_rollout_start(self):
            self._set_scale()

        def _on_step(self):
            if self.num_timesteps >= self.next_at:
                self.next_at += self.every
                self.logger.record("train/dr_scale", self.scale)
                evaluate_now(f"{self.num_timesteps / 1e6:5.2f}M dr {self.scale:.2f}")
            return True

    every = args.eval_every or int(tcfg["eval_every_steps"])
    curriculum = Curriculum(every)
    curriculum.scale = 1.0
    print(f"training {total:,} steps in {args.dir}, evaluating every {every:,}")
    evaluate_now("start")
    try:
        model.learn(
            total_timesteps=total,
            callback=curriculum,
            reset_num_timesteps=args.resume is None,
        )
        evaluate_now("final")
    finally:
        for e in (train_inner, nominal, field, half):
            e.close()
    print(
        f"\ndone in {(time.time() - started) / 60:.1f} min.  "
        f"Export:  python3 export_policy.py {args.dir / 'best_model.zip'}"
    )


if __name__ == "__main__":
    main()
