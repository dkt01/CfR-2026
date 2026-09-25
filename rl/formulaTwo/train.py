#!/usr/bin/env python3
"""PPO for formulaTwo: asymmetric actor-critic, started from v12.

    python3 train.py --dir runs/f2_v1
    python3 train.py --dir runs/f2_v2 --resume runs/f2_v1/best_model.zip
    python3 train.py --dir runs/scratch --no-init        # ignore init_from

Three things differ from formulaOne's train.py, each for a reason:

  ASYMMETRIC.  The observation is [actor | privileged].  The actor's input
  layer sees the privileged block as zeros (ActorSlice), so it can neither use
  nor depend on it, and export_policy.py ships only the actor rows.  The
  critic reads everything: under this much randomisation, a value function
  that has to guess which car and which course it drew from noisy inputs is
  the bottleneck, and it never goes to the car.

  STARTED AS v12.  The first 35 actor inputs are v12's inputs, bit for bit
  (checked: identical trajectories).  So the actor is v12's weights with zero
  columns for the depth stack -- at step 0 it IS v12 -- and the critic trains
  alone for `critic_warmup_steps` before the actor is allowed to move, so the
  first policy updates are not driven by a random value function.

  CURRICULUM ON RANDOMISATION.  `dr_scale` ramps from `start` to 1.0 over the
  first `ramp` of the run.  Evaluation is always at full strength.

Evaluation, as in formulaOne: `nominal` (randomisation off, grid start: how
fast) and `field` (everything redrawn, grid start: will it get round on the
day).  `finish` means THREE laps AND at rest.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

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
        gradient, so those columns stay exactly as initialised and the export
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


def init_actor_from(model, checkpoint, log_std):
    """Copy a formulaOne actor into this one, new input columns at zero."""
    import torch
    from stable_baselines3 import PPO

    src = PPO.load(checkpoint, device="cpu").policy.state_dict()
    dst = model.policy.state_dict()
    with torch.no_grad():
        w = dst["mlp_extractor.policy_net.0.weight"]
        v = src["mlp_extractor.policy_net.0.weight"]
        if w.shape[0] != v.shape[0]:
            raise SystemExit(f"pi_arch {tuple(w.shape)} cannot take {tuple(v.shape)}")
        w.zero_()
        w[:, : v.shape[1]] = v
        for key in (
            "mlp_extractor.policy_net.0.bias",
            "mlp_extractor.policy_net.2.weight",
            "mlp_extractor.policy_net.2.bias",
            "action_net.weight",
            "action_net.bias",
        ):
            dst[key].copy_(src[key])
        dst["log_std"].copy_(torch.as_tensor(log_std, dtype=dst["log_std"].dtype))
    model.policy.load_state_dict(dst)
    return v.shape[1]


def throttle_to_edge(model, env, steps=200):
    """Shift the throttle output bias so its raw mean sits at -1, the floor.

    PPO's Gaussian lives on the raw action and the env clips it.  Once the
    raw throttle mean drifts to ~-2.2 (v12 and f2_v1 both did: 100% of ticks
    at the floor), exploration at sigma 0.44 samples above the floor 0.3% of
    the time, so there is no gradient about driving faster at all.  At the
    edge, the deterministic policy still drives the floor, and half of the
    samples ask for more.
    """
    import torch

    obs = env.reset()
    raw = []
    for _ in range(steps):
        with torch.no_grad():
            dist = model.policy.get_distribution(torch.as_tensor(obs))
            mean = dist.distribution.mean.cpu().numpy()
        raw.append(mean[:, 1])
        obs, _, _, _, _ = env.step(np.clip(mean, -1, 1))
    shift = -1.0 - float(np.median(np.concatenate(raw)))
    with torch.no_grad():
        model.policy.action_net.bias[1] += shift
    return shift


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
    return [o for o in out[:episodes] if o is not None]


def summarise(records):
    if not records:
        return dict(ret=-1e9, finish=0.0, n=0)
    done = [r for r in records if r["stopped"]]
    times = [r["race_time"] for r in done]
    clear = np.array([r["min_clearance"] for r in records])
    return dict(
        ret=float(np.mean([r["episode"]["r"] for r in records])),
        finish=len(done) / len(records),
        crashed=float(np.mean([r["crashed"] for r in records])),
        time=float(np.mean(times)) if times else float("nan"),
        lap=float(np.mean(times)) / 3 if times else float("nan"),
        best_lap=float(np.nanmean([r["best_lap"] for r in done])) if done else float("nan"),
        distance=float(np.mean([r["distance"] for r in records])),
        clearance=float(clear.min()),
        clearance_p10=float(np.percentile(clear, 10)),
        cte=float(np.mean([r["mean_cte"] for r in records])),
        max_cte=float(np.max([r["max_cte"] for r in records])),
        gate=float(np.mean([r["mean_gate"] for r in records])),
        jerk=float(np.mean([r["steer_jerk_rms"] for r in records])),
        overspeed=float(np.max([r["max_overspeed"] for r in records])),
        stop=float(np.mean([r["stop_distance"] for r in done])) if done else float("nan"),
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
    ap.add_argument("--dir", type=Path, default=HERE / "runs/f2_v1")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--no-init", action="store_true")
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
        "--throttle-to-edge",
        action="store_true",
        help="on --resume, shift the throttle bias so its raw mean sits at -1",
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
    total = args.steps or int(tcfg["total_steps"])
    args.dir.mkdir(parents=True, exist_ok=True)
    (args.dir / "config.yaml").write_text(args.config.read_text())

    n_envs = int(tcfg["n_envs"])
    train_inner = ParallelEnv(cfg, n_envs, args.workers, seed=int(tcfg["seed"]))
    train_env = sb3_adapter(train_inner)
    n_eval = args.eval_episodes
    nominal = ParallelEnv(cfg, 32, 2, seed=10_001, deterministic=True, random_start=False)
    field = ParallelEnv(cfg, n_eval, 4, seed=10_002, random_start=False)
    # Checkpoints are SELECTED on a half-strength field: at full strength the
    # fixed-gain prior finishes 0% at any speed, so a policy's full-field
    # return is mostly noise about how far it got before the 20 m chicane.
    half = ParallelEnv(cfg, 64, 2, seed=10_003, random_start=False)
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
    warmup = 0
    if args.resume:
        model = PPO.load(
            args.resume,
            env=train_env,
            device="cpu",
            custom_objects=dict(policy_class=Policy),
        )
        for name in ("ent_coef", "clip_range", "gamma", "gae_lambda"):
            want = float(tcfg[name])
            was = getattr(model, name, None)
            if isinstance(was, float) and abs(was - want) > 1e-12:
                setattr(model, name, want)
                print(f"  {name}: {was} -> {want} (from config)")
        sched = lr_schedule(tcfg)
        if callable(sched):
            model.lr_schedule = sched
        print(f"resumed from {args.resume}")
        if args.throttle_to_edge:
            shift = throttle_to_edge(model, train_inner)
            print(f"  throttle bias {shift:+.3f}: raw mean now at the floor edge")
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
        init = tcfg.get("init_from")
        if init and not args.no_init:
            ckpt = (HERE / init).resolve()
            # v12's own steering std collapsed to 0.067 (log -2.70); started
            # there the actor could never explore its way into using the
            # depth stack.  Its weights, a fresh steering std.
            width = init_actor_from(model, ckpt, [-1.6, -1.0])
            warmup = int(tcfg.get("critic_warmup_steps", 0))
            print(f"actor initialised from {ckpt} ({width} inputs copied, rest zero)")

    history = []
    best = [-1e18]
    started = time.time()

    def evaluate_now(label):
        def predict(obs):
            return model.predict(obs, deterministic=True)[0]

        t0 = time.time()
        nom = summarise(rollout(nominal, predict, 32))
        fld = summarise(rollout(field, predict, n_eval))
        hlf = summarise(rollout(half, predict, 64))
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
        (args.dir / "history.json").write_text(json.dumps(history, indent=1))
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
        if hlf["ret"] > best[0]:
            best[0] = hlf["ret"]
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
            self.frozen = False
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
            if warmup > 0:
                for p in model.policy.actor_parameters():
                    p.requires_grad_(False)
                self.frozen = True
                print(f"  actor frozen for {warmup:,} steps of critic warm-up")

        def _on_rollout_start(self):
            self._set_scale()
            if self.frozen and self.num_timesteps - self.first >= warmup:
                for p in model.policy.actor_parameters():
                    p.requires_grad_(True)
                self.frozen = False
                print(f"  actor released at {self.num_timesteps:,}", flush=True)

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
