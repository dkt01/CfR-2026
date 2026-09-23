#!/usr/bin/env python3
"""PPO against the vectorised env.

    python3 train.py --dir runs/v1
    python3 train.py --dir runs/v2 --resume runs/v1/best_model.zip

The env steps ~19k car-steps a second on one core at `n_envs: 256`, so the
whole 12M-step budget is minutes of environment time and a couple of hours
wall clock once the network is in the loop.  That is the point of not
training inside Gazebo: the same budget there is weeks.

Two evaluations run side by side throughout, because they answer different
questions and only one of them is about lap time:

    nominal   randomisation off, standing start.  "How fast is it?"
    field     randomisation on, standing start.   "Will it get round on the
              day, on a car whose dead time, drag, steering trim and
              localisation are not the ones it trained on?"

The best model is chosen on the FIELD result first and the time second.  A
policy that is half a second quicker and finishes four runs in five is not
the one to take to a competition.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

import track as track_mod
from env import FormulaOneEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sb3_adapter(env):
    """Wrap FormulaOneEnv in the VecEnv interface SB3 expects."""
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
                # Tells PPO to bootstrap the value of a timed-out episode
                # instead of treating the cut as a real terminal state.
                info[i]["TimeLimit.truncated"] = True
            return obs, reward, terminated | truncated, info

        def close(self):
            pass

        def get_attr(self, name, indices=None):
            return [getattr(self.inner, name)] * self.num_envs

        def set_attr(self, name, value, indices=None):
            setattr(self.inner, name, value)

        def env_method(self, name, *a, indices=None, **kw):
            return [getattr(self.inner, name)(*a, **kw)] * self.num_envs

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * self.num_envs

    return Adapter(env)


def rollout(env, predict, episodes):
    """Collect `episodes` finished episodes from a batched env."""
    obs = env.reset()
    out = []
    for _ in range(20000):
        action = predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        for i in np.flatnonzero(terminated | truncated):
            out.append(info[i])
        if len(out) >= episodes:
            break
    return out[:episodes]


def summarise(records):
    if not records:
        return dict(
            ret=-1e9,
            finish=0.0,
            time=float("nan"),
            clearance=0.0,
            overspeed=0.0,
            cte=float("nan"),
            max_cte=float("nan"),
            jerk=float("nan"),
            gain=float("nan"),
            stop=0.0,
            n=0,
        )
    # `stopped` is the run the brief actually asks for: two laps AND at rest.
    # `finished` alone is the car crossing the line still doing 5 m/s, which
    # on a course with no run-off is not a completed run.
    finished = [r for r in records if r["stopped"]]
    times = [r["race_time"] for r in finished]
    gains = [
        r["last_lap"] - r["best_lap"]
        for r in finished
        if r["last_lap"] > 0 and np.isfinite(r["best_lap"])
    ]
    return dict(
        # MEAN EPISODE RETURN is what picks the best model, not finish rate.
        # Finish-rate-first selection silently re-imposes a priority the
        # reward no longer holds: once `time` was raised to 8.0, a policy
        # finishing 75% of runs in 89 s is worth LESS than one finishing 61%
        # in 78 s, and a finish-first rule would have kept the slow one
        # forever.  The return is the one number that already encodes the
        # speed-against-reliability trade the reward defines, so selection
        # follows the reward instead of arguing with it.
        ret=float(np.mean([r["episode"]["r"] for r in records])),
        finish=len(finished) / len(records),
        time=float(np.mean(times)) if times else float("nan"),
        clearance=float(np.min([r["min_clearance"] for r in records])),
        overspeed=float(np.max([r["max_overspeed"] for r in records])),
        # The other three requirements, measured rather than assumed: mean
        # cross-track error, the worst of it, and how much the steering
        # command shakes.  A policy can win on return while quietly getting
        # worse at one of these, and then it is not the policy that was asked
        # for -- so they are printed on every evaluation line.
        cte=float(np.mean([r["mean_cte"] for r in records])),
        max_cte=float(np.max([r["max_cte"] for r in records])),
        jerk=float(np.mean([r["steer_jerk_rms"] for r in records])),
        # Seconds the quickest lap took off the slowest -- "beat its previous
        # lap time", after the fact.
        gain=float(np.mean(gains)) if gains else 0.0,
        stop=float(np.mean([r["stop_distance"] for r in finished]))
        if finished
        else float("nan"),
        n=len(records),
    )


def lr_schedule(tcfg):
    """Linear decay from `learning_rate` to `lr_final_frac` of it.

    SB3 hands the callable `progress_remaining`, which runs 1 -> 0 over the
    budget passed to `learn`.  See the note on `lr_final_frac` in config.yaml
    for what a constant rate did to the first run of this environment.
    """
    lr0 = float(tcfg["learning_rate"])
    final = lr0 * float(tcfg.get("lr_final_frac", 1.0))
    if final >= lr0:
        return lr0
    return lambda remaining: final + (lr0 - final) * max(remaining, 0.0)


def make_eval_envs(cfg, track, n=64):
    nominal = FormulaOneEnv(cfg, track, n, seed=10_001, deterministic=True)
    nominal.random_start = False
    field = FormulaOneEnv(cfg, track, n, seed=10_002, deterministic=False)
    field.random_start = False  # a real run starts on the grid
    return nominal, field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=HERE / "runs/v1")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="steps between evaluations; default from config",
    )
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    cfg = yaml.safe_load(args.config.read_text())
    tcfg = cfg["train"]
    total = args.steps or int(tcfg["total_steps"])
    args.dir.mkdir(parents=True, exist_ok=True)
    (args.dir / "config.yaml").write_text(args.config.read_text())

    track = track_mod.build(cfg, ROOT)
    train_env = sb3_adapter(
        FormulaOneEnv(cfg, track, int(tcfg["n_envs"]), seed=int(tcfg["seed"]))
    )
    nominal, field = make_eval_envs(cfg, track)

    kwargs = dict(
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
        policy_kwargs=dict(
            net_arch=list(tcfg["net_arch"]), log_std_init=float(tcfg["log_std_init"])
        ),
    )
    if args.resume:
        model = PPO.load(args.resume, env=train_env, device="cpu")
        # PPO.load restores the hyperparameters saved INSIDE the checkpoint,
        # so a resumed run would otherwise ignore every tuning change made to
        # config.yaml since -- silently, with the old value still in force.
        # The ones worth re-applying are re-applied here, and printed, so a
        # resume says out loud what it is actually running with.
        for name in ("ent_coef", "clip_range", "gamma", "gae_lambda"):
            want = tcfg[name] if not callable(tcfg.get(name)) else None
            if want is None:
                continue
            was = getattr(model, name, None)
            if isinstance(was, float) and abs(was - float(want)) > 1e-12:
                setattr(model, name, float(want))
                print(f"  {name}: {was} -> {float(want)} (from config)")
        # The schedule lives in the checkpoint too, so a resume would carry
        # the old run's decay -- including its endpoint -- into the new one.
        sched = lr_schedule(tcfg)
        if callable(sched):
            model.lr_schedule = sched
        print(f"resumed from {args.resume}")
    else:
        model = PPO("MlpPolicy", train_env, **kwargs)

    history = []
    best = (-1e9, 0.0, float("inf"))
    started = time.time()

    def evaluate_now(model, label):
        """One nominal + one field evaluation, recorded and reported."""
        nonlocal best

        def predict(obs):
            return model.predict(obs, deterministic=True)[0]

        nom = summarise(rollout(nominal, predict, 64))
        fld = summarise(rollout(field, predict, 256))
        history.append(
            dict(
                steps=int(model.num_timesteps),
                wall=time.time() - started,
                nominal=nom,
                field=fld,
            )
        )
        (args.dir / "history.json").write_text(json.dumps(history, indent=1))
        print(
            f"  [{label}] "
            f"nominal {100 * nom['finish']:3.0f}% {nom['time']:6.2f}s | "
            f"field {100 * fld['finish']:3.0f}% {fld['time']:6.2f}s "
            f"ret {fld['ret']:7.1f} | "
            f"clearance {fld['clearance']:+.3f} m over {fld['overspeed']:+.2f} m/s"
        )
        print(
            f"           cte {nom['cte']:.3f}/{nom['max_cte']:.3f} m  "
            f"jerk {nom['jerk']:.4f}  lap gain {nom['gain']:+.2f}s  "
            f"stop +{nom['stop']:.1f} m   (nominal)"
        )
        if fld["ret"] > best[0]:
            best = (fld["ret"], fld["finish"], fld["time"])
            model.save(args.dir / "best_model")
            print(f"      new best -> {args.dir / 'best_model.zip'}")
        model.save(args.dir / "last_model")
        # Keep the ladder too: selection is a judgement call, and a run that
        # only ever saves its own favourite leaves nothing to re-judge later.
        model.save(args.dir / f"step_{model.num_timesteps // 1000}k")

    class Evaluate(BaseCallback):
        def __init__(self, every):
            super().__init__()
            self.every = every
            self.next_at = every
            self.primed = False

        def _on_step(self):
            if not self.primed:
                # A resumed run starts at the checkpoint's step count, not at
                # zero, so the first deadline has to be placed relative to
                # where it actually starts -- otherwise every missed interval
                # fires back to back before the schedule catches up.
                self.primed = True
                self.next_at = self.num_timesteps + self.every
                return True
            if self.num_timesteps < self.next_at:
                return True
            self.next_at += self.every
            evaluate_now(self.model, f"{self.num_timesteps / 1e6:5.2f}M")
            return True

    every = args.eval_every or int(tcfg["eval_every_steps"])
    sched = lr_schedule(tcfg)
    print(f"training {total:,} steps in {args.dir}, evaluating every {every:,}")
    if callable(sched):
        print(f"  learning rate {sched(1.0):.2e} -> {sched(0.0):.2e}, linear")
    model.learn(
        total_timesteps=total,
        callback=Evaluate(every),
        reset_num_timesteps=args.resume is None,
    )
    # Always score the finished model, so a run shorter than one eval interval
    # still leaves a best_model.zip behind rather than an empty directory.
    evaluate_now(model, "final")
    print(
        f"\ndone in {(time.time() - started) / 60:.1f} min. "
        f"Best field return {best[0]:.1f}: {100 * best[1]:.0f}% at {best[2]:.2f} s.\n"
        f"Export it:  python3 export_policy.py {args.dir / 'best_model.zip'}"
    )


if __name__ == "__main__":
    main()
