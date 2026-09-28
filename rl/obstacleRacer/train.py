#!/usr/bin/env python3
"""PPO on the numpy Obstacle Course.

    python3 train.py --dir runs/v1
    python3 train.py --dir runs/v2 --resume runs/v1/best_model.zip
    python3 train.py --dir runs/smoke --steps 1000000 --eval-every 250000

Every evaluation drives deterministic episodes from the start box (+/-0.1 m,
+/-5 deg, from rest) on ten of the TRAINING layouts and on the four HELD-OUT
ones, plus a per-obstacle check on the held-out layouts (dealt 2 m before
each obstacle: does the car get through it?), and appends a record to
<dir>/history.json -- the file the rl-train skill's tui.py watches.  The
record also carries, from the training rollouts, how often each obstacle was
met and failed at.  The best model is the one that finishes the most
held-out runs, then scores best on held-out progress and obstacles cleared,
then laps quickest: a policy that only knows its training courses is not the one
to take to the event.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import course_model
import layouts
import observation as O
import reward
from env import ObstacleEnv
import ppo_policy
from ppo_policy import Driver, SquashedMeanPolicy

HERE = Path(__file__).resolve().parent


def sb3_adapter(env, reward_scale=1.0):
    """ObstacleEnv as the VecEnv stable-baselines3 expects.

    Rewards reach PPO multiplied by `reward_scale`, so the value net's
    targets are of order 1.  Unscaled, v3's returns sat at -70..-140; Adam
    moves the output bias only ~lr a step, so the value net made -63 by
    driving all its tanh units to +/-1 instead, and never predicted anything
    again (explained variance ~0 for 20M steps).  A constant factor changes
    no episode's rank, and the episode returns logged come from the env's
    own info, unscaled.
    """
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
            obs, rew, terminated, truncated, info = self.inner.step(self._actions)
            rew = rew * reward_scale
            for i in np.flatnonzero(truncated):
                # PPO bootstraps a cut episode instead of treating it as over.
                info[i]["TimeLimit.truncated"] = True
            return obs, rew, terminated | truncated, info

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


def evaluate(env, driver, per_layout):
    """Deterministic start-box episodes, `per_layout` on each of env's layouts.

    `driver(n)` makes a fresh ppo_policy.Driver (or anything with the same
    call and `ended`) for the env's n cars.
    """
    want = per_layout * len(env.layout_ids)
    # One car per episode slot, layouts dealt round-robin so every layout
    # gets its share regardless of how fast its episodes end.
    env.layout_ids_cycle = np.tile(env.layout_ids, per_layout)
    obs = env.reset()
    predict = driver(env.n)
    out = []
    for _ in range(int(env.cfg["env"]["episode_s"] * env.cfg["env"]["control_hz"]) + 5):
        obs, _, term, trunc, info = env.step(predict(obs))
        predict.ended(term | trunc)
        out += [i for i in info if i]
        if len(out) >= want:
            break
    return out[:want]


RECOVERED_M = 2.0  # a stuck start that got this far along has backed out
SECTION_BACKOFF_M = 2.0
SECTION_PAST_M = 1.5  # past the obstacle's end: a hoop missed is due by 1 m
# v5's 30 s failed Wide Section runs that got through, slowly, after it:
# the section is 6-8 m of bales to find a way through.
SECTION_TIME_S = 60.0


def section_eval(env, driver, per_layout):
    """Per obstacle: dealt SECTION_BACKOFF_M before it, does the car get past?

    Never dealt behind the end of the obstacle before (env.section_floor), so
    each rate is that obstacle alone; before v5 at 22M, the buckets' and the
    hoops' rates also charged the Wide Section's and the buckets' failures.

    One car per (layout, obstacle, repeat).  A car clears the obstacle once it
    has driven SECTION_PAST_M beyond the obstacle's end; an episode ending
    first, or SECTION_TIME_S running out, is a fail.  Returns
    {obstacle: share cleared}.
    """
    plan = [
        (lay, z)
        for lay in env.layout_ids
        for z in env.section_s[lay]
        for _ in range(per_layout)
    ]
    assert len(plan) == env.n, (len(plan), env.n)
    lays = np.array([p[0] for p in plan])
    zones = np.array([p[1] for p in plan])
    env.forced_lay = lays
    starts = [
        env.section_target(lay, z, SECTION_BACKOFF_M, any_hoop=False) for lay, z in plan
    ]
    env.forced_s = np.array([s for s, _ in starts])
    env.forced_floor = np.array([f for _, f in starts])
    obs = env.reset()
    predict = driver(env.n)
    ends = np.array([env.section_s[lay][z][1] for lay, z in plan])
    need = np.mod(ends - env.s_start, env.loop[lays]) + SECTION_PAST_M
    cleared = np.full(env.n, np.nan)
    for _ in range(int(SECTION_TIME_S * float(env.cfg["env"]["control_hz"]))):
        obs, _, term, trunc, info = env.step(predict(obs))
        predict.ended(term | trunc)
        open_ = np.isnan(cleared)
        cleared[open_ & (env.dist >= need)] = 1.0
        for i in np.flatnonzero(open_):
            if info[i] and np.isnan(cleared[i]):
                cleared[i] = 1.0 if info[i]["outcome"] == "finish" else 0.0
        if not np.isnan(cleared).any():
            break
    cleared = np.nan_to_num(cleared, nan=0.0)
    return {
        env.zone_names[z]: float(cleared[zones == z].mean())
        for z in sorted(
            set(zones.tolist()), key=lambda z: env.section_s[lays[0]].get(z, (0,))[0]
        )
    }


def summarize(records):
    if not records:
        return {}
    finished = [r for r in records if r["outcome"] == "finish"]
    frac = [r["dist"] / r["goal"] for r in records]
    by_seed = {}
    for r in records:
        by_seed.setdefault(r["seed"], []).append(r["outcome"] == "finish")
    ends = Counter()
    for r in records:
        if r["outcome"] != "finish":
            ends[f"{r['outcome']}@{r['zone']}"] += 1
    attitude = {}
    for r in records:
        for zone, (p, rl) in r["attitude"].items():
            a = attitude.setdefault(zone, [0.0, 0.0])
            a[0], a[1] = max(a[0], p), max(a[1], rl)
    return dict(
        n=len(records),
        finish=len(finished) / len(records),
        progress=float(np.mean(frac)),
        progress_m=float(np.mean([r["dist"] for r in records])),
        lap_time=float(np.mean([r["time"] for r in finished])) if finished else None,
        best_lap=float(np.min([r["time"] for r in finished])) if finished else None,
        hoops=float(np.mean([r["hoops"] for r in records])),
        # Centerline meters per second over the whole run, finished or not,
        # and the fastest the car went; lap_time only exists once laps do.
        avg_speed=float(np.mean([r["dist"] / max(r["time"], 1e-3) for r in records])),
        top_speed=float(np.mean([r.get("top_speed", 0.0) for r in records])),
        ret=float(np.mean([r["episode"]["r"] for r in records])),
        min_clearance=float(np.min([r["min_clearance"] for r in records])),
        touches=float(np.mean([r["touches"] for r in records])),
        outcomes=dict(Counter(r["outcome"] for r in records)),
        ends=dict(ends.most_common(12)),
        finish_by_seed={str(k): float(np.mean(v)) for k, v in sorted(by_seed.items())},
        attitude_deg=attitude,
    )


def lr_schedule(tcfg):
    lr0 = float(tcfg["learning_rate"])
    final = lr0 * float(tcfg.get("lr_final_fraction", 1.0))
    if final >= lr0:
        return lr0
    return lambda remaining: final + (lr0 - final) * max(remaining, 0.0)


class EvalEnv(ObstacleEnv):
    """Start box only, and layouts dealt round-robin rather than at random."""

    layout_ids_cycle = None
    _next = 0

    def reset(self):
        self._next = 0
        return super().reset()

    def _pick_layouts(self, k):
        if self.layout_ids_cycle is None:
            return super()._pick_layouts(k)
        ids = self.layout_ids_cycle
        pick = ids[(self._next + np.arange(k)) % len(ids)]
        self._next += k
        return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=HERE / "runs/v1")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="save step_<N>k.zip this often, between evals (default: at evals only)",
    )
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--gazebo",
        type=int,
        default=None,
        help="Gazebo cars beside the numpy batch, one sim container each "
        "(default: the config's gazebo.instances, else 0)",
    )
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    recurrent = bool(yaml.safe_load(args.config.read_text())["train"].get("recurrent"))
    if recurrent:
        from sb3_contrib import RecurrentPPO as Algo

        policy_class = ppo_policy.SquashedMeanLstmPolicy
    else:
        Algo, policy_class = PPO, SquashedMeanPolicy

    cfg = yaml.safe_load(args.config.read_text())
    tcfg = cfg["train"]
    if recurrent:
        import torch

        # Short, frequently reset LSTM sequences cost more in MKLDNN and with
        # four CPU threads on the training laptop (see bench_ppo.py).
        torch.set_num_threads(
            int(tcfg.get("torch_cpu_threads", torch.get_num_threads()))
        )
        torch.backends.mkldnn.enabled = bool(tcfg.get("torch_mkldnn", True))
        print(
            f"PyTorch CPU: {torch.get_num_threads()} threads, "
            f"MKLDNN {'on' if torch.backends.mkldnn.enabled else 'off'}",
            flush=True,
        )
    # The reward is checked against the config this run actually uses, not
    # only when someone remembers to run reward.py.
    reward.assert_reachable(cfg)
    reward.assert_episode_incentives(cfg)

    total = args.steps or int(tcfg["total_steps"])
    every = args.eval_every or int(tcfg["eval_every_steps"])
    args.dir.mkdir(parents=True, exist_ok=True)
    (args.dir / "config.yaml").write_text(args.config.read_text())

    seeds = layouts.TRAIN_SEEDS + layouts.HELDOUT_SEEDS
    model_ = course_model.CourseModel(seeds)
    train_ids = np.arange(len(layouts.TRAIN_SEEDS))
    held_ids = np.arange(len(layouts.TRAIN_SEEDS), len(seeds))
    gcfg = cfg.get("gazebo", {})
    n_gazebo = args.gazebo if args.gazebo is not None else int(gcfg.get("instances", 0))
    n_envs = (
        int(gcfg.get("numpy_envs", tcfg["n_envs"])) if n_gazebo else int(tcfg["n_envs"])
    )
    inner_env = ObstacleEnv(cfg, model_, n_envs, train_ids, seed=args.seed)
    # Gazebo-paced runs make ~50 steps/s, so an eval's worth of steps is
    # hours; checkpoints come far more often than the ~10 min full eval.
    ckpt_every = args.checkpoint_every or int(
        (gcfg if n_gazebo else tcfg).get("checkpoint_every_steps", 0)
    )
    workers = []
    if n_gazebo:
        import gazebo_vec

        # Gazebo trains on a spread of the training layouts; its course model
        # is baked in each container, so fewer layouts start faster.
        stride = max(1, len(layouts.TRAIN_SEEDS) // int(gcfg.get("layouts", 40)))
        gz_seeds = layouts.TRAIN_SEEDS[::stride]
        repo = str(HERE.parents[1])
        for k in range(n_gazebo):
            name = f"{gcfg.get('container_prefix', 'cfr-racer-gz')}{k + 1}"
            print(f"starting Gazebo car {k + 1} in {name}", flush=True)
            gazebo_vec.ensure_container(name, repo)
            w = gazebo_vec.GazeboWorker(
                name,
                cfg,
                gz_seeds,
                args.seed + 1000 * (k + 1),
                args.dir / f"gazebo_{k + 1}.log",
                rtf=float(gcfg.get("rtf", 1.0)),
                domain=41 + k,
            )
            w.start()
            workers.append(w)
        inner_env = gazebo_vec.MixedEnv(inner_env, workers)
    train_env = sb3_adapter(inner_env, float(tcfg["reward_scale"]))
    per = int(tcfg["eval_episodes_per_layout"])
    # Every training layout is trained on; the start-box check drives the
    # first ten (layouts.TRAIN_EVAL_SEEDS) so it costs what it did in v5.
    train_eval_ids = np.arange(len(layouts.TRAIN_EVAL_SEEDS))
    eval_train = EvalEnv(
        cfg,
        model_,
        per * len(train_eval_ids),
        train_eval_ids,
        seed=10_001,
        start_box_only=True,
    )
    # More held-out layouts, fewer runs on each: layouts differ far more
    # than repeats on one layout do.
    held_per = int(tcfg.get("heldout_episodes_per_layout", per))
    eval_held = EvalEnv(
        cfg,
        model_,
        held_per * len(held_ids),
        held_ids,
        seed=10_002,
        start_box_only=True,
    )
    sec_per = int(tcfg["section_eval_per_layout"])
    probe = ObstacleEnv(cfg, model_, 1, held_ids, seed=0)
    n_sec = sec_per * sum(len(probe.section_s[k]) for k in held_ids)
    eval_sections = ObstacleEnv(cfg, model_, n_sec, held_ids, seed=10_003)
    del probe

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
        seed=args.seed,
        device="cpu",
        policy_kwargs=dict(
            net_arch=list(tcfg["net_arch"]),
            log_std_init=float(tcfg["log_std_init"]),
            log_std_range=tuple(tcfg["log_std_range"]),
            log_std_shift=(O.steer_log_std_shift(cfg), 0.0),
        ),
    )
    if recurrent:
        # One LSTM for the actor and one for the critic, as sb3-contrib
        # defaults; gradients run back through each n_steps rollout.
        kwargs["policy_kwargs"].update(
            lstm_hidden_size=int(tcfg["lstm_hidden_size"]),
            n_lstm_layers=int(tcfg.get("n_lstm_layers", 1)),
        )
    if args.resume:
        model = Algo.load(args.resume, env=train_env, device="cpu")
        # PPO.load restores the checkpoint's own hyperparameters; re-apply
        # the config's, out loud, so a resume runs what config.yaml says.
        for name in (
            "ent_coef",
            "clip_range",
            "gamma",
            "gae_lambda",
            "n_epochs",
            "batch_size",
        ):
            was, want = getattr(model, name, None), tcfg[name]
            if isinstance(was, (int, float)) and was != want:
                setattr(model, name, type(was)(want))
                print(f"  {name}: {was} -> {want} (from config)")
        sched = lr_schedule(tcfg)
        model.lr_schedule = sched if callable(sched) else (lambda _: sched)
        print(f"resumed from {args.resume} at {model.num_timesteps:,} steps")
    else:
        model = Algo(policy_class, train_env, **kwargs)

    history_path = args.dir / "history.json"
    history = (
        json.loads(history_path.read_text())
        if args.resume and history_path.exists()
        else []
    )
    started = time.time()
    best = [(-1.0, -1.0, 0.0)]
    target = model.num_timesteps + total if args.resume else total
    # Stuck starts (env.stuck_start_prob): how many, and how many backed out
    # and drove on at least RECOVERED_M -- the thing v5 is meant to learn.
    rollout_stats = {"ep_rew": [], "outcomes": Counter(), "stuck": [0, 0]}
    # Gazebo cars' episodes, kept apart: they are the ones this is all for.
    gazebo_stats = {"runs": []}

    def evaluate_now(label):
        def driver(n):
            return Driver(model, n)

        tr = summarize(evaluate(eval_train, driver, per))
        he = summarize(evaluate(eval_held, driver, held_per))
        he["sections"] = section_eval(eval_sections, driver, sec_per)
        inner = train_env.inner
        met = {
            inner.zone_names[z]: [
                int(inner.zone_attempts[z]),
                int(inner.zone_fails[z]),
            ]
            for z in np.flatnonzero(inner.zone_attempts)
        }
        inner.zone_attempts[:] = 0
        inner.zone_fails[:] = 0
        record = dict(
            steps=int(model.num_timesteps),
            target=int(target),
            wall=time.time() - started,
            pid=os.getpid(),
            train=tr,
            heldout=he,
            rollout=dict(
                ep_rew_mean=float(np.mean(rollout_stats["ep_rew"]))
                if rollout_stats["ep_rew"]
                else None,
                outcomes=dict(rollout_stats["outcomes"]),
                zones=met,
                # Share of section starts each obstacle is getting now.
                practice={
                    inner.zone_names[z]: float(p)
                    for z, p in zip(*inner.section_weights())
                },
                stuck_starts=rollout_stats["stuck"][0],
                stuck_recovered=rollout_stats["stuck"][1]
                / max(rollout_stats["stuck"][0], 1),
            ),
        )
        if workers:
            runs = gazebo_stats["runs"]
            box = [r for r in runs if r["start"] == "box"]
            walls = inner.step_wall[-2000:]
            record["gazebo"] = dict(
                episodes=len(runs),
                steps=int(sum(r["episode"]["l"] for r in runs)),
                outcomes=dict(Counter(r["outcome"] for r in runs)),
                ends=dict(
                    Counter(
                        f"{r['outcome']}@{r['zone']}"
                        for r in runs
                        if r["outcome"] != "finish"
                    ).most_common(8)
                ),
                ep_rew_mean=float(np.mean([r["episode"]["r"] for r in runs]))
                if runs
                else None,
                dist_mean=float(np.mean([r["dist"] for r in runs])) if runs else None,
                box_runs=len(box),
                box_finish=float(np.mean([r["outcome"] == "finish" for r in box]))
                if box
                else None,
                box_dist=float(np.mean([r["dist"] for r in box])) if box else None,
                lap_times=[r["time"] for r in box if r["outcome"] == "finish"],
                restarts=int(sum(w.restarts for w in workers)),
                step_s_numpy=float(np.mean([a for a, _ in walls])) if walls else None,
                step_s=float(np.mean([b for _, b in walls])) if walls else None,
            )
            gazebo_stats["runs"] = []
            g = record["gazebo"]
            print(
                f"  [gazebo] {g['episodes']} runs, {g['steps']} steps, "
                f"mean {g['dist_mean'] or 0:.1f} m; start box {g['box_runs']} runs, "
                f"finish {100 * (g['box_finish'] or 0):.0f}%, {g['box_dist'] or 0:.1f} m | "
                f"ends {dict(list(g['ends'].items())[:3])} | step {g['step_s'] or 0:.3f} s "
                f"(numpy {g['step_s_numpy'] or 0:.3f} s), restarts {g['restarts']}",
                flush=True,
            )
        rollout_stats["ep_rew"].clear()
        rollout_stats["outcomes"].clear()
        rollout_stats["stuck"] = [0, 0]
        history.append(record)
        history_path.write_text(json.dumps(history, indent=1))

        def lap(x):
            return f"{x:5.1f}s" if x else "  -  "

        print(
            f"  [{label}] train {100 * tr['finish']:3.0f}% {lap(tr['lap_time'])} "
            f"{100 * tr['progress']:3.0f}% of lap | held-out {100 * he['finish']:3.0f}% "
            f"{lap(he['lap_time'])} {100 * he['progress']:3.0f}% of lap, hoops {he['hoops']:.1f}, "
            f"{he['avg_speed']:.2f} m/s (top {he['top_speed']:.2f}), "
            f"obstacles {100 * float(np.mean(list(he['sections'].values()))):3.0f}% | "
            f"ends {dict(list(he['ends'].items())[:3])}",
            flush=True,
        )
        sections = float(np.mean(list(he["sections"].values())))
        key = (
            he["finish"],
            0.5 * he["progress"] + 0.5 * sections,
            -(he["lap_time"] or 1e9),
        )
        if key > best[0]:
            best[0] = key
            model.save(args.dir / "best_model")
            print(f"      new best -> {args.dir / 'best_model.zip'}", flush=True)
        model.save(args.dir / "last_model")
        model.save(args.dir / f"step_{model.num_timesteps // 1000}k")

    class Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_at = None
            self.next_ckpt = None
            self.marks = []

        def _on_rollout_end(self):
            # ETAs in SB3's table, at the median rate of recent rollouts so
            # one that paid for a 10 min eval does not skew them.
            self.marks = (self.marks + [(time.time(), self.num_timesteps)])[-9:]
            rates = [
                (s1 - s0) / (t1 - t0)
                for (t0, s0), (t1, s1) in zip(self.marks, self.marks[1:])
                if t1 > t0
            ]
            if not rates:
                return
            rate = float(np.median(rates))
            self.logger.record("eta/steps_per_s", round(rate, 1))
            now = time.time()
            live = dict(pid=os.getpid(), time=now, steps=self.num_timesteps, rate=rate)
            for name, at in (
                ("checkpoint", self.next_ckpt if ckpt_every else None),
                ("eval", self.next_at),
                ("done", target),
            ):
                if at is None:
                    continue
                live[f"next_{name}"] = at
                s = max(0.0, (at - self.num_timesteps) / rate)
                self.logger.record(
                    f"eta/{name}",
                    f"{int(s // 3600)}:{int(s % 3600 // 60):02d} "
                    f"(at {time.strftime('%H:%M', time.localtime(now + s))})",
                )
            # history.json only changes per eval; the rl-train tui reads this
            # between them.  Replaced whole so a reader never sees half a file.
            tmp = args.dir / "live.json.tmp"
            tmp.write_text(json.dumps(live))
            os.replace(tmp, args.dir / "live.json")

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if info and info.get("sim") == "gazebo":
                    if "episode" in info:
                        gazebo_stats["runs"].append(
                            {
                                k: info[k]
                                for k in (
                                    "episode",
                                    "outcome",
                                    "zone",
                                    "dist",
                                    "time",
                                    "start",
                                    "hoops",
                                    "seed",
                                )
                            }
                        )
                    continue
                if info and "episode" in info:
                    rollout_stats["ep_rew"].append(info["episode"]["r"])
                    rollout_stats["outcomes"][info["outcome"]] += 1
                    if info.get("start") == "stuck":
                        rollout_stats["stuck"][0] += 1
                        rollout_stats["stuck"][1] += info["dist"] >= RECOVERED_M
            if self.next_at is None:
                self.next_at = self.num_timesteps + every
            if self.num_timesteps >= self.next_at:
                self.next_at += every
                evaluate_now(f"{self.num_timesteps / 1e6:5.2f}M")
            elif ckpt_every:
                if self.next_ckpt is None:
                    self.next_ckpt = self.num_timesteps + ckpt_every
                if self.num_timesteps >= self.next_ckpt:
                    self.next_ckpt += ckpt_every
                    model.save(args.dir / "last_model")
                    model.save(args.dir / f"step_{model.num_timesteps // 1000}k")
                    runs = gazebo_stats["runs"]
                    print(
                        f"  [checkpoint {self.num_timesteps / 1e6:.2f}M] "
                        f"step_{model.num_timesteps // 1000}k.zip"
                        + (
                            f" | gazebo since eval: {len(runs)} runs, "
                            f"{sum(r['outcome'] == 'finish' for r in runs)} finished, "
                            f"mean {np.mean([r['dist'] for r in runs]):.1f} m"
                            if runs
                            else ""
                        ),
                        flush=True,
                    )
            return True

    print(
        f"training to {target:,} steps in {args.dir}, evaluating every {every:,}"
        + (f", checkpoints every {ckpt_every:,}" if ckpt_every else ""),
        flush=True,
    )
    evaluate_now("start")
    model.learn(
        total_timesteps=total,
        callback=Callback(),
        reset_num_timesteps=args.resume is None,
    )
    evaluate_now("final")
    if workers:
        inner_env.close()
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
