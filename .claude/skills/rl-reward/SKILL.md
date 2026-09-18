---
name: rl-reward
description: Review, diagnose and change the reward (cost) shaping for the CfR-2026 RL learner in rl/bale_follower -- the reward.py terms, the reward block in config.yaml, and the CasADi smoother / MPC tracker cost weights -- with a before-and-after probe of what each term actually pays at real driving states. Use this whenever the user asks to look at, tune, rebalance, add to or explain the reward or cost function, says the policy hugs walls, saws the steering, crashes at hairpins, stalls, brushes bales, drives too slowly or games the reward, or asks why a trained policy behaves the way it does -- and prefer it over editing coefficients straight into config.yaml and starting a run.
---

# CfR-2026 reward and cost shaping

Reward changes are expensive to test: the only honest verdict comes from a
training run that costs five hours or more (`rl-train` skill). So the point of
this skill is to get as much wrong as possible *before* that run — by reading
what the current shaping actually pays, at the states the car really visits,
and by keeping the invariants that stop PPO from finding a shortcut.

```bash
python .claude/skills/rl-reward/scripts/reward_probe.py
python .claude/skills/rl-reward/scripts/reward_probe.py --set k_touch=200 --set k_center=6
```

The probe prints each term's contribution at eight representative states, the
same value held over a whole episode, and the ordering invariants — for the
values in `config.yaml`, then for the proposal, then the delta.

## Where the shaping lives

| file | what it holds |
|---|---|
| `rl/bale_follower/reward.py` | the terms themselves, and `RewardConfig`'s **defaults** |
| `rl/bale_follower/config.yaml` (`reward:`) | the values a run actually trains with |
| `env.py`, `compute_reward` call site | what each term is measured *from* |
| each checkpoint's `.json` | the reward config that checkpoint was trained under |
| `REPORT.md` | why the current values are what they are, with measurements |

**`python reward.py` does not test `config.yaml`.** Its `__main__` block builds
`RewardConfig()` with dataclass defaults, and those have drifted well away from
the trained values — defaults carry `collision_penalty` 50 and `k_touch` 40
where the config trains with 200 and 120. The ordering assertions there are
still the right guardrail, which is why `reward_probe.py` re-runs the same
invariants against the loaded config instead of replacing them.

## Diagnosing before changing

Start from evidence, not from the coefficient that looks wrong. The useful
inputs are the eval JSON written by `evaluate.py` (`collision_rate`,
`mean_speed`, `min_clearance`, `mean_steer_jerk_rad`, per-episode rows) and the
deterministic-eval trend from a run.

| symptom | first suspect | why |
|---|---|---|
| collides in hairpins, fine on straights | `max_speed` / curriculum, not reward | the hairpins have a physical speed limit; no penalty makes an infeasible corner feasible |
| drives fast, scrapes along a wall | `k_center`, `k_proximity` | progress at speed can out-earn a gentle clearance slope |
| brushes bales and keeps going | `k_touch`, `touch_clearance` | contact must score negative at *top* speed, not at cruise |
| saws the steering left-right | `k_steer_reversal` | `k_smooth` charges for changing steering, not for reversing it; only the sign change separates a held corner from chatter |
| creeps, or stops moving entirely | `collision_penalty` too large vs `k_progress` | if crashing costs more than a whole episode of progress can earn, standing still is the rational policy |
| reverses more than it drives | reward pays forward progress only — check the action mapping, not the reward | `forward_progress` projects onto the heading, so reversing already earns negative |
| good stochastic reward, bad deterministic distance | not a reward problem | the mean action is degenerate; see the `rl-train` skill |

## Making a change

1. **Probe first.** `reward_probe.py --set k=v ...` and read the delta column.
   If a change does not move any state you can name, it will not change the
   policy either.
2. **Prefer `config.yaml` to `reward.py`.** Coefficients are meant to be swept
   from the config; the dataclass exists so they can be. Touch `reward.py` only
   to add or remove a *term*.
3. **Keep the invariants true.** Centered beats hugging, clearing beats
   brushing, contact never pays, moving beats standing, anything beats
   crashing, steady steering beats sawing, and a survivable hairpin still pays.
   The last one is the easiest to break by accident: raising `k_center` far
   enough makes slow, tightly-cornered driving score negative, and the policy
   learns to avoid hairpins rather than take them.
4. **Weigh per-step against once.** Every shaping term is charged at
   `control_hz` for up to `episode_time_limit_s` — 600 times per episode today.
   `collision_penalty` is charged once. The probe's `/episode` column is there
   so that comparison is visible rather than imagined.
5. **Adding a term:** give it a field on `RewardConfig` with a comment saying
   what behaviour it exists to remove and what it is measured from, thread it
   through `compute_reward` and its `RewardResult`, add the matching argument
   at the `env.py` call site, and extend both `reward.py`'s `__main__` ordering
   check and the probe's invariants. A term with no invariant is a term nobody
   will notice breaking.
6. **Ground truth is allowed in the reward, never in the observation.**
   `center_error` and progress are computed from the true pose on purpose —
   they shape the line without giving the policy information a real onboard
   sensor could not provide. Keep new terms on that side of the line.
7. **Record the why.** Update the comment above the coefficient in
   `config.yaml` with the measured number that motivated it (that file's
   existing comments are the model — "at v7's values a bale brush at 5.5 m/s
   still scored +0.99 ... now -9.52"), and add a REPORT.md section when a run
   confirms or refutes it.

## After the change

A reward change invalidates comparisons across checkpoints: distance in metres
is still comparable, but reward totals are not, and `evaluate.py` reads the
reward config out of the checkpoint's metadata rather than from `config.yaml`,
so old checkpoints keep scoring under their old shaping. Say which is which
when reporting numbers.

Resuming an existing policy under new shaping is legitimate and much cheaper
than starting over — that is how the curriculum works — but the first few
thousand steps after the change are re-anchoring, not regression. Retrain from
scratch when the change is structural (a new term, a sign flip) rather than a
rebalance.

Hand the run itself to the `rl-train` skill.

## The other cost functions in this directory

"Cost function" in `rl/bale_follower` can mean three different things, and
conflating them wastes an afternoon:

- **`reward.py` + `config.yaml`'s `reward:` block** — what PPO maximizes.
  Everything above.
- **`casadi_smoother.py` (`smoother:` in config.yaml)** — `w_speed`,
  `w_steer`, `w_accel`, `w_steer_rate`: a QP that tracks the policy's requested
  command while penalizing acceleration and steering rate. It is **not** in the
  deployment path for v6 and later policies, which train against the actuator
  limits directly and lose distance when it is added (91.5 m with, 120.5 m
  without). Retune it only for pre-v6 checkpoints or for `path_racer.py`.
- **`mpc_tracker.py`** — `w_position`, `w_terminal`, `w_speed`, `w_accel`,
  `w_steer_rate`: the planned-line racer's tracking cost. Nothing to do with
  RL; changing it does not affect any policy.

If the user says "the cost function" without qualifying, ask which — or infer
from the symptom: policy behaviour is reward, command chatter or lag in the
racer is the smoother or the tracker.
