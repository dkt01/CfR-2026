#!/usr/bin/env python3
"""What every reward term is actually worth, over a real run.

    python3 reward_probe.py                      # the scripted floor
    python3 reward_probe.py runs/v1/policy.npz   # a trained policy

Reward weights are only meaningful next to each other, and "next to each
other" means at the states the car really visits -- not at the state you had
in mind when you picked the number.  A term can look decisive in config.yaml
and contribute 2% of the return; that exact mistake (`time` at 1.0) once cost
four evaluations of a policy getting steadily slower while three rounds of
tuning went looking for the problem somewhere else.

So this prints two things:

  totals    what each term paid over a whole run, and its share of the
            positive or negative side.  A constraint that is under 5% of the
            negative side is not constraining anything.
  moments   the same terms at the four states that decide a lap -- the fastest
            point, the hairpin entry, the closest approach to a bale, and the
            worst cross-track error -- because a term that is small on average
            can still be the one that decides what the car does at the only
            places that matter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import track as track_mod
from baseline import BaselineDriver
from env import FormulaOneEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", type=Path)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    trk = track_mod.build(cfg, ROOT)
    env = FormulaOneEnv(cfg, trk, 1, seed=3, deterministic=True)
    env.random_start = False

    if args.policy:
        from policy import NumpyPolicy

        net = NumpyPolicy.load(args.policy)
        act = lambda env, obs: net.act(obs)  # noqa: E731
        label = str(args.policy)
    else:
        drv = BaselineDriver(trk, cfg)
        act = lambda env, obs: env.scripted_action(drv)  # noqa: E731
        label = "scripted baseline"

    # reward.step returns the terms, but the env does not keep them, so the
    # terms are recomputed here from the same call the env makes -- by
    # monkeypatching the Reward instance rather than by reimplementing it,
    # because a probe that reimplements the thing it is probing agrees with
    # itself and with nothing else.
    history = []
    inner = env.reward.step

    def spy(**kw):
        total, terms = inner(**kw)
        history.append(
            (
                float(np.ravel(total)[0]),
                {k: float(np.ravel(v)[0]) for k, v in terms.items()},
                dict(
                    speed=float(kw["speed"][0]),
                    v_cap=float(kw["v_cap"][0]),
                    clearance=float(kw["clearance"][0]),
                    lateral=float(kw["lateral"][0]),
                    stopping=float(np.ravel(kw["stopping"])[0]),
                ),
            )
        )
        return total, terms

    env.reward.step = lambda **kw: spy(**kw)

    obs = env.reset()
    rec = None
    for _ in range(6000):
        obs, _, term, trunc, info = env.step(act(env, obs))
        if term[0] or trunc[0]:
            rec = info[0]
            break

    names = list(history[0][1])
    totals = {n: sum(h[1][n] for h in history) for n in names}
    pos = sum(v for v in totals.values() if v > 0)
    neg = -sum(v for v in totals.values() if v < 0)
    ret = sum(totals.values())

    print(f"\n  {label}")
    print(
        f"  {len(history)} ticks, return {ret:.1f}   "
        f"(+{pos:.0f} earned, -{neg:.0f} paid)"
    )
    if rec:
        print(
            f"  race {rec['race_time']:.2f} s, stopped={rec['stopped']}, "
            f"clearance {rec['min_clearance']:+.3f} m, "
            f"cte {rec['mean_cte']:.3f}/{rec['max_cte']:.3f} m, "
            f"jerk {rec['steer_jerk_rms']:.4f}"
        )

    print(f"\n  {'term':<12} {'total':>9} {'share':>7}   {'per second':>11}")
    secs = len(history) * env.dt_nominal
    for n in sorted(names, key=lambda k: -abs(totals[k])):
        v = totals[n]
        share = abs(v) / (pos if v > 0 else neg) * 100 if (pos and neg) else 0.0
        flag = ""
        if v < 0 and 0 < share < 3:
            flag = "  <- under 3% of the cost side: not constraining anything"
        print(f"  {n:<12} {v:+9.1f} {share:6.1f}%   {v / secs:+11.2f}{flag}")

    # The four states that decide a lap.
    racing = [h for h in history if h[2]["stopping"] < 0.5]
    picks = [
        ("fastest", max(racing, key=lambda h: h[2]["speed"])),
        ("slowest", min(racing, key=lambda h: h[2]["speed"])),
        ("closest to a bale", min(racing, key=lambda h: h[2]["clearance"])),
        ("worst cross-track", max(racing, key=lambda h: abs(h[2]["lateral"]))),
    ]
    print(
        f"\n  {'':<20} {'v':>5} {'clear':>7} {'cte':>7}   "
        + " ".join(f"{n[:9]:>9}" for n in names)
    )
    for why, (total, terms, st) in picks:
        print(
            f"  {why:<20} {st['speed']:5.2f} {st['clearance']:+7.3f} "
            f"{st['lateral']:+7.3f}   " + " ".join(f"{terms[n]:>9.2f}" for n in names)
        )
    print()


if __name__ == "__main__":
    main()
