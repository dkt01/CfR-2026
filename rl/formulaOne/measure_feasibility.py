#!/usr/bin/env python3
"""How fast can each randomised car actually be driven cleanly?

    python3 measure_feasibility.py

Four 30M-step runs have now landed on the same speed/reliability frontier:
about 60% field finish at 100 s, 30% at 75 s, 20% at 67 s. Reward reweighting
moved along it; fixing the steering prior's speed ceiling did not move it
either. So the question is whether the frontier is a property of the POLICY or
a property of the ENVIRONMENT -- specifically of how wide `randomize` is.

The trap this is looking for has cost this project a restart before: widen a
range past what is winnable and the policy, which cannot tell which car it
drew, correctly drives EVERY car as slowly as the worst one. The symptom is
exactly what four runs have shown -- a policy that is reliable and far too
slow, and reward changes that only trade one for the other.

Method. Draw N cars from the full joint `randomize` distribution, freeze them,
and for each one find the fastest speed profile the SCRIPTED driver can
complete cleanly (two laps, at rest, never inside the graze band). The driver
is fixed and cannot adapt, so this is a lower bound on what a policy could do
-- but it is measured per car, and that is what makes it useful:

  * the DISTRIBUTION of those speeds says what a policy must drive to cover a
    given fraction of cars.  If the 10th-percentile car tops out at 0.65 of
    the feasible profile, then a policy aiming at 90% finish has to drive
    roughly there, and no reward weight will change that;
  * the CORRELATION between each car's max clean speed and each drawn
    parameter says WHICH axis is paying for it.

Cars are drawn once and replayed at every speed, so the comparison across
speeds is the same car, not a fresh sample.
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

# Per-car quantities worth correlating against.  Plant draws plus the sensor
# draws the env holds itself.
PLANT_KEYS = ("dead_time", "coast_f0", "accel", "slew", "understeer",
              "tire_scrub", "steer_gain", "steer_asym", "steer_offset",
              "steer_limit", "dropout", "yaw_tau", "steer_tau")
ENV_KEYS = ("noise_xy", "noise_yaw", "drift_rate", "jitter")


class Scaled(BaselineDriver):
    """The scripted driver, at a fraction of its own speed profile."""

    def __init__(self, track, config, scale):
        super().__init__(track, config)
        self.scale = scale

    # Signature forwarded rather than spelled out: env.scripted_action grew a
    # v_floor argument and this override silently stopped matching it, which
    # is a TypeError on the first step rather than a wrong answer - but only
    # for whoever runs it next.
    def act(self, *args, **kwargs):
        a = super().act(*args, **kwargs)
        a[:, 1] = 2.0 * np.clip(0.5 * (a[:, 1] + 1.0) * self.scale, 0.0, 1.0) - 1.0
        return a


def run_all(cfg, trk, n, seed, scale, graze_margin):
    """One batched run of n frozen cars.  Returns (clean mask, drawn params)."""
    env = FormulaOneEnv(cfg, trk, n, seed=seed, deterministic=False)
    env.random_start = False
    drv = Scaled(trk, cfg, scale)
    env.reset()
    # Captured AFTER reset and before any stepping: this is the car, and the
    # seed makes it identical at every speed.
    drawn = {k: np.asarray(getattr(env.plant, k)).copy() for k in PLANT_KEYS}
    drawn.update({k: np.asarray(getattr(env, k)).copy() for k in ENV_KEYS})

    out = [None] * n
    for _ in range(9000):
        _, _, term, trunc, info = env.step(env.scripted_action(drv))
        for i in np.flatnonzero(term | trunc):
            if out[i] is None:
                out[i] = info[i]
        if all(o is not None for o in out):
            break
    clean = np.array([
        bool(o) and o["stopped"] and o["min_clearance"] >= graze_margin
        for o in out
    ])
    return clean, drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--cars", type=int, default=96)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    trk = track_mod.build(cfg, ROOT)
    graze = float(cfg["reward"]["graze_margin"])
    ideal = float((trk.ds / __import__("baseline").feasible_profile(trk, cfg)).sum())

    scales = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
    best = np.zeros(args.cars)          # fastest scale each car survived
    drawn = None
    print(f"\n  {args.cars} cars from the full randomize distribution, "
          f"scripted driver, clean = two laps at rest and never inside "
          f"{graze:.2f} m\n")
    print(f"  {'scale':>6} {'~lap':>7} {'clean':>7}")
    for sc in scales:
        clean, d = run_all(cfg, trk, args.cars, args.seed, sc, graze)
        if drawn is None:
            drawn = d
        best = np.where(clean & (best == 0.0), sc, best)
        print(f"  {sc:6.2f} {2*ideal/sc:6.1f}s {100*clean.mean():6.0f}%")

    never = best == 0.0
    print(f"\n  --- what a policy has to drive to cover N% of cars ---")
    print(f"  {'cover':>7} {'max scale':>10} {'~two laps':>11}")
    for pct in (50, 70, 80, 90, 95):
        # The scale at which `pct` of cars are still clean.
        q = np.percentile(best, 100 - pct)
        lap = f"{2*ideal/q:.1f}s" if q > 0 else "impossible"
        print(f"  {pct:6d}% {q:10.2f} {lap:>11}")
    print(f"\n  cars no speed got round cleanly: {100*never.mean():.0f}% "
          f"({never.sum()} of {args.cars})")

    print(f"\n  --- which axis costs the speed ---")
    print(f"  {'parameter':>14} {'range drawn':>22} {'corr with max speed':>21}")
    rows = []
    for k, v in drawn.items():
        v = np.asarray(v, dtype=float)
        if v.std() < 1e-12:
            continue
        c = float(np.corrcoef(v, best)[0, 1])
        rows.append((abs(c), k, v, c))
    for _, k, v, c in sorted(rows, reverse=True):
        flag = "   <-- costs speed" if c < -0.2 else ""
        print(f"  {k:>14} {v.min():9.3f} to {v.max():-9.3f} {c:+21.2f}{flag}")
    print("\n  Negative correlation = cars with MORE of this had to be driven"
          "\n  slower.  A range that dominates here is a range to question.\n")


if __name__ == "__main__":
    main()
