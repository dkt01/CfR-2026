#!/usr/bin/env python3
"""Tune the steering prior against the model, before spending training on it.

    python3 sweep_prior.py               # coarse sweep, prints the top rows
    python3 sweep_prior.py --write       # and writes the winner to config.yaml

The prior is not a detail of the policy, it IS most of the steering: `env.py`
scales the network's steering output as a residual on top of it, so a prior
that cannot get round the course on its own is a prior the policy spends its
whole budget fighting.  Measured previously on this course: with no prior at
all PPO's `ep_len_mean` sat flat at 20 steps over five million.

`plant.yaw_response_tau` made the old gains wrong.  The prior was tuned when
the model believed a car adopts the yaw rate its wheels imply immediately; at
0.34 s of chassis lag the same gains saturate the steering lock to lock in the
first chicane and put the car into the bales at 21 m -- which is, to within a
metre, where the same scripted driver has always beached in Gazebo.

Scored on the NOMINAL car first (does it get round at all, how fast, how
close) and then on the randomised field, because a prior that only works on
one draw is not a floor.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def drive(cfg, trk, n, seed, deterministic, random_start=False):
    """Run the scripted driver to the end and return the episode records.

    The driver's speed comes from `env.baseline_speed_scale`, and that is the
    single most important thing about this sweep: gains tuned against a slow
    driver are gains that are only good when driven slowly.

    The first version of this file swept at 0.85, whose top speed is 3.76 m/s,
    and the resulting prior held 0.182 m of clearance there and 0.032 m at
    5.08 m/s.  A policy given that prior learns to drive at 3 m/s -- not
    because the reward told it to, but because that is where its steering
    still works.  Sweep at the speed you actually want to drive.
    """
    from baseline import BaselineDriver
    from env import FormulaOneEnv

    env = FormulaOneEnv(cfg, trk, n, seed=seed, deterministic=deterministic)
    env.random_start = random_start
    drv = BaselineDriver(trk, cfg)
    env.reset()
    out = [None] * n
    for _ in range(6000):
        obs, _, term, trunc, info = env.step(env.scripted_action(drv))
        for i in np.flatnonzero(term | trunc):
            if out[i] is None:
                out[i] = info[i]
        if all(o is not None for o in out):
            break
    return [o for o in out if o is not None]


def score(cfg, trk, field_n=64):
    """(nominal record, field finish rate) for one prior setting."""
    nom = drive(cfg, trk, 1, seed=0, deterministic=True)
    if not nom:
        return None, 0.0
    fld = drive(cfg, trk, field_n, seed=7, deterministic=False)
    finish = sum(r["finished"] for r in fld) / max(len(fld), 1)
    return nom[0], finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--field", type=int, default=64)
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=None,
        help="override env.baseline_speed_scale for the sweep; "
        "tune the prior at the speed you want to drive",
    )
    args = ap.parse_args()

    # CFR_FF_HORIZON would override config.yaml inside ObservationBuilder and
    # silently flatten the horizon axis of this sweep into one value.
    os.environ.pop("CFR_FF_HORIZON", None)
    import track as track_mod

    base = yaml.safe_load(args.config.read_text())
    if args.speed_scale is not None:
        base = {
            **base,
            "env": {**base["env"], "baseline_speed_scale": args.speed_scale},
        }
        print(f"  sweeping at baseline_speed_scale {args.speed_scale}")
    trk = track_mod.build(base, ROOT)

    horizons = [0.25, 0.35, 0.45, 0.55]
    leads = [0.0, 0.25, 0.5, 0.75]
    k_lats = [0.4, 0.8, 1.6, 2.4]
    k_rates = [0.0, 0.08, 0.16]
    k_heads = [0.1, 0.2, 0.4]

    rows = []
    total = len(horizons) * len(leads) * len(k_lats) * len(k_rates) * len(k_heads)
    print(f"  {total} prior settings, scored on the nominal car\n")
    for i, (h, lead, kl, kr, kh) in enumerate(
        itertools.product(horizons, leads, k_lats, k_rates, k_heads)
    ):
        cfg = {
            **base,
            "env": {
                **base["env"],
                "steer_prior": {
                    **base["env"]["steer_prior"],
                    "horizon_s": h,
                    "lead_scale": lead,
                    "k_lateral": kl,
                    "k_rate": kr,
                    "k_heading": kh,
                },
            },
        }
        nom = drive(cfg, trk, 1, seed=0, deterministic=True)
        if not nom:
            continue
        r = nom[0]
        rows.append(
            dict(
                h=h,
                lead=lead,
                kl=kl,
                kr=kr,
                kh=kh,
                rec=r,
                ok=bool(r["finished"]),
                t=r["episode"]["t"],
                clear=r["min_clearance"],
                dist=r["distance"],
            )
        )
        if (i + 1) % 25 == 0:
            done = sum(x["ok"] for x in rows)
            print(f"    {i + 1}/{total}   {done} get round so far", flush=True)

    finishers = [r for r in rows if r["ok"]]
    print(
        f"\n  {len(finishers)} of {len(rows)} settings complete two laps "
        f"on the nominal car"
    )
    if not finishers:
        best = sorted(rows, key=lambda r: -r["dist"])[:10]
        print("\n  NONE finish.  Furthest, so the next sweep knows where to look:")
        print(
            f"  {'horizon':>8} {'lead':>6} {'k_lat':>6} {'k_rate':>7} "
            f"{'k_head':>7} {'metres':>8}"
        )
        for r in best:
            print(
                f"  {r['h']:8.2f} {r['lead']:6.2f} {r['kl']:6.2f} "
                f"{r['kr']:7.2f} {r['kh']:7.2f} {r['dist']:8.1f}"
            )
        sys.exit(1)

    # Among the ones that get round, prefer clearance and then time: the
    # prior is a floor to be stable, not a lap record.  The policy makes it
    # fast.
    finishers.sort(key=lambda r: (-min(r["clear"], 0.20), r["t"]))
    print(
        f"\n  {'horizon':>8} {'lead':>6} {'k_lat':>6} {'k_rate':>7} "
        f"{'k_head':>7} {'time':>7} {'min clear':>10} {'field':>7}"
    )
    shown = finishers[:16]
    for r in shown:
        cfg = {
            **base,
            "env": {
                **base["env"],
                "steer_prior": {
                    **base["env"]["steer_prior"],
                    "horizon_s": r["h"],
                    "lead_scale": r["lead"],
                    "k_lateral": r["kl"],
                    "k_rate": r["kr"],
                    "k_heading": r["kh"],
                },
            },
        }
        fld = drive(cfg, trk, args.field, seed=7, deterministic=False)
        r["field"] = sum(x["stopped"] for x in fld) / max(len(fld), 1)
        print(
            f"  {r['h']:8.2f} {r['lead']:6.2f} {r['kl']:6.2f} {r['kr']:7.2f} "
            f"{r['kh']:7.2f} {r['t']:7.2f} {r['clear']:+10.3f} "
            f"{100 * r['field']:6.0f}%",
            flush=True,
        )

    # The field rate is the one that matters -- it is the same question the
    # trained policy is finally judged on -- with clearance breaking ties.
    win = max(shown, key=lambda r: (round(r["field"], 2), min(r["clear"], 0.20)))
    print(
        f"\n  best: horizon {win['h']:.2f}  lead {win['lead']:.2f}  "
        f"k_lateral {win['kl']:.2f}  k_rate {win['kr']:.2f}  "
        f"k_heading {win['kh']:.2f}  "
        f"-> {win['t']:.2f} s nominal, {100 * win['field']:.0f}% field"
    )

    if args.write:
        text = args.config.read_text()
        for key, val in (
            ("horizon_s", win["h"]),
            ("lead_scale", win["lead"]),
            ("k_lateral", win["kl"]),
            ("k_rate", win["kr"]),
            ("k_heading", win["kh"]),
        ):
            import re

            text = re.sub(rf"(\n    {key}: )[0-9.]+", rf"\g<1>{val}", text, count=1)
        args.config.write_text(text)
        print(f"  written to {args.config}")


if __name__ == "__main__":
    main()
