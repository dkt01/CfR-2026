#!/usr/bin/env python3
"""Tune the centerline steering prior for formulaTwo -- to HOLD THE LINE.

    python3 sweep_prior.py                       # Gazebo-fitted car, prints the table
    python3 sweep_prior.py --write               # and writes the winner to config.yaml
    python3 sweep_prior.py --plant 0.19,0.05,0.34   # dead time, servo lag, chassis lag

formulaOne's sweep_prior.py, with four differences, each from a measurement:

  THE CAR IS GAZEBO'S.  fit_gazebo.py replayed a Gazebo run's recorded
  steering through plant.py: the steering chain Gazebo actually has is dead
  time 0.10 s, servo lag 0.02 s and CHASSIS LAG 0.75 s -- against the 0.19 /
  0.05 / 0.34 the old gains were tuned for.  At the weave frequency (0.4 Hz,
  > 3.5 m/s) that fit matches Gazebo's gain (2.92 vs 2.89) and phase (-40 vs
  -40 deg); the old model had 3.88 and -34.  The prior predicts through the
  same numbers, so they are changed for the car AND for the prior's own
  internal model.

  IT DRIVES AT THE SPEED FLOOR, which is where the policy drives (throttle at
  the floor on 100% of ticks) -- 5.0 m/s on the straights.  Gains that hold
  the line slowly are not gains that hold it at race speed.

  IT IS SCORED ON THE CENTERLINE, not on finishing.  Ranked by RMS and worst
  cross-track error on the straights (v > 3.5 m/s, where the weave lives),
  among settings that finish with >= 5 cm of body clearance -- and then
  RANKED ON THE RANDOMISED FIELD: a setting must finish >= 90% of cars
  around the fitted one (or as many as the current gains do) to count.

  IT IS FAST.  Every setting is one car in one batched env -- the prior's
  gains are per-car arrays -- so ~1200 settings drive at once.  The camera is
  not rendered: the prior does not read it.

The top settings are then re-driven on 32 cars randomised around the fitted
car, so the winner is not tuned to one guess of Gazebo.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import re
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEYS = ("horizon_s", "lead_scale", "k_lateral", "k_rate", "k_heading")


def sweep_config(base, dead, servo, lag, laps):
    """The config the sweep drives: Gazebo's steering chain, ranges around it."""
    cfg = copy.deepcopy(base)
    p, r = cfg["plant"], cfg["randomize"]
    p["command_dead_time"], p["steering_tau"], p["yaw_response_tau"] = dead, servo, lag
    # Re-centred on the fit.  dead_time and steering_tau are absolute in the
    # config; yaw_tau_scale multiplies the (new) nominal.
    r["dead_time"] = [max(dead - 0.04, 0.0), dead + 0.06, dead]
    r["steering_tau"] = [max(servo - 0.015, 0.0), servo + 0.04, servo]
    r["yaw_tau_scale"] = [0.75, 1.3, 1.0]
    cfg["env"]["laps"] = laps
    return cfg


def _tile_blocks(obj, n, block, skip=()):
    """Copy the first `block` cars' per-car state onto every later block.

    COMMON RANDOM NUMBERS: every candidate is then driven on exactly the same
    randomised cars (parameters, start pose, sensor draws), so differences
    between candidates are the gains, not the dice.  Without this the SAME
    gains scored 88% and 75% on two draws of 24 cars.
    """
    reps = n // block
    for name, val in list(vars(obj).items()):
        if name in skip:
            continue
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == n:
            val[:] = np.tile(val[:block], (reps,) + (1,) * (val.ndim - 1))
        elif isinstance(val, dict):
            for v in val.values():
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                    v[:] = np.tile(v[:block], (reps,) + (1,) * (v.ndim - 1))


def drive(cfg, trk, gains, deterministic, seed=0, dr_scale=1.0, block=None):
    """Drive the prior alone, at the floor, one car per row of `gains`.

    Returns per-car finish, min clearance, and cross-track statistics on the
    straights and overall.
    """
    from env import FormulaTwoEnv

    n = len(gains["k_lateral"])
    env = FormulaTwoEnv(cfg, trk, n, seed=seed, deterministic=deterministic)
    env.random_start = False
    env.dr_scale = dr_scale
    # No camera: the prior never reads it, and rendering 1200 cars is most of
    # the cost.
    env._capture = lambda mask, delayed=True: np.zeros(
        (int(np.sum(mask)), env.camera.width), dtype=np.float32
    )
    env._camera_tick = lambda: None
    ob = env.obs_builder
    ob.ff_horizon = np.asarray(gains["horizon_s"], float)
    ob.lead_scale = np.asarray(gains["lead_scale"], float)
    ob.k_lateral = np.asarray(gains["k_lateral"], float)
    ob.k_rate = np.asarray(gains["k_rate"], float)
    ob.k_heading = np.asarray(gains["k_heading"], float)

    env.reset()
    if block:
        for obj in (env, env.plant, env.world, env.scans):
            _tile_blocks(obj, n, block)
        _tile_blocks(ob, n, block, skip=("ff_horizon", "lead_scale", "k_lateral", "k_rate", "k_heading"))
        env._sense()
        env._observe()
    action = np.tile([[0.0, -1.0]], (n, 1))  # prior steering, speed floor
    done = np.zeros(n, bool)
    finished = np.zeros(n, bool)
    clear = np.full(n, 9.0)
    s_sum, s_n, s_max = np.zeros(n), np.zeros(n), np.zeros(n)
    a_sum, a_n = np.zeros(n), np.zeros(n)
    for _ in range(int(cfg["env"]["episode_timeout_s"] * 20) + 10):
        _, _, te, tr, info = env.step(action)
        live = ~done
        lat = np.abs(env.lateral_true)
        fast = live & (env.plant.speed > 3.5) & ~env.stopping
        s_sum += np.where(fast, lat**2, 0)
        s_n += fast
        s_max = np.maximum(s_max, np.where(fast, lat, 0))
        racing = live & ~env.stopping
        a_sum += np.where(racing, lat**2, 0)
        a_n += racing
        for i in np.flatnonzero((te | tr) & live):
            done[i] = True
            finished[i] = info[i]["stopped"]
            clear[i] = info[i]["min_clearance"]
        if done.all():
            break
    return dict(
        finished=finished,
        clear=clear,
        rms_straight=np.sqrt(s_sum / np.maximum(s_n, 1)),
        max_straight=s_max,
        rms_all=np.sqrt(a_sum / np.maximum(a_n, 1)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument(
        "--plant",
        default="0.10,0.02,0.75",
        help="dead time, servo lag, chassis lag (s); default: fitted to Gazebo",
    )
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--field", type=int, default=32)
    ap.add_argument("--top", type=int, default=120)
    ap.add_argument("--min-clear", type=float, default=0.05)
    ap.add_argument("--min-field", type=float, default=0.90)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    import track as track_mod

    dead, servo, lag = (float(v) for v in args.plant.split(","))
    base = yaml.safe_load(args.config.read_text())
    cfg = sweep_config(base, dead, servo, lag, args.laps)
    trk = track_mod.build(cfg, ROOT)
    current = {k: float(base["env"]["steer_prior"][k]) for k in KEYS}
    print(
        f"  car: dead time {dead} s, servo lag {servo} s, chassis lag {lag} s; {args.laps} laps at the speed floor"
    )

    grid = dict(
        # The first sweep's winners sat on the low edges of horizon and lead
        # and the high edge of k_heading, so the grid extends past them.
        horizon_s=[0.20, 0.25, 0.30, 0.35, 0.45],
        lead_scale=[0.0, 0.10, 0.25, 0.50],
        k_lateral=[0.8, 1.2, 1.6, 2.0, 2.4],
        k_rate=[0.08, 0.16, 0.24, 0.32],
        k_heading=[0.4, 0.6, 0.8, 1.0],
    )
    combos = [dict(zip(KEYS, c)) for c in itertools.product(*(grid[k] for k in KEYS))]
    combos.append(current)  # the "before", on the same car
    gains = {k: np.array([c[k] for c in combos]) for k in KEYS}
    print(f"  {len(combos)} settings, driven together on the nominal fitted car...")
    nom = drive(cfg, trk, gains, deterministic=True)

    def row(i, extra=""):
        c = combos[i]
        return (
            f"  {c['horizon_s']:5.2f} {c['lead_scale']:5.2f} {c['k_lateral']:5.2f} "
            f"{c['k_rate']:6.2f} {c['k_heading']:6.2f} | "
            f"{'yes' if nom['finished'][i] else ' no'} {nom['clear'][i]:+.3f} "
            f"{nom['rms_straight'][i]:.3f} {nom['max_straight'][i]:.3f} {nom['rms_all'][i]:.3f}{extra}"
        )

    head = (
        f"  {'horiz':>5} {'lead':>5} {'k_lat':>5} {'k_rate':>6} {'k_head':>6} | "
        f"fin clear   rms_st max_st rms_all"
    )
    cur = len(combos) - 1
    print("\n  CURRENT GAINS on the Gazebo-fitted car:")
    print(head)
    print(row(cur))

    ok = np.flatnonzero(nom["finished"] & (nom["clear"] >= args.min_clear))
    ok = ok[ok != cur]
    print(
        f"\n  {len(ok)} of {len(combos) - 1} settings finish with >= {args.min_clear:.2f} m clearance"
    )
    if not len(ok):
        print("  none -- relax --min-clear, or the car cannot be held at this speed")
        return 1
    # ROBUSTNESS IS RANKED, NOT CHECKED AFTERWARDS.  The first version picked
    # the tightest line on the one fitted car and re-drove those on
    # randomised cars: they held 2.8 cm RMS there and finished 25-38% of the
    # field, against the current gains' 100%.  So a broad pool goes to the
    # field, all of it in one batch, and the field decides.
    key = nom["rms_straight"][ok] + 0.5 * nom["max_straight"][ok]
    pool = list(ok[np.argsort(key)[: args.top]]) + [cur]
    F = args.field
    g = {k: np.repeat([combos[i][k] for i in pool], F) for k in KEYS}
    print(
        f"  {len(pool)} candidates x {F} randomised cars = {len(pool) * F} cars in one batch..."
    )
    f = drive(cfg, trk, g, deterministic=False, seed=11, dr_scale=0.5, block=F)
    results = []
    for j, i in enumerate(pool):
        sl = slice(j * F, (j + 1) * F)
        results.append(
            (
                i,
                f["finished"][sl].mean(),
                np.percentile(f["clear"][sl], 10),
                np.median(f["rms_straight"][sl]),
                np.median(f["max_straight"][sl]),
            )
        )
    cur_r = results[-1]
    cands = [r for r in results[:-1] if r[1] >= min(args.min_field, cur_r[1])]
    print(
        f"  {len(cands)} candidates finish >= {100 * min(args.min_field, cur_r[1]):.0f}% of the field "
        f"(current gains: {100 * cur_r[1]:.0f}%)"
    )
    if not cands:
        cands = sorted(results[:-1], key=lambda r: -r[1])[:10]
        print("  none reach it -- showing the most robust instead")
    # Among the robust ones: the line on the field, then clearance.
    cands.sort(key=lambda r: (r[3] + 0.5 * r[4], -r[2]))
    print(head + " | field fin  clr p10  rms_st max_st")
    for r in cands[:12] + [cur_r]:
        tag = "   <- current" if r[0] == cur else ""
        print(
            row(
                r[0], f" | {100 * r[1]:5.0f}%  {r[2]:+.3f}  {r[3]:.3f}  {r[4]:.3f}{tag}"
            ),
            flush=True,
        )
    win = cands[0]
    w = combos[win[0]]
    print(
        "\n  best: "
        + "  ".join(f"{k} {w[k]:.2f}" for k in KEYS)
        + f"\n        nominal fitted car: straights rms {nom['rms_straight'][win[0]]:.3f} m, worst {nom['max_straight'][win[0]]:.3f} m, clearance {nom['clear'][win[0]]:+.3f} m"
        + f"\n        randomised field:   {100 * win[1]:.0f}% finish, clearance p10 {win[2]:+.3f} m, straights rms {win[3]:.3f} m, worst {win[4]:.3f} m"
        + f"\n  current: nominal rms {nom['rms_straight'][cur]:.3f} / worst {nom['max_straight'][cur]:.3f} / clear {nom['clear'][cur]:+.3f}; "
        f"field {100 * cur_r[1]:.0f}%, rms {cur_r[3]:.3f}, worst {cur_r[4]:.3f}"
    )

    if args.write:
        text = args.config.read_text()
        for k in KEYS:
            text = re.sub(rf"(\n    {k}: )[0-9.]+", rf"\g<1>{w[k]}", text, count=1)
        args.config.write_text(text)
        print(f"  written to {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
