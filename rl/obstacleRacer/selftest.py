#!/usr/bin/env python3
"""Checks that need neither ROS nor Gazebo, run before every training run.

  python3 selftest.py            # all of it
  python3 selftest.py --quick    # skip the long prior-only rollout

1. layouts match the randomizer's draw
2. reward: reachable thresholds and episode-level incentives
3. plant: static attitudes on the ramp and bank, a wheel in a recess, a
   bump transient, and the front going light off the deck crest
4. the env steps, and episodes end for the reasons they should
5. the prior alone, from the start box: how far it gets, and that the car
   pitched and rolled on the 3D sections (a regression that flattens the
   car back to planar fails here, before a run)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import course_model
import layouts
import plant as P
import reward

HERE = Path(__file__).resolve().parent


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    return 0 if ok else 1


def plant_checks(cfg, model):
    failures = 0
    cfg = dict(cfg, randomize=dict(cfg["randomize"], enabled=False))
    spots = [
        ("ramp", 5.0, 0.0, 0.31, 0.0),
        ("bank", -4.03, -9.37, 0.09, -math.pi / 2),
        ("flat", 0.0, 0.1, 0.0, 0.0),
    ]
    pl = P.Plant(cfg, model, len(spots), np.random.default_rng(0))
    v = np.array([s[1:] for s in spots])
    z = np.zeros(len(spots))
    pl.reset(np.arange(len(spots)), 0, v[:, 0], v[:, 1], v[:, 2], v[:, 3], z)
    ramp, bank, flat = pl.state
    failures += check(
        "resting on the ramp pitches the body with it",
        abs(math.degrees(ramp[P.S_PITCH]) - math.degrees(math.atan(0.19))) < 1.2,
        f"{math.degrees(ramp[P.S_PITCH]):.2f} deg vs {math.degrees(math.atan(0.19)):.2f}",
    )
    failures += check(
        "resting on the bank rolls the body with it",
        abs(abs(math.degrees(bank[P.S_ROLL])) - 8.5) < 1.2,
        f"{math.degrees(bank[P.S_ROLL]):.2f} deg vs 8.5",
    )
    failures += check(
        "on the flat the springs sit at ride height",
        np.abs(flat[P.S_Q : P.S_Q + 4]).max() < 0.001 and abs(flat[P.S_Z]) < 0.001,
        f"q {np.abs(flat[P.S_Q : P.S_Q + 4]).max() * 1000:.2f} mm",
    )

    # Drive straight across the pothole board and off the deck crest, and
    # watch the body.
    runs = [
        ("potholes", -1.2, -8.3, 0.0, 0.0, 1.5, 60),
        ("deck crest", 6.2, 0.0, 0.52, 0.0, 3.0, 30),
    ]
    pl = P.Plant(cfg, model, len(runs), np.random.default_rng(0))
    v = np.array([r[1:5] for r in runs])
    speeds = np.array([r[5] for r in runs])
    pl.reset(np.arange(len(runs)), 0, v[:, 0], v[:, 1], v[:, 2], v[:, 3], speeds)
    trace = []
    for _ in range(max(r[6] for r in runs)):
        pl.step(np.zeros(len(runs)), speeds, 10)
        trace.append(pl.state.copy())
    trace = np.array(trace)
    pot = trace[:, 0]
    on_board = (pot[:, P.S_X] > 0.1) & (pot[:, P.S_X] < 2.4)
    pitch_sd = np.degrees(pot[on_board, P.S_PITCH]).std()
    roll_sd = np.degrees(pot[on_board, P.S_ROLL]).std()
    failures += check(
        "crossing the potholes moves the body in pitch and roll",
        pitch_sd > 0.1 and roll_sd > 0.1,
        f"pitch sd {pitch_sd:.2f} deg, roll sd {roll_sd:.2f} deg",
    )
    loads = trace[:, 0, P.S_LOAD : P.S_LOAD + 4][on_board]
    failures += check(
        "wheels load and unload over the recesses and bumps",
        (loads.max(0) - loads.min(0)).min() > 1.0,
        f"per-wheel load swing {np.round(loads.max(0) - loads.min(0), 1)} N",
    )
    crest = trace[:, 1]
    front = crest[:, P.S_LOAD] + crest[:, P.S_LOAD + 1]
    failures += check(
        "the front goes light over the deck crest at 3 m/s",
        front.min() < 0.6 * front[:4].mean(),
        f"front load min {front.min():.1f} N vs {front[:4].mean():.1f} N on the ramp",
    )
    return failures


def gate_parity(cfg, model):
    """sensor.py's gate features == observation.gate_features on the same gate.

    The car's node builds gate features from the segmenter's gates with
    observation.gate_features; training builds them in sensor.py's kernel.
    Put a car in front of hoop_0 and compare.
    """
    import observation as O
    import sensor as S

    cfg = dict(cfg, randomize=dict(cfg["randomize"], enabled=False))
    cfg["sensor"] = dict(
        cfg["sensor"], noise_a=0.0, noise_b=0.0, dropout=0.0, phantom=0.0
    )
    gates = S.gate_table(model)[0]
    hoop = next(g for g in gates if g[0] == S.HOOP)
    x, y, yaw = hoop[1] + 1.6, hoop[2] + 0.2, math.pi + 0.15
    pl = P.Plant(cfg, model, 1, np.random.default_rng(0))
    pl.reset(
        np.arange(1),
        0,
        np.array([x]),
        np.array([y]),
        np.zeros(1),
        np.array([yaw]),
        np.zeros(1),
    )
    sen = S.Sensor(cfg, model, 1, np.random.default_rng(0))
    _, gate = sen.read(pl.lay, pl.state)
    st = pl.state[0]
    cp = math.cos(st[P.S_PITCH])
    cx = st[P.S_X] + 0.315 * cp * math.cos(st[P.S_YAW])
    cy = st[P.S_Y] + 0.315 * cp * math.sin(st[P.S_YAW])
    c, s = math.cos(-st[P.S_YAW]), math.sin(-st[P.S_YAW])
    dx, dy = hoop[1] - cx, hoop[2] - cy
    ax, ay = hoop[3], hoop[4]
    local = (
        O.HOOP_KIND,
        c * dx - s * dy,
        s * dx + c * dy,
        c * ax - s * ay,
        s * ax + c * ay,
    )
    want = O.gate_features([local], cfg)
    err = float(np.abs(gate[0, :4] - want[:4]).max())
    return check(
        "sensor.py gate features match observation.gate_features",
        gate[0, 0] == 1.0 and err < 1e-6,
        f"max diff {err:.2e}",
    )


def rollout(cfg, model, cars, seconds, start_box_only=True):
    import env as env_module

    env = env_module.ObstacleEnv(
        cfg,
        model,
        cars,
        np.arange(len(model.layouts)),
        seed=1,
        start_box_only=start_box_only,
    )
    env.reset()
    infos = []
    t = time.time()
    steps = int(seconds * float(cfg["env"]["control_hz"]))
    for _ in range(steps):
        _, _, term, trunc, info = env.step(env.scripted_action())
        infos += [i for i in info if i]
    rate = cars * steps / (time.time() - t)
    return env, infos, rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    failures = 0

    print("layouts")
    stale = [
        s
        for s in layouts.TRAIN_SEEDS + layouts.HELDOUT_SEEDS
        if layouts.load(s) != layouts.draw(s)
    ]
    failures += check(
        "exported layouts match the randomizer's draw", not stale, str(stale)
    )

    print("reward")
    try:
        reward.assert_reachable(cfg)
        reward.assert_episode_incentives(cfg)
        failures += check("thresholds reachable, episode incentives in order", True)
    except AssertionError as error:
        failures += check(
            "thresholds reachable, episode incentives in order", False, str(error)
        )

    model = course_model.CourseModel(layouts.TRAIN_SEEDS + layouts.HELDOUT_SEEDS)
    print("plant")
    failures += plant_checks(cfg, model)
    print("sensor")
    failures += gate_parity(cfg, model)

    print("env")
    env, infos, rate = rollout(cfg, model, 64, 4.0 if args.quick else 60.0)
    failures += check("env steps", rate > 0, f"{rate:.0f} car-steps/s")
    if infos:
        outcomes = Counter(i["outcome"] for i in infos)
        reach = np.array([i["s_end"] for i in infos])
        print(
            f"  prior alone, start box: {len(infos)} episodes, outcomes {dict(outcomes)}"
        )
        print(
            f"  distance: median {np.median(reach):.1f} m, best {reach.max():.1f} m, "
            f"hoops {Counter(i['hoops'] for i in infos)}"
        )
        print(
            "  where they ended:",
            dict(Counter(i["zone"] for i in infos).most_common(8)),
        )
        if not args.quick:
            att = {}
            for i in infos:
                for zone, (p, r) in i["attitude"].items():
                    a = att.setdefault(zone, [0.0, 0.0])
                    a[0], a[1] = max(a[0], p), max(a[1], r)
            for zone in ("overpass_ramp", "helical_ramp", "banked_turn", "potholes"):
                if zone in att:
                    p, r = att[zone]
                    failures += check(
                        f"body moves on {zone}",
                        p > 0.5 or r > 0.5,
                        f"max |pitch| {p:.1f} deg, |roll| {r:.1f} deg",
                    )
                else:
                    print(f"  ---- {zone}: not reached by the prior; not checked")
    print("PASS" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
