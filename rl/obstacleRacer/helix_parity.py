#!/usr/bin/env python3
"""Where the numpy plant and Gazebo part ways on the helix (or any section).

Reads a `gazebo_env.py trace` file (the policy driven into a section from a
fixed arc length before it, every control step saved) and, on the host:

  numpy     the same policy from the same starts in the numpy env, many cars
            per layout: how its runs end, against Gazebo's;
  replay    Gazebo's own commands, 1 s at a time, through the numpy plant from
            Gazebo's pose: position, heading and speed drift per window, and
            whether the plant touches where Gazebo touched;
  crashes   what the Gazebo judge saw at each crash: speed from the pose,
            tach, clearance, pose age.

    python3 helix_parity.py runs/helix/gz_trace_a.json --policy runs/helix/v11_198M.npz \\
        --config runs/v11/config.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import course_model
import observation as O
import plant as P
from centerline import Centerlines
from env import CLEARANCE_RINGS, ObstacleEnv
from policy import NumpyPolicy

WINDOW = 20  # control steps per open-loop replay (1 s)


class FloorTach:
    """The tach before observation.Tach: |v|, zero under the floor, at once."""

    def reset(self, idx, speed):
        self.v = np.abs(np.asarray(speed, dtype=np.float64))

    def read(self, idx):
        return np.where(self.v < O.TACH_FLOOR, 0.0, self.v)

    def update(self, speed, idx=None):
        self.reset(idx, speed)
        return self.read(idx)


def numpy_runs(cfg, seeds, section, before, policy_path, per, seconds, dealt, old_tach):
    model = course_model.CourseModel(seeds)
    n = per * len(seeds)
    # Gazebo deals every start at rest (a teleport sets no speed).
    cfg = dict(cfg, env=dict(cfg["env"], dealt_speed=[dealt, dealt]))
    env = ObstacleEnv(cfg, model, n, np.arange(len(seeds)), seed=7)
    if old_tach:
        env.tach = FloorTach()
    lays = np.repeat(np.arange(len(seeds)), per)
    env.forced_lay = lays
    env.forced_s = np.array([env.section_entry(k, section) - before for k in lays])
    obs = env.reset()
    policy = NumpyPolicy.load(policy_path)
    policy.reset(n)
    out = [None] * n
    for _ in range(int(seconds * float(cfg["env"]["control_hz"]))):
        obs, _, term, trunc, info = env.step(policy.act(obs))
        live = np.array([o is None for o in out])
        for i in np.flatnonzero((term | trunc) & live):
            out[i] = (info[i]["outcome"], info[i]["zone"])
        if all(o is not None for o in out):
            break
    return [o or ("cut", None) for o in out], lays


def replay(cfg, seeds, run, k):
    """Gazebo's commands through the numpy plant, WINDOW steps at a time."""
    cfg = copy.deepcopy(cfg)
    cfg["randomize"]["enabled"] = False
    model = course_model.CourseModel([seeds[k]])
    plant = P.Plant(cfg, model, 1, np.random.default_rng(0))
    line = Centerlines(model.layouts, model=model).lines[0]
    rows = run["rows"]
    out = []
    for i in range(1, len(rows) - WINDOW, WINDOW // 2):
        x, y, zg, yaw, _, _ = rows[i - 1]["pose"]
        z = float(line.points[line.nearest_index((x, y, zg)), 2])
        plant.reset([0], [0], [x], [y], [z], [yaw], [rows[i - 1]["v"]])
        st = plant.state[0]
        st[P.S_R] = rows[i - 1]["yaw_rate"]
        prev = rows[i - 2]["cmd"] if i >= 2 else rows[i]["cmd"]
        st[P.S_STEER] = prev[0]
        plant.history[0, :, 0] = prev[0]
        plant.history[0, :, 1] = prev[1]
        touch_np = lost_np = 0.0
        clear_np = math.inf
        for j in range(i, i + WINDOW):
            steer, speed = rows[j]["cmd"]
            plant.push_command(np.array([steer]), np.array([speed]))
            t, lost, _ = plant.step(np.array([steer]), np.array([speed]), 10)
            touch_np += float(t[0])
            lost_np = max(lost_np, float(lost[0]))
            clear_np = min(
                clear_np,
                float(
                    P.body_clearance(
                        plant.OBS, plant.lay, plant.state, CLEARANCE_RINGS
                    )[0]
                ),
            )
        g = rows[i + WINDOW - 1]
        st = plant.state[0]
        dyaw = math.degrees(
            (st[P.S_YAW] - g["pose"][3] + math.pi) % (2 * math.pi) - math.pi
        )
        gz_clear = min(r["clear"] for r in rows[i : i + WINDOW])
        out.append(
            dict(
                t=i * 0.05,
                x=x,
                y=y,
                pos_err=math.hypot(st[P.S_X] - g["pose"][0], st[P.S_Y] - g["pose"][1]),
                yaw_err=dyaw,
                v_np=float(st[P.S_V]),
                v_gz=g["v"],
                clear_np=clear_np,
                clear_gz=gz_clear,
                touch_np=touch_np,
                lost_np=lost_np,
                lost_gz=max(r["lost"] for r in rows[i : i + WINDOW]),
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--per", type=int, default=32, help="numpy cars per layout")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument(
        "--dealt-speed", type=float, default=0.0, help="m/s; Gazebo's are at rest"
    )
    ap.add_argument(
        "--old-tach", action="store_true", help="numpy with the floor-only tach"
    )
    ap.add_argument("--numpy-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data = json.loads(args.trace.read_text())
    runs = data["runs"]
    seeds = sorted({r["seed"] for r in runs}, key=[r["seed"] for r in runs].index)

    section = data.get("section", "helical_ramp")  # early traces were helix only
    print(f"== outcomes from {data['before']} m before {section}, {args.seconds:.0f} s")
    gz = Counter(f"{r['outcome']}@{r['zone']}" for r in runs)
    print(f"gazebo ({len(runs)} runs): {dict(gz.most_common())}")
    ends, lays = numpy_runs(
        cfg,
        seeds,
        section,
        data["before"],
        args.policy,
        args.per,
        args.seconds,
        args.dealt_speed,
        args.old_tach,
    )
    npc = Counter(f"{o}@{z}" for o, z in ends)
    tach = "floor-only tach" if args.old_tach else "firmware tach"
    print(f"numpy  ({len(ends)} runs, {tach}): {dict(npc.most_common(8))}")
    if args.numpy_only:
        return

    print("\n== crashes and rollovers in Gazebo: the last 6 steps")
    crash_speed = float(cfg["env"]["crash_speed"])
    for r in runs:
        if r["outcome"] not in ("crash", "rollover", "pinned"):
            continue
        rows = r["rows"]
        print(f"seed {r['seed']} run {r['rep']}: {r['outcome']} at {r['zone']}")
        for j in range(max(0, len(rows) - 6), len(rows)):
            w = rows[j]
            dstamp = w["stamp"] - rows[j - 1]["stamp"] if j else 0.0
            x, y, z, yaw, pitch, roll = w["pose"]
            print(
                f"   t {j * 0.05:5.2f} ({x:5.2f},{y:5.2f},{z:4.2f}) yaw {math.degrees(yaw):6.1f} "
                f"pitch {math.degrees(pitch):5.1f} roll {math.degrees(roll):5.1f} | "
                f"v {w['v']:5.2f} tach {w['tach']:5.2f} cmd ({w['cmd'][0]:+.2f},{w['cmd'][1]:4.2f}) "
                f"clear {100 * w['clear']:5.1f} cm lost {w['lost']:4.2f}"
                f"{' >crash' if w['lost'] > crash_speed and w['clear'] <= 0.03 else ''} "
                f"| pose age {1000 * (w['now'] - w['stamp']):4.0f} ms, d {1000 * dstamp:4.0f} ms"
            )

    print(
        "\n== open-loop replay: Gazebo's commands through the numpy plant, 1 s windows"
    )
    agg = []
    for r in runs:
        k = seeds.index(r["seed"])
        for w in replay(cfg, seeds, r, k):
            agg.append(w)
            flag = ""
            if w["clear_gz"] <= 0.03 and w["touch_np"] == 0:
                flag = "  <- Gazebo touched, numpy did not"
            elif w["clear_gz"] > 0.03 and w["touch_np"] > 0:
                flag = "  <- numpy touched, Gazebo did not"
            print(
                f"seed {r['seed']} run {r['rep']} t {w['t']:5.2f} ({w['x']:5.2f},{w['y']:5.2f}): "
                f"pos {100 * w['pos_err']:5.1f} cm yaw {w['yaw_err']:+6.1f} deg "
                f"v np {w['v_np']:5.2f} gz {w['v_gz']:5.2f} | clear np {100 * w['clear_np']:5.1f} "
                f"gz {100 * w['clear_gz']:5.1f} cm lost np {w['lost_np']:4.2f} gz {w['lost_gz']:4.2f}{flag}"
            )
    if agg:
        pe = np.array([w["pos_err"] for w in agg])
        ye = np.abs([w["yaw_err"] for w in agg])
        print(
            f"\nper 1 s window: position error median {100 * np.median(pe):.1f} cm, "
            f"p90 {100 * np.percentile(pe, 90):.1f} cm; yaw error median {np.median(ye):.1f} deg, "
            f"p90 {np.percentile(ye, 90):.1f} deg over {len(agg)} windows"
        )


if __name__ == "__main__":
    main()
