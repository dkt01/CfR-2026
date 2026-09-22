#!/usr/bin/env python3
"""Choose which checkpoint to ship, by re-scoring the whole ladder.

Not named `select.py`: that shadows the standard library module of the same
name, which rclpy and subprocess both import, and this directory is on the
ROS node's sys.path.

    python3 select_policy.py runs/final2           # score every checkpoint
    python3 select_policy.py runs/final2 --export  # ...and export the winner

Why this exists rather than trusting `best_model.zip`:

  * The in-loop evaluation runs 256 field episodes, and the field metric
    carries about +/- 5 points of noise at that size -- measured, by scoring
    one unchanged policy four times and getting 66, 71, 75 and 69%.  A ladder
    picked on single 256-episode scores is partly picking noise.
  * "Best" is a judgement, not a scalar.  Mean return encodes the reward's own
    speed-against-reliability trade, which is the right default, but the rules
    we race to say the car should not touch a bale -- so a policy that is two
    seconds quicker and hits one run in three is not the one to take.  Both
    numbers are printed, and `--min-finish` lets the reliability floor be
    stated out loud instead of hidden in a weighting.

Everything is scored against the SAME seeds, so the comparison between
checkpoints is paired rather than each one getting its own luck.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

import track as track_mod
from env import FormulaOneEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def score(model, cfg, trk, episodes, seed, deterministic):
    env = FormulaOneEnv(cfg, trk, 64, seed=seed, deterministic=deterministic)
    env.random_start = False
    obs = env.reset()
    out = []
    for _ in range(60000):
        action = model.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action)
        for i in np.flatnonzero(term | trunc):
            out.append(info[i])
        if len(out) >= episodes:
            break
    out = out[:episodes]
    # STOPPED, not `finished`: the run is two laps and at rest.  Selecting on
    # the line being crossed would happily pick a checkpoint that arrives at
    # 5 m/s and then coasts into a bale.
    done = [r for r in out if r["stopped"]]
    times = [r["race_time"] for r in done]
    return dict(
        finish=len(done) / len(out),
        time=float(np.mean(times)) if times else float("nan"),
        ret=float(np.mean([r["episode"]["r"] for r in out])),
        clearance=float(np.min([r["min_clearance"] for r in out])),
        grazed=float(np.mean([r["min_clearance"] < cfg["reward"]["graze_margin"] for r in out])),
        overspeed=float(np.max([r["max_overspeed"] for r in out])),
        cte=float(np.mean([r["mean_cte"] for r in out])),
        max_cte=float(np.max([r["max_cte"] for r in out])),
        jerk=float(np.mean([r["steer_jerk_rms"] for r in out])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--min-finish", type=float, default=0.0,
                    help="reliability floor; the fastest checkpoint above it wins")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    cfg_path = args.config or (args.run / "config.yaml")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    trk = track_mod.build(cfg, ROOT)

    ckpts = sorted(args.run.glob("step_*.zip"),
                   key=lambda p: int(p.stem.split("_")[1].rstrip("k")))
    for extra in ("best_model.zip", "last_model.zip"):
        if (args.run / extra).exists():
            ckpts.append(args.run / extra)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {args.run}")

    print(f"\nscoring {len(ckpts)} checkpoints, {args.episodes} field episodes each "
          f"(paired seeds)\n")
    print(f"  {'checkpoint':<18} {'nominal':>8} {'field':>7} {'two laps':>9} "
          f"{'return':>8} {'clear':>7} {'grazed':>7} {'over':>6} "
          f"{'cte':>13} {'jerk':>7}")
    rows = []
    for c in ckpts:
        model = PPO.load(c, device="cpu")
        nom = score(model, cfg, trk, 64, 777, True)
        fld = score(model, cfg, trk, args.episodes, 778, False)
        rows.append((c, nom, fld))
        print(f"  {c.stem:<18} {nom['time']:7.2f}s {100*fld['finish']:6.1f}% "
              f"{fld['time']:8.2f}s {fld['ret']:8.1f} {fld['clearance']:+7.3f} "
              f"{100*fld['grazed']:6.1f}% {fld['overspeed']:+6.2f} "
              f"{nom['cte']:.3f}/{nom['max_cte']:.3f} {nom['jerk']:7.4f}")

    eligible = [r for r in rows if r[2]["finish"] >= args.min_finish]
    if not eligible:
        print(f"\nNothing clears the {100*args.min_finish:.0f}% reliability floor. "
              f"Lower it, or train more.")
        eligible = rows
        pick = max(eligible, key=lambda r: r[2]["ret"])
    elif args.min_finish > 0:
        # Floor stated, so speed decides above it.
        pick = min(eligible, key=lambda r: r[2]["time"])
    else:
        pick = max(eligible, key=lambda r: r[2]["ret"])

    c, nom, fld = pick
    print(f"\n  pick: {c.stem} -- {nom['time']:.2f} s nominal, "
          f"{100*fld['finish']:.1f}% field at {fld['time']:.2f} s, return {fld['ret']:.1f}, "
          f"cte {nom['cte']:.3f} m, jerk {nom['jerk']:.4f}")
    if args.export:
        out = args.run / "policy.npz"
        subprocess.run([sys.executable, str(HERE / "export_policy.py"), str(c),
                        "-o", str(out)], check=True)
        print(f"\n  ship:  {out}")
        print(f"  check: ./validate.sh --policy {out}")


if __name__ == "__main__":
    main()
