#!/usr/bin/env python3
"""Rescore checkpoints on the same start-box laps, to pick a race policy.

One deterministic start-box lap per car, 256 cars: the 32 held-out layouts
x4 and 64 never-seen layouts (seeds 5101-5164) x2 -- the laps v9 was chosen
on (bestModel/v9/graded_eval.txt).  Every checkpoint drives the same laps
(same env seed), so their rates compare directly.

    python3 grade.py runs/v10/step_*.zip runs/v9/step_196001k.zip
    python3 grade.py runs/v10/step_1980*.zip --config runs/v10/config.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import yaml

import course_model
import layouts
from ppo_policy import Driver, load
from train import EvalEnv, evaluate, summarize

HERE = Path(__file__).resolve().parent
NEW_SEEDS = list(range(5101, 5165))


def grade(paths, cfg):
    held = layouts.HELDOUT_SEEDS
    seeds = held + NEW_SEEDS
    model_ = course_model.CourseModel(seeds)
    held_ids = np.arange(len(held))
    new_ids = np.arange(len(held), len(seeds))
    rows = []
    for path in paths:
        model = load(path, device="cpu")
        out = {}
        for name, ids, per in (("held", held_ids, 4), ("new", new_ids, 2)):
            env = EvalEnv(
                cfg, model_, per * len(ids), ids, seed=20_000, start_box_only=True
            )
            out[name] = summarize(evaluate(env, lambda n: Driver(model, n), per))
        n_h, n_n = out["held"]["n"], out["new"]["n"]
        fin = (out["held"]["finish"] * n_h + out["new"]["finish"] * n_n) / (n_h + n_n)
        dist = (out["held"]["progress_m"] * n_h + out["new"]["progress_m"] * n_n) / (
            n_h + n_n
        )
        laps = [x for x in (out["held"]["lap_time"], out["new"]["lap_time"]) if x]
        row = dict(
            path=str(path),
            finish=fin,
            dist=dist,
            lap=float(np.mean(laps)) if laps else None,
            held=out["held"]["finish"],
            held_m=out["held"]["progress_m"],
            new=out["new"]["finish"],
            new_m=out["new"]["progress_m"],
            ends_held=out["held"]["ends"],
            ends_new=out["new"]["ends"],
        )
        rows.append(row)
        print(
            f"{Path(path).name:22s} all fin {100 * fin:5.1f}%  {dist:4.1f} m lap "
            f"{row['lap'] or 0:5.1f} s | held {100 * row['held']:5.1f}%  {row['held_m']:4.1f} m "
            f"| new {100 * row['new']:5.1f}%  {row['new_m']:4.1f} m",
            flush=True,
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument(
        "--config", type=Path, default=None, help="default: beside the first checkpoint"
    )
    ap.add_argument("--out", type=Path, default=None, help="JSON of every row")
    args = ap.parse_args()
    paths = sorted({p for pat in args.checkpoints for p in glob.glob(pat)})
    config = args.config or Path(paths[0]).parent / "config.yaml"
    cfg = yaml.safe_load(config.read_text())
    rows = grade(paths, cfg)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
