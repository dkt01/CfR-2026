#!/usr/bin/env python3
"""The Obstacle Course layouts the racer trains and is judged on.

Each layout is what obstacle_randomizer_node puts in Gazebo for one seed --
bucket positions, hoop positions and which wall bale is parked -- drawn by
the node's own ROS-free draw module, so "seed 104" means the same course in
the numpy trainer, in Gazebo validation and in the node's log line.

Ten seeds are trained on.  Four more are held out, one per entrance slot,
and only ever evaluated, so a policy that has memorized its ten courses
rather than learned to drive shows up as a gap between the two finish rates.

    python3 layouts.py            # (re)write layouts/seed_*.json
    python3 layouts.py --check    # fail if the files disagree with the draw
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = REPO / "jetson" / "cfr_arduino_bridge"
LAYOUT_YAML = PACKAGE / "config" / "obstacle_course_layout.yaml"
LAYOUT_DIR = HERE / "layouts"

TRAIN_SEEDS = list(range(101, 111))
# One per gap slot, so held-out finish rate covers every entrance.
HELDOUT_SEEDS = [201, 202, 208, 218]


def _draw_module():
    path = PACKAGE / "src" / "obstacle_layout_draw.py"
    spec = importlib.util.spec_from_file_location("obstacle_layout_draw", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def layout_spec() -> dict:
    return yaml.safe_load(LAYOUT_YAML.read_text())["obstacle_randomizer"][
        "ros__parameters"
    ]


def draw(seed: int) -> dict:
    layout = _draw_module().draw(layout_spec(), seed)
    return {
        "seed": seed,
        "buckets": [list(p) for p in layout["buckets"]],
        "hoops": {k: list(v) for k, v in layout["hoops"].items()},
        "gap_bale": layout["gap_bale"],
    }


def load(seed: int) -> dict:
    path = LAYOUT_DIR / f"seed_{seed}.json"
    return json.loads(path.read_text())


def load_all(seeds) -> list[dict]:
    return [load(seed) for seed in seeds]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    failures = 0
    LAYOUT_DIR.mkdir(exist_ok=True)
    for seed in TRAIN_SEEDS + HELDOUT_SEEDS:
        layout = draw(seed)
        path = LAYOUT_DIR / f"seed_{seed}.json"
        text = json.dumps(layout, indent=1) + "\n"
        if args.check:
            if not path.exists() or json.loads(path.read_text()) != layout:
                print(f"STALE {path.name}")
                failures += 1
            continue
        path.write_bytes(text.encode())
        role = "train" if seed in TRAIN_SEEDS else "held out"
        print(
            f"seed {seed} ({role}): {len(layout['buckets'])} buckets, "
            f"gap at {layout['gap_bale']}"
        )
    if args.check:
        print("layouts match the draw" if not failures else f"{failures} stale")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
