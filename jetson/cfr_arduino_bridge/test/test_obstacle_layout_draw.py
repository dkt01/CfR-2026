"""Tests for the ROS-free Obstacle Course layout draw.

obstacle_randomizer_node calls obstacle_layout_draw for its layouts, and
rl/obstacleRacer trains against layouts exported from the same functions, so
the two only agree if a seed means the same layout in both.  These check the
draw keeps the drawing's rules and is a pure function of the seed, and that
the exported training layouts are still what the draw produces today.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

PACKAGE = Path(__file__).parents[1]
MODULE = PACKAGE / "src" / "obstacle_layout_draw.py"
_spec = importlib.util.spec_from_file_location("obstacle_layout_draw", MODULE)
draw_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = draw_module
_spec.loader.exec_module(draw_module)

LAYOUT_YAML = PACKAGE / "config" / "obstacle_course_layout.yaml"
EXPORTED = PACKAGE.parents[1] / "rl" / "obstacleRacer" / "layouts"


@pytest.fixture(scope="module")
def spec():
    return yaml.safe_load(LAYOUT_YAML.read_text())["obstacle_randomizer"][
        "ros__parameters"
    ]


@pytest.mark.parametrize("seed", range(40))
def test_draw_keeps_the_drawing_rules(spec, seed):
    layout = draw_module.draw(spec, seed)
    buckets = spec["buckets"]
    low = [v + buckets["wall_clearance"] for v in buckets["region_min"]]
    high = [v - buckets["wall_clearance"] for v in buckets["region_max"]]
    points = layout["buckets"]
    assert buckets["count_min"] <= len(points) <= buckets["count_max"]
    for x, y in points:
        assert low[0] - 1e-9 <= x <= high[0] + 1e-9
        assert low[1] - 1e-9 <= y <= high[1] + 1e-9
    for index, a in enumerate(points):
        for b in points[index + 1 :]:
            assert math.dist(a, b) >= buckets["min_spacing"] - 1e-9

    for hoop in spec["hoops"]["names"]:
        start, end = spec["hoops"][hoop]["from"], spec["hoops"][hoop]["to"]
        x, y = layout["hoops"][hoop]
        margin = spec["hoops"][hoop]["base_length"] / 2
        along = math.dist(start, (x, y))
        assert margin - 1e-9 <= along <= math.dist(start, end) - margin + 1e-9
        # On the line, not just near it.
        assert math.dist(start, (x, y)) + math.dist((x, y), end) == pytest.approx(
            math.dist(start, end)
        )
    assert layout["gap_bale"] in spec["gap_bales"]["names"]


def test_same_seed_same_layout(spec):
    assert draw_module.draw(spec, 1234) == draw_module.draw(spec, 1234)
    assert draw_module.draw(spec, 1234) != draw_module.draw(spec, 1235)


def test_every_bucket_count_fits(spec):
    for count in range(spec["buckets"]["count_min"], spec["buckets"]["count_max"] + 1):
        assert len(draw_module.draw(spec, 7, bucket_count=count)["buckets"]) == count


def test_exported_training_layouts_match_the_draw(spec):
    files = sorted(EXPORTED.glob("seed_*.json"))
    if not files:
        pytest.skip("rl/obstacleRacer/layouts not present in this checkout")
    for path in files:
        exported = json.loads(path.read_text())
        drawn = draw_module.draw(spec, exported["seed"])
        assert [list(p) for p in drawn["buckets"]] == exported["buckets"], path.name
        assert {k: list(v) for k, v in drawn["hoops"].items()} == exported["hoops"]
        assert drawn["gap_bale"] == exported["gap_bale"], path.name
