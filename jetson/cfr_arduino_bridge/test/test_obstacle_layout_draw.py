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
        assert {k: list(v) for k, v in drawn["wide_bales"].items()} == exported.get(
            "wide_bales"
        ), path.name


def _independent_track(spec, gap_bale, poses, step=0.02):
    """Is there a path min_track_width wide from entry to the open slot?

    Written separately from the module's check, on a finer grid and with
    8-connected moves, so the two do not share a bug.  Clearance is the exact
    distance from each cell center to every wall and bale.
    """
    wide = spec["wide_bales"]
    length, width = wide["size"]
    boxes = draw_module.wide_walls(spec, gap_bale) + [
        (x, y, yaw, length, width) for x, y, yaw in poses.values()
    ]
    x0, y0, x1, y1 = wide["path_bounds"]
    nx, ny = int((x1 - x0) / step), int((y1 - y0) / step)
    half = wide["min_track_width"] / 2

    def clear(i, j):
        px, py = x0 + (i + 0.5) * step, y0 + (j + 0.5) * step
        for bx, by, yaw, bl, bw in boxes:
            c, s = math.cos(yaw), math.sin(yaw)
            lx = c * (px - bx) + s * (py - by)
            ly = -s * (px - bx) + c * (py - by)
            if math.hypot(max(abs(lx) - bl / 2, 0), max(abs(ly) - bw / 2, 0)) < half:
                return False
        return True

    def cell(x, y):
        return int((x - x0) / step), int((y - y0) / step)

    start, goal = cell(*wide["entry"]), cell(*draw_module.wide_exit(spec, gap_bale))
    seen, frontier = {start}, [start]
    if not clear(*start):
        return False
    while frontier:
        i, j = frontier.pop()
        if (i, j) == goal:
            return True
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = (i + di, j + dj)
                if n not in seen and 0 <= n[0] < nx and 0 <= n[1] < ny and clear(*n):
                    seen.add(n)
                    frontier.append(n)
    return False


@pytest.mark.parametrize("seed", range(12))
def test_wide_section_bales_keep_the_course_rules(spec, seed):
    """Anywhere in the section, any yaw, overlapping nothing, 20 in track kept."""
    layout = draw_module.draw(spec, seed)
    wide = spec["wide_bales"]
    poses = layout["wide_bales"]
    assert sorted(poses) == sorted(wide["names"])
    length, width = wide["size"]
    boxes = [(x, y, yaw, length, width) for x, y, yaw in poses.values()]
    x0, y0, x1, y1 = wide["bounds"]
    fixed = draw_module.wide_walls(spec, layout["gap_bale"]) + [tuple(wide["keep_out"])]
    for index, box in enumerate(boxes):
        for px, py in draw_module._corners(box):
            assert x0 - 1e-9 <= px <= x1 + 1e-9 and y0 - 1e-9 <= py <= y1 + 1e-9
        for other in fixed + boxes[index + 1 :]:
            assert not draw_module._overlap(box, other)
    assert wide["min_track_width"] == pytest.approx(20 * 0.0254)
    assert _independent_track(spec, layout["gap_bale"], poses)


def test_wide_section_yaws_are_not_confined(spec):
    yaws = [
        yaw % math.pi
        for seed in range(30)
        for _, _, yaw in draw_module.draw(spec, seed)["wide_bales"].values()
    ]
    # Any angle: spread over the half turn, not stuck on a few headings.
    assert min(yaws) < 0.3 and max(yaws) > math.pi - 0.3
    assert len({round(y, 2) for y in yaws}) > 150


def test_the_drawing_keeps_the_track_width(spec):
    """The drawn Wide Section itself leaves the 20 in track to every slot."""
    poses = draw_module.nominal(spec)["wide_bales"]
    for slot in spec["gap_bales"]["names"]:
        assert _independent_track(spec, slot, poses), slot


def test_a_blocked_section_is_caught(spec):
    """The path check is live: a bale across the lane's mouth fails it."""
    wide = spec["wide_bales"]
    poses = dict(draw_module.nominal(spec)["wide_bales"])
    first = wide["names"][0]
    poses[first] = (3.9, -8.2, math.pi / 2)  # across the potholes lane's mouth
    assert not draw_module.wide_path_ok(spec, "gap_bale_0", poses)
    assert not _independent_track(spec, "gap_bale_0", poses)


def test_courses_without_a_wide_section_draw_none():
    assert draw_module.draw_wide_bales({}, __import__("random").Random(0)) == {}
