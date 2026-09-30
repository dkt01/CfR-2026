#!/usr/bin/env python3
"""Standalone FormulaSubZero planner and safety checks; no ROS or Gazebo."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from formulaTwo import track as track_mod

from controller import FormulaSubZeroDriver, coast_visible_speed_cap
from depth_planner import DepthPlanner, depth_points
from planner import SpeedCoursePlan


def check_failure(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError(f"expected ValueError containing {text!r}")


def main():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    track = track_mod.build(cfg, ROOT)
    plan = SpeedCoursePlan(track, cfg)
    assert plan.min_map_clearance >= cfg["formula_sub_zero"]["min_racing_line_clearance_m"]
    print(f"racing line minimum body clearance: {plan.min_map_clearance:.3f} m")

    bad = copy.deepcopy(cfg)
    bad["formula_sub_zero"]["min_racing_line_clearance_m"] = plan.min_map_clearance + 0.001
    check_failure(lambda: SpeedCoursePlan(track, bad), "racing line body clearance")
    for name in ("mid_hairpin_center_m", "approach_corner_center_m"):
        bad = copy.deepcopy(cfg)
        bad["formula_sub_zero"][name] = track.length
        check_failure(lambda: SpeedCoursePlan(track, bad), name)

    plant = cfg["plant"]
    a0 = plant["coast_f0"] / plant["mass"]
    for room in (0.01, 0.1, 1.0, 7.5):
        v = coast_visible_speed_cap(room, 0.25, plant, 0.70)
        assert v * 0.25 + v * v / (2 * a0) <= room + 1e-9
    assert coast_visible_speed_cap(0, 0.25, plant, 0.70) == 0
    print(f"coast stop bound uses {min(0.70, a0):.3f} m/s²")

    image = np.full((5, 5), 2.0, dtype=float)
    projection = dict(intrinsics=(10.0, 10.0, 2.0, 2.0), height=0.2,
                      pitch=0.0, roll=0.0, camera_x=0.0, stride=1,
                      min_range=0.3, max_range=20.0, local_horizon=10.0)
    assert len(depth_points(image, band=[0.07, 0.33], **projection)) > 0
    assert len(depth_points(image, band=[0.21, 0.33], **projection)) == 0
    projection["max_range"] = 1.0
    assert len(depth_points(image, band=[0.07, 0.33], **projection)) == 0
    print("camera band and range control projected bale points")

    depth = DepthPlanner(track, plan, cfg)
    depth._update_map_consistency(0, 100)
    assert depth.map_consistent
    depth._update_map_consistency(8, 100)
    assert not depth.map_consistent
    for count in (4, 2, 4, 2, 2):
        depth._update_map_consistency(count, 100)
        assert not depth.map_consistent
    depth._update_map_consistency(2, 100)
    assert depth.map_consistent
    print("map trust has immediate hazard response and clean-frame debounce")

    driver = FormulaSubZeroDriver(track, cfg)
    assert driver.n > driver.delay_steps
    print("MPC initialized without ROS or Gazebo")

    class BadOpt:
        def set_value(self, *_args):
            raise TypeError("bad solver parameter shape")

    driver.opt = BadOpt()
    driver.depth.replan = lambda *_args: True
    driver.depth.blocked_distance = 8.0
    station = float(track.start_station)
    x = float(track.at(station, track.x))
    y = float(track.at(station, track.y))
    yaw = float(np.arctan2(track.at(station, track.ty), track.at(station, track.tx)))
    frame = {"station": np.array([station]), "v_cap": np.array([5.0]),
             "v_floor": np.array([0.0]), "steer_ff": np.array([0.0])}
    action = driver.act_frame(frame, x, y, yaw, 1.0, 0.0, 0.0, 1.0,
                              (x, y, yaw), 0.0, False)
    assert action.shape == (1, 2) and driver.failures == 1
    assert "TypeError" in driver.last_solver_error
    print("MPC parameter errors fall back to a finite command")
    print("FormulaSubZero self-test passed")


if __name__ == "__main__":
    main()
