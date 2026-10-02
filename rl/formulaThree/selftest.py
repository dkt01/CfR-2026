#!/usr/bin/env python3
"""Offline numerical regressions; no ROS or simulator required."""

from copy import deepcopy
from pathlib import Path
import unittest
import time

import numpy as np
import yaml

from baseline import BaselineDriver
from env import FormulaThreeEnv
from metrics import selection_key
from observation import scale_action
from perception import Camera, INVALID
from policy import NumpyPolicy, config_digest
from reward import Reward
from reward_probe import state
import track
from validation import verdict

HERE = Path(__file__).resolve().parent


class FormulaThreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((HERE / "config.yaml").read_text())
        cls.track = track.build(cls.cfg, HERE.parents[1])

    def test_full_lift_and_reference_speed(self):
        act = np.array([[0, -1], [0, 0], [0, 1]])
        steer, speed = scale_action(
            act, np.full(3, 5.2), np.zeros(3), 0.5, np.full(3, 4.0)
        )
        np.testing.assert_allclose(speed, [0, 4.0, 5.2])
        np.testing.assert_allclose(steer, 0)

    def test_baseline_inverse(self):
        driver = BaselineDriver(self.track, self.cfg)
        station = self.track.s[::10]
        speed = np.full_like(station, 3.0)
        cap = self.track.at(station, self.track.v_cap)
        ref = cap * 0.8
        act = driver.act(station, speed, cap, ref)
        _, actual = scale_action(act, cap, np.zeros_like(cap), 0.5, ref)
        expected = np.minimum(
            self.track.at(station + speed * driver.dead_time, driver.v_ref), cap
        )
        np.testing.assert_allclose(actual, expected)

    def test_no_contact_finish_bonus(self):
        values = state(clearance=0, crash=True)
        values.update(lapped=1.0, finished=1.0, stopped=1.0)
        reward, terms = Reward(self.cfg).step(**values)
        self.assertLess(reward, 0)
        self.assertEqual(float(terms["finish"] + terms["stop"] + terms["lap"]), 0)

    def test_clearance_covers_bumper_midpoint(self):
        env = FormulaThreeEnv(self.cfg, self.track, 1, deterministic=True)
        env.reset()
        # A bale corner can meet the middle of the bumper before its corners.
        hl = self.cfg["collision"]["chassis_half_length"]
        self.assertLess(np.linalg.norm(env.world.body - [hl, 0], axis=1).min(), 0.025)
        self.assertGreater(env.world.clearance_error, 0)

    def test_single_substep_and_collision_on_stop(self):
        cfg = deepcopy(self.cfg)
        cfg["env"]["substeps"] = 1
        env = FormulaThreeEnv(cfg, self.track, 1, deterministic=True)
        obs = env.reset()
        self.assertTrue(np.isfinite(obs).all())
        env.stopping[:] = True
        env.world.body_clearance = lambda x, y, yaw: np.zeros(len(x))
        _, _, term, _, info = env.step(np.zeros((1, 2)))
        self.assertTrue(term[0])
        self.assertTrue(info[0]["crashed"])
        self.assertFalse(info[0]["clean_finish"])
        self.assertFalse(info[0]["stopped"])

    def test_depth_invalid_is_not_open_road(self):
        camera = Camera(self.cfg)
        image = np.full((360, 640), np.nan)
        sampled = camera.sample_depth(image, 220, 220, 320, 180)
        encoded = camera.encode(camera.depth_to_scan(sampled))
        np.testing.assert_array_equal(encoded, np.full_like(encoded, INVALID))
        ranges = np.full((1, camera.width), 2.0)
        rendered = camera.render(ranges, np.zeros(1), np.full(1, camera.z))
        np.testing.assert_allclose(camera.depth_to_scan(rendered), ranges, atol=1e-6)

    def test_scan_reorigin_to_trained_camera(self):
        # The car's lens sits 0.06 m left of the camera the policy trained
        # with.  In a 0.92 m corridor, read raw, the walls come out 13% off
        # and lopsided; moved onto the trained camera they must match it.
        camera = Camera(self.cfg)

        def walls(dy):
            s, c = np.sin(camera.azimuth), np.cos(camera.azimuth)
            with np.errstate(divide="ignore"):
                side = np.where(
                    s > 0, (0.46 - dy) / s, np.where(s < 0, (-0.46 - dy) / s, np.inf)
                )
            return np.minimum(side, 6.0 / c)

        depth = camera.render(walls(0.06)[None], np.zeros(1), np.full(1, camera.z))
        truth = walls(0.0)
        fixed = camera.depth_to_scan(depth, origin=(0.0, 0.06, 0.0))[0]
        # Edge columns the shifted lens cannot see hold the outermost wall
        # range rather than going invalid (a vanishing wall stalled the car).
        self.assertTrue(np.isfinite(fixed).all())
        self.assertLess(np.median(np.abs(fixed - truth) / truth), 0.02)
        # ...but open track the lens can see stays open.
        s = np.sin(camera.azimuth)
        with np.errstate(divide="ignore"):
            left_only = np.where(s > 0, 0.40 / s, np.inf)
        open_right = camera.render(left_only[None], np.zeros(1), np.full(1, camera.z))
        scan = camera.depth_to_scan(open_right, origin=(0.0, 0.06, 0.0))[0]
        self.assertTrue(np.isinf(scan[camera.azimuth < -0.05]).all())
        raw = camera.depth_to_scan(depth)[0]
        self.assertGreater(np.median(np.abs(raw - truth) / truth), 0.1)
        np.testing.assert_array_equal(
            camera.depth_to_scan(depth, origin=(0, 0, 0)), camera.depth_to_scan(depth)
        )

    def test_config_pairing(self):
        policy = NumpyPolicy(
            [],
            [],
            meta={
                "meta_schema": "formulaThree-v1",
                "meta_config_sha256": config_digest(self.cfg),
            },
        )
        policy.check_config(self.cfg)
        changed = deepcopy(self.cfg)
        changed["env"]["steer_residual"] *= 0.5
        with self.assertRaises(ValueError):
            policy.check_config(changed)

    def test_reliability_precedes_speed(self):
        good = dict(finish=1.0, lap=36.0, distance=330.0, clearance_p10=0.1)
        fast = {**good, "finish": 0.99, "lap": 28.0}
        self.assertGreater(selection_key(good, good), selection_key(good, fast))
        faster = {**good, "lap": 33.0}
        self.assertGreater(selection_key(good, faster), selection_key(good, good))

    def test_strict_sim_verdict(self):
        log = "FINISHED 3 laps in 99.0 s\nSTOPPED"
        m = dict(samples=100, min_clearance=0.1, speed_samples=100, max_overspeed=0.0)
        m.update(
            last_pose_wall_time=time.time(),
            last_speed_wall_time=time.time(),
            last_speed=0.0,
        )
        limits = self.cfg["validation"]
        self.assertEqual(verdict(m, log, 3, limits), [])
        for bad in (
            {},
            {**m, "min_clearance": 0.01},
            {**m, "max_overspeed": 0.4},
            {**m, "pose_jumps": 1},
            {**m, "depth_lost": True},
        ):
            self.assertTrue(verdict(bad, log, 3, limits))
        self.assertTrue(verdict(m, log + " TIMED OUT, still moving", 3, limits))
        self.assertTrue(verdict({**m, "last_speed": 1.0}, log, 3, limits))
        self.assertTrue(verdict({**m, "last_pose_wall_time": 0.0}, log, 3, limits))


if __name__ == "__main__":
    unittest.main()
