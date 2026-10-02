"""camera_extrinsics.py and calibrate_camera.py against scenes with a KNOWN mount.

A depth image is rendered from a chosen camera pose (flat floor, two boxes),
then calibrated; the mount has to come back.  The lever-arm fit gets a
kinematic-bicycle drive with a known pose point.  Plain unittest, no ROS.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts",
    ),
)

import calibrate_camera as cc  # noqa: E402
import camera_extrinsics as ce  # noqa: E402

W, H = 640, 360
F = (W / 2) / math.tan(math.radians(55))
CX, CY = W / 2 - 0.5, H / 2 - 0.5
LENS = (-0.01, 0.06, 0.015)


def render(mount, boxes, noise=0.004, seed=0):
    """Optical-axis depth seen from `mount`: flat floor plus axis-aligned
    boxes (x0, x1, y0, y1, height) in the vehicle frame."""
    rot = ce.rotation(mount.roll, mount.pitch, mount.yaw)
    origin = mount.depth_origin()
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([np.ones((H, W)), (CX - u) / F, (CY - v) / F], -1) @ rot.T
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(rays[..., 2] < 0, -origin[2] / rays[..., 2], np.inf)
        for x0, x1, y0, y1, top in boxes:
            for face_x in (x0,):
                z = (face_x - origin[0]) / rays[..., 0]
                y = origin[1] + z * rays[..., 1]
                h = origin[2] + z * rays[..., 2]
                hit = (z > 0) & (y > y0) & (y < y1) & (h > 0) & (h < top)
                depth = np.where(hit & (z < depth), z, depth)
    depth = np.where(np.isfinite(depth), depth, np.nan)
    return depth + np.random.default_rng(seed).normal(0, noise, depth.shape)


def capture(mount, boxes, frames=4):
    g = ce.up_from_attitude(mount.roll, mount.pitch) * 9.81
    return {
        "depth": np.stack([render(mount, boxes, seed=i) for i in range(frames)]),
        "k": np.array([F, F, CX, CY]),
        "imu": np.tile(g, (50, 1)),
        "gyro": np.zeros((50, 3)),
        "quat": np.zeros((0, 4)),
        "pose": np.zeros((0, 3)),
        "lens": np.array(LENS),
        "lens_source": np.array("tf"),
    }


TRUE = ce.Mount(
    0.29,
    0.012,
    0.205,
    math.radians(-0.8),
    math.radians(3.5),
    math.radians(1.2),
    lens=LENS,
)
PRIOR = ce.Mount(0.315, 0.0, 0.185, 0.0, 0.0, 0.0, lens=(0.0, 0.06, 0.015))
BOXES = [(2.0, 2.4, -0.25, 0.25, 0.4), (3.2, 3.6, 0.45, 0.95, 0.4)]


class TestFloor(unittest.TestCase):
    def test_floor_alone_gives_height_and_attitude(self):
        res = cc.analyze(capture(TRUE, []), PRIOR, [], None)
        got = res["measured"]
        self.assertAlmostEqual(
            got["depth_origin"][2], TRUE.depth_origin()[2], delta=0.002
        )
        self.assertAlmostEqual(
            got["mount"]["pitch"], TRUE.pitch, delta=math.radians(0.05)
        )
        self.assertAlmostEqual(
            got["mount"]["roll"], TRUE.roll, delta=math.radians(0.05)
        )
        self.assertNotIn("camera_mount.x", res["values"])
        self.assertNotIn("camera_mount.yaw", res["values"])
        # Lens offset from TF lands in the patch.
        self.assertAlmostEqual(res["values"]["camera_mount.lens_offset_x"], -0.01)

    def test_imu_agrees_on_level_floor(self):
        res = cc.analyze(capture(TRUE, []), PRIOR, [], None)
        self.assertAlmostEqual(res["imu"]["accel_pitch"], TRUE.pitch, delta=1e-6)
        self.assertFalse([n for n in res["notes"] if "disagree" in n])


class TestTargets(unittest.TestCase):
    def test_two_targets_give_position_and_yaw(self):
        truth = [(2.0, 0.0), (3.2, 0.70)]
        res = cc.analyze(capture(TRUE, BOXES), PRIOR, truth, None)
        v = res["values"]
        self.assertAlmostEqual(v["camera_mount.x"], TRUE.x, delta=0.01)
        self.assertAlmostEqual(v["camera_mount.y"], TRUE.y, delta=0.01)
        self.assertAlmostEqual(v["camera_mount.z"], TRUE.z, delta=0.003)
        self.assertAlmostEqual(v["camera_mount.yaw"], TRUE.yaw, delta=math.radians(0.3))

    def test_report_flags_the_changes(self):
        text = cc.report(cc.analyze(capture(TRUE, []), PRIOR, [], None))
        self.assertIn("**update**", text)
        self.assertIn("obstacle band from", text)


class TestLeverArm(unittest.TestCase):
    @staticmethod
    def drive(ahead, yaw_offset, v=1.5, dt=0.002, seconds=60):
        wheelbase, th, x, y = 0.324, 0.0, 0.0, 0.0
        rows = []
        for t in np.arange(0, seconds, dt):
            w = v * math.tan(0.3 * math.sin(2 * math.pi * t / 12)) / wheelbase
            x += v * math.cos(th + w * dt / 2) * dt
            y += v * math.sin(th + w * dt / 2) * dt
            th += w * dt
            rows.append(
                (t, x + ahead * math.cos(th), y + ahead * math.sin(th), th + yaw_offset)
            )
        return np.array(rows[::10]).T

    def test_recovers_distance_ahead_of_rear_axle_and_yaw(self):
        fit = ce.fit_lever_arm(*self.drive(0.47, math.radians(2.0)))
        self.assertAlmostEqual(fit["ahead_of_rear_axle_m"], 0.47, delta=0.01)
        self.assertAlmostEqual(fit["yaw_deg"], 2.0, delta=0.1)

    def test_wheelbase_midpoint_reads_half_wheelbase(self):
        # Gazebo's pose is the midpoint: the fit must say 0.162.
        fit = ce.fit_lever_arm(*self.drive(0.162, 0.0))
        self.assertAlmostEqual(fit["ahead_of_rear_axle_m"], 0.162, delta=0.005)

    def test_parked_track_gives_nothing(self):
        t = np.arange(0, 10, 0.02)
        self.assertIsNone(ce.fit_lever_arm(t, 0 * t, 0 * t, 0 * t))


class TestVehicleYaml(unittest.TestCase):
    def test_section_loads(self):
        mount = ce.load_mount()
        self.assertIsNotNone(mount)
        self.assertEqual(len(mount.depth_origin()), 3)


if __name__ == "__main__":
    unittest.main()
