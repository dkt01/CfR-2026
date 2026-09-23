"""Checks the physical follower's scan conventions and steering response."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from left_wall_cloud import cloud_command, ranges_from_cloud  # noqa: E402


def wall(y):
    x = np.linspace(0.18, 3.0, 1000)
    left = np.full_like(x, y)
    return np.arctan2(left, x), np.hypot(x, left)


def test_steers_toward_a_distant_left_wall_and_away_from_a_close_one():
    close = cloud_command(*wall(0.25))[1]
    far = cloud_command(*wall(0.75))[1]
    assert close < 0 < far


def test_empty_cloud_commands_stop():
    assert cloud_command(np.array([]), np.array([])) == (0.0, 0.0)


def test_registered_cloud_uses_body_axes_and_rejects_floor():
    points = np.array(
        [(1.0, 0.5, 0.0), (1.0, -0.5, 0.0), (1.0, 0.5, -0.4)], dtype=np.float32
    )
    fields = [
        SimpleNamespace(name=name, offset=offset)
        for name, offset in (("x", 0), ("y", 4), ("z", 8))
    ]
    message = SimpleNamespace(
        fields=fields, data=points.tobytes(), point_step=12, is_bigendian=False
    )
    bearings, ranges = ranges_from_cloud(message)
    assert len(bearings) == 2
    assert math.isclose(bearings[0], math.atan2(0.5, 1.0), abs_tol=1e-6)
    assert bearings[1] < 0
    assert np.allclose(ranges, math.hypot(1.0, 0.5))
