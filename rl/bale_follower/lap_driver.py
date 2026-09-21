"""A scripted driver for the planned loop, with no policy involved.

Shared by `lap_env_selftest.py` (stubbed ROS, kinematic stand-in) and
`lap_live_check.py` (real Gazebo), so the same reference driver is what both
report a lap time for and the two numbers can be compared.

Deliberately simple -- pure pursuit on the planned line with a
curvature-dependent speed -- because its job is to prove the environment
works, not to be fast. It is also the floor any trained policy should beat:
it drives the plan with no lookahead braking and no idea where the bales are.

Imports nothing but math and numpy, so it survives the self-test's stubbing
of ROS and Gymnasium.
"""

from __future__ import annotations

import math

import numpy as np


class PursuitDriver:
    def __init__(self, environment, lookahead: float = 1.2,
                 target_speed: float = 3.2, corner_scale: float = 0.6,
                 use_plan_speed: bool = True, plan_scale: float = 1.0,
                 brake_horizon_m: float = 4.0) -> None:
        self.env = environment
        self.lookahead = lookahead
        self.target_speed = target_speed
        # How much of the target speed a full-lock corner gives up, when the
        # plan's own speed profile is not being used.
        self.corner_scale = corner_scale
        # The plan already carries a friction-circle min-time speed profile,
        # and taking its MINIMUM over the next few metres is what turns pure
        # pursuit into something that can finish a lap: without it the car
        # arrives at the first hairpin at straight-line speed and understeers
        # into the bales (measured -- 22.5 m, then a collision).
        self.use_plan_speed = use_plan_speed
        self.plan_scale = plan_scale
        self.brake_horizon_m = brake_horizon_m

    def action(self, x: float, y: float, yaw: float) -> np.ndarray:
        from env import MAX_STEERING_ANGLE, WHEELBASE

        environment = self.env
        track = environment.track
        index = track.nearest_index(x, y, environment._path_index)
        ahead = (index + int(self.lookahead / track.spacing)) % len(track.x)
        bearing = math.atan2(track.y[ahead] - y, track.x[ahead] - x) - yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        wanted = math.atan2(2.0 * WHEELBASE * math.sin(bearing), self.lookahead)
        wanted_fraction = float(np.clip(wanted / MAX_STEERING_ANGLE, -1.0, 1.0))

        # The action is a steering RATE, so ask for the rate that closes the
        # gap to the angle wanted, saturating at the servo's limit.
        per_step = environment.steer_rate_fraction_per_s / environment.control_hz
        rate = float(np.clip(
            (wanted_fraction - environment._cmd_steer_fraction) / per_step,
            -1.0, 1.0))
        if self.use_plan_speed:
            horizon = max(1, int(self.brake_horizon_m / track.spacing))
            window = (index + np.arange(horizon)) % len(track.x)
            speed = float(track.reference_speed[window].min()) * self.plan_scale
            speed = min(speed, self.target_speed)
        else:
            speed = self.target_speed * (
                1.0 - self.corner_scale * abs(wanted_fraction))
        speed = min(speed, environment.max_speed)
        return np.array([
            2.0 * (speed + environment.reverse_speed)
            / (environment.max_speed + environment.reverse_speed) - 1.0,
            rate,
        ], dtype=np.float32)
