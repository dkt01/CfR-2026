"""Centerline steering and a coast-feasible speed reference for formulaThree."""

from __future__ import annotations

import numpy as np


def feasible_profile(track, config):
    """Fastest speed at each station that coast drag can still slow in time.

    Backward pass under m dv/dt = -(f0 + f1 v), then a forward pass under the
    bridge's 2.0 m/s^2 target slew.  Iterated because the course is a loop and
    the two passes constrain each other across the start line.

    It is the BRAKING ideal and nothing more, which is what makes it a useful
    reference: it is the lap time the drag curve allows, so the gap between it
    and a driven lap is the whole of what driving costs.  It does NOT know
    about `plant.yaw_response_tau` -- a car that cannot start turning for
    0.34 s cannot actually hold this profile through the chicanes, and
    `BaselineDriver` backs off by `baseline_speed_scale` to account for that.
    Folding the lag in here instead was tried and is the wrong place for it:
    it changes the meaning of the word "ideal" in every report that prints it.
    """
    p = config["plant"]
    mass = float(p["mass"])
    f0, f1 = float(p["coast_f0"]) / mass, float(p["coast_f1"]) / mass
    accel = min(float(p["speed_slew_rate"]), float(p["max_accel"]))
    ds = track.ds
    v = track.v_cap.copy()
    n = len(v)
    for _ in range(8):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            decel = f0 + f1 * v[j]
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * decel * ds))
        for i in range(n):
            j = (i - 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * accel * ds))
    return v


def residual_action(absolute, steer_prior, residual):
    """Absolute steering command -> the residual the env expects.

    `env.py` scales the network's steering as
    `clip(steer_prior + residual * a0)`, so anything that thinks in absolute
    commands -- a bench test, a replayed log -- has to be converted or it gets
    added to the prior twice.
    """
    out = np.array(absolute, dtype=float, copy=True)
    out[:, 0] = np.clip((out[:, 0] - steer_prior) / residual, -1.0, 1.0)
    return out


class BaselineDriver:
    """Centerline prior plus a reference profile, backed off for chassis lag."""

    def __init__(self, track, config):
        self.track = track
        self.cfg = config
        self.v_ref = feasible_profile(track, config) * float(
            config["env"]["baseline_speed_scale"]
        )
        self.dead_time = float(config["plant"]["command_dead_time"])

    def reset(self, n=1):
        pass

    def act(self, station, speed, v_cap, v_floor=None):
        """Invert the same reference-speed mapping used by the learned actor."""
        ahead = (station + speed * self.dead_time) % self.track.length
        want = np.clip(self.track.at(ahead, self.v_ref), 0, v_cap)
        ref = 0.5 * v_cap if v_floor is None else np.clip(v_floor, 0, v_cap)
        throttle = np.where(
            want < ref,
            want / np.maximum(ref, 1e-6) - 1,
            (want - ref) / np.maximum(v_cap - ref, 1e-6),
        )
        return np.stack([np.zeros_like(throttle), np.clip(throttle, -1, 1)], axis=1)
