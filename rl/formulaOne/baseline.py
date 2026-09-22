#!/usr/bin/env python3
"""The scripted driver: the centerline prior, plus a coast-feasible speed profile.

This is the floor a learned policy has to beat, and it is deliberately made of
exactly the two pieces the policy is given for free and the one piece it is
not:

    steering   the centerline prior from `observation.steer_prior` -- the SAME
               code the policy trims, so the floor and the prior cannot drift
               apart.  In action terms the baseline's steering residual is
               zero.
    speed      the coast-feasible profile below, which is the interesting
               half and the only thing the policy has to discover on its own.

The speed profile is where the course's real difficulty lives.  Because the
car has no brakes, the fastest speed at a station is not the cap there -- it
is the largest speed from which COASTING still meets every cap ahead.  That is
a backward pass through the coast-drag curve, and it is what puts the lift-off
point roughly nine metres before each hairpin.

The baseline runs that profile open-loop off a map.  It scores 100% on the
nominal car and about 13% once the dead time, drag, steering authority and
localisation are redrawn, because a fixed profile cannot adapt to a car that
is not the one it was computed for.  Closing that gap is the policy's job.
"""

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
    """The floor: the prior's steering, and the coast profile backed off.

    `baseline_speed_scale` is not a fudge, it is the price of the chassis
    lag, and it was measured rather than chosen.  Driving the raw
    coast-feasible profile under `yaw_response_tau` leaves this driver
    scraping the first chicane at 0.056 m of clearance -- inside the graze
    band, and close enough to the edge that changing any other number flips
    it into the bales.  Stepping the whole profile down buys clearance
    smoothly and predictably:

        scale   two laps   min clearance
        1.00     57.1 s       0.056 m     <- inside the graze band
        0.90     63.1 s       0.116 m
        0.85     66.7 s       0.131 m     <- here
        0.80     70.7 s       0.179 m

    The floor's job is to be a stable reference and a bootstrap, not to set a
    lap record.  The POLICY is the thing that gets to decide, corner by
    corner and car by car, how much of that margin it can afford to give
    back -- which is the entire difference between a fixed profile and a
    learned one, and why a trained policy can be faster than this and still
    cleaner.
    """

    def __init__(self, track, config):
        self.track = track
        self.cfg = config
        self.v_ref = feasible_profile(track, config) * float(
            config["env"]["baseline_speed_scale"])
        self.dead_time = float(config["plant"]["command_dead_time"])

    def reset(self, n=1):
        pass

    def act(self, station, speed, v_cap):
        """(B, 2): zero steering residual, and the profile's throttle."""
        t = self.track
        # Ask for the speed that will be right where the command LANDS, not
        # where the car is now: 0.19 s of dead time is a metre at racing speed.
        ahead = (station + speed * self.dead_time) % t.length
        want = t.at(ahead, self.v_ref)
        throttle = 2.0 * np.clip(want / np.maximum(v_cap, 1e-6), 0.0, 1.0) - 1.0
        return np.stack([np.zeros_like(throttle), throttle], axis=1)
