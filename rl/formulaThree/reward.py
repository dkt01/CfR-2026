"""Rate-scaled centerline racing reward.

A clearance violation charges `graze` continuously and blocks success
bonuses; whether it ends the episode is the environment's call (env.py gives
a mid-race touch a bounded recovery window before it counts as `crashed`).
"""

from __future__ import annotations

import numpy as np


class Reward:
    def __init__(self, config: dict):
        r = config["reward"]
        self.w_progress = float(r["progress"])
        self.sigma_y = float(r["sigma_y"])
        self.sigma_psi = float(r["sigma_psi"])
        self.w_track = float(r["track"])
        self.w_time = float(r["time"])
        self.w_over_lin = float(r["overspeed_linear"])
        self.w_over_quad = float(r["overspeed_quad"])
        self.w_graze = float(r["graze"])
        self.graze_margin = float(r["graze_margin"])
        self.w_steer_rate = float(r["steer_rate"])
        self.w_steer_jerk = float(r["steer_jerk"])
        self.lap_bonus = float(r["lap_bonus"])
        self.finish_bonus = float(r["finish_bonus"])
        self.stop_bonus = float(r["stop_bonus"])
        self.crash = float(r["crash"])
        self.stall = float(r["stall"])

    def gate(self, lateral, psi):
        """1 on the centerline, pointing along it; falls off smoothly."""
        return np.exp(
            -0.5 * (lateral / self.sigma_y) ** 2 - 0.5 * (psi / self.sigma_psi) ** 2
        )

    def step(
        self,
        dt,
        advance,
        speed,
        v_cap,
        clearance,
        lateral,
        psi,
        steer_step,
        steer_jerk,
        lapped,
        finished,
        crashed,
        stalled,
        stopping,
        stopped,
        lap_gain,
        touching=None,
    ):
        racing = 1.0 - stopping
        gate = self.gate(lateral, psi)
        over = np.maximum(speed - v_cap, 0.0)
        bite = np.clip((self.graze_margin - clearance) / self.graze_margin, 0.0, 1.0)
        terms = {
            # Backwards progress is charged at full price, not gated: a car
            # reversing along the line must not earn a discount for it.
            "progress": self.w_progress
            * np.where(advance > 0, advance * gate, advance)
            * racing,
            "track": -self.w_track * (1.0 - gate) * dt,
            "time": -self.w_time * dt * racing,
            "overspeed": -(self.w_over_lin * over + self.w_over_quad * over**2) * dt,
            "graze": -self.w_graze * bite**2 * dt,
            "steer_rate": -self.w_steer_rate * steer_step**2 / dt,
            "steer_jerk": -self.w_steer_jerk * steer_jerk**2 / dt,
            "lap": self.lap_bonus * lapped,
            "lap_improve": lap_gain,
            "finish": self.finish_bonus * finished,
            "stop": self.stop_bonus * stopped,
            "crash": self.crash * crashed,
            "stall": self.stall * stalled,
        }
        # A boundary violation on the finish/stop tick cannot collect success
        # bonuses. The environment also makes clean_finish mutually exclusive.
        # `touching` extends this to a tick spent touching a bale that has
        # not (yet, or ever) become a terminal crash -- a recovery window is
        # not grounds to bank a lap crossed mid-graze. Callers that do not
        # pass it (the reward probe, direct-call selftests) keep the old,
        # crash-only gate exactly.
        blocked = crashed if touching is None else np.maximum(crashed, touching)
        for key in ("lap", "finish", "stop", "lap_improve"):
            terms[key] = np.where(blocked > 0, 0.0, terms[key])
        return sum(terms.values()), terms
