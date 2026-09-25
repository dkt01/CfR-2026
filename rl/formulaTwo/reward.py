"""The expert reward: progress along the TRUE centerline, gated by how well the
car sits on it.

The simulator knows the exact centerline of the course as built (the SDF's
corridor, pushed through the episode's layout warp -- see world.py), so the
reward does not have to guess what good driving is.  It is

    progress * metres_advanced * gate,
    gate = exp(-e_y^2 / 2 sy^2) * exp(-e_psi^2 / 2 spsi^2),

with e_y and e_psi from the TRUE pose against the TRUE centerline.  The
policy never sees either: it drives on a noisy map pose and a depth scan.
That asymmetry is the point -- the reward can be exact because it only has to
exist in simulation.

Everything else is formulaOne's, with the reasons that set each weight kept
there (formulaOne/config.yaml): time, overspeed, graze, crash, stall, the two
steering smoothness terms, the lap/finish/stop bonuses and the dense
lap-improvement shaping.  formulaOne's additive `lateral` term is gone; the
gate replaces it, and `track` keeps the same gate priced while the car is in
the stopping phase, where progress no longer pays.
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
        return sum(terms.values()), terms
