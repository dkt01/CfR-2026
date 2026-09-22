#!/usr/bin/env python3
"""What driving well is worth, term by term.

Kept separate from the env so that `reward_probe.py` can price a handful of
real driving states against every term at once, and so that retuning is a
diff to one file rather than an archaeology exercise across a step function.

The shape of the problem, which is what the weights are answering:

  * Progress is the objective.  Everything else is a constraint expressed as
    a price, and each price is set against what breaking that constraint is
    WORTH in progress -- not picked to look severe.
  * The two speed limits are already structural: `scale_action` cannot ask
    for more than the cap at the current station.  The overspeed term is
    therefore not about the command, it is about ARRIVING too fast, which on
    a car with no brakes is decided ten metres earlier.
  * Grazing is priced, contact is terminal.  "Should not graze" is a band of
    clearance that costs to be in, not a second crash test.
  * Cross-track error and steering smoothness are REQUIREMENTS, so they are
    priced against progress rather than under it -- see `lateral` and the two
    steering terms in config.yaml for what each one costs at a real driving
    state.
  * The run does not end at the finish line, it ends at rest.  During the
    stopping phase progress and time both stop paying, so there is nothing to
    be gained by carrying speed over the line beyond what the race itself
    already pays for; what remains live is everything about staying inside
    the corridor, because the car is still in it.
"""

from __future__ import annotations

import numpy as np


class Reward:
    def __init__(self, config: dict):
        r = config["reward"]
        self.w_progress = float(r["progress"])
        self.w_time = float(r["time"])
        self.w_over_lin = float(r["overspeed_linear"])
        self.w_over_quad = float(r["overspeed_quad"])
        self.w_graze = float(r["graze"])
        self.graze_margin = float(r["graze_margin"])
        self.w_steer_rate = float(r["steer_rate"])
        self.w_steer_jerk = float(r["steer_jerk"])
        self.w_lateral = float(r["lateral"])
        self.lateral_scale = float(r["lateral_scale"])
        self.lap_bonus = float(r["lap_bonus"])
        self.finish_bonus = float(r["finish_bonus"])
        self.stop_bonus = float(r["stop_bonus"])
        self.w_lap_improve = float(r["lap_improve"])
        self.lap_improve_cap = float(r["lap_improve_cap"])
        self.crash = float(r["crash"])
        self.stall = float(r["stall"])

    def step(self, dt, advance, speed, v_cap, clearance, lateral, steer_step,
             steer_jerk, lapped, finished, crashed, stalled, stopping,
             stopped, lap_gain):
        """Per-step reward for a batch, plus the terms that made it up.

        `stopping` is 1.0 for the cars that have finished their two laps and
        are coasting to rest.  It gates progress and time off rather than
        ending the episode, because on a car with no brakes the coast down is
        fifteen metres of corridor the policy still has to steer.
        """
        racing = 1.0 - stopping
        over = np.maximum(speed - v_cap, 0.0)
        # Clearance is measured from the car's body, so it is already the
        # distance that matters; below zero the car is inside a bale and the
        # episode is over anyway.
        bite = np.clip((self.graze_margin - clearance) / self.graze_margin, 0.0, 1.0)

        terms = {
            "progress": self.w_progress * advance * racing,
            "time": -self.w_time * dt * racing,
            "overspeed": -(self.w_over_lin * over + self.w_over_quad * over**2) * dt,
            "graze": -self.w_graze * bite**2 * dt,
            # Hold the centerline.  Quadratic, so it is nearly free to be a
            # couple of centimetres out and expensive to be half a corridor
            # out -- a shape that asks for zero cross-track error without
            # asking for a twitch every time the estimate moves 5 mm.
            "lateral": -self.w_lateral * (lateral / self.lateral_scale) ** 2 * dt,
            # Divided by dt, not multiplied: these are differences of a
            # command sampled every dt, so squaring them and dividing gives a
            # rate that does not change meaning if control_hz does.
            "steer_rate": -self.w_steer_rate * steer_step**2 / dt,
            "steer_jerk": -self.w_steer_jerk * steer_jerk**2 / dt,
            "lap": self.lap_bonus * lapped,
            # BEATING THE PREVIOUS LAP, PAID AS IT IS EARNED.
            #
            # `lap_gain` arrives already differenced by env.py: it is the
            # step-to-step change in w*clip(pace), where pace is how many
            # seconds up on the previous lap's pace the car is right now.
            # Over a whole lap those increments telescope to exactly
            # w*(previous lap - this lap) -- the same total as paying one
            # lump at the line -- but spread over the 660 steps that earned
            # it.
            #
            # That matters because of gamma.  At 0.995 and 20 Hz the
            # discount horizon is ~200 steps, or 10 s of a 33 s lap, so a
            # lump sum at the finish line is invisible from the first two
            # thirds of the lap it is meant to reward: the policy would be
            # paid for something it could not see itself causing.  Dense is
            # the same objective with the credit attached to the actions
            # that produced it.
            "lap_improve": lap_gain,
            "finish": self.finish_bonus * finished,
            "stop": self.stop_bonus * stopped,
            "crash": self.crash * crashed,
            "stall": self.stall * stalled,
        }
        total = sum(terms.values())
        return total, terms
