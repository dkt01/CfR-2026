#!/usr/bin/env python3
"""The policy's input vector, built in exactly one place.

Both sides of the sim-to-real boundary import this file: `env.py` calls it for
256 cars at once during training, `formula_one_node.py` calls it for one car at
20 Hz on the Orin.  Any observation that is assembled twice eventually gets
assembled two different ways -- a sign flip, a different normaliser, a stale
lookahead table -- and the policy that results drives beautifully in the
simulator and into the first bale on the course.  So there is one builder, it
is batched, and B == 1 is not a special case.

Everything in here is derivable on the real car from:

    /zed/zed_node/pose         map-frame x, y, yaw  (loop-closure corrected)
    /arduino_bridge/status     ground speed from the spur tachometer
    the track file, which is the same bytes in both places

There is deliberately no camera or point-cloud channel.  The bales do not
move, the corridor is 0.92 m wide, and a known map read against a corrected
pose is a far shorter path to the real car than a depth cloud whose noise,
field of view and latency would all have to be modelled convincingly.  The
cost of that choice is that everything rides on localisation, which is why
`randomize` in config.yaml spends most of its budget on pose error.
"""

from __future__ import annotations

import os

import numpy as np

# Kept next to the builder so the node cannot drift from the trainer.
OBS_DIM = 35

# Normalisers.  Fixed constants rather than running statistics: a policy that
# depends on a normaliser fitted during training needs that normaliser shipped
# with it and applied identically, which is one more thing to get wrong.
_LATERAL_SCALE = 0.5     # m -- half the corridor, near enough
_ROOM_SCALE = 0.6        # m -- distance to a wall along the track normal
_CLEAR_SCALE = 0.3       # m -- body clearance to the nearest bale
_KAPPA_SCALE = 2.0       # 1/m -- |k| peaks at 0.68 on this course
_PACE_SCALE = 5.0        # s -- how far ahead or behind the last lap's pace
_RATE_SCALE = 2.0        # rad/s -- how far the chassis is behind its wheels


class ObservationBuilder:
    """Turns (pose, speed, last action) into the network's input.

    Holds the station hint, so projection stays local.  The course runs two
    straights 1.4 m apart in opposite directions; a global nearest-point
    search lands on the wrong one about as often as not.
    """

    def __init__(self, track, config: dict, n: int = 1):
        self.track = track
        self.n = n
        env = config["env"]
        self.lookahead = np.asarray(env["lookahead_m"], dtype=float)
        self.v_ref = float(config["track"]["v_straight"])
        veh = config["vehicle"]
        self.half_length = float(veh["length"]) / 2
        self.half_width = float(veh["width"]) / 2
        # The corridor is ~0.92 m, so anything past 1.5 m is the normal
        # escaping through the mouth of a hairpin rather than real room.
        self.half_left = np.minimum(track.half_left, 1.5)
        self.half_right = np.minimum(track.half_right, 1.5)
        self.hint = np.zeros(n, dtype=np.int64)
        p = config["plant"]
        self.wheelbase = float(p["wheelbase"])
        self.understeer = float(p["understeer_gradient"])
        # Nominal tire_scrub, not a per-car draw: this prior is a FIXED
        # geometric calculation the policy trims, not itself something that
        # gets randomised per episode (that happens in plant.py, to the
        # engine the prior is steering).  Baking in the measured 1.10 keeps
        # the prior calibrated to the car it will typically face instead of
        # asking the residual to correct a further 10% every single tick.
        self.tire_scrub = float(p["tire_scrub"])
        # The chassis lag plant.py now carries.  The prior has to know it for
        # two separate reasons -- it predicts forward through it, and it
        # inverts it in the feedforward -- and both are below.
        self.yaw_tau = float(p["yaw_response_tau"])
        self.steer_pts = np.asarray(p["steering_command_points"], dtype=float)
        self.steer_ang = np.asarray(p["steering_angle_points"], dtype=float)
        # d(kappa)/ds, periodic and smoothed.  The lead term below needs the
        # RATE at which the required steering angle is changing, and a central
        # difference on a 0.05 m grid amplifies whatever noise survived the
        # curvature smoothing, so it gets the same treatment curvature did.
        k = track.kappa
        dk = (np.roll(k, -1) - np.roll(k, 1)) / (2.0 * track.ds)
        win = max(int(round(float(config["track"]["curvature_smooth_m"]) / track.ds)), 1)
        kern = np.ones(win) / win
        pad = np.concatenate([dk[-win:], dk, dk[:win]])
        self.dkappa = np.convolve(pad, kern, mode="same")[win:win + len(dk)]
        # How far ahead the prior evaluates itself.  Three delays stack up
        # between deciding on a command and the car actually being somewhere
        # else: command_dead_time (0.19 s) before the wheels move,
        # steering_tau (0.05 s) for the servo, and yaw_response_tau (0.34 s)
        # for the chassis to take up the yaw rate those wheels imply.  It is
        # a tuned constant in config.yaml rather than their sum, because the
        # lead term below already cancels part of the third one and summing
        # them double-counts it.  CFR_FF_HORIZON overrides it for a sweep.
        self.ff_horizon = float(os.environ.get(
            "CFR_FF_HORIZON", config["env"]["steer_prior"]["horizon_s"]))
        prior = config["env"]["steer_prior"]
        self.k_lateral = float(prior["k_lateral"])
        self.k_heading = float(prior["k_heading"])
        self.speed_floor = float(prior["speed_floor"])
        # How much of the lag inversion to actually apply.  1.0 is the exact
        # algebraic inverse; it is not automatically the best number, because
        # evaluating the feedforward `horizon_s` ahead ALREADY turns the
        # wheels in early, and the two together ask for the corner twice.
        self.lead_scale = float(prior["lead_scale"])
        self.k_rate = float(prior["k_rate"])
        if 2 * len(self.lookahead) + 15 != OBS_DIM:
            raise ValueError(
                f"lookahead_m has {len(self.lookahead)} entries, which makes a "
                f"{2 * len(self.lookahead) + 15}-wide observation; OBS_DIM is "
                f"{OBS_DIM}. Retrain, or put the count back."
            )

    def set_station(self, mask, station):
        """Seed the hint after a reset or a relocalisation."""
        idx = np.searchsorted(self.track.s, np.asarray(station) % self.track.length)
        self.hint[mask] = np.clip(idx, 0, len(self.track.s) - 1)

    def steer_prior(self, station, x, y, yaw, speed, yaw_rate, last_steer):
        """The absolute steering command that holds the centerline here.

        This is the competent prior the network trims, and it is the reason
        the policy learns anything at all.  Measured on this course:

            zero steering              crashes in 2.8 m
            curvature feedforward only crashes in 3.6 m
            this                       completes two laps

        Feedforward alone is not enough because open-loop curvature tracking
        accumulates error and there is only about +/- 0.26 m of corridor to
        accumulate it in.  What makes the difference is feedback, evaluated
        at where the car WILL BE when the command lands -- 0.19 s of dead
        time plus 0.05 s of servo lag is most of a car length at 5 m/s, and
        feeding back on where the car IS puts that lag inside the loop, which
        oscillates and then diverges as soon as curvature picks up.

        `yaw_rate` is passed in rather than differenced here, and that is
        load-bearing: predicting forward from the angle THIS CALL is about to
        return closes an algebraic loop that saturates the servo lock to lock
        on alternate ticks.  It has to be a measurement.  `last_steer` is a
        different thing -- that command was issued a tick ago and is already
        on its way to the wheels -- so predicting through it is a Smith
        predictor, not a loop.

        THE CHASSIS LAGS, AND BOTH HALVES OF THIS FUNCTION ACCOUNT FOR IT.
        `plant.yaw_response_tau` (0.34 s, fitted to Gazebo step-steer traces)
        is long enough to destabilise a controller that ignores it, and the
        failure is not subtle: extrapolating the heading at a CONSTANT yaw
        rate through a chicane predicts the car still rotating left when its
        wheels have already gone hard right, so the feedback asks for the
        opposite lock again, and the prior saturates for eight ticks together
        and drives into the bales at 21 m.  That is what the scripted baseline
        did in Gazebo all along; it only started doing it HERE once the lag
        was in the model.  So:

          * the heading prediction RELAXES toward the yaw rate the last
            command implies instead of holding the present one -- the lag's
            own solution rather than a tangent to it;
          * the feedforward INVERTS the lag.  To hold curvature k the yaw rate
            must track v*k, and under dw/dt = (w_kin - w)/tau that needs
            w_kin = v*k + tau*d(v*k)/dt, i.e. an angle set by
            k + tau*v*dk/ds rather than by k.  That is the term that turns the
            wheels in BEFORE the corner arrives instead of when it does.

        Steering geometry is known, the corridor is 0.92 m wide, and nobody
        should ship a raw-network steering command into it.  What is NOT
        primed is the throttle -- the speed profile is the whole lap time and
        the whole difficulty (the car cannot brake), and the policy learns
        all of it.
        """
        t = self.track
        h = self.ff_horizon
        eff_wheelbase = (self.wheelbase + self.understeer * speed**2) * self.tire_scrub

        # Where the yaw rate is HEADED, from the command already in flight.
        last_angle = np.interp(np.clip(last_steer, -1.0, 1.0),
                               self.steer_pts, self.steer_ang)
        rate_kin = speed * np.tan(last_angle) / eff_wheelbase
        # Integral of the first-order relaxation from yaw_rate toward it.
        decay = np.exp(-h / max(self.yaw_tau, 1e-6))
        d_yaw = rate_kin * h + (yaw_rate - rate_kin) * self.yaw_tau * (1.0 - decay)
        yaw_p = yaw + d_yaw
        mid = yaw + 0.5 * d_yaw
        x_p = x + speed * np.cos(mid) * h
        y_p = y + speed * np.sin(mid) * h
        station_p = (station + speed * h) % t.length

        idx = np.clip(np.searchsorted(t.s, station_p), 0, len(t.s) - 1)
        ex, ey = x_p - t.x[idx], y_p - t.y[idx]
        lateral = -ex * t.ty[idx] + ey * t.tx[idx]
        ref = np.arctan2(t.ty[idx], t.tx[idx])
        psi = np.arctan2(np.sin(yaw_p - ref), np.cos(yaw_p - ref))

        # Curvature, plus as much of the lag inversion as `lead_scale` asks
        # for.  The inverse itself is exact -- holding curvature k under
        # dw/dt = (w_kin - w)/tau needs an angle set by k + tau*v*dk/ds -- but
        # at 5 m/s into a hairpin mouth that term alone is 1.2 1/m against a
        # curvature of 0.68 and saturates the lock, so it is scaled.
        kappa_lead = (t.at(station_p, t.kappa) + self.lead_scale * self.yaw_tau
                      * speed * t.at(station_p, self.dkappa))
        angle = np.arctan(eff_wheelbase * kappa_lead)
        # Stanley form: firm in a 2.5 m/s hairpin, gentle on a 5.2 m/s
        # straight, from one constant.  The yaw-rate term is damping: with a
        # 0.34 s chassis lag in the loop, position-and-heading feedback alone
        # is a second-order system with almost no damping in it, and it rings.
        rate_error = yaw_rate - speed * t.at(station_p, t.kappa)
        angle = angle - self.k_heading * psi - self.k_rate * rate_error - np.arctan(
            self.k_lateral * lateral / np.maximum(speed, self.speed_floor)
        )
        angle = np.clip(angle, self.steer_ang[0], self.steer_ang[-1])
        # Through the measured table, so the command means the same thing here
        # as it does to the servo.  `rate_kin - yaw_rate` goes out with it:
        # see `compute` for why that one number is worth an observation
        # channel of its own.
        return np.interp(angle, self.steer_ang, self.steer_pts), rate_kin - yaw_rate

    def compute(self, x, y, yaw, speed, yaw_rate, prev_action, last_steer,
                lap_state):
        """(B, OBS_DIM) plus the frame quantities the caller also wants.

        `lap_state` is (B, 3): how far through the current lap the car is
        (0-1), how long it has been on it (s), and the lap time it is trying
        to beat (s, or 0 on the first lap).

        Those three are in the observation because the reward pays for
        BEATING THE PREVIOUS LAP, and a reward for something the policy
        cannot see is not a reward, it is noise -- the same return arrives
        whatever it did, so there is no gradient pointing at the behaviour
        that earned it.  What the network is actually handed is the useful
        combination: seconds up or down on the previous lap's pace, which is
        a quantity it can act on at any point in the lap rather than a
        stopwatch it has to integrate for itself.
        """
        t = self.track
        idx, station, lateral = t.project(x, y, self.hint)
        self.hint = idx
        psi = t.heading_error(yaw, idx)

        v_cap_now = t.v_cap[idx]
        room_left = np.clip(self.half_left[idx] - lateral, -1.5, 1.5)
        room_right = np.clip(self.half_right[idx] + lateral, -1.5, 1.5)
        clear = t.body_clearance(x, y, yaw, self.half_length, self.half_width)

        lap_state = np.asarray(lap_state, dtype=float).reshape(-1, 3)
        lap_progress, lap_elapsed, lap_target = lap_state.T
        has_target = (lap_target > 1e-6).astype(float)
        # Where the previous lap would have been by now, minus where this one
        # is.  Positive means ahead of it.  A uniform-pace reference is crude
        # -- the previous lap was not uniform -- but it is monotone in the
        # right direction everywhere, which is all a gradient needs.
        pace = has_target * (lap_target * lap_progress - lap_elapsed)

        cap_ahead = t.lookahead(station, self.lookahead, t.v_cap)
        kappa_ahead = t.lookahead(station, self.lookahead, t.kappa)
        steer_ff, rate_lag = self.steer_prior(station, x, y, yaw, speed,
                                              yaw_rate, last_steer)

        obs = np.concatenate(
            [
                (speed / self.v_ref)[:, None],
                (lateral / _LATERAL_SCALE)[:, None],
                np.sin(psi)[:, None],
                np.cos(psi)[:, None],
                prev_action[:, 0:1],
                prev_action[:, 1:2],
                ((speed - v_cap_now) / self.v_ref)[:, None],
                (room_left / _ROOM_SCALE)[:, None],
                (room_right / _ROOM_SCALE)[:, None],
                np.clip(clear / _CLEAR_SCALE, -1.0, 3.0)[:, None],
                steer_ff[:, None],
                # HOW FAR BEHIND ITS OWN WHEELS THE CHASSIS IS, right now:
                # the yaw rate the command already in flight implies, minus
                # the yaw rate the car actually has.
                #
                # This is here to make the chassis lag OBSERVABLE.  The
                # policy is a plain MLP with one step of history, and
                # `yaw_tau_scale` redraws the lag every episode over a range
                # wide enough to change how early a hairpin has to be set up
                # by half a metre.  Without this channel the policy cannot
                # tell which car it drew and its only safe answer is to drive
                # every car like the slowest one -- which is exactly how an
                # over-wide randomisation once turned into a policy that took
                # 101 s a run and three rounds of reward tuning looking for a
                # problem that was not in the reward.  With it, the lag is a
                # state variable it can read off a single frame and respond
                # to, because for a first-order lag this difference IS the
                # state.
                np.clip(rate_lag / _RATE_SCALE, -3.0, 3.0)[:, None],
                lap_progress[:, None],
                has_target[:, None],
                np.clip(pace / _PACE_SCALE, -3.0, 3.0)[:, None],
                cap_ahead / self.v_ref,
                np.clip(kappa_ahead * _KAPPA_SCALE, -2.0, 2.0),
            ],
            axis=1,
        )
        frame = dict(
            index=idx, station=station, lateral=lateral, psi=psi,
            v_cap=v_cap_now, clearance=clear, steer_ff=steer_ff,
            room_left=room_left, room_right=room_right, pace=pace,
            rate_lag=rate_lag,
        )
        return np.clip(obs, -10.0, 10.0).astype(np.float32), frame


def scale_action(action, v_cap, steer_ff, residual):
    """Raw network output -> (steering command, speed target in m/s).

    The steering action is a TRIM on the centerline feedforward, not the whole
    command; `residual` is how much authority it has.  The speed action is a
    FRACTION OF THE LOCAL CAP, not a speed.  That makes
    the two rule limits -- 2.5 m/s in the hairpins, 5.2 on the straights --
    structural: there is no action, in or out of distribution, that asks for
    more than the cap at the station the car is standing on.  A reward
    penalty alone would only make exceeding it expensive, and a policy that
    has found something worth more would pay it.

    What the cap cannot do is stop the car ARRIVING at a hairpin too fast,
    because the car has no brakes.  That part is on the policy, and on the
    overspeed term in reward.py that teaches it.
    """
    throttle = np.clip(action[:, 1], -1.0, 1.0)
    speed = v_cap * 0.5 * (throttle + 1.0)
    # RESIDUAL steering: the network trims the centerline feedforward rather
    # than producing the whole command.  A zero action now follows the track,
    # which is what gives PPO something to improve on instead of a wall.
    steer = np.clip(steer_ff + residual * np.clip(action[:, 0], -1.0, 1.0), -1.0, 1.0)
    return steer, speed
