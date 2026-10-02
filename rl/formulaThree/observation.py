"""Shared training/deployment pose features and residual action mapping."""

from __future__ import annotations

import numpy as np

from perception import Camera

# Default map-feature width; use the builder for custom lookahead settings.
OBS_DIM = 36

# Normalizers.  Fixed constants rather than running statistics: a policy that
# depends on a normalizer fitted during training needs that normalizer shipped
# with it and applied identically, which is one more thing to get wrong.
_LATERAL_SCALE = 0.5  # m -- half the corridor, near enough
_ROOM_SCALE = 0.6  # m -- distance to a wall along the track normal
_CLEAR_SCALE = 0.3  # m -- body clearance to the nearest bale
_KAPPA_SCALE = 2.0  # 1/m -- |k| peaks at 0.68 on this course
_PACE_SCALE = 5.0  # s -- how far ahead or behind the last lap's pace
_RATE_SCALE = 2.0  # rad/s -- how far the chassis is behind its wheels
_ACCEL_SCALE = 2.0  # m/s^2 -- how hard the car is actually slowing


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
        self.v_floor = speed_floor(track, config)
        self.hint = np.zeros(n, dtype=np.int64)
        # Absent means FALSE, deliberately: every config.yaml snapshot saved
        # into a run directory before this channel existed lacks the key, and
        # those policies are 35 wide.  Defaulting to True would silently
        # widen them and refuse to load the best artifact this project has.
        self.observe_accel = bool(config["env"].get("observe_accel", False))
        p = config["plant"]
        self.wheelbase = float(p["wheelbase"])
        self.understeer = float(p["understeer_gradient"])
        # Nominal tire_scrub, not a per-car draw: this prior is a FIXED
        # geometric calculation the policy trims, not itself something that
        # gets randomized per episode (that happens in plant.py, to the
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
        win = max(
            int(round(float(config["track"]["curvature_smooth_m"]) / track.ds)), 1
        )
        kern = np.ones(win) / win
        pad = np.concatenate([dk[-win:], dk, dk[:win]])
        self.dkappa = np.convolve(pad, kern, mode="same")[win : win + len(dk)]
        # How far ahead the prior evaluates itself.  Three delays stack up
        # between deciding on a command and the car actually being somewhere
        # else: command_dead_time (0.19 s) before the wheels move,
        # steering_tau (0.05 s) for the servo, and yaw_response_tau (0.34 s)
        # for the chassis to take up the yaw rate those wheels imply.  It is
        # a tuned constant in config.yaml rather than their sum, because the
        # lead term below already cancels part of the third one and summing
        # them double-counts it.  Keep it in the exported config to preserve inference parity.
        self.ff_horizon = float(config["env"]["steer_prior"]["horizon_s"])
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
        self.obs_dim = 2 * len(self.lookahead) + 15 + int(self.observe_accel)

        # Depth-scan gap centering: same table and (k_lateral, speed_floor)
        # shape as the centerline Stanley term above, applied to a lateral
        # offset read off the current scan instead of the pose. Camera is
        # geometry only (config-derived, stateless), so building one here
        # does not duplicate the env's/node's own Camera state.
        self.camera = Camera(config)
        dc = config["env"].get("depth_center", {})
        self.depth_center_gain = float(dc.get("gain", 0.0))
        self.depth_center_deadzone = np.deg2rad(float(dc.get("deadzone_deg", 10.0)))
        self.depth_center_max_range = float(dc.get("max_range_m", 2.5))
        # A driven apex on this course legitimately runs a few cm of
        # clearance (measured: 7 cm nominal minimum), well inside max_range_m
        # -- so gating on range alone fires the correction almost everywhere,
        # fighting the racing line instead of the wall it is meant for. Gate
        # on the SAME contact threshold env.py uses for `touching` instead:
        # then it is silent through a confidently-driven apex and only wakes
        # up exactly where recovery already applies.
        self.depth_center_touch_margin = float(
            dc.get("touch_margin_m", config["collision"]["safety_margin"])
        )

    def set_station(self, mask, station):
        """Seed the hint after a reset or a relocalization."""
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
        last_angle = np.interp(
            np.clip(last_steer, -1.0, 1.0), self.steer_pts, self.steer_ang
        )
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
        kappa_lead = t.at(
            station_p, t.kappa
        ) + self.lead_scale * self.yaw_tau * speed * t.at(station_p, self.dkappa)
        angle = np.arctan(eff_wheelbase * kappa_lead)
        # Stanley form: firm in a 2.5 m/s hairpin, gentle on a 5.2 m/s
        # straight, from one constant.  The yaw-rate term is damping: with a
        # 0.34 s chassis lag in the loop, position-and-heading feedback alone
        # is a second-order system with almost no damping in it, and it rings.
        rate_error = yaw_rate - speed * t.at(station_p, t.kappa)
        angle = (
            angle
            - self.k_heading * psi
            - self.k_rate * rate_error
            - np.arctan(self.k_lateral * lateral / np.maximum(speed, self.speed_floor))
        )
        angle = np.clip(angle, self.steer_ang[0], self.steer_ang[-1])
        # Through the measured table, so the command means the same thing here
        # as it does to the servo.  `rate_kin - yaw_rate` goes out with it:
        # see `compute` for why that one number is worth an observation
        # channel of its own.
        return np.interp(angle, self.steer_ang, self.steer_pts), rate_kin - yaw_rate

    def compute(
        self,
        x,
        y,
        yaw,
        speed,
        yaw_rate,
        speed_rate,
        prev_action,
        last_steer,
        lap_state,
        scan=None,
    ):
        """(B, OBS_DIM) plus the frame quantities the caller also wants.

        `lap_state` is (B, 3): how far through the current lap the car is
        (0-1), how long it has been on it (s), and the lap time it is trying
        to beat (s, or 0 on the first lap).

        Those three are in the observation because the reward pays for
        BEATING THE PREVIOUS LAP, and a reward for something the policy
        cannot see is not a reward, it is noise -- the same return arrives
        whatever it did, so there is no gradient pointing at the behavior
        that earned it.  What the network is actually handed is the useful
        combination: seconds up or down on the previous lap's pace, which is
        a quantity it can act on at any point in the lap rather than a
        stopwatch it has to integrate for itself.

        `scan` is the newest ENCODED depth-scan row (B, camera.width), the
        same one already stacked into the actor's own input -- callers pass
        `self.scans.frames[:, 0]`. Omit it (default) to skip the depth
        centering term entirely, matching old behavior exactly.
        """
        t = self.track
        idx, station, lateral = t.project(x, y, self.hint)
        self.hint = idx
        psi = t.heading_error(yaw, idx)

        v_cap_now = t.v_cap[idx]
        v_floor_now = self.v_floor[idx]
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
        steer_ff, rate_lag = self.steer_prior(
            station, x, y, yaw, speed, yaw_rate, last_steer
        )
        depth_offset = np.zeros_like(speed)
        if scan is not None and self.depth_center_gain > 0.0:
            depth_offset, depth_valid = self.camera.gap_offset(
                scan, self.depth_center_deadzone, self.depth_center_max_range
            )
            depth_valid = depth_valid & (clear <= self.depth_center_touch_margin)
            # Same Stanley MAGNITUDE as the centerline lateral term, opposite
            # sign: `lateral` above is the CAR's offset from the target (left
            # is positive, and steer_prior steers right -- subtracts -- to
            # correct it), but `depth_offset` is the TARGET's offset from the
            # CAR (the gap's midpoint, camera-frame, left positive), so a
            # positive value means steer toward it, i.e. add. Composed in
            # ANGLE space (not added to steer_ff directly) so it goes through
            # the same table steer_prior used, and stays a residual the
            # network can still trim.
            corr = np.arctan(
                self.depth_center_gain * depth_offset / np.maximum(speed, self.speed_floor)
            )
            base_angle = np.interp(steer_ff, self.steer_pts, self.steer_ang)
            angle = np.clip(
                base_angle + np.where(depth_valid, corr, 0.0),
                self.steer_ang[0],
                self.steer_ang[-1],
            )
            steer_ff = np.interp(angle, self.steer_ang, self.steer_pts)

        accel_channel = (
            [np.clip(speed_rate / _ACCEL_SCALE, -3.0, 3.0)[:, None]]
            if self.observe_accel
            else []
        )
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
                # by half a meter.  Without this channel the policy cannot
                # tell which car it drew and its only safe answer is to drive
                # every car like the slowest one -- which is exactly how an
                # over-wide randomization once turned into a policy that took
                # 101 s a run and three rounds of reward tuning looking for a
                # problem that was not in the reward.  With it, the lag is a
                # state variable it can read off a single frame and respond
                # to, because for a first-order lag this difference IS the
                # state.
                np.clip(rate_lag / _RATE_SCALE, -3.0, 3.0)[:, None],
                # HOW HARD THIS CAR IS ACTUALLY SLOWING, when enabled.  The
                # car has no brakes, so 5.2 -> 2.5 m/s is 9.3 m of coasting
                # against ~12 m straights, and `coast_scale` redraws the drag
                # +/-20% each episode -- nearly two meters of stopping
                # distance the policy otherwise cannot see.
                #
                # MEASURED RESULT: it raised top speed 2.7 -> 3.3 m/s and did
                # NOT pay for itself overall (run v4 scored below v3.5 on the
                # same reward).  Kept, off by default, because the finding is
                # that the straights are limited by something else -- and the
                # next person should not spend another run rediscovering it.
                *accel_channel,
                # HOW HARD THIS PARTICULAR CAR IS ACTUALLY SLOWING DOWN.
                #
                # This is the straights.  The car has no brakes, so shedding
                # 5.2 -> 2.5 m/s is 9.3 m of coasting against straights of
                # about 12 m -- and `coast_scale` redraws the drag +/-20%
                # every episode, which moves that stopping distance by nearly
                # two meters.  A policy that cannot tell which car it drew
                # cannot know where to lift off, so its only safe answer is
                # to never use the straight at all: run v3 topped out at
                # 2.7 m/s where the cap allowed 5.2.
                #
                # Achieved dv/dt is the drag, measured, once per tick.  Same
                # argument as the chassis-lag channel above: a randomization
                # the policy can OBSERVE costs lap time; one it cannot costs
                # far more.
                lap_progress[:, None],
                has_target[:, None],
                np.clip(pace / _PACE_SCALE, -3.0, 3.0)[:, None],
                cap_ahead / self.v_ref,
                np.clip(kappa_ahead * _KAPPA_SCALE, -2.0, 2.0),
            ],
            axis=1,
        )
        frame = dict(
            index=idx,
            station=station,
            lateral=lateral,
            psi=psi,
            v_cap=v_cap_now,
            v_floor=v_floor_now,
            clearance=clear,
            steer_ff=steer_ff,
            depth_center_offset=depth_offset,
            room_left=room_left,
            room_right=room_right,
            pace=pace,
            rate_lag=rate_lag,
        )
        return np.clip(obs, -10.0, 10.0).astype(np.float32), frame


def speed_floor(track, config):
    """Coast-feasible reference profile. The v_floor field is retained for telemetry."""
    from baseline import feasible_profile

    cfg = config["track"]
    # FEASIBLE FOR THE SLIPPERIEST CAR, not the nominal one.  The floor is
    # structural -- no action can go below it -- and the car has no brakes,
    # so a floor clipped against nominal drag demands up to 0.35 m/s more
    # than a low-drag draw can shed before a hairpin (measured: drag x0.77,
    # 2.8% of the lap).  That is an infeasible task, not a driving mistake.
    # Clipped against `floor_drag_scale` of the drag instead, every car in the
    # randomization range can meet every cap; it is only a minimum, so a car
    # with more drag is still free to carry more speed.
    scale = float(cfg.get("floor_drag_scale", 1.0))
    plant = dict(config["plant"])
    plant["coast_f0"] = float(plant["coast_f0"]) * scale
    plant["coast_f1"] = float(plant["coast_f1"]) * scale
    feasible = feasible_profile(track, {**config, "plant": plant})

    # RED is the rule: the stations whose cap is the hairpin limit.
    red = track.v_cap <= float(cfg["v_hairpin"]) + 1e-6
    # YELLOW is every station where a hairpin is DICTATING the speed even
    # though the local geometry would allow more -- i.e. the coast-down into
    # one and the acceleration out of it.  `feasible_profile` already knows
    # exactly where that is: it is below the straight limit precisely where a
    # corner ahead (or behind, through the acceleration pass) is binding.
    #
    # Defining the band by the v_cap taper instead makes it ~4% of the lap,
    # because that taper is only a meter or two wide.  This is the band the
    # color is really pointing at: the ~9 m of coasting each hairpin entry
    # needs on a car with no brakes, plus the ~5 m of getting back up to
    # speed afterwards.
    yellow = ~red & (feasible < float(cfg["v_straight"]) - 1e-6)

    raw = np.where(
        red,
        float(cfg["speed_floor_hairpin"]),
        np.where(
            yellow, float(cfg["speed_floor_taper"]), float(cfg["speed_floor_open"])
        ),
    )
    return np.minimum(raw, feasible)


def scale_action(action, v_cap, steer_ff, residual, v_floor=None):
    """Residual steering and speed around a coast-feasible reference.

    Throttle -1 coasts, 0 follows the reference, +1 requests the local cap.
    The policy can sacrifice speed to recover clearance without fighting a floor.
    """
    throttle = np.clip(action[:, 1], -1.0, 1.0)
    ref = 0.5 * v_cap if v_floor is None else np.clip(v_floor, 0, v_cap)
    speed = np.where(throttle < 0, ref * (1 + throttle), ref + (v_cap - ref) * throttle)
    steer = np.clip(steer_ff + residual * np.clip(action[:, 0], -1, 1), -1, 1)
    return steer, speed
