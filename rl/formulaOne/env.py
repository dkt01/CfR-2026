#!/usr/bin/env python3
"""Two laps of the Speed Course, vectorised over B cars.

Plain numpy, no Gazebo, no ROS.  On one CPU core this steps ~10^5 car-steps a
second against Gazebo's ~10, which is the difference between a policy that has
seen 10^7 steps and one that has seen 10^5.  Gazebo is still where a trained
policy is CHECKED -- `validate.sh` runs it there with the real nodes and the
real physics -- but it is not where it is trained.

The env keeps two versions of the car's pose and never confuses them:

    plant.x/y/yaw     where the car actually is.  Collisions, lap counting and
                      the reward are scored against this.
    sensed pose       what the ZED would have reported: delayed, noisy, and
                      slowly drifting.  This is the only pose the policy sees.

That split is what makes the localisation error a thing the policy is trained
THROUGH rather than a thing that surprises it on the day.
"""

from __future__ import annotations

import numpy as np

import observation as obs_mod
import track as track_mod
from observation import ObservationBuilder, scale_action
from plant import Plant
from reward import Reward


class FormulaOneEnv:
    """B cars racing the same course, auto-resetting like an SB3 VecEnv."""

    def __init__(self, config, track, n_envs=1, seed=0, deterministic=False):
        self.cfg = config
        self.track = track
        self.n = int(n_envs)
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic

        env = config["env"]
        self.control_hz = float(env["control_hz"])
        self.dt_nominal = 1.0 / self.control_hz
        self.substeps = int(env["substeps"])
        self.laps = int(env["laps"])
        self.timeout = float(env["episode_timeout_s"])
        self.random_start = bool(env["random_start"]) and not deterministic
        self.start_speed_range = env["random_start_speed"]
        self.stall_speed = float(env["stall_speed"])
        self.stall_patience = float(env["stall_patience_s"])
        self.stop_speed = float(env["stop_speed"])
        self.stop_timeout = float(env["stop_timeout_s"])
        self.residual = float(env["steer_residual"])
        self.yaw_filter = float(env["yaw_rate_filter"])
        # Defaulted, like `observe_accel`: a config.yaml snapshot saved into a
        # run directory before this existed has no such key, and those runs
        # still have to be replayable.
        self.accel_filter = float(env.get("speed_rate_filter", 0.25))

        veh = config["vehicle"]
        self.half_length = float(veh["length"]) / 2
        self.half_width = float(veh["width"]) / 2

        if deterministic:
            # A copy, so turning randomisation off for evaluation cannot leak
            # back into the training config object.
            self.cfg = {**config, "randomize": {**config["randomize"], "enabled": False}}

        self.plant = Plant(self.cfg, self.n, self.rng)
        self.reward = Reward(config)
        self.obs_builder = ObservationBuilder(track, config, self.n)
        # The BUILDER's width, not the module default: it depends on config
        # (see `observe_accel`), and a policy trained before that channel
        # existed must still load.
        self.obs_dim = self.obs_builder.obs_dim
        self.act_dim = 2

        self.target_distance = self.laps * track.length
        self.pose_history = np.zeros((self.n, 8, 3))
        self.pose_head = 0
        self._alloc()

    def _alloc(self):
        z = np.zeros(self.n)
        self.hint = np.zeros(self.n, dtype=np.int64)
        self.station = z.copy()
        self.distance = z.copy()
        self.elapsed = z.copy()
        self.laps_done = np.zeros(self.n, dtype=np.int32)
        self.stalled_for = z.copy()
        self.prev_action = np.zeros((self.n, 2))
        self.last_steer = np.zeros(self.n)
        self.last_steer_step = z.copy()
        # The stopping phase: two laps are done and the car is coasting to
        # rest.  `race_time` freezes the clock at the line so the lap time
        # reported is the RACE, not the race plus however long the coast took.
        self.stopping = np.zeros(self.n, dtype=bool)
        self.stopping_for = z.copy()
        self.race_time = np.full(self.n, np.nan)
        # Lap timing, for the reward that pays for beating the previous lap.
        self.lap_started_at = z.copy()
        self.last_lap_time = z.copy()
        self.best_lap_time = np.full(self.n, np.inf)
        # Potential for the lap-improvement shaping: w * clip(pace), where
        # pace is seconds up on the previous lap.  The reward pays its
        # step-to-step CHANGE, which telescopes over a lap to exactly
        # w * (previous lap - this lap) -- see reward.py.
        self.pace_potential = z.copy()
        self.sensed_x = np.zeros(self.n)
        self.sensed_y = np.zeros(self.n)
        self.sensed_yaw = np.zeros(self.n)
        self.yaw_rate = np.zeros(self.n)
        self.speed_rate = np.zeros(self.n)
        self.prev_obs_speed = np.zeros(self.n)
        self.dt = np.full(self.n, self.dt_nominal)
        self.latency_steps = np.zeros(self.n, dtype=np.int32)
        self.noise_xy = z.copy()
        self.noise_yaw = z.copy()
        self.drift_dir = z.copy()
        self.drift_rate = z.copy()
        self.drift_yaw_rate = z.copy()
        self.jitter = np.full(self.n, 1.0)
        self.ep_return = z.copy()
        self.ep_min_clearance = np.full(self.n, 9.0)
        self.ep_overspeed = z.copy()
        self.ep_lateral_sum = z.copy()
        self.ep_lateral_n = z.copy()
        self.ep_lateral_max = z.copy()
        self.ep_steer_jerk_sum = z.copy()
        self.ep_steps = z.copy()
        self.ep_floor_deficit = z.copy()

    # ----------------------------------------------------------------- reset

    def _sample_start(self, k):
        t = self.track
        r = self.cfg["randomize"]
        on = r["enabled"]
        if self.random_start:
            station = self.rng.uniform(0.0, t.length, k)
            lo, hi = self.start_speed_range
            speed = self.rng.uniform(lo, hi, k)
            # Never start above the cap at the station drawn, or the first
            # thing the policy learns is that overspeed is unavoidable.
            speed = np.minimum(speed, t.at(station, t.v_cap))
        else:
            station = np.full(k, t.start_station)
            speed = np.zeros(k)

        lat = self.rng.uniform(*r["start_lateral"], k) if on else np.zeros(k)
        head = self.rng.uniform(*r["start_heading"], k) if on else np.zeros(k)
        idx = np.searchsorted(t.s, station % t.length) % len(t.s)
        x = t.x[idx] - t.ty[idx] * lat
        y = t.y[idx] + t.tx[idx] * lat
        yaw = np.arctan2(t.ty[idx], t.tx[idx]) + head
        return station, x, y, yaw, speed

    def _sample_sensor(self, mask, k):
        r = self.cfg["randomize"]
        if not r["enabled"]:
            self.latency_steps[mask] = 0
            self.noise_xy[mask] = 0.0
            self.noise_yaw[mask] = 0.0
            self.drift_rate[mask] = 0.0
            self.drift_yaw_rate[mask] = 0.0
            self.jitter[mask] = 1.0
            return
        u = self.rng.uniform
        lat = u(*r["pose_latency"], k)
        self.latency_steps[mask] = np.clip(
            np.round(lat * self.control_hz), 0, self.pose_history.shape[1] - 1
        ).astype(np.int32)
        self.noise_xy[mask] = u(*r["pose_noise_xy"], k)
        self.noise_yaw[mask] = u(*r["pose_noise_yaw"], k)
        self.drift_dir[mask] = u(-np.pi, np.pi, k)
        self.drift_rate[mask] = u(*r["pose_drift_m_per_lap"], k)
        self.drift_yaw_rate[mask] = u(*r["pose_drift_rad_per_lap"], k) * self.rng.choice(
            [-1.0, 1.0], k
        )
        self.jitter[mask] = u(*r["control_jitter"], k)

    def _reset_idx(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        station, x, y, yaw, speed = self._sample_start(k)
        self.plant.reset(mask, x, y, yaw, speed)
        self._sample_sensor(mask, k)
        self.station[mask] = station
        self.distance[mask] = 0.0
        self.elapsed[mask] = 0.0
        self.laps_done[mask] = 0
        self.stalled_for[mask] = 0.0
        self.prev_action[mask] = 0.0
        self.last_steer[mask] = 0.0
        self.last_steer_step[mask] = 0.0
        self.stopping[mask] = False
        self.stopping_for[mask] = 0.0
        self.race_time[mask] = np.nan
        self.lap_started_at[mask] = 0.0
        self.last_lap_time[mask] = 0.0
        self.best_lap_time[mask] = np.inf
        self.pace_potential[mask] = 0.0
        self.dt[mask] = self.dt_nominal * self.jitter[mask]
        self.ep_return[mask] = 0.0
        self.ep_min_clearance[mask] = 9.0
        self.ep_overspeed[mask] = 0.0
        self.ep_lateral_sum[mask] = 0.0
        self.ep_lateral_n[mask] = 0.0
        self.ep_lateral_max[mask] = 0.0
        self.ep_steer_jerk_sum[mask] = 0.0
        self.ep_steps[mask] = 0.0
        self.ep_floor_deficit[mask] = 0.0
        idx = np.searchsorted(self.track.s, station % self.track.length)
        self.hint[mask] = np.clip(idx, 0, len(self.track.s) - 1)
        self.obs_builder.set_station(mask, station)
        self.pose_history[mask] = np.stack([x, y, yaw], axis=1)[:, None, :]
        self.sensed_x[mask], self.sensed_y[mask] = x, y
        self.sensed_yaw[mask] = yaw
        self.yaw_rate[mask] = 0.0
        # Seed the previous speed with the speed the car is actually placed
        # at, so a car dropped in mid-course at 4 m/s does not read as having
        # just accelerated from rest.
        self.speed_rate[mask] = 0.0
        self.prev_obs_speed[mask] = np.where(speed < 0.3, 0.0, speed)

    def reset(self):
        self._reset_idx(np.ones(self.n, dtype=bool))
        self._record_pose()
        self._sense()
        return self._observe()

    # ----------------------------------------------------------------- sense

    def _record_pose(self):
        """Append the true pose to the ring, ONCE per control step.

        Separate from `_observe` because a step that ends an episode observes
        twice -- once for the terminal observation and once for the reset
        one -- and a ring that advanced on each of those would sample the
        delay from the wrong place.  Getting this wrong is silent: the policy
        simply reads a pose that never changes, and the run looks like a
        localisation failure rather than a bookkeeping one.
        """
        p = self.plant
        self.pose_history[:, self.pose_head] = np.stack([p.x, p.y, p.yaw], axis=1)
        self.pose_head = (self.pose_head + 1) % self.pose_history.shape[1]

    def _sense(self, mask=None):
        """Sensed pose and yaw rate, ONCE per control step.

        Split out from `_observe` because a step that ends an episode observes
        twice -- once for the terminal observation and once for the reset one
        -- and the yaw rate is a difference, so computing it inside `_observe`
        would difference the same pose against itself and hand the steering
        prior a zero rate on every terminal step.
        """
        p = self.plant
        rows = np.arange(self.n)
        # pose_head points at the NEXT slot to write, so the freshest sample
        # is one behind it.
        head = (self.pose_head - 1) % self.pose_history.shape[1]
        idx = (head - self.latency_steps) % self.pose_history.shape[1]
        delayed = self.pose_history[rows, idx]

        laps_travelled = self.distance / self.track.length
        drift = self.drift_rate * laps_travelled
        x = delayed[:, 0] + np.cos(self.drift_dir) * drift
        y = delayed[:, 1] + np.sin(self.drift_dir) * drift
        yaw = delayed[:, 2] + self.drift_yaw_rate * laps_travelled
        if self.cfg["randomize"]["enabled"]:
            x = x + self.rng.normal(0.0, 1.0, self.n) * self.noise_xy
            y = y + self.rng.normal(0.0, 1.0, self.n) * self.noise_xy
            yaw = yaw + self.rng.normal(0.0, 1.0, self.n) * self.noise_yaw

        step = np.arctan2(np.sin(yaw - self.sensed_yaw), np.cos(yaw - self.sensed_yaw))
        # A car at full lock and full speed turns at under 6 rad/s; anything
        # past that is a reset or a pose jump, not a yaw rate.
        rate = np.clip(step / self.dt, -8.0, 8.0)
        rate = (1 - self.yaw_filter) * self.yaw_rate + self.yaw_filter * rate

        # Achieved acceleration, from the speed the POLICY sees (tachometer
        # blind spot included), differenced once per control step for the same
        # reason the yaw rate is: computing it inside `_observe` would
        # difference a step against itself on every terminal tick.
        obs_speed = np.where(p.speed < 0.3, 0.0, p.speed)
        raw_accel = np.clip((obs_speed - self.prev_obs_speed) / self.dt, -20.0, 20.0)
        accel = ((1 - self.accel_filter) * self.speed_rate
                 + self.accel_filter * raw_accel)

        keep = np.ones(self.n, dtype=bool) if mask is None else mask
        self.speed_rate = np.where(keep, accel, self.speed_rate)
        self.prev_obs_speed = np.where(keep, obs_speed, self.prev_obs_speed)
        self.sensed_x = np.where(keep, x, self.sensed_x)
        self.sensed_y = np.where(keep, y, self.sensed_y)
        self.sensed_yaw = np.where(keep, yaw, self.sensed_yaw)
        self.yaw_rate = np.where(keep, rate, self.yaw_rate)

    def _observe(self):
        p = self.plant
        # The tachometer cannot resolve below ~0.3 m/s (one magnet, 0.4 s
        # stall timeout), so the policy must not be handed a clean crawl.
        speed = np.where(p.speed < 0.3, 0.0, p.speed)
        obs, self.frame = self.obs_builder.compute(
            self.sensed_x, self.sensed_y, self.sensed_yaw, speed,
            self.yaw_rate, self.speed_rate, self.prev_action, self.last_steer,
            self._lap_state(),
        )
        return obs

    def _lap_state(self):
        """(B, 3) lap progress, time on this lap, time to beat."""
        lap_progress = np.clip(
            (self.distance - self.laps_done * self.track.length)
            / self.track.length, 0.0, 1.0)
        return np.stack([lap_progress,
                         self.elapsed - self.lap_started_at,
                         self.last_lap_time], axis=1)

    # ------------------------------------------------------------------ step

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(self.n, 2)
        p = self.plant
        t = self.track

        # Scale the action with what the car can KNOW -- the sensed station's
        # cap and feedforward -- not the true one.  The reward below still
        # scores overspeed against the true cap, because that is the rule the
        # car is actually judged by.  Using truth here instead would train a
        # policy that silently depends on perfect localisation.
        steer_cmd, speed_cmd = scale_action(
            action, self.frame["v_cap"], self.frame["steer_ff"], self.residual,
            self.frame["v_floor"],
        )
        # TWO LAPS DONE MEANS STOP.  The throttle is taken away and the
        # steering is not: the car has no brakes, so it has fifteen metres of
        # corridor to coast through and it still has to stay in it.
        speed_cmd = np.where(self.stopping, 0.0, speed_cmd)
        steer_cmd, speed_cmd = p.push_command(steer_cmd, speed_cmd)
        steer_step = steer_cmd - self.last_steer
        steer_jerk = steer_step - self.last_steer_step
        v_cap_true = t.v_cap[self.hint]

        sub_dt = self.dt / self.substeps
        min_clear = np.full(self.n, 9.0)
        for _ in range(self.substeps):
            p.substep(steer_cmd, speed_cmd, sub_dt)
            clear = t.body_clearance(p.x, p.y, p.yaw, self.half_length, self.half_width)
            min_clear = np.minimum(min_clear, clear)

        self._record_pose()
        self._sense()
        prev_station = self.station
        self.hint, self.station, lateral = t.project(p.x, p.y, self.hint)
        # Wrapped, and clipped: a step longer than half the lap is the
        # projection having jumped, not the car having driven.
        advance = (self.station - prev_station + t.length / 2) % t.length - t.length / 2
        advance = np.clip(advance, -2.0, 2.0)
        lateral_true = lateral
        self.distance += advance
        self.elapsed += self.dt

        crossed = np.floor(np.maximum(self.distance, 0.0) / t.length).astype(np.int32)
        lapped_mask = crossed > self.laps_done
        lapped = lapped_mask.astype(float)
        self.laps_done = np.maximum(self.laps_done, crossed)

        # Close the lap that just ended and open the next one.
        lap_time = self.elapsed - self.lap_started_at
        self.last_lap_time = np.where(lapped_mask, lap_time, self.last_lap_time)
        self.best_lap_time = np.where(lapped_mask,
                                      np.minimum(self.best_lap_time, lap_time),
                                      self.best_lap_time)
        self.lap_started_at = np.where(lapped_mask, self.elapsed, self.lap_started_at)

        crashed = min_clear <= 0.0
        # Crossing the line starts the stopping phase; it does not end the run.
        reached = self.distance >= self.target_distance
        just_finished = reached & ~self.stopping
        self.race_time = np.where(just_finished, self.elapsed, self.race_time)
        self.stopping = self.stopping | reached
        self.stopping_for = np.where(self.stopping, self.stopping_for + self.dt, 0.0)
        stopped = self.stopping & (p.speed <= self.stop_speed)
        stop_failed = self.stopping & (self.stopping_for >= self.stop_timeout)

        self.stalled_for = np.where(
            p.speed < self.stall_speed, self.stalled_for + self.dt, 0.0
        )
        # A car coasting down on purpose is not a car that has stalled.
        stalled = (self.stalled_for >= self.stall_patience) & ~self.stopping
        timed_out = self.elapsed >= self.timeout

        # --- how much of the previous lap's pace has been taken back.
        # `pace` is where the previous lap would have been by now minus where
        # this one is; the potential is that, weighted and capped, and what
        # the reward is paid is its CHANGE.  Two guards:
        #   * on the step a lap closes, pace jumps discontinuously (progress
        #     falls back to 0 and the target changes) -- that jump is
        #     bookkeeping, not driving, so the potential is re-seeded and
        #     nothing is paid for it;
        #   * during the stopping phase the clock runs on with no progress
        #     left to make, so the potential would drain away at 2.0/s for
        #     something the policy was told to do.
        r = self.cfg["reward"]
        lap_progress = np.clip(
            (self.distance - self.laps_done * self.track.length)
            / self.track.length, 0.0, 1.0)
        pace = np.where(self.last_lap_time > 0.0,
                        self.last_lap_time * lap_progress
                        - (self.elapsed - self.lap_started_at), 0.0)
        potential = float(r["lap_improve"]) * np.clip(
            pace, -float(r["lap_improve_cap"]), float(r["lap_improve_cap"]))
        quiet = lapped_mask | self.stopping
        lap_gain = np.where(quiet, 0.0, potential - self.pace_potential)
        self.pace_potential = potential

        reward, terms = self.reward.step(
            dt=self.dt,
            advance=advance,
            speed=p.speed,
            v_cap=v_cap_true,
            clearance=min_clear,
            lateral=lateral_true,
            steer_step=steer_step,
            steer_jerk=steer_jerk,
            lapped=lapped,
            finished=just_finished.astype(float),
            crashed=crashed.astype(float),
            stalled=stalled.astype(float),
            stopping=self.stopping.astype(float),
            stopped=stopped.astype(float),
            lap_gain=lap_gain,
        )

        # The RAW action is what the network emitted and what it should see
        # next tick; the realised command is tracked separately for the rate
        # penalty.
        self.prev_action = np.clip(action, -1.0, 1.0)
        self.last_steer = steer_cmd
        self.last_steer_step = steer_step
        self.ep_return += reward
        self.ep_min_clearance = np.minimum(self.ep_min_clearance, min_clear)
        self.ep_overspeed = np.maximum(self.ep_overspeed, p.speed - v_cap_true)
        # Cross-track error, tracked while RACING only: the reward asks for
        # zero of it and a mean that also averaged the coast-down would be
        # reporting a different quantity from the one being paid for.
        live = ~self.stopping
        self.ep_lateral_sum += np.abs(lateral_true) * live
        self.ep_lateral_n += live
        self.ep_lateral_max = np.maximum(self.ep_lateral_max,
                                         np.abs(lateral_true) * live)
        self.ep_steer_jerk_sum += steer_jerk**2
        self.ep_steps += 1
        # How far under the floor the car actually ran, while racing.  The
        # floor is structural on the COMMAND, so any shortfall here is the
        # car failing to reach what it asked for -- accelerating out of a
        # corner, or a draw whose slew rate cannot keep up -- not the policy
        # disobeying it.
        # Only while NOT meaningfully accelerating.  The floor binds the
        # command, and the car takes the bridge's 2.0 m/s^2 to get there, so
        # counting every wind-up would report 3.5 m/s of "shortfall" at a
        # standing start and after every corner exit.  Cars below 1 m/s are
        # excluded for the same reason -- at the line the car is stationary
        # and the floor is 3.5, which is not a violation, it is a launch.
        # What this measures is settled-but-too-slow.
        winding_up = (self.speed_rate > 0.5) | (p.speed < 1.0)
        self.ep_floor_deficit = np.maximum(
            self.ep_floor_deficit,
            np.where(live & ~winding_up,
                     np.maximum(self.frame["v_floor"] - p.speed, 0.0), 0.0))

        terminated = crashed | stopped | stop_failed | stalled
        truncated = timed_out & ~terminated
        done = terminated | truncated

        obs = self._observe()
        info = [{} for _ in range(self.n)]
        if done.any():
            final_obs = obs
            for i in np.flatnonzero(done):
                info[i] = {
                    "terminal_observation": final_obs[i],
                    "episode": {
                        "r": float(self.ep_return[i]),
                        "l": int(self.elapsed[i] * self.control_hz),
                        "t": float(self.elapsed[i]),
                    },
                    "distance": float(self.distance[i]),
                    "laps": int(self.laps_done[i]),
                    # `finished` is the two laps; `stopped` is the two laps
                    # AND at rest, which is what the run actually asks for.
                    "finished": bool(self.stopping[i]),
                    "stopped": bool(stopped[i]),
                    "crashed": bool(crashed[i]),
                    "stalled": bool(stalled[i]),
                    "min_clearance": float(self.ep_min_clearance[i]),
                    "max_overspeed": float(self.ep_overspeed[i]),
                    "race_time": float(self.race_time[i]),
                    "stop_distance": float(self.distance[i] - self.target_distance),
                    "lap_time": float(self.race_time[i] / self.laps)
                                if np.isfinite(self.race_time[i])
                                else float(self.elapsed[i] / max(self.laps_done[i], 1)),
                    "best_lap": float(self.best_lap_time[i])
                                if np.isfinite(self.best_lap_time[i]) else float("nan"),
                    "last_lap": float(self.last_lap_time[i]),
                    "mean_cte": float(self.ep_lateral_sum[i]
                                      / max(self.ep_lateral_n[i], 1)),
                    "max_cte": float(self.ep_lateral_max[i]),
                    "floor_deficit": float(self.ep_floor_deficit[i]),
                    "steer_jerk_rms": float(np.sqrt(
                        self.ep_steer_jerk_sum[i] / max(self.ep_steps[i], 1))),
                }
            self._reset_idx(done)
            # A car that has just been placed has no history to be delayed
            # against, so its ring is refilled with where it now stands.
            self.pose_history[done] = np.stack(
                [p.x[done], p.y[done], p.yaw[done]], axis=1
            )[:, None, :]
            # Re-sense ONLY the cars that moved, so the others' yaw rates are
            # not differenced against a pose they have not left yet.
            self._sense(done)
            obs = self._observe()
        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------------- state

    def scripted_action(self, driver):
        """Action array from the scripted baseline, in the policy's own space."""
        return driver.act(self.frame["station"], self.plant.speed,
                          self.frame["v_cap"], self.frame["v_floor"])

    def snapshot(self):
        """Per-car truth, for plotting and for the self-test."""
        p = self.plant
        return dict(
            x=p.x.copy(), y=p.y.copy(), yaw=p.yaw.copy(), speed=p.speed.copy(),
            steer=p.steer_angle.copy(), station=self.station.copy(),
            distance=self.distance.copy(), elapsed=self.elapsed.copy(),
            v_cap=self.frame["v_cap"].copy(),
            steer_ff=self.frame["steer_ff"].copy(),
            clearance=self.track.body_clearance(
                p.x, p.y, p.yaw, self.half_length, self.half_width
            ),
        )


def make(config, n_envs=1, seed=0, deterministic=False, repo_root=None):
    """Build the track once and wrap it in an env."""
    from pathlib import Path

    root = repo_root or Path(__file__).resolve().parents[2]
    trk = track_mod.build(config, root)
    return FormulaOneEnv(config, trk, n_envs, seed, deterministic)
