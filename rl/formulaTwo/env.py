"""Three laps of the Speed Course as built, seen through a ZED, vectorised.

formulaOne's env, with three things added:

  THE WORLD IS NOT THE MAP.  `world.World` warps the bales per episode.  The
  plant drives, collides and is scored in the warped world; the policy's map
  features (prior, lookahead, clearance channel) read the NOMINAL map from a
  noisy, delayed, drifting pose -- exactly what the car will have.

  THE CAR CAN SEE.  At 10-15 Hz a depth frame is rendered from the true pose
  of `depth_latency` ago, corrupted with the ZED's artifacts, reduced to a
  64-beam virtual LiDAR by the same code the car runs (perception.py) and
  pushed onto a 4-frame stack.

  THE CRITIC KNOWS THE TRUTH.  The observation is [actor | privileged].  The
  actor reads only the first `actor_dim` columns; the critic reads all of it:
  true cross-track and heading error, true clearance, the map error around
  the car, the pose error, and the car's own draw (friction, mass, lags).
  Only the actor ships.

Three poses, never confused:
    plant.x/y/yaw   truth.  Collisions, laps and the reward.
    sensed pose     what the ZED reports.  The map features.
    camera pose     truth, `depth_latency` ago, through a mount that is not
                    quite where the car thinks.  The depth scan.
"""

from __future__ import annotations

import numpy as np

import track as track_mod
from observation import ObservationBuilder, scale_action
from perception import Camera, ScanStack, corrupt
from plant import Plant
from randomization import draw
from reward import Reward
from world import World

N_PRIVILEGED = 27


class FormulaTwoEnv:
    def __init__(self, config, track, n_envs=1, seed=0, deterministic=False):
        self.cfg = config
        self.track = track
        self.n = int(n_envs)
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic
        # Randomisation strength, 0..1; the training curriculum sets it.
        self.dr_scale = 1.0

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
        self.accel_filter = float(env.get("speed_rate_filter", 0.25))

        if deterministic:
            self.cfg = {
                **config,
                "randomize": {**config["randomize"], "enabled": False},
            }
        self.randomized = bool(self.cfg["randomize"]["enabled"])

        self.plant = Plant(self.cfg, self.n, self.rng)
        self.world = World(track, self.cfg, self.n, self.rng)
        self.camera = Camera(self.cfg)
        self.scans = ScanStack(self.n, self.camera.stack, self.camera.width)
        self.reward = Reward(config)
        self.obs_builder = ObservationBuilder(track, config, self.n)
        self.map_dim = self.obs_builder.obs_dim
        self.actor_dim = self.map_dim + self.camera.stack * self.camera.width + 1
        self.obs_dim = self.actor_dim + N_PRIVILEGED
        self.act_dim = 2

        self.target_distance = self.laps * track.length
        self.pose_history = np.zeros((self.n, 8, 3))
        self.pose_head = 0
        self._alloc()

    def _alloc(self):
        z = np.zeros(self.n)
        n = self.n
        self.hint = np.zeros(n, dtype=np.int64)
        self.station = z.copy()
        self.lateral_true = z.copy()
        self.psi_true = z.copy()
        self.clear_true = np.full(n, 9.0)
        self.distance = z.copy()
        self.elapsed = z.copy()
        self.laps_done = np.zeros(n, dtype=np.int32)
        self.stalled_for = z.copy()
        self.prev_action = np.zeros((n, 2))
        self.last_steer = z.copy()
        self.last_steer_step = z.copy()
        self.stopping = np.zeros(n, dtype=bool)
        self.stopping_for = z.copy()
        self.race_time = np.full(n, np.nan)
        self.lap_started_at = z.copy()
        self.last_lap_time = z.copy()
        self.best_lap_time = np.full(n, np.inf)
        self.lap_times = np.zeros((n, self.laps))
        self.pace_potential = z.copy()
        self.sensed_x, self.sensed_y, self.sensed_yaw = z.copy(), z.copy(), z.copy()
        self.yaw_rate = z.copy()
        self.speed_rate = z.copy()
        self.prev_obs_speed = z.copy()
        self.dt = np.full(n, self.dt_nominal)
        self.latency_steps = np.zeros(n, dtype=np.int32)
        self.noise_xy, self.noise_yaw = z.copy(), z.copy()
        self.drift_dir, self.drift_rate, self.drift_yaw_rate = z.copy(), z.copy(), z.copy()
        self.jitter = np.ones(n)
        # Camera draw.
        self.cam = {
            k: z.copy()
            for k in (
                "period",
                "timer",
                "latency",
                "drop",
                "noise_s0",
                "noise_s2",
                "pixel_dropout",
                "blob_rate",
                "edge_bleed",
                "pitch",
                "pitch_jitter",
                "yaw",
                "height",
                "hfov_half",
            )
        }
        self.cam_lat_steps = np.zeros(n, dtype=np.int32)
        # Episode statistics.
        self.ep_return = z.copy()
        self.ep_min_clearance = np.full(n, 9.0)
        self.ep_overspeed = z.copy()
        self.ep_lateral_sum = z.copy()
        self.ep_lateral_n = z.copy()
        self.ep_lateral_max = z.copy()
        self.ep_gate_sum = z.copy()
        self.ep_steer_jerk_sum = z.copy()
        self.ep_steps = z.copy()

    # ----------------------------------------------------------------- reset

    def _d(self, spec, k):
        return draw(self.rng, spec, k, self.dr_scale, self.randomized)

    def _sample_start(self, k):
        t = self.track
        r = self.cfg["randomize"]
        if self.random_start:
            station = self.rng.uniform(0.0, t.length, k)
            speed = self.rng.uniform(*self.start_speed_range, k)
            speed = np.minimum(speed, t.at(station, t.v_cap))
        else:
            station = np.full(k, t.start_station)
            speed = np.zeros(k)
        lat = self._d(r["start_lateral"], k)
        head = self._d(r["start_heading"], k)
        idx = np.searchsorted(t.s, station % t.length) % len(t.s)
        return station, idx, lat, head, speed

    def _sample_sensor(self, mask, k):
        r = self.cfg["randomize"]
        lat = self._d(r["pose_latency"], k)
        self.latency_steps[mask] = np.clip(
            np.round(lat * self.control_hz), 0, self.pose_history.shape[1] - 1
        ).astype(np.int32)
        self.noise_xy[mask] = self._d(r["pose_noise_xy"], k)
        self.noise_yaw[mask] = self._d(r["pose_noise_yaw"], k)
        self.drift_dir[mask] = self.rng.uniform(-np.pi, np.pi, k)
        self.drift_rate[mask] = self._d(r["pose_drift_m_per_lap"], k)
        self.drift_yaw_rate[mask] = self._d(
            r["pose_drift_rad_per_lap"], k
        ) * self.rng.choice([-1.0, 1.0], k)
        self.jitter[mask] = self._d(r["control_jitter"], k)

        s = self.cfg["sensor_noise"]
        c = self.cam
        c["period"][mask] = 1.0 / self._d(s["camera_hz"], k)
        c["timer"][mask] = self.rng.uniform(0, 1, k) * c["period"][mask]
        c["latency"][mask] = self._d(s["depth_latency"], k)
        for name in ("noise_s0", "noise_s2", "pixel_dropout", "blob_rate", "edge_bleed"):
            c[name][mask] = self._d(s[name], k)
        c["height"][mask] = self._d(s["height_err"], k)
        c["drop"][mask] = self._d(s["frame_drop"], k)
        c["pitch"][mask] = np.deg2rad(self._d(s["pitch_err_deg"], k))
        c["pitch_jitter"][mask] = np.deg2rad(self._d(s["pitch_jitter_deg"], k))
        c["yaw"][mask] = np.deg2rad(self._d(s["yaw_err_deg"], k))
        c["hfov_half"][mask] = np.deg2rad(self._d(s["hfov_half_deg"], k))
        self.cam_lat_steps[mask] = np.clip(
            np.round(c["latency"][mask] * self.control_hz),
            0,
            self.pose_history.shape[1] - 1,
        ).astype(np.int32)

    def _reset_idx(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        t = self.track
        self.world.sample(mask, self.dr_scale, self.randomized)
        station, idx, lat, head, speed = self._sample_start(k)
        # Place the car on the AS-BUILT centerline: the nominal station pushed
        # through this episode's layout warp.
        rows = np.flatnonzero(mask)
        cx, cy = t.x[idx], t.y[idx]
        dx, dy = self.world.displacement(cx[:, None], cy[:, None], True, rows)
        x = cx + dx[:, 0] - t.ty[idx] * lat
        y = cy + dy[:, 0] + t.tx[idx] * lat
        yaw = np.arctan2(t.ty[idx], t.tx[idx]) + head
        self.plant.reset(mask, x, y, yaw, speed, self.dr_scale)
        self._sample_sensor(mask, k)

        for name in ("distance", "elapsed", "stalled_for", "last_steer"):
            getattr(self, name)[mask] = 0.0
        self.station[mask] = station
        self.laps_done[mask] = 0
        self.prev_action[mask] = 0.0
        self.last_steer_step[mask] = 0.0
        self.stopping[mask] = False
        self.stopping_for[mask] = 0.0
        self.race_time[mask] = np.nan
        self.lap_started_at[mask] = 0.0
        self.last_lap_time[mask] = 0.0
        self.best_lap_time[mask] = np.inf
        self.lap_times[mask] = 0.0
        self.pace_potential[mask] = 0.0
        self.dt[mask] = self.dt_nominal * self.jitter[mask]
        for name in (
            "ep_return",
            "ep_overspeed",
            "ep_lateral_sum",
            "ep_lateral_n",
            "ep_lateral_max",
            "ep_gate_sum",
            "ep_steer_jerk_sum",
            "ep_steps",
        ):
            getattr(self, name)[mask] = 0.0
        self.ep_min_clearance[mask] = 9.0
        self.hint[mask] = np.clip(idx, 0, len(t.s) - 1)
        self.obs_builder.set_station(mask, station)
        self.pose_history[mask] = np.stack([x, y, yaw], axis=1)[:, None, :]
        self.sensed_x[mask], self.sensed_y[mask], self.sensed_yaw[mask] = x, y, yaw
        self.yaw_rate[mask] = 0.0
        self.speed_rate[mask] = 0.0
        self.prev_obs_speed[mask] = np.where(speed < 0.3, 0.0, speed)
        # Truth, so the first observation's privileged block is right.
        p = self.plant
        _, _, lt, pt = self.world.frenet(p.x, p.y, p.yaw, self.hint)
        self.lateral_true[mask], self.psi_true[mask] = lt[mask], pt[mask]
        self.clear_true[mask] = self.world.body_clearance(p.x, p.y, p.yaw)[mask]
        # A first frame, so the stack never starts empty.
        self.scans.reset(mask, self._capture(mask, delayed=False))

    def reset(self):
        self._reset_idx(np.ones(self.n, dtype=bool))
        self._record_pose()
        self._sense()
        return self._observe()

    # ---------------------------------------------------------------- camera

    def _capture(self, mask, delayed=True):
        """Render, corrupt and reduce one depth frame for the cars in `mask`."""
        rows = np.flatnonzero(mask)
        c = self.cam
        if delayed:
            head = (self.pose_head - 1) % self.pose_history.shape[1]
            idx = (head - self.cam_lat_steps[rows]) % self.pose_history.shape[1]
            pose = self.pose_history[rows, idx]
        else:
            p = self.plant
            pose = np.stack([p.x[rows], p.y[rows], p.yaw[rows]], 1)
        cam = self.camera
        x, y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
        ox = x + cam.x * np.cos(yaw)
        oy = y + cam.x * np.sin(yaw)
        angles = (yaw + c["yaw"][rows])[:, None] + cam.azimuth[None, :]
        r_h = self.world.raycast(ox, oy, angles, cam.scan_max + 2.0, rows)
        # Past the cast range there is still a bale somewhere: call it a far
        # wall, so the column reads "far" rather than "no data".
        r_h = np.where(np.isfinite(r_h), r_h, cam.scan_max + 5.0)
        pitch = c["pitch"][rows] + self.rng.standard_normal(len(rows)) * c[
            "pitch_jitter"
        ][rows]
        depth = cam.render(r_h, pitch, cam.z + c["height"][rows])
        if self.randomized:
            depth = corrupt(depth, cam, {k: v[rows] for k, v in c.items()}, self.rng)
        return cam.encode(cam.depth_to_scan(depth))

    def _camera_tick(self):
        c = self.cam
        c["timer"] += self.dt
        self.scans.tick(self.dt)
        fire = c["timer"] >= c["period"]
        c["timer"] = np.where(fire, c["timer"] - c["period"], c["timer"])
        if self.randomized:
            fire &= self.rng.random(self.n) >= c["drop"]
        if fire.any():
            self.scans.push(fire, self._capture(fire), c["latency"][fire])

    # ----------------------------------------------------------------- sense

    def _record_pose(self):
        p = self.plant
        self.pose_history[:, self.pose_head] = np.stack([p.x, p.y, p.yaw], axis=1)
        self.pose_head = (self.pose_head + 1) % self.pose_history.shape[1]

    def _sense(self, mask=None):
        """Sensed pose, yaw rate and achieved accel, once per control step.

        Unchanged from formulaOne -- including why it is split from
        `_observe` (a terminal step observes twice).
        """
        p = self.plant
        rows = np.arange(self.n)
        head = (self.pose_head - 1) % self.pose_history.shape[1]
        idx = (head - self.latency_steps) % self.pose_history.shape[1]
        delayed = self.pose_history[rows, idx]
        laps_travelled = self.distance / self.track.length
        drift = self.drift_rate * laps_travelled
        x = delayed[:, 0] + np.cos(self.drift_dir) * drift
        y = delayed[:, 1] + np.sin(self.drift_dir) * drift
        yaw = delayed[:, 2] + self.drift_yaw_rate * laps_travelled
        if self.randomized:
            x = x + self.rng.normal(0.0, 1.0, self.n) * self.noise_xy
            y = y + self.rng.normal(0.0, 1.0, self.n) * self.noise_xy
            yaw = yaw + self.rng.normal(0.0, 1.0, self.n) * self.noise_yaw
        step = np.arctan2(np.sin(yaw - self.sensed_yaw), np.cos(yaw - self.sensed_yaw))
        rate = np.clip(step / self.dt, -8.0, 8.0)
        rate = (1 - self.yaw_filter) * self.yaw_rate + self.yaw_filter * rate
        obs_speed = np.where(p.speed < 0.3, 0.0, p.speed)
        raw_accel = np.clip((obs_speed - self.prev_obs_speed) / self.dt, -20.0, 20.0)
        accel = (1 - self.accel_filter) * self.speed_rate + self.accel_filter * raw_accel
        keep = np.ones(self.n, dtype=bool) if mask is None else mask
        self.speed_rate = np.where(keep, accel, self.speed_rate)
        self.prev_obs_speed = np.where(keep, obs_speed, self.prev_obs_speed)
        self.sensed_x = np.where(keep, x, self.sensed_x)
        self.sensed_y = np.where(keep, y, self.sensed_y)
        self.sensed_yaw = np.where(keep, yaw, self.sensed_yaw)
        self.yaw_rate = np.where(keep, rate, self.yaw_rate)

    def _lap_state(self):
        lap_progress = np.clip(
            (self.distance - self.laps_done * self.track.length) / self.track.length,
            0.0,
            1.0,
        )
        return np.stack(
            [lap_progress, self.elapsed - self.lap_started_at, self.last_lap_time],
            axis=1,
        )

    def _privileged(self):
        p = self.plant
        c, s = np.cos(p.yaw), np.sin(p.yaw)

        def car_frame(dx, dy):
            return dx * c + dy * s, -dx * s + dy * c

        me = self.world.map_error(p.x, p.y)
        mex, mey = car_frame(me[:, 0], me[:, 1])
        pex, pey = car_frame(self.sensed_x - p.x, self.sensed_y - p.y)
        yaw_err = np.arctan2(np.sin(self.sensed_yaw - p.yaw), np.cos(self.sensed_yaw - p.yaw))
        nom = self.plant.nominal
        cols = [
            self.lateral_true / 0.5,
            np.sin(self.psi_true),
            np.cos(self.psi_true),
            np.clip(self.clear_true / 0.3, -1, 3),
            p.speed / 5.2,
            mex / 0.1,
            mey / 0.1,
            pex / 0.1,
            pey / 0.1,
            yaw_err / 0.05,
            p.mu,
            p.mass_scale,
            p.yaw_tau / 0.34,
            p.steer_tau / 0.05,
            p.dead_time / 0.19,
            p.throttle_delay / 0.06,
            p.motor_tau / 0.1,
            p.coast_f0 / nom["coast_f0"],
            p.steer_gain,
            p.steer_offset / 0.035,
            p.understeer / 0.02,
            p.steer_limit / 0.5,
            self.world.inflate / 0.02,
            self.stopping.astype(float),
            self.laps_done / self.laps,
            p.yaw_rate / 2.0,
            p.steer_angle / 0.5,
        ]
        return np.clip(np.stack(cols, 1), -10, 10)

    def _observe(self):
        p = self.plant
        speed = np.where(p.speed < 0.3, 0.0, p.speed)
        map_obs, self.frame = self.obs_builder.compute(
            self.sensed_x,
            self.sensed_y,
            self.sensed_yaw,
            speed,
            self.yaw_rate,
            self.speed_rate,
            self.prev_action,
            self.last_steer,
            self._lap_state(),
        )
        return np.concatenate(
            [map_obs, self.scans.features(), self._privileged()], 1
        ).astype(np.float32)

    # ------------------------------------------------------------------ step

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(self.n, 2)
        p = self.plant
        t = self.track

        steer_cmd, speed_cmd = scale_action(
            action,
            self.frame["v_cap"],
            self.frame["steer_ff"],
            self.residual,
            self.frame["v_floor"],
        )
        speed_cmd = np.where(self.stopping, 0.0, speed_cmd)
        steer_cmd, speed_cmd = p.push_command(steer_cmd, speed_cmd)
        steer_step = steer_cmd - self.last_steer
        steer_jerk = steer_step - self.last_steer_step
        v_cap_true = t.v_cap[self.hint]

        sub_dt = self.dt / self.substeps
        min_clear = np.full(self.n, 9.0)
        for i in range(self.substeps):
            p.substep(steer_cmd, speed_cmd, sub_dt)
            # Contact at 100 Hz: 5 cm of travel at full speed, and contact
            # here is a graze that builds over several centimetres.
            if i % 2 == 1:
                clear = self.world.body_clearance(p.x, p.y, p.yaw)
                min_clear = np.minimum(min_clear, clear)
        self.clear_true = clear

        self._record_pose()
        self._sense()
        self._camera_tick()
        prev_station = self.station
        self.hint, self.station, lateral, psi = self.world.frenet(
            p.x, p.y, p.yaw, self.hint
        )
        self.lateral_true, self.psi_true = lateral, psi
        advance = (self.station - prev_station + t.length / 2) % t.length - t.length / 2
        advance = np.clip(advance, -2.0, 2.0)
        self.distance += advance
        self.elapsed += self.dt

        crossed = np.floor(np.maximum(self.distance, 0.0) / t.length).astype(np.int32)
        lapped_mask = crossed > self.laps_done
        lap_time = self.elapsed - self.lap_started_at
        for i in np.flatnonzero(lapped_mask):
            j = min(int(self.laps_done[i]), self.laps - 1)
            self.lap_times[i, j] = lap_time[i]
        self.laps_done = np.maximum(self.laps_done, crossed)
        self.last_lap_time = np.where(lapped_mask, lap_time, self.last_lap_time)
        self.best_lap_time = np.where(
            lapped_mask, np.minimum(self.best_lap_time, lap_time), self.best_lap_time
        )
        self.lap_started_at = np.where(lapped_mask, self.elapsed, self.lap_started_at)

        crashed = min_clear <= 0.0
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
        stalled = (self.stalled_for >= self.stall_patience) & ~self.stopping
        timed_out = self.elapsed >= self.timeout

        r = self.cfg["reward"]
        lap_progress = np.clip(
            (self.distance - self.laps_done * t.length) / t.length, 0.0, 1.0
        )
        pace = np.where(
            self.last_lap_time > 0.0,
            self.last_lap_time * lap_progress - (self.elapsed - self.lap_started_at),
            0.0,
        )
        potential = float(r["lap_improve"]) * np.clip(
            pace, -float(r["lap_improve_cap"]), float(r["lap_improve_cap"])
        )
        quiet = lapped_mask | self.stopping
        lap_gain = np.where(quiet, 0.0, potential - self.pace_potential)
        self.pace_potential = potential

        reward, terms = self.reward.step(
            dt=self.dt,
            advance=advance,
            speed=p.speed,
            v_cap=v_cap_true,
            clearance=min_clear,
            lateral=lateral,
            psi=psi,
            steer_step=steer_step,
            steer_jerk=steer_jerk,
            lapped=lapped_mask.astype(float),
            finished=just_finished.astype(float),
            crashed=crashed.astype(float),
            stalled=stalled.astype(float),
            stopping=self.stopping.astype(float),
            stopped=stopped.astype(float),
            lap_gain=lap_gain,
        )
        self.last_terms = terms

        self.prev_action = np.clip(action, -1.0, 1.0)
        self.last_steer = steer_cmd
        self.last_steer_step = steer_step
        self.ep_return += reward
        self.ep_min_clearance = np.minimum(self.ep_min_clearance, min_clear)
        self.ep_overspeed = np.maximum(self.ep_overspeed, p.speed - v_cap_true)
        live = ~self.stopping
        self.ep_lateral_sum += np.abs(lateral) * live
        self.ep_lateral_n += live
        self.ep_lateral_max = np.maximum(self.ep_lateral_max, np.abs(lateral) * live)
        self.ep_gate_sum += self.reward.gate(lateral, psi) * live
        self.ep_steer_jerk_sum += steer_jerk**2
        self.ep_steps += 1

        terminated = crashed | stopped | stop_failed | stalled
        truncated = timed_out & ~terminated
        done = terminated | truncated

        obs = self._observe()
        info = [{} for _ in range(self.n)]
        if done.any():
            for i in np.flatnonzero(done):
                info[i] = self._episode_info(i, obs[i], stopped[i], crashed[i], stalled[i])
            self._reset_idx(done)
            self.pose_history[done] = np.stack(
                [p.x[done], p.y[done], p.yaw[done]], axis=1
            )[:, None, :]
            self._sense(done)
            obs = self._observe()
        return obs, reward, terminated, truncated, info

    def _episode_info(self, i, final_obs, stopped, crashed, stalled):
        rt = self.race_time[i]
        return {
            "terminal_observation": final_obs,
            "episode": {
                "r": float(self.ep_return[i]),
                "l": int(self.ep_steps[i]),
                "t": float(self.elapsed[i]),
            },
            "distance": float(self.distance[i]),
            "laps": int(self.laps_done[i]),
            "finished": bool(self.stopping[i]),
            "stopped": bool(stopped),
            "crashed": bool(crashed),
            "stalled": bool(stalled),
            "min_clearance": float(self.ep_min_clearance[i]),
            "max_overspeed": float(self.ep_overspeed[i]),
            "race_time": float(rt),
            "lap_time": float(rt / self.laps) if np.isfinite(rt) else float("nan"),
            "lap_times": self.lap_times[i].tolist(),
            "stop_distance": float(self.distance[i] - self.target_distance),
            "best_lap": float(self.best_lap_time[i])
            if np.isfinite(self.best_lap_time[i])
            else float("nan"),
            "last_lap": float(self.last_lap_time[i]),
            "mean_cte": float(self.ep_lateral_sum[i] / max(self.ep_lateral_n[i], 1)),
            "max_cte": float(self.ep_lateral_max[i]),
            "mean_gate": float(self.ep_gate_sum[i] / max(self.ep_lateral_n[i], 1)),
            "steer_jerk_rms": float(
                np.sqrt(self.ep_steer_jerk_sum[i] / max(self.ep_steps[i], 1))
            ),
        }

    # ----------------------------------------------------------------- state

    def scripted_action(self, driver):
        return driver.act(
            self.frame["station"],
            self.plant.speed,
            self.frame["v_cap"],
            self.frame["v_floor"],
        )

    def snapshot(self):
        p = self.plant
        return dict(
            x=p.x.copy(),
            y=p.y.copy(),
            yaw=p.yaw.copy(),
            speed=p.speed.copy(),
            station=self.station.copy(),
            distance=self.distance.copy(),
            elapsed=self.elapsed.copy(),
            v_cap=self.frame["v_cap"].copy(),
            lateral=self.lateral_true.copy(),
            psi=self.psi_true.copy(),
            clearance=self.clear_true.copy(),
            scan=self.scans.frames[:, 0].copy(),
        )


def make(config, n_envs=1, seed=0, deterministic=False, repo_root=None):
    from pathlib import Path

    root = repo_root or Path(__file__).resolve().parents[2]
    trk = track_mod.build(config, root)
    return FormulaTwoEnv(config, trk, n_envs, seed, deterministic)
