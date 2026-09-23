"""Gymnasium environment for the lap-time objective on the speed course.

Built on `BaleFollowerEnv`, which already owns everything that is about
talking to Gazebo -- the ground-truth pose subscription, the /clock feed, the
point-cloud path, the teleport reset, the traction clamp -- and is left
untouched so the v1-v11 checkpoints keep running. What this class replaces is
everything that is about the *task*:

| | `BaleFollowerEnv` | `LapRacerEnv` |
|---|---|---|
| goal | metres down a corridor before a collision | laps, as fast as possible |
| progress | displacement projected on the car's own heading | signed arc length along the planned loop (`lap_track`) |
| clearance | min of a 110 deg ray fan from the car's CENTRE | gap between the car's footprint and the nearest bale's |
| steering action | an angle, clamped to the servo's slew | the servo's RATE, integrated -- the angle cannot jump by construction |
| stuck | truncates the episode after 5 s | enters a recovery mode the policy can see, and is expected to drive out of |
| start pose | +/-2 m around the SDF spawn | anywhere on the loop, sometimes deliberately wedged |
| episode | 60 s | 150 s or 3 laps, so a lap bonus is something the policy actually experiences |

The observation stays sensor-only: a ZED-shaped forward scan plus the car's
own speed, yaw rate, steering angle and recovery flag. The loop, the arc
length, the lap clock and the body clearance are all privileged and live in
the reward, the episode logic and the metrics -- so nothing here needs a map
or a global pose at deployment, exactly as before.
"""

from __future__ import annotations

import math
import time
from collections import deque

import gymnasium
import numpy as np
from geometry_msgs.msg import Twist

import bale_geometry
import zed_sim
from cfr_interfaces.msg import ArduinoStatus
from env import (
    GRAVITY,
    MAX_STEERING_ANGLE,
    WHEELBASE,
    BaleFollowerEnv,
    Pose2D,
    _wrap_to_pi,
)
from lap_reward import LapRewardConfig, compute_lap_reward
from lap_track import LapCounter, LapTrack


class LapRacerEnv(BaleFollowerEnv):
    def __init__(
        self,
        *args,
        lap_reward_config: LapRewardConfig | None = None,
        course_path_json: str | None = None,
        max_laps: int = 3,
        scan_history: int = 1,
        # Rolling window over which "the car is not going anywhere" is judged,
        # as NET DISPLACEMENT from where the car was at the start of it --
        # not the sum of per-step distances. Summing lets a car rock back and
        # forth in place and never register: measured at 16k steps of the
        # first run, the policy found exactly that, sitting at 0.0 m/s mean
        # for 995 steps with -0.1 m of progress while the detector saw metres
        # of "travel". Displacement is also what `path_racer._check_stuck`
        # uses, and it stays computable from odometry alone, so the flag can
        # still be in the observation.
        recovery_window_s: float = 2.0,
        recovery_distance: float = 0.4,
        # Forward travel that ends a recovery. Long enough that the car has
        # actually pulled out and pointed somewhere, short enough that it is
        # back on the clock quickly.
        recovery_exit_distance: float = 1.0,
        # Give up on the episode only after this much dead time -- five times
        # the recovery window, so a policy gets several attempts at backing
        # out before the episode is written off.
        stuck_window_s: float = 10.0,
        stuck_distance: float = 1.0,
        # Fraction of episodes that start from a deliberately awkward pose:
        # angled across the corridor, nose near a bale. Recovery cannot be
        # learned from states the car only reaches by already being bad.
        wedged_start_prob: float = 0.15,
        # Where the analytic scan is cast from, in the chassis frame. The
        # simulated ZED sits at <pose>0.315 0 0.20</pose>
        # (sensors_world.py's SENSORS_CAMERA), so a ray fan cast from the
        # chassis origin measures a DIFFERENT scene from the one the camera
        # returns: 0.247 m of mean error against the rendered cloud, falling
        # to 0.113 m once the fan is moved to the lens. Without this, a
        # policy trained on `analytic` and deployed on `cloud` (or on the
        # real car) reads every range with a 0.315 m shift.
        scan_origin_x: float = 0.315,
        # The camera's frame rate, which is not the control rate. The ZED is
        # configured at 15 Hz (sensors_world.py) and measured at 13.7, against
        # a 20 Hz control loop -- so on the real car roughly a third of the
        # steps see a REPEATED scan, with a mean age of 55 ms (11 cm at
        # 2 m/s). An analytic scan recomputed fresh every step is a sensor
        # nobody owns; this holds the previous one between camera frames so
        # the policy trains against the staleness it will be deployed with.
        # 0 recomputes every step.
        scan_update_hz: float = 15.0,
        # Where the observation's speed channel comes from. "status" is the
        # tachometer the car actually has -- via /arduino_bridge/status, the
        # same topic in sim and on the vehicle -- including its ~0.3 m/s
        # blind spot and the ~27% of samples that read exactly zero. "truth"
        # differences the ground-truth pose, which is what the first runs
        # trained on and which does not exist outside the simulator.
        speed_source: str = "status",
        status_topic: str = "/arduino_bridge/status",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            stuck_window_s=stuck_window_s,
            stuck_distance=stuck_distance,
            **kwargs,
        )
        self.track = LapTrack(course_path_json) if course_path_json else LapTrack()
        # The plan is a loop with no inherent direction; the SDF spawn heading
        # is what says which way round the course is driven.
        if self.track.orient_to(*self.spawn_pose):
            print(
                "lap_env: planned loop reversed to match the spawn heading", flush=True
            )
        self.counter = LapCounter(self.track)
        self.lap_reward_config = lap_reward_config or LapRewardConfig()
        self.max_laps = max_laps
        self.scan_history = max(0, scan_history)
        self.recovery_window_s = recovery_window_s
        self.recovery_distance = recovery_distance
        self.recovery_exit_distance = recovery_exit_distance
        self.wedged_start_prob = wedged_start_prob
        self.scan_origin_x = scan_origin_x
        self.scan_update_hz = scan_update_hz
        self.speed_source = speed_source
        self._status_speed = 0.0
        self._status_stamp = -1.0
        self._status_misses = 0
        self._cached_scan: np.ndarray | None = None
        self._cached_scan_time = -1.0
        self._next_frame_time = -1.0
        if speed_source == "status":
            self._node.create_subscription(
                ArduinoStatus, status_topic, self._on_status, 10
            )

        # Steering is integrated from a rate command, so the servo's slew is
        # a property of the action space rather than a clamp applied after
        # the fact. In fraction-of-full-lock per second.
        self.steer_rate_fraction_per_s = self.max_steering_rate / MAX_STEERING_ANGLE

        obs_dim = self.num_lidar_bins * (1 + self.scan_history) + 4
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._scan_history: deque[np.ndarray] = deque(maxlen=max(1, self.scan_history))
        # Positions, newest last, long enough to serve both windows.
        self._recovery_steps = max(1, round(self.recovery_window_s * self.control_hz))
        self._stuck_steps = max(1, round(self.stuck_window_s * self.control_hz))
        self._pose_window: deque[tuple[float, float]] = deque(
            maxlen=max(self._recovery_steps, self._stuck_steps) + 1
        )
        self._path_index = 0
        self._recovering = False
        self._recovery_forward = 0.0
        self._prev_heading_error = 0.0
        self._prev_lateral = 0.0
        self._prev_steer_rate = 0.0
        self._episode_clearance = math.inf
        # How much SIMULATION time a control step actually covers. The step
        # loop sleeps on the wall clock, so this equals 1/control_hz only
        # when the simulator runs at exactly real time -- measured, it runs
        # at 1.19x without the camera (60 ms of sim per 50 ms step) and at
        # ~0.63x with it. The servo slew and the traction clamp are limits
        # per SECOND, so applying them per step against the nominal period
        # lets the policy command 1.6x the servo's real rate the moment the
        # ZED is switched on -- the same class of train/deploy mismatch that
        # cost v5 two thirds of its distance, hidden inside the real-time
        # factor. An EMA of the measured period is used instead.
        self._dt_estimate = 1.0 / self.control_hz
        self._failed_teleports = 0

    def _on_status(self, msg: ArduinoStatus) -> None:
        self._status_speed = float(msg.speed)
        self._status_stamp = self._sim_time

    def _observed_speed(self, true_speed: float) -> float:
        """The speed channel the POLICY sees, as opposed to the true one."""
        if self.speed_source != "status":
            return true_speed
        if self._status_stamp < 0.0:
            self._status_misses += 1
            if self._status_misses == 1:
                print(
                    "WARNING: speed_source=status but no ArduinoStatus yet; "
                    "using the true speed",
                    flush=True,
                )
            elif self._status_misses >= 200:
                raise RuntimeError(
                    "speed_source=status but /arduino_bridge/status has "
                    "produced nothing in 200 steps"
                )
            return true_speed
        return self._status_speed

    def _apply_traction(
        self, speed: float, steer_fraction: float, dt: float | None = None
    ) -> float:
        """As `BaleFollowerEnv._apply_traction`, but on the measured period."""
        dt = self._dt_estimate if dt is None else dt
        a_max = self.traction * GRAVITY
        speed = self._cmd_speed + min(
            max(speed - self._cmd_speed, -a_max * dt), a_max * dt
        )
        tan_delta = abs(math.tan(steer_fraction * MAX_STEERING_ANGLE))
        if tan_delta > 1e-6:
            grip_speed = math.sqrt(a_max * WHEELBASE / tan_delta)
            speed = min(max(speed, -grip_speed), grip_speed)
        self._cmd_speed = max(-self.reverse_speed, min(speed, self.max_speed))
        return self._cmd_speed

    # ------------------------------------------------------------- actions

    def decode_action(self, action: np.ndarray) -> tuple[float, float]:
        """(target speed m/s, steering RATE as a fraction of the servo limit).

        Note the second element is a rate, not an angle -- `encode_action`
        and anything that filters commands (the CasADi smoother) must be
        aware of that. The angle itself is state, integrated in `step`.
        """
        throttle = (float(np.clip(action[0], -1.0, 1.0)) + 1.0) / 2.0
        speed = -self.reverse_speed + throttle * (self.max_speed + self.reverse_speed)
        return speed, float(np.clip(action[1], -1.0, 1.0))

    def encode_action(self, speed: float, steer_rate_fraction: float) -> np.ndarray:
        throttle = (speed + self.reverse_speed) / (self.max_speed + self.reverse_speed)
        return np.array([2.0 * throttle - 1.0, steer_rate_fraction], dtype=np.float32)

    # -------------------------------------------------------- observations

    def _observe(
        self, scan: np.ndarray, signed_speed: float, yaw_rate: float
    ) -> np.ndarray:
        # `scan` arrives already corrupted, once per CAMERA frame -- see
        # `_scan_at`.
        normalized = (scan / self.lidar_max_range).astype(np.float32)
        # Newest first, then one frame back per `scan_history`. 50 ms of
        # history at 20 Hz is what makes closing rate on a bale observable;
        # at the start of an episode there is none, so it repeats the current
        # frame rather than feeding the policy zeros (a wall at zero range).
        while len(self._scan_history) < self.scan_history:
            self._scan_history.append(normalized)
        frames = [normalized] + list(self._scan_history)[: self.scan_history]
        if self.scan_history:
            self._scan_history.appendleft(normalized)

        speed_norm = np.clip(
            (signed_speed + self.reverse_speed) / (self.max_speed + self.reverse_speed),
            0.0,
            1.0,
        )
        yaw_norm = np.clip((yaw_rate + 3.0) / 6.0, 0.0, 1.0)
        steer_norm = (self._cmd_steer_fraction + 1.0) / 2.0
        return np.concatenate(
            frames
            + [
                np.array(
                    [
                        speed_norm,
                        yaw_norm,
                        steer_norm,
                        1.0 if self._recovering else 0.0,
                    ],
                    dtype=np.float32,
                )
            ]
        ).astype(np.float32)

    def _scan_at(self, pose: Pose2D) -> np.ndarray:
        # Between camera frames the policy sees the previous scan, not a new
        # one -- the camera is slower than the control loop on the car and in
        # the simulator alike.
        # A frame DEADLINE, not "time since the last frame". The camera runs
        # free at its own rate and the loop samples whatever is latest; a
        # since-last rule against a clock quantised to the 50 ms control step
        # can only fire every other step, which turns a 15 Hz camera into a
        # 10 Hz one (measured: 50% repeated frames where 25% was correct).
        # Advancing a deadline by the camera period reproduces the real
        # 1-step/2-step mix.
        #
        # Keyed on the POSE's simulation stamp rather than `self._sim_time`:
        # the pose carries the clock its own motion was measured against, and
        # it is the one signal guaranteed to advance wherever this env runs.
        if (
            self.scan_update_hz > 0.0
            and self._cached_scan is not None
            and pose.sim_stamp < self._next_frame_time
        ):
            return self._cached_scan
        scan = self._cloud_scan() if self.scan_source == "cloud" else None
        if scan is None:
            if self.scan_source == "cloud":
                self._cloud_misses += 1
                if self._cloud_misses == 1:
                    print(
                        "WARNING: scan_source=cloud but no cloud yet; "
                        "using analytic scan",
                        flush=True,
                    )
                elif self._cloud_misses >= 200:
                    raise RuntimeError(
                        f"scan_source=cloud but {self.cloud_topic} has produced "
                        "nothing in 200 steps -- launch with CFR_SENSORS=1"
                    )
            # From the camera's mount, not the chassis origin -- see
            # `scan_origin_x`.
            scan = bale_geometry.lidar_scan(
                self.bales,
                pose.x + self.scan_origin_x * math.cos(pose.yaw),
                pose.y + self.scan_origin_x * math.sin(pose.yaw),
                pose.yaw,
                self.num_lidar_bins,
                self.lidar_fov_deg,
                self.lidar_max_range,
            )
        else:
            self._cloud_hits += 1
            self._cloud_misses = 0
        # Corrupted here, once per camera frame, rather than once per
        # control step: a repeated frame carries the SAME noise and dropouts.
        # Re-rolling them on every step would let the policy average the
        # sensor's error away across the repeats, which no real camera
        # allows -- and it is the noise, not the geometry, that the averaging
        # would be cheating on.
        scan = zed_sim.apply(
            scan.copy(), self.zed_config, self.lidar_max_range, self._rng
        )
        self._cached_scan = scan
        self._cached_scan_time = pose.sim_stamp
        if self.scan_update_hz > 0.0:
            period = 1.0 / self.scan_update_hz
            if self._next_frame_time < 0.0:
                self._next_frame_time = pose.sim_stamp
            # Catch up rather than drift, if the sim skipped a beat.
            while self._next_frame_time <= pose.sim_stamp:
                self._next_frame_time += period
        return scan

    # ------------------------------------------------------------- spawning

    def _pick_start_pose(self) -> tuple[float, float, float]:
        """Anywhere on the loop, occasionally wedged across the corridor.

        Uniform over arc length is the point: with the old +/-2 m jitter
        around the SDF spawn, every state past the first hairpin is reachable
        only by a policy that can already drive to it, so the back half of the
        course gets a fraction of the samples the first corner does.
        """
        if not self.randomize_start:
            x, y, yaw = self.spawn_pose
            return x, y, yaw

        wedged = self._rng.random() < self.wedged_start_prob
        for _ in range(40):
            index = int(self._rng.integers(0, len(self.track.x)))
            if wedged:
                lateral = float(self._rng.choice([-1.0, 1.0])) * self._rng.uniform(
                    0.12, 0.22
                )
                heading = float(self._rng.choice([-1.0, 1.0])) * self._rng.uniform(
                    0.9, 1.8
                )
            else:
                lateral = float(self._rng.uniform(-0.12, 0.12))
                heading = float(self._rng.uniform(-0.25, 0.25))
            candidate = self.track.pose_at(index, lateral, heading)
            if bale_geometry.check_collision(self.bales, *candidate):
                continue
            if bale_geometry.body_clearance(self.bales, *candidate) < 0.06:
                continue
            return candidate
        return self.spawn_pose

    def _net_displacement(self, steps: int) -> float | None:
        """How far the car is from where it was `steps` ago, or None if the
        window is not full yet."""
        if len(self._pose_window) <= steps:
            return None
        x0, y0 = self._pose_window[-1 - steps]
        x1, y1 = self._pose_window[-1]
        return math.hypot(x1 - x0, y1 - y0)

    # --------------------------------------------------------------- episode

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Not super().reset(): the parent's reset builds the corridor
        # observation and resets state this class does not use.
        gymnasium.Env.reset(self, seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._settle(0.3)
        x, y, yaw = self._pick_start_pose()
        try:
            self._teleport(x, y, math.degrees(yaw))
        except RuntimeError as error:
            # `gz service` times out under load every 20-30 minutes, and the
            # parent env treats that as fatal: three chunks of the first
            # stage-1 run died this way, each losing the steps since its last
            # checkpoint and the eval it was about to run. A failed teleport
            # is not a broken simulator, it is one missed reposition -- so
            # the episode starts from wherever the car already is. The start
            # distribution degrades for one episode; the run survives.
            self._failed_teleports += 1
            print(
                f"reset: {error} -- starting in place "
                f"({self._failed_teleports} so far)",
                flush=True,
            )
        self._settle(0.3)

        pose = self._wait_for_pose(since=time.monotonic())

        self._episode_step = 0
        self._episode_time = 0.0
        self._prev_pose = pose
        self._cmd_speed = 0.0
        self._cmd_steer_fraction = 0.0
        self._scan_history.clear()
        self._pose_window.clear()
        self._cached_scan = None
        self._cached_scan_time = -1.0
        self._next_frame_time = -1.0
        self._recovering = False
        self._recovery_forward = 0.0
        self._prev_steer_rate = 0.0
        self._episode_clearance = math.inf
        self._dt_estimate = 1.0 / self.control_hz

        projection = self.track.project(pose.x, pose.y, pose.yaw)
        self._path_index = projection.index
        self._prev_heading_error = projection.heading_error
        self._prev_lateral = projection.lateral
        self.counter.reset(projection.s)

        scan = self._scan_at(pose)
        return self._observe(scan, 0.0, 0.0), {}

    def step(self, action: np.ndarray):
        step_dt = self._dt_estimate
        target_speed, steer_rate_fraction = self.decode_action(action)

        # Integrate the servo. The angle cannot move faster than the servo,
        # because the only thing the policy can ask for is a rate.
        max_delta = self.steer_rate_fraction_per_s * step_dt
        previous_steer = self._cmd_steer_fraction
        self._cmd_steer_fraction = float(
            np.clip(previous_steer + steer_rate_fraction * max_delta, -1.0, 1.0)
        )
        # What the servo actually did, which is what the smoothness term
        # charges for: a rate command into a saturated lock costs nothing.
        realized_rate = (self._cmd_steer_fraction - previous_steer) / max_delta

        speed = self._apply_traction(target_speed, self._cmd_steer_fraction, step_dt)
        angular_z = 0.0
        if abs(speed) > 1e-3:
            angular_z = (speed / WHEELBASE) * math.tan(
                self._cmd_steer_fraction * MAX_STEERING_ANGLE
            )
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = angular_z
        self._cmd_pub.publish(twist)

        before = time.monotonic()
        self._advance_sim()
        pose = self._wait_for_pose(since=before)

        dt = pose.sim_stamp - self._prev_pose.sim_stamp
        if not (1e-4 < dt < 1.0):
            dt = step_dt
        # Smoothed, because a single long step (a dropped frame, a GC pause)
        # should not slew the actuator limits with it.
        self._dt_estimate = float(
            np.clip(0.9 * self._dt_estimate + 0.1 * dt, 0.005, 0.25)
        )
        forward = (pose.x - self._prev_pose.x) * math.cos(self._prev_pose.yaw) + (
            pose.y - self._prev_pose.y
        ) * math.sin(self._prev_pose.yaw)
        signed_speed = forward / dt
        yaw_rate = _wrap_to_pi(pose.yaw - self._prev_pose.yaw) / dt

        collided = bale_geometry.check_collision(self.bales, pose.x, pose.y, pose.yaw)
        clearance = bale_geometry.body_clearance(self.bales, pose.x, pose.y, pose.yaw)
        self._episode_clearance = min(self._episode_clearance, clearance)

        projection = self.track.project(pose.x, pose.y, pose.yaw, hint=self._path_index)
        self._path_index = projection.index
        self._episode_time += dt
        delta_s, lap_time = self.counter.update(projection.s, self._episode_time)

        # Recovery: entered on net displacement over the window (observable
        # from odometry, so the policy can be told), left on forward travel
        # since entering.
        self._pose_window.append((pose.x, pose.y))
        if self._recovering:
            self._recovery_forward += max(0.0, forward)
            if self._recovery_forward >= self.recovery_exit_distance:
                self._recovering = False
        elif self._net_displacement(self._recovery_steps) is not None and (
            self._net_displacement(self._recovery_steps) < self.recovery_distance
        ):
            self._recovering = True
            self._recovery_forward = 0.0

        result = compute_lap_reward(
            self.lap_reward_config,
            delta_s=delta_s,
            dt=dt,
            speed=signed_speed,
            clearance=clearance,
            steer_rate_fraction=realized_rate,
            prev_steer_rate_fraction=self._prev_steer_rate,
            steer_fraction=self._cmd_steer_fraction,
            prev_steer_fraction=previous_steer,
            collided=collided,
            recovering=self._recovering,
            heading_error=projection.heading_error,
            prev_heading_error=self._prev_heading_error,
            lateral=projection.lateral,
            prev_lateral=self._prev_lateral,
            lap_time=lap_time,
            best_lap_time=self.counter.best_lap_time,
        )

        self._prev_pose = pose
        self._prev_heading_error = projection.heading_error
        self._prev_lateral = projection.lateral
        self._prev_steer_rate = realized_rate
        self._episode_step += 1

        net = self._net_displacement(self._stuck_steps)
        stuck = net is not None and net < self.stuck_distance

        terminated = collided
        truncated = (
            self._episode_time >= self.episode_time_limit_s
            or stuck
            or self.counter.laps >= self.max_laps
        )

        info = {
            "collided": collided,
            "stuck": stuck,
            "recovering": self._recovering,
            "progress_distance": delta_s,
            "s_progress": self.counter.travelled,
            "laps": self.counter.laps,
            "lap_time": lap_time,
            "lap_times": list(self.counter.lap_times),
            "min_clearance": clearance,
            "episode_min_clearance": self._episode_clearance,
            "failed_teleports": self._failed_teleports,
            "speed": signed_speed,
            "observed_speed": self._observed_speed(signed_speed),
            "steer_rate": abs(realized_rate) * self.max_steering_rate,
            "heading_error": projection.heading_error,
            "lateral": projection.lateral,
            "reward_terms": {
                "progress": result.progress,
                "time": result.time,
                "stall": result.stall,
                "clearance": result.clearance,
                "touch": result.touch,
                "steering": result.steering,
                "align": result.align,
                "shaping": result.shaping,
                "lap_bonus": result.lap_bonus,
            },
        }
        observation = self._observe(
            self._scan_at(pose), self._observed_speed(signed_speed), yaw_rate
        )
        return observation, result.total, terminated, truncated, info
