"""Batched model of the whole actuation chain, from DriveCommand to pose.

This is a transcription of what is already on the car, not a new model:

    cfr_arduino_bridge/src/sim_vehicle_node.cpp   plant response
    cfr_arduino_bridge/config/arduino_bridge.yaml bridge-side limits
    cfr_arduino_bridge/config/vehicle.yaml        measured constants

It exists separately from Gazebo because training needs ~10^7 steps and
Gazebo delivers ~10 per wall-clock second.  Vectorised over B cars in numpy it
delivers ~10^5.  The price of that is that this file has to stay honest, so
every term below names the thing on the car it stands for, and
`selftest.py` checks the two agree.

Three properties dominate how the policy must drive, and all three are here:

  * NO BRAKES.  brake_decel is 0.0 and must stay 0.0.  Slowing down is coast
    drag alone, m dv/dt = -(2.18 + 0.47 v), which is 9.3 m of track to go from
    5.2 to 2.5 m/s.  Every hairpin has to be set up nearly ten metres early.
  * THE BRIDGE SLEWS THE TARGET at 2.0 m/s^2, which is below the plant's own
    3.0 m/s^2 -- so acceleration is bridge-limited, not grip-limited.
  * STEERING IS ASYMMETRIC by ~34% at matched command, and the servo lags.
    A policy trained on a symmetric model steers one way into the bales.
  * THE SIMULATED CAR SCRUBS ~10% WIDER than a bare kinematic bicycle predicts
    (measure_turn_radius.py, 2026-09-21: teleport to open ground, drive a
    constant command, fit a circle).  Flat across both steering directions and
    across 1.5-3.0 m/s, so it is a fixed geometric loss -- Gazebo's tire
    friction cone letting go a little at any real lock -- not an
    understeer_gradient effect, which would grow with v^2.  `tire_scrub` is
    that number, applied as a second multiplier on the effective wheelbase
    alongside the v^2 term, and it is why the trained policy still beached in
    Gazebo after the steering-authority bug (saturated left lock, clamped by a
    stale `guess` in vehicle.yaml) was fixed: THAT bug was structural and
    truncated a third of the lock outright, but this uniform 1.10x was always
    going to be there underneath it, and plant.py had never modelled it.
"""

from __future__ import annotations

import numpy as np

from randomization import draw


class Plant:
    """B independent cars, stepped together."""

    def __init__(self, cfg: dict, n: int, rng: np.random.Generator):
        self.cfg = cfg
        self.n = n
        self.rng = rng
        p = cfg["plant"]
        self.max_speed = float(p["max_speed"])
        self.brake_decel = float(p["brake_decel"])
        self.wheelbase = float(p["wheelbase"])
        self.steer_pts = np.asarray(p["steering_command_points"], dtype=float)
        self.steer_ang = np.asarray(p["steering_angle_points"], dtype=float)
        # steering_slew stays a scalar: it saturates on 0.1% of control ticks
        # even at the slowest servo the 2075R datasheet allows, so randomising
        # it would add a parameter the policy cannot observe a consequence of.
        # steering_tau is the opposite - see `nominal` below.
        self.steering_slew = float(p["steering_slew"])
        self.steering_limit_nominal = max(
            abs(self.steer_ang[0]), abs(self.steer_ang[-1])
        )
        self.nominal = dict(
            dead_time=float(p["command_dead_time"]),
            coast_f0=float(p["coast_f0"]) / float(p["mass"]),
            coast_f1=float(p["coast_f1"]) / float(p["mass"]),
            accel=float(p["max_accel"]),
            slew=float(p["speed_slew_rate"]),
            understeer=float(p["understeer_gradient"]),
            tire_scrub=float(p["tire_scrub"]),
            yaw_tau=float(p["yaw_response_tau"]),
            steer_tau=float(p["steering_tau"]),
        )
        self.gravity = float(p["gravity"])
        # Command history is kept at substep resolution so the dead time is
        # applied where it happens -- between the Jetson and the wheels --
        # rather than rounded to a control tick.
        self.dt_sub = 1.0 / (float(cfg["env"]["control_hz"]) * cfg["env"]["substeps"])
        self.hist_len = int(np.ceil(0.45 / self.dt_sub)) + 2
        self.history = np.zeros((n, self.hist_len, 2))
        self.head = 0

    # ------------------------------------------------------------ parameters

    def sample_parameters(self, mask: np.ndarray, scale: float = 1.0):
        """Draw a fresh car for every env in `mask`.

        v12's draws, plus the four formulaTwo adds: ground friction, mass,
        motor lag and a throttle-only delay.  Mass is applied where it acts
        physically -- drag deceleration and acceleration go as 1/m -- rather
        than as one more independent scale.
        """
        r = self.cfg["randomize"]
        k = int(mask.sum())
        if k == 0:
            return
        on = bool(r["enabled"])

        def d(name):
            return draw(self.rng, r[name], k, scale, on)

        nom = self.nominal
        mass = d("mass_scale")
        self.mass_scale[mask] = mass
        self.dead_time[mask] = d("dead_time")
        self.throttle_delay[mask] = d("throttle_delay")
        # NOT scaled by mass: yaw_tau_scale is already fitted to the measured
        # per-trace spread (0.19-0.59 s), and stacking mass on top reaches
        # 0.63 s -- past anything measured, the over-wide-range trap.
        self.yaw_tau[mask] = nom["yaw_tau"] * d("yaw_tau_scale")
        self.steer_tau[mask] = d("steering_tau")
        coast = d("coast_scale") / mass
        self.coast_f0[mask] = nom["coast_f0"] * coast
        self.coast_f1[mask] = nom["coast_f1"] * coast
        self.accel[mask] = nom["accel"] * d("accel_scale") / mass
        self.slew[mask] = nom["slew"] * d("slew_scale")
        self.understeer[mask] = d("understeer")
        self.tire_scrub[mask] = nom["tire_scrub"] * d("scrub_scale")
        self.steer_gain[mask] = d("steer_gain")
        self.steer_asym[mask] = d("steer_asymmetry")
        self.steer_offset[mask] = d("steer_offset")
        self.steer_limit[mask] = d("steer_limit")
        self.dropout[mask] = d("action_dropout")
        self.mu[mask] = d("mu")
        self.motor_tau[mask] = d("motor_tau")

    def reset(self, mask, x, y, yaw, speed, scale=1.0):
        """Place the cars in `mask` and give them a fresh plant."""
        if not hasattr(self, "dead_time"):
            z = np.zeros(self.n)
            for name in (
                "dead_time",
                "coast_f0",
                "coast_f1",
                "accel",
                "slew",
                "understeer",
                "tire_scrub",
                "steer_gain",
                "steer_asym",
                "steer_offset",
                "steer_limit",
                "dropout",
                "yaw_tau",
                "steer_tau",
                "mass_scale",
                "throttle_delay",
                "mu",
                "motor_tau",
            ):
                setattr(self, name, z.copy())
            self.yaw_rate = z.copy()
            self.x, self.y, self.yaw = z.copy(), z.copy(), z.copy()
            self.speed, self.target, self.steer_angle = z.copy(), z.copy(), z.copy()
            self.last_command = np.zeros((self.n, 2))
            self.motor_target = z.copy()
        self.sample_parameters(mask, scale)
        self.x[mask], self.y[mask], self.yaw[mask] = x, y, yaw
        self.speed[mask] = speed
        self.target[mask] = speed
        self.motor_target[mask] = speed
        self.steer_angle[mask] = 0.0
        self.yaw_rate[mask] = 0.0
        self.last_command[mask] = 0.0
        self.history[mask] = 0.0
        # A car dropped in mid-course is already moving, so seed its command
        # history with its own speed rather than with a standstill it would
        # have to accelerate out of.
        self.history[mask, :, 1] = speed[:, None]

    # ----------------------------------------------------------------- steer

    def steering_angle(self, command):
        """Measured command -> effective angle, with per-car trim and gain.

        `steer_asym` scales only the left half of the table, which is the way
        the real asymmetry presents: the servo's two directions do not share a
        linkage ratio, so a policy must not assume |left| == |right|.
        """
        c = np.clip(command + self.steer_offset, -1.0, 1.0)
        angle = np.interp(c, self.steer_pts, self.steer_ang)
        angle = angle * self.steer_gain
        angle = np.where(angle > 0, angle * self.steer_asym, angle)
        # Whatever the table says, the linkage stops somewhere.
        return np.clip(angle, -self.steer_limit, self.steer_limit)

    # ------------------------------------------------------------------ step

    def push_command(self, steer, speed):
        """Record a command as issued now; it reaches the wheels later."""
        repeat = self.rng.random(self.n) < self.dropout
        steer = np.where(repeat, self.last_command[:, 0], steer)
        speed = np.where(repeat, self.last_command[:, 1], speed)
        self.last_command[:, 0] = steer
        self.last_command[:, 1] = speed
        return steer, speed

    def substep(self, steer_cmd, speed_cmd, dt):
        """Advance every car by `dt` under the command issued `dead_time` ago."""
        self.history[:, self.head, 0] = steer_cmd
        self.history[:, self.head, 1] = speed_cmd
        delay = np.clip((self.dead_time / dt).astype(np.int32), 0, self.hist_len - 1)
        rows = np.arange(self.n)
        idx = (self.head - delay) % self.hist_len
        eff_steer = self.history[rows, idx, 0]
        # The throttle path carries its own extra delay (ESC arming, motor
        # timing) on top of the shared Jetson -> Arduino dead time.
        t_delay = np.clip(
            ((self.dead_time + self.throttle_delay) / dt).astype(np.int32),
            0,
            self.hist_len - 1,
        )
        eff_speed = self.history[rows, (self.head - t_delay) % self.hist_len, 1]
        self.head = (self.head + 1) % self.hist_len

        # Bridge: clamp, then slew the TARGET.  This is why the car
        # accelerates at 2.0 m/s^2 and not at the plant's 3.0.
        want = np.clip(eff_speed, 0.0, self.max_speed)
        step = self.slew * dt
        self.target += np.clip(want - self.target, -step, step)

        # Motor: a first-order lag between the slewed target and what the
        # drivetrain actually chases.
        self.motor_target += dt / (self.motor_tau + dt) * (
            self.target - self.motor_target
        )
        # Plant: thrust up, coast down.  Asymmetric by measurement -- see
        # ApproachTarget in sim_vehicle_node.cpp.
        delta = self.motor_target - self.speed
        speeding_up = self.motor_target > self.speed
        coast = self.coast_f0 + self.coast_f1 * np.abs(self.speed) + self.brake_decel
        limit = np.where(speeding_up, self.accel, coast)
        self.speed += np.clip(delta, -limit * dt, limit * dt)
        self.speed = np.clip(self.speed, 0.0, self.max_speed)

        # Servo: rate limit, then first-order lag, on the ANGLE.
        want_angle = self.steering_angle(eff_steer)
        slew = self.steering_slew * dt
        reachable = self.steer_angle + np.clip(
            want_angle - self.steer_angle, -slew, slew
        )
        alpha = dt / (self.steer_tau + dt)
        self.steer_angle += alpha * (reachable - self.steer_angle)

        # Bicycle with an understeer gradient, R = (L + K v^2) / tan(delta),
        # times tire_scrub for the flat ~1.10x the simulated car turns wider
        # than that kinematic radius (measured; see the module docstring).
        # R_gazebo = tire_scrub * (L + K v^2) / tan(delta), so tire_scrub
        # multiplies the effective wheelbase exactly the way K v^2 does --
        # it is a second, speed-independent term in the same denominator.
        eff_wheelbase = (
            self.wheelbase + self.understeer * self.speed**2
        ) * self.tire_scrub
        kinematic_rate = self.speed * np.tan(self.steer_angle) / eff_wheelbase
        # GRIP.  The tyres carry at most mu*g of lateral acceleration; past
        # that the car slides and the yaw rate the wheels ask for is not the
        # one it gets.  Never binds on the nominal car (the course needs
        # 0.43 g), binds on a slippery draw driven too fast into a hairpin.
        grip = self.mu * self.gravity / np.maximum(self.speed, 0.1)
        kinematic_rate = np.clip(kinematic_rate, -grip, grip)
        # THE CHASSIS DOES NOT ADOPT THAT YAW RATE THE INSTANT THE WHEELS ARE
        # POINTED.  Measured (measure_step_steer.py, fitted by fit_yaw_lag.py):
        # after a step of the steering command Gazebo reaches half its steady
        # yaw rate 0.2-0.4 s later than a bare kinematic bicycle does, at every
        # speed from 1.5 to 4.5 m/s and in both directions.  Speed-independent,
        # so it is not tire relaxation length (sigma/v); it is the yaw inertia
        # and tire force build-up of a car that has to be pushed into rotating.
        #
        # This is the term that was missing while both the scripted baseline
        # and two trained policies beached ENTERING a hairpin: steady-state
        # radius was already modelled (tire_scrub), so the model cornered
        # correctly once settled and turned in roughly a metre of travel too
        # early -- which on a 0.92 m corridor is the whole corridor.
        beta = dt / (self.yaw_tau + dt)
        self.yaw_rate += beta * (kinematic_rate - self.yaw_rate)
        yaw_rate = self.yaw_rate
        self.yaw += yaw_rate * dt
        self.x += self.speed * np.cos(self.yaw) * dt
        self.y += self.speed * np.sin(self.yaw) * dt
        return yaw_rate
