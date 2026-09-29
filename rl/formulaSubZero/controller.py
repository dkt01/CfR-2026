"""CasADi lateral MPC for the planned Speed Course line."""

from __future__ import annotations

import math
import time

import casadi as ca
import numpy as np

from depth_planner import DepthPlanner
from planner import SpeedCoursePlan


class FormulaSubZeroDriver:
    def __init__(self, track, cfg):
        self.track = track
        self.cfg = cfg
        self.plan = SpeedCoursePlan(track, cfg)
        self.depth = DepthPlanner(track, self.plan, cfg)
        self.pc = cfg["formula_sub_zero"]
        self.plant = cfg["plant"]
        self.n = int(self.pc["horizon"])
        self.dt = float(self.pc["horizon_dt"])
        self.delay_steps = int(self.pc["command_delay_steps"])
        if self.n <= self.delay_steps:
            raise ValueError("MPC horizon must exceed command delay")
        self._build()
        self.reset()

    def reset(self, n=1):
        self.warm = None
        self.failures = 0
        self.last_solve_ms = 0.0

    def _build(self):
        n, dt = self.n, self.dt
        opt = ca.Opti()
        lateral = opt.variable(n + 1)
        heading = opt.variable(n + 1)
        yaw_rate = opt.variable(n + 1)
        steer = opt.variable(n - self.delay_steps)
        initial = opt.parameter(3)
        speed = opt.parameter(n)
        kappa = opt.parameter(n)
        previous = opt.parameter()
        prior = opt.parameter(n)
        risk_room = opt.parameter(n)
        opt.subject_to(lateral[0] == initial[0])
        opt.subject_to(heading[0] == initial[1])
        opt.subject_to(yaw_rate[0] == initial[2])
        wheelbase = float(self.plant["wheelbase"])
        understeer = float(self.plant["understeer_gradient"])
        scrub = float(self.plant["tire_scrub"])
        commands = self.plant["steering_command_points"]
        angles = self.plant["steering_angle_points"]
        left_slope = float(angles[-1]) / float(commands[-1])
        right_slope = float(angles[0]) / float(commands[0])
        tau = float(self.plant["yaw_response_tau"])
        cost = 0
        for k in range(n):
            u = previous if k < self.delay_steps else steer[k - self.delay_steps]
            angle = ca.if_else(u >= 0, left_slope * u, right_slope * u)
            r_target = speed[k] * ca.tan(angle) / (
                (wheelbase + understeer * speed[k] ** 2) * scrub
            )
            opt.subject_to(lateral[k + 1] == lateral[k] + dt * speed[k] * ca.sin(heading[k]))
            opt.subject_to(heading[k + 1] == heading[k] + dt * (yaw_rate[k] - speed[k] * kappa[k]))
            opt.subject_to(yaw_rate[k + 1] == yaw_rate[k] + dt / tau * (r_target - yaw_rate[k]))
            w = 2.0 if k == n - 1 else 1.0
            cost += w * (
                float(self.pc["w_lateral"]) * lateral[k + 1] ** 2
                + float(self.pc["w_heading"]) * heading[k + 1] ** 2
            )
            excess = ca.fmax(ca.fabs(lateral[k + 1]) - risk_room[k], 0)
            cost += float(self.pc["w_risk"]) * excess**2
            if k >= self.delay_steps:
                last = previous if k == self.delay_steps else steer[k - self.delay_steps - 1]
                cost += float(self.pc["w_steer_change"]) * (u - last) ** 2
                cost += float(self.pc["w_prior"]) * (u - prior[k]) ** 2
        opt.subject_to(opt.bounded(-1.0, steer, 1.0))
        opt.minimize(cost)
        opt.solver(
            "ipopt",
            {"print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 35,
                "tol": 1e-3,
                "acceptable_tol": 1e-2,
            },
        )
        self.opt = opt
        self.state = (lateral, heading, yaw_rate, steer)
        self.params = (initial, speed, kappa, previous, prior, risk_room)

    def act_frame(self, frame, x, y, yaw, speed, measured_rate, last_steer, residual, raw_pose):
        station = float(frame["station"][0])
        cap = float(frame["v_cap"][0])
        floor = float(frame["v_floor"][0])
        prior = float(frame["steer_ff"][0])
        delay = float(self.plant["command_dead_time"])
        safe_path = self.depth.replan(station, (x, y, yaw), raw_pose)
        want = float(self.plan.at(station + speed * delay, self.plan.speed)) if safe_path else 0.0
        # Coast-only speed bound for a newly blocked corridor or the end of
        # the depth-verified horizon. The map still supplies hairpin lookahead.
        room = max(0.0, self.depth.blocked_distance - 0.5)
        decel = float(self.pc["coast_decel_mps2"])
        reaction = float(self.pc["depth_reaction_s"])
        visible_cap = max(0.0, math.sqrt((decel * reaction) ** 2 + 2 * decel * room)
                          - decel * reaction)
        want = min(cap, max(floor, want), visible_cap)
        rx, ry, rpsi, _ = self.depth.reference(np.array([station]))
        rx, ry, rpsi = float(rx[0]), float(ry[0]), float(rpsi[0])
        lateral = -(x - rx) * math.sin(rpsi) + (y - ry) * math.cos(rpsi)
        heading = math.atan2(math.sin(yaw - rpsi), math.cos(yaw - rpsi))
        speeds = self.plan.at(
            station + max(speed, 0.8) * self.dt * np.arange(self.n),
            self.plan.speed,
        )
        # Use measured speed at the first step, then the feasible profile.
        speeds = np.asarray(speeds, dtype=float)
        speeds[0] = max(speed, 0.1)
        _, _, _, kappa = self.depth.reference(
            station + max(speed, 0.8) * self.dt * np.arange(self.n)
        )
        speeds = np.minimum(speeds, visible_cap)
        speeds[0] = max(speed, 0.1)
        initial_p, speed_p, kappa_p, prev_p, prior_p, room_p = self.params
        self.opt.set_value(initial_p, [lateral, heading, measured_rate])
        self.opt.set_value(speed_p, speeds)
        self.opt.set_value(kappa_p, kappa)
        self.opt.set_value(prev_p, last_steer)
        self.opt.set_value(prior_p, np.full(self.n, prior))
        sample_s = station + max(speed, 0.8) * self.dt * np.arange(1, self.n + 1)
        room = (np.full(self.n, 0.10) if self.depth.map_consistent else
                np.clip(self.depth.clearance_at(sample_s) - 0.10, 0.02, 0.20))
        self.opt.set_value(room_p, room)
        if self.warm is not None:
            for var, val in zip(self.state, self.warm):
                self.opt.set_initial(var, val)
        else:
            self.opt.set_initial(self.state[3], np.full(self.n - self.delay_steps, prior))
        start = time.monotonic()
        try:
            sol = self.opt.solve()
            steer = float(sol.value(self.state[3][0]))
            trim = min(0.35, float(self.pc["max_steer_trim"])
                       + 1.5 * (float(np.max(np.abs(self.depth.offsets)))
                                if self.depth.offsets is not None else 0.0))
            steer = float(np.clip(steer, prior - trim, prior + trim))
            self.warm = tuple(sol.value(var) for var in self.state)
            self.failures = 0
        except RuntimeError:
            self.warm = None
            self.failures += 1
            steer = prior
            want *= 0.75 if self.failures <= 3 else 0.0
        if not safe_path:
            want = 0.0
        self.last_solve_ms = (time.monotonic() - start) * 1000.0
        action_steer = np.clip((steer - prior) / max(residual, 1e-6), -1.0, 1.0)
        throttle = 2.0 * np.clip((want - floor) / max(cap - floor, 1e-6), 0.0, 1.0) - 1.0
        return np.array([[action_steer, throttle]], dtype=float)
