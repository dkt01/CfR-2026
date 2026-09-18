"""CasADi MPC path tracker for the course racer.

Pure pursuit cuts corners: chasing a lookahead point on a curving path pulls
the car 0.2-0.4 m inside the planned line, which in a 0.95 m corridor is the
difference between the racing line and a wall (measured in sim -- see
REPORT.md). MPC fixes that structurally: it forward-simulates the kinematic
bicycle over a 1 s horizon and picks the control sequence whose *whole
predicted trajectory* hugs the reference, so curvature is anticipated instead
of reacted to.

Formulation, solved with IPOPT each control tick (~10 ms for N=10):

    state    x, y, yaw, v          (kinematic bicycle, wheelbase L)
    input    a (accel), delta (steering angle; rate-limited between steps)
    cost     position error to time-parameterized reference points
             + speed tracking + input effort + steering rate
    subject  |delta| <= max steering, |a| <= traction*g,
             v in [0, v_max],  v^2 |tan delta| / L <= traction*g

Reference points are sampled along the planned path at the profile speed, so
braking for a hairpin enters the horizon a full second before the hairpin.
Falls back to the caller's pure-pursuit command if the solve fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import casadi
import numpy as np

GRAVITY = 9.81


@dataclass
class MpcConfig:
    horizon: int = 10
    dt: float = 0.1
    wheelbase: float = 0.324
    max_steering_angle: float = 0.40
    max_steering_rate: float = 3.5  # servo-limited; see casadi_smoother.py
    traction: float = 0.6
    max_speed: float = 4.0
    # Speed floor. Below roughly 0.8 m/s the steered wheels cannot overcome
    # tire scrub and the car stops rotating, so without this the optimizer
    # trades speed for tracking error, crawls, and wedges -- its cost function
    # has no notion that slow means unsteerable.
    #
    # A flat floor is not enough: scrub is worst at full lock, which is
    # exactly where the speed profile asks for the least speed. So the floor
    # is raised in proportion to how hard the reference is turning, up to
    # min_speed_turn. Calibration measured a clean 1.5 m/s at every steering
    # angle including full lock, so the turning floor is known-achievable
    # rather than assumed.
    min_speed: float = 0.9
    min_speed_turn: float = 1.4
    w_position: float = 10.0
    w_terminal: float = 30.0
    w_speed: float = 1.0
    w_accel: float = 0.02
    w_steer_rate: float = 0.5


class MpcTracker:
    def __init__(self, config: MpcConfig) -> None:
        self.config = config
        self._delta_prev = 0.0
        self._warm = None
        self._build()

    def _build(self) -> None:
        c = self.config
        n = c.horizon
        opti = casadi.Opti()

        x = opti.variable(n + 1)
        y = opti.variable(n + 1)
        yaw = opti.variable(n + 1)
        v = opti.variable(n + 1)
        a = opti.variable(n)
        delta = opti.variable(n)

        state0 = opti.parameter(4)  # x, y, yaw, v
        delta0 = opti.parameter()  # last applied steering, for rate limit
        ref_xy = opti.parameter(n, 2)  # reference points, one per step
        ref_v = opti.parameter(n)
        v_floor = opti.parameter()  # curvature-dependent minimum speed

        opti.subject_to(x[0] == state0[0])
        opti.subject_to(y[0] == state0[1])
        opti.subject_to(yaw[0] == state0[2])
        opti.subject_to(v[0] == state0[3])

        a_max = c.traction * GRAVITY
        rate = c.max_steering_rate * c.dt
        cost = 0
        for k in range(n):
            opti.subject_to(x[k + 1] == x[k] + v[k] * casadi.cos(yaw[k]) * c.dt)
            opti.subject_to(y[k + 1] == y[k] + v[k] * casadi.sin(yaw[k]) * c.dt)
            opti.subject_to(
                yaw[k + 1] == yaw[k] + v[k] / c.wheelbase * casadi.tan(delta[k]) * c.dt
            )
            opti.subject_to(v[k + 1] == v[k] + a[k] * c.dt)
            opti.subject_to(opti.bounded(-a_max, a[k], a_max))
            opti.subject_to(
                opti.bounded(-c.max_steering_angle, delta[k], c.max_steering_angle)
            )
            prev = delta0 if k == 0 else delta[k - 1]
            opti.subject_to(opti.bounded(-rate, delta[k] - prev, rate))
            lateral = v[k] ** 2 * casadi.tan(delta[k]) / c.wheelbase
            opti.subject_to(lateral**2 <= a_max**2)

            w_pos = c.w_terminal if k == n - 1 else c.w_position
            cost += w_pos * (
                (x[k + 1] - ref_xy[k, 0]) ** 2 + (y[k + 1] - ref_xy[k, 1]) ** 2
            )
            cost += c.w_speed * (v[k + 1] - ref_v[k]) ** 2
            cost += c.w_accel * a[k] ** 2
            cost += c.w_steer_rate * (delta[k] - prev) ** 2
        opti.subject_to(opti.bounded(v_floor, v[1:], c.max_speed))
        opti.subject_to(
            opti.bounded(0.0, v[0], c.max_speed)
        )  # current state may be slower
        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 60,
                "ipopt.tol": 1e-3,
                "ipopt.acceptable_tol": 1e-2,
            },
        )

        self._opti = opti
        self._vars = (x, y, yaw, v, a, delta)
        self._params = (state0, delta0, ref_xy, ref_v, v_floor)

    def reset(self) -> None:
        self._delta_prev = 0.0
        self._warm = None

    def solve(
        self,
        x: float,
        y: float,
        yaw: float,
        v: float,
        ref_xy: np.ndarray,
        ref_v: np.ndarray,
        ref_curvature: float = 0.0,
    ) -> tuple[float, float] | None:
        """One tick. Returns (commanded speed, steering angle) or None on failure.

        `ref_curvature` (1/m, unsigned) raises the speed floor where the path
        turns hard, because that is where tire scrub can stall the car.
        """
        opti = self._opti
        xs, ys, yaws, vs, a_var, delta_var = self._vars
        state0, delta0, ref_xy_p, ref_v_p, v_floor_p = self._params
        c = self.config
        # Full lock is the reference point: kappa at the car's tightest radius
        # maps to the full turning floor, straight-ahead to the base floor.
        kappa_full = math.tan(c.max_steering_angle) / c.wheelbase
        blend = min(1.0, abs(ref_curvature) / max(kappa_full, 1e-6))
        floor = c.min_speed + blend * (c.min_speed_turn - c.min_speed)
        opti.set_value(state0, [x, y, yaw, v])
        opti.set_value(delta0, self._delta_prev)
        opti.set_value(ref_xy_p, ref_xy)
        opti.set_value(ref_v_p, ref_v)
        opti.set_value(v_floor_p, min(floor, c.max_speed))
        if self._warm is not None:
            for var, val in zip(self._vars, self._warm):
                opti.set_initial(var, val)
        else:
            opti.set_initial(xs, np.linspace(x, ref_xy[-1, 0], self.config.horizon + 1))
            opti.set_initial(ys, np.linspace(y, ref_xy[-1, 1], self.config.horizon + 1))
            opti.set_initial(yaws, np.full(self.config.horizon + 1, yaw))
            opti.set_initial(vs, np.full(self.config.horizon + 1, max(v, 0.1)))
        try:
            solution = opti.solve()
        except RuntimeError:
            self._warm = None
            return None
        self._warm = tuple(solution.value(var) for var in self._vars)
        delta_cmd = float(solution.value(delta_var[0]))
        speed_cmd = float(solution.value(vs[1]))
        self._delta_prev = delta_cmd
        return speed_cmd, delta_cmd


if __name__ == "__main__":
    import time

    config = MpcConfig()
    tracker = MpcTracker(config)
    # Track a right-angle-ish arc: start on a straight, reference bends away.
    theta = np.linspace(0, 0.9, config.horizon)
    radius = 1.5
    ref = np.stack([radius * np.sin(theta), radius * (1 - np.cos(theta))], axis=1)
    ref_v = np.full(config.horizon, 1.2)

    state = [0.0, 0.0, 0.0, 1.0]
    start = time.perf_counter()
    ticks = 20
    for _ in range(ticks):
        result = tracker.solve(*state, ref, ref_v)
        assert result is not None, "MPC failed on a benign arc"
        speed, delta = result
        lateral = speed**2 * abs(math.tan(delta)) / config.wheelbase
        assert lateral <= config.traction * GRAVITY * 1.05, "friction circle violated"
        # crude plant: apply the command exactly for one dt
        x, y, yaw, v = state
        state = [
            x + speed * math.cos(yaw) * config.dt,
            y + speed * math.sin(yaw) * config.dt,
            yaw + speed / config.wheelbase * math.tan(delta) * config.dt,
            speed,
        ]
    elapsed_ms = (time.perf_counter() - start) / ticks * 1000
    error = math.hypot(state[0] - ref[-1, 0], state[1] - ref[-1, 1])
    print(
        f"arc tracking: final offset {error:.2f} m from horizon end, "
        f"{elapsed_ms:.1f} ms/solve (budget 50 ms at 20 Hz)"
    )
    assert error < 0.8, "did not follow the arc"
    print("MPC tracker checks passed")
