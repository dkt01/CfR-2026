"""CasADi short-horizon command smoother.

The PPO policy outputs a fresh (speed, steering) target every control tick
with no memory of what it asked for last tick, so raw commands can slam
between extremes -- fine in a kinematic sim, hard on a real drivetrain and
inaccurate once tires saturate. This module sits between the policy and
/cmd_vel and solves a small optimal-control problem each tick:

    state    v (speed), delta (steering angle)
    inputs   a (accel), r (steering rate)
    cost     track the policy's (v_ref, delta_ref) + penalize a^2, r^2
    subject  |a| <= traction * g                 (longitudinal grip)
             v^2 * |tan(delta)| / L <= traction*g (lateral grip, bicycle model)
             |delta| <= max steering angle, |r| <= steering slew rate
             0 <= v <= max_speed

The friction-circle constraints are what make this a traction model rather
than a low-pass filter: the optimizer will slow the car down *before* a
steering angle it cannot carry at speed, instead of understeering through it.
Traction (mu) and max speed are plain config numbers so track surface and
speed caps can be changed without retraining.

The lateral constraint makes the problem nonconvex, so it is solved with
IPOPT, warm-started from the previous solution. N=8 at dt=0.1 s solves in
about a millisecond -- well inside a 10 Hz tick. On solver failure the
fallback is the reference clamped to the same traction limits, so a bad
solve degrades to a slew-limited command, never a wild one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import casadi
import numpy as np

GRAVITY = 9.81


@dataclass
class SmootherConfig:
    enabled: bool = True
    horizon: int = 8
    dt: float = 0.1
    # Friction coefficient between tires and floor; a_max = traction * g.
    traction: float = 0.6
    max_speed: float = 4.0
    # Most negative allowed speed; 0.0 forbids reverse, a negative value (e.g.
    # -0.5 to mirror the env's reverse_speed) lets the policy back out.
    min_speed: float = 0.0
    max_steering_angle: float = 0.40  # rad, mirrors arduino_bridge.yaml
    # rad/s at the road wheel. Traxxas 2075 servo: 0.17 s/60 deg = 6.16 rad/s
    # at the horn, ~50-75% of that through the linkage -> lock-to-lock ~0.25 s.
    max_steering_rate: float = 3.5
    wheelbase: float = 0.324
    w_speed: float = 1.0
    w_steer: float = 4.0
    w_accel: float = 0.05
    w_steer_rate: float = 0.02


def smoother_from_metadata(
    metadata: dict, control_hz: float, traction: float, max_speed: float,
    wheelbase: float, max_steering_angle: float,
) -> "CommandSmoother":
    """Build a smoother from a checkpoint's metadata JSON (written by train.py).

    Timing and vehicle limits come from the env section so the smoother can
    never disagree with what the policy was trained against; the metadata's
    smoother section carries only the tuning knobs.
    """
    knobs = dict(metadata.get("smoother", {}))
    knobs.pop("enabled", None)
    return CommandSmoother(SmootherConfig(
        dt=1.0 / control_hz,
        traction=traction,
        max_speed=max_speed,
        min_speed=-metadata.get("env", {}).get("reverse_speed", 0.0),
        wheelbase=wheelbase,
        max_steering_angle=max_steering_angle,
        **knobs,
    ))


class CommandSmoother:
    def __init__(self, config: SmootherConfig) -> None:
        self.config = config
        self._v = 0.0
        self._delta = 0.0
        self._build()

    def _build(self) -> None:
        c = self.config
        n = c.horizon
        opti = casadi.Opti()
        v = opti.variable(n + 1)
        delta = opti.variable(n + 1)
        a = opti.variable(n)
        r = opti.variable(n)

        v0 = opti.parameter()
        delta0 = opti.parameter()
        v_ref = opti.parameter()
        delta_ref = opti.parameter()

        opti.subject_to(v[0] == v0)
        opti.subject_to(delta[0] == delta0)

        a_max = c.traction * GRAVITY
        cost = 0
        for k in range(n):
            opti.subject_to(v[k + 1] == v[k] + a[k] * c.dt)
            opti.subject_to(delta[k + 1] == delta[k] + r[k] * c.dt)
            opti.subject_to(opti.bounded(-a_max, a[k], a_max))
            opti.subject_to(opti.bounded(-c.max_steering_rate, r[k], c.max_steering_rate))
            # Lateral acceleration of the kinematic bicycle: v^2 tan(delta)/L.
            # tan(delta)^2 form keeps it smooth through delta = 0.
            lat = v[k + 1] ** 2 * casadi.tan(delta[k + 1]) / c.wheelbase
            opti.subject_to(lat**2 <= a_max**2)
            cost += (
                c.w_speed * (v[k + 1] - v_ref) ** 2
                + c.w_steer * (delta[k + 1] - delta_ref) ** 2
                + c.w_accel * a[k] ** 2
                + c.w_steer_rate * r[k] ** 2
            )
        opti.subject_to(opti.bounded(c.min_speed, v, c.max_speed))
        opti.subject_to(opti.bounded(-c.max_steering_angle, delta, c.max_steering_angle))
        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
             "ipopt.max_iter": 50, "ipopt.tol": 1e-4},
        )

        self._opti = opti
        self._vars = (v, delta, a, r)
        self._params = (v0, delta0, v_ref, delta_ref)
        self._warm = None

    def _clamp_fallback(self, v_ref: float, delta_ref: float) -> tuple[float, float]:
        """Traction-limited slew of the reference, used when IPOPT fails."""
        c = self.config
        a_max = c.traction * GRAVITY
        v = self._v + np.clip(v_ref - self._v, -a_max * c.dt, a_max * c.dt)
        delta = self._delta + np.clip(
            delta_ref - self._delta, -c.max_steering_rate * c.dt, c.max_steering_rate * c.dt
        )
        v = float(np.clip(v, c.min_speed, c.max_speed))
        delta = float(np.clip(delta, -c.max_steering_angle, c.max_steering_angle))
        if abs(math.tan(delta)) > 1e-6:
            grip = math.sqrt(a_max * c.wheelbase / abs(math.tan(delta)))
            v = min(max(v, -grip), grip)
        return v, delta

    def reset(self, v: float = 0.0, delta: float = 0.0) -> None:
        self._v = v
        self._delta = delta
        self._warm = None

    def smooth(self, v_ref: float, delta_ref: float) -> tuple[float, float]:
        """One tick: policy's raw (speed, steering angle) in, smoothed pair out."""
        if not self.config.enabled:
            self._v, self._delta = v_ref, delta_ref
            return v_ref, delta_ref
        opti = self._opti
        v, delta, a, r = self._vars
        v0, delta0, vr, dr = self._params
        opti.set_value(v0, self._v)
        opti.set_value(delta0, self._delta)
        opti.set_value(vr, np.clip(v_ref, self.config.min_speed, self.config.max_speed))
        opti.set_value(dr, np.clip(delta_ref, -self.config.max_steering_angle, self.config.max_steering_angle))
        if self._warm is not None:
            for var, val in zip(self._vars, self._warm):
                opti.set_initial(var, val)
        try:
            sol = opti.solve()
            self._warm = tuple(sol.value(var) for var in self._vars)
            v_next = float(sol.value(v[1]))
            delta_next = float(sol.value(delta[1]))
        except RuntimeError:
            self._warm = None
            v_next, delta_next = self._clamp_fallback(v_ref, delta_ref)
        self._v, self._delta = v_next, delta_next
        return v_next, delta_next


if __name__ == "__main__":
    import time

    cfg = SmootherConfig()
    smoother = CommandSmoother(cfg)
    a_max = cfg.traction * GRAVITY

    # A policy that slams full speed + full lock at once. The smoother must
    # respect accel/steer-rate slews and hold the friction circle.
    prev_v, prev_d = 0.0, 0.0
    start = time.perf_counter()
    ticks = 30
    for _ in range(ticks):
        v, d = smoother.smooth(cfg.max_speed, cfg.max_steering_angle)
        assert v - prev_v <= a_max * cfg.dt + 1e-6, "accel slew violated"
        assert abs(d - prev_d) <= cfg.max_steering_rate * cfg.dt + 1e-6, "steer slew violated"
        lat = v**2 * abs(math.tan(d)) / cfg.wheelbase
        assert lat <= a_max * 1.01, f"friction circle violated: {lat:.2f} > {a_max:.2f}"
        prev_v, prev_d = v, d
    per_tick_ms = (time.perf_counter() - start) / ticks * 1000
    print(f"steady state under full-lock request: v={v:.2f} m/s delta={d:.2f} rad "
          f"(lat accel {lat:.2f} <= {a_max:.2f} m/s^2)")
    print(f"solve time {per_tick_ms:.2f} ms/tick (budget 100 ms at 10 Hz)")

    # And it should track a benign reference essentially exactly.
    smoother.reset()
    for _ in range(30):
        v, d = smoother.smooth(1.0, 0.05)
    assert abs(v - 1.0) < 0.05 and abs(d - 0.05) < 0.01, f"benign tracking off: {v=} {d=}"
    print(f"benign reference tracked: v={v:.3f} delta={d:.3f}")
    print("\nCasADi smoother checks passed")
