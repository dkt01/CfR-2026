"""Clearance-gated speed ceiling, shared by training and deployment.

The policy sees a 6 m forward scan and has no notion of "straight" or
"corner", so trained at a single cap it drives every metre of the course at
that cap. This raises the ceiling only where the scan shows room down the
corridor AND the policy is not actually turning, leaving the trained cap
everywhere else.

It lives in its own module because env.py and run_policy.py must apply the
*same* envelope. A previous version of this project trained with one set of
actuator limits and deployed with another, and the policy lost two thirds of
its distance on deployment; anything shaping the action has to be shared code,
not two copies of the same constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BoostConfig:
    # Ceiling on clear straights. Equal to (or below) the trained cap disables
    # boosting entirely, which is the default so existing checkpoints are
    # unaffected.
    straight_speed: float = 0.0
    clear_lo: float = 2.0  # m of forward sight: below this, no boost
    clear_hi: float = 5.0  # m: at/above this, full boost
    # Narrow cone. The corridor is ~0.95 m wide, so a wide cone measures the
    # distance to the side walls rather than the road ahead: at 25 deg a ray
    # meets a wall 0.475 m away after 1.1 m, and forward clearance reads ~1 m
    # even on the longest straight (measured). At 6 deg that wall is 4.5 m
    # away, so the cone reports how far down the corridor the car can see.
    cone_deg: float = 6.0
    # Gate on the SIGNED steering average, not its magnitude. The policy steers
    # bang-bang at +/-1.0 on most ticks, including down a clear straight, so
    # averaging |steer| makes straights and corners both read 1.0 and the gate
    # never opens (measured). Signed, sawing cancels to ~0 and a sustained
    # corner holds ~+/-1.
    steer_gate: float = 0.45
    steer_tau: float = 0.5
    # Give speed back slowly, take it away fast: a corner enters the 6 m scan
    # with about a second of warning at boost speed, and braking has to be
    # able to use all of it.
    accel: float = 2.5
    decel: float = 6.0


class BoostLimiter:
    """Stateful speed ceiling. One instance per driving car."""

    def __init__(
        self,
        config: BoostConfig,
        base_speed: float,
        control_hz: float,
        lidar_fov_deg: float,
    ) -> None:
        self.config = config
        self.base_speed = base_speed
        self.control_hz = control_hz
        self.lidar_fov_deg = lidar_fov_deg
        self._ceiling = base_speed
        self._steer_avg = 0.0

    @property
    def enabled(self) -> bool:
        return self.config.straight_speed > self.base_speed

    @property
    def ceiling(self) -> float:
        return self._ceiling

    def reset(self) -> None:
        self._ceiling = self.base_speed
        self._steer_avg = 0.0

    def forward_clearance(self, scan: np.ndarray) -> float:
        angles = np.linspace(
            -self.lidar_fov_deg / 2.0, self.lidar_fov_deg / 2.0, len(scan)
        )
        cone = np.abs(angles) <= self.config.cone_deg
        return float(scan[cone].min()) if cone.any() else float(scan.min())

    def update(self, scan: np.ndarray, steer_fraction: float) -> float:
        """Advance one control step and return the current speed ceiling."""
        c = self.config
        if not self.enabled:
            return self.base_speed

        alpha = 1.0 - math.exp(-(1.0 / self.control_hz) / max(c.steer_tau, 1e-6))
        self._steer_avg += (steer_fraction - self._steer_avg) * alpha

        span = max(c.clear_hi - c.clear_lo, 1e-6)
        room = min(max((self.forward_clearance(scan) - c.clear_lo) / span, 0.0), 1.0)
        straightness = min(
            max(1.0 - abs(self._steer_avg) / max(c.steer_gate, 1e-6), 0.0), 1.0
        )
        target = self.base_speed + room * straightness * (
            c.straight_speed - self.base_speed
        )

        rate = c.accel if target > self._ceiling else c.decel
        step = rate / self.control_hz
        self._ceiling = min(max(target, self._ceiling - step), self._ceiling + step)
        self._ceiling = min(max(self._ceiling, self.base_speed), c.straight_speed)
        return self._ceiling

    def apply(self, speed: float, scan: np.ndarray, steer_fraction: float) -> float:
        """Scale a commanded speed up to the current ceiling.

        Forward commands only: reverse is recovery, which must keep its own
        modest speed regardless of how open the road ahead looks.
        """
        ceiling = self.update(scan, steer_fraction)
        if not self.enabled or speed <= 0.0:
            return speed
        # The action is a fraction of the trained range; apply the same
        # fraction to the raised ceiling so part throttle stays part throttle.
        return speed / self.base_speed * ceiling
