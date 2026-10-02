"""Depth-only bale corridor follower for the physical speed course."""

from __future__ import annotations

import math

import numpy as np


class CorridorFollower:
    def __init__(self, cfg):
        self.cfg = cfg["formula_sub_zero"]["corridor"]
        self.plant = cfg["plant"]
        self.collision = cfg["collision"]
        self.reset()

    def reset(self):
        self.half_width = float(self.cfg["nominal_half_width_m"])
        self.target_y = 0.0
        self.last_steer = 0.0
        self.status = "waiting for corridor depth"

    @staticmethod
    def _faces(points, minimum):
        """Separate the two bale faces even when both lie on one image side."""
        y = np.sort(points[:, 1])
        if len(y) < minimum:
            return None, None
        gap = np.diff(y)
        if len(gap):
            split = int(np.argmax(gap)) + 1
            if gap[split - 1] >= 0.28:
                if split >= minimum and len(y) - split >= minimum:
                    return (float(np.quantile(y[split:], 0.20)),
                            float(np.quantile(y[:split], 0.80)))
                return None, None
        if float(np.median(y)) >= 0:
            return float(np.quantile(y, 0.20)), None
        return None, float(np.quantile(y, 0.80))

    def _center_samples(self, points):
        lower = float(self.cfg["min_lookahead_m"])
        upper = float(self.cfg["max_lookahead_m"])
        edges = np.linspace(lower, upper, 7)
        x, y, weight, widths = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            group = points[(points[:, 0] >= lo) & (points[:, 0] < hi)]
            left, right = self._faces(
                group, max(2, int(self.cfg["min_points_per_wall"]) // 2),
            )
            if left is None and right is None:
                continue
            if left is not None and right is not None:
                width = left - right
                if width < float(self.cfg["min_corridor_width_m"]):
                    if lo < 1.5:
                        return np.array([]), np.array([]), np.array([]), [], "narrow"
                    continue
                center = (left + right) / 2
                confidence = 2.0
                widths.append(width)
            elif left is not None:
                center = left - self.half_width
                confidence = 1.0
            else:
                center = right + self.half_width
                confidence = 1.0
            x.append((lo + hi) / 2)
            y.append(float(np.clip(center, -1.2, 1.2)))
            weight.append(confidence)
        return np.asarray(x), np.asarray(y), np.asarray(weight), widths, "ok"

    def command(self, points, speed, yaw_rate):
        """Return steering and coast-safe speed from visible bale faces."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("corridor points must be an Nx2 array")
        near = points[(points[:, 0] >= float(self.cfg["min_lookahead_m"])) &
                      (points[:, 0] <= float(self.cfg["max_lookahead_m"]))]
        near = near[np.isfinite(near).all(axis=1)]
        xs, ys, weights, widths, state = self._center_samples(near)
        if state == "narrow":
            self.status = "visible corridor narrower than body clearance; coasting"
            return 0.0, 0.0
        if not len(xs):
            self.status = "no visible bale wall; coasting"
            return 0.0, 0.0
        if widths:
            self.half_width = 0.8 * self.half_width + 0.2 * min(float(np.median(widths)) / 2, 0.9)
        walls = "both walls" if widths else "one wall"
        center = float(ys[0])

        # A single midpoint over the whole visible wall erases bend geometry.
        # Sample wall faces by forward distance and fit the corridor center.
        lookahead = float(np.clip(1.0 + 0.6 * max(speed, 0.0), 1.2, 2.2))
        if len(xs) >= 3:
            coeff = np.polyfit(xs, ys, 2, w=weights)
            target = float(np.polyval(coeff, lookahead))
        elif len(xs) == 2:
            target = float(np.interp(lookahead, xs, ys, left=ys[0], right=ys[-1]))
        elif len(xs) == 1:
            target = float(ys[0])
        else:
            target = center
        target = float(np.clip(target, -0.9, 0.9))
        blend = 0.75 if widths else 0.25
        self.target_y = (1 - blend) * self.target_y + blend * target
        curvature = 2 * self.target_y / (lookahead * lookahead)
        angle = math.atan(float(self.plant["wheelbase"]) * curvature)
        steer = float(np.interp(angle, self.plant["steering_angle_points"],
                                self.plant["steering_command_points"]))
        steer -= float(self.cfg["yaw_rate_gain"]) * yaw_rate
        limit = float(self.cfg["max_steer_command"])
        change = float(self.cfg["max_steer_change_per_tick"])
        steer = float(np.clip(steer, -limit, limit))
        steer = float(np.clip(steer, self.last_steer - change,
                              self.last_steer + change))
        self.last_steer = steer

        # A bale in the predicted swept corridor limits speed. Keep a larger
        # coast margin than the physical nose because speed control also lags.
        if len(xs) >= 2:
            path_y = np.interp(points[:, 0], xs, ys, left=ys[0], right=ys[-1])
        else:
            path_y = np.full(len(points), center)
        front = points[(points[:, 0] > 0) &
                       ((np.abs(points[:, 1] - path_y) <
                         float(self.collision["wheel_outer_y"])
                         + float(self.cfg["front_margin_m"])) |
                        (np.abs(points[:, 1]) <
                         float(self.collision["wheel_outer_y"])
                         + float(self.cfg["front_margin_m"])))]
        nearest = (float(np.min(front[:, 0])) if len(front)
                   else float(self.cfg["max_lookahead_m"]))
        room = max(0.0, nearest - float(self.cfg["stop_margin_m"]))
        decel = min(float(self.cfg["coast_decel_mps2"]),
                    float(self.plant["coast_f0"]) / float(self.plant["mass"]))
        reaction = float(self.cfg["reaction_s"])
        stop_cap = max(0.0, math.sqrt((decel * reaction) ** 2 + 2 * decel * room)
                       - decel * reaction)
        curve_cap = math.sqrt(float(self.cfg["max_lateral_accel_mps2"])
                              / max(abs(curvature), 1e-3))
        command_speed = min(float(self.cfg["speed_mps"]), stop_cap, curve_cap)
        if not widths:
            command_speed = min(command_speed, float(self.cfg["single_wall_speed_mps"]))
        self.status = (f"{walls}, target {self.target_y:+.2f} m at {lookahead:.1f} m, "
                       f"bend {curvature:+.2f} 1/m, front {nearest:.1f} m, "
                       f"speed {command_speed:.2f} m/s")
        return steer, command_speed
