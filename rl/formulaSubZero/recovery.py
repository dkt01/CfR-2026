"""Bounded stall recovery using ZED pose progress and mapped rear clearance."""

from __future__ import annotations

import math

import numpy as np


class StallRecovery:
    def __init__(self, track, cfg):
        self.track = track
        self.cfg = cfg["formula_sub_zero"]["recovery"]
        self.plant = cfg["plant"]
        self.collision = cfg["collision"]
        self.reset()

    def reset(self):
        self.phase = "idle"
        self.attempts = 0
        self.stall_since = None
        self.stall_origin = None
        self.recovery_origin = None
        self.reverse_origin = None
        self.escape_origin = None
        self.reverse_yaw = 0.0
        self.reverse_steer = 0.0
        self.phase_since = 0.0
        self.settle_check_since = None
        self.settle_check_origin = None
        self.cooldown_until = 0.0
        self.event = None

    def _body_clearance(self, x, y, yaw):
        return self.track.body_clearance(
            np.asarray(x), np.asarray(y), np.asarray(yaw),
            float(self.collision["chassis_half_length"]),
            float(self.collision["wheel_outer_y"]),
        )

    def _reverse_steering(self, x, y, yaw, station):
        # Check past the commanded stopping point for actuator delay and coasting.
        distance = (float(self.cfg["reverse_distance_m"]) +
                    float(self.cfg["reverse_coast_margin_m"]))
        n = max(10, int(math.ceil(distance / 0.025)))
        ds = -distance / n
        initial_clear = float(self._body_clearance([x], [y], [yaw])[0])
        minimum = float(self.cfg["min_map_clearance_m"])
        commands = self.plant["steering_command_points"]
        angles = self.plant["steering_angle_points"]
        wheelbase = float(self.plant["wheelbase"]) * float(self.plant["tire_scrub"])
        ref_station = station - distance
        ref_x = float(self.track.at(ref_station, self.track.x))
        ref_y = float(self.track.at(ref_station, self.track.y))
        ref_yaw = math.atan2(
            float(self.track.at(ref_station, self.track.ty)),
            float(self.track.at(ref_station, self.track.tx)),
        )
        best = None
        for command in (-0.35, -0.18, 0.0, 0.18, 0.35):
            angle = float(np.interp(command, commands, angles))
            xx, yy, heading = x, y, yaw
            path_x, path_y, path_yaw = [xx], [yy], [heading]
            for step in range(n):
                # The command already in flight delays the new wheel angle.
                if step * abs(ds) >= 0.09:
                    heading += ds * math.tan(angle) / wheelbase
                xx += ds * math.cos(heading)
                yy += ds * math.sin(heading)
                path_x.append(xx)
                path_y.append(yy)
                path_yaw.append(heading)
            clearance = self._body_clearance(path_x, path_y, path_yaw)
            threshold = minimum if initial_clear >= minimum else initial_clear - 0.02
            if np.min(clearance[1:]) < threshold:
                continue
            if initial_clear < minimum and clearance[-1] < initial_clear + 0.03:
                continue
            lateral = -(xx - ref_x) * math.sin(ref_yaw) + (yy - ref_y) * math.cos(ref_yaw)
            heading_error = math.atan2(
                math.sin(heading - ref_yaw), math.cos(heading - ref_yaw)
            )
            score = 2.0 * abs(lateral) + 0.4 * abs(heading_error) - 0.2 * clearance[-1]
            if best is None or score < best[0]:
                best = score, command
        return None if best is None else best[1]

    def _fail(self, reason):
        self.phase = "failed"
        self.event = f"recovery stopped: {reason}; manual intervention required"
        return 0.0, 0.0

    def abort(self, reason):
        if self.phase in ("reverse", "settle", "escape"):
            self._fail(reason)

    def step(self, now, x, y, yaw, station, commanded_forward, measured_speed, enabled):
        """Return a direct (steering, signed speed) override or None."""
        self.event = None
        if not enabled:
            self.stall_since = None
            if self.phase in ("reverse", "settle", "escape"):
                return self._fail("pose, depth, or actuator status became unavailable")
            return (0.0, 0.0) if self.phase == "failed" else None
        if self.phase == "failed":
            return 0.0, 0.0
        if self.phase == "reverse":
            ox, oy = self.reverse_origin
            backed = -((x - ox) * math.cos(self.reverse_yaw) +
                       (y - oy) * math.sin(self.reverse_yaw))
            if backed >= float(self.cfg["reverse_distance_m"]):
                self.phase = "settle"
                self.phase_since = now
                self.settle_check_since = None
                self.event = f"backed {backed:.2f} m; waiting for stop"
                return 0.0, 0.0
            elapsed = now - self.phase_since
            if elapsed >= float(self.cfg["reverse_no_progress_s"]) and backed < 0.04:
                return self._fail("reverse made no measured progress")
            if elapsed >= float(self.cfg["reverse_timeout_s"]):
                return self._fail(f"reverse made only {max(backed, 0.0):.2f} m progress")
            if abs((x - ox) * -math.sin(self.reverse_yaw) +
                   (y - oy) * math.cos(self.reverse_yaw)) > 0.22:
                return self._fail("reverse moved too far sideways")
            return self.reverse_steer, -float(self.cfg["reverse_speed_mps"])
        if self.phase == "settle":
            elapsed = now - self.phase_since
            if elapsed >= float(self.cfg["settle_timeout_s"]):
                return self._fail("car did not stop after reverse")
            if elapsed < float(self.cfg["settle_s"]):
                return 0.0, 0.0
            if self.settle_check_since is None:
                self.settle_check_since = now
                self.settle_check_origin = (x, y)
                return 0.0, 0.0
            if now - self.settle_check_since >= 0.3:
                moved = math.hypot(x - self.settle_check_origin[0],
                                   y - self.settle_check_origin[1])
                if moved < 0.03 and measured_speed < 0.2:
                    self.phase = "escape"
                    self.phase_since = now
                    self.escape_origin = (x, y)
                    self.stall_since = None
                    self.cooldown_until = now + float(self.cfg["cooldown_s"])
                    self.event = "stopped after reverse; moving forward slowly with depth planner"
                else:
                    self.settle_check_since = now
                    self.settle_check_origin = (x, y)
            return 0.0, 0.0

        if self.phase == "escape":
            progress = math.hypot(x - self.escape_origin[0], y - self.escape_origin[1])
            if progress >= float(self.cfg["escape_distance_m"]):
                self.phase = "idle"
                self.stall_since = None
                self.event = "escaped bale; restoring planned speed"
            elif now - self.phase_since >= float(self.cfg["escape_retry_s"]):
                self.phase = "idle"
                self.stall_since = now - float(self.cfg["stall_s"])
                self.stall_origin = (x, y)
                self.cooldown_until = 0.0
            else:
                return None

        if self.recovery_origin is not None and math.hypot(
            x - self.recovery_origin[0], y - self.recovery_origin[1]
        ) > 1.0:
            self.attempts = 0
            self.recovery_origin = None
        if now < self.cooldown_until or commanded_forward < float(self.cfg["min_forward_cmd_mps"]):
            self.stall_since = None
            return None
        if self.stall_since is None:
            self.stall_since = now
            self.stall_origin = (x, y)
            return None
        if math.hypot(x - self.stall_origin[0], y - self.stall_origin[1]) >= float(
            self.cfg["min_progress_m"]
        ):
            self.stall_since = now
            self.stall_origin = (x, y)
            return None
        if now - self.stall_since < float(self.cfg["stall_s"]):
            return None
        if self.attempts >= int(self.cfg["max_attempts"]):
            return self._fail("still stalled after the allowed attempts")
        steer = self._reverse_steering(x, y, yaw, station)
        if steer is None:
            return self._fail("no mapped reverse path clears the bale envelope")
        self.attempts += 1
        if self.recovery_origin is None:
            self.recovery_origin = (x, y)
        self.phase = "reverse"
        self.phase_since = now
        self.reverse_origin = (x, y)
        self.reverse_yaw = yaw
        self.reverse_steer = steer
        self.stall_since = None
        self.event = f"stalled {float(self.cfg['stall_s']):.1f} s; backing attempt {self.attempts}"
        return steer, -float(self.cfg["reverse_speed_mps"])
