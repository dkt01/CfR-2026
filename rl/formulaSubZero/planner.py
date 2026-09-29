"""Map-based racing line and coast-feasible speed for the Speed Course."""

from __future__ import annotations

import numpy as np


class SpeedCoursePlan:
    def __init__(self, track, cfg):
        self.track = track
        pc = cfg["formula_sub_zero"]
        step = float(pc["path_step_m"])
        stations = np.arange(0.0, track.length, step)
        offsets = np.arange(
            -float(pc["path_max_offset_m"]),
            float(pc["path_max_offset_m"]) + 0.001,
            float(pc["path_offset_step_m"]),
        )
        target = float(pc["path_target_clearance_m"])
        x = track.at(stations, track.x)
        y = track.at(stations, track.y)
        tx = track.at(stations, track.tx)
        ty = track.at(stations, track.ty)
        yaw = np.arctan2(ty, tx)
        nx, ny = -ty, tx
        c = cfg["collision"]
        # A center point can look safe while the outside wheel touches a bale.
        clearance = np.column_stack(
            [
                track.body_clearance(
                    x + offset * nx,
                    y + offset * ny,
                    yaw,
                    float(c["chassis_half_length"]),
                    float(c["wheel_outer_y"]),
                )
                for offset in offsets
            ]
        )
        site = 2.0 * offsets[None, :] ** 2 + 150.0 * np.maximum(
            target - clearance, 0.0
        ) ** 2
        site = np.broadcast_to(site, clearance.shape).copy()
        # Dynamic programming discourages sudden shifts that the car could not
        # track. The first station is centered at the grid start.
        score = np.full_like(site, np.inf)
        center = int(np.argmin(abs(offsets)))
        score[0, center] = site[0, center]
        parent = np.zeros(site.shape, dtype=np.int16)
        for i in range(1, len(stations)):
            transition = score[i - 1, :, None] + 8.0 * (
                offsets[:, None] - offsets[None, :]
            ) ** 2
            parent[i] = np.argmin(transition, axis=0)
            score[i] = site[i] + transition[parent[i], np.arange(len(offsets))]
        chosen = np.zeros(len(stations), dtype=np.int16)
        chosen[-1] = int(np.argmin(score[-1] + 8.0 * offsets**2))
        for i in range(len(stations) - 1, 0, -1):
            chosen[i - 1] = parent[i, chosen[i]]
        offset = offsets[chosen]
        # Smooth the discrete offset while keeping the loop closed.
        pad = np.r_[offset[-3:], offset, offset[:3]]
        offset = np.convolve(pad, np.ones(7) / 7.0, mode="valid")
        dense_offset = np.interp(track.s, stations, offset, period=track.length)
        # Gazebo's measured yaw lag puts this hairpin consistently to the
        # right of the centerline; the map still gives >=14 cm body clearance
        # with this small, smooth left shift (three-lap trace, 2026-09-29).
        entry = float(pc["start_hairpin_entry_width_m"])
        exit_width = float(pc["start_hairpin_exit_width_m"])
        signed_s = (track.s + track.length / 2) % track.length - track.length / 2
        width = np.where(signed_s < 0, entry, exit_width)
        window = np.where(
            np.abs(signed_s) < width,
            0.5 * (1 + np.cos(np.pi * signed_s / width)),
            0.0,
        )
        dense_offset += float(pc["start_hairpin_bias_m"]) * window
        exit_length = float(pc["start_hairpin_exit_extra_length_m"])
        exit_window = np.where(
            (signed_s > 0) & (signed_s < exit_length),
            np.sin(np.pi * signed_s / exit_length) ** 2,
            0.0,
        )
        dense_offset += float(pc["start_hairpin_exit_extra_m"]) * exit_window
        center = float(pc["mid_hairpin_center_m"])
        distance = (track.s - center + track.length / 2) % track.length - track.length / 2
        width = np.where(
            distance < 0,
            float(pc["mid_hairpin_width_m"]),
            float(pc["mid_hairpin_exit_width_m"]),
        )
        window = np.where(
            np.abs(distance) < width,
            0.5 * (1 + np.cos(np.pi * distance / width)),
            0.0,
        )
        dense_offset += float(pc["mid_hairpin_bias_m"]) * window
        approach = float(pc["approach_corner_center_m"])
        approach_width = float(pc["approach_corner_width_m"])
        distance = (track.s - approach + track.length / 2) % track.length - track.length / 2
        window = np.where(
            np.abs(distance) < approach_width,
            0.5 * (1 + np.cos(np.pi * distance / approach_width)),
            0.0,
        )
        dense_offset += float(pc["approach_corner_bias_m"]) * window
        self.x = track.x - track.ty * dense_offset
        self.y = track.y + track.tx * dense_offset
        dx = np.roll(self.x, -1) - np.roll(self.x, 1)
        dy = np.roll(self.y, -1) - np.roll(self.y, 1)
        self.yaw = np.arctan2(dy, dx)
        ds = np.maximum(np.hypot(dx, dy), 1e-6)
        turn = np.roll(self.yaw, -1) - np.roll(self.yaw, 1)
        self.kappa = np.arctan2(np.sin(turn), np.cos(turn)) / ds
        window_size = max(3, int(round(0.35 / track.ds)) | 1)
        padded = np.r_[self.kappa[-window_size:], self.kappa, self.kappa[:window_size]]
        self.kappa = np.convolve(
            padded, np.ones(window_size) / window_size, mode="same"
        )[window_size : window_size + len(self.kappa)]
        from baseline import feasible_profile

        self.speed = feasible_profile(track, cfg) * float(pc["speed_fraction"])
        self.min_map_clearance = float(np.min(track.body_clearance(
            self.x, self.y, self.yaw,
            float(c["chassis_half_length"]), float(c["wheel_outer_y"]),
        )))

    def at(self, station, field):
        return self.track.at(np.asarray(station), field)

    def yaw_at(self, station):
        return np.arctan2(
            self.at(station, np.sin(self.yaw)),
            self.at(station, np.cos(self.yaw)),
        )

    def reference(self, station, speed, n, dt, delay):
        s = station + max(speed, 0.8) * (delay + dt * np.arange(n + 1))
        return (
            self.at(s, self.x),
            self.at(s, self.y),
            self.yaw_at(s),
            self.at(s, self.kappa),
        )
