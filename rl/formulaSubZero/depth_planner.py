"""Live bale geometry and a short, clearance-aware racing line."""

from __future__ import annotations

import math

import numpy as np


def depth_points(image, intrinsics, height, pitch, roll, camera_x, camera_y=0.0,
                 stride=4, *, band, min_range, max_range, local_horizon):
    """Project in-band depth pixels into the car's ground-plane frame."""
    fx, fy, cx, cy = intrinsics[:4]
    if len(band) != 2 or band[0] >= band[1] or min_range <= 0 or max_range < min_range:
        raise ValueError("invalid depth projection range or height band")
    v = np.arange(stride // 2, image.shape[0], stride)
    u = np.arange(stride // 2, image.shape[1], stride)
    depth = image[np.ix_(v, u)]
    a = (cx - u)[None, :] / fx
    b = (cy - v)[:, None] / fy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    lateral = a * cr - b * sr
    z1 = a * sr + b * cr
    forward = cp + z1 * sp
    up = -sp + z1 * cp
    with np.errstate(invalid="ignore"):
        x = camera_x + depth * forward
        y = camera_y + depth * lateral
        z = height + depth * up
        valid = (np.isfinite(depth) & (depth >= min_range) & (depth <= max_range)
                 & (z >= band[0]) & (z <= band[1])
                 & (x > 0.1) & (x <= local_horizon))
    x = np.broadcast_to(x, depth.shape)[valid]
    y = np.broadcast_to(y, depth.shape)[valid]
    if not x.size:
        return np.empty((0, 2), dtype=np.float32)
    cells = np.round(np.column_stack((x, y)) / 0.06).astype(np.int16)
    _, first = np.unique(cells, axis=0, return_index=True)
    return np.column_stack((x[first], y[first])).astype(np.float32)


def signal_point(image, intrinsics, pixel, image_size, height, pitch, roll,
                 camera_x, camera_y):
    """The closest above-ground depth cluster at the detected signal pixel."""
    fx, fy, cx, cy = intrinsics[:4]
    rgb_width, rgb_height = image_size
    if rgb_width <= 0 or rgb_height <= 0:
        return None
    u = int(round(pixel[0] * image.shape[1] / rgb_width))
    v = int(round(pixel[1] * image.shape[0] / rgb_height))
    if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
        return None
    uu, vv = np.meshgrid(
        np.arange(max(0, u - 5), min(image.shape[1], u + 6)),
        np.arange(max(0, v - 5), min(image.shape[0], v + 6)),
    )
    depth = image[vv, uu]
    a, b = (cx - uu) / fx, (cy - vv) / fy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    lateral = a * cr - b * sr
    z1 = a * sr + b * cr
    forward = cp + z1 * sp
    up = -sp + z1 * cp
    x = camera_x + depth * forward
    y = camera_y + depth * lateral
    z = height + depth * up
    valid = (np.isfinite(depth) & (depth >= 0.5) & (depth <= 8.0)
             & (z >= 0.4) & (z <= 1.5))
    if valid.sum() < 3:
        return None
    close = np.flatnonzero(valid.ravel())
    close = close[np.argsort(depth.ravel()[close])[:max(3, len(close) // 4)]]
    return float(np.median(x.ravel()[close])), float(np.median(y.ravel()[close]))


def _world_points(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * points[:, 0] - s * points[:, 1], y + s * points[:, 0] + c * points[:, 1]


class DepthPlanner:
    def __init__(self, track, plan, cfg):
        self.track = track
        self.plan = plan
        self.pc = cfg["formula_sub_zero"]
        self.points = np.empty((0, 2), dtype=np.float32)
        self.capture_pose = None
        self.capture_stamp = None
        self.scan = None
        self.azimuth = None
        self.camera_y = 0.0
        self.signal_raw_xy = None
        self.signal_stamp = None
        self.offset_stations = None
        self.offsets = None
        self.clearances = None
        self.blocked_distance = 0.0
        self.last_clearance = math.inf
        self.map_consistent = False
        self._consistency_seen = False
        self._clean_frames = 0
        self.status = "waiting for depth"
        collision = cfg["collision"]
        self.half_length = float(collision["chassis_half_length"])
        self.half_width = float(collision["wheel_outer_y"])

    def update_local(self, points, stamp):
        """Keep car-frame depth for corridor steering without a pose match."""
        self.points = np.asarray(points, dtype=np.float32)
        self.capture_stamp = stamp

    def update(self, points, raw_pose, stamp, scan, azimuth, camera_y):
        self.points = points
        self.capture_pose = tuple(raw_pose)
        self.capture_stamp = stamp
        self.scan = np.asarray(scan, dtype=float)
        self.azimuth = np.asarray(azimuth, dtype=float)
        self.camera_y = camera_y

    def update_signal(self, point, raw_pose, stamp):
        sx, sy = _world_points(np.asarray([point]), raw_pose)
        self.signal_raw_xy = float(sx[0]), float(sy[0])
        self.signal_stamp = stamp

    def current_points(self, raw_pose):
        if self.capture_pose is None:
            return np.empty((0, 2), dtype=np.float32)
        dx = raw_pose[0] - self.capture_pose[0]
        dy = raw_pose[1] - self.capture_pose[1]
        if math.hypot(dx, dy) > 1.0:
            return np.empty((0, 2), dtype=np.float32)
        wx, wy = _world_points(self.points, self.capture_pose)
        c, s = math.cos(raw_pose[2]), math.sin(raw_pose[2])
        x = c * (wx - raw_pose[0]) + s * (wy - raw_pose[1])
        y = -s * (wx - raw_pose[0]) + c * (wy - raw_pose[1])
        return np.column_stack((x, y))

    def _start_candidates(self, points, ds, lateral, heading):
        ss, ll, hh = np.meshgrid(ds, lateral, heading, indexing="ij")
        ss, ll, hh = ss.ravel(), ll.ravel(), hh.ravel()
        station = self.track.start_station + ss
        tx = self.track.at(station, self.track.tx)
        ty = self.track.at(station, self.track.ty)
        x = self.track.at(station, self.track.x) - ty * ll
        y = self.track.at(station, self.track.y) + tx * ll
        yaw = np.arctan2(ty, tx) + hh
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        px = x[:, None] + c * points[None, :, 0] - s * points[None, :, 1]
        py = y[:, None] + s * points[None, :, 0] + c * points[None, :, 1]
        error = np.abs(self.track.clearance(px.ravel(), py.ravel())).reshape(px.shape)
        kept = max(1, int(0.8 * points.shape[0]))
        point_score = np.partition(np.minimum(error, 0.35), kept - 1, axis=1)[:, :kept].mean(axis=1)
        prior = 0.01 * (ss / 1.5) ** 2 + 0.004 * (ll / 0.5) ** 2
        poses = np.column_stack((x, y, yaw))
        return poses, point_score + prior, np.column_stack((ss, ll, hh))

    def _map_ranges(self, poses, azimuth):
        """Raycast a small set of candidate starts against the mapped bales."""
        yaw = poses[:, 2, None]
        c, s = np.cos(yaw), np.sin(yaw)
        ox = poses[:, 0, None] + c * 0.315 - s * self.camera_y
        oy = poses[:, 1, None] + s * 0.315 + c * self.camera_y
        angles = yaw + azimuth[None, :]
        dx, dy = np.cos(angles), np.sin(angles)
        ray = np.full(dx.shape, 0.3)
        hit = np.zeros(dx.shape, dtype=bool)
        for _ in range(70):
            distance = self.track.clearance((ox + dx * ray).ravel(),
                                             (oy + dy * ray).ravel()).reshape(ray.shape)
            hit |= distance <= 0.025
            active = ~hit & (ray < 10.0)
            if not active.any():
                break
            ray += np.where(active, np.maximum(0.03, 0.7 * distance), 0.0)
        return np.where(hit & (ray <= 10.0), ray, np.inf)

    def _choose_start(self, poses, point_score, offsets, count):
        candidates = np.argsort(point_score)[:count]
        # A point can land on an adjacent repeated bale even for the wrong
        # pose. The range along the whole ray disambiguates those matches.
        valid = np.isfinite(self.scan) & (self.scan > 0.3) & (self.scan < 9.8)
        if valid.sum() < 12:
            return None
        observed = self.scan[valid]
        predicted = self._map_ranges(poses[candidates], self.azimuth[valid])
        range_error = np.minimum(np.abs(predicted - observed[None, :]), 2.0)
        score = point_score[candidates] + 0.25 * np.median(range_error, axis=1)
        best = int(np.argmin(score))
        return tuple(poses[candidates[best]]), float(score[best]), tuple(offsets[candidates[best]])

    def match_start(self, raw_pose):
        points = self.current_points(raw_pose)
        points = points[(points[:, 0] > 0.5) & (points[:, 0] < 6.0)]
        if len(points) < 35 or self.scan is None:
            return None, "too few bale depth points for start alignment"
        points = points[np.linspace(0, len(points) - 1, min(120, len(points)), dtype=int)]
        poses, point_score, offsets = self._start_candidates(
            points, np.arange(-1.5, 1.501, 0.25),
            np.arange(-0.45, 0.451, 0.15), np.arange(-0.35, 0.351, 0.07),
        )
        coarse = self._choose_start(poses, point_score, offsets, 80)
        if coarse is None:
            return None, "too few valid depth rays for start alignment"
        _, _, (ds, lateral, heading) = coarse
        poses, point_score, offsets = self._start_candidates(
            points, np.arange(ds - 0.15, ds + 0.151, 0.05),
            np.arange(lateral - 0.075, lateral + 0.076, 0.025),
            np.arange(heading - 0.07, heading + 0.071, 0.035),
        )
        refined = self._choose_start(poses, point_score, offsets, 80)
        if refined is None:
            return None, "too few valid depth rays for start alignment"
        pose, score, _ = refined
        if score > float(self.pc["start_match_max_error_m"]):
            return None, f"bale map match error {score:.2f} m"
        if (self.signal_raw_xy is None or self.signal_stamp is None or
                self.capture_stamp - self.signal_stamp > 5.0):
            return None, "no recent depth range to the start signal"
        dx = self.signal_raw_xy[0] - raw_pose[0]
        dy = self.signal_raw_xy[1] - raw_pose[1]
        c, s = math.cos(raw_pose[2]), math.sin(raw_pose[2])
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        c, s = math.cos(pose[2]), math.sin(pose[2])
        predicted_x = pose[0] + c * local_x - s * local_y
        predicted_y = pose[1] + s * local_x + c * local_y
        signal_x = float(self.pc["start_signal_x_m"])
        signal_y = float(self.pc["start_signal_y_m"])
        correction_x = signal_x - predicted_x
        residual_y = signal_y - predicted_y
        if abs(correction_x) > 1.5 or abs(residual_y) > 0.8:
            return None, "start signal and bale geometry disagree"
        # The detector pixel lands on a different part of the rendered arm
        # than the mapped mount center. Use bale geometry for lateral pose.
        pose = (pose[0] + correction_x, pose[1], pose[2])
        return (pose, f"bale error {score:.2f} m, signal x correction "
                f"{correction_x:+.2f} m, y residual {residual_y:+.2f} m")

    def _point_clearance(self, cx, cy, yaw, px, py):
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        dx, dy = px[None, :] - cx[:, None], py[None, :] - cy[:, None]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        qx, qy = np.abs(lx) - self.half_length, np.abs(ly) - self.half_width
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        return (outside + np.minimum(np.maximum(qx, qy), 0)).min(axis=1)

    def _update_map_consistency(self, unexpected_count, point_count):
        high = max(6, int(point_count * float(self.pc["unexpected_point_fraction"])))
        low = max(2, high // 2)
        if not self._consistency_seen:
            self.map_consistent = unexpected_count < high
            self._consistency_seen = True
        elif unexpected_count >= high:
            self.map_consistent = False
            self._clean_frames = 0
        elif not self.map_consistent:
            self._clean_frames = self._clean_frames + 1 if unexpected_count <= low else 0
            if self._clean_frames >= int(self.pc["map_clean_frames"]):
                self.map_consistent = True

    def replan(self, station, track_pose, raw_pose):
        points = self.current_points(raw_pose)
        if len(points) < int(self.pc["min_depth_points"]):
            self.blocked_distance = 0.0
            self.status = f"only {len(points)} bale points"
            return False
        px, py = _world_points(points, track_pose)
        # Keep the measured racing line when depth describes the mapped bales.
        # Replanning nominal frames can erase its hairpin corrections.
        unexpected = np.abs(self.track.clearance(px, py)) > float(
            self.pc["unexpected_surface_error_m"])
        count = int(unexpected.sum())
        # New geometry takes effect immediately; returning to the map needs
        # repeated clean frames.
        self._update_map_consistency(count, len(points))
        step = float(self.pc["local_path_step_m"])
        horizon = float(self.pc["local_path_horizon_m"])
        distances = np.arange(0.0, horizon + 0.001, step)
        stations = station + distances
        offsets = np.arange(
            -float(self.pc["local_max_offset_m"]),
            float(self.pc["local_max_offset_m"]) + 0.001,
            float(self.pc["local_offset_step_m"]),
        )
        prior_offset = (np.zeros_like(stations) if self.offset_stations is None else
                        np.interp(stations, self.offset_stations, self.offsets,
                                  left=self.offsets[0], right=self.offsets[-1]))
        base_x = self.plan.at(stations, self.plan.x)
        base_y = self.plan.at(stations, self.plan.y)
        yaw = self.plan.yaw_at(stations)
        x = base_x[:, None] - np.sin(yaw[:, None]) * offsets[None, :]
        y = base_y[:, None] + np.cos(yaw[:, None]) * offsets[None, :]
        shape = x.shape
        yyaw = np.broadcast_to(yaw[:, None], shape).ravel()
        map_clear = self.track.body_clearance(
            x.ravel(), y.ravel(), yyaw, self.half_length, self.half_width,
        ).reshape(shape)
        live_clear = self._point_clearance(x.ravel(), y.ravel(), yyaw, px, py).reshape(shape)
        hard = float(self.pc["min_live_clearance_m"])
        feasible = (live_clear >= hard) & (map_clear >= float(self.pc["min_map_clearance_m"]))
        score = (2.0 * offsets[None, :] ** 2
                 + (30.0 if self.map_consistent else 0.0) * offsets[None, :] ** 2
                 + float(self.pc["local_temporal_weight"]) * (
                     offsets[None, :] - prior_offset[:, None]) ** 2
                 + 130.0 * np.maximum(float(self.pc["target_clearance_m"]) - live_clear, 0) ** 2
                 + 60.0 * np.maximum(float(self.pc["target_clearance_m"]) - map_clear, 0) ** 2)
        score = np.broadcast_to(score, shape).copy()
        score[~feasible] = np.inf
        first_blocked = next((i for i in range(len(stations)) if not feasible[i].any()), len(stations))
        self.blocked_distance = float(distances[first_blocked]) if first_blocked < len(stations) else horizon
        if first_blocked < 2:
            self.status = f"no safe corridor within {self.blocked_distance:.1f} m"
            return False
        score = score[:first_blocked]
        parent = np.zeros(score.shape, dtype=np.int16)
        current = score[0].copy()
        current += 5.0 * offsets**2
        if self.offset_stations is not None:
            current += float(self.pc["local_temporal_weight"]) * (offsets - prior_offset[0]) ** 2
        max_step = float(self.pc["local_max_offset_change_m"])
        transition = 15.0 * (offsets[:, None] - offsets[None, :]) ** 2
        transition[np.abs(offsets[:, None] - offsets[None, :]) > max_step + 1e-6] = np.inf
        for i in range(1, len(score)):
            candidate = current[:, None] + transition
            parent[i] = np.argmin(candidate, axis=0)
            current = score[i] + candidate[parent[i], np.arange(len(offsets))]
        if not np.isfinite(current).any():
            self.status = "no reachable safe local path"
            return False
        chosen = np.zeros(len(score), dtype=np.int16)
        chosen[-1] = np.argmin(current)
        for i in range(len(score) - 1, 0, -1):
            chosen[i - 1] = parent[i, chosen[i]]
        self.offset_stations = stations[:first_blocked]
        self.offsets = offsets[chosen]
        rows = np.arange(first_blocked)
        self.clearances = np.minimum(map_clear[rows, chosen], live_clear[rows, chosen])
        self.last_clearance = float(live_clear[0, chosen[0]])
        mode = "mapped" if self.map_consistent else f"replanned ({int(unexpected.sum())} new points)"
        self.status = (f"{len(points)} points, {mode}, clear {self.last_clearance:.2f} m, "
                       f"path {self.blocked_distance:.1f} m")
        return True

    def clearance_at(self, station):
        station = np.asarray(station, dtype=float)
        if self.offset_stations is None:
            return np.full_like(station, 0.20)
        return np.interp(station, self.offset_stations, self.clearances,
                         left=self.clearances[0], right=self.clearances[-1])

    def reference(self, station):
        station = np.asarray(station, dtype=float)
        x = self.plan.at(station, self.plan.x)
        y = self.plan.at(station, self.plan.y)
        yaw = self.plan.yaw_at(station)
        kappa = self.plan.at(station, self.plan.kappa)
        if self.offset_stations is None:
            return x, y, yaw, kappa
        offset = np.interp(station, self.offset_stations, self.offsets,
                           left=self.offsets[0], right=self.offsets[-1])
        slope = np.gradient(self.offsets, self.offset_stations)
        local_slope = np.interp(station, self.offset_stations, slope,
                                left=slope[0], right=slope[-1])
        x = x - np.sin(yaw) * offset
        y = y + np.cos(yaw) * offset
        yaw = yaw + np.arctan(local_slope)
        return x, y, yaw, kappa
