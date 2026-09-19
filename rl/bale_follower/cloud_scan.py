"""ZED point cloud -> the 36-bin range scan the policy consumes.

This is the piece that makes a trained policy deployable. Until now the
observation came from `bale_geometry.lidar_scan`, which ray-casts against bale
positions read out of the world SDF using ground-truth pose -- neither of
which exists on the robot. A policy trained on that cannot run outside the
simulator at all, however good its numbers look.

So the scan is built from a point cloud instead, by code that does not know
whether the cloud was rendered by Gazebo or captured by a real ZED 2i. Train
against the simulated camera, deploy against the real one, same function.

The conversion is deliberately crude, because every step of cleverness is a
step the real camera can fail differently at:

  1. drop points outside a height band -- the ground plane and anything
     overhead are not obstacles;
  2. drop points beyond max_range or behind the camera;
  3. bin what is left by bearing and take the nearest return in each bin;
  4. empty bins read max_range, which is what "no bale" has always meant to
     this policy.

Points arrive in the REP-103 body convention: +x forward, +y left, +z up.
That is what Gazebo's rgbd_camera puts on its `points` topic (verified by
inspecting a live cloud: x spanned 0.25-12.7 m while y and z stayed near
zero), and what the ZED ROS 2 wrapper publishes on
`point_cloud/cloud_registered`. It is NOT the optical convention (+z forward,
+y down) used by raw depth images -- assuming that silently returns an empty
scan, because every point fails the forward test.
"""

from __future__ import annotations

import math

import numpy as np


def _scan_tracking_the_ground(
    ranges: np.ndarray,
    bearings: np.ndarray,
    heights: np.ndarray,
    num_bins: int,
    fov_deg: float,
    max_range: float,
    min_range: float,
    ground_step: float,
    cell_m: float,
) -> np.ndarray:
    """Nearest *obstacle* per bearing, following the ground as it climbs.

    A fixed height band cannot do this course. The ramp climbs 0.635 m in
    3.3 m, so 2 m ahead its surface is 0.38 m above the car's own ground --
    higher than the top of a 0.356 m bale at the same range. Any threshold
    that lets the ramp through lets the bales through with it. Measured
    against `min_height` alone, the ramp read as a wall 2.18 m ahead and was
    indistinguishable from a bale at 3.0 m, so climbing it looked exactly
    like driving into one.

    Height does not separate them; *shape* does. Ground and ramp rise
    smoothly with range, a bale or a bucket is a step. So walk outward along
    each bearing carrying an estimate of where the drivable surface is: a
    return that continues it (within `ground_step`) updates the estimate and
    is not an obstacle, and the first return standing higher than that is
    the obstacle for that bearing. A bale standing *on* the ramp is still
    found, because it steps above the ramp the walk has been tracking.

    Deliberately a rule about the world rather than about the simulator: the
    real car climbs the same ramp with the same camera and needs the same
    distinction.
    """
    scan = np.full(num_bins, max_range, dtype=np.float32)
    half = fov_deg / 2.0
    edges = np.linspace(-half, half, num_bins + 1)
    index = np.clip(np.digitize(bearings, edges) - 1, 0, num_bins - 1)

    for current in range(num_bins):
        in_bin = index == current
        if not in_bin.any():
            continue
        bin_ranges = ranges[in_bin]
        bin_heights = heights[in_bin]
        order = np.argsort(bin_ranges)
        bin_ranges = bin_ranges[order]
        bin_heights = bin_heights[order]

        # Seed on the closest returns, which are the ground the car is
        # standing on; if the near field dropped out, the first cell seeds
        # it instead.
        surface = float(bin_heights[: max(1, min(8, bin_heights.size))].min())
        start = float(bin_ranges[0])
        cells = np.floor((bin_ranges - start) / cell_m).astype(np.int64)
        for cell in range(int(cells[-1]) + 1):
            here = cells == cell
            if not here.any():
                continue  # a gap in the cloud is not a step
            cell_heights = bin_heights[here]
            above = cell_heights > surface + ground_step
            if above.any():
                scan[current] = max(min_range, float(bin_ranges[here][above].min()))
                break
            # Everything in this cell continues the drivable surface, so
            # follow it -- this is what lets the walk climb the ramp and
            # descend the helix without either reading as a wall.
            surface = float(cell_heights.min())
    return scan


def scan_from_points(
    points: np.ndarray,
    num_bins: int,
    fov_deg: float,
    max_range: float,
    min_height: float = -0.12,
    max_height: float = 0.80,
    min_range: float = 0.15,
    pitch: float = 0.0,
    roll: float = 0.0,
    ground_step: float | None = None,
    cell_m: float = 0.25,
) -> np.ndarray:
    """Reduce an (N, 3) cloud in the body frame to a bearing-binned scan.

    Heights are relative to the camera, which sits ~0.25 m above the ground,
    so a level camera sees the ground at about -0.25. The default band starts
    above that to discard it and extends past the top of a 0.356 m bale.

    `pitch` and `roll` (radians, nose-down positive) level the cloud before
    the height test, and passing them accurately matters more than any other
    parameter here. The height band only rejects the ground while the camera
    is level: measured on a synthetic ground plane, 3 degrees of nose-down
    pitch fills every bin with phantom returns at 2.57 m, and 10 degrees puts
    an apparent wall 0.70 m ahead. A braking RC car pitches that much easily,
    so an unlevelled scan hallucinates an obstacle exactly when the car is
    working hardest. In sim these come from the pose quaternion; on the robot,
    from the ZED's IMU gravity vector.
    """
    scan = np.full(num_bins, max_range, dtype=np.float32)
    if points.size == 0:
        return scan

    if pitch or roll:
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        # Body frame: +x forward, +y left, +z up. Undo pitch about +y, then
        # roll about +x, so "height" below is measured against gravity.
        x0, y0, z0 = points[:, 0], points[:, 1], points[:, 2]
        x1 = x0 * cp + z0 * sp
        z1 = -x0 * sp + z0 * cp
        y1 = y0 * cr - z1 * sr
        z2 = y0 * sr + z1 * cr
        points = np.stack([x1, y1, z2], axis=1)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    keep = finite & (x > 0.0) & (z < max_height)
    # The ground-tracking walk needs the ground returns the height band is
    # there to throw away, so it does its own rejection further down.
    if ground_step is None:
        keep &= z > min_height
    if not keep.any():
        return scan

    if ground_step is not None:
        kx, ky, kz = x[keep], y[keep], z[keep]
        kranges = np.hypot(kx, ky)
        kbearings = np.degrees(np.arctan2(ky, kx))
        window = (
            (kranges <= max_range)
            & (kbearings >= -fov_deg / 2.0)
            & (kbearings <= fov_deg / 2.0)
        )
        if not window.any():
            return scan
        return _scan_tracking_the_ground(
            kranges[window],
            kbearings[window],
            kz[window],
            num_bins,
            fov_deg,
            max_range,
            min_range,
            ground_step,
            cell_m,
        )

    x, y = x[keep], y[keep]
    # Ground-plane range and bearing. Bearing is positive to the LEFT, to
    # match bale_geometry.lidar_scan, whose bin 0 is the rightmost ray.
    ranges = np.hypot(x, y)
    bearings = np.degrees(np.arctan2(y, x))

    half = fov_deg / 2.0
    inside = (
        (ranges >= min_range)
        & (ranges <= max_range)
        & (bearings >= -half)
        & (bearings <= half)
    )
    if not inside.any():
        return scan

    ranges, bearings = ranges[inside], bearings[inside]
    edges = np.linspace(-half, half, num_bins + 1)
    index = np.clip(np.digitize(bearings, edges) - 1, 0, num_bins - 1)
    # Nearest return per bin. np.minimum.at is the unbuffered form, so
    # repeated indices all take effect instead of only the last one.
    np.minimum.at(scan, index, ranges.astype(np.float32))
    return scan


def scan_from_depth(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    num_bins: int,
    fov_deg: float,
    max_range: float,
    stride: int = 2,
    **kwargs,
) -> np.ndarray:
    """Same scan, projected from a depth image instead of a cloud.

    Preferred over the cloud path for sim-to-real work. A point cloud arrives
    already reprojected by whoever produced it, so it carries that producer's
    intrinsics, frame convention and invalid-point encoding -- and Gazebo and
    the ZED wrapper do not agree on all three. (This module's first version
    assumed the optical convention and returned an entirely empty scan against
    Gazebo's body-frame cloud.) Projecting here instead means the same code,
    with intrinsics carried in-band by camera_info, runs in both worlds.

    `stride` subsamples rows and columns. A 640x360 depth image is 230k
    points and the scan keeps only the nearest return per bearing bin, so
    every second pixel loses nothing that matters and quarters the work.
    """
    depth = np.asarray(depth, dtype=np.float32)[::stride, ::stride]
    rows, cols = np.indices(depth.shape, dtype=np.float32)
    u = cols * stride
    v = rows * stride

    z = depth.ravel()  # optical +z, forward
    u = u.ravel()
    v = v.ravel()
    valid = np.isfinite(z) & (z > 0.0) & (z <= max_range * 1.5)
    if not valid.any():
        return np.full(num_bins, max_range, dtype=np.float32)

    z = z[valid]
    x_optical = (u[valid] - cx) * z / fx  # +x right
    y_optical = (v[valid] - cy) * z / fy  # +y down

    # Optical -> body (REP 103): forward, left, up. scan_from_points works in
    # body frame, so converting here keeps one binning implementation.
    points = np.stack([z, -x_optical, -y_optical], axis=1)
    return scan_from_points(points, num_bins, fov_deg, max_range, **kwargs)


def depth_from_image_msg(msg) -> np.ndarray:
    """(H, W) float32 metres from a sensor_msgs/Image.

    Accepts the two encodings that turn up in practice: 32FC1 metres (Gazebo's
    rgbd_camera and the ZED's depth_registered) and 16UC1 millimetres, which
    some depth drivers publish instead.
    """
    if msg.encoding == "32FC1":
        data = np.frombuffer(msg.data, dtype=np.float32)
    elif msg.encoding == "16UC1":
        data = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32) / 1000.0
    else:
        raise ValueError(f"unsupported depth encoding {msg.encoding!r}")
    return data.reshape(msg.height, msg.width)


def points_from_pointcloud2(msg) -> np.ndarray:
    """(N, 3) float32 xyz from a sensor_msgs/PointCloud2, without ros_numpy.

    Reads the x/y/z field offsets out of the message rather than assuming the
    usual 16-byte layout: Gazebo's rgbd_camera and the ZED wrapper do not
    agree on what else rides along in each point.
    """
    offsets = {f.name: f.offset for f in msg.fields if f.name in ("x", "y", "z")}
    if len(offsets) != 3:
        raise ValueError(
            f"cloud has no xyz fields, only {[f.name for f in msg.fields]}"
        )

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    stride = msg.point_step
    count = len(raw) // stride
    raw = raw[: count * stride].reshape(count, stride)
    return np.stack(
        [
            raw[:, o : o + 4].copy().view(np.float32).ravel()
            for o in (offsets["x"], offsets["y"], offsets["z"])
        ],
        axis=1,
    )
