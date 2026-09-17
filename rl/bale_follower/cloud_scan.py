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
    keep = finite & (x > 0.0) & (z > min_height) & (z < max_height)
    if not keep.any():
        return scan

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
    fx: float, fy: float, cx: float, cy: float,
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

    z = depth.ravel()          # optical +z, forward
    u = u.ravel()
    v = v.ravel()
    valid = np.isfinite(z) & (z > 0.0) & (z <= max_range * 1.5)
    if not valid.any():
        return np.full(num_bins, max_range, dtype=np.float32)

    z = z[valid]
    x_optical = (u[valid] - cx) * z / fx   # +x right
    y_optical = (v[valid] - cy) * z / fy   # +y down

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
        raise ValueError(f"cloud has no xyz fields, only {[f.name for f in msg.fields]}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    stride = msg.point_step
    count = len(raw) // stride
    raw = raw[: count * stride].reshape(count, stride)
    return np.stack(
        [raw[:, o : o + 4].copy().view(np.float32).ravel() for o in
         (offsets["x"], offsets["y"], offsets["z"])],
        axis=1,
    )
