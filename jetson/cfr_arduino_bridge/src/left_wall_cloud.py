"""Cloud-based left-wall control for the physical ZED camera."""

import math

import numpy as np


def ranges_from_cloud(message):
    """Return bearing/range arrays from ZED's body-frame registered cloud.

    The ZED wrapper publishes +x forward, +y left, +z up. Heights are
    relative to the camera. Discard the floor below the bale faces.
    """
    fields = {field.name: field.offset for field in message.fields}
    if not all(axis in fields for axis in ("x", "y", "z")):
        raise ValueError("point cloud has no x/y/z fields")
    count = len(message.data) // message.point_step
    if count == 0:
        return np.empty(0), np.empty(0)
    raw = np.frombuffer(message.data, dtype=np.uint8, count=count * message.point_step)
    raw = raw.reshape(count, message.point_step)
    endian = ">f4" if message.is_bigendian else "<f4"
    xyz = [
        raw[:, fields[axis] : fields[axis] + 4].copy().view(endian).ravel()
        for axis in ("x", "y", "z")
    ]
    x, y, z = xyz
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x > 0.18)
        & (z > -0.12)
        & (z < 0.70)
    )
    x, y = x[valid], y[valid]
    return np.arctan2(y, x), np.hypot(x, y)


def sector_range(bearings, ranges, center_deg, half_width_deg=5, max_range=3.0):
    within = np.abs(bearings - math.radians(center_deg)) < math.radians(half_width_deg)
    values = ranges[within & (ranges < max_range)]
    return float(np.percentile(values, 10)) if values.size >= 5 else max_range


def cloud_command(bearings, ranges):
    """Slow forward speed and steering angle from two left rays and front space."""
    if bearings.size < 20:
        return 0.0, 0.0
    near = sector_range(bearings, ranges, 50)
    far = sector_range(bearings, ranges, 30)
    front = sector_range(bearings, ranges, 0, 15)
    if near >= 3.0 and far >= 3.0:
        return 0.35, 0.25  # reacquire the left boundary
    if near >= 3.0:
        side, tangent = far * math.sin(math.radians(30)), 0.0
    elif far >= 3.0:
        side, tangent = near * math.sin(math.radians(50)), 0.0
    else:
        side = near * math.sin(math.radians(50))
        ahead = far * math.sin(math.radians(30))
        forward_gap = far * math.cos(math.radians(30)) - near * math.cos(
            math.radians(50)
        )
        tangent = math.atan2(ahead - side, max(0.25, forward_gap))
    steer = 0.8 * tangent + 0.65 * (side - 0.50)
    if front < 1.1:
        steer += 0.9 * (1.1 - front)  # oval bends left at its ends
    steer = max(-0.40, min(0.40, steer))
    speed = 0.35 if front < 1.1 or abs(steer) > 0.3 else 0.60
    return speed, steer
