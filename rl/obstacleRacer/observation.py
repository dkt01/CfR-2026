"""The policy's observation, its steering prior, and its action mapping.

ONE module, imported by both the numpy trainer (env.py) and the car's ROS
node (obstacle_racer_node.py), so the vector the policy was trained on and
the vector it drives on are built by the same lines.  The last obstacle
course attempt built 44 values per frame in training and 38 on the car.

Everything in here must be computable on the car: numpy only, and only from
the segmented ZED cloud (scan + gates, as `cloud_segmentation.py`'s
`scan_from_segmentation` and `Segmentation.gates` give them), the wheel speed
the Arduino reports, and the yaw rate.  No pose, no map, no centerline.

Per frame (FRAME_DIM = 49):

    scan      36  nearest blocking return per bearing bin / max_range,
                  bin 0 rightmost (scan_from_segmentation's order)
    hoop       4  valid, bearing / half-FOV, range / max_range,
                  gate normal relative to heading / (pi/2)
    car wash   4  the same, for the nearest visible arch
    speed      1  tachometer speed / v_cap (reads 0 below 0.3 m/s)
    yaw rate   1  rad/s / 3
    action     2  previous raw action
    prior      1  the prior's steering command

Two frames are stacked, newest first.
"""

from __future__ import annotations

import math

import numpy as np

SCAN_BINS = 36
GATE_DIM = 8
FRAME_DIM = SCAN_BINS + GATE_DIM + 5
TACH_FLOOR = 0.30  # m/s: ArduinoStatus.speed cannot resolve below this
YAW_RATE_SCALE = 3.0

# Measured steering table, arduino_bridge.yaml: command -> road-wheel angle.
STEER_LEFT = 0.512
STEER_RIGHT = 0.382


def angle_to_command(angle):
    """Road-wheel angle (rad, +left) -> DriveCommand.steering in [-1, 1]."""
    angle = np.asarray(angle, dtype=np.float64)
    return np.clip(
        np.where(angle >= 0, angle / STEER_LEFT, angle / STEER_RIGHT), -1.0, 1.0
    )


def tach(speed):
    """What the hall-sensor tachometer reports for a true ground speed."""
    speed = np.asarray(speed, dtype=np.float64)
    return np.where(np.abs(speed) < TACH_FLOOR, 0.0, speed)


def prior_steer(scan, gate, yaw_rate, cfg):
    """Map-free steering: at a visible gate's center, else the best free gap.

    scan (B, bins) in meters, gate (B, 8) as the observation carries it,
    yaw_rate (B,) as measured.  Returns a steering command per car.

    The aim is damped by the yaw rate: the car answers the wheel 0.19 s
    (dead time) plus 0.34 s (chassis lag) late, and an undamped aim at the
    gap weaves the car into the wall within a few meters.

    The gap is the widest run of bins that read at least `gap_min_depth`,
    scored against its angle off dead ahead so a straight lane is not traded
    for a slightly wider gap to one side; with no open run, the single
    deepest bin.  This is the whole prior.  It does not drive the course --
    it keeps a fresh policy's steering exploration pointed down the lane
    instead of into the wall beside it, which is how most of the last
    attempt's episodes ended.
    """
    p = cfg["prior"]
    s = cfg["sensor"]
    scan = np.asarray(scan, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    B, bins = scan.shape
    half = math.radians(s["fov_deg"]) / 2
    width = 2 * half / bins
    centers = -half + width * (np.arange(bins) + 0.5)

    # Runs of open bins, all cars at once: a run starts where the padded row
    # steps 0 -> 1 and ends (exclusive) where it steps 1 -> 0; row-major
    # order pairs every start with its own end.
    open_ = scan >= float(p["gap_min_depth"])
    padded = np.zeros((B, bins + 2), np.int8)
    padded[:, 1:-1] = open_
    step = np.diff(padded, axis=1)
    rows, starts = np.nonzero(step == 1)
    _, ends = np.nonzero(step == -1)
    bearing = centers[np.argmax(scan, axis=1)]
    if len(rows):
        mid = 0.5 * (centers[starts] + centers[ends - 1])
        score = (ends - starts) * width - float(p["straight_bias"]) * np.abs(mid)
        order = np.lexsort((-score, rows))
        first = np.unique(rows[order], return_index=True)[1]
        pick = order[first]
        # Aim inside the chosen run at its depth-weighted centroid, not its
        # midpoint: in a curving lane (the helix) the deep bins are the ones
        # that point round the curve.
        depth = float(p["gap_min_depth"])
        weight = np.where(open_, (scan - depth + 0.1) ** 2, 0.0)
        cols = np.arange(bins)
        for q in pick:
            r = rows[q]
            span = (cols >= starts[q]) & (cols < ends[q])
            wr = weight[r] * span
            bearing[r] = (wr @ centers) / max(wr.sum(), 1e-9)
    damping = float(p["yaw_damping"]) * np.asarray(yaw_rate, dtype=np.float64)
    angle = float(p["gap_gain"]) * bearing

    # A visible gate wins: hoop first, then the car wash.
    for off in (4, 0):
        valid = gate[:, off] > 0.5
        angle = np.where(valid, float(p["gate_gain"]) * gate[:, off + 1] * half, angle)
    return angle_to_command(angle - damping)


HOOP_KIND, CARWASH_KIND = 2, 3


def gate_features(gates, cfg):
    """The 8 gate values from the segmenter's gates, for one car.

    `gates` is an iterable of (kind, center_x, center_y, axis_x, axis_y) in
    the leveled camera frame -- `cloud_segmentation.Gate`'s kind, center and
    axis.  For each kind, the nearest one gives: valid, bearing / half-FOV,
    range / max_range, and the gate's normal (turned away from the car) as an
    angle off the heading / (pi/2).  sensor.py computes the same four numbers
    from the course model; selftest.py checks the two agree.
    """
    s = cfg["sensor"]
    half = math.radians(float(s["fov_deg"])) / 2
    max_range = float(s["max_range"])
    out = np.zeros(GATE_DIM)
    for slot, kind in enumerate((HOOP_KIND, CARWASH_KIND)):
        best = None
        for g_kind, cx, cy, ax, ay in gates:
            if int(g_kind) != kind:
                continue
            r = math.hypot(cx, cy)
            if best is None or r < best[0]:
                best = (r, cx, cy, ax, ay)
        if best is None:
            continue
        r, cx, cy, ax, ay = best
        nx, ny = -ay, ax
        if nx * cx + ny * cy < 0:
            nx, ny = -nx, -ny
        out[4 * slot : 4 * slot + 4] = (
            1.0,
            math.atan2(cy, cx) / half,
            r / max_range,
            float(np.clip(math.atan2(ny, nx) / (0.5 * math.pi), -2.0, 2.0)),
        )
    return out


def smooth_prior(previous, raw, fresh, cfg):
    """First-order low-pass on the prior's command, tau = prior.smoothing_s.

    The gap search can jump between two near-equal gaps from one frame to the
    next; unfiltered, that is steering chatter the policy would have to spend
    its residual cancelling.  `fresh` marks cars on their first frame, which
    take the raw value.
    """
    dt = 1.0 / float(cfg["env"]["control_hz"])
    alpha = dt / (float(cfg["prior"]["smoothing_s"]) + dt)
    out = np.asarray(previous) + alpha * (np.asarray(raw) - np.asarray(previous))
    if fresh is not None:
        out = np.where(fresh, raw, out)
    return out


def frame(scan, gate, speed_meas, yaw_rate_meas, prev_action, prior, cfg):
    """One frame of the observation, (B, FRAME_DIM) float32."""
    s = cfg["sensor"]
    v_cap = float(cfg["env"]["v_cap"])
    return np.concatenate(
        [
            np.asarray(scan) / float(s["max_range"]),
            np.asarray(gate),
            (np.asarray(speed_meas) / v_cap)[:, None],
            np.clip(np.asarray(yaw_rate_meas) / YAW_RATE_SCALE, -3, 3)[:, None],
            np.asarray(prev_action),
            np.asarray(prior)[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def action_to_command(action, prior, cfg):
    """Raw policy action (B, 2) -> (steering command, target speed m/s)."""
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    scale = float(cfg["prior"]["residual_scale"])
    steer = np.clip(prior + scale * action[:, 0], -1.0, 1.0)
    speed = float(cfg["env"]["v_cap"]) * 0.5 * (action[:, 1] + 1.0)
    return steer, speed


class Stack:
    """The last `depth` frames, newest first, flattened."""

    def __init__(self, n, depth):
        self.buf = np.zeros((n, depth, FRAME_DIM), np.float32)

    def reset(self, idx, first):
        self.buf[idx] = first[:, None, :]

    def push(self, f):
        self.buf[:, 1:] = self.buf[:, :-1]
        self.buf[:, 0] = f
        return self.buf.reshape(len(self.buf), -1)

    @property
    def obs(self):
        return self.buf.reshape(len(self.buf), -1)
