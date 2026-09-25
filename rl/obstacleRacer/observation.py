"""The policy's observation, its steering prior, and its action mapping.

ONE module, imported by both the numpy trainer (env.py) and the car's ROS
node (obstacle_racer_node.py), so the vector the policy was trained on and
the vector it drives on are built by the same lines.  The last obstacle
course attempt built 44 values per frame in training and 38 on the car.

Everything in here must be computable on the car: numpy only, and only from
the segmented ZED cloud (scan + gates, as `cloud_segmentation.py`'s
`scan_from_segmentation` and `Segmentation.gates` give them), the wheel speed
the Arduino reports, and the yaw rate.  No pose, no map, no centerline.

Per frame (FRAME_DIM = 49, or 52 with env.memory_features):

    scan      36  nearest blocking return per bearing bin / max_range,
                  bin 0 rightmost (scan_from_segmentation's order)
    hoop       4  valid, bearing / half-FOV, range / max_range,
                  gate normal relative to heading / (pi/2)
    car wash   4  the same, for the nearest visible arch
    speed      1  ArduinoStatus.speed / v_cap: tachometer magnitude (0 below
                  0.3 m/s), signed by the speed controller's direction
    yaw rate   1  rad/s / 3
    action     2  previous raw action
    prior      1  the prior's steering command
    memory     3  with env.memory_features (Memory, below): time the tach has
                  read stopped / 3 s, the tach speed averaged over ~2 s / v_cap,
                  and the direction the controller last reported (+1 / -1)

Frames are stacked newest first: env.frame_offsets control steps back (v5:
0, 2, 10, 20 -- now, 0.1, 0.5 and 1.0 s ago), or the last env.frame_stack
steps in older configs.  A second of scan history covers what the 110 deg
camera loses beside the car: hoop posts as it threads them, bales as it
turns past them.
"""

from __future__ import annotations

import math

import numpy as np

SCAN_BINS = 36
GATE_DIM = 8
FRAME_DIM = SCAN_BINS + GATE_DIM + 5
MEMORY_DIM = 3
STOP_CAP_S = 3.0  # Memory's stopped time saturates here
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
    """What ArduinoStatus.speed reports for a signed speed.

    The sensor gives the magnitude; the sign is the controller's direction
    estimate, so the caller passes |v| times that estimate, not the true
    signed velocity.
    """
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


def frame_dim(cfg):
    """Values per frame under this config."""
    return FRAME_DIM + (MEMORY_DIM if cfg["env"].get("memory_features") else 0)


def frame_offsets(cfg):
    """Control steps back of each stacked frame, newest first."""
    e = cfg["env"]
    if "frame_offsets" in e:
        return [int(k) for k in e["frame_offsets"]]
    return list(range(int(e["frame_stack"])))


def obs_dim(cfg):
    return frame_dim(cfg) * len(frame_offsets(cfg))


def frame(scan, gate, speed_meas, yaw_rate_meas, prev_action, prior, cfg, memory=None):
    """One frame of the observation, (B, frame_dim(cfg)) float32.

    `memory` is Memory.update's (B, 3), required when the config turns
    memory_features on and ignored otherwise.
    """
    s = cfg["sensor"]
    v_cap = float(cfg["env"]["v_cap"])
    parts = [
        np.asarray(scan) / float(s["max_range"]),
        np.asarray(gate),
        (np.asarray(speed_meas) / v_cap)[:, None],
        np.clip(np.asarray(yaw_rate_meas) / YAW_RATE_SCALE, -3, 3)[:, None],
        np.asarray(prev_action),
        np.asarray(prior)[:, None],
    ]
    if cfg["env"].get("memory_features"):
        if memory is None:
            raise ValueError("memory_features is on: pass Memory.update's output")
        parts.append(np.asarray(memory))
    return np.concatenate(parts, axis=1).astype(np.float32)


class Memory:
    """Slow state the policy cannot see in a second of frames, from the tach.

    All three come from ArduinoStatus.speed alone, updated once per control
    step, so the car computes them exactly as training does:

        stopped    s the tach has read zero, capped at STOP_CAP_S, / STOP_CAP_S:
                   how long the car has been stuck, which the frames cannot
                   tell apart from a moment ago
        mean speed the tach reading low-passed over memory_tau_s, / v_cap:
                   signed, so it says whether the car has been backing out
        direction  the sign of the last nonzero reading: the controller's
                   direction, which the tach hides below 0.3 m/s -- whether
                   reverse has engaged yet (+1 until anything is read)
    """

    def __init__(self, n, cfg):
        self.dt = 1.0 / float(cfg["env"]["control_hz"])
        self.tau = float(cfg["env"].get("memory_tau_s", 2.0))
        self.v_cap = float(cfg["env"]["v_cap"])
        self.stopped = np.zeros(n)
        self.mean_v = np.zeros(n)
        self.direction = np.ones(n)

    def update(self, speed_meas, idx=None, fresh=None):
        """Advance cars `idx` (all by default) by one step; returns (k, 3).

        `fresh` marks cars on their first frame: they start from their own
        reading (stopped 0, mean speed = the reading) rather than from the
        last episode's.
        """
        if idx is None:
            idx = np.arange(len(self.stopped))
        v = np.asarray(speed_meas, dtype=np.float64)
        if fresh is not None and np.any(fresh):
            f = idx[np.asarray(fresh, bool)]
            self.stopped[f] = 0.0
            self.mean_v[f] = v[np.asarray(fresh, bool)]
            self.direction[f] = 1.0
        moving = v != 0.0
        self.stopped[idx] = np.where(
            moving, 0.0, np.minimum(self.stopped[idx] + self.dt, STOP_CAP_S)
        )
        self.direction[idx] = np.where(moving, np.sign(v), self.direction[idx])
        a = self.dt / (self.tau + self.dt)
        self.mean_v[idx] += a * (v - self.mean_v[idx])
        return np.stack(
            [
                self.stopped[idx] / STOP_CAP_S,
                self.mean_v[idx] / self.v_cap,
                self.direction[idx],
            ],
            axis=1,
        )


def action_to_command(action, prior, cfg):
    """Raw policy action (B, 2) -> (steering command, target speed m/s).

    Speed runs linearly from -v_reverse at -1 to v_cap at +1.  A reverse
    target while rolling forward is the Arduino's cue to coast to a stop
    (it drives backwards only once the wheels read stopped).
    """
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    scale = float(cfg["prior"]["residual_scale"])
    steer = np.clip(prior + scale * action[:, 0], -1.0, 1.0)
    lo = -float(cfg["env"].get("v_reverse", 0.0))
    hi = float(cfg["env"]["v_cap"])
    speed = lo + (hi - lo) * 0.5 * (action[:, 1] + 1.0)
    return steer, speed


def speed_to_action(speed, cfg):
    """Inverse of action_to_command's speed mapping: m/s -> action[1]."""
    lo = -float(cfg["env"].get("v_reverse", 0.0))
    hi = float(cfg["env"]["v_cap"])
    return 2.0 * (np.asarray(speed, dtype=np.float64) - lo) / (hi - lo) - 1.0


class Stack:
    """Frames `offsets` control steps back (0 = newest), newest first, flattened.

    Keeps every frame back to the oldest offset; a reset fills the history
    with the first frame, so a fresh episode reads as having stood there.
    """

    def __init__(self, n, offsets, dim=FRAME_DIM):
        if isinstance(offsets, int):  # a plain depth: the last `offsets` frames
            offsets = range(offsets)
        self.offsets = np.asarray(list(offsets), np.int64)
        assert self.offsets[0] == 0 and (np.diff(self.offsets) > 0).all()
        self.buf = np.zeros((n, int(self.offsets[-1]) + 1, dim), np.float32)

    def reset(self, idx, first):
        self.buf[idx] = first[:, None, :]

    def push(self, f):
        self.buf[:, 1:] = self.buf[:, :-1]
        self.buf[:, 0] = f
        return self.obs

    @property
    def obs(self):
        return self.buf[:, self.offsets].reshape(len(self.buf), -1)
