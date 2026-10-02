"""Plan Z's driving core: scan + pose hint in, steering and speed out.

No learning and no map of the obstacles.  Each camera frame (`observe`):

    remember   scan returns go into a short memory of points in the pose
               frame; a point a newer scan sees past, or measures again from
               nearer, is replaced.  This is what covers the sides of the
               car, which the 110 degree camera cannot see while it turns
               past a bale or a hoop post.
    path       in a walled lane, the route itself (route.py), slid sideways
               point by point to the middle of the walls beside it, measured
               off the remembered points -- to the centimeter, which the 20 in
               lane needs -- and eased across to the middle of each hoop
               ahead.  The same measurement moves the route to where the
               lane really is, which is how a drifting pose and a course
               built off the drawing are taken out.
    field      where the route gives no line (the Wide Section, the buckets,
               or something standing in the lane): a wavefront from the goal
               over a coarse grid of what is remembered.  Ground the camera
               has not looked at costs more, and beside a wall it is wall.

and each control tick (`step`):

    steer      in a lane: the path's own bend, read where the car will be
               once the command has taken hold, plus a pull onto the path
               from where the commands already sent will have put it.  By
               the field: whichever of the car's own arcs ends furthest down
               it, and when none gets nearer, the backing move after which
               one does.
    veto       the command is rolled out through the car's lag and checked
               against the remembered points with the body's real outline;
               if it would touch, the nearest command that does not is sent.
    speed      the least of the route's cap, what the bend allows, what can
               be coasted down from in the clear length ahead, and what the
               car's own steering lag can answer for.
    recovery   stopped while commanding speed, or nothing clear: back off,
               steering the nose toward the path, and plan again.

Three things about the car are learned as it drives, each bounded and each
with a switch: the steering trim, the lag between the wheel and the yaw, and
the camera's yaw on its mount.

ROS-free and numpy only, so selftest.py and sim_numpy.py drive exactly the
code plan_z_node.py runs on the car.

Frames: the car frame is at the chassis center, x forward, y left.  Poses are
(x, y, yaw) of the chassis center in the pose frame.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

WHEELBASE = 0.324
AXLE_BACK = 0.162  # the rear axle, behind the chassis center
BODY_HALF_WIDTH = 0.1625  # over the tires
# The body, from the rear axle: 0.55 long.
BODY_FRONT = 0.275 + AXLE_BACK
BODY_REAR = 0.275 - AXLE_BACK
# Measured steering table (arduino_bridge.yaml): command -> road-wheel angle.
STEER_LEFT = 0.512
STEER_RIGHT = 0.382
HOOP, CARWASH = 2, 3
SQRT2 = math.sqrt(2.0)

# The wavefront's grid, in the frame of the car at the last camera frame.
COARSE = 0.10
COARSE_X0, COARSE_Y0, COARSE_NX, COARSE_NY = -4.0, -6.0, 120, 120
CELL = 0.05  # memory keeps one point per cell this size
# Where the camera has looked and found nothing, as the time it last did, on
# a grid fixed in the pose frame and centered on the start.
SEEN_N = 1600  # 160 m square: the Speed Course is 41 m long

ARC_REACH = 6.5  # m: as far as a command is checked, sight at v_max
CANDIDATES = 25
MIN_FREE = 0.30  # m of clear way below which it is not worth starting
PATH_STEP = 0.10


def angle_to_command(angle):
    """Road-wheel angle (rad, + left) -> DriveCommand.steering in [-1, 1]."""
    limit = STEER_LEFT if angle >= 0 else STEER_RIGHT
    return max(-1.0, min(1.0, angle / limit))


def command_to_angle(command):
    return command * (STEER_LEFT if command >= 0 else STEER_RIGHT)


def to_frame(points, pose):
    """Pose-frame points (N, 2+) seen from `pose`, as (N, 2)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = points[:, 0] - pose[0], points[:, 1] - pose[1]
    return np.c_[c * dx + s * dy, -s * dx + c * dy]


def from_frame(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.c_[
        pose[0] + c * points[:, 0] - s * points[:, 1],
        pose[1] + s * points[:, 0] + c * points[:, 1],
    ]


def arc_clearance(x, y, kappa, half_width, lead=BODY_FRONT):
    """Distance an arc can be driven before the body meets a point.

    x, y (..., N) are points in the rear-axle frame; kappa and half_width
    broadcast against them (kappa with a trailing axis of 1).  The result
    drops the point axis and is ARC_REACH where nothing is met.  Exact for
    the body as a rectangle `half_width` either side of the axle's path: a
    point is met by whichever part of the body, front bumper to outer front
    corner, comes round to it first.  `lead` is how far the body's leading
    end is ahead of the axle: BODY_REAR, with x negated, checks backing up.
    """
    shape = np.broadcast_shapes(x.shape[:-1], kappa.shape[:-1])
    if x.shape[-1] == 0:
        return np.full(shape, ARC_REACH)
    straight = np.abs(kappa) < 1e-4
    radius = 1.0 / np.where(straight, 1.0, np.abs(kappa))
    yy = np.sign(kappa) * y  # mirrored so every turn is a left one
    rho2 = x**2 + (radius - yy) ** 2
    phi = np.arctan2(x, radius - yy)
    inner = radius - half_width
    x_max = np.minimum(lead, np.sqrt(np.maximum(rho2 - inner**2, 0.0)))
    x_min2 = rho2 - (radius + half_width) ** 2
    theta = phi - np.arcsin(
        np.clip(x_max / np.sqrt(np.maximum(rho2, 1e-12)), -1.0, 1.0)
    )
    turning = np.where(
        (rho2 >= inner**2) & (x_min2 <= x_max**2) & (theta > 0.0),
        radius * theta,
        np.inf,
    )
    ahead = np.where((np.abs(y) <= half_width) & (x > lead), x - lead, np.inf)
    return np.minimum(np.where(straight, ahead, turning).min(-1), ARC_REACH)


def _dilate(a):
    b = a.copy()
    b[1:] |= a[:-1]
    b[:-1] |= a[1:]
    c = b.copy()
    c[:, 1:] |= b[:, :-1]
    c[:, :-1] |= b[:, 1:]
    return c


def _smooth(path, passes=2):
    """A polyline with its corners rounded; the ends stay put."""
    for _ in range(passes):
        if len(path) < 3:
            break
        path = np.concatenate(
            [path[:1], 0.25 * path[:-2] + 0.5 * path[1:-1] + 0.25 * path[2:], path[-1:]]
        )
    return path


def _resample(path, step=PATH_STEP):
    seg = np.hypot(*np.diff(path, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] < step:
        return path
    at = np.arange(0.0, arc[-1] + 1e-9, step)
    return np.c_[np.interp(at, arc, path[:, 0]), np.interp(at, arc, path[:, 1])]


def wavefront(cost, blocked, goal, near, max_iter=400):
    """Cost-to-goal over an 8-connected grid; inf where it cannot be reached.

    `cost` is paid per cell entered (times sqrt 2 diagonally).  `goal` is
    (rows, columns, starting costs): several cells, each with the cost still
    to pay beyond it.  Sweeps until nothing changes, or until the cells in
    `near` (slices round the car) have been reached and have stopped
    changing: the far side of the grid does not matter to the car.
    """
    h, w = blocked.shape
    D = np.full((h + 2, w + 2), np.inf)
    inner = D[1:-1, 1:-1]
    np.minimum.at(inner, (goal[0], goal[1]), goal[2])
    diag = cost * SQRT2
    quiet = 0
    for _ in range(max_iter):
        straight = np.minimum(
            np.minimum(D[:-2, 1:-1], D[2:, 1:-1]), np.minimum(D[1:-1, :-2], D[1:-1, 2:])
        )
        corner = np.minimum(
            np.minimum(D[:-2, :-2], D[:-2, 2:]), np.minimum(D[2:, :-2], D[2:, 2:])
        )
        new = np.minimum(inner, np.minimum(straight + cost, corner + diag))
        new[blocked] = np.inf
        changed = new < inner
        inner[...] = new
        if not changed.any():
            break
        if changed[near].any() or not np.isfinite(inner[near]).any():
            quiet = 0
        else:
            quiet += 1
            if quiet >= 6:
                break
    return inner.copy()


class PoseDrift:
    """A test fault: the pose frame creeping away from the ground, as visual
    odometry does.  Position error grows by `per_m` and heading error by
    `yaw_per_m` for every meter driven."""

    def __init__(self, per_m=0.0, yaw_per_m=0.0):
        self.per_m, self.yaw_per_m = per_m, yaw_per_m
        self.last = None
        self.distance = 0.0

    def __call__(self, pose):
        if self.per_m == 0.0 and self.yaw_per_m == 0.0:
            return pose
        if self.last is not None:
            self.distance += math.hypot(pose[0] - self.last[0], pose[1] - self.last[1])
        self.last = pose
        psi = self.yaw_per_m * self.distance
        e = self.per_m * self.distance
        c, s = math.cos(psi), math.sin(psi)
        return (
            c * pose[0] - s * pose[1] + e * 0.6,
            s * pose[0] + c * pose[1] + e * 0.8,
            pose[2] + psi,
        )


# The yaw lags tried against the car's own answer to the wheel (auto_lag).
LAG_BANK = np.array([0.10, 0.15, 0.22, 0.33, 0.50, 0.75, 1.10])


class Planner:
    def __init__(self, knobs, route=None):
        self.k = knobs
        self.route = route if knobs["route_hint"] else None
        self.dt = 1.0 / float(knobs["control_hz"])
        self.bins = int(knobs["scan_bins"])
        self.half_fov = math.radians(float(knobs["fov_deg"])) / 2
        self.bin_width = 2 * self.half_fov / self.bins
        self.bearings = -self.half_fov + self.bin_width * (np.arange(self.bins) + 0.5)
        self.max_range = float(knobs["max_range"])
        self.cam_x = float(knobs["camera_forward"])
        ci, cj = np.meshgrid(np.arange(COARSE_NX), np.arange(COARSE_NY), indexing="ij")
        self.coarse_ij = (ci, cj)
        cell_x = COARSE_X0 + COARSE * (ci + 0.5)
        cell_y = COARSE_Y0 + COARSE * (cj + 0.5)
        self.cell_xy = np.stack([cell_x.ravel(), cell_y.ravel()], 1)
        self.cell_range = np.hypot(cell_x - self.cam_x, cell_y).ravel()
        self.cell_bearing = np.arctan2(cell_y, cell_x - self.cam_x).ravel()
        self.cell_near = (np.hypot(cell_x, cell_y) < 0.45).ravel()
        # The car turns wider than the bicycle model says (tires scrub):
        # measured 1.10 times the radius, either way, at any speed.
        self.wheelbase = WHEELBASE * float(knobs["turn_radius_scale"])
        use = float(knobs["curvature_use"])
        self.k_left = math.tan(STEER_LEFT * use) / self.wheelbase
        self.k_right = math.tan(STEER_RIGHT * use) / self.wheelbase
        half = CANDIDATES // 2
        u = np.linspace(0.0, 1.0, half + 1)[1:]
        self.kappa = np.concatenate([-self.k_right * u[::-1], [0.0], self.k_left * u])
        self.reset()

    def reset(self, pose=None, t=0.0):
        """Start of a run: forget everything, tie the route to this pose."""
        # Remembered returns: x, y in the pose frame, and the range noise
        # each was measured with (1 sigma, m).
        self.mem = np.zeros((0, 3))
        self.mem_t = np.zeros(0)
        self.far = np.zeros((0, 3))
        self.candidates = np.zeros((0, 2))
        self.scan = None
        self.seen_t = np.full((SEEN_N, SEEN_N), -np.inf, np.float32)
        self.seen_origin = (
            np.zeros(2) if pose is None else np.asarray(pose[:2], float)
        ) - 0.5 * SEEN_N * COARSE
        self.points = np.zeros((0, 3))
        self.path = None  # (M, 2) in the pose frame, PATH_STEP apart
        self.searched = False  # the path was searched for, not the route's
        self.searched_at = -1e9
        self.frames = 0
        self.grid_pose = None
        self.goal_local = np.zeros(2)
        self.D = None
        self.goal = None
        self.in_open = False
        self.gate = None  # (center xy, normal xy, half span), pose frame
        self.hoops = []  # every hoop seen ahead and not yet driven through
        self.passed = deque(maxlen=8)  # (t, center) of gates driven through
        self.mode = "drive"
        self.reverse = None
        self.history = deque()
        self.cmd_since = None
        self.last_pose = pose
        self.kappa_cmd = 0.0
        self.v_cmd = 0.0
        self.steer_out = 0.0
        self.sent = deque()  # (t, curvature) of the commands still in flight
        self.trim_est = 0.0
        self.angle_hist = deque()
        self.angle_lag = 0.0
        # How late the yaw follows the wheel, as the car itself shows it
        # (auto_lag): yaw_tau until it has turned enough to tell.
        self.tau = float(self.k["yaw_tau"])
        self.lag_f = np.zeros(len(LAG_BANK))
        self.lag_ff = np.zeros(len(LAG_BANK))
        self.lag_fr = np.zeros(len(LAG_BANK))
        self.lag_rr = 0.0
        self.lag_known = False
        self.speed = 0.0
        self.blocked_since = None
        self.push_until = -1e9
        self.boxed, self.boxed_at = 0, (0.0, 0.0)
        self.search_until = -1e9  # the lane's line is given up for a search till then
        self.field_best, self.field_best_t, self.field_off_until = np.inf, 0.0, -1e9
        self.center_off = None
        self.cam_yaw_est = 0.0  # rad the camera is found to point left of true
        self.reverse_count = 0
        self.last_reverse_end = -1e9
        self.info = {}
        self.now = t
        self.events = []  # (t, why, x, y) of every reversal, for the log
        if self.route is not None and pose is not None:
            self.route.anchor(pose)

    # ------------------------------------------------------------ observe

    def observe(self, scan, gates, pose, t):
        """A new camera frame.

        scan   (bins,) m, nearest blocking return per bearing, bin 0 rightmost
        gates  iterable of (kind, cx, cy, nx, ny, half_span) in the leveled
               camera frame; (nx, ny) is the gate's normal, either way round
        pose   the chassis pose when the frame was captured
        """
        k = self.k
        scan = np.asarray(scan, float)
        self.scan = scan
        cam_yaw = math.radians(float(k["camera_yaw_deg"])) + self.cam_yaw_est
        bearings = self.bearings + cam_yaw
        cam = (
            pose[0] + self.cam_x * math.cos(pose[2]),
            pose[1] + self.cam_x * math.sin(pose[2]),
            pose[2],
        )

        self._forget(scan, cam, cam_yaw, t)
        self._see(scan, cam_yaw, pose, t)
        hit = scan < self.max_range - 0.05
        local = np.c_[scan * np.cos(bearings), scan * np.sin(bearings)]
        # Midpoints between neighboring returns on one surface, so a wall
        # seen at range has no holes between bins.
        pair = hit[:-1] & hit[1:] & (np.abs(np.diff(scan)) < 0.3)
        local_all = np.concatenate([local[hit], 0.5 * (local[:-1] + local[1:])[pair]])
        r_all = np.concatenate([scan[hit], 0.5 * (scan[:-1] + scan[1:])[pair]])
        world = from_frame(local_all, cam)
        # A return is believed once two frames running put something there:
        # stereo throws single-frame phantoms close in, and one of those dead
        # ahead would stop the car.
        if len(world) and len(self.candidates):
            gap = np.hypot(
                world[:, None, 0] - self.candidates[None, :, 0],
                world[:, None, 1] - self.candidates[None, :, 1],
            ).min(1)
            seen_twice = gap < 0.12 + 0.06 * r_all
        else:
            seen_twice = np.zeros(len(world), bool)
        self.candidates = world
        keep = r_all <= float(k["insert_range"])
        # Stereo range noise grows with the square of range.
        sigma = float(k["noise_a"]) + float(k["noise_b"]) * r_all**2
        world = np.c_[world, sigma]
        self.far = world[seen_twice & ~keep]
        self._remember(world[seen_twice & keep], t)

        if self.route is not None and self.route.anchored:
            self.route.update(pose[0], pose[1])
        self._track_gate(gates, cam, cam_yaw, pose, t)
        self._plan(pose)

    def _seen_index(self, points):
        idx = np.floor((points - self.seen_origin) / COARSE).astype(int)
        return np.clip(idx[:, 0], 0, SEEN_N - 1), np.clip(idx[:, 1], 0, SEEN_N - 1)

    def _see(self, scan, cam_yaw, pose, t):
        """Stamp the cells this frame looked through, and the ones under the car."""
        b = self.cell_bearing - cam_yaw
        idx = np.clip(
            ((b + self.half_fov) / self.bin_width).astype(int), 0, self.bins - 1
        )
        empty = (
            (np.abs(b) < self.half_fov)
            & (self.cell_range < scan[idx] - 0.15)
            & (self.cell_range < float(self.k["see_range"]))
        ) | self.cell_near
        i, j = self._seen_index(from_frame(self.cell_xy[empty], pose))
        self.seen_t[i, j] = t
        self.now = t

    def _forget(self, scan, cam, cam_yaw, t):
        if not len(self.mem):
            return
        k = self.k
        life = float(k["open_memory_s"] if self.in_open else k["memory_s"])
        keep = t - self.mem_t <= life
        rel = to_frame(self.mem, cam)
        r = np.hypot(rel[:, 0], rel[:, 1])
        b = np.arctan2(rel[:, 1], rel[:, 0]) - cam_yaw
        inside = (np.abs(b) < self.half_fov - self.bin_width) & (
            r > float(k["min_range"]) + 0.1
        )
        idx = np.clip(
            ((b + self.half_fov) / self.bin_width).astype(int), 0, self.bins - 1
        )
        # Seen past on its own bearing and both neighbors: one dropped bin
        # must not erase a post.
        pad = np.concatenate([scan[:1], scan, scan[-1:]])
        through = np.minimum(np.minimum(pad[:-2], pad[1:-1]), pad[2:])
        seen_past = through[idx] > r + float(k["carve_tolerance"]) + 0.04 * r
        # The same surface measured again, from nearer: range noise grows
        # with the square of range, so the old point is the worse one, and
        # left in it would stand a few centimeters inside the lane.
        again = np.abs(scan[idx] - r) < 0.10 + 0.08 * r
        keep &= ~(inside & (seen_past | again))
        self.mem, self.mem_t = self.mem[keep], self.mem_t[keep]

    def _remember(self, points, t):
        if not len(points):
            return
        pts = np.concatenate([points, self.mem])
        stamps = np.concatenate([np.full(len(points), t), self.mem_t])
        # One point per 5 cm cell, the newest.
        key = np.round(pts[:, :2] / CELL).astype(np.int64)
        key = key[:, 0] * 1_000_003 + key[:, 1]
        _, first = np.unique(key, return_index=True)
        self.mem, self.mem_t = pts[first], stamps[first]

    def _track_gate(self, gates, cam, cam_yaw, pose, t):
        """Keep the hoops ahead: every one the segmenter reports, not only
        the nearest, because the line through one has to be set up for the
        next, two and a half meters on and off to the other side."""
        k = self.k
        if not k["gate_aim"]:
            self.gate, self.hoops = None, []
            return
        here = np.asarray(pose[:2])
        ahead = []
        for hoop in self.hoops:
            if (here - hoop["c"]) @ hoop["n"] > 0.30:
                self.passed.append((t, hoop["c"]))
            elif t - hoop["t"] < float(k["gate_memory_s"]):
                ahead.append(hoop)
        self.hoops = ahead
        for kind, gx, gy, nx, ny, half in gates:
            if int(kind) != HOOP or math.hypot(gx, gy) > float(k["gate_range"]):
                continue
            c, s = math.cos(cam_yaw), math.sin(cam_yaw)
            local = np.array([[c * gx - s * gy, s * gx + c * gy]])
            center = from_frame(local, cam)[0]
            yaw = cam[2] + cam_yaw
            c, s = math.cos(yaw), math.sin(yaw)
            normal = np.array([c * nx - s * ny, s * nx + c * ny])
            # The normal points the way the car is going through.
            if normal @ (center - here) < 0:
                normal = -normal
            if any(
                t - tp < 30.0 and np.hypot(*(center - cp)) < 0.8
                for tp, cp in self.passed
            ):
                continue
            if self.route is not None and self.route.anchored:
                off = self.route.distance_to(center[0], center[1])
                if off > float(k["gate_route_tolerance"]):
                    continue
            for hoop in self.hoops:
                if np.hypot(*(center - hoop["c"])) < 0.7:
                    # The same one, measured again: nearer, so better.
                    if normal @ hoop["n"] < 0:
                        normal = -normal
                    hoop.update(c=0.5 * (hoop["c"] + center), n=normal, t=t)
                    break
            else:
                self.hoops.append(dict(c=center, n=normal, half=float(half), t=t))
        self.gate = None
        if self.hoops:
            hoop = min(self.hoops, key=lambda h: float(np.hypot(*(h["c"] - here))))
            self.gate = (hoop["c"], hoop["n"], hoop["half"])

    # --------------------------------------------------------------- plan

    def _goal(self, pose):
        """The points to find a way to, (M, 2) in the pose frame: any one."""
        k = self.k
        self.in_open = False
        if self.gate is not None:
            c, n, _ = self.gate
            return (c + float(k["gate_through"]) * n)[None, :]
        if self.route is not None and self.route.anchored:
            goals, self.in_open = self.route.goal()
            return np.asarray(goals)
        return self._reactive_goal(pose)[None, :]

    def _reactive_goal(self, pose):
        """No route: head for the deep end of the widest open stretch of scan,
        as rl/obstacleRacer's steering prior does."""
        scan = self.scan
        open_ = scan >= 1.5
        bearing, depth = float(self.bearings[int(np.argmax(scan))]), float(scan.max())
        if open_.any():
            edges = np.diff(np.concatenate([[0], open_.astype(np.int8), [0]]))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            mid = 0.5 * (self.bearings[starts] + self.bearings[ends - 1])
            pick = int(np.argmax((ends - starts) * self.bin_width - 0.15 * np.abs(mid)))
            run = slice(starts[pick], ends[pick])
            weight = (scan[run] - 1.4) ** 2
            bearing = float(weight @ self.bearings[run] / weight.sum())
            depth = float(np.median(scan[run]))
        reach = min(max(depth - 0.5, 1.0), 4.0)
        heading = (
            pose[2]
            + bearing
            + math.radians(float(self.k["camera_yaw_deg"]))
            + self.cam_yaw_est
        )
        return np.asarray(
            [
                pose[0] + self.cam_x * math.cos(pose[2]) + reach * math.cos(heading),
                pose[1] + self.cam_x * math.sin(pose[2]) + reach * math.sin(heading),
            ]
        )

    def _plan(self, pose):
        k = self.k
        goals = self._goal(pose)
        self.goal = goals[
            int(np.argmin(np.hypot(goals[:, 0] - pose[0], goals[:, 1] - pose[1])))
        ]
        points = np.concatenate([self.mem, self.far]) if len(self.far) else self.mem
        if self.gate is not None:
            c, n, half = self.gate
            a = np.array([-n[1], n[0]])
            rel = points[:, :2] - c
            along, across = rel @ n, rel @ a
            # The opening is open, whatever a noisy return says; its posts
            # stay.  Either side of it is wall: the goal is through it.
            inside = (np.abs(across) < half - 0.10) & (np.abs(along) < 0.30)
            points = points[~inside]
        if self.route is not None and self.route.anchored and len(points):
            soft = self.route.soft_line()
            if soft is not None:
                # The car wash's strands hang across the lane and read as a
                # wall; they are there to be driven through.  What stands
                # within the lane's own width of the route there is not an
                # obstacle.  The arches' posts, outside it, still are.
                gap = np.hypot(
                    points[:, None, 0] - soft[None, :, 0],
                    points[:, None, 1] - soft[None, :, 1],
                ).min(1)
                points = points[gap > float(k["soft_half_width"])]
        # What the lane is slid between: the walls, not the lines drawn
        # below to keep a search or a swerve from going round a hoop.
        seen, seen_local = (
            points,
            (to_frame(points, pose) if len(points) else np.zeros((0, 2))),
        )
        if self.gate is not None:
            w = np.arange(half + 0.05, half + float(k["gate_wing"]), 0.04)
            wings = np.concatenate([c + w[:, None] * a, c - w[:, None] * a], axis=0)
            points = np.concatenate([points, np.c_[wings, np.zeros(len(wings))]])
        if self.route is not None and self.route.anchored:
            shut = self.route.way_in(pose[0], pose[1])
            if shut is not None:
                # The lane the car came into an open region by is closed
                # behind it: the way on is not back out.
                t = np.linspace(0.0, 1.0, 66)[:, None]
                line = shut[0] + t * (shut[1] - shut[0])
                points = np.concatenate([points, np.c_[line, np.zeros(len(line))]])
            fence = self.route.fence(pose[0], pose[1])
            if fence is not None and len(fence):
                # And the region's own walls, drawn a little outside them:
                # the way through is not round the back of one.
                points = np.concatenate([points, np.c_[fence, np.zeros(len(fence))]])
        self.points = points
        local = to_frame(points, pose) if len(points) else np.zeros((0, 2))
        self.grid_pose = pose
        self.goal_local = to_frame(self.goal[None, :], pose)[0]

        # In a lane the route IS the way: it has the lane's own shape, bend
        # for bend, which a search over a grid of what the camera happens to
        # see does not (round the helix, whose inside wall cannot be seen,
        # the search goes straight across the middle).  It only has to be
        # slid to the middle of the walls actually there, and over to the
        # middle of a hoop standing off the line.  The way is searched for
        # where the route says nothing -- an open region -- or where
        # something is standing on it.
        lane_way = None
        if self.route is not None and self.route.anchored:
            # Long enough to be read where the car is steering for.
            lead = (
                float(k["dead_time"])
                + float(k["feedforward_lead_s"])
                + self._bend_lead()
            )
            lane = self.route.ahead(
                max(float(k["path_length"]), self.speed * lead + 1.5)
            )
            if lane is not None:
                self.center_off = None
                way = self._center(to_frame(lane, pose), seen_local, seen[:, 2])
                off = self.center_off
                way = self._through_gates(way, pose)
                self.frames += 1
                stopped = self._blocked(way, seen_local, seen[:, 2])
                if stopped:
                    # Once more, in case the route is further off the lane
                    # than one sliding takes out.
                    way = self._center(way, seen_local, seen[:, 2])
                    way = self._through_gates(way, pose)
                    stopped = self._blocked(way, seen_local, seen[:, 2])
                # Not on the first frames: a wall takes two to be believed,
                # and half a lane is not a lane.
                if (not stopped or self.frames < 4) and self.now >= self.search_until:
                    threading = self.hoops or any(
                        self.now - at < 10.0 for at, _ in self.passed
                    )
                    if off is not None and not threading and self.mode == "drive":
                        # Walls either side say where the lane is, to the
                        # centimeter: the route is moved over to it, a share
                        # each frame, and comes to carry the pose's drift
                        # and the course's own offset from the drawing.
                        c, s = math.cos(pose[2]), math.sin(pose[2])
                        gain = float(k["align_wall_gain"])
                        self.route.nudge(
                            gain
                            * np.array(
                                [c * off[0] - s * off[1], s * off[0] + c * off[1]]
                            )
                        )
                    self.path = from_frame(way, pose)
                    self.D = None
                    self.searched = False
                    return
                lane_way = way[: int(3.0 / PATH_STEP)]
        # A search takes tens of milliseconds, and what it finds stays good
        # while the car drives it: not every frame.
        # And a new search from a slightly different place can come out the
        # other side of a bale, and the car then dithers between the two: so
        # the way in hand is kept while it is still clear and still ahead.
        age = self.now - self.searched_at
        keep = self.searched and self.path is not None
        # (Driving by the field, it is searched afresh each frame: there is
        # no line to hold to, and the field has to be the latest.)
        if keep and not k["field_drive"]:
            ahead = to_frame(self.path, pose)
            nearest = int(np.argmin(np.hypot(ahead[:, 0], ahead[:, 1])))
            left = (len(ahead) - 1 - nearest) * PATH_STEP
            on_it = float(np.hypot(*ahead[nearest])) < 0.35
            clear = not self._blocked(ahead[nearest:], local, points[:, 2])
            if on_it and clear and left > 0.8 and age < float(k["search_keep_s"]):
                return
            if age < float(k["search_period_s"]) and clear and left > 0.3:
                return
        self.searched = True
        self.searched_at = self.now
        if lane_way is not None:
            # Something is standing in the lane: the way round it is to
            # where the lane itself goes, not to where the route's hint of
            # it would be.
            goals = from_frame(lane_way[-1:], pose)
            self.goal = goals[0]
            self.goal_local = lane_way[-1]

        # The grid: walls, dearer the nearer one is; dearer off the route;
        # dearer over ground the camera has not looked at.
        ci = np.floor((local[:, 0] - COARSE_X0) / COARSE).astype(int)
        cj = np.floor((local[:, 1] - COARSE_Y0) / COARSE).astype(int)
        ok = (ci >= 0) & (ci < COARSE_NX) & (cj >= 0) & (cj < COARSE_NY)
        occ = np.zeros((COARSE_NX, COARSE_NY), bool)
        occ[ci[ok], cj[ok]] = True
        w = float(k["clearance_weight"])
        cost = np.ones((COARSE_NX, COARSE_NY))
        ring = occ
        for weight in (3.0, 1.5, 0.8, 0.4, 0.2):
            ring = _dilate(ring)
            cost += w * weight * ring
        ring1 = _dilate(occ)
        lane = None
        if self.gate is None and self.route is not None and self.route.anchored:
            lane = self.route.corridor()
        if lane is not None:
            # Rings out from the route: what has not been seen is free, so
            # without this a goal round a bend is headed for across the
            # inside of it.
            li = np.floor((to_frame(lane, pose) - (COARSE_X0, COARSE_Y0)) / COARSE)
            li = li.astype(int)
            ok = (
                (li[:, 0] >= 0)
                & (li[:, 0] < COARSE_NX)
                & (li[:, 1] >= 0)
                & (li[:, 1] < COARSE_NY)
            )
            reach = np.zeros((COARSE_NX, COARSE_NY), bool)
            reach[li[ok, 0], li[ok, 1]] = True
            free = int(round(float(k["lane_half_width"]) / COARSE))
            away = np.zeros((COARSE_NX, COARSE_NY))
            for ring_no in range(free + 10):
                reach = _dilate(reach)
                if ring_no >= free:
                    away += ~reach
            cost += float(k["lane_weight"]) * away
        # The camera sees 55 degrees either side, so the inside of a tight
        # turn is ground nobody has looked at, and the wall that ends at a
        # corner usually carries on round it.  Not a wall, or the car could
        # never head for a goal it cannot see the way to.
        i, j = self._seen_index(from_frame(self.cell_xy, pose))
        unseen = self.now - self.seen_t[i, j] > float(k["seen_memory_s"])
        cost += float(k["unseen_cost"]) * unseen.reshape(cost.shape)
        # But beside a wall it is wall, as far as the search is told at
        # first: a wall is seen a stretch at a time, and the unseen ground
        # at the end of a stretch is nearly always more of it.  Left free,
        # the car backs and fills against a bale wall to get at the gap it
        # takes to be just out of sight.  Ground the camera has looked
        # across and found empty is a gap; that is what a gap is.
        kept = float(k["open_memory_s"] if self.in_open else k["seen_memory_s"])
        dark = (self.now - self.seen_t[i, j] > kept).reshape(cost.shape)
        beside = occ
        for _ in range(int(round(float(k["shadow_m"]) / COARSE))):
            beside = _dilate(beside)
        shadow = dark & beside
        car_i = int((0.0 - COARSE_X0) / COARSE)
        car_j = int((0.0 - COARSE_Y0) / COARSE)
        blocked = occ.copy()
        blocked[car_i - 1 : car_i + 2, car_j - 1 : car_j + 2] = False

        g = to_frame(goals, pose)
        # A goal off the grid is brought to its edge, toward the goal.
        lo_x, hi_x = COARSE_X0 + 0.3, COARSE_X0 + COARSE * COARSE_NX - 0.3
        lo_y, hi_y = COARSE_Y0 + 0.3, COARSE_Y0 + COARSE * COARSE_NY - 0.3
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.minimum(
                np.where(g[:, 0] > hi_x, hi_x / g[:, 0], 1.0),
                np.where(g[:, 0] < lo_x, lo_x / g[:, 0], 1.0),
            )
            scale = np.minimum(scale, np.where(g[:, 1] > hi_y, hi_y / g[:, 1], 1.0))
            scale = np.minimum(scale, np.where(g[:, 1] < lo_y, lo_y / g[:, 1], 1.0))
        g = g * scale[:, None]
        gi = ((g[:, 0] - COARSE_X0) / COARSE).astype(int)
        gj = ((g[:, 1] - COARSE_Y0) / COARSE).astype(int)
        # Not the goals themselves: a slot's far side is always out of sight.
        for a, b in zip(gi, gj):
            shadow[max(a - 2, 0) : a + 3, max(b - 2, 0) : b + 3] = False
        open_ = ~ring1[gi, gj]
        if open_.any():
            seeds = (gi[open_], gj[open_], np.zeros(int(open_.sum())))
        else:
            # Every goal is in something: the nearest free cell to the first.
            ii, jj = self.coarse_ij
            d2 = np.where(ring1, np.inf, (ii - gi[0]) ** 2 + (jj - gj[0]) ** 2)
            cell = np.unravel_index(int(np.argmin(d2)), d2.shape)
            seeds = (np.array([cell[0]]), np.array([cell[1]]), np.zeros(1))
        goal_local = self.goal_local
        r = int(2.6 / COARSE)
        near = (
            slice(max(car_i - r, 0), car_i + r + 1),
            slice(max(car_j - r, 0), car_j + r + 1),
        )
        # The body's half width off every wall, and unseen ground beside a
        # wall shut; then with less of each if that closes the only way.
        thick = _dilate(ring1)
        tries = (thick | shadow, ring1 | shadow, thick, ring1)
        for walls in tries:
            shut = walls.copy()
            shut[car_i - 2 : car_i + 3, car_j - 2 : car_j + 3] = False
            D = wavefront(cost, shut, seeds, near)
            if np.isfinite(D[car_i, car_j]):
                break
        else:
            D = wavefront(cost, blocked, seeds, near)
        self.D, self.D_pose = D, pose

        path = self._descend(D, car_i, car_j, goal_local)
        path = self._center(path, local, points[:, 2])
        self.path = from_frame(path, pose)

    def _through_gates(self, way, pose):
        """The lane's way (car frame), bent over to pass through the middle
        of the hoop ahead, and of those just passed.

        A hoop stands up to half a meter off the lane's middle, and the next
        is two and a half meters on and off to the other side.  A way
        searched for through it and then on to the next comes out as a
        three-point turn; what the car can drive is the lane's own line
        eased across by the hoop's offset, over gate_blend_m either side.
        """
        centers = [(c, None) for stamp, c in self.passed if self.now - stamp < 10.0]
        centers += [(hoop["c"], hoop["n"]) for hoop in self.hoops]
        if not centers or len(way) < 5:
            return way
        tangent = np.gradient(way, axis=0)
        tangent /= np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-9)[:, None]
        normal = np.c_[-tangent[:, 1], tangent[:, 0]]
        blend = float(self.k["gate_blend_m"]) / PATH_STEP
        at = np.arange(len(way))
        shift = np.zeros(len(way))
        weight = np.zeros(len(way))
        cos, sin = math.cos(pose[2]), math.sin(pose[2])
        for center, through in centers:
            c = to_frame(center[None, :], pose)[0]
            rel = c[None, :] - way
            j = int(np.argmin(np.hypot(rel[:, 0], rel[:, 1])))
            off = float(rel[j] @ normal[j])
            if j >= len(way) - 3 or abs(float(rel[j] @ tangent[j])) > 0.3:
                continue  # off the end of the way: not yet
            if abs(off) > float(self.k["gate_route_tolerance"]):
                continue  # not on this lane
            if through is not None:
                # Nor one the way does not go through: a hoop round the
                # corner stands side on to the lane the car is still in.
                n = np.array(
                    [
                        cos * through[0] + sin * through[1],
                        -sin * through[0] + cos * through[1],
                    ]
                )
                if abs(float(n @ tangent[j])) < 0.6:
                    continue
            w = 0.5 * (1.0 + np.cos(np.pi * np.clip(np.abs(at - j) / blend, 0.0, 1.0)))
            shift += off * w
            weight += w
        # Between two hoops each has its say in proportion.
        shift = np.where(weight > 1.0, shift / np.maximum(weight, 1e-9), shift)
        return _resample(way + shift[:, None] * normal)

    def _blocked(self, way, points, sigma):
        """Whether something stands on the first few meters of `way`."""
        if not len(points) or len(way) < 2:
            return False
        # From a little ahead of the car: the route can start beside a wall
        # the car is not at (it is only a hint), and that is not an obstacle.
        ahead = np.hypot(way[:, 0], way[:, 1]) > 0.5
        way = way[ahead][: int(float(self.k["block_check_m"]) / PATH_STEP)]
        if not len(way):
            return False
        gap = (
            np.hypot(
                points[None, :, 0] - way[:, None, 0],
                points[None, :, 1] - way[:, None, 1],
            )
            + float(self.k["noise_sigmas"]) * sigma
        )
        return bool((gap < BODY_HALF_WIDTH).any())

    def _descend(self, D, i, j, goal_local):
        """The way to the goal: downhill on the wavefront from the car."""
        cells = [(i, j)]
        if np.isfinite(D[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]).any():
            for _ in range(int(float(self.k["path_length"]) / COARSE)):
                lo_i, lo_j = max(i - 1, 0), max(j - 1, 0)
                box = D[lo_i : i + 2, lo_j : j + 2]
                at = np.unravel_index(int(np.argmin(box)), box.shape)
                ni, nj = lo_i + at[0], lo_j + at[1]
                if (ni, nj) == (i, j) or not np.isfinite(box[at]):
                    break
                i, j = ni, nj
                cells.append((i, j))
        if len(cells) < 4:
            # Walled in on the grid: straight at the goal, for the veto to
            # make what it can of.
            reach = max(float(np.hypot(*goal_local)), 0.5)
            t = np.arange(0.0, min(reach, 3.0), PATH_STEP)[:, None]
            return t * (goal_local / reach)[None, :]
        cells = np.asarray(cells, float)
        path = np.c_[
            COARSE_X0 + COARSE * (cells[:, 0] + 0.5),
            COARSE_Y0 + COARSE * (cells[:, 1] + 0.5),
        ]
        path[0] = (0.0, 0.0)
        return _resample(_smooth(path, 3))

    def _center(self, path, points, sigma):
        """Slide the path sideways to the middle of what stands beside it.

        The wavefront is on a 10 cm grid; the 20 in lane leaves the car 9 cm
        a side.  So each path point is moved along its own normal: to the
        middle, where there is a wall within reach on both sides, and off
        the wall where there is one on one side only.  Distances are to the
        remembered points themselves, each given the benefit of its range
        noise.
        """
        k = self.k
        if len(path) < 3 or not len(points):
            return path
        reach = float(k["center_reach"])
        want = BODY_HALF_WIDTH + float(k["center_air"])
        slack = float(k["noise_sigmas"]) * sigma
        for _ in range(2):
            tangent = np.gradient(path, axis=0)
            tangent /= np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-9)[:, None]
            dx = points[None, :, 0] - path[:, None, 0]
            dy = points[None, :, 1] - path[:, None, 1]
            along = dx * tangent[:, None, 0] + dy * tangent[:, None, 1]
            across = -dx * tangent[:, None, 1] + dy * tangent[:, None, 0]
            beside = np.abs(along) < 0.25
            left = np.where(beside & (across > 0), across + slack, np.inf).min(1)
            right = np.where(beside & (across < 0), -across + slack, np.inf).min(1)
            both = (left < reach) & (right < reach)
            with np.errstate(invalid="ignore"):
                shift = np.where(both, 0.5 * (left - right), 0.0)
            shift = np.where(~both & (left < want), left - want, shift)
            shift = np.where(~both & (right < want), want - right, shift)
            shift = np.clip(shift, -0.3, 0.3)
            near = both & (np.hypot(path[:, 0], path[:, 1]) < 2.0)
            if self.center_off is None and near.sum() >= 8:
                # How far the path as given stood from the middle of the
                # walls beside the car (car frame), for the route's keeping.
                normal = np.c_[-tangent[:, 1], tangent[:, 0]]
                self.center_off = (shift[near, None] * normal[near]).mean(0)
            # Where the path has no wall on both sides to be put between
            # (returns thin out with range, and the far wall goes first), it
            # is moved as the nearest stretch that has them was: an offset is
            # the route's, not that stretch's.  Left where it is instead, the
            # path bends back to the route two meters out, which at speed is
            # where the car is steering for.
            known = both | (shift != 0.0)
            if known.any() and not known.all():
                at = np.arange(len(path))
                shift = np.interp(at, at[known], shift[known])
            # A wall is a row of points, and the nearest of them to each
            # path point jumps about by centimeters: the shift is averaged
            # over most of a meter, or the path comes out wavy and the car
            # is steered by the waves.
            pad = np.concatenate([np.full(4, shift[0]), shift, np.full(4, shift[-1])])
            shift = np.convolve(pad, np.full(9, 1.0 / 9.0), mode="valid")
            normal = np.c_[-tangent[:, 1], tangent[:, 0]]
            path = _smooth(path + shift[:, None] * normal, 2)
        return _resample(path)

    # --------------------------------------------------------------- step

    def step(self, pose, speed, yaw_rate, t):
        """One control tick -> (steering command, speed command m/s)."""
        k = self.k
        if self.last_pose is not None:
            jump = math.hypot(pose[0] - self.last_pose[0], pose[1] - self.last_pose[1])
            if jump > float(k["relocalize_step"]):
                # A loop closure moved the pose frame under what is remembered.
                self.mem, self.mem_t = np.zeros((0, 3)), np.zeros(0)
                self.far = np.zeros((0, 3))
                self.points = np.zeros((0, 3))
                self.path = None
                self.history.clear()
                if self.route is not None and self.route.anchored:
                    self.route.update(pose[0], pose[1], search_m=6.0)
        self.last_pose = pose
        self.speed = abs(speed)
        if self.route is not None and self.route.anchored:
            self.route.update(pose[0], pose[1])
        if self.path is None:
            return 0.0, 0.0

        self.history.append((t, pose[0], pose[1]))
        while self.history and t - self.history[0][0] > float(k["stuck_s"]) + 0.5:
            self.history.popleft()

        if self.mode == "reverse":
            out = self._reverse_step(pose, t)
            if out is not None:
                return out

        choice = None
        by_field = self.searched and self.D is not None and k["field_drive"]
        if by_field:
            # Getting nowhere by it -- shuttling in a pocket between bales
            # whose way out is a gap the arcs do not line up with -- it is
            # given a rest, and the line drawn over the grid is followed
            # instead, with the plain back-and-fill: clumsier, and it
            # wriggles through places this does not.
            here = float(self._field(np.zeros((1, 2)), pose)[0])
            if here < self.field_best - 3.0:
                self.field_best, self.field_best_t = here, t
            elif t - self.field_best_t > float(k["field_patience_s"]):
                self.field_off_until = t + float(k["field_rest_s"])
                self.field_best, self.field_best_t = np.inf, self.field_off_until
        else:
            self.field_best, self.field_best_t = np.inf, t
        if by_field and t >= self.field_off_until:
            # Where the way has had to be searched for, the search's own
            # field is driven on directly: see _field_drive.
            choice = self._field_drive(pose, speed, yaw_rate, t)
            if choice is None:
                out = self._field_reverse(pose, speed, yaw_rate, t)
                if out is not None:
                    return out
            else:
                want = choice[0]
                self.info.update(vetoed=False, tight=False)
        if choice is None:
            want = self._pursue(pose, speed, yaw_rate, t)
            self._learn_camera_yaw(speed, yaw_rate)
            choice = self._veto(pose, speed, yaw_rate, t, want)
        if t < self.push_until:
            kappa = choice[0] if choice is not None else want
            self.info.update(mode="boxed")
            return self._steer_command(kappa, yaw_rate, t=t), float(k["v_min"])
        if choice is None:
            if self.blocked_since is None:
                self.blocked_since = t
            if not self.searched and k["field_drive"] and self.hoops:
                # The line through a hoop cannot be driven from here (the
                # car has come up square on to a post): for a while the way
                # is searched for instead, which can back and fill its way
                # round to face the opening; _plan sees to it from the next
                # frame.  Only for a hoop: in a plain lane backing off and
                # taking the line again is quicker and surer.
                self.search_until = t + float(k["search_keep_s"])
                if t - self.blocked_since < 1.0:
                    return self._steer_command(self.kappa_cmd, yaw_rate, t=t), 0.0
            if t - self.blocked_since > 0.3:
                return self._start_reverse(pose, t, "blocked")
            return self._steer_command(self.kappa_cmd, yaw_rate, t=t), 0.0
        self.blocked_since = None
        kappa, free, limited = choice

        # Speed: route cap, the bend, and the clear length ahead.
        v = float(k["v_max"])
        v_route = v
        if self.route is not None and self.route.anchored:
            v_route = self.route.speed_cap(abs(speed))
        elif self.route is None:
            v_route = float(k["sections"]["default"])
        v_bend = math.sqrt(float(k["lat_accel"]) / max(abs(kappa), 1e-3))
        v_sight = (
            math.sqrt(
                2.0
                * float(k["sight_decel"])
                * max(free - float(k["sight_margin"]), 0.0)
            )
            if limited
            else v
        )
        # And no faster than the car is already turning allows: out of a
        # hairpin it is still coming round after the wheel is straight, and
        # the throttle then (Gazebo's car, at least) keeps it coming round.
        v_yaw = float(k["lat_accel"]) / max(abs(yaw_rate), 1e-3)
        lag = self.tau if self.lag_known else max(self.tau, float(k["lag_cautious"]))
        v_lag = float(k["lag_reach"]) / lag
        v = max(min(v, v_route, v_bend, v_sight, v_yaw, v_lag), float(k["v_min"]))
        v_cmd = max(v * float(k["speed_scale"]), float(k["v_min"]))
        # Speed is added gently; it is taken off at once.
        self.v_cmd = min(
            v_cmd, max(self.v_cmd, abs(speed)) + float(k["throttle_rate"]) * self.dt
        )
        v_cmd = max(self.v_cmd, float(k["v_min"]))

        if self._stuck(t, v_cmd):
            self._bumped(pose, t, front=True)
            return self._start_reverse(pose, t, "stuck")
        if self.route is not None and self.route.anchored and speed > 0.3:
            # Only between walls: that is what makes the car's path the lane.
            beside = to_frame(self.mem, pose) if len(self.mem) else np.zeros((0, 2))
            beside = beside[np.abs(beside[:, 0]) < 0.6]
            # And only while it is simply driving: backing and filling, or
            # dodging something, is not the lane's shape.
            settled = t - self.last_reverse_end > 4.0 and not self.info.get("vetoed")
            # Nor is the line it takes through the hoops, which is off the
            # lane's on purpose.
            settled = settled and not self.hoops
            settled = settled and not any(t - stamp < 10.0 for stamp, _ in self.passed)
            if settled and (beside[:, 1] > 0.1).any() and (beside[:, 1] < -0.1).any():
                self.route.align(pose[0], pose[1])
        steer = self._steer_command(kappa, yaw_rate, speed, t)
        self.info.update(
            mode="drive",
            kappa=kappa,
            want=want,
            free=free,
            v_route=v_route,
            v_bend=v_bend,
            v_sight=v_sight,
        )
        return steer, v_cmd

    def _sent_at(self, when):
        """The curvature command that was on the wire at time `when`."""
        for stamp, kappa in reversed(self.sent):
            if stamp <= when:
                return kappa
        return self.sent[0][1] if self.sent else 0.0

    def _rollout(self, kappa, speed, yaw_rate, t, seconds):
        """Where each command in `kappa` would take the car over `seconds`:
        the rear axle's (x, y, heading), each (K, S), in the car frame, and
        the step length.

        The car answers the wheel late: nothing for dead_time, while the
        commands already sent are still arriving, then the yaw rate eases
        toward what was asked with the time constant the car has shown
        (yaw_tau to begin with).
        """
        k = self.k
        v = max(speed, 0.5)
        h = min(0.05, 0.10 / v)
        dead, tau = float(k["dead_time"]), self.tau
        steps = max(1, int(round(seconds / h)))
        gain = 1.0 - math.exp(-h / tau)
        n = len(kappa)
        r = np.full(n, float(yaw_rate))
        th = np.zeros(n)
        x = np.full(n, -AXLE_BACK)
        y = np.zeros(n)
        X, Y, TH = (np.empty((n, steps)) for _ in range(3))
        for j in range(steps):
            at = j * h
            applied = self._sent_at(t - dead + at) if at < dead else kappa
            r = r + (v * applied - r) * gain
            th = th + r * h
            x = x + v * h * np.cos(th)
            y = y + v * h * np.sin(th)
            X[:, j], Y[:, j], TH[:, j] = x, y, th
        return X, Y, TH, v * h

    def _pursue(self, pose, speed, yaw_rate, t):
        """The curvature to ask for now, to follow the path.

        Two parts.  Feedforward: the path's own bend, read where the car
        will be once this command has taken hold (dead_time plus
        feedforward_lead_s on, and further for a car whose yaw is slow to
        follow: lead_per_tau), because a bend steered for when the car
        reaches it is a bend entered a car length late.  Feedback: the
        sideways and heading errors where the commands already sent will
        have put the car, as a second-order pull onto the path
        (track_omega rad/s, track_damping).  Pure pursuit alone had to
        choose between a short lookahead that weaves under the lag and a
        long one that runs wide of every bend.

        Where the car will be depends on the command being chosen, so the
        command is solved for: each candidate is rolled out, and the one
        taken is the one that is what the path asks for where it leads.
        Rolling out the last command instead and steering for where that
        leads works while the lag is short; with a second of it (Gazebo's
        car) each command undoes the one before, lock to lock, every tick.
        """
        k = self.k
        dead = float(k["dead_time"])
        v = max(speed, float(k["v_min"]))
        path = self.path
        if len(path) < 12:
            # Too short to read a bend off: straight at its end.
            g = to_frame(path[-1:], pose)[0]
            kappa = 2.0 * g[1] / max(float(g @ g), 0.05)
            return max(-self.k_right, min(self.k_left, kappa))
        cand = self.kappa
        X, Y, TH, _ = self._rollout(
            cand, speed, yaw_rate, t, dead + float(k["feedforward_lead_s"])
        )
        # The chassis center then, in the pose frame.
        th = pose[2] + TH[:, -1]
        at = from_frame(
            np.c_[
                X[:, -1] + AXLE_BACK * np.cos(TH[:, -1]),
                Y[:, -1] + AXLE_BACK * np.sin(TH[:, -1]),
            ],
            pose,
        )
        d = np.hypot(
            path[None, :, 0] - at[:, None, 0], path[None, :, 1] - at[:, None, 1]
        )
        j = np.clip(d.argmin(1), 4, len(path) - 5)
        tangent = path[j + 4] - path[j - 4]
        heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        # + when the path is to the left of the car.
        offset = (path[j, 0] - at[:, 0]) * -np.sin(heading) + (
            path[j, 1] - at[:, 1]
        ) * np.cos(heading)
        turn = (heading - th + math.pi) % (2 * math.pi) - math.pi
        # The pull back onto the path, as a length: v / track_omega m of
        # travel, but never under track_length_min -- slower than that the
        # gains would grow without limit and the wheel chatter lock to lock.
        # And no quicker than the yaw can follow: a pull the car answers a
        # second late is a weave, wall to wall, at speed.
        omega = min(float(k["track_omega"]), float(k["track_lag"]) / self.tau)
        reach = max(v / omega, float(k["track_length_min"]))
        # ... and with more damping on the heading: the lag is a third
        # state the pull has to be damped against.
        slow = max(0.0, self.tau - 0.4)
        zeta = float(k["track_damping"]) + float(k["lag_zeta"]) * slow
        feedback = offset / reach**2 + 2.0 * zeta * turn / reach
        # The bend, over a car length of path: further on for a slow yaw,
        # which has to be asked that much sooner.
        j = np.clip(j + int(round(v * self._bend_lead() / PATH_STEP)), 4, len(path) - 5)
        span = np.minimum(5, np.minimum(j, len(path) - 1 - j))
        before, after = path[j] - path[j - span], path[j + span] - path[j]
        bend = np.arctan2(
            before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0],
            before[:, 0] * after[:, 0] + before[:, 1] * after[:, 1],
        ) / (span * PATH_STEP)
        # And against the yaw rate itself, where it is more than the path's
        # bend calls for: a car that swings on past what the model takes it
        # to do is under-damped by the rest.
        feedback -= float(k["yaw_damping"]) * (yaw_rate - v * bend) / v
        asked = np.clip(bend + feedback, -self.k_right, self.k_left)
        # More left than a candidate where it leads to the right of the
        # path, less where it leads to the left: the crossing is the command.
        more = asked - cand
        cross = np.flatnonzero((more[:-1] >= 0.0) & (more[1:] < 0.0))
        if len(cross):
            i = int(cross[np.argmin(np.abs(cand[cross] - self.kappa_cmd))])
            part = more[i] / (more[i] - more[i + 1])
            kappa = float(cand[i] + part * (cand[i + 1] - cand[i]))
            i += int(part > 0.5)
        else:
            i = len(cand) - 1 if more[-1] >= 0.0 else 0
            kappa = float(cand[i])
        self.info.update(
            offset=float(offset[i]), turn=float(turn[i]), bend=float(bend[i])
        )
        return kappa

    # ------------------------------------------------------- open ground

    def _sweep(self, pose, kappa, speed, yaw_rate, t, seconds, width):
        """Each command in `kappa` rolled out for `seconds` against what is
        remembered: the rollout (as _rollout gives it) and, for each, the
        step at which the body first meets a point (the number of steps if
        it never does)."""
        X, Y, TH, ds = self._rollout(kappa, speed, yaw_rate, t, seconds)
        steps = X.shape[1]
        pts = to_frame(self.points, pose) if len(self.points) else np.zeros((0, 2))
        near = (pts[:, 0] > -1.0) & (np.hypot(pts[:, 0], pts[:, 1]) < 3.5)
        pts = pts[near]
        slack = np.minimum(
            float(self.k["noise_sigmas"]) * self.points[near, 2], 0.5 * width
        )
        # What the body is already against is not counted: see _veto.
        now = (
            (pts[:, 0] + AXLE_BACK >= -BODY_REAR)
            & (pts[:, 0] + AXLE_BACK <= BODY_FRONT)
            & (np.abs(pts[:, 1]) + slack <= width)
        )
        p, sl = pts[~now], slack[~now]
        if not len(p):
            return X, Y, TH, ds, np.full(len(kappa), steps)
        cos, sin = np.cos(TH)[..., None], np.sin(TH)[..., None]
        dx = p[None, None, :, 0] - X[..., None]
        dy = p[None, None, :, 1] - Y[..., None]
        fore = cos * dx + sin * dy
        side = np.abs(-sin * dx + cos * dy) + sl
        hit = ((fore >= -BODY_REAR) & (fore <= BODY_FRONT) & (side <= width)).any(-1)
        return X, Y, TH, ds, np.where(hit.any(1), hit.argmax(1), steps)

    def _field(self, xy, pose):
        """The search's cost-to-goal at points `xy` (M, 2) given in the frame
        of `pose`; where a point is in a wall, the least beside it and a
        little, so that an arc ending close to one still has a value."""
        g = to_frame(from_frame(xy, pose), self.D_pose)
        i = np.floor((g[:, 0] - COARSE_X0) / COARSE).astype(int)
        j = np.floor((g[:, 1] - COARSE_Y0) / COARSE).astype(int)
        inside = (i >= 1) & (i < COARSE_NX - 1) & (j >= 1) & (j < COARSE_NY - 1)
        value = np.full(len(xy), np.inf)
        i, j = i[inside], j[inside]
        best = self.D[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                best = np.minimum(best, self.D[i + di, j + dj] + 2.0)
        value[inside] = best
        return value

    def _field_drive(self, pose, speed, yaw_rate, t):
        """(curvature, clear length m, whether something ends it) by the
        search's field: of the commands the car can actually carry out from
        here -- each rolled out through its lag for field_look_m and stopped
        short of what it would hit -- the one that ends nearest the goal as
        the field measures it.  None when none of them gets nearer.

        Among bales and buckets a line drawn over the grid asks for turns
        the car cannot make, and following one ends nose to a bale with the
        way on a meter to the side.  The field says how good anywhere is;
        the car's own arcs say where it can get to.
        """
        k = self.k
        v = max(speed, float(k["v_min"]))
        seconds = float(k["dead_time"]) + float(k["field_look_m"]) / v
        here = float(self._field(np.zeros((1, 2)), pose)[0])
        if not np.isfinite(here):
            return None
        margin = float(k["body_margin"])
        for width in (BODY_HALF_WIDTH + margin, BODY_HALF_WIDTH):
            X, Y, TH, ds, first = self._sweep(
                pose, self.kappa, speed, yaw_rate, t, seconds, width
            )
            steps = X.shape[1]
            end = np.minimum(first - int(round(0.10 / ds)), steps) - 1
            able = end * ds >= float(k["field_min_m"])
            if able.any():
                break
        else:
            return None
        e = np.clip(end, 0, steps - 1)
        n = np.arange(len(e))
        at = np.c_[
            X[n, e] + AXLE_BACK * np.cos(TH[n, e]),
            Y[n, e] + AXLE_BACK * np.sin(TH[n, e]),
        ]
        cost = self._field(at, pose) + float(k["field_steady"]) * np.abs(
            self.kappa - self.kappa_cmd
        )
        # And it should end pointing the way the field falls from there:
        # an arc that gets near the goal facing a wall has only got to the
        # start of a three-point turn.
        ahead = at + 0.3 * np.c_[np.cos(TH[n, e]), np.sin(TH[n, e])]
        with np.errstate(invalid="ignore"):
            gain = np.nan_to_num(
                (self._field(at, pose) - self._field(ahead, pose)) / 0.3
            )
        cost = cost - float(k["field_heading"]) * np.clip(gain, -20.0, 20.0)
        cost = np.where(able, cost, np.inf)
        best = int(np.argmin(cost))
        if not cost[best] < here - float(k["field_gain"]):
            return None
        self.info.update(field=here, field_end=float(cost[best]))
        return (
            float(self.kappa[best]),
            float(first[best] * ds),
            bool(first[best] < steps),
        )

    def _field_reverse(self, pose, speed, yaw_rate, t):
        """No command forward gets nearer the goal: the backing move after
        which one does, if there is one.  Each of wheel left, straight and
        right, backed a little and a little more as far as there is room,
        is judged by the best the car could then do forward.  Returns the
        step's (steer, speed), or None when backing does not help either.
        """
        k = self.k
        here = float(self._field(np.zeros((1, 2)), pose)[0])
        if not np.isfinite(here) or t - self.last_reverse_end < 0.5:
            return None
        if abs(speed) > 0.35:
            # No brakes: come to rest first, wheel as it was.
            self.info.update(mode="stopping")
            return self._steer_command(self.kappa_cmd, yaw_rate, speed, t), 0.0
        look = float(k["field_look_m"])
        lock = float(k["reverse_lock"])
        width = BODY_HALF_WIDTH
        best = None
        for wheel in (1.0, 0.0, -1.0):
            room = self._room_behind(pose, wheel * lock)
            kappa = math.tan(command_to_angle(wheel * lock)) / self.wheelbase
            for length in (0.3, 0.5, 0.8):
                if length > room - 0.08:
                    break
                # The rear axle backed `length` along its arc, and the
                # chassis center that goes with it.
                th = -kappa * length
                if abs(kappa) < 1e-6:
                    ax, ay = -AXLE_BACK - length, 0.0
                else:
                    ax = -AXLE_BACK - math.sin(kappa * length) / kappa
                    ay = (1.0 - math.cos(kappa * length)) / kappa
                center = np.array(
                    [[ax + AXLE_BACK * math.cos(th), ay + AXLE_BACK * math.sin(th)]]
                )
                world = from_frame(center, pose)[0]
                then = (float(world[0]), float(world[1]), pose[2] + th)
                # From there, forward: plain arcs, from rest.
                pts = (
                    to_frame(self.points, then)
                    if len(self.points)
                    else np.zeros((0, 2))
                )
                free = arc_clearance(
                    pts[None, :, 0] + AXLE_BACK,
                    pts[None, :, 1],
                    self.kappa[:, None],
                    width,
                )
                reach = np.minimum(free - 0.10, look)
                able = reach >= float(k["field_min_m"])
                if not able.any():
                    continue
                kap = np.where(np.abs(self.kappa) < 1e-6, 1e-6, self.kappa)
                ex = np.sin(kap * reach) / kap - AXLE_BACK
                ey = (1.0 - np.cos(kap * reach)) / kap
                eth = kap * reach
                at = np.c_[ex + AXLE_BACK * np.cos(eth), ey + AXLE_BACK * np.sin(eth)]
                value = np.where(able, self._field(at, then), np.inf).min()
                # Backing costs its length twice over, and the stop.
                value += float(k["field_reverse_cost"]) + 20.0 * length
                if best is None or value < best[0]:
                    best = (value, wheel, length)
        if (
            best is None
            or not best[0]
            < here - float(k["field_gain"]) + float(k["field_reverse_cost"]) + 16.0
        ):
            return None
        return self._start_reverse(pose, t, "field", best[1], best[2])

    def _veto(self, pose, speed, yaw_rate, t, want):
        """(curvature, clear length m, whether something ends it) for the
        command nearest `want` that does not put the body into a remembered
        point, or None when nothing leaves room to move."""
        k = self.k
        kappa = np.concatenate([[want], self.kappa])
        X, Y, TH, ds = self._rollout(
            kappa,
            speed,
            yaw_rate,
            t,
            float(k["dead_time"]) + float(k["veto_horizon_s"]),
        )
        steps = X.shape[1]
        pts = to_frame(self.points, pose)
        near = (pts[:, 0] > -1.0) & (np.hypot(pts[:, 0], pts[:, 1]) < ARC_REACH + 1.5)
        pts = pts[near]
        # A return measured from far off may stand centimeters inside the
        # lane it is the wall of.  So a point is taken to be as far from the
        # path as its noise allows: one seen again from close has next to none.
        sigma = float(k["noise_sigmas"]) * self.points[near, 2]
        best = None
        for width in (BODY_HALF_WIDTH + float(k["body_margin"]), BODY_HALF_WIDTH):
            slack = np.minimum(sigma, 0.5 * width)
            # What the body is already against cannot be driven away from by
            # counting it as hit: only what the car would go deeper into does.
            now = (
                (pts[:, 0] + AXLE_BACK >= -BODY_REAR)
                & (pts[:, 0] + AXLE_BACK <= BODY_FRONT)
                & (np.abs(pts[:, 1]) + slack <= width)
            )
            p, sl = pts[~now], slack[~now]
            if len(p):
                cos, sin = np.cos(TH)[..., None], np.sin(TH)[..., None]
                dx = p[None, None, :, 0] - X[..., None]
                dy = p[None, None, :, 1] - Y[..., None]
                fore = cos * dx + sin * dy
                side = np.abs(-sin * dx + cos * dy) + sl
                hit = (
                    (fore >= -BODY_REAR) & (fore <= BODY_FRONT) & (side <= width)
                ).any(-1)
            else:
                hit = np.zeros(X.shape, bool)
            any_hit = hit.any(1)
            first = np.where(any_hit, hit.argmax(1), steps)
            # Past the rollout: carrying on round the same bend.
            ex, ey, eth = X[:, -1], Y[:, -1], TH[:, -1]
            ec, es = np.cos(eth), np.sin(eth)
            qdx, qdy = p[None, :, 0] - ex[:, None], p[None, :, 1] - ey[:, None]
            onward = arc_clearance(
                ec[:, None] * qdx + es[:, None] * qdy,
                -es[:, None] * qdx + ec[:, None] * qdy,
                kappa[:, None],
                width - sl[None, :],
            )
            free = np.where(any_hit, first * ds, steps * ds + onward)
            if (~any_hit).any():
                # Pursuit's own command if it stays clear while it can be
                # judged, else the nearest that does.  Past that the path
                # will have been drawn again.
                pick = int(np.argmin(np.where(~any_hit, np.abs(kappa - want), np.inf)))
                self.info.update(tight=width == BODY_HALF_WIDTH, vetoed=pick != 0)
                return (
                    float(kappa[pick]),
                    float(free[pick]),
                    bool(free[pick] < ARC_REACH),
                )
            if best is None or free.max() > best[1]:
                best = (float(kappa[int(np.argmax(free))]), float(free.max()))
        # Against something whatever it does, but not yet up to it: creep
        # along whichever has most room rather than back off.
        if best is not None and best[1] >= 0.12:
            self.info.update(tight=True, vetoed=True)
            return best[0], best[1], True
        return None

    def _steer_command(self, kappa, yaw_rate, speed=0.0, t=0.0):
        k = self.k
        self.kappa_cmd = kappa
        self.sent.append((t, kappa))
        while len(self.sent) > 2 and t - self.sent[1][0] > float(k["dead_time"]) + 0.1:
            self.sent.popleft()
        want = math.atan(self.wheelbase * kappa)
        # Auto trim: what the car actually turned against what was asked for,
        # once the command has had time to act.
        self.angle_hist.append(want)
        delay = int(round(float(k["dead_time"]) / self.dt))
        delayed = self.angle_hist[0] if len(self.angle_hist) > delay else 0.0
        while len(self.angle_hist) > delay:
            self.angle_hist.popleft()
        self.angle_lag += self.dt / (self.tau + self.dt) * (delayed - self.angle_lag)
        self._learn_lag(math.tan(delayed) / self.wheelbase, yaw_rate, speed)
        # Only while the wheel is near straight and has been left alone:
        # turning, the difference is as much the lag model's error as trim.
        steady = abs(self.angle_lag) < 0.06 and abs(delayed - self.angle_lag) < 0.03
        if k["auto_trim"] and speed > 0.9 and steady:
            got = math.atan(self.wheelbase * yaw_rate / speed)
            self.trim_est += (
                float(k["auto_trim_rate"]) * self.dt * (got - self.angle_lag)
            )
            lim = float(k["auto_trim_limit"])
            self.trim_est = max(-lim, min(lim, self.trim_est))
        angle = want - float(k["steer_trim"]) - self.trim_est
        raw = angle_to_command(angle)
        a = self.dt / (float(k["smoothing_s"]) + self.dt)
        self.steer_out += a * (raw - self.steer_out)
        self.info.update(trim=self.trim_est, tau=self.tau, camera_yaw=self.cam_yaw_est)
        return self.steer_out

    def _learn_camera_yaw(self, speed, yaw_rate):
        """The camera's yaw on its mount, from a contradiction: the car is
        running straight down a straight lane, not turning and not closing
        on either wall, and the walls say the lane points off to one side.
        Then it is the camera that points off to the other.

        A degree of it is 5 cm of wall a few meters on, which the tracker
        answers by holding the car that far off the middle, and more in a
        bend; so it is taken out, slowly, where it can be told from a real
        heading error by the car being settled.
        """
        k, info = self.k, self.info
        if not k["auto_camera_yaw"] or self.searched or self.in_open or self.hoops:
            return
        if (
            self.mode != "drive"
            or speed < 1.2
            or abs(yaw_rate) > 0.12
            or info.get("vetoed")
        ):
            return
        turn, bend, offset = (
            info.get(name, 1.0) for name in ("turn", "bend", "offset")
        )
        if abs(bend) > 0.04 or abs(turn) > 0.2 or abs(offset) > 0.2:
            return
        # A camera turned left sees the lane turned right: turn < 0.
        limit = math.radians(float(k["auto_camera_yaw_limit_deg"]))
        est = self.cam_yaw_est - float(k["auto_camera_yaw_rate"]) * self.dt * turn
        self.cam_yaw_est = max(-limit, min(limit, est))

    def _bend_lead(self):
        """s further ahead than feedforward_lead_s the path's bend is read."""
        k = self.k
        return max(
            0.0, float(k["lead_per_tau"]) * self.tau - float(k["feedforward_lead_s"])
        )

    def _learn_lag(self, asked, yaw_rate, speed):
        """How late the yaw follows the wheel, from the car's own answer.

        The curvature asked for a dead time ago is run through a bank of
        lags, and the one whose output the measured curvature has followed
        best (to within a gain: the turn radius is not what is being judged)
        becomes the model the tracker and the veto predict with.  It
        matters: Gazebo's car takes about a second (its Ackermann plugin
        steers at steer_p_gain 1/s), a real servo a fraction of that, and a
        tracker tuned for either weaves or cuts in on the other.
        """
        k = self.k
        self.lag_f += self.dt / (LAG_BANK + self.dt) * (asked - self.lag_f)
        # Only while it is simply following a lane at speed: backing and
        # filling round bales, or below the speed the throttle runs smoothly
        # at, the tach dithers and the wheel is thrown lock to lock, and
        # neither says much about the lag.
        slow = speed < float(k["auto_lag_min_speed"])
        if (
            not k["auto_lag"]
            or slow
            or self.mode != "drive"
            or self.searched
            or self.in_open
        ):
            return
        got = yaw_rate / speed
        keep = 1.0 - self.dt / float(k["auto_lag_memory_s"])
        self.lag_ff = keep * self.lag_ff + self.dt * self.lag_f**2
        self.lag_fr = keep * self.lag_fr + self.dt * self.lag_f * got
        self.lag_rr = keep * self.lag_rr + self.dt * got * got
        # Only once it has turned enough to tell one lag from another.
        if self.lag_ff.min() > float(k["auto_lag_excite"]):
            error = self.lag_rr - self.lag_fr**2 / self.lag_ff
            best = float(LAG_BANK[int(np.argmin(error))])
            self.tau += self.dt / float(k["auto_lag_settle_s"]) * (best - self.tau)
            self.lag_known = True
        if self.lag_known and self.route is not None:
            self.route.set_lag(self.tau)

    # ----------------------------------------------------------- recovery

    def _stuck(self, t, v_cmd):
        k = self.k
        if v_cmd <= 0.0:
            self.cmd_since = None
            return False
        if self.cmd_since is None:
            self.cmd_since = t
        window = float(k["stuck_s"])
        if t - self.cmd_since < window or t - self.last_reverse_end < window:
            return False
        past = [h for h in self.history if t - h[0] >= window]
        if not past:
            return False
        _, x0, y0 = past[-1]
        _, x1, y1 = self.history[-1]
        return math.hypot(x1 - x0, y1 - y0) < float(k["stuck_distance"])

    def _bumped(self, pose, t, front):
        """The car is being driven and is not moving: whatever holds it is
        something the camera did not see or no longer remembers (a bale's
        far corner behind the tail, a bucket under the nose).  It is put
        into the memory as a row of points across that end of the body, so
        that the next way found does not lead back into it."""
        across = np.linspace(-BODY_HALF_WIDTH, BODY_HALF_WIDTH, 7)
        at = (BODY_FRONT + 0.04) if front else -(BODY_REAR + 0.04)
        row = from_frame(np.c_[np.full(7, at - AXLE_BACK), across], pose)
        # Remembered for bump_memory_s only: it is a guess at where the
        # thing is, and the camera cannot look at it to take it back.
        life = float(self.k["open_memory_s"] if self.in_open else self.k["memory_s"])
        self._remember(
            np.c_[row, np.zeros(7)], t - max(life - float(self.k["bump_memory_s"]), 0.0)
        )
        self.points = np.concatenate([self.points, np.c_[row, np.zeros(7)]])

    def _path_turn(self, pose):
        """How far the path, most of a meter on, points off the car's
        heading (rad, + left): which way the car has to come round."""
        if self.path is None or len(self.path) < 3:
            g = to_frame(self.goal[None, :], pose)[0]
            return math.atan2(g[1], g[0])
        d = np.hypot(self.path[:, 0] - pose[0], self.path[:, 1] - pose[1])
        j = min(int(np.argmin(d)) + 5, len(self.path) - 2)
        ahead = self.path[min(j + 3, len(self.path) - 1)] - self.path[j]
        heading = math.atan2(ahead[1], ahead[0])
        return (heading - pose[2] + math.pi) % (2 * math.pi) - math.pi

    def _room_behind(self, pose, steer):
        """m the car can back up with the wheel at `steer` before its tail
        meets a remembered point."""
        pts = to_frame(self.points, pose)
        if not len(pts):
            return ARC_REACH
        kappa = math.tan(command_to_angle(steer)) / self.wheelbase
        # Backing up is driving forward in the mirror, the tail leading.
        x = -(pts[:, 0] + AXLE_BACK)
        return float(
            arc_clearance(
                x[None, :],
                pts[None, :, 1],
                np.array([[kappa]]),
                BODY_HALF_WIDTH,
                BODY_REAR,
            )[0]
        )

    def _start_reverse(self, pose, t, why, wheel=None, length=None):
        """Back and fill: reverse with the wheel turned so the nose comes
        round toward the path, as far as there is room behind."""
        self.events.append((round(t, 2), why, round(pose[0], 2), round(pose[1], 2)))
        turn = self._path_turn(pose)
        # Backing up with the wheel left swings the nose right.
        side = -1.0 if turn >= 0 else 1.0
        if self.reverse_count % 2 == 1 and abs(turn) < 0.5:
            side = -side  # the last try did not free it: the other way
        lock = float(self.k["reverse_lock"])
        options = (
            [side * lock, 0.0, -side * lock] if abs(turn) < 0.5 else [side * lock, 0.0]
        )
        room = [self._room_behind(pose, steer) for steer in options]
        if max(room) >= 0.35:
            steer = options[int(np.argmax([r >= 0.35 for r in room]))]
        elif max(room) >= 0.12:
            # Wedged between two things: whichever way gives most, however
            # little.  A hand's width back with the wheel over is the first
            # move of working free.
            steer = options[int(np.argmax(room))]
        else:
            steer = None
        if wheel is not None:
            # The search's own move: its wheel, as far as it said.
            steer = wheel * lock
            if self._room_behind(pose, steer) < 0.15:
                steer = None
        self.cmd_since = None
        self.blocked_since = None
        if steer is None:
            # Nothing behind either, as far as is remembered: rock it.  A
            # second pushing on along whichever way has most room; if that
            # has not moved it, back off regardless, wheel one way and then
            # the other.  What is remembered behind the car is a guess (a
            # bale's corner, a bump put there by the last failed try), and a
            # car held by something has nothing to lose by finding out.
            if math.hypot(pose[0] - self.boxed_at[0], pose[1] - self.boxed_at[1]) > 0.4:
                self.boxed, self.boxed_at = 0, (pose[0], pose[1])
            self.boxed += 1
            if self.boxed % 2 == 1:
                self.last_reverse_end = t
                self.history.clear()
                self.push_until = t + 1.0
                self.info.update(mode="boxed")
                return self._steer_command(self.kappa_cmd, 0.0, t=t), float(
                    self.k["v_min"]
                )
            steer = lock if (self.boxed // 2) % 2 else -lock
            length, forced = 0.3, True
        else:
            forced = False
        self.reverse_count += 1
        self.mode = "reverse"
        self.reverse = dict(
            t=t,
            x=pose[0],
            y=pose[1],
            steer=steer,
            turn=turn,
            yaw=pose[2],
            length=length,
            forced=forced,
        )
        return self._reverse_step(pose, t)

    def _reverse_step(self, pose, t):
        k, r = self.k, self.reverse
        moved = math.hypot(pose[0] - r["x"], pose[1] - r["y"])
        swung = (pose[2] - r["yaw"] + math.pi) % (2 * math.pi) - math.pi
        # Far enough, or come round to the path, or out of room behind.
        planned = r["length"] is not None
        if t - r["t"] > float(k["stuck_s"]) + 1.0 and moved < 0.05:
            # Not moving: the tail is against something behind the camera.
            self._bumped(pose, t, front=False)
            r["length"], planned = 0.0, True
        done = (
            moved >= (r["length"] if planned else float(k["reverse_distance"]))
            or (not planned and moved > 0.15 and abs(swung) >= abs(r["turn"]) - 0.15)
            or (
                moved > 0.05
                and not r["forced"]
                and self._room_behind(pose, r["steer"]) < 0.08
            )
            or t - r["t"] > float(k["reverse_timeout_s"])
        )
        if done:
            self.mode = "drive"
            self.last_reverse_end = t
            self.history.clear()
            self.cmd_since = None
            if moved >= 0.25:
                self.reverse_count = 0
            return None
        self.steer_out = r["steer"]
        self.kappa_cmd = 0.0
        self.sent.clear()
        self.info.update(mode="reverse")
        return r["steer"], -float(k["reverse_speed"])
