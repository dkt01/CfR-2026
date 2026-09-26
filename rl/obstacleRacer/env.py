"""The batched Obstacle Course environment: B cars, stepped together.

Per control step (20 Hz):

  1. the policy's action becomes a DriveCommand (observation.action_to_command:
     steering = prior + residual, speed in [-v_reverse, v_cap]) and enters
     the plant's dead-time history, with the throttle dither a slow target
     gets on the car;
  2. the plant runs 10 substeps -- wheels on the collision surface, springs,
     body pitch and roll, traction from load, reverse as the Arduino does it
     -- checking the chassis against obstacles every other substep and
     sliding a car that touched one back out along it;
  3. privileged bookkeeping: arc length along the layout's centerline, hoop
     crossings (hoop_monitor.py's gate test), clearance, terminations (an
     impact over crash_speed is a crash; a touch is only a cost);
  4. the sensor model reads the course from where the body put the camera,
     and the next observation frame is built exactly as the car's node builds
     it.

Episodes start in the start box ((-0.7, 0), yaw 0, +/-0.1 m in x and y,
+/-5 deg, from rest) or, for training coverage, dealt part way round with the
same noise: before a recent failure, before an obstacle picked uniformly from
all of them, or anywhere -- or, with stuck_start_prob, exactly where a recent
episode ended pinned against something, stopped, to practice backing out.  Every episode is one full lap from wherever it started: progress
wraps round the loop at the timing line, all three hoops have to be threaded
during the episode, and the finish counts only once the car is back past its
own start point (and, from the start box, over the timing line).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import centerline as centerline_module
import observation as O
import plant as P
import reward as R
import sensor as S

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CLEARANCE_RINGS = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
START_BOX = (-0.70, 0.0, 0.0)

# hoop_monitor.py's gate test: threaded within the posts, missed within the
# attempt gate, ignored beyond it (a far crossing of the hoop's infinite plane).
HOOP_PASS_HALF = 0.275
HOOP_ATTEMPT_HALF = 1.5
# A hoop the car is this far past along the line without having threaded it
# was driven round, which is a miss however it was done.
HOOP_LATE_M = 1.0

FAILURES = (
    "crash",
    "rollover",
    "hoop_miss",
    "pinned",
    "stall",
    "off_course",
    "no_progress",
)
OUTCOMES = (
    "finish",
    "crash",
    "rollover",
    "hoop_miss",
    "pinned",
    "stall",
    "off_course",
    "no_progress",
    "timeout",
)


def _regions():
    sys.path.insert(
        0, str(REPO / ".agents" / "skills" / "obstacle-course-regions" / "scripts")
    )
    import regions

    return regions


# Regions whose drivable surface is up off the floor.  The regions are 2D, and
# the overpass deck runs over the tunnel, so where a centerline point is in
# both it is the line's height that says which one the car is driving: on the
# deck it is the overpass, on the floor under it the tunnel.  Picking the
# first 2D hit instead labeled the tunnel exit (floor level, s ~15.6-16.3 m)
# "overpass_ramp": every one of v4's 19 held-out "pinned@overpass_ramp"
# endings at 20M was there, and the overpass section ran on to 16.3 m,
# through the helix and the tunnel.
ELEVATED_REGIONS = ("overpass_ramp", "helical_ramp")
ELEVATED_Z = 0.15  # m; the deck is at 0.64, the tunnel floor at 0


def zone_labels(lines):
    """Per layout, a zone index for every centerline point, and the names.

    Named regions come from the obstacle-course-regions skill; the lane
    between two of them is 'after_<previous>'.  Where 2D regions overlap,
    the one at the line's height wins (ELEVATED_REGIONS), and an obstacle
    the line has left is never re-entered: each is driven once a lap, so a
    later point inside its outline is the floor under it (past the tunnel,
    under the ramp), not the obstacle again.
    """
    regions = _regions()
    names = ["start"]
    per_layout = []
    for line in lines:
        labels = np.zeros(len(line.points), np.int64)
        current = "start"
        left = set()
        for i, (x, y, z) in enumerate(line.points):
            hits = [
                h for h in regions.classify(float(x), float(y)) if h.name not in left
            ]
            if len(hits) > 1:
                up = z > ELEVATED_Z
                level = [h for h in hits if (h.name in ELEVATED_REGIONS) == up]
                hits = level or hits
            if hits:
                if hits[0].name != current and current != "start":
                    left.add(current)
                current = hits[0].name
                name = current
            else:
                name = current if current == "start" else f"after_{current}"
            if name not in names:
                names.append(name)
            labels[i] = names.index(name)
        per_layout.append(labels)
    return names, per_layout


class ObstacleEnv:
    def __init__(
        self, cfg, model, n, layout_ids, seed=0, start_box_only=False, randomize=True
    ):
        self.cfg = cfg
        self.model = model
        self.n = n
        self.layout_ids = np.asarray(layout_ids, np.int64)
        self.start_box_only = start_box_only
        self.rng = np.random.default_rng(seed)
        e = cfg["env"]
        self.dt = 1.0 / float(e["control_hz"])
        self.substeps = int(e["substeps"])
        if not randomize:
            cfg = dict(cfg, randomize=dict(cfg["randomize"], enabled=False))
            self.cfg = cfg
        self.plant = P.Plant(cfg, model, n, self.rng)
        self.sensor = S.Sensor(cfg, model, n, self.rng)
        self.lines = centerline_module.Centerlines(model.layouts)
        self.zone_names, zl = zone_labels(self.lines.lines)
        self.zone_of = np.zeros(self.lines.points.shape[:2], np.int64)
        for k, labels in enumerate(zl):
            self.zone_of[k, : len(labels)] = labels
            self.zone_of[k, len(labels) :] = labels[-1]
        self._loop_geometry()
        self._hoop_geometry()
        self._deal_table()
        self._section_geometry()
        self.stack = O.Stack(n, O.frame_offsets(cfg), O.frame_dim(cfg))
        self.obs_dim = O.obs_dim(cfg)
        self.memory = O.Memory(n, cfg)
        self.heading = O.Heading(n, cfg)
        self.heading_bias = np.zeros(n)  # rad/s the heading drifts at
        # The camera: the car's ZED publishes point clouds at 12 Hz
        # (cfr_zed2i.yaml pub_frame_rate), and a cloud reaches the policy a
        # segmentation later; the control loop runs at 20 Hz on whatever
        # arrived last.  Scans are rendered every control step, kept for
        # `latency` steps, and delivered only when a frame is due.
        sc = cfg["sensor"]
        self.cam_period = 1.0 / float(sc.get("camera_hz", e["control_hz"]))
        lat = sc.get("latency_s", [0.0, 0.0])
        self.lat_steps = (
            int(round(float(lat[0]) / self.dt)),
            int(round(float(lat[1]) / self.dt)),
        )
        depth = self.lat_steps[1] + 1
        self.scan_hist = np.zeros((n, depth, int(sc["bins"])))
        self.gate_hist = np.zeros((n, depth, 2 * S.GATE_FEATURES))
        self.scan_held = np.zeros((n, int(sc["bins"])))
        self.gate_held = np.zeros((n, 2 * S.GATE_FEATURES))
        self.cam_phase = np.zeros(n)
        self.cam_lat = np.zeros(n, np.int64)
        self.act_dim = 2
        z = np.zeros(n)
        self.t = z.copy()
        self.s = z.copy()
        self.s_start = z.copy()
        self.dist = z.copy()  # arc length driven this episode, across the wrap
        self.goal = z.copy()  # dist at which the lap is complete
        self.hoop_d = np.zeros((n, 3))  # dist at which each hoop is due
        self.idx = np.zeros(n, np.int64)
        self.hoop_state = np.zeros((n, 3), np.int64)
        self.hoop_u = np.zeros((n, 3))
        self.stall_t = z.copy()
        self.touch_age = z.copy()  # s since the car last touched anything
        self.hoop_phi = z.copy()  # hoop-alignment potential, reward.py
        self.progress_mark = np.zeros((n, 2))  # (time, s) of the window start
        self.prev_action = np.zeros((n, 2))
        self.prev_steer = z.copy()
        self.prev_dsteer = z.copy()
        self.prior = z.copy()
        self.ret = z.copy()
        self.min_clear = z.copy()
        self.top_speed = z.copy()
        self.terms = {k: z.copy() for k in R.TERMS}
        self.yaw_noise = z.copy()
        self.speed_noise = z.copy()
        self.start_kind = np.zeros(n, np.int64)
        self.attitude = np.zeros((n, len(self.zone_names), 2))
        # Per car, a layout and arc length to start at instead of dealing
        # (the per-obstacle eval); None deals as usual.
        self.forced_lay = None
        self.forced_s = None
        self.forced_floor = None  # with forced_s: never dealt behind these
        self.dither = z.copy()  # the wander on a slow speed target, m/s
        self.dither_scale = np.ones(n)
        self.prev_speed_cmd = z.copy()
        self.touches = np.zeros(n, np.int64)
        self.v_lost = z.copy()
        # Obstacles met this episode, and, over training, how often each was
        # met and failed at (read and cleared by train.py per eval).
        self.visited = np.zeros((n, len(self.zone_names)), bool)
        self.zone_attempts = np.zeros(len(self.zone_names), np.int64)
        self.zone_fails = np.zeros(len(self.zone_names), np.int64)
        # Decaying counts, per obstacle, of training runs that met it and of
        # those that failed at it, for weighting section starts toward what
        # the policy cannot yet do.  A run that ends in the lane after an
        # obstacle ("after_tunnel") failed that obstacle.
        self.practice_met = np.zeros(len(self.zone_names))
        self.practice_fail = np.zeros(len(self.zone_names))
        self.section_of_zone = np.array(
            [
                self.zone_names.index(n[len("after_") :])
                if n.startswith("after_")
                else (-1 if n == "start" else z)
                for z, n in enumerate(self.zone_names)
            ],
            np.int64,
        )

    # ------------------------------------------------------------ geometry

    def _loop_geometry(self):
        """Per layout: the arc length where the line's end rejoins its start.

        The line starts at the start box, 0.7 m before the timing line, and
        ends on the timing line.  Driving on from its end is driving on from
        the point of its first pass over the timing line, `wrap_s`; the loop
        round the course is `lap_length - wrap_s` long.
        """
        L = len(self.lines.lines)
        self.wrap_s = np.zeros(L)
        for k, line in enumerate(self.lines.lines):
            head = line.points[:40]
            j = int(np.argmin(np.linalg.norm(head - line.points[-1], axis=1)))
            self.wrap_s[k] = line.arc[j]
        self.loop = self.lines.lap_length - self.wrap_s

    def _section_geometry(self):
        """Per layout: where each named obstacle's stretch of line starts and ends.

        `section_s[k][z]` is (s_in, s_out) for zone z on layout k, or absent
        where the line never enters it.  The lanes between obstacles
        ('after_*') and the start are not sections.
        """
        self.sections = [
            z
            for z, name in enumerate(self.zone_names)
            if name != "start" and not name.startswith("after_")
        ]
        self.section_s = []
        for k, line in enumerate(self.lines.lines):
            zones = self.zone_of[k, : len(line.points)]
            spans = {}
            for z in self.sections:
                hit = np.flatnonzero(zones == z)
                if len(hit):
                    spans[z] = (float(line.arc[hit[0]]), float(line.arc[hit[-1]]))
            self.section_s.append(spans)

    def _hoop_geometry(self):
        """Per layout: hoop centers, span axes, travel normals, and arc lengths."""
        spec = self.model.spec["hoops"]
        L = len(self.model.layouts)
        self.hoop_c = np.zeros((L, 3, 2))
        self.hoop_a = np.zeros((L, 3, 2))
        self.hoop_n = np.zeros((L, 3, 2))
        self.hoop_s = np.zeros((L, 3))
        for k, layout in enumerate(self.model.layouts):
            line = self.lines.lines[k]
            for h, name in enumerate(spec["names"]):
                x, y = layout["hoops"][name]
                yaw = float(spec[name]["yaw"])
                a = np.array([math.cos(yaw), math.sin(yaw)])
                nrm = np.array([-a[1], a[0]])
                i = line.nearest_index((x, y, 0.0))
                j = min(i + 3, len(line.points) - 1)
                tangent = line.points[j, :2] - line.points[max(i - 3, 0), :2]
                if nrm @ tangent < 0:
                    nrm = -nrm
                self.hoop_c[k, h] = (x, y)
                self.hoop_a[k, h] = a
                self.hoop_n[k, h] = nrm
                self.hoop_s[k, h] = line.arc[i]

    def _deal_table(self):
        """Per layout: arc lengths a car can be dealt in at, part way round.

        Anywhere round the loop, not on top of a bucket, a hoop or a Wide
        Section bale.
        The helix is dealt too (place() sets the car on its segments): it is
        where v1 crashed, and a policy that only meets it at the end of a
        ramp climb practices it a few times per million steps.
        """
        e = self.cfg["env"]
        self.deal = []
        for k, (line, layout) in enumerate(zip(self.lines.lines, self.model.layouts)):
            buckets = np.asarray(layout["buckets"]).reshape(-1, 2)
            ok = []
            for s in np.arange(0.5, line.lap_length - 0.5, 0.25):
                if not e.get("deal_on_helix", True) and (
                    line.helix_start_s - 0.8 <= s <= line.helix_end_s + 0.3
                ):
                    continue
                x, y, _, _ = line.pose_at(s)
                if (
                    len(buckets)
                    and np.min(np.hypot(buckets[:, 0] - x, buckets[:, 1] - y)) < 0.65
                ):
                    continue
                if (
                    np.min(np.hypot(self.hoop_c[k, :, 0] - x, self.hoop_c[k, :, 1] - y))
                    < 0.6
                ):
                    continue
                if centerline_module.near_wide_bale(layout, x, y, 0.45):
                    continue
                ok.append(s)
            self.deal.append(np.asarray(ok))
        # Where recent training episodes failed, per layout, for dealing
        # starts a few meters before the obstacles the policy cannot yet do.
        memory = int(e.get("fail_memory", 400))
        self.fail_s = np.zeros((len(self.deal), memory))
        self.fail_n = np.zeros(len(self.deal), np.int64)
        # Where recent episodes ended pinned, as (s, x, y, z, yaw), for
        # starting a car right there, stopped, to learn to back out.
        self.stuck_pose = np.zeros((len(self.deal), memory, 5))
        self.stuck_n = np.zeros(len(self.deal), np.int64)

    # --------------------------------------------------------------- reset

    def reset(self):
        self._reset_idx(np.arange(self.n))
        return self.stack.obs.copy()

    def _reset_idx(self, idx, attempt=0, lay=None):
        e = self.cfg["env"]
        k = len(idx)
        if lay is None:
            lay = (
                self._pick_layouts(k)
                if self.forced_lay is None
                else self.forced_lay[idx]
            )
        box = self.start_box_only | (self.rng.random(k) < float(e["start_box_prob"]))
        if self.forced_s is not None:
            box[:] = False
        xy = float(e["start_xy_noise"])
        yawn = math.radians(float(e["start_yaw_noise_deg"]))
        x = np.empty(k)
        y = np.empty(k)
        z = np.empty(k)
        yaw = np.empty(k)
        speed = np.zeros(k)
        s0 = np.zeros(k)
        stuck = np.zeros(k, bool)
        p_stuck = (
            0.0
            if self.forced_s is not None or self.start_box_only
            else float(e.get("stuck_start_prob", 0.0))
        )
        for i in range(k):
            n_stuck = min(self.stuck_n[lay[i]], self.stuck_pose.shape[1])
            if not box[i] and n_stuck and self.rng.random() < p_stuck:
                pose = self.stuck_pose[lay[i], self.rng.integers(n_stuck)]
                s0[i], x[i], y[i], z[i], yaw[i] = pose
                stuck[i] = True
            elif box[i]:
                x[i], y[i], z[i] = START_BOX
                yaw[i] = 0.0
            else:
                table = self.deal[lay[i]]
                s0[i] = table[self.rng.integers(len(table))]
                n_fail = min(self.fail_n[lay[i]], self.fail_s.shape[1])
                pick = self.rng.random()
                p_fail = float(e.get("fail_start_prob", 0.0)) if n_fail else 0.0
                p_sec = float(e.get("section_start_prob", 0.0))
                target, floor = None, -np.inf
                if pick < p_fail:
                    back = self.rng.uniform(*e["fail_backoff_m"])
                    target = self.fail_s[lay[i], self.rng.integers(n_fail)] - back
                elif pick < p_fail + p_sec:
                    target, floor = self.section_target(lay[i])
                if self.forced_s is not None:
                    target = self.forced_s[idx[i]]
                    if self.forced_floor is not None:
                        floor = self.forced_floor[idx[i]]
                if target is not None:
                    s0[i] = self.snap(lay[i], target, floor)
                x[i], y[i], z[i], yaw[i] = self.lines.lines[lay[i]].pose_at(s0[i])
                speed[i] = self.rng.uniform(*e["dealt_speed"])
        # A stuck start goes exactly where the car stopped: noise could put
        # it inside what it was stuck on.
        free = ~stuck
        x[free] += self.rng.uniform(-xy, xy, free.sum())
        y[free] += self.rng.uniform(-xy, xy, free.sum())
        yaw[free] += self.rng.uniform(-yawn, yawn, free.sum())
        self.plant.reset(idx, lay, x, y, z, yaw, speed)

        # A dealt start that lands the car against something is re-dealt;
        # a stuck start is against something on purpose.
        touching = (
            P.body_contact(
                self.plant.OBS, self.plant.lay[idx], self.plant.state[idx], 0.03
            )
            & ~stuck
        )
        if touching.any() and attempt < 4:
            self._reset_idx(idx[touching], attempt + 1, lay[touching])
            keep = ~touching
            idx, lay, s0 = idx[keep], lay[keep], s0[keep]
            box, stuck = box[keep], stuck[keep]
            if len(idx) == 0:
                return

        st = self.plant.state[idx]
        self.idx[idx] = self.lines.index_at(lay, s0)
        self.idx[idx], self.s[idx], _ = self.lines.project(
            lay, self.idx[idx], st[:, P.S_X], st[:, P.S_Y], st[:, P.S_Z]
        )
        self.s_start[idx] = self.s[idx]
        self.dist[idx] = 0.0
        loop = self.loop[lay]
        # A full loop back to the start point; from the start box (before the
        # timing line) that is also over the timing line.
        self.goal[idx] = np.maximum(loop, self.lines.lap_length[lay] - self.s[idx])
        # Every hoop is ahead: one just behind a dealt start is due at the
        # end of the lap.
        self.hoop_d[idx] = np.mod(self.hoop_s[lay] - self.s[idx, None], loop[:, None])
        self.t[idx] = 0.0
        self.stall_t[idx] = 0.0
        # A stuck start has just touched: the recovery window applies.
        self.touch_age[idx] = np.where(stuck, 0.0, np.inf)
        self.progress_mark[idx] = 0.0
        self.hoop_state[idx] = 0
        self.hoop_u[idx] = self._hoop_u(idx)
        self.hoop_phi[idx] = 0.0
        self.prev_action[idx] = 0.0
        self.prev_steer[idx] = 0.0
        self.prev_dsteer[idx] = 0.0
        self.ret[idx] = 0.0
        self.min_clear[idx] = np.inf
        self.top_speed[idx] = 0.0
        for v in self.terms.values():
            v[idx] = 0.0
        self.attitude[idx] = 0.0
        self.dither[idx] = 0.0
        self.prev_speed_cmd[idx] = self.plant.state[idx, P.S_V]
        self.touches[idx] = 0
        self.v_lost[idx] = 0.0
        self.visited[idx] = False
        self.start_kind[idx] = np.where(box, 0, np.where(stuck, 2, 1))
        r = self.cfg["randomize"]
        if r["enabled"]:
            self.yaw_noise[idx] = self.rng.uniform(*r["yaw_rate_noise"], len(idx))
            self.speed_noise[idx] = self.rng.uniform(*r["speed_noise"], len(idx))
            self.dither_scale[idx] = self.rng.uniform(*r["dither_scale"], len(idx))
        else:
            self.yaw_noise[idx] = 0.0
            self.speed_noise[idx] = 0.0
            self.dither_scale[idx] = 1.0

        # Heading since the start box: the car's own estimate starts off by
        # its placement error (and, dealt part way round, by what it would
        # have drifted getting there), then drifts at a per-run bias.
        hn = math.radians(float(self.cfg["env"].get("heading_init_noise_deg", 0.0)))
        self.heading.reset(
            idx, self.plant.state[idx, P.S_YAW] + self.rng.uniform(-hn, hn, len(idx))
        )
        hb = r.get("heading_bias", [0.0, 0.0]) if r["enabled"] else [0.0, 0.0]
        self.heading_bias[idx] = self.rng.uniform(*hb, len(idx))

        scan, gate = self.sensor.read(self.plant.lay, self.plant.state, idx)
        self.scan_hist[idx] = scan[:, None, :]
        self.gate_hist[idx] = gate[:, None, :]
        self.scan_held[idx] = scan
        self.gate_held[idx] = gate
        self.cam_phase[idx] = self.rng.uniform(0.0, self.cam_period, len(idx))
        self.cam_lat[idx] = self.rng.integers(
            self.lat_steps[0], self.lat_steps[1] + 1, len(idx)
        )
        fresh = np.zeros(self.n, bool)
        fresh[idx] = True
        self.stack.reset(idx, self._frame(scan, gate, idx, fresh))

    def _pick_layouts(self, k):
        return self.layout_ids[self.rng.integers(len(self.layout_ids), size=k)]

    def section_weights(self, lay=0):
        """Chance of each obstacle being the one a section start is dealt before.

        In proportion to its recent failure rate, (fails + 1) / (met + 2),
        mixed with uniform by env.section_uniform_mix so the obstacles the
        policy already clears stay in practice.  v4 spent as many section
        starts on gravel and the car wash (100% clear) as on the tunnel and
        the bank (50%) or the buckets and hoops (12%).  A mix of 1 is uniform.
        """
        cands = list(self.section_s[lay])
        rate = (self.practice_fail[cands] + 1.0) / (self.practice_met[cands] + 2.0)
        mix = float(self.cfg["env"].get("section_uniform_mix", 1.0))
        p = (1.0 - mix) * rate / rate.sum() + mix / len(cands)
        return cands, p

    def snap(self, lay, target, floor=-np.inf):
        """The dealable arc length at or before `target` on layout `lay`,
        unless that is behind `floor`: then the first one past the floor."""
        table = self.deal[lay]
        j = np.searchsorted(table, target, side="right") - 1
        if j < 0 or table[j] < floor:
            j = min(np.searchsorted(table, floor), len(table) - 1)
        return table[max(j, 0)]

    def section_floor(self, lay, zone):
        """Where the obstacle before `zone` ends on layout `lay` (-inf if none).

        A section start must not be dealt behind this: the lanes between
        obstacles are short (0.2-0.7 m before the buckets and the hoops), so
        a start a few meters back lands inside the obstacle before.  v5 dealt
        397 of 400 bucket starts inside the Wide Section and most hoop starts
        inside the buckets, so the buckets and hoops were practiced only by
        the few cars that got through what came first.
        """
        entry = self.section_s[lay][zone][0]
        ends = [
            b for z, (a, b) in self.section_s[lay].items() if z != zone and b <= entry
        ]
        return max(ends, default=-np.inf)

    def section_target(self, lay, zone=None, back=None, any_hoop=True):
        """(arc length, floor): a few meters before an obstacle on layout `lay`.

        The obstacle is `zone`, or one picked by section_weights from those
        the line passes through.  The start is never behind the end of the
        obstacle before it (section_floor).  For the hoops, it goes before
        one of the three hoops at random (any_hoop), so the second and third
        are practiced without first threading the one before.
        """
        spans = self.section_s[lay]
        if zone is None:
            cands, p = self.section_weights(lay)
            zone = cands[self.rng.choice(len(cands), p=p)]
        if back is None:
            back = self.rng.uniform(*self.cfg["env"]["section_backoff_m"])
        entry = spans[zone][0]
        if any_hoop and self.zone_names[zone] == "hoops":
            entry = max(entry, float(self.rng.choice(self.hoop_s[lay])))
        floor = self.section_floor(lay, zone)
        return max(entry - back, floor), floor

    # ---------------------------------------------------------------- step

    def _hoop_u(self, idx):
        lay = self.plant.lay[idx]
        st = self.plant.state[idx]
        d = np.stack([st[:, P.S_X], st[:, P.S_Y]], 1)[:, None, :] - self.hoop_c[lay]
        return np.einsum("khd,khd->kh", d, self.hoop_n[lay])

    def _frame(self, scan, gate, idx=None, fresh=None):
        """Observation frame for all cars, or for cars `idx`."""
        if idx is None:
            idx = np.arange(self.n)
        st = self.plant.state[idx]
        k = len(idx)
        # The Arduino reports the tachometer's magnitude with its own
        # direction estimate for the sign.
        signed = np.abs(st[:, P.S_V]) * np.where(st[:, P.S_DIR] < 0, -1.0, 1.0)
        speed = O.tach(signed + self.rng.normal(0, 1, k) * self.speed_noise[idx])
        yaw_rate = st[:, P.S_R] + self.rng.normal(0, 1, k) * self.yaw_noise[idx]
        raw = O.prior_steer(scan, gate, yaw_rate, self.cfg)
        first = fresh[idx] if fresh is not None else None
        prior = O.smooth_prior(self.prior[idx], raw, first, self.cfg)
        self.prior[idx] = prior
        memory = self.memory.update(speed, idx, first)
        heading = self.heading.update(yaw_rate + self.heading_bias[idx], idx)
        return O.frame(
            scan,
            gate,
            speed,
            yaw_rate,
            self.prev_action[idx],
            prior,
            self.cfg,
            memory,
            heading,
        )

    def _camera(self, scan, gate):
        """What the policy sees this step: the last frame to have arrived.

        A frame is due every cam_period; the one delivered was rendered
        cam_lat control steps ago.  Between frames the last one is held.
        """
        self.scan_hist = np.roll(self.scan_hist, 1, axis=1)
        self.gate_hist = np.roll(self.gate_hist, 1, axis=1)
        self.scan_hist[:, 0] = scan
        self.gate_hist[:, 0] = gate
        self.cam_phase += self.dt
        due = self.cam_phase >= self.cam_period
        self.cam_phase[due] -= self.cam_period
        rows = np.flatnonzero(due)
        self.scan_held[rows] = self.scan_hist[rows, self.cam_lat[rows]]
        self.gate_held[rows] = self.gate_hist[rows, self.cam_lat[rows]]
        return self.scan_held.copy(), self.gate_held.copy()

    def _track(self, st, lay):
        """Privileged bookkeeping for a step: arc length round the loop, hoops.

        Returns (ds, distance off the line, hoops threaded this step, a hoop
        missed this step, the hoop-alignment potential now).
        """
        s_prev = self.s.copy()
        self.idx, self.s, off = self.lines.project(
            lay, self.idx, st[:, P.S_X], st[:, P.S_Y], st[:, P.S_Z]
        )
        ds = self.s - s_prev
        self.dist += ds
        # Past the end of the line is round the loop again: carry on from
        # the same place on the line's first pass.
        wrap = np.flatnonzero(self.s >= self.lines.lap_length[lay] - 0.3)
        if len(wrap):
            lw = lay[wrap]
            self.idx[wrap] = self.lines.index_at(lw, self.s[wrap] - self.loop[lw])
            self.idx[wrap], self.s[wrap], _ = self.lines.project(
                lw, self.idx[wrap], st[wrap, P.S_X], st[wrap, P.S_Y], st[wrap, P.S_Z]
            )

        # Hoops: crossings of each hoop's plane, forward, near the hoop.
        all_idx = np.arange(self.n)
        u = self._hoop_u(all_idx)
        d = np.stack([st[:, P.S_X], st[:, P.S_Y]], 1)[:, None, :] - self.hoop_c[lay]
        w = np.abs(np.einsum("khd,khd->kh", d, self.hoop_a[lay]))
        crossed = (self.hoop_u < 0) & (u >= 0) & (self.hoop_state == 0)
        passed = crossed & (w <= HOOP_PASS_HALF)
        missed = (crossed & (w > HOOP_PASS_HALF) & (w <= HOOP_ATTEMPT_HALF)) | (
            (self.hoop_state == 0) & (self.dist[:, None] > self.hoop_d + HOOP_LATE_M)
        )
        self.hoop_state[passed] = 1
        self.hoop_state[missed & ~passed] = 2
        self.hoop_u = u
        hoops_now = passed.sum(1)
        hoop_missed = (missed & ~passed).any(1)
        # Alignment potential: in the last `window` before a hoop not yet
        # threaded, higher the nearer the car is to its center line.
        r = self.cfg["reward"]
        window = float(r["hoop_align_window_m"])
        cap = float(r["hoop_align_cap_m"])
        near = (
            (self.hoop_state == 0) & (u < 0) & (u >= -window) & (w <= HOOP_ATTEMPT_HALF)
        )
        phi = (near * (1.0 - np.minimum(w, cap) / cap)).sum(1) * float(r["hoop_align"])
        return ds, off, hoops_now, hoop_missed, phi

    def step(self, action):
        cfg, e = self.cfg, self.cfg["env"]
        action = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        steer, speed_cmd = O.action_to_command(action, self.prior, cfg)
        steer, speed_cmd = self.plant.push_command(steer, speed_cmd)
        speed = speed_cmd + self._dither(speed_cmd)
        touched, lost, rolled = self.plant.step(steer, speed, self.substeps)
        crashed = lost > float(e["crash_speed"])
        self.touches += touched
        self.v_lost += lost
        st = self.plant.state
        lay = self.plant.lay
        self.t += self.dt
        ds, off, hoops_now, hoop_missed, phi = self._track(st, lay)

        # Terminations.
        v = np.abs(st[:, P.S_V])
        self.stall_t = np.where(
            v < float(e["stall_speed"]), self.stall_t + self.dt, 0.0
        )
        # Stopped against something it touched in the last pinned_window_s
        # is "pinned".  It gets pinned_s, not stall_s, before the run ends:
        # time for the Arduino to change direction (~0.5 s under the tach
        # floor), back off and drive on.  v4 at 20M: 61% of pinned cars never
        # commanded reverse in their last 2.5 s -- too short to learn in.
        self.touch_age = np.where(touched, 0.0, self.touch_age + self.dt)
        recovering = self.touch_age <= float(e["pinned_window_s"])
        limit = np.where(
            recovering, float(e.get("pinned_s", e["stall_s"])), float(e["stall_s"])
        )
        stalled = self.stall_t >= limit
        all_idx = np.arange(self.n)
        off_course = off > float(e["off_course_m"])
        finished = (self.dist >= self.goal) & (self.hoop_state == 1).all(1)
        window = float(e["progress_window_s"])
        due = self.t - self.progress_mark[:, 0] >= window
        # Backing off a wall is progress lost on purpose: a window with a
        # touch in it is not judged (the stall clock still is).
        no_progress = (
            due
            & (self.touch_age >= window)
            & (self.dist - self.progress_mark[:, 1] < float(e["progress_window_m"]))
        )
        self.progress_mark[due] = np.c_[self.t[due], self.dist[due]]
        # Circling or edging about without getting anywhere is stopping by
        # another name: it ends the run the same way.
        stopped = stalled | no_progress
        timeout = self.t >= float(e["episode_s"])
        crash = crashed | rolled
        # Stuck against what it just hit is charged reward.pinned, set so
        # that with the longer wait it costs no more than hitting the wall
        # hard enough to crash (reward.py checks it).  Otherwise a head-on
        # touch at walking pace costs more than a crash -- v3/v4-smoke: every
        # stall at step 0 came within 3 s of a contact, and the policy learned
        # to crawl.  Parking in the open stays the worst ending.
        pinned = stopped & recovering

        # Reward.
        clearance = P.body_clearance(self.plant.OBS, lay, st, CLEARANCE_RINGS)
        self.min_clear = np.minimum(self.min_clear, clearance)
        self.top_speed = np.maximum(self.top_speed, np.abs(st[:, P.S_V]))
        dsteer = steer - self.prev_steer
        ddsteer = dsteer - self.prev_dsteer
        lap_time = self.t * self.goal / np.maximum(self.dist, 1e-3)
        # Potential-based shaping, gamma * phi' - phi, with phi' = 0 once the
        # run is over, so it can steer the car to the hoop's center without
        # changing which way of driving pays best.
        over = finished | crash | hoop_missed | stopped | off_course
        gamma = float(cfg["train"]["gamma"])
        align = np.where(over, 0.0, gamma * phi) - self.hoop_phi
        self.hoop_phi = phi
        total, terms = R.step_reward(
            cfg,
            ds,
            self.dt,
            hoops_now,
            finished,
            lap_time,
            crash,
            hoop_missed,
            stopped & ~pinned,
            off_course,
            clearance,
            dsteer,
            ddsteer,
            lost,
            speed_cmd - self.prev_speed_cmd,
            align,
            pinned,
        )
        self.prev_speed_cmd = speed_cmd
        self.prev_dsteer = dsteer
        self.prev_steer = steer
        self.prev_action = action
        self.ret += total
        for k, v_ in terms.items():
            self.terms[k] += v_

        # Attitude per zone, for the check that the car is not flat.
        zone = self.zone_of[lay, self.idx]
        self.visited[all_idx, zone] = True
        ap = np.abs(st[:, P.S_PITCH])
        ar = np.abs(st[:, P.S_ROLL])
        self.attitude[all_idx, zone, 0] = np.maximum(
            self.attitude[all_idx, zone, 0], ap
        )
        self.attitude[all_idx, zone, 1] = np.maximum(
            self.attitude[all_idx, zone, 1], ar
        )

        terminated = finished | crash | hoop_missed | stopped | off_course
        truncated = ~terminated & timeout
        scan, gate = self._camera(*self.sensor.read(lay, st))
        obs = self.stack.push(self._frame(scan, gate)).copy()

        infos = [{} for _ in range(self.n)]
        done = np.flatnonzero(terminated | truncated)
        if len(done):
            causes = np.select(
                [
                    finished,
                    rolled,
                    crashed,
                    hoop_missed,
                    pinned,
                    stalled,
                    off_course,
                    no_progress,
                ],
                [
                    "finish",
                    "rollover",
                    "crash",
                    "hoop_miss",
                    "pinned",
                    "stall",
                    "off_course",
                    "no_progress",
                ],
                "timeout",
            )
            for i in done:
                infos[i] = self._episode_info(i, causes[i])
                met = self.visited[i]
                self.zone_attempts[met] += 1
                if causes[i] in FAILURES:
                    self.zone_fails[self.zone_of[lay[i], self.idx[i]]] += 1
                if not self.start_box_only and self.forced_s is None:
                    decay = float(e.get("section_weight_decay", 0.999))
                    self.practice_met *= decay
                    self.practice_fail *= decay
                    sections = self.section_of_zone[np.flatnonzero(met)]
                    self.practice_met[np.unique(sections[sections >= 0])] += 1.0
                    if causes[i] in FAILURES:
                        where = self.section_of_zone[self.zone_of[lay[i], self.idx[i]]]
                        if where >= 0:
                            self.practice_fail[where] += 1.0
                if not self.start_box_only and causes[i] in FAILURES:
                    L = int(lay[i])
                    self.fail_s[L, self.fail_n[L] % self.fail_s.shape[1]] = self.s[i]
                    self.fail_n[L] += 1
                    if causes[i] == "pinned":
                        slot = self.stuck_n[L] % self.stuck_pose.shape[1]
                        self.stuck_pose[L, slot] = (
                            self.s[i],
                            st[i, P.S_X],
                            st[i, P.S_Y],
                            st[i, P.S_Z],
                            st[i, P.S_YAW],
                        )
                        self.stuck_n[L] += 1
                infos[i]["terminal_observation"] = obs[i].copy()
            self._reset_idx(done)
            obs[done] = self.stack.obs[done]
        return obs, total.astype(np.float32), terminated, truncated, infos

    def _dither(self, speed_cmd):
        """The wander on a slow target: first-order noise, faded out with speed."""
        e = self.cfg["env"]
        a = self.dt / (float(e["dither_tau_s"]) + self.dt)
        # Scaled so the stationary sd is dither_sd whatever the time constant.
        kick = math.sqrt((2 - a) / a)
        self.dither += a * (
            self.rng.normal(0.0, float(e["dither_sd"]) * kick, self.n) - self.dither
        )
        lo, hi = float(e["dither_full_below"]), float(e["dither_gone_above"])
        mag = np.abs(speed_cmd)
        fade = np.clip((hi - mag) / (hi - lo), 0.0, 1.0)
        # A zero target is a neutral pulse: nothing to dither.
        fade = np.where(mag < 0.05, 0.0, fade)
        return self.dither * fade * self.dither_scale

    def _episode_info(self, i, cause):
        lay = int(self.plant.lay[i])
        zone = self.zone_names[int(self.zone_of[lay, self.idx[i]])]
        att = {
            self.zone_names[z]: (
                float(np.degrees(self.attitude[i, z, 0])),
                float(np.degrees(self.attitude[i, z, 1])),
            )
            for z in np.flatnonzero(self.attitude[i, :, 0] > 0)
        }
        return {
            "episode": {
                "r": float(self.ret[i]),
                "l": int(round(self.t[i] / self.dt)),
                "t": float(self.t[i]),
            },
            "outcome": str(cause),
            "seed": int(self.model.seeds[lay]),
            "start": ("box", "dealt", "stuck")[int(self.start_kind[i])],
            "s_start": float(self.s_start[i]),
            "s_end": float(self.s[i]),
            "dist": float(self.dist[i]),
            "goal": float(self.goal[i]),
            "lap_length": float(self.lines.lap_length[lay]),
            "time": float(self.t[i]),
            "hoops": int((self.hoop_state[i] == 1).sum()),
            "zone": zone,
            "pose": [
                float(v)
                for v in self.plant.state[
                    i, [P.S_X, P.S_Y, P.S_Z, P.S_YAW, P.S_PITCH, P.S_ROLL, P.S_V]
                ]
            ],
            "min_clearance": float(self.min_clear[i]),
            "top_speed": float(self.top_speed[i]),
            "touches": int(self.touches[i]),
            "v_lost": float(self.v_lost[i]),
            "terms": {k: float(v[i]) for k, v in self.terms.items()},
            "attitude": att,
        }

    # ------------------------------------------------------------ helpers

    def scripted_action(self):
        """The prior alone: zero residual, a fixed cruising throttle."""
        a = np.zeros((self.n, 2))
        a[:, 1] = O.speed_to_action(1.5, self.cfg)
        return a
