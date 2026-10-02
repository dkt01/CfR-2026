"""Plan Z's route hint: which way the course goes next, and how fast.

The route is the course's loop as points in the drawing's frame
(routes/*.yaml, written by make_routes.py).  It is tied to the car's own pose
frame once, at the start, by the pose the car is parked at -- the way
lap_counter finds the line -- and from then on gives:

    progress   the nearest route point, searched only a little way either
               side of the last one, so a neighboring lane or the deck over
               the tunnel is never mistaken for where the car is
    the lane   the stretch of route ahead, which the planner slides to the
               middle of the walls it sees and drives
    goals      where there is no lane (an open region: the Wide Section, the
               buckets): the places to make for, stage by stage, and a fence
               the search is kept inside
    speed      a cap per point: the section's, the bend's, what the car's
               steering lag allows, and what it can coast down to in time
               for the caps ahead

and is kept on the course as the car drives: moved over to the middle of
the walls each frame they are seen on both sides (`nudge`), and turned to
the path driven along the straights (`align`).  That is what takes out a
car parked askew, a pose that drifts and a course built off the drawing.
numpy and PyYAML only.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
WINDOW_BACK_M = 1.0
SOFT_SECTIONS = ("car_wash",)
WINDOW_AHEAD_M = 4.0


def flatten(config):
    """Config groups -> one dict of knobs; `sections` stays a dict."""
    out = {}
    for key, value in config.items():
        if isinstance(value, dict) and key != "sections":
            out.update(value)
        else:
            out[key] = value
    return out


def load_config(path=None, overrides=None):
    """config.yaml as flat knobs, with `overrides` (leaf name -> value) applied.

    A `sections.<name>` override sets one section's speed cap.
    """
    knobs = flatten(yaml.safe_load(Path(path or HERE / "config.yaml").read_text()))
    knobs["sections"] = dict(knobs["sections"])
    for name, value in (overrides or {}).items():
        if name.startswith("sections."):
            knobs["sections"][name.split(".", 1)[1]] = float(value)
            continue
        if name not in knobs:
            raise KeyError(f"no Plan Z knob named {name}")
        kind = type(knobs[name])
        if kind is bool:
            value = str(value).lower() in ("true", "1", "on", "yes")
        knobs[name] = kind(value) if kind is not str else str(value)
    return knobs


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Route:
    def __init__(self, course, knobs):
        if isinstance(course, dict):
            data = course
        else:
            path = Path(course)
            if not path.suffix:
                path = HERE / "routes" / f"{course}.yaml"
            data = yaml.safe_load(path.read_text())
        self.k = knobs
        self.course = data["course"]
        self.laps = int(data["laps"])
        self.spacing = float(data["spacing"])
        self.start_pose = tuple(float(v) for v in data["start_pose"])
        self.course_xy = np.asarray(data["points"], float)
        self.n = len(self.course_xy)
        runs = data["sections"]
        self.section = []
        for (name, first), nxt in zip(runs, runs[1:] + [[None, self.n]]):
            self.section += [name] * (nxt[1] - first)
        # Open regions: [first, last] route indices, and optionally the
        # places to make for on the way through, stage by stage -- each stage
        # a list of points any one of which will do (the Wide Section's way
        # out is whichever of four wall slots is open).
        # And optionally a fence per stage: a box (x0, y0, x1, y1) the way
        # is kept inside until the stage is reached, open where the stage's
        # points are.
        self.open, self.stages, self.fences, extra = [], [], [], []
        for region in data.get("open_regions", []):
            if isinstance(region, dict):
                self.open.append((int(region["first"]), int(region["last"])))
                stages = []
                for points in region.get("stages", []):
                    stages.append(
                        list(
                            range(
                                self.n + len(extra), self.n + len(extra) + len(points)
                            )
                        )
                    )
                    extra += [[float(x), float(y)] for x, y in points]
                self.stages.append(stages)
                fences = []
                for box in region.get("fences", []):
                    if box is None:
                        fences.append(None)
                        continue
                    x0, y0, x1, y1 = (float(v) for v in box)
                    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    rail = []
                    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
                        steps = max(2, int(math.hypot(bx - ax, by - ay) / 0.08))
                        t = np.arange(steps) / steps
                        rail += list(zip(ax + t * (bx - ax), ay + t * (by - ay)))
                    first = self.n + len(extra)
                    extra += [[x, y] for x, y in corners + rail]
                    fences.append((first, first + 4, first + 4 + len(rail)))
                self.fences.append(fences)
            else:
                self.open.append((int(region[0]), int(region[1])))
                self.stages.append([])
                self.fences.append([])
        self.extra = np.asarray(extra, float).reshape(-1, 2)
        self.stage = 0
        self.xy = self.course_xy.copy()
        # The yaw lag the bends are slowed for: lag_cautious until the
        # driver has measured the car's own (set_lag).
        self.lag = float(knobs["lag_cautious"])
        self.v = self._profile()
        self.i = 0
        self.anchored = False

    # ------------------------------------------------------------- speed

    def set_lag(self, lag):
        """Slow the bends for a car whose yaw follows the wheel in `lag` s."""
        if abs(lag - self.lag) > 0.07:
            self.lag = float(lag)
            self.v = self._profile()

    def _profile(self):
        """Speed cap per point, m/s before speed_scale."""
        k, n, ds = self.k, self.n, self.spacing
        caps = k["sections"]
        v = np.array([float(caps.get(s, caps["default"])) for s in self.section], float)
        v = np.minimum(v, float(k["v_max"]))
        # Never faster than the steering can answer for: lag_reach m is as
        # far as the car may go in the time its yaw takes to follow the wheel.
        v = np.minimum(v, float(k["lag_reach"]) / max(self.lag, 0.05))
        # Bend radius over a meter of route: the hand-placed line has corners.
        span = max(1, int(round(0.5 / ds)))
        idx = np.arange(n)
        ahead = self.course_xy[(idx + span) % n] - self.course_xy[idx]
        behind = self.course_xy[idx] - self.course_xy[(idx - span) % n]
        turn = np.abs(
            wrap(
                np.arctan2(ahead[:, 1], ahead[:, 0])
                - np.arctan2(behind[:, 1], behind[:, 0])
            )
        )
        curvature = turn / (span * ds)
        self.heading = np.arctan2(ahead[:, 1], ahead[:, 0])
        in_open = np.zeros(n, bool)
        for a, b in self.open:
            in_open[a:b] = True
        bend = np.sqrt(float(k["lat_accel"]) / np.maximum(curvature, 1e-3))
        # And no faster than the yaw can follow the bend's onset: a car
        # that answers the wheel in `lag` s is v * lag m late turning in,
        # and runs wide (or, steered early, cuts in) by that length squared
        # over the radius.
        late = np.sqrt(float(k["lag_room"]) / np.maximum(curvature, 1e-3)) / max(
            self.lag, 0.05
        )
        bend = np.minimum(bend, late)
        v = np.where(in_open, v, np.minimum(v, bend))
        v = np.maximum(v, float(k["v_min"]))
        # No brakes: backward under coast drag, forward under the bridge's slew.
        f0 = float(k["coast_f0"]) / float(k["mass"]) * float(k["route_decel_trust"])
        f1 = float(k["coast_f1"]) / float(k["mass"]) * float(k["route_decel_trust"])
        accel = float(k["accel"])
        for _ in range(3):
            for i in range(n - 1, -n - 1, -1):
                j = (i + 1) % n
                v[i % n] = min(
                    v[i % n], math.sqrt(v[j] ** 2 + 2 * (f0 + f1 * v[j]) * ds)
                )
            for i in range(2 * n):
                j = (i - 1) % n
                v[i % n] = min(v[i % n], math.sqrt(v[j] ** 2 + 2 * accel * ds))
        return v

    # ------------------------------------------------------------ anchor

    def anchor(self, pose):
        """Tie the route to the pose frame: the car is at the start pose now."""
        sx, sy, syaw = self.start_pose
        ax, ay, ayaw = pose
        c, s = math.cos(-syaw), math.sin(-syaw)
        rel = np.concatenate([self.course_xy, self.extra]) - (sx, sy)
        car = np.c_[c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]]
        # The course's shift from the drawing, in the parked car's frame.
        oyaw = math.radians(float(self.k["route_offset_yaw_deg"]))
        c, s = math.cos(oyaw), math.sin(oyaw)
        car = np.c_[c * car[:, 0] - s * car[:, 1], s * car[:, 0] + c * car[:, 1]]
        car += (float(self.k["route_offset_x"]), float(self.k["route_offset_y"]))
        c, s = math.cos(ayaw), math.sin(ayaw)
        self.xy = np.c_[
            ax + c * car[:, 0] - s * car[:, 1], ay + s * car[:, 0] + c * car[:, 1]
        ]
        self.i = int(
            np.argmin(np.hypot(self.xy[: self.n, 0] - ax, self.xy[: self.n, 1] - ay))
        )
        self.stage = 0
        idx = np.arange(self.n)
        span = max(1, int(round(0.5 / self.spacing)))
        ahead = self.xy[(idx + span) % self.n] - self.xy[idx]
        self.heading = np.arctan2(ahead[:, 1], ahead[:, 0])
        self.anchored = True
        self.trail = []  # (route index, x, y) where the car has driven
        self.trail_fit = 0
        self.align_yaw = 0.0  # total correction applied, for the log
        self.align_shift = np.zeros(2)

    def place(self, pose):
        """For tests that start part way round: the pose frame IS the
        drawing's, so the route stands where it is drawn."""
        self.anchor(self.start_pose)
        loop = self.xy[: self.n]
        self.i = int(np.argmin(np.hypot(loop[:, 0] - pose[0], loop[:, 1] - pose[1])))

    # ---------------------------------------------------------- progress

    def _open_region(self, i):
        # Entered open_lead_m early: the way through has to be in hand
        # before the car is among the bales, not after.
        lead = int(float(self.k["open_lead_m"]) / self.spacing)
        for a, b in self.open:
            if a - lead <= i < b:
                return a, b
        return None

    def _stage_points(self, region):
        """The points of the stage the car is on in `region`, or None once
        only the exit is left."""
        stages = self.stages[self.open.index(region)]
        return self.xy[stages[self.stage]] if self.stage < len(stages) else None

    def update(self, x, y, search_m=None):
        """Advance to the route point nearest (x, y); returns its index."""
        back = int((search_m or WINDOW_BACK_M) / self.spacing)
        ahead = int((search_m or WINDOW_AHEAD_M) / self.spacing)
        window = (self.i + np.arange(-back, ahead + 1)) % self.n
        d = np.hypot(self.xy[window, 0] - x, self.xy[window, 1] - y)
        new = int(window[int(np.argmin(d))])
        for a, b in self.open:
            # Out of an open region is out: its last points are still the
            # nearest for a meter, and going back to them is going back to
            # searching for the way out the car has just taken.
            if b <= self.i < b + 40 and new < b:
                new = self.i
        region = self._open_region(self.i)
        if region is None:
            self.stage = 0
        else:
            via = self._stage_points(region)
            if via is not None and np.min(
                np.hypot(via[:, 0] - x, via[:, 1] - y)
            ) < float(self.k["stage_radius"]):
                self.stage += 1
        if region is not None:
            # Inside an open region the car is wherever the obstacles let it
            # be, so the nearest point says little: it is in the region until
            # it reaches the exit.
            a, b = region
            ex, ey = self.xy[b]
            tx, ty = self.xy[(b + 3) % self.n] - self.xy[b]
            past = (x - ex) * tx + (y - ey) * ty > 0.0
            near = math.hypot(x - ex, y - ey)
            radius = float(self.k["open_exit_radius"])
            if near < radius or (past and near < 1.5):
                new = b
            else:
                lead = int(float(self.k["open_lead_m"]) / self.spacing)
                new = min(max(new, self.i), b - 1) if a - lead <= new < b else self.i
        self.i = new
        return new

    def align(self, x, y):
        """Pull the route onto where the car has actually driven.

        The car is parked by hand, give or take 5 degrees and 10 cm, and 5
        degrees at the start is 0.8 m at the helix; the pose drifts as well.
        But between walls the car is where the lane is, because the walls
        put it there.  So the last few meters of its path, each point paired
        with the route point it was nearest, are fitted with a turn and a
        shift of the route, and part of that is applied.  What is left of
        the route is its shape ahead, which is all a hint has to be.

        Not inside an open region, where the car leaves the line on purpose.
        """
        k = self.k
        if not k["route_align"] or self._open_region(self.i) is not None:
            self.trail = []
            return
        if (
            self.trail
            and math.hypot(x - self.trail[-1][1], y - self.trail[-1][2]) < 0.1
        ):
            return
        self.trail.append((self.i, x, y))
        keep = int(float(k["align_window_m"]) / 0.1)
        self.trail = self.trail[-keep:]
        self.trail_fit += 1
        if len(self.trail) < 15 or self.trail_fit < 5:
            return
        self.trail_fit = 0
        trail = np.asarray(self.trail)
        car = trail[:, 1:]
        index = trail[:, 0].astype(int)
        # Only along a straight: round a bend the car's line is not the
        # route's (it runs wide, cuts in, backs and fills), and fitting one
        # to the other turns the route a degree at a time -- four degrees
        # by the end of the bank's U-turn, which is half a meter at the
        # far side of the course.  On a straight the path driven between
        # walls is the lane's direction and nothing else.
        if np.abs(wrap(self.heading[index] - self.heading[index[-1]])).max() > 0.2:
            return
        on = self.xy[index]
        car_mid, on_mid = car.mean(0), on.mean(0)
        dc, do = car - car_mid, on - on_mid
        turn = math.atan2(
            float(np.sum(do[:, 0] * dc[:, 1] - do[:, 1] * dc[:, 0])),
            float(np.sum(do * dc)),
        )
        # Small steps: a turn about here moves the far side of the course by
        # its distance, 15 m off, and the path driven is only roughly the
        # route's (it cuts corners, backs and fills).  A degree a meter is
        # still twenty times what a pose drifts.
        gain = float(k["align_gain"])
        limit = float(k["align_turn_max"])
        turn = max(-limit, min(limit, gain * turn))
        shift = gain * (car_mid - on_mid)
        size = float(np.hypot(*shift))
        limit = float(k["align_shift_max"])
        if size > limit:
            shift *= limit / size
        c, s = math.cos(turn), math.sin(turn)
        rel = self.xy - on_mid
        self.xy = (
            on_mid
            + shift
            + np.c_[c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]]
        )
        self.heading = self.heading + turn
        self.align_yaw += turn
        self.align_shift += shift

    def nudge(self, shift):
        """Move the whole route by `shift` (pose frame): the walls say the
        lane is that far from where the route has it."""
        if not self.k["route_align"] or self._open_region(self.i) is not None:
            return
        self.xy = self.xy + shift
        self.align_shift += shift

    def goal(self):
        """Points (M, 2) to head for when the way has to be found -- any one
        will do -- and whether the car is in an open region.

        In an open region they are its current stage's, then its exit.  In a
        lane it is the route
        point goal_lookahead ahead, but no further round a bend than
        goal_turn_deg: what has not been seen counts as free, and a goal
        across the helix's middle would be headed for straight through it.
        """
        region = self._open_region(self.i)
        if region is not None:
            via = self._stage_points(region)
            if via is not None:
                return via, True
            # A way on past the exit, not the exit itself: the car should
            # come out of it pointing down the lane beyond.
            on = int(float(self.k["open_exit_ahead_m"]) / self.spacing)
            return self.xy[[(region[1] + on) % self.n]], True
        look = int(float(self.k["goal_lookahead"]) / self.spacing)
        turned = np.abs(
            wrap(
                self.heading[(self.i + np.arange(look + 1)) % self.n]
                - self.heading[self.i]
            )
        )
        over = np.flatnonzero(turned > math.radians(float(self.k["goal_turn_deg"])))
        if len(over):
            look = max(int(over[0]), int(1.0 / self.spacing))
        return self.xy[[(self.i + look) % self.n]], False

    def way_in(self, x, y):
        """Inside an open region, the lane the car came in by, as the two ends
        of a line across it (pose frame); None elsewhere, or until the car is
        well in.

        What the camera has not seen counts as free when a way is searched
        for, and the lane behind is ground it knows: left open, the way to a
        goal on the far side of a wall is back out and round.
        """
        region = self._open_region(self.i)
        if region is None:
            return None
        a = region[0] - int(float(self.k["open_lead_m"]) / self.spacing)
        at = self.xy[(a - 6) % self.n]
        along = self.xy[a % self.n] - self.xy[(a - 12) % self.n]
        along = along / max(float(np.hypot(*along)), 1e-9)
        if (x - at[0]) * along[0] + (y - at[1]) * along[1] < 1.2:
            return None
        across = np.array([-along[1], along[0]])
        return at - 1.3 * across, at + 1.3 * across

    def fence(self, x, y):
        """Inside an open region with a fence for the stage the car is on,
        and once the car is inside it: the fence as points (pose frame),
        less the stretch beside the stage's own points.  None otherwise."""
        region = self._open_region(self.i)
        if region is None:
            return None
        fences = self.fences[self.open.index(region)]
        if self.stage >= len(fences) or fences[self.stage] is None:
            return None
        first, rail, end = fences[self.stage]
        corners = self.xy[first:rail]
        # Inside, by a margin: the car comes in through it.
        edge = np.roll(corners, -1, axis=0) - corners
        to_car = np.array([x, y]) - corners
        inside = (edge[:, 0] * to_car[:, 1] - edge[:, 1] * to_car[:, 0]) / np.hypot(
            edge[:, 0], edge[:, 1]
        )
        if inside.min() < float(self.k["fence_inside_m"]):
            return None
        points = self.xy[rail:end]
        via = self._stage_points(region)
        if via is not None:
            gap = np.hypot(
                points[:, None, 0] - via[None, :, 0],
                points[:, None, 1] - via[None, :, 1],
            ).min(1)
            points = points[gap > float(self.k["fence_gap_m"])]
        return points

    def soft_line(self, behind_m=1.5, ahead_m=6.0):
        """The route through what is to be driven THROUGH, near the car (the
        car wash's hanging strands, which the camera sees as a wall across
        the lane): (M, 2) in the pose frame, or None when there is none."""
        index = (
            self.i
            + np.arange(-int(behind_m / self.spacing), int(ahead_m / self.spacing))
        ) % self.n
        soft = np.array([self.section[i] in SOFT_SECTIONS for i in index])
        if not soft.any():
            return None
        # A little either end: the first curtain hangs at the section's edge.
        for _ in range(4):
            soft = soft | np.roll(soft, 1) | np.roll(soft, -1)
        return self.xy[index[soft]]

    def ahead(self, meters):
        """The route from just behind the car to `meters` on, or None inside
        an open region or where the stretch runs into one."""
        steps = np.arange(-3, int(meters / self.spacing) + 1)
        index = (self.i + steps) % self.n
        if self._open_region(self.i) is not None:
            return None
        for a, b in self.open:
            inside = (index >= a) & (index < b)
            if inside.any():
                # Up to the region's edge: beyond it the line means nothing.
                index = index[: int(np.argmax(inside))]
        return self.xy[index] if len(index) >= 8 else None

    def corridor(self, behind_m=1.5, ahead_m=5.0):
        """Route points round the car, for the planner's lane-keeping cost;
        None inside an open region, where the lane is not known."""
        if self._open_region(self.i) is not None:
            return None
        window = (
            self.i
            + np.arange(-int(behind_m / self.spacing), int(ahead_m / self.spacing) + 1)
        ) % self.n
        return self.xy[window]

    def speed_cap(self, speed=0.0):
        """Cap at the point the command lands on, lag_s ahead."""
        lead = int(speed * float(self.k["lag_s"]) / self.spacing)
        return float(self.v[(self.i + lead) % self.n])

    def distance_to(self, x, y, ahead_m=6.0, back_m=1.0):
        """How far (x, y) is from the route between back_m behind and ahead_m on."""
        window = (
            self.i
            + np.arange(-int(back_m / self.spacing), int(ahead_m / self.spacing) + 1)
        ) % self.n
        return float(np.min(np.hypot(self.xy[window, 0] - x, self.xy[window, 1] - y)))
