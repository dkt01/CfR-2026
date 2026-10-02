#!/usr/bin/env python3
"""Plan Z's core against small made-up courses: no ROS, no Gazebo, numpy only.

Each case is a lane drawn from wall segments, a route down its middle, and a
car with the real one's steering lag (0.19 s dead time, 0.34 s yaw lag, the
measured steering table, a turn radius 1.10 times the bicycle model's).  The
car is given what the node would give it -- a noisy 36-bin scan from the
camera's place, a frame or two late, and its pose -- and has to get round
without touching.

    python3 selftest.py              # every case; exits non-zero on a failure
    python3 selftest.py oval -v      # one case, with its trace
    python3 selftest.py --plot DIR   # a picture of each run
    python3 selftest.py --set lag_room=0.8 speed_course

The cases are the lanes the two courses are made of -- an oval, the helix's
bend with and without an inside wall to see, a 0.66 m lane, the 20 in pinch,
the Speed Course's own bales -- each driven by the car as modeled and by
cars that are not it: one whose yaw answers the wheel three times as slowly
(`_lazy`), one twice as quickly (`_quick`), one like Gazebo's (`_gazebo`:
slow to wind lock on, quick to let it off), one with the camera turned 4
degrees on its mount, and one parked 5 degrees and 10 cm off the route's
start.  The driver is told none of that.  A change to the planner that breaks
one of these breaks a course; they take a few minutes, where the courses in
Gazebo take hours.

The open regions (Wide Section, buckets), the hoops and the car wash are not
here: sim_numpy.py has them.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import planner as PL  # noqa: E402
from route import Route, load_config  # noqa: E402

DT = 0.05
BINS, FOV, MAX_RANGE, MIN_RANGE = 36, math.radians(110.0), 6.0, 0.30
RAYS = 3
CAMERA_FORWARD = 0.315
BODY_FRONT, BODY_REAR, HALF_WIDTH = 0.275, 0.275, 0.1625


# ------------------------------------------------------------------ shapes


def line(a, b):
    return [(*a, *b)]


def arc(center, radius, start_deg, sweep_deg, steps=None):
    """A circular wall as short segments."""
    steps = steps or max(4, int(abs(sweep_deg) / 6))
    t = np.radians(start_deg + sweep_deg * np.arange(steps + 1) / steps)
    pts = np.c_[center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]
    return [(*pts[i], *pts[i + 1]) for i in range(steps)]


def path(points, spacing=0.1):
    """A polyline resampled every `spacing`, as a route's points."""
    points = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_len = np.concatenate([[0.0], np.cumsum(seg)])
    at = np.arange(0.0, arc_len[-1], spacing)
    return np.c_[
        np.interp(at, arc_len, points[:, 0]), np.interp(at, arc_len, points[:, 1])
    ]


def arc_points(center, radius, start_deg, sweep_deg, step_deg=3.0):
    t = np.radians(
        start_deg + np.arange(0.0, abs(sweep_deg), step_deg) * np.sign(sweep_deg)
    )
    return np.c_[center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]


def oval(width, radius=1.2, straight=4.0, inner=True):
    """Two straights joined by two left-hand half turns of `radius`: the
    helix's bend, twice, with a lane `width` wide.  Without `inner` the
    inside of the first turn cannot be seen, as on the helix: it is returned
    as hidden walls."""
    h = width / 2
    walls = []
    hidden = []
    # Bottom straight (driven +x), top straight (driven -x).
    for y0, sign in ((0.0, 1), (2 * radius, -1)):
        walls += line((0, y0 - h), (straight, y0 - h)) + line(
            (0, y0 + h), (straight, y0 + h)
        )
    walls += arc((straight, radius), radius + h, -90, 180)
    walls += arc((0, radius), radius + h, 90, 180)
    (walls if inner else hidden).extend(arc((straight, radius), radius - h, -90, 180))
    walls += arc((0, radius), radius - h, 90, 180)
    center = np.concatenate(
        [
            path([(0.5, 0), (straight, 0)]),
            arc_points((straight, radius), radius, -90, 180),
            path([(straight, 2 * radius), (0, 2 * radius)]),
            arc_points((0, radius), radius, 90, 180),
            path([(0, 0), (0.5, 0)]),
        ]
    )
    return (walls, hidden), path(center), (0.5, 0.0, 0.0)


def pinch(width, length=5.0):
    """An oval whose first straight narrows to `width` for `length`: the
    course's 20 in lane, with the 32 in lane either end of it."""
    (walls, hidden), center, start = oval(0.81, radius=1.4, straight=length + 4.0)
    h = width / 2
    for side in (-1, 1):
        walls += line((1.5, side * 0.405), (2.0, side * h))
        walls += line((2.0, side * h), (2.0 + length, side * h))
        walls += line((2.0 + length, side * h), (2.5 + length, side * 0.405))
    return (walls, hidden), center, start


# -------------------------------------------------------------------- world


class World:
    def __init__(self, walls, hidden=()):
        """`hidden` walls are there to hit but not to see, as the inside of
        the helix is."""
        self.seg = np.asarray(walls, float)
        walls = list(walls) + list(hidden)
        # Points along every wall, for the contact test.
        pts = []
        for x0, y0, x1, y1 in walls:
            n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 0.02))
            t = np.linspace(0, 1, n)
            pts.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0)])
        self.points = np.concatenate(pts)

    def scan(self, pose, yaw_offset=0.0, rng=None):
        """Nearest wall per bearing bin from the camera, as the segmenter's scan."""
        cx = pose[0] + CAMERA_FORWARD * math.cos(pose[2])
        cy = pose[1] + CAMERA_FORWARD * math.sin(pose[2])
        n = BINS * RAYS
        bearing = -FOV / 2 + FOV * (np.arange(n) + 0.5) / n
        a = pose[2] + yaw_offset + bearing
        dx, dy = np.cos(a)[:, None], np.sin(a)[:, None]
        x0, y0, x1, y1 = (self.seg[:, k][None, :] for k in range(4))
        ex, ey = x1 - x0, y1 - y0
        den = dx * ey - dy * ex
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        t = ((x0 - cx) * ey - (y0 - cy) * ex) / den
        u = ((x0 - cx) * dy - (y0 - cy) * dx) / den
        t = np.where((t > 0) & (u >= 0) & (u <= 1), t, MAX_RANGE)
        ranges = np.clip(t.min(1), MIN_RANGE, MAX_RANGE).reshape(BINS, RAYS).min(1)
        if rng is not None:
            # rl/obstacleRacer/sensor.py's noise: stereo range noise, bins
            # that read nothing, and single-frame phantoms close in.
            hit = ranges < MAX_RANGE
            ranges = ranges + hit * rng.normal(0.0, 0.5 * (0.01 + 0.008 * ranges**2))
            u = rng.random(BINS)
            ranges = np.where(u < 0.02, MAX_RANGE, ranges)
            ranges = np.where(
                (u >= 0.02) & (u < 0.025), MIN_RANGE + 0.7 * rng.random(BINS), ranges
            )
            ranges = np.clip(ranges, MIN_RANGE, MAX_RANGE)
        return ranges

    def clearance(self, pose):
        """Distance from the body's outline to the nearest wall point; <= 0
        is contact."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        dx, dy = self.points[:, 0] - pose[0], self.points[:, 1] - pose[1]
        fx, fy = c * dx + s * dy, -s * dx + c * dy
        ox = np.maximum(np.abs(fx) - BODY_FRONT, 0.0)
        oy = np.maximum(np.abs(fy) - HALF_WIDTH, 0.0)
        return float(np.hypot(ox, oy).min())


class Car:
    """The plant: rl/obstacleRacer/plant.py's steering chain on flat ground."""

    def __init__(
        self,
        pose,
        steer_bias=0.0,
        steer_gain=1.0,
        dead_time=0.19,
        yaw_tau=0.34,
        downhill=None,
        yaw_tau_on=None,
    ):
        """`downhill` is (x0, speed): past x0 the car rolls up to `speed`
        whatever is commanded, as it does down the helix with no brakes."""
        self.downhill = downhill
        self.x, self.y, self.th = pose
        self.v = 0.0
        self.r = 0.0
        self.bias, self.gain = steer_bias, steer_gain
        self.tau = yaw_tau
        # Gazebo's car: slow to wind lock on (yaw_tau_on), quick to let it
        # off (yaw_tau), as its step responses show.
        self.tau_on = yaw_tau_on
        self.queue = deque([(0.0, 0.0)] * max(1, int(round(dead_time / DT))))

    def step(self, steer, speed):
        self.queue.append((steer, speed))
        steer, speed = self.queue.popleft()
        angle = PL.command_to_angle(max(-1.0, min(1.0, steer))) * self.gain + self.bias
        angle = max(-0.512, min(0.512, angle))
        # The bridge's slew up, coast drag down: no brakes.
        if speed > self.v:
            self.v = min(speed, self.v + 2.0 * DT)
        else:
            self.v = max(speed, self.v - (0.61 + 0.13 * abs(self.v)) * DT)
        if self.downhill and self.x > self.downhill[0]:
            self.v = min(
                max(self.v, 0.0) + 2 * 0.45 * DT, max(self.v, self.downhill[1])
            )
        wheelbase = (0.324 + 0.007 * self.v**2) * 1.10
        for _ in range(5):
            h = DT / 5
            kin = self.v * math.tan(angle) / wheelbase
            tau = self.tau
            if self.tau_on is not None and kin * self.r > 0 and abs(kin) > abs(self.r):
                tau = self.tau_on
            self.r += (kin - self.r) * h / (tau + h)
            self.th += self.r * h
            self.x += self.v * h * math.cos(self.th)
            self.y += self.v * h * math.sin(self.th)

    @property
    def pose(self):
        return (self.x, self.y, self.th)


# --------------------------------------------------------------------- run


def run(case, overrides=None, verbose=False, plot=None):
    (walls, hidden), center, start = case["build"]()
    world = World(walls, hidden)
    knobs = load_config(overrides={**case.get("knobs", {}), **(overrides or {})})
    route = Route(
        dict(
            course=case["name"],
            laps=1,
            spacing=0.1,
            start_pose=list(case.get("route_start", start)),
            open_regions=case.get("open", []),
            points=center.tolist(),
            sections=[[case.get("section", "default"), 0]],
        ),
        knobs,
    )
    car = Car(start, **case.get("car", {}))
    drive = PL.Planner(knobs, route)
    drive.reset(car.pose, 0.0)
    camera_yaw = math.radians(case.get("camera_yaw_deg", 0.0))
    rng = (
        np.random.default_rng(case.get("seed", 1)) if case.get("noise", True) else None
    )
    poses = deque([car.pose] * 3, maxlen=3)
    frame_due = 0.0
    lap = len(center) * 0.1
    driven, closest, closest_at, trace = 0.0, np.inf, None, []
    outcome = "timeout"
    for step in range(int(case.get("seconds", 60.0) / DT)):
        t = step * DT
        # The camera's 12 Hz, each frame one or two ticks old on arrival.
        if t >= frame_due:
            frame_due += 1.0 / 12.0
            seen_from = (
                poses[-1 - int(rng.integers(1, 3))] if rng is not None else poses[-2]
            )
            drive.observe(world.scan(seen_from, camera_yaw, rng), [], seen_from, t)
        yaw_rate = car.r + (rng.normal(0.0, 0.02) if rng is not None else 0.0)
        steer, speed = drive.step(car.pose, car.v, yaw_rate, t)
        before = car.pose
        car.step(steer, speed)
        poses.append(car.pose)
        driven += math.hypot(car.x - before[0], car.y - before[1]) * (
            1 if car.v >= 0 else -1
        )
        gap = world.clearance(car.pose)
        if gap < closest:
            closest, closest_at = (
                gap,
                (round(car.x, 2), round(car.y, 2), round(car.v, 2)),
            )
        trace.append(
            (t, car.x, car.y, car.v, steer, speed, gap, drive.mode == "reverse")
        )
        if verbose and step % 2 == 0:
            print(
                f"  t {t:5.2f}  ({car.x:6.2f}, {car.y:6.2f}) yaw {math.degrees(car.th):+6.1f}  v {car.v:4.2f}"
                f"  steer {steer:+.2f} cmd {speed:.2f}  gap {gap:.3f}  {drive.mode}  i {route.i}"
                f"  kappa {drive.kappa_cmd:+.2f} want {drive.info.get('want', 0):+.2f} veto {drive.info.get('vetoed')} n {len(drive.points)}"
                f" align {math.degrees(route.align_yaw):+.1f}"
            )
        if gap <= 0.0:
            outcome = "contact"
            break
        if driven >= lap * case.get("laps", 1.0):
            outcome = "finish"
            break
    result = dict(
        outcome=outcome,
        time=len(trace) * DT,
        driven=driven,
        closest=closest,
        closest_at=closest_at,
        reversals=len(drive.events),
        tau=drive.tau,
        camera_yaw=math.degrees(drive.cam_yaw_est),
        events=drive.events[:6],
        end=(round(car.x, 2), round(car.y, 2)),
    )
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 9))
        for x0, y0, x1, y1 in world.seg:
            ax.plot([x0, x1], [y0, y1], "k-", lw=1)
        for x0, y0, x1, y1 in hidden:
            ax.plot([x0, x1], [y0, y1], "r-", lw=1)
        ax.plot(center[:, 0], center[:, 1], "b:", lw=0.6)
        tr = np.asarray(trace)
        ax.scatter(tr[:, 1], tr[:, 2], c=tr[:, 3], s=4, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(f"{case['name']}: {result}")
        fig.savefig(Path(plot) / f"{case['name']}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
    return result


def speed_course():
    """The Speed Course itself, flat: its bales from the world file as walls,
    and the route the driver is given for it."""
    import re

    import yaml

    world = (
        HERE.parents[1]
        / "jetson"
        / "cfr_arduino_bridge"
        / "worlds"
        / "speed_course.sdf"
    )
    walls = []
    for match in re.finditer(
        r'<collision name="bale[^"]*">\s*<pose>([^<]+)</pose>\s*<geometry>\s*<box>\s*<size>([^<]+)</size>',
        world.read_text(),
    ):
        x, y, _, _, _, yaw = (float(v) for v in match.group(1).split())
        hl, hw = (float(v) / 2 for v in match.group(2).split()[:2])
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [
            (x + c * a - s * b, y + s * a + c * b)
            for a, b in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))
        ]
        walls += [(*corners[i], *corners[(i + 1) % 4]) for i in range(4)]
    route = yaml.safe_load((HERE / "routes" / "speed.yaml").read_text())
    return (walls, []), np.asarray(route["points"]), tuple(route["start_pose"])


def parked(build, dx=0.0, dy=0.0, yaw_deg=0.0):
    """The same course, with the car set down off the start pose."""

    def moved():
        walls, center, start = build()
        return (
            walls,
            center,
            (start[0] + dx, start[1] + dy, start[2] + math.radians(yaw_deg)),
        )

    return moved


CASES = [
    dict(name="oval_quiet", build=lambda: oval(1.05), seconds=40, noise=False),
    dict(name="oval", build=lambda: oval(1.05), seconds=40),
    dict(
        name="lane_066", build=lambda: oval(0.66, radius=1.6, straight=8.0), seconds=60
    ),
    dict(
        name="pinch_051",
        build=lambda: pinch(0.51),
        seconds=60,
        knobs={"sections.default": 0.9},
    ),
    dict(name="oval_no_inner", build=lambda: oval(1.05, inner=False), seconds=40),
    dict(
        name="helix",
        build=lambda: oval(1.05, inner=False),
        seconds=40,
        knobs={"sections.default": 1.0},
        car=dict(downhill=(3.5, 1.8)),
    ),
    dict(
        name="oval_fast",
        build=lambda: oval(1.05, inner=False),
        seconds=40,
        knobs={"sections.default": 1.9},
    ),
    dict(
        name="speedway",
        build=lambda: oval(0.9, radius=1.5, straight=25.0),
        seconds=60,
        knobs={"sections.default": 3.0},
        laps=1.0,
    ),
    dict(name="speed_course", build=speed_course, seconds=120, section="track"),
    # Gazebo's car: its Ackermann plugin turns the wheels toward the angle
    # asked at 1/s of what is left, so the yaw answers in about a second.
    dict(
        name="speed_course_lazy",
        build=speed_course,
        seconds=150,
        section="track",
        car=dict(yaw_tau=1.0),
    ),
    dict(
        name="speedway_lazy",
        build=lambda: oval(0.9, radius=1.5, straight=25.0),
        seconds=90,
        knobs={"sections.default": 3.0},
        car=dict(yaw_tau=1.0),
    ),
    dict(
        name="oval_lazy",
        build=lambda: oval(1.05, inner=False),
        seconds=40,
        car=dict(yaw_tau=1.0),
    ),
    dict(
        name="lane_066_lazy",
        build=lambda: oval(0.66, radius=1.6, straight=8.0),
        seconds=60,
        car=dict(yaw_tau=1.0),
    ),
    dict(
        name="speed_course_gazebo",
        build=speed_course,
        seconds=150,
        section="track",
        car=dict(yaw_tau=0.4, yaw_tau_on=1.0),
    ),
    dict(
        name="speedway_gazebo",
        build=lambda: oval(0.9, radius=1.5, straight=25.0),
        seconds=90,
        knobs={"sections.default": 3.0},
        car=dict(yaw_tau=0.4, yaw_tau_on=1.0),
    ),
    # And a car quicker than the model: the real servo may well be.
    dict(
        name="speed_course_quick",
        build=speed_course,
        seconds=120,
        section="track",
        car=dict(yaw_tau=0.12, steer_gain=1.1),
    ),
    dict(
        name="lane_066_quick",
        build=lambda: oval(0.66, radius=1.6, straight=8.0),
        seconds=60,
        car=dict(yaw_tau=0.12, steer_gain=1.1),
    ),
    # The camera turned on its mount, which the driver is not told.
    dict(
        name="speed_course_camera_left",
        build=speed_course,
        seconds=120,
        section="track",
        camera_yaw_deg=4.0,
    ),
    dict(
        name="speed_course_camera_right",
        build=speed_course,
        seconds=120,
        section="track",
        camera_yaw_deg=-4.0,
    ),
    # Less of it in a lane with 17 cm a side: the first bend comes before
    # there has been a straight to find the yaw on.
    dict(
        name="lane_066_camera_left",
        build=lambda: oval(0.66, radius=1.6, straight=8.0),
        seconds=60,
        camera_yaw_deg=2.5,
    ),
    dict(
        name="oval_parked_left",
        build=parked(lambda: oval(1.05, inner=False), 0.1, 0.1, 5.0),
        route_start=(0.5, 0.0, 0.0),
        seconds=40,
    ),
    dict(
        name="oval_parked_right",
        build=parked(lambda: oval(1.05, inner=False), -0.1, -0.1, -5.0),
        route_start=(0.5, 0.0, 0.0),
        seconds=40,
    ),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="*")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--plot")
    ap.add_argument("--set", action="append", default=[], help="knob=value")
    args = ap.parse_args()
    overrides = dict(o.split("=", 1) for o in args.set)
    failed = 0
    for case in CASES:
        if args.cases and case["name"] not in args.cases:
            continue
        result = run(case, overrides, args.verbose, args.plot)
        ok = result["outcome"] == case.get("expect", "finish")
        failed += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {case['name']:18s} {result['outcome']:8s} "
            f"{result['time']:5.1f} s  {result['driven']:5.1f} m  closest {result['closest']:.3f} m  "
            f"reversals {result['reversals']}  lag {result['tau']:.2f}  camera {result['camera_yaw']:+.1f} deg  end {result['end']}"
            + f"  closest at {result['closest_at']}"
            + (f"  {result['events']}" if result["events"] else "")
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
