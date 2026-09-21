#!/usr/bin/env python3
"""Race the course with no offline plan at all: follow-the-gap + pure pursuit
+ a small CasADi speed/steering coupling, replanned from a live range scan
every tick.

Why this exists (2026-09-20): path_racer.py drives off course_path.json, a
racing line and speed profile computed once, offline, from the bale SDF.
That plan is only as good as its assumptions about grip and tracking error --
raise the assumed grip a little (--traction 0.6 -> 0.75) and a real run wedged
repeatedly at the first hairpin, because the plan's minimum speed rose with
it and the plan itself has no way to notice, mid-corner, that reality
disagrees. A wall-follower trim bolted on top (a few-cm nudge to the aim
point) cannot fix an entry speed that is wrong by 60%.

This drops the offline plan entirely. There is no course_path.json, no path
index, no pre-computed speed profile: every tick reads the corridor that is
actually in front of the car right now and picks a speed and steering angle
from *that*, not from a number computed offline against assumptions nobody
re-checked. It cannot be wrong about the hairpin ahead in the way the offline
plan was, because it never assumed anything about the hairpin ahead -- it
looked.

The scan is analytic: bale_geometry.lidar_scan ray-casts against the bale
OBBs parsed from the SDF, standing in for the ZED (see bale_geometry.py's
own docstring on that substitution). Ranges are then made body-relative
rather than centre-relative -- see footprint_extent, without which every
threshold in this file is optimistic by most of a car.

Ingredients, genuinely combined rather than layered as patches on a plan:

  * strict corridor/wall following: the free span that CONTAINS straight
    ahead, at the deepest probe depth that still has one, is "the corridor";
    aim at its centre, then bias toward whichever side has more room so the
    car holds the middle rather than merely pointing along it. This is
    deliberately NOT widest-gap-anywhere follow-the-gap: see
    CORRIDOR_PROBE_DEPTHS for how that put the nose into walls.
  * pure pursuit: the corridor bearing becomes a steering angle via the
    standard lookahead-curvature formula.
  * CasADi: not tracking a spatial reference (there is no path to track) --
    a short joint optimisation over just (speed, steering) that couples them
    through the friction circle, so the accel limit and the steering-rate
    limit are satisfied *together*. Clamping each independently can still
    land on a (v, delta) pair that violates the combined grip limit; solving
    them jointly cannot.

Speed comes from two live caps every tick, not a lookup table: the friction
circle applied to whatever curvature the CURRENT steering command needs, and
a braking-distance cap from the ACTUAL sensed range in the chosen direction.

    ros2 launch cfr_arduino_bridge training.launch.py    # or validate.sh's stack
    python medley_racer.py

    DEMO_PROGRAM=medley_racer.py ./validate.sh              # watch in the viewer
    python medley_racer.py --self-test                      # no ROS, no Gazebo
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

import casadi
import numpy as np

import bale_geometry
from env import MAX_STEERING_ANGLE, WHEELBASE, _unpause_world, _yaw_from_quaternion

GRAVITY = 9.81
DEFAULT_SDF = Path(__file__).resolve().parents[2] / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"

# Follow-the-gap scan. FOV and max range match config_lap.yaml's lidar
# (150 deg / 6 m is a forward-biased window -- the corridor behind the car
# does not matter to where it is about to steer).
SCAN_BINS = 61
SCAN_FOV_DEG = 150.0
SCAN_MAX_RANGE = 6.0
# Minimum ANGULAR width (in bins) for a free span to count as "the corridor"
# rather than a sliver between two close readings.
MIN_GAP_BINS = 3
BEARING_SMOOTHING = 0.35  # low-pass on the chosen bearing, 0=no smoothing

# Depths the corridor search probes, deepest first: the corridor direction is
# the centre of the free span that CONTAINS straight ahead at the deepest
# deapth that still has one.
#
# This replaced a plain widest-gap-anywhere follow-the-gap, which is what put
# the nose into walls. Picking the WIDEST span across a 150 deg FOV means a
# diagonal sightline past a bend can beat the corridor the car is actually
# in, so the aim point swung to +-1.3 rad (+-75 deg, the FOV edges) and the
# car drove at the wall between them -- measured live, repeatedly, with the
# bearing flipping sign every couple of seconds. Anchoring the span to
# straight ahead instead means the answer is always "where does the corridor
# I am in actually go", which in a 0.95 m corridor is the only question worth
# asking.
CORRIDOR_PROBE_DEPTHS = (4.0, 3.0, 2.0, 1.4, 1.0, 0.7)

# Strict wall following on top of the corridor direction: hold the middle by
# correcting toward whichever side has more room. Deliberately gentle and
# clamped -- in a corner one wall is legitimately closer, and over-centering
# fights the corner instead of taking it.
SIDE_BAND_DEG = 50.0  # bins beyond this angle count as "beside", not "ahead"
CENTERING_GAIN = 0.25  # rad of aim-point bias per metre of left/right imbalance
CENTERING_MAX = 0.30  # rad
# Full width of a forward cone used only for the braking-distance speed cap
# (see plan_command). Follow-the-gap deliberately seeks the WIDEST apparent
# opening, which is often a diagonal sightline through a bend rather than
# straight ahead -- using that gap's own depth for braking distance is
# exactly backwards, since the gap-follow finding a wide-looking way through
# is what makes its own depth reading optimistic. A self-test collision at
# the first hairpin measured the straight-ahead reading at 1.36 m while the
# chosen gap read 3.83 m at the same instant; braking distance has to see
# the smaller number.
#
# Narrow on purpose: the corridor is 0.95 m wide, so a WIDE cone reads its
# own side walls at a few tens of degrees off-axis regardless of whether the
# corridor ahead is straight or curving -- half-width / sin(angle) is ~1.1 m
# at 25 deg on a dead-straight section, which capped speed to the same ~3.5
# m/s everywhere, straights included, the first time this used 50 deg. 8 deg
# reads the full 6 m scan range on a straight (measured) while still reading
# 1.2 m approaching the hairpin that caused the collision above (also
# measured) -- narrow enough to see past the corridor's own width, wide
# enough to see a real bend closing in ahead.
FORWARD_CONE_DEG = 8.0

# Pure pursuit lookahead, same shape as path_racer.py's (this is a standard
# formula, not something specific to the old plan).
MIN_LOOKAHEAD = 0.6
MAX_LOOKAHEAD = 2.0
LOOKAHEAD_GAIN = 0.4  # seconds of travel
# A short, FIXED lookahead checked alongside the speed-scaled one above, for
# the speed law only (never for the actual steering command). At speed, the
# scaled lookahead grows past 1 m, so the same gap bearing reads as a gentle
# curvature right up until the car is almost on top of a sharp bend -- there
# is otherwise no anticipation at all, which is exactly how a self-test run
# hit 7.7 m/s on the entry straight and collided at the first hairpin before
# the reactive cap ever saw it coming. Same bearing at a shorter assumed
# distance implies more curvature, so the speed cap sees the bend early even
# though the steering command itself still uses the smooth, speed-scaled one.
NEAR_LOOKAHEAD = 1.0

# Speed law.
# Speed held through corners, blended by curvature between these two. These
# are FLOORS, not caps: the friction circle and the braking-distance cap
# still bound speed from above, and the turn floor only lifts it when there
# is room (see speed_target). TURN_SPEED_MIN_LOCK is the "maintain this
# through a hairpin" number -- 2.4 m/s, which sits under the 2.6 m/s the
# friction circle allows at full lock on traction 0.9, so it is a speed the
# tires can actually hold rather than one that asks them to slide.
TURN_SPEED_MIN = 0.9  # tire-scrub floor straight-ahead
TURN_SPEED_MIN_LOCK = 2.4  # held through full-lock hairpins
BRAKE_STOP_BUFFER = 0.50  # m; from the BODYWORK (see footprint_extent), not the centre
# The car NEVER commands less than this while it still has room to move.
# Without it the braking cap returns a hard 0 inside BRAKE_STOP_BUFFER, the
# turn floor is gated off that close, and whatever is left lands under
# sim_vehicle_node's 0.05 m/s neutral deadband and is discarded -- a dead
# stop the controller has no way out of, which is exactly what a live run
# did (sat still at full lock commanding ~0.02 m/s). 0.9 also clears the
# ~0.8 m/s below which the steered wheels cannot overcome tire scrub (see
# mpc_tracker.py), so creeping still STEERS rather than just scrubbing.
# The car never commands below this, full stop -- reversing is disabled, so a
# zero command is not a pause, it is a permanent stall with nothing left to
# recover it. Steering only works while rolling, so the one move always
# available has to be "keep rolling and steer out of it".
CREEP_SPEED = 0.9
# Room demanded beyond what the bodywork strictly needs, in corridor_target's
# span-width check. The ranges it checks are already body-relative (see
# footprint_extent), so this is no longer compensating for the scan measuring
# from the wrong place -- it is purely the allowance for everything the
# idealised kinematic self-test does not have: sensor noise, drivetrain lag,
# control-loop jitter. Measured: 0.15 -> 0.05 m worst-case clearance,
# 0.24 -> 0.09 m, 0.28 -> over-constrains (the corridor search starts falling
# through to its least-bad fallback and clearance gets WORSE, 0.04 m).
CLEARANCE_MARGIN = 0.24

# CasADi speed/steering coupling: a few steps, not a spatial horizon -- there
# is no path to look ahead along, only a (v, delta) target to arrive at
# without breaking the drivetrain's own limits.
SMOOTH_HORIZON = 5
SMOOTH_DT = 0.08
MAX_STEERING_RATE = 3.5  # rad/s, servo-limited; see casadi_smoother.py

# Direction of travel is locked to the heading the car started with: the
# corridor is a closed loop and its walls look much the same driven either
# way round, so nothing in a purely reactive scan says which way is "forward"
# -- left to itself the car will happily pick the way it came and lap
# backwards, which is what a live run did after a reversal spun it round.
#
# Absolute heading cannot express this on a serpentine (a lap legitimately
# points the car at every compass direction), so what is held is a
# slow-tracking heading of travel: it follows the car's actual yaw with a
# long time constant, so real cornering carries it along, while a sudden
# swing to the opposite direction outruns it and gets clamped. 90 deg is the
# limit because that is the boundary between "turning" and "turning back".
TRAVEL_TRACK_GAIN = 0.06  # per tick; ~0.8 s time constant at 20 Hz
MAX_TRAVEL_DEVIATION = math.radians(150.0)


def footprint_extent(offsets: np.ndarray) -> np.ndarray:
    """Distance from the car's centre to its own bodywork, per scan bearing.

    bale_geometry.lidar_scan ray-casts from the car's CENTRE, so every range
    it reports includes the half-car the rays start inside of -- its own
    docstring for body_clearance says as much ("a scan reading of 0.15 m is
    already inside the 0.30 m-wide chassis"). Measured over a full lap, that
    makes the scan overstate the room the car actually has by 0.18 m on
    average and up to 0.44 m, which is a systematic optimism sitting under
    every threshold in this file: BRAKE_STOP_BUFFER, the corridor probe
    depths, the gap-width check.

    Subtracting this turns "distance from a point at the car's centre" into
    "distance from the bodywork", which is what those thresholds were always
    meant to be about. Exact for a rectangle: a ray leaving the centre at
    body-frame angle t exits the footprint at whichever of the two half-
    extents it reaches first.
    """
    half_l = bale_geometry.CHASSIS_LENGTH / 2.0
    half_w = bale_geometry.CHASSIS_WIDTH / 2.0
    cos_t = np.abs(np.cos(offsets))
    sin_t = np.abs(np.sin(offsets))
    with np.errstate(divide="ignore"):
        along = np.where(cos_t > 1e-9, half_l / np.maximum(cos_t, 1e-9), np.inf)
        across = np.where(sin_t > 1e-9, half_w / np.maximum(sin_t, 1e-9), np.inf)
    return np.minimum(along, across)


def corridor_target(
    ranges: np.ndarray,
    offsets: np.ndarray,
    prev_bearing: float,
    car_half_width: float,
) -> tuple[float, float]:
    """(bearing, depth) of the corridor the car is actually in.

    Strict corridor following, not widest-gap-anywhere: take the free span
    that CONTAINS straight ahead (or the nearest free bin to it when the nose
    itself is blocked) at the deepest probe depth that still yields one, and
    aim at that span's centre. See CORRIDOR_PROBE_DEPTHS for why the anchor
    matters -- unanchored, the widest span in a 150 deg FOV is regularly a
    diagonal past a bend rather than the way the corridor goes.

    A wall-following centring bias is then layered on top, so the car holds
    the middle of the corridor rather than merely pointing along it.

    `offsets` are the scan's angular offsets from the car's current heading
    (radians); `ranges` are the matching distances. Pure function, no ROS, so
    --self-test exercises exactly this.
    """
    n = len(ranges)
    centre_bin = n // 2
    bin_width = float(offsets[1] - offsets[0]) if n > 1 else 0.1

    chosen = None
    for depth in CORRIDOR_PROBE_DEPTHS:
        free = ranges >= depth
        if not free.any():
            continue
        if free[centre_bin]:
            seed = centre_bin
        else:
            free_idx = np.flatnonzero(free)
            seed = int(free_idx[np.argmin(np.abs(free_idx - centre_bin))])
        lo = hi = seed
        while lo - 1 >= 0 and free[lo - 1]:
            lo -= 1
        while hi + 1 < n and free[hi + 1]:
            hi += 1
        # The span has to be wide enough for the body, not just for a ray:
        # a scan is infinitely thin lines and a corner just inside the
        # nominal opening still clips a fender.
        needed_half_angle = math.atan2(car_half_width + CLEARANCE_MARGIN, max(depth, 0.1))
        needed_bins = max(MIN_GAP_BINS, int(math.ceil(2.0 * needed_half_angle / max(bin_width, 1e-3))))
        if hi - lo + 1 >= needed_bins:
            chosen = (lo, hi, depth)
            break

    if chosen is None:
        # Nothing anywhere is both deep and wide enough. Aim at the single
        # most open direction and creep at it -- with reversing disabled
        # this is the only move left, and it is "least bad", not "good".
        seed = int(np.argmax(ranges))
        chosen = (seed, seed, float(ranges[seed]))

    lo, hi, depth = chosen
    bearing = float(offsets[(lo + hi) // 2])

    # Wall following: bias toward whichever side has more room, so the car
    # holds the corridor's middle instead of drifting onto a wall it is
    # nominally parallel to.
    side = math.radians(SIDE_BAND_DEG)
    left = ranges[offsets >= side]
    right = ranges[offsets <= -side]
    if len(left) and len(right):
        imbalance = float(left.min()) - float(right.min())
        bearing += float(np.clip(CENTERING_GAIN * imbalance, -CENTERING_MAX, CENTERING_MAX))
    bearing = float(np.clip(bearing, offsets[0], offsets[-1]))

    # Low-pass so a span whose centre hops a bin tick-to-tick does not read
    # as a steering command that hops with it.
    bearing = prev_bearing + BEARING_SMOOTHING * (bearing - prev_bearing)
    return bearing, float(depth)


def _wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def pure_pursuit_steer(bearing: float, lookahead: float) -> float:
    """Standard lookahead-curvature steering law, clamped to the servo limit."""
    delta = math.atan2(2.0 * WHEELBASE * math.sin(bearing), max(lookahead, 0.3))
    return max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, delta))


def speed_target(delta: float, gap_range: float, traction: float, max_speed: float) -> float:
    """Live speed cap: friction circle on the commanded curvature, and a
    braking-distance cap on the actually-sensed range in that direction.
    Neither is a number computed offline against an assumption -- both are
    evaluated against what this tick's own command and this tick's own scan
    say, so a corner that turns out tighter than expected caps speed itself
    instead of relying on a plan to have anticipated it.
    """
    a_max = traction * GRAVITY
    kappa = abs(math.tan(delta)) / WHEELBASE
    v_grip = math.sqrt(a_max / max(kappa, 1e-6)) if kappa > 1e-6 else max_speed
    v_range = math.sqrt(max(2.0 * a_max * (gap_range - BRAKE_STOP_BUFFER), 0.0))

    kappa_full = math.tan(MAX_STEERING_ANGLE) / WHEELBASE
    blend = min(1.0, kappa / max(kappa_full, 1e-6))
    turn_floor = TURN_SPEED_MIN + blend * (TURN_SPEED_MIN_LOCK - TURN_SPEED_MIN)

    capped = min(max_speed, v_grip, v_range)
    if capped < turn_floor and gap_range > BRAKE_STOP_BUFFER + 0.3:
        capped = turn_floor
    # Creep rather than freeze, unconditionally. The braking cap returns a
    # hard 0 anywhere inside BRAKE_STOP_BUFFER, and with reversing disabled a
    # zero command is not a pause -- it is a permanent stall, because a
    # stationary car cannot steer out of whatever it is pointed at and
    # nothing else is coming to move it. Rolling is the only state in which
    # the steering has any authority at all, so the floor has no exception.
    return max(capped, CREEP_SPEED)


def plan_command(
    ranges: np.ndarray,
    offsets: np.ndarray,
    prev_bearing: float,
    v_meas: float,
    car_half_width: float,
    traction: float,
    max_speed: float,
    yaw: float = 0.0,
    travel_heading: float | None = None,
) -> tuple[float, float, float, float, float]:
    """One tick's (bearing, steering command, speed target, gap range,
    braking range).

    Shared by the ROS node and --self-test so the near-lookahead anticipation
    fix applies identically to both rather than living as two copies that can
    drift apart.

    `braking_range` is returned separately from `gap_range` because it, not
    the gap, is what actually caps speed -- and a live run where the car sat
    still commanding ~0.02 m/s was unreadable from the logs precisely because
    only `gap_range` was being printed (a healthy-looking 1.4-1.7 m) while
    `braking_range` was under BRAKE_STOP_BUFFER and zeroing the speed.
    """
    bearing, gap_range = corridor_target(ranges, offsets, prev_bearing, car_half_width)

    # Hold the direction of travel the car started in. The corridor reads the
    # same driven either way round, so without this the reactive scan has no
    # opinion about which way is forward and will lap backwards given the
    # chance. See MAX_TRAVEL_DEVIATION.
    if travel_heading is not None:
        deviation = _wrap_angle(yaw + bearing - travel_heading)
        clamped = max(-MAX_TRAVEL_DEVIATION, min(MAX_TRAVEL_DEVIATION, deviation))
        if clamped != deviation:
            bearing = _wrap_angle(travel_heading + clamped - yaw)
            bearing = float(np.clip(bearing, offsets[0], offsets[-1]))

    lookahead = float(np.clip(LOOKAHEAD_GAIN * v_meas, MIN_LOOKAHEAD, MAX_LOOKAHEAD))
    delta_cmd = pure_pursuit_steer(bearing, lookahead)
    # Speed sees whichever lookahead implies the sharper turn -- see
    # NEAR_LOOKAHEAD's comment. The steering command itself always uses the
    # smooth, speed-scaled one; only the speed cap gets the conservative one.
    delta_near = pure_pursuit_steer(bearing, NEAR_LOOKAHEAD)
    delta_for_speed = delta_cmd if abs(delta_cmd) > abs(delta_near) else delta_near
    # Braking distance uses whichever is closer: the chosen gap's own depth,
    # or the nearest thing in a forward cone around the CURRENT heading --
    # see FORWARD_CONE_DEG. The gap can be a wide-looking diagonal through a
    # bend; the cone is "what's actually coming up if I do nothing".
    half_cone = math.radians(FORWARD_CONE_DEG) / 2.0
    forward_mask = np.abs(offsets) <= half_cone
    forward_min_range = float(ranges[forward_mask].min()) if forward_mask.any() else gap_range
    braking_range = min(gap_range, forward_min_range)
    v_target = speed_target(delta_for_speed, braking_range, traction, max_speed)
    return bearing, delta_cmd, v_target, gap_range, braking_range


class SpeedSteerSmoother:
    """Joint (speed, steering) feasibility solve -- see module docstring for
    why this earns being a CasADi problem rather than two clamp() calls: the
    friction-circle constraint couples v and delta, so clamping each to its
    own rate limit independently can still land outside the combined grip
    limit. Solving them together cannot.
    """

    def __init__(self, traction: float, max_speed: float) -> None:
        self.traction = traction
        self.max_speed = max_speed
        self._delta_prev = 0.0
        self._v_prev = 0.0
        self._build()

    def _build(self) -> None:
        n = SMOOTH_HORIZON
        opti = casadi.Opti()
        v = opti.variable(n + 1)
        a = opti.variable(n)
        delta = opti.variable(n)

        v0 = opti.parameter()
        delta0 = opti.parameter()
        v_target = opti.parameter()
        delta_target = opti.parameter()

        a_max = self.traction * GRAVITY
        rate = MAX_STEERING_RATE * SMOOTH_DT
        opti.subject_to(v[0] == v0)
        cost = 0
        for k in range(n):
            opti.subject_to(v[k + 1] == v[k] + a[k] * SMOOTH_DT)
            opti.subject_to(opti.bounded(-a_max, a[k], a_max))
            opti.subject_to(opti.bounded(-MAX_STEERING_ANGLE, delta[k], MAX_STEERING_ANGLE))
            prev = delta0 if k == 0 else delta[k - 1]
            opti.subject_to(opti.bounded(-rate, delta[k] - prev, rate))
            # Friction circle: the joint constraint a clamp cannot see.
            lateral = v[k + 1] ** 2 * casadi.tan(delta[k]) / WHEELBASE
            opti.subject_to(lateral ** 2 <= a_max ** 2)
            cost += (v[k + 1] - v_target) ** 2 + 4.0 * (delta[k] - delta_target) ** 2
            cost += 0.01 * a[k] ** 2 + 0.1 * (delta[k] - prev) ** 2
        opti.subject_to(opti.bounded(0.0, v, self.max_speed))
        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": 40,
                "ipopt.tol": 1e-3,
                "ipopt.acceptable_tol": 1e-2,
            },
        )
        self._opti = opti
        self._vars = (v, a, delta)
        self._params = (v0, delta0, v_target, delta_target)

    def reset(self) -> None:
        self._delta_prev = 0.0
        self._v_prev = 0.0

    def solve(self, v_meas: float, v_target: float, delta_target: float) -> tuple[float, float]:
        opti = self._opti
        v, a, delta = self._vars
        v0, delta0, v_target_p, delta_target_p = self._params
        opti.set_value(v0, v_meas)
        opti.set_value(delta0, self._delta_prev)
        opti.set_value(v_target_p, v_target)
        opti.set_value(delta_target_p, delta_target)
        opti.set_initial(v, np.full(SMOOTH_HORIZON + 1, v_meas))
        opti.set_initial(delta, np.full(SMOOTH_HORIZON, self._delta_prev))
        try:
            solution = opti.solve()
        except RuntimeError:
            # Fall back to independent clamps -- degraded, never wild. The
            # friction circle can be violated here in the rare case the
            # solver itself fails; that is the one thing the joint solve
            # buys that this fallback cannot.
            a_max = self.traction * GRAVITY
            rate = MAX_STEERING_RATE * SMOOTH_DT
            v_cmd = float(np.clip(v_target, self._v_prev - a_max * SMOOTH_DT, self._v_prev + a_max * SMOOTH_DT))
            d_cmd = float(np.clip(delta_target, self._delta_prev - rate, self._delta_prev + rate))
            self._v_prev, self._delta_prev = v_cmd, d_cmd
            return v_cmd, d_cmd
        # End of the horizon, not v[1]. /cmd_vel carries a speed SETPOINT that
        # the vehicle's own plant chases (sim_vehicle_node ramps toward it at
        # its measured 3.0 m/s^2 through a 0.19 s dead time); it is not a
        # trajectory the controller has to hand-feed one step at a time.
        # Returning v[1] capped every command at v_meas + a_max*SMOOTH_DT
        # (~0.7 m/s), so the setpoint could only ever climb as fast as the
        # plant had ALREADY climbed -- a ratchet that measured ~1.5 m/s live
        # while this same code reached 7.7 in the self-test, where the
        # "plant" is instantaneous. The whole horizon is still friction-circle
        # and rate feasible, so asking for its endpoint asks for nothing the
        # tires cannot do; the drivetrain clamps the rest by itself.
        v_cmd = float(solution.value(v[SMOOTH_HORIZON]))
        d_cmd = float(solution.value(delta[0]))
        self._v_prev, self._delta_prev = v_cmd, d_cmd
        return v_cmd, d_cmd


class _KinematicCar:
    """Bicycle-model stand-in for Gazebo, used only by --self-test."""

    def __init__(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw, self.v = x, y, yaw, 0.0

    def step(self, v_cmd: float, delta_cmd: float, dt: float) -> None:
        self.v = v_cmd  # ideal drivetrain; the point of this test is the
        # steering/speed law's geometry, not vehicle plant fidelity
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WHEELBASE * math.tan(delta_cmd) * dt


def run_self_test(traction: float, max_speed: float, duration_s: float, debug: bool) -> None:
    """No ROS, no Gazebo: drives a kinematic car around the real bale
    geometry using this module's own control law, the way lap_env_selftest.py
    tests lap_env's bookkeeping without a vehicle. Checks it does not hit a
    bale and makes real progress -- not a substitute for a live run (the
    drivetrain, sensor noise and control-loop timing are all idealised here),
    but it is the fastest way to catch a broken control law before spending a
    Gazebo session on it.
    """
    bales = bale_geometry.parse_bales(str(DEFAULT_SDF))
    spawn = bale_geometry.parse_vehicle_spawn(str(DEFAULT_SDF))
    car = _KinematicCar(*spawn)
    smoother = SpeedSteerSmoother(traction, max_speed)
    car_half_width = bale_geometry.CHASSIS_WIDTH / 2.0

    half_fov = math.radians(SCAN_FOV_DEG) / 2.0
    offsets = np.linspace(-half_fov, half_fov, SCAN_BINS)
    extents = footprint_extent(offsets)

    dt = 1.0 / 20.0
    steps = int(duration_s / dt)
    bearing = 0.0
    travel_heading = spawn[2]
    x0, y0 = spawn[0], spawn[1]
    left_start = False
    lap_started = 0.0
    laps = []
    min_clearance = math.inf
    distance = 0.0

    for step in range(steps):
        now = step * dt
        ranges = np.maximum(
            bale_geometry.lidar_scan(
                bales, car.x, car.y, car.yaw, SCAN_BINS, SCAN_FOV_DEG, SCAN_MAX_RANGE
            )
            - extents,
            0.0,
        )
        travel_heading = _wrap_angle(
            travel_heading + TRAVEL_TRACK_GAIN * _wrap_angle(car.yaw - travel_heading)
        )
        bearing, delta_target, v_target_value, gap_range, braking_range = plan_command(
            ranges, offsets, bearing, car.v, car_half_width, traction, max_speed,
            car.yaw, travel_heading,
        )
        v_cmd, delta_cmd = smoother.solve(car.v, v_target_value, delta_target)

        prev_x, prev_y = car.x, car.y
        car.step(v_cmd, delta_cmd, dt)
        distance += math.hypot(car.x - prev_x, car.y - prev_y)

        if bale_geometry.check_collision(bales, car.x, car.y, car.yaw):
            print(f"COLLISION at t={now:.1f}s, ({car.x:.2f},{car.y:.2f}) -- self-test FAILED")
            return
        clearance = bale_geometry.body_clearance(bales, car.x, car.y, car.yaw)
        min_clearance = min(min_clearance, clearance)

        dist_to_start = math.hypot(car.x - x0, car.y - y0)
        if not left_start and dist_to_start > 3.0:
            left_start = True
        elif left_start and dist_to_start < 1.5:
            laps.append(now - lap_started)
            lap_started = now
            left_start = False
            print(f"lap {len(laps)}: {laps[-1]:.2f} s")

        if debug and step % 20 == 0:
            print(
                f"t={now:5.1f} pos=({car.x:6.2f},{car.y:6.2f}) v={car.v:4.2f} "
                f"v_tgt={v_target_value:4.2f} delta={delta_cmd:+.2f} bearing={bearing:+.2f} "
                f"gap={gap_range:.2f} brake={braking_range:.2f} clearance={clearance:.2f}"
            )

    print(
        f"\n{duration_s:.0f}s self-test: {distance:.1f} m travelled, "
        f"{len(laps)} laps, min clearance {min_clearance:.2f} m, no collision"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-name", default="cfr_speed_course")
    parser.add_argument("--pose-topic", default=None)
    parser.add_argument(
        "--pose-msg",
        choices=["pose", "tf", "odom"],
        default="pose",
        help="pose: /zed/zed_node/pose (geometry_msgs/PoseStamped), the "
        "unambiguous single-vehicle topic simulation.launch.py's gazebo_bridge "
        "always bridges (default -- see the module-level note on why 'tf' is "
        "unsafe). tf: tf2_msgs/TFMessage on the world's dynamic_pose/info, "
        "the ground-truth bridge validate.sh's _glue role sets up -- kept for "
        "compatibility, NOT recommended, see below. odom: nav_msgs/Odometry, "
        "e.g. a QuestNav republish on real hardware.",
    )
    parser.add_argument(
        "--traction",
        type=float,
        default=0.9,
        help="grip fraction (of g) the friction-circle speed cap plans "
        "against. 0.9 puts the full-lock hairpin speed at ~2.6 m/s "
        "(v = sqrt(traction*g/kappa_full), kappa_full = tan(0.40)/0.324), "
        "still under the sim's own modelled tire grip (vehicle.yaml "
        "lateral.mu_lateral=1.0) -- unlike course_path.py's offline profile, "
        "this is evaluated fresh against the ACTUAL commanded curvature "
        "every tick, so there is no static plan for a wrong grip guess to "
        "silently outlive.",
    )
    parser.add_argument("--max-speed", type=float, default=5.0)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="no ROS/Gazebo; drive a kinematic stand-in")
    parser.add_argument("--self-test-seconds", type=float, default=90.0)
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args.traction, args.max_speed, args.self_test_seconds, args.debug)
        return

    # Imported here, not at module scope, so --self-test needs neither ROS
    # nor a running Gazebo.
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from tf2_msgs.msg import TFMessage

    if args.pose_topic is None:
        args.pose_topic = {
            "pose": "/zed/zed_node/pose",
            "tf": f"/world/{args.world_name}/dynamic_pose/info",
            "odom": "/zed/zed_node/odom",
        }[args.pose_msg]

    class MedleyRacer(Node):
        def __init__(self) -> None:
            super().__init__("medley_racer")
            self._bales = bale_geometry.parse_bales(str(DEFAULT_SDF))
            self._smoother = SpeedSteerSmoother(args.traction, args.max_speed)
            self._lock = threading.Lock()
            self._pose = None
            self._bearing = 0.0
            self._v_meas = 0.0
            self._prev_pose_time: tuple[float, float, float] | None = None
            self._start_xy: tuple[float, float] | None = None
            self._left_start = False
            self._lap_started = time.monotonic()
            self._lap_count = 0
            self._pose_history: list[tuple[float, float, float]] = []
            self._travel_heading: float | None = None

            half_fov = math.radians(SCAN_FOV_DEG) / 2.0
            self._offsets = np.linspace(-half_fov, half_fov, SCAN_BINS)
            self._extents = footprint_extent(self._offsets)
            self._car_half_width = bale_geometry.CHASSIS_WIDTH / 2.0

            if args.pose_msg == "pose":
                self.create_subscription(PoseStamped, args.pose_topic, self._on_pose_stamped, 10)
            elif args.pose_msg == "tf":
                self.create_subscription(TFMessage, args.pose_topic, self._on_tf, 10)
            else:
                self.create_subscription(Odometry, args.pose_topic, self._on_odom, 10)
            self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.create_timer(1.0 / args.control_hz, self._on_tick)
            self.get_logger().info(
                f"medley racer: follow-the-gap + pure pursuit + CasADi, no offline "
                f"plan; waiting for pose on {args.pose_topic}"
            )

        def _update_pose(self, x: float, y: float, yaw: float) -> None:
            # Speed is measured here, off the pose message's own arrival, not
            # in _on_tick off the control timer: the timer fires at a fixed
            # 20 Hz regardless of whether a new pose has actually shown up
            # since the last tick, and this topic does not publish in lockstep
            # with it. A tick that reads the same (x, y) as the tick before it
            # (because no new message had arrived yet) computed a spurious
            # raw=0 and dragged the exponential average down -- measured
            # live, this landed v_meas at roughly half of v_cmd, STEADY, not
            # narrowing, for 8+ seconds, which a real acceleration lag would
            # not do. Keying the difference to whenever a message actually
            # arrives removes the spurious zeros instead of averaging them in.
            now = time.monotonic()
            with self._lock:
                self._pose = (x, y, yaw)
                if self._prev_pose_time is not None:
                    pt, px, py = self._prev_pose_time
                    dt = now - pt
                    if dt > 1e-3:
                        raw = min(math.hypot(x - px, y - py) / dt, 8.0)
                        self._v_meas = 0.6 * self._v_meas + 0.4 * raw
                self._prev_pose_time = (now, x, y)

        def _on_pose_stamped(self, msg: "PoseStamped") -> None:
            p = msg.pose
            self._update_pose(
                p.position.x,
                p.position.y,
                _yaw_from_quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w),
            )

        def _on_tf(self, msg: TFMessage) -> None:
            # NOT RECOMMENDED -- see --pose-msg's help. ros_gz's Pose_V ->
            # TFMessage conversion does not populate child_frame_id at all
            # (gazebosim/ros_gz#172, #410, the latter closed "not planned"),
            # so there is no reliable way to tell which transform in this
            # message is the vehicle. dynamic_pose/info also carries the
            # start signal's "arms" joint transform on the same topic, so
            # transforms[0] is a guess, not an identification -- this is
            # what produced "pulls backward and tries a U-turn at the start"
            # in a live run: reading the wrong entity's pose entirely.
            # Kept only for a pose source that genuinely has no better
            # option; --pose-msg pose does not have this problem, because
            # /zed/zed_node/pose only ever contains the vehicle.
            if not msg.transforms:
                return
            t = msg.transforms[0].transform
            q = t.rotation
            self._update_pose(t.translation.x, t.translation.y, _yaw_from_quaternion(q.x, q.y, q.z, q.w))

        def _on_odom(self, msg: Odometry) -> None:
            p = msg.pose.pose
            self._update_pose(
                p.position.x,
                p.position.y,
                _yaw_from_quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w),
            )

        def _check_boxed_in(self, now: float, x: float, y: float) -> bool:
            self._pose_history.append((now, x, y))
            while self._pose_history and now - self._pose_history[0][0] > 2.5:
                self._pose_history.pop(0)
            if len(self._pose_history) < 2 or now - self._pose_history[0][0] < 2.0:
                return False
            oldest = self._pose_history[0]
            return math.hypot(x - oldest[1], y - oldest[2]) < 0.12

        def _on_tick(self) -> None:
            with self._lock:
                pose = self._pose
            if pose is None:
                return
            x, y, yaw = pose
            now = time.monotonic()

            if self._start_xy is None:
                self._start_xy = (x, y)
            if self._travel_heading is None:
                self._travel_heading = yaw
            else:
                # Track the car's actual heading slowly, so real cornering
                # carries the locked direction along with it but a swing
                # toward the way it came outruns it and gets clamped.
                self._travel_heading = _wrap_angle(
                    self._travel_heading
                    + TRAVEL_TRACK_GAIN * _wrap_angle(yaw - self._travel_heading)
                )

            # Body-relative, not centre-relative -- see footprint_extent.
            ranges = np.maximum(
                bale_geometry.lidar_scan(
                    self._bales, x, y, yaw, SCAN_BINS, SCAN_FOV_DEG, SCAN_MAX_RANGE
                )
                - self._extents,
                0.0,
            )
            self._bearing, delta_target, v_target_value, gap_range, braking_range = plan_command(
                ranges, self._offsets, self._bearing, self._v_meas,
                self._car_half_width, args.traction, args.max_speed,
                yaw, self._travel_heading,
            )

            # No reversing: the speed law's CREEP_SPEED floor means the car is
            # always rolling, and rolling is the only state its steering has
            # authority in. Being stuck is still worth saying out loud, but
            # the answer is to keep driving out of it, not to back up.
            if self._check_boxed_in(now, x, y):
                self.get_logger().warning(
                    f"not making progress (braking_range={braking_range:.2f} m); "
                    f"steering out of it at creep speed"
                )
                self._pose_history.clear()

            v_cmd, delta_cmd = self._smoother.solve(self._v_meas, v_target_value, delta_target)

            twist = Twist()
            twist.linear.x = v_cmd
            if abs(v_cmd) > 1e-3:
                twist.angular.z = (v_cmd / WHEELBASE) * math.tan(delta_cmd)
            self._cmd_pub.publish(twist)

            x0, y0 = self._start_xy
            dist_to_start = math.hypot(x - x0, y - y0)
            if not self._left_start and dist_to_start > 3.0:
                self._left_start = True
            elif self._left_start and dist_to_start < 1.5:
                self._lap_count += 1
                lap_time = now - self._lap_started
                self.get_logger().info(f"LAP {self._lap_count}: {lap_time:.2f} s")
                self._lap_started = now
                self._left_start = False

            if args.debug and now - getattr(self, "_last_debug", 0.0) > 0.5:
                self._last_debug = now
                print(
                    f"DBG t={now:.1f} pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):+.1f}deg "
                    f"v_cmd={v_cmd:.2f} v_tgt={v_target_value:.2f} v_meas={self._v_meas:.2f} "
                    f"delta={delta_cmd:+.2f} bearing={self._bearing:+.2f} "
                    f"gap={gap_range:.2f} brake={braking_range:.2f}",
                    flush=True,
                )

    # "odom" is the real-hardware pose source (e.g. a QuestNav republish) --
    # there is no Gazebo world to unpause there. Both sim-backed sources
    # ("pose", the default, and "tf") need it.
    if args.pose_msg != "odom" and not _unpause_world(args.world_name):
        raise SystemExit(f"could not start world '{args.world_name}' running")

    rclpy.init()
    node = MedleyRacer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._cmd_pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
