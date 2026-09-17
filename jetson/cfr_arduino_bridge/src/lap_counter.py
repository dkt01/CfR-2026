"""Counting laps of the start/finish line, free of ROS.

The car waits behind the line, on the lane centerline, pointed down the lane --
0.70 m behind it on both courses (`generate_speed_course.py`'s
`LINE_AHEAD_OF_CAR`, and the obstacle course's `VEHICLE_START`).  So the line
needs no map and no survey: latch the pose the car held at the start, express
every later pose relative to it, and the line is the plane `x = line_offset`.

Which leaves the question of what stops the back straight counting.  On the
speed course the car leaves the line along +x, runs to the far end, turns back,
and returns down the other side of the oval -- crossing the plane of the line a
second time, 14 m off to the side, travelling the other way.

The gate is heading, not distance.  At the line the car travels the way the run
started; on the back straight it travels the opposite way.  That is true of any
start/finish line on any closed course -- it is what a start/finish line means
-- whereas "within 3 m of where we started" is a claim about how wide this
particular course happens to be, and the course built on the day will not match
the one in the simulator.  So nothing here knows the course's shape, length or
layout.  It knows how far ahead of the parked car the line is, and that a lap
is longer than `min_lap_distance`.

Loop closure shows up as a pose discontinuity -- a step no ground vehicle can
drive.  It is counted and reported, and kept out of the travelled distance so
it cannot push a lap over the line, but it does not re-seed anything: a closure
moves the estimate towards truth, and the latched reference lives in the same
corrected frame.  Being carried under e-stop is the opposite case, and the
caller signals that by dropping `counting`: that really is the car moving, so
the baseline is re-seeded, and if it was set down somewhere else the distance
it had driven is thrown away with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Mirrors LapCount.msg.  lap_counter_node asserts these against the generated
# message at import time, so the two cannot drift apart unnoticed.
STATE_IDLE = 0
STATE_ARMED = 1
STATE_RUNNING = 2
STATE_DONE = 3

# Why a plane crossing did or did not score.
COUNTED = "counted"
DEPARTURE = "departure"  # the outbound crossing, which arms rather than scores
REJECT_HEADING = "heading"
REJECT_DISTANCE = "distance"
REJECT_LATERAL = "lateral"
REJECT_DISABLED = "disabled"


# ----------------------------------------------------------------------- pose


@dataclass(frozen=True)
class Pose2D:
    """Planar pose.  Mirrors the C++ Pose2D in path_geometry.hpp."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


def yaw_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    """Extract yaw, assuming roll and pitch are small.

    A Python mirror of YawFromQuaternion in path_geometry.cpp; the C++ one
    cannot be reached from here, so test_lap_counter reuses the C++ test's
    cases to keep the two honest.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_relative_to(pose: Pose2D, reference: Pose2D) -> Pose2D:
    """Express `pose` in a frame whose origin and heading are `reference`."""
    dx = pose.x - reference.x
    dy = pose.y - reference.y
    cos_yaw = math.cos(reference.yaw)
    sin_yaw = math.sin(reference.yaw)
    return Pose2D(
        x=dx * cos_yaw + dy * sin_yaw,
        y=-dx * sin_yaw + dy * cos_yaw,
        yaw=wrap_to_pi(pose.yaw - reference.yaw),
    )


# ------------------------------------------------------------------- geometry


@dataclass
class Geometry:
    """What the counter assumes about the course.  Deliberately very little."""

    # How far ahead of the parked car the line is.  A setup fact, measurable
    # with a tape at the venue; both generated courses use 0.70 m.
    line_offset: float = 0.70
    # How far off the starting heading the car may be and still be taken to be
    # crossing the line rather than passing its plane elsewhere.  60 degrees is
    # loose enough for a car still unwinding out of the last corner and far
    # from the 180 degrees the return leg sits at.
    heading_tolerance: float = 1.05
    # A lap is at least this long.  Both courses are about ten times this; the
    # gate is here only so a car stopped astride the line cannot score twice.
    min_lap_distance: float = 10.0
    # Optional backstop on cross-track distance, for a course that runs a
    # second leg parallel to the start straight in the same direction.  Zero
    # disables it, which is the default: a distance gate is exactly the kind of
    # course-shape assumption this counter is trying not to make.
    lateral_gate: float = 0.0
    # A step longer than this between consecutive samples is a loop closure or
    # a tracking reset, not driving.  At 8 m/s and 30 Hz a real step is 0.27 m.
    max_step: float = 1.0
    # How far the car may be found from where it was e-stopped before the
    # distance it had travelled is thrown away.  Stopping in place and
    # carrying on keeps its lap; being moved starts the lap distance again, so
    # a car set down behind the line cannot score on the way back over it
    # using distance it drove before the e-stop.
    carry_tolerance: float = 0.5

    def __post_init__(self) -> None:
        if self.heading_tolerance <= 0.0 or self.heading_tolerance > math.pi:
            raise ValueError("heading_tolerance must be in (0, pi]")
        if self.min_lap_distance < 0.0:
            raise ValueError("min_lap_distance must not be negative")
        if self.max_step <= 0.0:
            raise ValueError("max_step must be positive")
        if self.lateral_gate < 0.0:
            raise ValueError("lateral_gate must not be negative; 0 disables it")
        if self.carry_tolerance < 0.0:
            raise ValueError("carry_tolerance must not be negative")


@dataclass(frozen=True)
class Crossing:
    """One pass through the plane of the line, and what became of it."""

    verdict: str
    along_track: float
    cross_track: float
    heading_error: float
    travelled: float

    @property
    def counted(self) -> bool:
        return self.verdict == COUNTED


@dataclass(frozen=True)
class Observation:
    """The tracker's view after one pose sample."""

    state: int
    laps: int
    done: bool
    along_track: float
    cross_track: float
    heading_error: float
    lap_distance: float
    rejected: int
    loop_closures: int
    # The crossing this sample produced, if it produced one.
    crossing: Crossing | None = None
    # Magnitude of the discontinuity this sample showed, 0.0 if it was driving.
    loop_closure: float = 0.0
    # How far the car was moved while counting was suspended, 0.0 if it was
    # not moved or not suspended.
    carried: float = 0.0


# -------------------------------------------------------------------- tracker


class LapTracker:
    """Counts returns to the start/finish line.

    Feed it poses in any consistent frame -- the map frame, in practice, so
    that loop closure keeps the latched reference meaningful over a long run.
    """

    def __init__(self, target_laps: int, geometry: Geometry | None = None) -> None:
        if target_laps < 1:
            raise ValueError("target_laps must be at least 1")
        self.target_laps = target_laps
        self.geometry = geometry if geometry is not None else Geometry()
        self.reset()

    def reset(self) -> None:
        """Forget the run: no reference, no laps, no latch."""
        self.reference: Pose2D | None = None
        self.previous: Pose2D | None = None
        self.laps = 0
        self.done = False
        self.state = STATE_IDLE
        self.lap_distance = 0.0
        self.rejected = 0
        self.loop_closures = 0
        self.suspended_at: Pose2D | None = None

    def arm(self, pose: Pose2D) -> None:
        """Latch `pose` as the start, and with it the position of the line."""
        self.reference = pose
        self.previous = None
        self.state = STATE_ARMED
        self.lap_distance = 0.0
        self.suspended_at = None

    def resume(self) -> None:
        """Drop the motion baseline, so the next sample re-seeds it.

        Called when counting resumes after an e-stop.  The car may have been
        carried, and that displacement must not read as driving.
        """
        self.previous = None

    @property
    def armed(self) -> bool:
        return self.reference is not None

    def update(self, pose: Pose2D, counting: bool = True) -> Observation:
        """Fold in one pose sample."""
        if self.reference is None:
            return self._observe(Pose2D(), None, 0.0)

        local = pose_relative_to(pose, self.reference)

        # Not racing: keep reporting position, but accumulate nothing and drop
        # the baseline so whatever happens under e-stop cannot look like a lap.
        if not counting:
            crossing = None
            if self.previous is not None and self._crosses(self.previous, local):
                crossing = self._crossing(REJECT_DISABLED, local)
                self.rejected += 1
            if self.suspended_at is None:
                self.suspended_at = local
            self.previous = None
            return self._observe(local, crossing, 0.0)

        if self.previous is None:
            carried = self._resume_from(local)
            self.previous = local
            return self._observe(local, None, 0.0, carried)

        step = math.hypot(local.x - self.previous.x, local.y - self.previous.y)
        jump = 0.0
        if step > self.geometry.max_step:
            # A discontinuity: loop closure, or tracking picking itself back
            # up.  Real, but not distance the car drove.
            jump = step
            self.loop_closures += 1
        else:
            self.lap_distance += step

        crossing = None
        if self._crosses(self.previous, local):
            crossing = self._resolve(local)

        self.previous = local
        return self._observe(local, crossing, jump)

    # ------------------------------------------------------------- internals

    def _resume_from(self, local: Pose2D) -> float:
        """Come back under autonomy, and decide what the lap is worth.

        Stopping in place and carrying on keeps the distance already driven.
        Being set down somewhere else does not: the rules allow the car to be
        lifted past an obstacle, and a car put down behind the line would
        otherwise score on the way back over it using distance it drove before
        the e-stop -- a lap it never completed.
        """
        if self.suspended_at is None:
            return 0.0
        moved = math.hypot(local.x - self.suspended_at.x, local.y - self.suspended_at.y)
        self.suspended_at = None
        if moved <= self.geometry.carry_tolerance:
            return 0.0
        self.lap_distance = 0.0
        return moved

    def _crosses(self, previous: Pose2D, local: Pose2D) -> bool:
        """True when the car passed through the plane of the line, forward.

        A backward pass is not a crossing at all, which is what makes a car
        that rolls back over the line harmless.
        """
        offset = self.geometry.line_offset
        return previous.x < offset <= local.x

    def _resolve(self, local: Pose2D) -> Crossing:
        """Decide what a forward plane crossing is worth."""
        heading_error = local.yaw
        if abs(heading_error) > self.geometry.heading_tolerance:
            return self._reject(REJECT_HEADING, local)

        # The outbound crossing: the car leaving the line it was parked behind.
        # It starts the run rather than completing a lap, and it is not a
        # rejection either -- nothing went wrong.
        if self.state == STATE_ARMED:
            self.state = STATE_RUNNING
            self.lap_distance = 0.0
            return self._crossing(DEPARTURE, local)

        gate = self.geometry.lateral_gate
        if gate > 0.0 and abs(local.y) > gate:
            return self._reject(REJECT_LATERAL, local)

        if self.lap_distance < self.geometry.min_lap_distance:
            return self._reject(REJECT_DISTANCE, local)

        crossing = self._crossing(COUNTED, local)
        self.laps += 1
        self.lap_distance = 0.0
        if self.laps >= self.target_laps:
            self.done = True
            self.state = STATE_DONE
        return crossing

    def _reject(self, verdict: str, local: Pose2D) -> Crossing:
        self.rejected += 1
        return self._crossing(verdict, local)

    def _crossing(self, verdict: str, local: Pose2D) -> Crossing:
        return Crossing(
            verdict=verdict,
            along_track=local.x,
            cross_track=local.y,
            heading_error=local.yaw,
            travelled=self.lap_distance,
        )

    def _observe(
        self,
        local: Pose2D,
        crossing: Crossing | None,
        jump: float,
        carried: float = 0.0,
    ) -> Observation:
        return Observation(
            state=self.state,
            laps=self.laps,
            done=self.done,
            along_track=local.x,
            cross_track=local.y,
            heading_error=local.yaw,
            lap_distance=self.lap_distance,
            rejected=self.rejected,
            loop_closures=self.loop_closures,
            crossing=crossing,
            loop_closure=jump,
            carried=carried,
        )
