"""Tracking whether the car threaded every hoop, free of ROS.

Same plane-crossing approach as lap_counter.py, applied per hoop instead of
per lap: express the car's pose in a frame centered on the hoop, and score
whichever sample steps across the plane through its center. A hoop is not a
start/finish line though -- the rules only ask that the car pass *through*
it, not that it do so travelling a particular way -- so a crossing counts in
either direction, unlike lap_counter's forward-only gate.

`yaw`, as generate_obstacle_course.py's HOOPS records it and
obstacle_course_layout.yaml carries it forward, is the SDF model's own pose
yaw: build_hoops() sets the two uprights at local x = +/-0.292 m, so that is
the axis yaw rotates into the world -- the gate's *width*, not the direction
of travel through it. A car drives through along the perpendicular, so the
crossing plane here is built from a frame rotated 90 degrees from the stored
yaw. This has not been checked against a real sim run of the course; if a
hoop the car visibly drove through is not recognized as passed, look here
first -- see test_hoop_monitor.py for the traces this was reasoned out
against.

A crossing only resolves a hoop if it happens near that hoop. The plane
through a hoop is infinite and the lane crosses two of them metres away
from the hoop itself, so without that gate driving the course correctly
marks hoops as missed -- see DEFAULT_ATTEMPT_HALF_WIDTH.

Missing even one hoop fails the whole run per the rules, so once a hoop is
marked missed it stays missed, and `any_missed` latches for the rest of the
run -- nothing here clears it short of `HoopMonitor.reset()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Half the clear opening between the uprights: build_hoops() places them at
# local x = +/-0.292 m with a 0.017 m radius, so the true gap between their
# inner edges is +/-(0.292 - 0.017) m. A car that crosses the gate's plane
# further out than this hit an upright, not the opening.
DEFAULT_GATE_HALF_WIDTH = 0.292 - 0.017

# How far either side of a hoop still counts as *attempting* it. The plane
# through a hoop is infinite, and the lane crosses two of them a long way
# from the hoop itself: the gravel-run-to-bank turn crosses hoop_0's plane
# 7.9 m south of hoop_0, and the car wash run back to the finish crosses
# hoop_2's plane 7.2 m east of hoop_2. Without this, driving the course
# correctly marks both as missed, and no clean lap is possible -- which is
# what a training run showed, as a 0.43 hoop-miss rate among cars that had
# never been near a hoop. Beyond this distance the car is not interacting
# with that hoop at all, so the crossing resolves nothing; inside it, the
# old pass/miss test applies unchanged. Generous next to the 0.275 m gate
# so that genuinely swerving around a hoop still counts as missing it.
DEFAULT_ATTEMPT_HALF_WIDTH = 1.5


@dataclass(frozen=True)
class Pose2D:
    """Planar pose. Mirrors lap_counter.Pose2D."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _pose_relative_to(pose: Pose2D, reference: Pose2D) -> Pose2D:
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


@dataclass
class Hoop:
    name: str
    x: float
    y: float
    yaw: float  # the SDF model's own pose yaw -- the uprights' axis, not travel
    half_width: float = DEFAULT_GATE_HALF_WIDTH
    attempt_half_width: float = DEFAULT_ATTEMPT_HALF_WIDTH

    def travel_frame(self) -> Pose2D:
        """Reference frame whose local +x is the direction of travel through the gate."""
        return Pose2D(self.x, self.y, wrap_to_pi(self.yaw + math.pi / 2.0))


@dataclass
class HoopState:
    passed: bool = False
    missed: bool = False
    _previous_local_x: float | None = field(default=None, repr=False)

    @property
    def resolved(self) -> bool:
        return self.passed or self.missed


class HoopMonitor:
    """Resolves every hoop on the course to passed or missed, once each."""

    def __init__(self, hoops: list[Hoop]) -> None:
        self.hoops: dict[str, Hoop] = {hoop.name: hoop for hoop in hoops}
        self.state: dict[str, HoopState] = {}
        self.reset()

    def reset(self) -> None:
        self.state = {name: HoopState() for name in self.hoops}

    @property
    def any_missed(self) -> bool:
        return any(state.missed for state in self.state.values())

    @property
    def all_passed(self) -> bool:
        return bool(self.state) and all(state.passed for state in self.state.values())

    def update_hoop(
        self,
        name: str,
        x: float,
        y: float,
        yaw: float | None = None,
        half_width: float | None = None,
    ) -> None:
        """(Re)place a hoop -- e.g. when the randomizer draws a new layout.

        `yaw` and `half_width` default to whatever the hoop already had: the
        randomizer only ever moves a hoop along its line, never turns it, so
        a position update alone (yaw=None) is the common case.
        """
        existing = self.hoops.get(name)
        self.hoops[name] = Hoop(
            name=name,
            x=x,
            y=y,
            yaw=yaw if yaw is not None else (existing.yaw if existing else 0.0),
            half_width=(
                half_width
                if half_width is not None
                else (existing.half_width if existing else DEFAULT_GATE_HALF_WIDTH)
            ),
        )
        self.state.setdefault(name, HoopState())

    def update(self, pose: Pose2D) -> dict[str, list[str]]:
        """Fold in one pose sample against every hoop not yet resolved.

        Returns the hoops (by name) this sample resolved, `{"passed": [...],
        "missed": [...]}`, so the caller can log or publish just the change
        rather than diffing the whole state every step.
        """
        newly_passed: list[str] = []
        newly_missed: list[str] = []
        for name, hoop in self.hoops.items():
            state = self.state[name]
            if state.resolved:
                continue
            local = _pose_relative_to(pose, hoop.travel_frame())
            previous_x = state._previous_local_x
            if previous_x is not None and (
                (previous_x < 0.0 <= local.x) or (local.x < 0.0 <= previous_x)
            ):
                if abs(local.y) <= hoop.half_width:
                    state.passed = True
                    newly_passed.append(name)
                elif abs(local.y) <= hoop.attempt_half_width:
                    state.missed = True
                    newly_missed.append(name)
                # Further out than that and the car is simply somewhere else
                # on the course that happens to lie on this hoop's infinite
                # plane -- see DEFAULT_ATTEMPT_HALF_WIDTH. Resolve nothing.
            state._previous_local_x = local.x
        return {"passed": newly_passed, "missed": newly_missed}
