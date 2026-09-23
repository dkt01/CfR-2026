"""Where the car is around the loop, and how fast it got there.

The v1-v11 policies measured progress as displacement projected onto the
car's own heading (`reward.forward_progress`). That is the right signal when
there is no centerline -- it was written before `course_path.py` existed --
but it cannot express a lap: it pays the same for a metre driven the wrong
way round the loop as for a metre driven the right way, it cannot tell that
the car has come back to where it started, and it has no clock.

This module projects the true pose onto the planned loop from
`course_path.json` (1102 samples, ~10 cm apart, closed, 110.1 m) and turns
that into the three things a lap-time objective needs:

  * `delta_s`  -- signed arc length gained along the loop this step. Negative
                  when the car reverses or drives backwards up the course, so
                  the progress reward cannot be farmed by shuffling.
  * `laps`     -- cumulative arc length crossing multiples of the loop
                  length. Equivalent to `lap_counter.py`'s start/finish plane
                  (a lap is one full circuit, heading-gated by construction
                  because arc length is signed) but valid from any start
                  pose, which matters because training spawns all over the
                  loop rather than behind the line.
  * `heading_error` -- angle between the car's heading and the direction of
                  travel along the loop. This is what "turn back into the
                  travel direction" is measured against when the car has
                  wedged and has to reverse out.

Everything here is PRIVILEGED: it comes from the true pose and the offline
plan, and it is used only by the reward, the episode logic and the metrics --
never by the observation. The policy still sees nothing but its forward scan
and its own proprioception, so it stays deployable without a map.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent / "course_path.json"


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Projection:
    index: int  # nearest path sample
    s: float  # arc length at that sample, metres from the plan's origin
    lateral: float  # signed offset from the line, +left
    heading_error: float  # wrapped(yaw - path tangent), rad
    curvature: float  # 1/m of the plan at this point
    reference_speed: float  # m/s the min-time plan asks for here


class LapTrack:
    def __init__(self, path_json: str | Path = DEFAULT_PATH) -> None:
        with open(path_json) as handle:
            plan = json.load(handle)
        self.x = np.asarray(plan["x"], dtype=float)
        self.y = np.asarray(plan["y"], dtype=float)
        self.curvature = np.asarray(plan["curvature"], dtype=float)
        self.reference_speed = np.asarray(plan["v"], dtype=float)
        self.closed = bool(plan.get("closed", True))
        self.reference_lap_time = float(plan.get("estimated_time_s", 0.0))
        self.reversed = False
        self._rebuild()

    def _rebuild(self) -> None:
        """Arc length and tangents from the points themselves.

        Not from the plan's own `s` array, so that a reversed loop is
        described exactly the way a forward one is.
        """
        self._points = np.column_stack([self.x, self.y])
        steps = np.linalg.norm(np.roll(self._points, -1, axis=0) - self._points, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(steps[:-1])])
        self.length = float(steps.sum()) if self.closed else float(self.s[-1])
        self.spacing = float(np.median(steps))
        # Tangent from the neighbouring points. `np.gradient` on a closed loop
        # would make the seam's tangent point across the loop, so the ends
        # wrap explicitly.
        dx = np.roll(self.x, -1) - np.roll(self.x, 1)
        dy = np.roll(self.y, -1) - np.roll(self.y, 1)
        self.tangent = np.arctan2(dy, dx)

    def orient_to(self, x: float, y: float, yaw: float) -> bool:
        """Point the loop the way the car is facing. Returns True if flipped.

        `course_path.py` walks the corridor skeleton in whichever direction
        the graph search happened to take, so the plan's stored direction is
        arbitrary -- `path_racer.py` flips it at runtime for the same reason.
        Get this wrong and every sign in the reward inverts: driving the
        course as intended scores as driving backwards up it.
        """
        index = self.nearest_index(x, y)
        tangent = float(self.tangent[index])
        if math.cos(yaw - tangent) >= 0.0:
            return False
        self.x = self.x[::-1].copy()
        self.y = self.y[::-1].copy()
        # Curvature and the reference speed are indexed by the same samples,
        # so they reverse with them; curvature also changes sign, because
        # left and right swap when the direction of travel does.
        self.curvature = -self.curvature[::-1].copy()
        self.reference_speed = self.reference_speed[::-1].copy()
        self.reversed = not self.reversed
        self._rebuild()
        return True

    # ------------------------------------------------------------ projection

    def nearest_index(
        self, x: float, y: float, hint: int | None = None, window: int = 80
    ) -> int:
        """Index of the closest path sample.

        `hint` restricts the search to +/-`window` samples (8 m at the plan's
        10 cm spacing) around the last known index. Without it a car running
        the back straight of an oval projects onto the parallel straight
        14 m away -- which is the same failure mode `lap_counter.py` gates
        with heading, seen from the other side.
        """
        if hint is None:
            deltas = self._points - np.array([x, y])
            return int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
        count = len(self.x)
        offsets = (np.arange(hint - window, hint + window + 1)) % count
        deltas = self._points[offsets] - np.array([x, y])
        return int(offsets[np.argmin(np.einsum("ij,ij->i", deltas, deltas))])

    def project(
        self, x: float, y: float, yaw: float, hint: int | None = None
    ) -> Projection:
        index = self.nearest_index(x, y, hint)
        tangent = float(self.tangent[index])
        dx, dy = x - self.x[index], y - self.y[index]
        # Along-track residual refines s between samples; cross-track is the
        # signed lateral offset (+left of the direction of travel).
        along = dx * math.cos(tangent) + dy * math.sin(tangent)
        lateral = -dx * math.sin(tangent) + dy * math.cos(tangent)
        return Projection(
            index=index,
            s=float(self.s[index]) + along,
            lateral=lateral,
            heading_error=_wrap_to_pi(yaw - tangent),
            curvature=float(self.curvature[index]),
            reference_speed=float(self.reference_speed[index]),
        )

    def delta_s(self, previous_s: float, current_s: float) -> float:
        """Signed arc length gained, with the seam wrapped.

        Any single step is far shorter than half the loop (0.55 m at 20 Hz
        and 11 m/s against 55 m), so the shorter of the two ways round is
        always the one the car actually drove.
        """
        delta = current_s - previous_s
        if not self.closed:
            return delta
        half = self.length / 2.0
        if delta > half:
            delta -= self.length
        elif delta < -half:
            delta += self.length
        return delta

    # --------------------------------------------------------------- spawning

    def pose_at(
        self, index: int, lateral: float = 0.0, heading_offset: float = 0.0
    ) -> tuple[float, float, float]:
        """A world pose `lateral` metres left of the line at `index`."""
        tangent = float(self.tangent[index])
        x = self.x[index] - lateral * math.sin(tangent)
        y = self.y[index] + lateral * math.cos(tangent)
        return x, y, _wrap_to_pi(tangent + heading_offset)


class LapCounter:
    """Cumulative arc length -> laps and lap times.

    Counts a lap every `track.length` metres of NET forward progress, so a
    car that reverses 5 m has to re-drive them. Lap time is wall-of-sim time
    between crossings, which is the number the objective is written against.
    """

    def __init__(self, track: LapTrack) -> None:
        self.track = track
        self.reset(0.0)

    def reset(self, s: float) -> None:
        self.previous_s = s
        self.travelled = 0.0  # net signed arc length since reset
        self.laps = 0
        self.lap_times: list[float] = []
        self._last_lap_stamp = 0.0
        self.best_lap_time = math.inf

    def update(self, s: float, elapsed: float) -> tuple[float, float | None]:
        """Advance with the current arc length and episode clock.

        Returns (delta_s, lap_time) where lap_time is not None exactly on the
        step a lap completes.
        """
        delta = self.track.delta_s(self.previous_s, s)
        self.previous_s = s
        self.travelled += delta
        lap_time = None
        # 1 mm of tolerance: the plan's own arc length and its sample sum
        # differ in the last few digits, so an exact `>=` can leave a car that
        # drove a perfect circuit one float short of its lap.
        if self.travelled >= (self.laps + 1) * self.track.length - 1e-3:
            self.laps += 1
            lap_time = elapsed - self._last_lap_stamp
            self._last_lap_stamp = elapsed
            self.lap_times.append(lap_time)
            self.best_lap_time = min(self.best_lap_time, lap_time)
        return delta, lap_time


if __name__ == "__main__":
    track = LapTrack()
    print(
        f"loop {track.length:.1f} m, {len(track.x)} samples "
        f"@ {track.spacing * 100:.0f} cm, plan {track.reference_lap_time:.1f} s"
    )
    print(
        f"reference speed {track.reference_speed.min():.2f}"
        f"-{track.reference_speed.max():.2f} m/s, "
        f"tightest radius {1.0 / np.abs(track.curvature).max():.2f} m"
    )

    # Drive the plan itself and check the machinery closes a lap in the
    # planned time to within the sampling resolution.
    counter = LapCounter(track)
    projection = track.project(track.x[0], track.y[0], float(track.tangent[0]))
    counter.reset(projection.s)
    elapsed, laps = 0.0, []
    for lap in range(2):
        for i in range(1, len(track.x) + 1):
            index = i % len(track.x)
            pose = track.pose_at(index)
            step = track.project(*pose, hint=index)
            elapsed += float(track.spacing / track.reference_speed[index])
            _, lap_time = counter.update(step.s, elapsed)
            if lap_time is not None:
                laps.append(lap_time)
    print(f"two laps driven at the plan's own speed: {[round(t, 2) for t in laps]} s")
    assert counter.laps == 2, counter.laps
    assert abs(laps[0] - track.reference_lap_time) < 1.0, laps

    # Lateral and heading errors are zero on the line, and signed off it.
    on_line = track.project(*track.pose_at(300), hint=300)
    left = track.project(*track.pose_at(300, lateral=0.25), hint=300)
    turned = track.project(*track.pose_at(300, heading_offset=0.5), hint=300)
    assert abs(on_line.lateral) < 1e-6 and abs(on_line.heading_error) < 1e-6
    assert abs(left.lateral - 0.25) < 1e-6, left.lateral
    assert abs(turned.heading_error - 0.5) < 1e-6, turned.heading_error

    # Reversing scores negative, and the seam does not spike.
    forward = track.delta_s(track.s[-1], track.s[0])
    backward = track.delta_s(track.s[0], track.s[-1])
    assert 0.0 < forward < 0.5 and -0.5 < backward < 0.0, (forward, backward)

    # Orientation: the plan's stored direction is arbitrary, so a car facing
    # the other way must flip it, and flipping must leave a consistent loop.
    fresh = LapTrack()
    head_on = fresh.pose_at(500)
    assert not fresh.orient_to(*head_on), "flipped a loop already facing right"
    backwards = fresh.pose_at(500, heading_offset=math.pi)
    assert fresh.orient_to(*backwards), "did not flip for a car facing the other way"
    assert abs(fresh.length - track.length) < 1e-6, (fresh.length, track.length)
    after = fresh.project(*backwards)
    assert abs(after.heading_error) < 1e-6, after.heading_error
    assert not fresh.orient_to(*backwards), "flipped twice for the same pose"
    print("projection, lap counting, seam wrap, sign and orientation checks passed")
