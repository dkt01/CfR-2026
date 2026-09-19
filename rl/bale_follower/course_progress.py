"""Arc-length progress along the planned course centerline.

The reward needs to pay for getting around the lap, not for moving. Those
differ: displacement projected on the car's own heading (what reward.py used
to pay) is earned just as well by circling in a wide section or by driving
the loop backwards. This projects the car onto `course_path.json` -- the
closed centerline course_path.py already plans -- and reports how far along
that loop it has come.

This is a privileged signal: the real car has no map. It is legitimate in
the *reward*, which is computed by the simulator and never reaches the
policy, and it must not become an observation.

The projection is windowed around the previous match on purpose. The course
is a serpentine whose adjacent passes run antiparallel about a metre apart,
so a global nearest-point search snaps between corridors -- and a snap to
the neighbouring pass reads as tens of metres of instant progress or loss.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent / "course_path.json"


class CourseProgress:
    """Stateful pose -> arc-length tracker. One instance per driving car."""

    def __init__(
        self,
        path_file: str | Path = DEFAULT_PATH,
        search_window_m: float = 8.0,
    ) -> None:
        with open(path_file) as handle:
            data = json.load(handle)
        self.x = np.asarray(data["x"], dtype=float)
        self.y = np.asarray(data["y"], dtype=float)
        self.s = np.asarray(data["s"], dtype=float)
        self.closed = bool(data["closed"])
        self.length = float(data["length_m"])
        self.search_window_m = search_window_m

        following = np.roll(np.arange(len(self.x)), -1)
        self.heading = np.arctan2(
            self.y[following] - self.y, self.x[following] - self.x
        )
        self._spacing = self.length / max(1, len(self.x))

        self._index = 0
        self._s = 0.0
        self.travelled = 0.0

    def _wrap(self, delta: float) -> float:
        """Shortest signed arc between two s values on a closed loop."""
        if not self.closed:
            return delta
        half = self.length / 2.0
        return (delta + half) % self.length - half

    def _segment_projection(self, i: int, j: int, x: float, y: float):
        ax, ay = self.x[i], self.y[i]
        dx, dy = self.x[j] - ax, self.y[j] - ay
        denominator = dx * dx + dy * dy
        if denominator < 1e-12:
            t = 0.0
        else:
            t = ((x - ax) * dx + (y - ay) * dy) / denominator
            t = min(1.0, max(0.0, t))
        lateral = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
        return self.s[i] + t * self._wrap(self.s[j] - self.s[i]), lateral

    def _refine(self, index: int, x: float, y: float):
        """Interpolate onto the better of the two segments meeting at `index`.

        Samples sit ~10 cm apart and a step covers up to ~55 cm, so taking the
        nearest sample's own s would quantize every delta by about a fifth.
        """
        count = len(self.x)
        best = None
        for i, j in ((index - 1) % count, index), (index, (index + 1) % count):
            candidate = self._segment_projection(i, j, x, y)
            if best is None or candidate[1] < best[1]:
                best = candidate
        return best[0] % self.length, best[1]

    def reset(self, x: float, y: float, yaw: float) -> float:
        """Re-acquire the loop from scratch after a teleport.

        Heading-gated: the serpentine's neighbouring passes run antiparallel,
        so nearest-point alone can lock onto the corridor heading the other
        way and then report the whole lap as negative progress.
        """
        squared = (self.x - x) ** 2 + (self.y - y) ** 2
        forward = np.cos(self.heading - yaw) > 0.0
        if forward.any():
            squared = np.where(forward, squared, np.inf)
        self._index = int(np.argmin(squared))
        self._s, _ = self._refine(self._index, x, y)
        self.travelled = 0.0
        return self._s

    def update(self, x: float, y: float):
        """Advance the tracker. Returns (s, delta_s, lateral_error)."""
        span = max(1, int(self.search_window_m / self._spacing))
        indices = (self._index + np.arange(-span, span + 1)) % len(self.x)
        nearest = int(
            indices[np.argmin((self.x[indices] - x) ** 2 + (self.y[indices] - y) ** 2)]
        )
        s, lateral = self._refine(nearest, x, y)
        delta = self._wrap(s - self._s)
        self._index = nearest
        self._s = s
        self.travelled += delta
        return s, delta, lateral

    @property
    def laps(self) -> float:
        return self.travelled / self.length
