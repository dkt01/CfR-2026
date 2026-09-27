"""One way to draw a randomised quantity, used by the plant, world and camera.

Every range in config.yaml is [lo, hi, nominal].  `scale` shrinks the range
toward its nominal -- 0 is the nominal car, 1 the full range -- which is what
the training curriculum turns up, and `enabled=False` is the nominal car.
"""

from __future__ import annotations

import numpy as np


def draw(rng, spec, k, scale=1.0, enabled=True):
    lo, hi, nom = (float(v) for v in spec)
    if not enabled:
        return np.full(k, nom)
    return rng.uniform(nom + scale * (lo - nom), nom + scale * (hi - nom), k)
