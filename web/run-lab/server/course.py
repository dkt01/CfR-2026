"""The Speed Course, loaded through rl/formulaOne/track.py.

The analyser must judge a run against the same centerline, rule cap and bale
distance field the driver used -- a second copy of the geometry that rounds a
corner differently would report clearances the car never had.  So this module
does not build anything itself; it imports the driver's own builder and its
cache.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
F1 = REPO / "rl" / "formulaOne"
if str(F1) not in sys.path:
    sys.path.insert(0, str(F1))

import track as track_mod  # noqa: E402


def load_config(run_dir: Path | None = None):
    """The run's own config when it shipped one, else the tree's.

    The run's copy may predate newer track keys (runs/v3.5 has no speed
    floors); missing keys are filled from the tree's config so the geometry
    still builds.  The track section decides geometry and it has not changed.
    """
    base = yaml.safe_load((F1 / "config.yaml").read_text())
    if run_dir is not None:
        shipped = run_dir / "policy" / "config.yaml"
        if shipped.exists():
            try:
                own = yaml.safe_load(shipped.read_text())
                for section, values in own.items():
                    if isinstance(values, dict):
                        base.setdefault(section, {}).update(values)
                    else:
                        base[section] = values
            except yaml.YAMLError:
                pass
    return base


@lru_cache(maxsize=4)
def _track_for(config_key: str):
    config = yaml.safe_load(config_key)
    return track_mod.build(config, REPO)


def get_track(config: dict):
    return _track_for(yaml.safe_dump({"track": config["track"]}, sort_keys=True))


def geometry(config: dict):
    """JSON-ready course description for the browser."""
    t = get_track(config)
    bales = track_mod.load_bales(REPO / config["track"]["world_sdf"])
    step = max(1, int(round(0.10 / t.ds)))
    lo, hi = float(config["track"]["v_hairpin"]), float(config["track"]["v_straight"])
    zone = np.where(t.hairpin, 0, np.where(t.v_cap >= hi - 1e-3, 2, 1))
    i = int(np.clip(np.searchsorted(t.s, t.start_station), 0, len(t.s) - 1))
    return {
        "length": float(t.length),
        "start": {
            "station": float(t.start_station),
            "x": float(t.x[i]),
            "y": float(t.y[i]),
            "yaw": float(np.arctan2(t.ty[i], t.tx[i])),
        },
        "centerline": {
            "s": t.s[::step].round(3).tolist(),
            "x": t.x[::step].round(3).tolist(),
            "y": t.y[::step].round(3).tolist(),
            "v_cap": t.v_cap[::step].round(2).tolist(),
            "zone": zone[::step].astype(int).tolist(),
            "half_left": t.half_left[::step].round(3).tolist(),
            "half_right": t.half_right[::step].round(3).tolist(),
        },
        "bales": [
            {
                "x": round(float(b[0]), 3),
                "y": round(float(b[1]), 3),
                "yaw": round(float(b[2]), 4),
            }
            for b in bales
        ],
        "bale_size": [track_mod.BALE_LENGTH, track_mod.BALE_WIDTH],
        "car": [float(config["vehicle"]["length"]), float(config["vehicle"]["width"])],
        "v_hairpin": lo,
        "v_straight": hi,
        "graze_band": 0.12,
    }


def zones(config: dict, piece: float = 10.0):
    """Contiguous course sections: (name, s0, s1, kind), for the section table.

    Kind follows the three colours the driver's RViz view already uses:
    hairpin (red, at v_hairpin), taper (yellow) and straight (green).
    """
    t = get_track(config)
    hi = float(config["track"]["v_straight"])
    kind = np.where(
        t.hairpin, "hairpin", np.where(t.v_cap >= hi - 1e-3, "straight", "taper")
    )
    out = []
    start = 0
    for k in range(1, len(kind) + 1):
        if k == len(kind) or kind[k] != kind[start]:
            out.append([str(kind[start]), float(t.s[start]), float(t.s[k - 1] + t.ds)])
            start = k
    # The loop wraps: merge the last section into the first if they match.
    if len(out) > 1 and out[0][0] == out[-1][0]:
        out[0][1] = out[-1][1] - t.length
        out.pop()
    # Drop slivers under half a metre into their neighbour; they are noise.
    merged = []
    for sec in out:
        if merged and sec[2] - sec[1] < 0.5:
            merged[-1][2] = sec[2]
        else:
            merged.append(sec)
    # A 48 m "straight" holds sweepers and a chicane; saying a problem was
    # "somewhere in Straight 1" locates nothing.  Split long zones into equal
    # pieces of at most `piece` metres.
    counters = {}
    named = []
    for kind_name, s0, s1 in merged:
        counters[kind_name] = counters.get(kind_name, 0) + 1
        pieces = max(1, int(np.ceil((s1 - s0) / piece)))
        edges = np.linspace(s0, s1, pieces + 1)
        for j in range(pieces):
            suffix = f"{chr(ord('a') + j)}" if pieces > 1 else ""
            named.append(
                {
                    "name": f"{kind_name.title()} {counters[kind_name]}{suffix}",
                    "kind": kind_name,
                    "s0": round(float(edges[j]), 2),
                    "s1": round(float(edges[j + 1]), 2),
                    # Sections that straddle the start have a negative s0.
                    "range": f"{edges[j] % t.length:.0f}-{edges[j + 1] % t.length:.0f} m",
                }
            )
    return named
