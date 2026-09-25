#!/usr/bin/env python3
"""The reward, and the checks that it pays for the right episodes.

Everything here is PRIVILEGED -- centerline arc length, ground-truth contact
and clearance, which hoops the car threaded -- and none of it reaches the
observation.  The terms follow rl/formulaOne/reward.py, which is what got a
numpy-trained policy round the Speed Course:

    progress    +k per meter of centerline arc length gained
    time        -k per second
    hoop        +k for each hoop threaded, in order, the right way
    finish      +k for crossing the line with all three hoops threaded,
                plus a pace bonus below the target lap time
    crash       terminal; the chassis touched an obstacle, or rolled
    hoop miss   terminal; passing a hoop outside its posts fails the run
    stall       terminal; stopped for stall_s
    off course  terminal; left the lane where the course is open
    graze       per second, (bite into the clearance band)^2
    steer rate  change in steering command, squared, per second
    steer jerk  change in that change, squared, per second

There is no centerline-offset term.  The buckets are redrawn per layout and
the line through their section is a progress coordinate, not a path; paying
the car to stay near it would pay it to hit buckets.

    python3 reward.py    # per-step prices and the EPISODE-level invariants

Episode level matters: a per-step ordering (fast > slow > stopped) can hold
while standing still or crashing early beats trying over a whole episode --
that is how an earlier obstacle-course run converged on standing still.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
TERMS = (
    "progress",
    "time",
    "hoop",
    "finish",
    "crash",
    "hoop_miss",
    "stall",
    "off_course",
    "graze",
    "steer_rate",
    "steer_jerk",
)


def step_reward(
    cfg,
    ds,
    dt,
    hoops_now,
    finished,
    lap_time,
    crashed,
    missed,
    stalled,
    off_course,
    clearance,
    dsteer,
    ddsteer,
):
    """Per-car reward for one control step, and the per-term breakdown."""
    r = cfg["reward"]
    band = float(r["graze_band"])
    bite = np.clip((band - clearance) / band, 0.0, 1.0)
    pace = np.maximum(0.0, float(r["target_lap_s"]) - lap_time)
    terms = {
        "progress": float(r["progress"]) * ds,
        "time": -float(r["time"]) * dt * np.ones_like(ds),
        "hoop": float(r["hoop"]) * hoops_now,
        "finish": np.where(
            finished, float(r["finish"]) + float(r["finish_pace"]) * pace, 0.0
        ),
        "crash": np.where(crashed, float(r["crash"]), 0.0),
        "hoop_miss": np.where(missed, float(r["hoop_miss"]), 0.0),
        "stall": np.where(stalled, float(r["stall"]), 0.0),
        "off_course": np.where(off_course, float(r["off_course"]), 0.0),
        "graze": -float(r["graze"]) * bite**2 * dt,
        "steer_rate": -float(r["steer_rate"]) * dsteer**2 / dt,
        "steer_jerk": -float(r["steer_jerk"]) * ddsteer**2 / dt,
    }
    total = sum(terms.values())
    return total, terms


def assert_reachable(cfg):
    """Every threshold has to sit inside the range its signal can take.

    Clearance is measured in rings 0.02 m apart out to CLEARANCE_RINGS[-1];
    a graze band beyond the last ring could never register.
    """
    import env as env_module

    band = float(cfg["reward"]["graze_band"])
    rings = env_module.CLEARANCE_RINGS
    assert rings[0] < band <= rings[-1] + 1e-9, (
        f"graze_band {band} m is outside the clearance rings {rings[0]}..{rings[-1]}"
    )
    assert float(cfg["env"]["stall_speed"]) > 0.0
    assert float(cfg["env"]["stall_speed"]) < float(cfg["env"]["v_cap"])


def episode_return(cfg, speed, meters, outcome, hoops, lap_length=74.0, grazing=0.0):
    """Return of an idealized episode: `meters` at constant `speed`, then outcome."""
    r = cfg["reward"]
    dt = 1.0 / float(cfg["env"]["control_hz"])
    seconds = meters / speed if speed > 0 else float(cfg["env"]["stall_s"])
    total = float(r["progress"]) * meters - float(r["time"]) * seconds
    total += float(r["hoop"]) * hoops
    total -= float(r["graze"]) * grazing**2 * seconds
    if outcome == "finish":
        total += float(r["finish"]) + float(r["finish_pace"]) * max(
            0.0, float(r["target_lap_s"]) - seconds
        )
    elif outcome in ("crash", "hoop_miss", "stall", "off_course"):
        total += float(r[outcome])
    del dt, lap_length
    return total


def assert_episode_incentives(cfg, lap=74.0):
    """What a whole episode pays must rank the strategies the right way."""
    E = lambda *a, **k: episode_return(cfg, *a, **k)  # noqa: E731
    fast = E(2.5, lap, "finish", 3)
    slow = E(1.0, lap, "finish", 3)
    grazing = E(2.5, lap, "finish", 3, grazing=0.8)
    crash_late = E(2.5, lap - 5, "crash", 3)
    miss_last = E(2.5, lap - 8, "hoop_miss", 2)
    stand = E(0.0, 0.0, "stall", 0)
    crash_early = E(2.0, 2.0, "crash", 0)
    try_10m = E(2.0, 10.0, "crash", 0)
    checks = {
        "fast lap > slow lap": fast > slow,
        "slow lap > crash near the end": slow > crash_late,
        "slow lap > missing the last hoop": slow > miss_last,
        "clean lap > grazing lap": fast > grazing,
        "driving 10 m then crashing > standing still": try_10m > stand,
        "driving 10 m then crashing > crashing at once": try_10m > crash_early,
        "standing still < 0": stand < 0,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError("reward incentives inverted: " + "; ".join(failed))
    return dict(
        fast=fast,
        slow=slow,
        grazing=grazing,
        crash_late=crash_late,
        miss_last=miss_last,
        stand=stand,
        crash_early=crash_early,
        try_10m=try_10m,
    )


def main() -> int:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    assert_reachable(cfg)
    table = assert_episode_incentives(cfg)
    print("episode returns (idealized):")
    for name, value in table.items():
        print(f"  {name:12s} {value:8.1f}")
    dt = 1.0 / float(cfg["env"]["control_hz"])
    print("per-step prices at 20 Hz:")
    for label, v, clear, dsteer in [
        ("3.0 m/s clean", 3.0, 1.0, 0.0),
        ("1.0 m/s clean", 1.0, 1.0, 0.0),
        ("stopped", 0.0, 1.0, 0.0),
        ("2.5 m/s, 4 cm off a wall", 2.5, 0.04, 0.0),
        ("2.5 m/s sawing the wheel", 2.5, 1.0, 0.6),
    ]:
        one = np.array([1.0])
        total, terms = step_reward(
            cfg,
            v * dt * one,
            dt,
            0 * one,
            one < 0,
            99 * one,
            one < 0,
            one < 0,
            one < 0,
            one < 0,
            clear * one,
            dsteer * one,
            2 * dsteer * one,
        )
        print(f"  {label:28s} {total[0]:+7.3f}")
    print("incentives ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
