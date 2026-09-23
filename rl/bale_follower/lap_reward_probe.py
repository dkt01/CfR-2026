#!/usr/bin/env python3
"""What the lap reward pays, term by term, at states the car really visits.

The lap-time objective is a difference of two large numbers -- progress and
clock -- so a coefficient that looks harmless can flip the sign of a whole
regime (make hairpins unprofitable, make crashing cheaper than recovering,
make brushing a bale worth it at speed). This prints every term at ten
representative states, converts the per-step numbers into per-second and
per-lap ones, and asserts the orderings that must hold.

    python3 lap_reward_probe.py                        # config_lap.yaml
    python3 lap_reward_probe.py --set k_time=8 --set k_touch=60

No ROS, no Gazebo, no simulator: run it before every training run.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

from lap_reward import LapRewardConfig, compute_lap_reward

CONFIG = Path(__file__).resolve().parent / "config_lap.yaml"


def scenarios(config: LapRewardConfig, dt: float, top_speed: float):
    """(label, kwargs) pairs in physical units at the configured control rate."""
    clear = 0.32  # a centred car in a 0.95 m corridor
    hairpin_speed = 1.9  # the plan's slowest point
    states = [
        (
            "racing, clear, top speed",
            dict(
                delta_s=top_speed * dt,
                dt=dt,
                speed=top_speed,
                clearance=clear,
                steer_rate_fraction=0.0,
                steer_fraction=0.1,
                prev_steer_fraction=0.1,
                collided=False,
            ),
        ),
        (
            "hairpin, 1.9 m/s, held lock",
            dict(
                delta_s=hairpin_speed * dt,
                dt=dt,
                speed=hairpin_speed,
                clearance=0.20,
                steer_rate_fraction=0.05,
                steer_fraction=0.95,
                prev_steer_fraction=0.95,
                collided=False,
            ),
        ),
        (
            "turn-in, full servo rate",
            dict(
                delta_s=3.0 * dt,
                dt=dt,
                speed=3.0,
                clearance=0.28,
                steer_rate_fraction=1.0,
                steer_fraction=0.5,
                prev_steer_fraction=0.06,
                collided=False,
            ),
        ),
        (
            "sawing at top speed",
            dict(
                delta_s=top_speed * dt,
                dt=dt,
                speed=top_speed,
                clearance=clear,
                steer_rate_fraction=1.0,
                steer_fraction=0.4,
                prev_steer_fraction=-0.4,
                collided=False,
            ),
        ),
        (
            "drifting off the line",
            dict(
                delta_s=top_speed * dt,
                dt=dt,
                speed=top_speed,
                clearance=clear,
                steer_rate_fraction=0.0,
                steer_fraction=0.1,
                prev_steer_fraction=0.1,
                collided=False,
                lateral=0.30,
                prev_lateral=0.20,
                heading_error=0.25,
                prev_heading_error=0.20,
            ),
        ),
        (
            "coming back to the line",
            dict(
                delta_s=top_speed * dt,
                dt=dt,
                speed=top_speed,
                clearance=clear,
                steer_rate_fraction=0.0,
                steer_fraction=0.1,
                prev_steer_fraction=0.1,
                collided=False,
                lateral=0.20,
                prev_lateral=0.30,
                heading_error=0.20,
                prev_heading_error=0.25,
            ),
        ),
        (
            "crawling, clear",
            dict(
                delta_s=0.5 * dt,
                dt=dt,
                speed=0.5,
                clearance=clear,
                steer_rate_fraction=0.0,
                steer_fraction=0.0,
                prev_steer_fraction=0.0,
                collided=False,
            ),
        ),
        (
            "stopped, clear",
            dict(
                delta_s=0.0,
                dt=dt,
                speed=0.0,
                clearance=clear,
                steer_rate_fraction=0.0,
                steer_fraction=0.0,
                prev_steer_fraction=0.0,
                collided=False,
            ),
        ),
        (
            "top speed, brushing a bale",
            dict(
                delta_s=top_speed * dt,
                dt=dt,
                speed=top_speed,
                clearance=0.05,
                steer_rate_fraction=0.0,
                steer_fraction=0.1,
                prev_steer_fraction=0.1,
                collided=False,
            ),
        ),
        (
            "top speed, 0.15 m gap",
            dict(
                delta_s=top_speed * dt,
                dt=dt,
                speed=top_speed,
                clearance=0.15,
                steer_rate_fraction=0.0,
                steer_fraction=0.1,
                prev_steer_fraction=0.1,
                collided=False,
            ),
        ),
        (
            "reversing, NOT stuck",
            dict(
                delta_s=-1.2 * dt,
                dt=dt,
                speed=-1.2,
                clearance=clear,
                steer_rate_fraction=0.0,
                steer_fraction=0.0,
                prev_steer_fraction=0.0,
                collided=False,
            ),
        ),
        (
            "rocking in place (the exploit)",
            dict(
                delta_s=0.0,
                dt=dt,
                speed=1.0,
                clearance=clear,
                steer_rate_fraction=0.3,
                prev_steer_rate_fraction=-0.3,
                steer_fraction=0.1,
                prev_steer_fraction=-0.1,
                collided=False,
                recovering=True,
                heading_error=0.4,
                prev_heading_error=0.4,
            ),
        ),
        (
            "wedged, sitting still",
            dict(
                delta_s=0.0,
                dt=dt,
                speed=0.0,
                clearance=0.12,
                steer_rate_fraction=0.0,
                steer_fraction=1.0,
                prev_steer_fraction=1.0,
                collided=False,
                recovering=True,
                heading_error=1.2,
                prev_heading_error=1.2,
            ),
        ),
        (
            "wedged, nose against a bale",
            dict(
                delta_s=0.0,
                dt=dt,
                speed=0.0,
                clearance=0.0,
                steer_rate_fraction=0.0,
                steer_fraction=1.0,
                prev_steer_fraction=1.0,
                collided=False,
                recovering=True,
                heading_error=1.2,
                prev_heading_error=1.2,
            ),
        ),
        (
            "wedged, reversing and turning out",
            dict(
                delta_s=-1.2 * dt,
                dt=dt,
                speed=-1.2,
                clearance=0.18,
                steer_rate_fraction=0.2,
                steer_fraction=1.0,
                prev_steer_fraction=1.0,
                collided=False,
                recovering=True,
                heading_error=1.125,
                prev_heading_error=1.2,
            ),
        ),
        (
            "collision",
            dict(
                delta_s=0.3 * dt,
                dt=dt,
                speed=0.3,
                clearance=0.0,
                steer_rate_fraction=0.5,
                steer_fraction=0.8,
                prev_steer_fraction=0.3,
                collided=True,
            ),
        ),
    ]
    # Unless a state says otherwise it is steady: the rate command is the
    # same as it was last step, so the jerk term charges nothing.
    for _, kwargs in states:
        kwargs.setdefault("prev_steer_rate_fraction", kwargs["steer_rate_fraction"])
    # Chatter: same average |rate| as the smooth turn-in above, opposite sign
    # every step. Only the jerk term separates the two.
    states.insert(
        4,
        (
            "chattering, top speed",
            dict(
                delta_s=top_speed * dt,
                dt=dt,
                speed=top_speed,
                clearance=clear,
                steer_rate_fraction=0.5,
                prev_steer_rate_fraction=-0.5,
                steer_fraction=0.05,
                prev_steer_fraction=-0.02,
                collided=False,
            ),
        ),
    )
    states.insert(
        4,
        (
            "smooth turn-in, same mean rate",
            dict(
                delta_s=top_speed * dt,
                dt=dt,
                speed=top_speed,
                clearance=clear,
                steer_rate_fraction=0.5,
                prev_steer_rate_fraction=0.5,
                steer_fraction=0.3,
                prev_steer_fraction=0.25,
                collided=False,
            ),
        ),
    )
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="override one coefficient",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    values = dict(raw["reward"])
    for override in args.set:
        key, _, value = override.partition("=")
        values[key] = float(value)
    config = LapRewardConfig(**values)

    dt = 1.0 / float(raw["env"]["control_hz"])
    top = float(raw["env"]["max_speed"])
    loop = 110.1

    columns = [
        "progress",
        "time",
        "stall",
        "clearance",
        "touch",
        "steering",
        "align",
        "shaping",
        "lap_bonus",
    ]
    print(
        f"control {1 / dt:.0f} Hz (dt {dt * 1000:.0f} ms), "
        f"max_speed {top} m/s, loop {loop} m"
    )
    print(
        f"break-even speed k_time/k_progress = "
        f"{config.k_time / config.k_progress:.2f} m/s\n"
    )
    header = (
        f"{'state':<32}"
        + "".join(f"{c[:8]:>9}" for c in columns)
        + f"{'TOTAL':>9}{'per s':>9}"
    )
    print(header)
    print("-" * len(header))
    results = {}
    for label, kwargs in scenarios(config, dt, top):
        result = compute_lap_reward(config, **kwargs)
        results[label] = result
        row = f"{label:<32}"
        for column in columns:
            value = getattr(result, column)
            row += f"{value:>9.2f}" if abs(value) > 1e-9 else f"{'.':>9}"
        print(row + f"{result.total:>9.2f}{result.total / dt:>9.1f}")

    print("\nlap bonus vs lap time")
    for lap_time in (28.4, 30.0, 33.0, 37.0, 42.0, 50.0):
        bonus = compute_lap_reward(
            config,
            delta_s=0.0,
            dt=dt,
            speed=3.0,
            clearance=0.3,
            steer_rate_fraction=0.0,
            prev_steer_rate_fraction=0.0,
            steer_fraction=0.0,
            prev_steer_fraction=0.0,
            collided=False,
            lap_time=lap_time,
        ).lap_bonus
        dense = config.k_progress * loop - config.k_time * lap_time
        print(
            f"  {lap_time:>5.1f} s lap: dense {dense:>7.1f} + bonus "
            f"{bonus:>6.1f} = {dense + bonus:>7.1f}"
        )

    print("\nper-episode weights (150 s, 20 Hz, 3000 steps)")
    steps = int(150 / dt)
    print(f"  clock alone, whole episode      {-config.k_time * 150:>8.1f}")
    print(
        f"  wedged for the full stuck window"
        f"{-(config.k_time + config.k_stall) * float(raw['env']['stuck_window_s']):>8.1f}"
    )
    print(f"  one collision                   {-config.collision_penalty:>8.1f}")
    print(
        f"  3 clean 33 s laps               "
        f"{3 * (config.k_progress * loop - config.k_time * 33) + 3 * (config.k_lap_base + config.k_lap_pace * max(0.0, config.target_lap_time - 33)):>8.1f}"
    )
    del steps

    # ---------------------------------------------------------- invariants
    racing = results["racing, clear, top speed"].total
    hairpin = results["hairpin, 1.9 m/s, held lock"].total
    sawing = results["sawing at top speed"].total
    crawling = results["crawling, clear"].total
    stopped = results["stopped, clear"].total
    brushing = results["top speed, brushing a bale"].total
    near = results["top speed, 0.15 m gap"].total
    reversing = results["reversing, NOT stuck"].total
    wedged_still = results["wedged, sitting still"].total
    wedged_out = results["wedged, reversing and turning out"].total
    collision = results["collision"].total

    assert racing > crawling > stopped, (racing, crawling, stopped)
    assert stopped < 0.0, "a stopped car must lose ground every step"
    assert hairpin > 0.0, (
        "a hairpin at the speed the plan asks for must still pay, or the "
        "policy learns to avoid the part of the course it cannot avoid"
    )
    assert config.k_time / config.k_progress < 1.4, (
        "break-even speed must sit below the ~1.4 m/s the vehicle needs to "
        "rotate at full lock, or hairpins are not worth driving"
    )
    assert sawing < racing, "sawing must cost against holding a line"
    assert (
        results["chattering, top speed"].total
        < results["smooth turn-in, same mean rate"].total
    ), (
        "chatter and a smooth turn-in have the same mean steering rate; only "
        "the jerk term separates them, so this is the invariant that keeps "
        "k_steer_jerk honest"
    )
    assert brushing < 0.0 and brushing < near, (brushing, near)
    assert near < racing, "clearing a bale must beat skimming it"
    assert reversing < 0.0, "reverse is a recovery move, not a strategy"
    rocking = results["rocking in place (the exploit)"].total
    assert rocking < crawling, (
        "rocking in place must cost at least as much as crawling: a car that "
        "keeps its speed up while going nowhere was the first run's policy"
    )
    assert wedged_out > rocking, "backing out must beat rocking in place"
    assert wedged_out > 0.0, (
        "a purposeful reverse-and-turn out of a wedge must pay, not merely "
        "cost less than the alternatives"
    )
    assert wedged_out > wedged_still, (
        "backing out and turning down the course must beat sitting in the wedge"
    )
    assert collision < min(racing, hairpin, stopped, wedged_still, brushing)
    # The escape-by-crashing check. The worst stream the car can sit in is a
    # wedge with its nose on a bale; the episode logic lets that run for
    # `stuck_window_s` before truncating, and truncation (unlike a collision)
    # bootstraps, so a collision is the only way to make the stream stop.
    window = float(raw["env"]["stuck_window_s"])
    control_hz = float(raw["env"]["control_hz"])
    worst_rate = (config.k_time + config.k_stall) + control_hz * (
        config.recovery_contact_scale
        * (config.k_clearance + config.k_touch * config.touch_clearance)
    )
    print(
        f"\nworst wedged stream {worst_rate:.1f}/s over the {window:.0f} s "
        f"stuck window = {worst_rate * window:.0f}, "
        f"collision_penalty = {config.collision_penalty:.0f}"
    )
    assert config.collision_penalty > worst_rate * window, (
        f"a wedged car loses {worst_rate * window:.0f} waiting out the stuck "
        f"window but only {config.collision_penalty:.0f} by driving into the "
        "bale -- the policy will learn to crash its way out"
    )
    fast = compute_lap_reward(
        config,
        delta_s=0.0,
        dt=dt,
        speed=3.0,
        clearance=0.3,
        steer_rate_fraction=0.0,
        prev_steer_rate_fraction=0.0,
        steer_fraction=0.0,
        prev_steer_fraction=0.0,
        collided=False,
        lap_time=30.0,
    )
    slow = compute_lap_reward(
        config,
        delta_s=0.0,
        dt=dt,
        speed=3.0,
        clearance=0.3,
        steer_rate_fraction=0.0,
        prev_steer_rate_fraction=0.0,
        steer_fraction=0.0,
        prev_steer_fraction=0.0,
        collided=False,
        lap_time=40.0,
    )
    # Shaping must be directional...
    assert (
        results["coming back to the line"].total
        > results["drifting off the line"].total
    ), "shaping must pay for closing on the line and charge for leaving it"
    # ...and it must telescope to zero round a closed path, or it is not
    # potential-based and it IS changing the optimal policy.
    import random

    rng = random.Random(0)
    state = (0.0, 0.0)
    loop_total = 0.0
    for _ in range(200):
        nxt = (rng.uniform(-0.8, 0.8), rng.uniform(-1.0, 1.0))
        loop_total += compute_lap_reward(
            config,
            delta_s=0.0,
            dt=dt,
            speed=2.0,
            clearance=0.32,
            steer_rate_fraction=0.0,
            prev_steer_rate_fraction=0.0,
            steer_fraction=0.0,
            prev_steer_fraction=0.0,
            collided=False,
            lateral=nxt[0],
            prev_lateral=state[0],
            heading_error=nxt[1],
            prev_heading_error=state[1],
        ).shaping
        state = nxt
    loop_total += compute_lap_reward(
        config,
        delta_s=0.0,
        dt=dt,
        speed=2.0,
        clearance=0.32,
        steer_rate_fraction=0.0,
        prev_steer_rate_fraction=0.0,
        steer_fraction=0.0,
        prev_steer_fraction=0.0,
        collided=False,
        lateral=0.0,
        prev_lateral=state[0],
        heading_error=0.0,
        prev_heading_error=state[1],
    ).shaping
    # It does not sum to exactly zero and should not be expected to: with
    # gamma < 1 the residual is (gamma-1) * sum(Phi), which for 201 steps at
    # gamma 0.998 is a fraction of a reward unit. What matters is that it is
    # negligible against what driving the same 201 steps pays -- otherwise
    # the shaping is a bias on the objective rather than a hint about it.
    reference = config.k_progress * 2.0 * dt * 201
    print(
        f"shaping round a 201-step closed walk: {loop_total:+.3f}, "
        f"against {reference:.0f} of progress over the same steps "
        f"({abs(loop_total) / reference:.1%})"
    )
    assert abs(loop_total) < 0.05 * reference, (
        f"shaping summed to {loop_total:+.3f} round a closed path, "
        f"{abs(loop_total) / reference:.1%} of the progress paid over it -- "
        "that is a bias on the objective, not a hint about it"
    )

    assert fast.lap_bonus > slow.lap_bonus, "a faster lap must pay more"
    assert math.isclose(
        fast.lap_bonus - slow.lap_bonus, config.k_lap_pace * 10.0, rel_tol=1e-6
    )
    print("\nall invariants hold")


if __name__ == "__main__":
    main()
