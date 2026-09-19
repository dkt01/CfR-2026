"""Reward shaping for BaleFollowerEnv.

The goal is the fastest lap without touching a wall, so progress is measured
as arc length gained along the planned course centerline (course_progress.py)
rather than displacement in the direction the car happens to face. The
difference is the whole point: heading-projected displacement is earned just
as well by circling in a wide section or by running the loop backwards, and
an earlier version of this file paid exactly that.

Episodes are time-limited and the loop is closed, so total arc length gained
over an episode *is* average speed around the course -- maximizing it is
maximizing lap pace, with no separate per-step time cost to balance against
the collision penalty. That balance is worth avoiding: a per-step time cost
large enough to drive pace also makes an early crash the cheapest way to
stop the bleeding, which is a local optimum a policy finds long before it
finds driving.

There is deliberately no centering term. The fast line cuts corners and runs
wide on exit; paying the car to sit mid-corridor charges it for the racing
line. Wall avoidance is left to the clearance terms, which is what they are
for.

Coefficients live in a dataclass so they're easy to sweep from config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    # Per metre of centerline arc length gained. Negative arc (running the
    # loop backwards) is charged at the same rate, which is what stops a
    # reversing recovery from becoming a reward source.
    k_progress: float = 10.0
    k_proximity: float = 2.0
    k_smooth: float = 0.05
    collision_penalty: float = 200.0
    # Wall clearance below which the proximity penalty starts biting. The
    # corridors are ~0.95 m wide and the car is 0.30 m wide, so this has to
    # stay well under half a corridor width or every pose is penalized.
    safe_clearance: float = 0.35
    # Brushing a bale without the collision check firing. k_proximity is a
    # gentle slope from 0.35 m that a fast lap can profitably pay; this is the
    # steep part close in, so "touched it but kept going" stops being a
    # winning trade. Separate from collision_penalty because a touch should
    # not end the episode -- the car should learn to recover from it.
    k_touch: float = 120.0
    touch_clearance: float = 0.15
    # Steering oscillation. k_smooth charges for changing the steering angle;
    # this charges for REVERSING it, which is what the left-right sawing is.
    # Penalising magnitude alone does not separate a sustained corner (good)
    # from bang-bang chatter (bad) -- only the sign change does.
    k_steer_reversal: float = 0.8


@dataclass
class RewardResult:
    total: float
    progress: float
    proximity: float
    smoothness: float
    touch: float
    collided: bool
    touched: bool


def compute_reward(
    config: RewardConfig,
    arc_progress: float,
    min_clearance: float,
    angular_z: float,
    prev_angular_z: float,
    collided: bool,
    steer_fraction: float = 0.0,
    prev_steer_fraction: float = 0.0,
) -> RewardResult:
    progress = config.k_progress * arc_progress
    proximity = -config.k_proximity * max(0.0, config.safe_clearance - min_clearance)

    smoothness = -config.k_smooth * abs(angular_z - prev_angular_z)
    # Sign reversal, charged in proportion to how far the steering swung
    # through zero. Sustained lock costs nothing here; sawing costs on every
    # reversal, which is the behaviour to remove.
    if steer_fraction * prev_steer_fraction < 0.0:
        smoothness -= config.k_steer_reversal * min(
            abs(steer_fraction), abs(prev_steer_fraction)
        )

    touched = min_clearance < config.touch_clearance
    touch = -config.k_touch * max(0.0, config.touch_clearance - min_clearance)

    total = progress + proximity + smoothness + touch
    if collided:
        total -= config.collision_penalty
    return RewardResult(
        total=total,
        progress=progress,
        proximity=proximity,
        smoothness=smoothness,
        touch=touch,
        collided=collided,
        touched=touched,
    )


if __name__ == "__main__":
    cfg = RewardConfig()

    def step(**kwargs):
        base = {
            "arc_progress": 0.40,
            "min_clearance": 0.45,
            "angular_z": 0.1,
            "prev_angular_z": 0.08,
            "collided": False,
        }
        return compute_reward(cfg, **{**base, **kwargs})

    fast = step(arc_progress=0.55)
    slow = step(arc_progress=0.20)
    backwards = step(arc_progress=-0.20)
    circling = step(arc_progress=0.0)
    scraping = step(min_clearance=0.05)
    near_miss = step(min_clearance=0.15)
    sawing = step(steer_fraction=1.0, prev_steer_fraction=-1.0)
    crashed = step(arc_progress=0.05, min_clearance=0.0, collided=True)

    print(f"fast lap step (0.55 m):  {fast.total:+.3f}")
    print(f"slow step (0.20 m):      {slow.total:+.3f}")
    print(f"circling (0.00 m):       {circling.total:+.3f}")
    print(f"backwards (-0.20 m):     {backwards.total:+.3f}")
    print(f"same step, sawing:       {sawing.total:+.3f}")
    print(f"0.15 m clearance:        {near_miss.total:+.3f}")
    print(f"scraping at 0.05 m:      {scraping.total:+.3f}")
    print(f"collision:               {crashed.total:+.3f}")

    assert fast.total > slow.total, "faster progress along the lap should pay more"
    assert slow.total > circling.total, "moving along the lap should beat circling"
    assert circling.total > backwards.total, "running the loop backwards must cost"
    assert near_miss.total > scraping.total, "clearance should be preferred"
    assert fast.total > sawing.total, "steady steering should beat sawing"
    assert crashed.total < backwards.total, "anything should beat crashing"
    # A crash must never be an escape from a bad episode: every other term is
    # bounded well inside the collision penalty.
    assert crashed.total < -100.0, "collision has to dominate the shaping terms"
    print("\nordering: fast > slow > circling > backwards > crashed")
