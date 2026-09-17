"""Reward shaping for BaleFollowerEnv.

There is no course centerline to measure progress against. The bale order in
speed_course.sdf is DXF drawing order, not a traversal path -- consecutive
indices jump up to 26 m between disjoint wall segments -- and
jetson/scripts/generate_speed_course.py discards the source outline. So
progress is measured as distance actually travelled in the direction the car
was facing, and the bale walls themselves stop the car from cutting corners
or circling: the course is a corridor, so "go forward without touching a
wall" is the task.

Coefficients live in a dataclass so they're easy to sweep from config.yaml.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardConfig:
    k_progress: float = 10.0
    k_proximity: float = 1.0
    k_smooth: float = 0.05
    # Penalty per metre of imbalance between the nearest bale on the left and
    # on the right of the car -- zero when centered in the corridor. Uses both
    # walls, unlike k_proximity which only reacts to the nearer one.
    k_center: float = 2.0
    collision_penalty: float = 50.0
    # Wall clearance below which the proximity penalty starts biting. The
    # corridors are ~0.95 m wide and the car is 0.30 m wide, so this has to
    # stay well under half a corridor width or every pose is penalized.
    safe_clearance: float = 0.35
    # Brushing a bale without the collision check firing. k_proximity is a
    # gentle slope from 0.35 m that a fast lap can profitably pay; this is the
    # steep part close in, so "touched it but kept going" stops being a
    # winning trade. Separate from collision_penalty because a touch should
    # not end the episode -- the car should learn to recover from it.
    k_touch: float = 40.0
    touch_clearance: float = 0.10
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
    centering: float
    touch: float
    collided: bool
    touched: bool


def forward_progress(
    prev_x: float, prev_y: float, prev_yaw: float, x: float, y: float
) -> float:
    """Displacement projected onto the heading the car started the step with."""
    return (x - prev_x) * math.cos(prev_yaw) + (y - prev_y) * math.sin(prev_yaw)


def compute_reward(
    config: RewardConfig,
    progress_distance: float,
    min_clearance: float,
    angular_z: float,
    prev_angular_z: float,
    collided: bool,
    center_error: float = 0.0,
    steer_fraction: float = 0.0,
    prev_steer_fraction: float = 0.0,
) -> RewardResult:
    progress = config.k_progress * progress_distance
    proximity = -config.k_proximity * max(0.0, config.safe_clearance - min_clearance)
    centering = -config.k_center * center_error

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

    total = progress + proximity + smoothness + centering + touch
    if collided:
        total -= config.collision_penalty
    return RewardResult(
        total=total,
        progress=progress,
        proximity=proximity,
        smoothness=smoothness,
        centering=centering,
        touch=touch,
        collided=collided,
        touched=touched,
    )


if __name__ == "__main__":
    cfg = RewardConfig()
    good = compute_reward(cfg, progress_distance=0.4, min_clearance=0.45, angular_z=0.1, prev_angular_z=0.08, collided=False, center_error=0.0)
    hugging = compute_reward(cfg, progress_distance=0.4, min_clearance=0.45, angular_z=0.1, prev_angular_z=0.08, collided=False, center_error=0.35)
    scraping = compute_reward(cfg, progress_distance=0.4, min_clearance=0.05, angular_z=0.1, prev_angular_z=0.08, collided=False, center_error=0.4)
    crashed = compute_reward(cfg, progress_distance=0.05, min_clearance=0.0, angular_z=0.4, prev_angular_z=-0.4, collided=True, center_error=0.5)
    stalled = compute_reward(cfg, progress_distance=0.0, min_clearance=0.5, angular_z=0.0, prev_angular_z=0.0, collided=False, center_error=0.0)

    sawing = compute_reward(cfg, progress_distance=0.4, min_clearance=0.45, angular_z=0.1, prev_angular_z=0.08, collided=False, center_error=0.0, steer_fraction=1.0, prev_steer_fraction=-1.0)
    touching = compute_reward(cfg, progress_distance=0.4, min_clearance=0.04, angular_z=0.1, prev_angular_z=0.08, collided=False, center_error=0.1)
    near_miss = compute_reward(cfg, progress_distance=0.4, min_clearance=0.15, angular_z=0.1, prev_angular_z=0.08, collided=False, center_error=0.1)

    print(f"clean centered step: {good.total:+.3f}")
    print(f"same step, sawing:   {sawing.total:+.3f}")
    print(f"fast but touching:   {touching.total:+.3f}")
    print(f"same, 0.15 m clear:  {near_miss.total:+.3f}")
    print(f"fast, hugging wall:  {hugging.total:+.3f}")
    print(f"fast but scraping:   {scraping.total:+.3f}")
    print(f"stalled in open:     {stalled.total:+.3f}")
    print(f"collision:           {crashed.total:+.3f}")

    assert good.total > hugging.total, "centered should beat wall-hugging"
    assert hugging.total > scraping.total, "clearance should be preferred"
    assert scraping.total > stalled.total, "progress should beat sitting still"
    assert stalled.total > crashed.total, "anything should beat crashing"
    assert good.total > sawing.total, "steady steering should beat sawing"
    # Same pose and speed, differing only in whether the car brushes a bale.
    assert touching.total < near_miss.total, "touching a bale must cost more than clearing it"
    print("\nordering check passed: centered > hugging > scraping > stalled > crashed")
    print("sawing and bale-touching both rank below the clean step")
