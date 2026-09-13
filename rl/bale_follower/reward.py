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


@dataclass
class RewardResult:
    total: float
    progress: float
    proximity: float
    smoothness: float
    centering: float
    collided: bool


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
) -> RewardResult:
    progress = config.k_progress * progress_distance
    proximity = -config.k_proximity * max(0.0, config.safe_clearance - min_clearance)
    smoothness = -config.k_smooth * abs(angular_z - prev_angular_z)
    centering = -config.k_center * center_error
    total = progress + proximity + smoothness + centering
    if collided:
        total -= config.collision_penalty
    return RewardResult(
        total=total,
        progress=progress,
        proximity=proximity,
        smoothness=smoothness,
        centering=centering,
        collided=collided,
    )


if __name__ == "__main__":
    cfg = RewardConfig()
    good = compute_reward(cfg, progress_distance=0.4, min_clearance=0.45, angular_z=0.1, prev_angular_z=0.08, collided=False, center_error=0.0)
    hugging = compute_reward(cfg, progress_distance=0.4, min_clearance=0.45, angular_z=0.1, prev_angular_z=0.08, collided=False, center_error=0.35)
    scraping = compute_reward(cfg, progress_distance=0.4, min_clearance=0.05, angular_z=0.1, prev_angular_z=0.08, collided=False, center_error=0.4)
    crashed = compute_reward(cfg, progress_distance=0.05, min_clearance=0.0, angular_z=0.4, prev_angular_z=-0.4, collided=True, center_error=0.5)
    stalled = compute_reward(cfg, progress_distance=0.0, min_clearance=0.5, angular_z=0.0, prev_angular_z=0.0, collided=False, center_error=0.0)

    print(f"clean centered step: {good.total:+.3f}")
    print(f"fast, hugging wall:  {hugging.total:+.3f}")
    print(f"fast but scraping:   {scraping.total:+.3f}")
    print(f"stalled in open:     {stalled.total:+.3f}")
    print(f"collision:           {crashed.total:+.3f}")

    assert good.total > hugging.total, "centered should beat wall-hugging"
    assert hugging.total > scraping.total, "clearance should be preferred"
    assert scraping.total > stalled.total, "progress should beat sitting still"
    assert stalled.total > crashed.total, "anything should beat crashing"
    print("\nordering check passed: centered > hugging > scraping > stalled > crashed")
