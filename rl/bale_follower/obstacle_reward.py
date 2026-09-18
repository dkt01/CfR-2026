"""Reward shaping for ObstacleCourseEnv.

The goal stated for this course is the fastest possible lap, so every step
pays a flat time cost regardless of what the car does: minimizing total
reward lost over an episode is exactly minimizing the time spent driving it.
Everything else here is either a shaping term that makes that sparse time
cost learnable from scratch (progress, proximity) or a terminal penalty for
a rule violation severe enough to end the run (collision, a missed hoop).

Unlike BaleFollowerEnv's corridor, the Obstacle Course has no consistent
left/right wall to center between -- the car wash, bucket room, tunnel and
bank are different shapes -- so there is no centering term here. Clearance
and collision both come from the ZED's simulated point cloud
(env.scan_source == "cloud") rather than analytic course geometry: the
course's ramps, tunnel and helix are 3D in a way a top-down bale-style OBB
model cannot represent (see obstacle_env.py's module docstring), and the
point cloud is what the real car senses too.

A missed hoop fails the run outright per the rules, so it costs more than a
collision: a policy that would rather clip a bale than skip a hoop is
learning the right ordering. hoop_pass_bonus is paid once per hoop, on the
step hoop_monitor reports it newly passed, which -- because time keeps
draining reward every step regardless -- rewards passing it *and* rewards
passing it sooner.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObstacleRewardConfig:
    # Cost per second elapsed. The dominant term: minimizing cumulative
    # reward lost to this is minimizing total episode time, which is the
    # stated objective. Sized so a clean step's progress term roughly offsets
    # it at a competitive speed -- see the ordering check below.
    k_time: float = 20.0
    # Small shaping bonus for forward progress. Time cost alone is a valid
    # but sparse signal for "go fast"; this keeps early learning (a policy
    # that has not yet discovered driving forward at all) from looking flat.
    k_progress: float = 2.0
    k_proximity: float = 1.0
    safe_clearance: float = 0.35
    # Brushing something, short of a logged collision -- same steep-close-in
    # shape as BaleFollowerEnv's k_touch, just against the point cloud's
    # range instead of an analytic wall distance.
    k_touch: float = 80.0
    touch_clearance: float = 0.15
    # Point-cloud range below which a step counts as a genuine hit rather
    # than a close pass. There is no analytic ground truth to check this
    # against on this course (see obstacle_env.py), so this threshold is the
    # collision detector, not just a shaping cutoff -- tune it against actual
    # sensor noise before trusting it, not just against these defaults.
    collision_clearance: float = 0.06
    collision_penalty: float = 300.0
    # Heavier than a collision: missing a hoop fails the run outright by the
    # rules, a collision does not.
    hoop_miss_penalty: float = 500.0
    hoop_pass_bonus: float = 30.0
    k_smooth: float = 0.05
    k_steer_reversal: float = 0.8


@dataclass
class ObstacleRewardResult:
    total: float
    time_cost: float
    progress: float
    proximity: float
    touch: float
    smoothness: float
    hoop: float
    collided: bool
    touched: bool
    hoop_missed: bool
    hoops_passed: int


def compute_reward(
    config: ObstacleRewardConfig,
    dt: float,
    progress_distance: float,
    min_clearance: float,
    angular_z: float,
    prev_angular_z: float,
    collided: bool,
    hoop_missed: bool = False,
    hoops_passed_this_step: int = 0,
    steer_fraction: float = 0.0,
    prev_steer_fraction: float = 0.0,
) -> ObstacleRewardResult:
    time_cost = -config.k_time * dt
    progress = config.k_progress * progress_distance
    proximity = -config.k_proximity * max(0.0, config.safe_clearance - min_clearance)
    touched = min_clearance < config.touch_clearance
    touch = -config.k_touch * max(0.0, config.touch_clearance - min_clearance)

    smoothness = -config.k_smooth * abs(angular_z - prev_angular_z)
    if steer_fraction * prev_steer_fraction < 0.0:
        smoothness -= config.k_steer_reversal * min(
            abs(steer_fraction), abs(prev_steer_fraction)
        )

    hoop = config.hoop_pass_bonus * hoops_passed_this_step
    if hoop_missed:
        hoop -= config.hoop_miss_penalty

    total = time_cost + progress + proximity + touch + smoothness + hoop
    if collided:
        total -= config.collision_penalty

    return ObstacleRewardResult(
        total=total,
        time_cost=time_cost,
        progress=progress,
        proximity=proximity,
        touch=touch,
        smoothness=smoothness,
        hoop=hoop,
        collided=collided,
        touched=touched,
        hoop_missed=hoop_missed,
        hoops_passed=hoops_passed_this_step,
    )


if __name__ == "__main__":
    cfg = ObstacleRewardConfig()
    dt = 0.1

    fast = compute_reward(
        cfg,
        dt=dt,
        progress_distance=0.45,
        min_clearance=0.5,
        angular_z=0.1,
        prev_angular_z=0.08,
        collided=False,
    )
    slow = compute_reward(
        cfg,
        dt=dt,
        progress_distance=0.15,
        min_clearance=0.5,
        angular_z=0.1,
        prev_angular_z=0.08,
        collided=False,
    )
    stalled = compute_reward(
        cfg,
        dt=dt,
        progress_distance=0.0,
        min_clearance=0.5,
        angular_z=0.0,
        prev_angular_z=0.0,
        collided=False,
    )
    touching = compute_reward(
        cfg,
        dt=dt,
        progress_distance=0.45,
        min_clearance=0.05,
        angular_z=0.1,
        prev_angular_z=0.08,
        collided=False,
    )
    crashed = compute_reward(
        cfg,
        dt=dt,
        progress_distance=0.05,
        min_clearance=0.0,
        angular_z=0.4,
        prev_angular_z=-0.4,
        collided=True,
    )
    passed_hoop = compute_reward(
        cfg,
        dt=dt,
        progress_distance=0.45,
        min_clearance=0.5,
        angular_z=0.1,
        prev_angular_z=0.08,
        collided=False,
        hoops_passed_this_step=1,
    )
    missed_hoop = compute_reward(
        cfg,
        dt=dt,
        progress_distance=0.45,
        min_clearance=0.5,
        angular_z=0.1,
        prev_angular_z=0.08,
        collided=False,
        hoop_missed=True,
    )
    sawing = compute_reward(
        cfg,
        dt=dt,
        progress_distance=0.45,
        min_clearance=0.5,
        angular_z=0.1,
        prev_angular_z=0.08,
        collided=False,
        steer_fraction=1.0,
        prev_steer_fraction=-1.0,
    )

    print(f"fast step:           {fast.total:+.3f}")
    print(f"slow step:            {slow.total:+.3f}")
    print(f"stalled:              {stalled.total:+.3f}")
    print(f"touching:             {touching.total:+.3f}")
    print(f"crashed:              {crashed.total:+.3f}")
    print(f"passed a hoop:        {passed_hoop.total:+.3f}")
    print(f"missed a hoop:        {missed_hoop.total:+.3f}")
    print(f"sawing:               {sawing.total:+.3f}")

    assert fast.total > slow.total, "covering more ground per step should pay more"
    assert slow.total > stalled.total, "any progress should beat none"
    assert stalled.total > touching.total, (
        "brushing something should cost more than sitting still"
    )
    assert touching.total > crashed.total, (
        "a logged collision must cost more than a brush"
    )
    assert missed_hoop.total < crashed.total, (
        "missing a hoop must cost more than a collision"
    )
    assert passed_hoop.total > fast.total, (
        "clearing a hoop should pay on top of a clean step"
    )
    assert fast.total > sawing.total, "steady steering should beat sawing"
    print(
        "\nordering check passed: fast > slow > stalled > touching > crashed > missed hoop"
    )
