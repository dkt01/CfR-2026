"""Reward shaping for ObstacleCourseEnv.

The objective is the fastest lap that does not hit anything, so the return
for an episode is built to make *finishing* the best outcome available and
*finishing sooner* better than finishing later:

    k_progress * (s_end - s_start)  - k_time * T  + lap_bonus + hoop bonuses
                                     - penalties for what went wrong

`s` is arc length round the course centerline (see obstacle_course_path.py).
Rewarding its *difference* has two properties worth stating, because the
previous version of this file had neither:

- It telescopes. Whatever route the car takes, the progress terms sum to
  `s_end - s_start`, so there is nothing to farm by driving back and forth,
  and the centerline only has to be approximately right.
- It is signed and course-relative. Driving the lap backwards scores
  negative and spinning on the spot scores zero, neither of which was true
  of the heading-relative `forward_progress` this replaces.

**Why the episode-level check at the bottom of this file exists.** The
previous reward passed a per-step ordering check identical in spirit to the
one below, and was still catastrophically wrong: `k_time` cost 2.0 a step
while progress paid at most 0.4 at the car's then-top speed, so every step of
driving was strictly negative and nothing was ever paid for finishing. The
best possible lap scored -230 against -57 for standing still until the stuck
detector fired, and 182k steps of PPO duly converged on standing perfectly
still. Per-step ordering cannot see that, because the failure is entirely in
how the steps *sum*. Both checks now run.

A missed hoop fails the run outright per the rules, so it costs more than a
collision: a policy that would rather clip a bale than skip a hoop is
learning the right ordering. hoop_pass_bonus is paid once per hoop, on the
step hoop_monitor reports it newly passed.

Clearance and collision both come from the ZED's simulated point cloud
(env.scan_source == "cloud") rather than analytic course geometry: the
course's ramps, tunnel and helix are 3D in a way a top-down bale-style OBB
model cannot represent (see obstacle_env.py's module docstring), and the
point cloud is what the real car senses too.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObstacleRewardConfig:
    # Paid per metre of course progress. The lap is ~75 m, so finishing one
    # is worth ~750 before bonuses -- large enough that no degenerate
    # strategy competes with driving, which is the property the previous
    # tuning lacked.
    k_progress: float = 10.0
    # Cost per second. This is what makes it a *fastest* lap rather than
    # merely a completed one: total time cost is -k_time * T, so among laps
    # that finish, the quick one wins. Deliberately far below k_progress --
    # a step is net positive above 0.5 m/s of course progress, so crawling
    # is discouraged while any real driving pays.
    k_time: float = 5.0
    # Paid once, on crossing the finish line. On top of the progress already
    # banked, so completing is strictly better than stopping just short.
    lap_bonus: float = 300.0
    # Charged when the episode ends because the car stopped making progress.
    # With a positive-sum reward this is belt and braces rather than the
    # load-bearing term it would be otherwise, but it closes the escape
    # hatch that the previous reward turned into its global optimum.
    stuck_penalty: float = 200.0

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
    # ~30 m of course progress. Steep enough to matter on a 75 m lap without
    # making the car so timid it will not pass a bucket.
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
    lap: float
    collided: bool
    touched: bool
    hoop_missed: bool
    hoops_passed: int


def compute_reward(
    config: ObstacleRewardConfig,
    dt: float,
    progress_s: float,
    min_clearance: float,
    angular_z: float,
    prev_angular_z: float,
    collided: bool,
    hoop_missed: bool = False,
    hoops_passed_this_step: int = 0,
    steer_fraction: float = 0.0,
    prev_steer_fraction: float = 0.0,
    lap_completed: bool = False,
    stuck: bool = False,
) -> ObstacleRewardResult:
    """One step's reward.

    `progress_s` is the *signed* advance in course arc length over this step
    -- negative when the car moved backwards round the course.
    """
    time_cost = -config.k_time * dt
    progress = config.k_progress * progress_s
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

    lap = config.lap_bonus if lap_completed else 0.0

    total = time_cost + progress + proximity + touch + smoothness + hoop + lap
    if collided:
        total -= config.collision_penalty
    if stuck:
        total -= config.stuck_penalty

    return ObstacleRewardResult(
        total=total,
        time_cost=time_cost,
        progress=progress,
        proximity=proximity,
        touch=touch,
        smoothness=smoothness,
        hoop=hoop,
        lap=lap,
        collided=collided,
        touched=touched,
        hoop_missed=hoop_missed,
        hoops_passed=hoops_passed_this_step,
    )


if __name__ == "__main__":
    cfg = ObstacleRewardConfig()
    dt = 0.1

    def step(progress_s, min_clearance=0.5, **kwargs):
        return compute_reward(
            cfg,
            dt=dt,
            progress_s=progress_s,
            min_clearance=min_clearance,
            angular_z=0.1,
            prev_angular_z=0.08,
            collided=False,
            **kwargs,
        )

    fast = step(0.45)
    slow = step(0.15)
    stalled = step(0.0)
    backwards = step(-0.30)
    touching = step(0.45, min_clearance=0.05)
    crashed = compute_reward(
        cfg,
        dt=dt,
        progress_s=0.05,
        min_clearance=0.0,
        angular_z=0.4,
        prev_angular_z=-0.4,
        collided=True,
    )
    passed_hoop = step(0.45, hoops_passed_this_step=1)
    missed_hoop = step(0.45, hoop_missed=True)
    sawing = step(0.45, steer_fraction=1.0, prev_steer_fraction=-1.0)

    print("--- per step ---")
    for label, result in (
        ("fast", fast),
        ("slow", slow),
        ("stalled", stalled),
        ("backwards", backwards),
        ("touching", touching),
        ("crashed", crashed),
        ("passed a hoop", passed_hoop),
        ("missed a hoop", missed_hoop),
        ("sawing", sawing),
    ):
        print(f"  {label:16s} {result.total:+9.3f}")

    assert fast.total > slow.total, "covering more ground per step should pay more"
    assert slow.total > stalled.total, "any progress should beat none"
    assert stalled.total > backwards.total, "going the wrong way must cost"
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

    # --- episode level -------------------------------------------------
    # The check the previous reward did not have. Per-step ordering above is
    # necessary and nowhere near sufficient: what PPO maximizes is the sum.
    from obstacle_course_path import CourseProgress

    lap_length = CourseProgress().lap_length
    stuck_window_s = 5.0
    hoops = 3

    def episode(distance_m, speed, finished, ended_stuck=False, hit=False):
        """Undiscounted return for driving `distance_m` at `speed`."""
        seconds = distance_m / speed if speed > 0 else stuck_window_s
        steps = max(1, round(seconds / dt))
        per_step = distance_m / steps if steps else 0.0
        total = sum(step(per_step).total for _ in range(steps))
        total += cfg.hoop_pass_bonus * (hoops if finished else 0)
        if finished:
            total += cfg.lap_bonus
        if ended_stuck:
            total -= cfg.stuck_penalty
        if hit:
            total -= cfg.collision_penalty
        return total

    stand_still = episode(0.0, 0.0, finished=False, ended_stuck=True)
    lap_2ms = episode(lap_length, 2.0, finished=True)
    lap_3ms = episode(lap_length, 3.0, finished=True)
    half_lap = episode(lap_length / 2, 2.0, finished=False)
    early_crash = episode(5.0, 2.0, finished=False, hit=True)

    print(f"\n--- whole episode (lap = {lap_length:.1f} m) ---")
    for label, value in (
        ("stand still until stuck", stand_still),
        ("crash 5 m in", early_crash),
        ("half a lap, then time out", half_lap),
        ("finish the lap at 2.0 m/s", lap_2ms),
        ("finish the lap at 3.0 m/s", lap_3ms),
    ):
        print(f"  {label:28s} {value:+10.1f}")

    assert lap_3ms > lap_2ms, "a quicker lap must beat a slower one"
    assert lap_2ms > half_lap, "finishing must beat stopping half way"
    assert half_lap > stand_still, "partial progress must beat none"
    assert half_lap > early_crash, "getting somewhere must beat crashing early"
    assert stand_still < 0, "standing still must not be a viable strategy"
    print("\nordering check passed, per step AND over a whole episode")
