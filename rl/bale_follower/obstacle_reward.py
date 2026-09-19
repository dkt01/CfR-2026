"""Reward shaping for ObstacleCourseEnv.

The objective is the fastest lap that does not hit anything, so the return
for an episode is built to make *finishing* the best outcome available and
*finishing sooner* better than finishing later:

    k_progress * (s_end - s_start) + hoop bonuses
        + lap_bonus + k_finish_speed * (time_limit - T)   [on finishing]
        - penalties for what went wrong

`s` is arc length round the course centerline (see obstacle_course_path.py).
Rewarding its *difference* has two properties worth stating:

- It telescopes. Whatever route the car takes, the progress terms sum to
  `s_end - s_start`, so there is nothing to farm by driving back and forth,
  and the centerline only has to be approximately right.
- It is signed and course-relative. Driving the lap backwards scores
  negative and spinning on the spot scores zero.

**There is deliberately no per-step time cost.** Two runs have now died on
this, and the second death is the more instructive one:

1. The original reward charged `k_time` 2.0 a step while progress paid at
   most 0.4, so every step of driving was strictly negative and nothing was
   paid for finishing. PPO learned to stand still. Obvious in hindsight.
2. The fix made driving pay and added `lap_bonus`, but *kept* a -5.0/s time
   cost. That is a constant drain a policy can only escape by ending the
   episode, so for any policy that cannot yet drive the course the ranking
   was: trip the stuck detector at 5.1 s (-225) >> survive the 90 s cap
   (-450) >> and the +1012 lap sat behind 75 m of clean driving through
   0.63 m gaps, which random exploration never reaches. PPO learned to
   quit: over 6.6k steps `ep_len_mean` fell 85 -> 51 (the stuck window,
   exactly) while `ep_rew_mean` stayed flat at ~-217.

The invariant that kills this whole family: **a stationary car in open
space scores exactly 0 per step, and ending an episode is never worth more
than continuing it.** A negative per-step floor turns every episode-ending
condition -- stuck, collision, missed hoop -- into an escape hatch worth
more than playing on. `assert_no_quit_incentive` below checks this directly
and is the check both dead runs would have failed.

"Fastest" is then carried by two things that do not touch the floor: gamma
(0.99 at 10 Hz is a ~6.9 s half-life, so progress banked sooner is worth
more) and `k_finish_speed`, paid per second of the episode cap left unused
at the finish line.

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
    # strategy competes with driving.
    k_progress: float = 10.0
    # Paid once, on crossing the finish line, on top of the progress already
    # banked, so completing is strictly better than stopping just short.
    lap_bonus: float = 300.0
    # Paid per second of `episode_time_limit_s` still unused at the finish.
    # This is what makes it a *fastest* lap rather than merely a completed
    # one, and it does it without charging anything per step: a 37 s lap
    # under a 90 s cap collects 53 * this, a 60 s lap only 30 * this.
    # Deliberately NOT a per-step time penalty -- see the module docstring
    # for the two runs that killed.
    k_finish_speed: float = 10.0

    k_proximity: float = 1.0
    # Mild shaping only. Kept below the clearance real driving leaves: the
    # tightest bucket pair this course generates is 0.63 m of gap for a
    # 0.30 m car (the env logs this every layout), i.e. 0.165 m a side when
    # perfectly centred. A safe_clearance above that would charge the car
    # for driving the course correctly.
    safe_clearance: float = 0.25
    # Brushing something, short of a logged collision. Was 80.0 at 0.15 m,
    # which is inside that same 0.165 m gap clearance -- it billed up to
    # 7.2 a step, twice what the best possible step of progress pays, for
    # threading a gap exactly as intended. Now it only bites in the 0.02 m
    # between here and a genuine collision.
    k_touch: float = 30.0
    touch_clearance: float = 0.08
    # Point-cloud range below which a step counts as a genuine hit rather
    # than a close pass. There is no analytic ground truth to check this
    # against on this course (see obstacle_env.py), so this threshold is the
    # collision detector, not just a shaping cutoff -- tune it against actual
    # sensor noise before trusting it, not just against these defaults.
    collision_clearance: float = 0.06
    # 5 m of course progress -- deliberately modest, because it is not the
    # real deterrent. A collision *ends the episode*, so it already forfeits
    # every metre the car would have gone on to bank plus the lap bonus and
    # the finish-speed bonus; that is what stops a trained policy ramming
    # through. Charging 300 on top double-counted it, and with the per-step
    # floor now at 0 that made freezing the optimal policy: sitting still
    # scored 0 while attempting the course and crashing scored -300. The
    # first rollout after the curriculum went in duly came back at -296.
    # See assert_attempting_beats_freezing.
    collision_penalty: float = 50.0
    # Heavier than a collision: missing a hoop fails the run outright by the
    # rules, a collision does not. Scaled down with it for the same reason.
    hoop_miss_penalty: float = 150.0
    hoop_pass_bonus: float = 30.0
    k_smooth: float = 0.05
    k_steer_reversal: float = 0.8
    # Charged when the episode ends because the car stopped making progress.
    # Deliberately 0.0: any positive value makes ending early cheaper than
    # playing on for a policy that cannot yet drive, which is exactly the
    # trap run 2 fell into. The stuck detector still truncates the episode
    # (worth doing -- it frees the sim for a useful one), and truncation
    # already discourages itself, because SB3 bootstraps the value estimate
    # on truncation and a car mid-course has positive value to lose.
    stuck_penalty: float = 0.0


@dataclass
class ObstacleRewardResult:
    total: float
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
    time_remaining_s: float = 0.0,
) -> ObstacleRewardResult:
    """One step's reward.

    `progress_s` is the *signed* advance in course arc length over this step
    -- negative when the car moved backwards round the course.
    `time_remaining_s` is only read on the step that completes the lap.
    """
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

    lap = 0.0
    if lap_completed:
        lap = config.lap_bonus + config.k_finish_speed * max(0.0, time_remaining_s)

    total = progress + proximity + touch + smoothness + hoop + lap
    if collided:
        total -= config.collision_penalty
    if stuck:
        total -= config.stuck_penalty

    return ObstacleRewardResult(
        total=total,
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


def assert_no_quit_incentive(config: ObstacleRewardConfig) -> None:
    """The check both dead runs would have failed.

    A policy that cannot yet drive the course is choosing between ending the
    episode early and playing it out. If ending early pays more, PPO finds
    that long before it finds a lap, and the run is over before it starts.
    So: an idle car in open space must score exactly 0 a step, and no
    episode-ending condition may be worth more than continuing.
    """
    idle = compute_reward(
        config,
        progress_s=0.0,
        min_clearance=1.0,
        angular_z=0.0,
        prev_angular_z=0.0,
        collided=False,
    )
    assert idle.total == 0.0, (
        f"an idle car must score exactly 0 a step, not {idle.total:+.3f}; "
        "any per-step drain makes ending the episode an escape hatch"
    )
    quit_now = -config.stuck_penalty
    play_on = 0.0
    assert quit_now <= play_on, (
        f"tripping the stuck detector pays {quit_now:+.1f} against "
        f"{play_on:+.1f} for playing on -- that is the run-2 trap"
    )
    assert config.collision_penalty > 0.0, "crashing must not be a free exit"
    assert config.hoop_miss_penalty > config.collision_penalty, (
        "missing a hoop fails the run outright; it must cost more than a hit"
    )


def assert_attempting_beats_freezing(config: ObstacleRewardConfig) -> None:
    """The mirror image of the quit trap.

    Once an idle car scores 0 a step, any large terminal penalty makes doing
    nothing the safest policy available: a car that tries the course and
    crashes must not end up behind one that never moved, or PPO learns to
    freeze instead of to drive. The real deterrent against crashing is that
    it ends the episode and forfeits the rest of the lap, not the penalty
    itself.

    The bar: a car that gets a few metres in before hitting something must
    already be ahead of one that sat on the line.
    """
    attempt_m = config.collision_penalty / config.k_progress
    assert attempt_m <= 10.0, (
        f"a crash costs {attempt_m:.0f} m of progress, so the car must cover "
        f"{attempt_m:.0f} m before trying beats sitting still -- too far to "
        "find by exploration; lower collision_penalty or raise k_progress"
    )


if __name__ == "__main__":
    cfg = ObstacleRewardConfig()
    dt = 0.1

    def step(progress_s, min_clearance=0.5, **kwargs):
        return compute_reward(
            cfg,
            progress_s=progress_s,
            min_clearance=min_clearance,
            angular_z=0.1,
            prev_angular_z=0.08,
            collided=False,
            **kwargs,
        )

    fast = step(0.35)
    slow = step(0.15)
    stalled = step(0.0)
    backwards = step(-0.30)
    touching = step(0.35, min_clearance=0.05)
    threading = step(0.35, min_clearance=0.165)
    crashed = compute_reward(
        cfg,
        progress_s=0.05,
        min_clearance=0.0,
        angular_z=0.4,
        prev_angular_z=-0.4,
        collided=True,
    )
    passed_hoop = step(0.35, hoops_passed_this_step=1)
    missed_hoop = step(0.35, hoop_missed=True)
    sawing = step(0.35, steer_fraction=1.0, prev_steer_fraction=-1.0)

    print("--- per step ---")
    for label, result in (
        ("fast", fast),
        ("slow", slow),
        ("stalled", stalled),
        ("backwards", backwards),
        ("threading a gap", threading),
        ("touching", touching),
        ("crashed", crashed),
        ("passed a hoop", passed_hoop),
        ("missed a hoop", missed_hoop),
        ("sawing", sawing),
    ):
        print(f"  {label:18s} {result.total:+9.3f}")

    assert fast.total > slow.total, "covering more ground per step should pay more"
    assert slow.total > stalled.total, "any progress should beat none"
    assert stalled.total > backwards.total, "going the wrong way must cost"
    # Like for like: same progress, one of them brushing. (Comparing a
    # brushing step against a *stalled* one no longer means anything now
    # that progress, not a time cost, dominates -- a car that is moving and
    # scraping can out-score a car doing nothing, and should.)
    assert touching.total < fast.total, "brushing something must cost"
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
    # The tightest legal gap this course generates must not be billed as a
    # brush: 0.63 m of gap for a 0.30 m car is 0.165 m a side. safe_clearance
    # still reaches a little past that on purpose -- a gentle pull towards
    # the middle of a gap is worth having -- but it must stay a nudge, not a
    # bill. The old 0.15 m / k_touch 80 pairing charged twice a step of
    # progress for driving the course exactly as intended.
    assert touching.total < threading.total, "a real brush must cost more than a gap"
    assert (fast.total - threading.total) < 0.05 * fast.total, (
        "threading the tightest legal gap must cost near nothing, not "
        f"{fast.total - threading.total:.3f} of {fast.total:.3f}"
    )

    # --- the two degenerate-strategy checks -----------------------------
    assert_no_quit_incentive(cfg)
    assert_attempting_beats_freezing(cfg)

    # --- episode level -------------------------------------------------
    # Per-step ordering above is necessary and nowhere near sufficient:
    # what PPO maximizes is the sum.
    from obstacle_course_path import CourseProgress

    lap_length = CourseProgress().lap_length
    time_limit = 90.0
    stuck_window_s = 5.0
    hoops = 3

    def episode(
        distance_m, speed, finished, ended_stuck=False, hit=False, seconds=None
    ):
        """Undiscounted return for driving `distance_m` at `speed`."""
        if seconds is None:
            seconds = distance_m / speed if speed > 0 else stuck_window_s
        steps = max(1, round(seconds / dt))
        per_step = distance_m / steps if steps else 0.0
        total = sum(step(per_step).total for _ in range(steps))
        total += cfg.hoop_pass_bonus * (hoops if finished else 0)
        if finished:
            total += cfg.lap_bonus + cfg.k_finish_speed * max(0.0, time_limit - seconds)
        if ended_stuck:
            total -= cfg.stuck_penalty
        if hit:
            total -= cfg.collision_penalty
        return total

    quit_early = episode(0.0, 0.0, finished=False, ended_stuck=True)
    wander_full = episode(0.0, 0.0, finished=False, seconds=time_limit)
    try_and_crash = episode(8.0, 2.0, finished=False, hit=True)
    lap_2ms = episode(lap_length, 2.0, finished=True)
    lap_3ms = episode(lap_length, 3.0, finished=True)
    half_lap = episode(lap_length / 2, 2.0, finished=False)
    early_crash = episode(5.0, 2.0, finished=False, hit=True)

    print(f"\n--- whole episode (lap = {lap_length:.1f} m) ---")
    for label, value in (
        ("quit at the stuck window", quit_early),
        ("wander the full 90 s cap", wander_full),
        ("crash 5 m in", early_crash),
        ("try, crash 8 m in", try_and_crash),
        ("half a lap, then time out", half_lap),
        ("finish the lap at 2.0 m/s", lap_2ms),
        ("finish the lap at 3.0 m/s", lap_3ms),
    ):
        print(f"  {label:28s} {value:+10.1f}")

    assert lap_3ms > lap_2ms, "a quicker lap must beat a slower one"
    assert lap_2ms > half_lap, "finishing must beat stopping half way"
    assert half_lap > wander_full, "partial progress must beat none"
    assert half_lap > early_crash, "getting somewhere must beat crashing early"
    # The freeze trap: trying the course and failing part way must already
    # beat never having moved, or doing nothing is the safest policy there
    # is. The first curriculum rollout came back at -296 for exactly this.
    assert try_and_crash > wander_full, (
        f"attempting the course and crashing 8 m in scores {try_and_crash:+.1f} "
        f"against {wander_full:+.1f} for never moving -- PPO will learn to freeze"
    )
    # The one that matters: run 2 had quit_early (-225) against wander_full
    # (-450) and converged straight onto that +225.
    #
    # The bound is "worth less than one metre of progress" rather than
    # "worth nothing". Demanding exactly nothing would outlaw the smoothness
    # shaping entirely -- a car that spends 90 s turning does pay a whisker
    # for jerk, which is intended -- while still leaving the real failure
    # (a quit worth tens of metres of driving) impossible.
    quit_advantage = wander_full - quit_early
    assert quit_advantage < cfg.k_progress, (
        f"ending an episode early is worth {quit_advantage:+.1f}, i.e. "
        f"{quit_advantage / cfg.k_progress:.1f} m of progress -- PPO will "
        "find that before it finds a lap, exactly as it did twice before"
    )
    print("\nordering check passed: per step, no-quit invariant, whole episode")
