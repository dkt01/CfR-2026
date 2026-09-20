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
    # A crash already forfeits every metre left in the episode: `terminated`
    # zeroes the value bootstrap, so ending at 100 s of a 120 s limit gives up
    # ~500 of progress on its own, and that deterrent grows as the policy gets
    # faster. This penalty only has to stop a crash being an ESCAPE from a bad
    # episode, and 50 does that (an episode cut by the progress floor is worth
    # ~0, so crashing is still strictly worse).
    #
    # It was 200, which the policy could not get past. Driving only beats
    # circling once the car survives collision_penalty / k_progress metres --
    # 20 m at 200, against the ~8 m a fresh policy manages. So circling at 0
    # dominated driving at -120, and a 61k-step run sat at zero progress for
    # six straight evals. At 50 the break-even is 5 m, inside what the policy
    # can already do, so the gradient points at driving instead of away.
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
    # A crash must never be an escape from a bad episode. The test is not that
    # the penalty is large in absolute terms -- it is that no single step of
    # shaping can pay for one, so the only way to profit from a crash would be
    # to earn the penalty back before hitting the wall.
    assert crashed.total < -cfg.collision_penalty / 2.0, (
        "collision has to dominate the shaping terms"
    )
    assert crashed.total < min(
        fast.total, slow.total, circling.total, backwards.total, scraping.total
    ), "no shaping term may outweigh a collision on a single step"
    print("\nordering: fast > slow > circling > backwards > crashed")

    # Per-step ordering is not enough. An Obstacle Course run converged on
    # "stand perfectly still" for 182k steps while its own per-step check
    # passed the whole time: a per-step time cost made every episode a pure
    # cost, and the stuck truncation was the cheapest way out. So compare
    # whole-episode returns for the strategies a policy can actually find.
    LOOP_M, LIMIT_S, HZ = 110.096, 120.0, 10.0
    FLOOR_S = 5.0  # progress_window_s: how fast a hopeless episode is cut

    def episode(pace_m_s, crash_after_s=None):
        """Return for driving at a steady pace, optionally ending in a wall."""
        duration = (
            crash_after_s
            if crash_after_s
            else (LIMIT_S if pace_m_s * FLOOR_S >= 1.0 * FLOOR_S else FLOOR_S)
        )
        steps = duration * HZ
        total = steps * step(arc_progress=pace_m_s / HZ).total
        if crash_after_s:
            total -= cfg.collision_penalty
        return total

    still = episode(0.0)
    crash_early = episode(2.5, crash_after_s=2.0)
    crawl = episode(0.5)
    steady = episode(2.5)
    quick = episode(3.2)

    print("\nepisode return over a %.0f s limit (loop is %.1f m):" % (LIMIT_S, LOOP_M))
    print(f"  stand still (cut at {FLOOR_S:.0f} s): {still:+9.1f}")
    print(f"  crash after 2 s:                 {crash_early:+9.1f}")
    print(f"  crawl 0.5 m/s (cut):             {crawl:+9.1f}")
    print(
        f"  steady 2.5 m/s:                  {steady:+9.1f}  "
        f"({steady / LOOP_M / 10:.2f} laps)"
    )
    print(
        f"  quick 3.2 m/s:                   {quick:+9.1f}  "
        f"({quick / LOOP_M / 10:.2f} laps)"
    )

    assert quick > steady > still, "going faster must pay more than going slow"
    assert crawl < steady, "being cut for slow progress must cost the episode"

    # This used to assert `still > crash_early` -- that crashing must be worth
    # less than standing still. That cannot hold at the same time as the
    # break-even check below, because both are the same quantity: a crash
    # after d metres returns k_progress * d - collision_penalty, and standing
    # still returns ~0, so crashing beats standing still exactly when
    # d > collision_penalty / k_progress -- the break-even distance. One of
    # the two has to give.
    #
    # The break-even check is the one that matters. The old assertion was
    # guarding against crash-as-escape, which is a real failure only when a
    # per-step time cost makes an episode negative-sum and ending it early is
    # itself the prize. There is deliberately no time cost here (see the
    # module docstring), so continuing an episode is free and there is nothing
    # to escape. What actually deters a crash is the progress it forfeits,
    # which is checked directly below and, unlike a flat penalty, grows as the
    # policy gets faster.
    crash_at_60s = episode(2.5, crash_after_s=60.0)
    assert crash_at_60s < steady, (
        "crashing must cost the rest of the episode's progress"
    )
    assert steady - crash_at_60s > cfg.collision_penalty, (
        "forfeited progress, not the flat penalty, has to be the main deterrent"
    )

    # The trap that actually cost a run. A policy learns wall avoidance long
    # before it learns to lap, so for thousands of steps its best episode is
    # "drive a few seconds, then hit something". If that is worth less than
    # circling until the progress floor cuts the episode (~0), the gradient
    # points at circling and the policy never gets the practice it needs to
    # improve. Break-even is collision_penalty / k_progress metres, so this
    # asserts that distance stays inside what an unskilled policy manages.
    REACHABLE_M = 8.0  # measured: ~4 s at ~2 m/s, from the 61k-step run
    breakeven_m = cfg.collision_penalty / cfg.k_progress
    early_learner = episode(2.0, crash_after_s=4.0)
    print(
        f"\nbreak-even survival before a crash beats circling: {breakeven_m:.1f} m"
        f"  (an early policy reaches ~{REACHABLE_M:.0f} m)"
    )
    print(f"  drive 2.0 m/s for 4 s, then a wall:  {early_learner:+9.1f}")
    assert breakeven_m < REACHABLE_M, (
        f"a crash costs {cfg.collision_penalty} and progress pays "
        f"{cfg.k_progress}/m, so driving only beats circling after "
        f"{breakeven_m:.1f} m -- further than a learning policy gets, which "
        "makes circling the optimum and stalls the run at zero progress"
    )
    assert early_learner > still, (
        "trying and crashing must beat circling, or the policy never practises"
    )
    # The real trap: a negative-sum reward makes doing nothing the best play.
    assert still >= -1.0, (
        "standing still must not be profitable relative to driving -- if this "
        "goes negative, every episode is a pure cost and the progress floor "
        "becomes an escape hatch"
    )
    print(
        "\nepisode-level: quick > steady > crawl > still. Trying and crashing "
        f"(+{early_learner:.0f}) beats circling ({still:+.0f}), and a crash at "
        f"60 s forfeits {steady - crash_at_60s:.0f} against finishing."
    )
