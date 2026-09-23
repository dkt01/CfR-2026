"""Reward shaping for the lap-time objective.

`reward.py` asks "how far did you get down a corridor without hitting a
wall". This asks a different question -- "how many seconds did that lap take"
-- and the two need different arithmetic, so this is a separate module rather
than more coefficients in the old one. Old checkpoints keep scoring under
their own shaping (`evaluate.py` reads the reward config out of the
checkpoint metadata), and nothing here changes what they mean.

The objective in one line:

    sum over a lap of (k_progress * ds - k_time * dt)
        = k_progress * loop_length - k_time * lap_time

The loop length is a constant, so maximising the return over a lap IS
minimising the lap time -- densely, one step at a time, instead of as a
sparse bonus 600 steps after the corner that earned it. Every other term
exists to stop that objective being satisfied in a way nobody wants:

| term | what it stops |
|---|---|
| `k_time` charged every step | dawdling; it is also the stoppage penalty, since a stopped car earns nothing and pays anyway |
| `k_stall` below `stall_speed` | the specific case of *stopped*, charged on top of clock time so a wedged car is losing ground fast |
| `k_clearance` / `k_touch` on the BODY gap | driving fast by using the bales as guide rails |
| `collision_penalty` | trading a crash for a fast split |
| `k_steer_rate` / `k_steer_jerk` / `k_steer_reversal` | getting round a corner by sawing at the servo's rate limit |
| `k_align` while recovering | sitting nose-in against a bale rather than reversing and turning back down the course |
| `k_lateral` / `k_heading`, potential-based | nothing -- it cannot change the optimal policy; it exists so an untrained one can find the corridor before the collision penalty teaches it to stop |
| lap bonus scaled by lap time | making the last corner of a lap worth the same as any other corner |

Two properties worth keeping when editing:

* **The break-even speed is `k_time / k_progress`.** Below it, driving scores
  worse than the clock alone; above it, every extra metre per second pays.
  At 6.0 / 6.0 that is 1.0 m/s -- under the 1.4 m/s the vehicle needs to
  rotate at full lock, so no hairpin is ever worth refusing.
* **`k_align` is potential-based** (it pays the *decrease* in |heading
  error|, and charges the increase symmetrically), so a car cannot farm it by
  swinging its nose back and forth: any closed cycle sums to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LapRewardConfig:
    # --- the objective ---------------------------------------------------
    # Per metre of signed arc length gained along the planned loop. Signed:
    # driving backwards up the course pays negative, so progress cannot be
    # farmed by shuffling back and forth.
    k_progress: float = 6.0
    # Per second of episode time, charged unconditionally. This is what makes
    # the return a lap time rather than a distance.
    k_time: float = 6.0

    # --- stoppage --------------------------------------------------------
    # Extra per-second charge while the car is not getting anywhere, on top
    # of k_time. A stopped car pays (k_time + k_stall) per second and earns
    # nothing, so five seconds wedged costs about a third of a lap's
    # earnings.
    #
    # "Not getting anywhere" is instantaneous speed below `stall_speed` OR
    # the recovery flag, which the env raises on net displacement. Speed
    # alone is not enough: a car rocking forwards and backwards is never
    # slow, never stalls, and never makes progress -- which is the policy the
    # first run converged on by 16k steps.
    k_stall: float = 10.0
    stall_speed: float = 0.35  # m/s, below which the car counts as stopped

    # --- the bales -------------------------------------------------------
    # Both are measured on `bale_geometry.body_clearance` -- the gap between
    # the car's footprint and the bale's, not a ray from the car's centre.
    # The corridors are ~0.95 m wide and the car is 0.30 m, so a perfectly
    # centred car has ~0.32 m per side: safe_clearance must stay under that
    # or every pose on the course is penalised.
    k_clearance: float = 2.0
    safe_clearance: float = 0.25
    # The steep part. Contact must lose at top speed, not just at cruise:
    # at 4 m/s a step earns +1.2 from progress, and a 5 cm gap costs -3.3.
    k_touch: float = 40.0
    touch_clearance: float = 0.10
    # Charged once, and the episode ends.
    #
    # Deliberately SMALL, because termination is already the deterrent: a
    # collision forfeits every remaining lap, which for a competent policy is
    # a thousand points of future reward, scaled automatically to how good
    # the policy is. An explicit penalty on top only has to break ties.
    #
    # Measured why it matters, over three runs: at 400 and then 250 the
    # policy's choice early in training was between driving (+18/s until a
    # crash it cannot yet avoid) and creeping (-16/s until the stuck
    # truncation). Driving only won if it could survive 22 s, then 8.6 s --
    # neither reachable by an untrained policy -- so it learned to stand
    # still, twice. At 100 it wins after 0.2 s of survival, which is
    # reachable immediately, and the deterrent grows on its own as the policy
    # gets better and has more to lose.
    #
    # The floor under it: a wedged car must never find crashing cheaper than
    # waiting out the stuck window (16/s * 5 s = 80 here). The probe asserts
    # exactly that.
    collision_penalty: float = 100.0
    # Zero: the clearance and touch charges are there to stop the car
    # CHOOSING to drive at a bale. A car already wedged against one has made
    # that choice, and charging it again only raises the price of the wedge
    # until crashing out of it looks cheap.
    recovery_contact_scale: float = 0.0

    # --- steering --------------------------------------------------------
    # The action is a steering RATE, so this charges the servo actually
    # moving. A held corner costs nothing; only changing the angle does.
    k_steer_rate: float = 0.30
    # Change in the rate command between steps -- steering jerk. This is the
    # term that actually removes chatter: a quadratic on the rate alone
    # charges alternating +/-0.25 exactly what it charges a smooth held
    # 0.25, so it cannot tell sawing from turning. The difference between
    # them is entirely in how fast the rate itself changes.
    k_steer_jerk: float = 0.8
    # Sign reversals of the ANGLE, in proportion to how far the wheel swung
    # through centre -- the large-amplitude version of the same thing.
    k_steer_reversal: float = 0.5

    # --- getting unstuck -------------------------------------------------
    # Per radian of heading error removed while in recovery. Pays for
    # reversing and swinging the nose back down the course; potential-based,
    # so it cannot be cycled for profit.
    #
    # It has to outweigh the stall charge that now applies throughout a
    # recovery, or backing out is merely the least-bad option rather than a
    # good one. At 12.0 a purposeful reverse-and-turn scores positive, while
    # rocking in place -- which moves without reducing the heading error --
    # still scores -23/s. That difference is the whole point of the term.
    k_align: float = 12.0

    # --- learning the corridor at all ------------------------------------
    # Potential-based shaping (Ng, Harada & Russell 1999) on distance from
    # the planned line and heading error against it:
    #
    #     F = gamma * Phi(s') - Phi(s),  Phi = -(k_lateral*|lat| + k_heading*|psi|)
    #
    # which provably leaves the optimal policy unchanged -- over any closed
    # path the terms telescope to zero, so the car is still free to cut an
    # apex or run wide if that is quicker. What it changes is the LEARNING
    # problem. Without it the only thing telling an untrained policy which
    # way to steer is the cosine in the progress term, until it is within
    # 0.25 m of a bale and it is already too late; the first two runs both
    # converged on creeping instead of driving because of it.
    k_lateral: float = 3.0
    k_heading: float = 2.0
    # Past this the car is off the line in a way the shaping cannot help
    # with, and an uncapped term would swamp everything else.
    lateral_cap: float = 1.0
    # Must match the training discount, or the telescoping is not exact.
    shaping_gamma: float = 0.998

    # --- lap bonus -------------------------------------------------------
    # Paid on the step a lap completes, and larger the faster that lap was:
    #   bonus = k_lap_base + k_lap_pace * max(0, target_lap_time - lap_time)
    # The planner's optimum is 28.4 s and the MPC racer's best is 29.95 s, so
    # a 45 s target makes every lap a real policy can drive sit on the sloped
    # part of the curve rather than pinned at either end.
    k_lap_base: float = 50.0
    k_lap_pace: float = 10.0
    target_lap_time: float = 45.0
    # Extra per second faster than this env's own best lap so far. Explicitly
    # non-stationary -- the same lap pays less once it has been beaten -- so
    # it defaults off; the pace bonus above already pays more for a faster
    # lap, without moving the target under the value function.
    k_lap_record: float = 0.0


@dataclass
class LapRewardResult:
    total: float
    progress: float
    time: float
    stall: float
    clearance: float
    touch: float
    steering: float
    align: float
    shaping: float
    lap_bonus: float
    collided: bool
    touched: bool


def compute_lap_reward(
    config: LapRewardConfig,
    *,
    delta_s: float,
    dt: float,
    speed: float,
    clearance: float,
    steer_rate_fraction: float,
    prev_steer_rate_fraction: float,
    steer_fraction: float,
    prev_steer_fraction: float,
    collided: bool,
    recovering: bool = False,
    heading_error: float = 0.0,
    prev_heading_error: float = 0.0,
    lateral: float = 0.0,
    prev_lateral: float = 0.0,
    lap_time: float | None = None,
    best_lap_time: float = math.inf,
) -> LapRewardResult:
    """One step of reward.

    `delta_s` is signed arc length along the loop (lap_track), `clearance` is
    the body-to-bale gap in metres, `steer_rate_fraction` is the commanded
    servo rate as a fraction of its limit, and `recovering` says the episode
    logic has declared the car stuck.
    """
    # While recovering, backing up is not charged as negative progress: the
    # way out of a nose-in wedge is a few metres of reverse, and paying the
    # progress penalty for them would make sitting still the better move.
    # The clock keeps running throughout, so recovery is never free.
    progress_distance = max(0.0, delta_s) if recovering else delta_s
    progress = config.k_progress * progress_distance
    time_cost = -config.k_time * dt
    stalled = abs(speed) < config.stall_speed or recovering
    stall = -config.k_stall * dt if stalled else 0.0

    contact_scale = config.recovery_contact_scale if recovering else 1.0
    clearance_cost = 0.0
    if clearance < config.safe_clearance:
        deficit = (config.safe_clearance - clearance) / config.safe_clearance
        clearance_cost = -config.k_clearance * deficit * deficit * contact_scale
    touched = clearance < config.touch_clearance
    touch = (
        -config.k_touch * max(0.0, config.touch_clearance - clearance) * contact_scale
    )

    steering = -config.k_steer_rate * steer_rate_fraction * steer_rate_fraction
    jerk = steer_rate_fraction - prev_steer_rate_fraction
    steering -= config.k_steer_jerk * jerk * jerk
    if steer_fraction * prev_steer_fraction < 0.0:
        steering -= config.k_steer_reversal * min(
            abs(steer_fraction), abs(prev_steer_fraction)
        )

    align = 0.0
    if recovering:
        align = config.k_align * (abs(prev_heading_error) - abs(heading_error))

    # Potential-based shaping towards the line. Charged always, including
    # during a recovery, where it points the same way k_align does.
    def potential(lat: float, psi: float) -> float:
        return -(
            config.k_lateral * min(abs(lat), config.lateral_cap)
            + config.k_heading * abs(psi)
        )

    shaping = config.shaping_gamma * potential(lateral, heading_error) - potential(
        prev_lateral, prev_heading_error
    )

    lap_bonus = 0.0
    if lap_time is not None:
        lap_bonus = config.k_lap_base + config.k_lap_pace * max(
            0.0, config.target_lap_time - lap_time
        )
        if config.k_lap_record > 0.0 and math.isfinite(best_lap_time):
            lap_bonus += config.k_lap_record * max(0.0, best_lap_time - lap_time)

    total = (
        progress
        + time_cost
        + stall
        + clearance_cost
        + touch
        + steering
        + align
        + shaping
        + lap_bonus
    )
    if collided:
        total -= config.collision_penalty
    return LapRewardResult(
        total=total,
        progress=progress,
        time=time_cost,
        stall=stall,
        clearance=clearance_cost,
        touch=touch,
        steering=steering,
        align=align,
        shaping=shaping,
        lap_bonus=lap_bonus,
        collided=collided,
        touched=touched,
    )
