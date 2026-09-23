# Lap-time RL policy (`lap_*`)

A second RL stack in this directory, sharing its geometry, camera model and
simulator plumbing with the bale follower but optimising a different thing:
**laps of the speed course, as fast as the car can drive them without
touching a bale.** The corridor-following stack (`env.py`, `reward.py`,
`config.yaml`, `train.py`, checkpoints v1-v11) is untouched and still runs.

| | corridor follower | lap racer |
|---|---|---|
| objective | metres before a collision | lap time |
| files | `env.py` `reward.py` `train.py` `evaluate.py` `config.yaml` | `lap_env.py` `lap_reward.py` `lap_track.py` `train_lap.py` `evaluate_lap.py` `config_lap.yaml` |
| progress | displacement on the car's own heading | signed arc length round the planned loop |
| contact | min of a 110 deg ray fan from the car's **centre** | gap between the car's **footprint** and the bale's |
| steering action | an angle, then clamped to the servo's slew | the servo's **rate**, integrated |
| stuck | truncates the episode | recovery mode the policy drives out of |
| episode | 60 s from near the spawn | 150 s or 3 laps, starting anywhere on the loop |
| benchmark | 130 m clean-episode distance (v6) | planner 28.4 s, MPC racer 29.95 s |

## The objective, in one line

    sum over a lap of (k_progress * ds - k_time * dt)
        = k_progress * 110.1 m - k_time * lap_time

The loop length is a constant, so **maximising the return over a lap is
minimising the lap time** -- densely, one step at a time, rather than as a
sparse bonus hundreds of steps after the corner that earned it. Everything
else in `lap_reward.py` exists to stop that being satisfied in a way nobody
wants; `lap_reward_probe.py` prints what each term pays and asserts the
orderings.

Four properties are worth knowing before touching a coefficient:

* **Break-even speed is `k_time / k_progress`** (1.0 m/s). Below it the clock
  outruns the progress. It must stay under the ~1.4 m/s the car needs to
  rotate at full lock, or hairpins stop being worth driving.
* **Stoppage is charged twice**: the clock runs regardless (`k_time`), and
  `k_stall` adds to it below 0.35 m/s. A stopped car loses 16/s.
* **The lap bonus rises as the lap time falls**
  (`k_lap_base + k_lap_pace * max(0, target_lap_time - lap_time)`): 200 for a
  30 s lap, 100 for a 40 s lap, 50 for anything at or over the 45 s target.
  Episodes run to three laps, so a policy is paid three times for improving.
* **Contact is measured on the bodywork.** `bale_geometry.body_clearance`
  returns the gap between the car's footprint and the nearest bale's, in
  every direction. The old scan-based figure is a ray from the car's centre
  inside a 110 deg fan: a bale 8 cm off the flank reads as 0.28 m of
  clearance there, and is invisible if it is off the quarter panel.

## Escape by crashing, and the inequality that prevents it

A wedged car pressed against a bale sits in a stream of negative reward.
Truncation (the stuck timeout) bootstraps, so it does not end that stream in
the policy's eyes -- but a **collision terminates**, and a terminal state has
no future cost at all. If the stream is expensive enough, driving into the
bale becomes the cheapest way out, and the reward pays for the one outcome
the objective exists to prevent.

Two things keep that from happening, and both are asserted in the probe:

1. `recovery_contact_scale` (0.25) reduces the clearance and touch charges
   while the car is already in recovery. Those terms are there to stop the
   car *choosing* to drive at a bale; charging a car already against one
   -120/s buys no better behaviour.
2. `collision_penalty` (400) must exceed the worst stream times the stuck
   window: 46/s x 8 s = 368.

Change `control_hz`, `stuck_window_s`, `k_touch` or `k_stall` and that
inequality moves. Run the probe.

## Rocking in place, and why the detectors measure displacement

The first training run found an exploit inside 16k steps, and it is worth
keeping written down because the fix is not obvious from the reward alone.

Both the recovery and stuck detectors originally summed per-step distances
over their window. A car that rocks forwards and backwards accumulates
"travel" without going anywhere, so it never registered as stuck, never
triggered the truncation, and -- because its instantaneous speed stayed above
`stall_speed` -- never paid the stall charge either. It paid only the 6/s
clock, and in exchange it never risked the -400 collision. The deterministic
evaluation at 16k steps: three episodes, mean speeds 0.12, 0.00 and
0.41 m/s, one of them running 995 steps to cover **-0.1 m**.

Two changes close it:

* Both windows now measure **net displacement** from where the car was at the
  start of them (what `path_racer._check_stuck` always did), so rocking reads
  as exactly the zero progress it is.
* `k_stall` is charged whenever the car is slow **or** the recovery flag is
  up, not on speed alone. Going nowhere costs 16/s however energetically it
  is done.

`k_align` went 8 -> 12 at the same time, because the stall charge now applies
through a recovery too: at 8 a purposeful reverse-and-turn was merely the
least-bad option (-5/s), where it should be a good one (+1/s). Rocking, which
moves without reducing the heading error, stays at -23/s. That gap is what
the alignment term is for.

The probe carries both as invariants (`rocking < crawling`,
`wedged_out > 0`), so the next person to rebalance these coefficients finds
out immediately rather than 16k steps in.

## Getting unstuck

`reverse_speed` is 1.2 m/s, and while the recovery flag is up the progress
term is clipped at zero, so backing up is not charged as negative progress --
the clock still runs, so recovery is never free. `k_align` pays for each
radian of heading error removed while recovering, which is what "turn back
into the direction of travel" means; it is potential-based (the increase is
charged symmetrically), so it cannot be farmed by swinging the nose.

The flag itself is derived from travelled distance, not arc length: under
0.4 m in 2 s enters recovery, 1.0 m of forward travel leaves it. Both are
computable from odometry alone, which is why the flag can be -- and is -- in
the observation. A policy cannot be asked to behave differently in a regime
it cannot see.

15% of episodes start deliberately wedged (angled 0.9-1.8 rad across the
corridor with a bale close in), because recovery cannot be learned from
states a good policy never reaches.

## Observation and action

    observation = [scan_t (36 bins / 6 m)] + [scan_t-1] +
                  [speed, yaw_rate, steering_angle, recovering]   -> 76 floats
    action      = [throttle -> target speed in [-1.2, 4.5] m/s,
                   steering RATE as a fraction of 3.5 rad/s]

Sensor-only, as before: the loop, the arc length, the lap clock and the body
clearance are privileged and live in the reward and the metrics. Nothing in
the observation needs a map or a global pose.

The steering action is the significant change. Making the *rate* the action
means the commanded angle cannot jump further than the servo can move,
regardless of what the policy asks for -- the train/deploy mismatch that cost
v5 two thirds of its distance is structurally impossible. It also makes the
smoothness terms meaningful: a held corner costs nothing (the rate is zero),
turning costs a little, and chatter costs a lot, because `k_steer_jerk`
charges the *change* in rate. A quadratic on the rate alone cannot tell
sawing from turning -- both have the same mean.

## Running it

Everything that needs neither ROS nor Gazebo, and should pass before a run:

    python3 lap_track.py          # projection, lap counting, seam, orientation
    python3 lap_reward_probe.py   # term-by-term table and the invariants
    python3 lap_env_selftest.py   # a scripted lap, a wedge, a recovery

`lap_env_selftest.py` stubs ROS and replaces Gazebo with the same kinematic
bicycle the env commands, so it tests the environment's *bookkeeping*, not
the vehicle. A scripted pure-pursuit driver laps in 37.8 s on it.

Training, in two stages (the sim stack is brought up and torn down for you;
`CFR_SENSORS=1` renders the ZED, which `scan_source: cloud` requires):

    export CFR_SENSORS=1
    ./launch_lap_training.sh --max-speed 3.0 --total-timesteps 150000 \
        --checkpoint-dir checkpoints_lap1
    ./launch_lap_training.sh --resume-from checkpoints_lap1/best_model.zip \
        --total-timesteps 450000 --checkpoint-dir checkpoints_lap2

Two stages because a fresh policy that meets 4.5 m/s learns to brake with its
exploration noise rather than its mean action -- the v3 failure, which cost a
100k-step run. Gazebo's pose stream still dies after about five hours, so use
`train_resilient.sh`'s pattern for anything longer, and watch
`[lap eval]` lines rather than the PPO reward curve: they are deterministic,
which is what deployment runs.

    ./test_lap_policy.sh --checkpoint checkpoints_lap2/best_model.zip \
        --episodes 5 --from-start

`--from-start` starts every episode at the SDF spawn pose, the way a
competition run begins.

## What is not done

* **No deployment runner.** `run_policy.py` builds the old observation and
  treats the second action as an angle, so it cannot run these checkpoints.
  Porting it is the step between "fast in the training env" and "fast in the
  simulation the rest of the stack runs", and it needs the recovery flag
  computed from odometry -- deliberately possible, but not written.
* **No run has been done.** Every number above is either measured elsewhere
  (the planner, the MPC racer, the vehicle calibration) or comes from the
  offline tests. What the policy actually laps in is unknown until it trains.
* **`k_lap_record` is off.** Paying extra for beating the env's own best lap
  is explicitly non-stationary; it is there for a plateau where laps are
  clean but the pace has stopped improving, not for the first run.
