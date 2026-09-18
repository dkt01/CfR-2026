# RL Training & Test Pipeline — What Was Built and Why

*2026-09-09, branch `simulation-prep`, directory `rl/bale_follower/`*

## Deliverables

| Piece | File | Why |
|---|---|---|
| Training launcher | `launch_training.sh` | One command: brings up the headless Gazebo stack, waits for the teleport API and ground-truth pose bridge (the two things the env dies without), runs PPO, tears everything down on exit. Refuses to start a second Gazebo server — duplicate servers on the same topics look like broken physics. `CFR_USE_RUNNING_SIM=1` reuses a sim you started yourself. |
| Test launcher | `test_policy.sh` | Same lifecycle around `evaluate.py`: N deterministic episodes of a checkpoint, printing and saving distance, mean speed, collision rate, minimum wall clearance, and steering jerk. `--no-smoother` gives a raw-policy A/B baseline; `--max-speed` / `--traction` override the trained caps at test time. |
| CasADi smoother | `casadi_smoother.py` | Sits between the policy and `/cmd_vel` at test/deployment time. Solves an 8-step optimal-control problem each tick (IPOPT, ~15 ms against a 100 ms budget at 10 Hz): track the policy's (speed, steering) target subject to a kinematic bicycle model, accel ≤ traction·g, steering-rate limits, and the friction-circle constraint v²·tan(δ)/L ≤ traction·g. |
| Traction model | `env.py` + `config.yaml` `traction:` | The same two grip limits, applied greedily during training, so the policy never learns commands the drivetrain won't deliver and the train/test dynamics match. |
| Simulated ZED 2i | `zed_sim.py` | 110° FOV clamp, range-squared stereo noise (~1% at 3 m, ~3% at 6 m, matching the ZED 2i spec), 3% per-bin dropout — applied to the observation only; reward and collisions stay on ground truth, like a real robot. |

## Key decisions and why

**Why RL + bales for wall-following.** The course has no usable centerline
(the SDF's bale order is DXF drawing order), so the task is framed as "make
forward progress in a corridor without touching the walls" — the existing
env's reward already encodes exactly that, and the bales are both the walls
the policy must avoid and the only feature it observes. That framing was kept.

**Why the ZED point cloud is modelled statistically, not rendered.** The ZED
SDK (`pyzed`) is not installed and the Gazebo world has no rendering-sensor
plugin (the on-model ZED 2i is cosmetic pending a GPU/EGL context). So the
camera's *statistics* — FOV, quadratic range noise, dropout — are applied to
the analytic range scan instead. This is the honest middle ground: it removes
the "trained on perfect 180° ground truth" gap the README flagged, without
pretending a depth pipeline exists. Rendering a real simulated point cloud
remains future work (rgbd_camera sensor + render engine + EGL context).

**Why CasADi at test time but a greedy clamp at train time.** The optimizer
is what makes driving accurate and smooth: it brakes *before* a corner the
tires can't carry at speed rather than understeering through it, and it rate-
limits steering so commands are drivetrain-feasible. Running IPOPT inside
every PPO step would add fragility and wall-clock cost to a training loop
that is already capped at 10 steps/s by the wall-clock-paced sim; enforcing
the *same* constraint set greedily in the env keeps the dynamics the policy
learns consistent with what the smoother will execute. If a solve ever fails,
the fallback is the reference clamped to the same limits — degraded, never
wild.

**Why max speed and traction are config, not code.** `max_speed` and
`traction` live in `config.yaml` (env section), are stamped into each
checkpoint's metadata JSON, and can be overridden per evaluation run. Small
changes (a slicker floor, a lower speed cap for a demo) need no retrain; the
metrics run will show if a big change degrades the policy enough to warrant
fine-tuning at the new limits.

## Verified

- `python zed_sim.py` — noise grows quadratically with range, dropout at the
  configured rate.
- `python casadi_smoother.py` — under a full-speed + full-lock request, accel
  and steering-rate slews hold, lateral acceleration pins at traction·g
  (5.89 m/s² at μ=0.6), benign references track exactly; ~15 ms per solve.
- `./launch_training.sh` — end-to-end against the live Gazebo stack: episodes
  reset, rewards flow, checkpoints + metadata written.
- A real 20k-step run (`checkpoints_v2/final_model.zip`, ~45 min at the sim's
  ~8 steps/s): mean episode reward climbed from −21 to ~44, episode length to
  ~26 s of driving.

## Results of the 20k-step checkpoint (5 episodes each)

| | with CasADi smoother | raw policy |
|---|---|---|
| mean distance before collision | 7.0 m | 14.2 m |
| mean speed | 1.60 m/s | 1.92 m/s |
| steering jerk (rad/tick) | **0.068** | 0.085 |
| collision rate | 1.0 | 1.0 |

Two honest readings. First, 20k steps is a short run: the policy makes real
progress down the corridor (up to 18 m) but always finds a bale eventually —
a longer run (100k+, ~3.5 h) is the obvious next step before judging the
approach. Second, the smoother currently trades distance for smoothness: its
steering-rate limit lags a policy that never trained under that limit, so
avoidance maneuvers arrive late. Two fixes, either or both: raise
`max_steering_rate` in `config.yaml` (the physical servo is fast), and/or
train longer so the policy steers earlier and gentler instead of relying on
last-moment flicks the filter suppresses.

## 100k-step run with centering reward (checkpoints_v3)

Changes for this run: `max_steering_rate` raised 4.0 → 8.0 (full steering
range within one 10 Hz tick, removing the lag measured above), and a new
**both-walls centering term** — narrow ground-truth ray fans at ±90° measure
the nearest bale left and right, and `k_center: 2.0` penalizes the imbalance
per metre, so the optimum line is the corridor center (side distances capped
at 1.5 m so open hairpins aren't punished). Reward-only privileged signal,
like progress; the policy still observes only the forward ZED-style scan.

Training was healthy: mean episode reward climbed monotonically −44 → +22
over ~3.5 h, episode length to ~390 steps (~39 s of driving).

Evaluation (5 episodes each) then exposed the real finding of this run:

| v3 checkpoint | mean dist | mean speed | collisions | clean-run dist |
|---|---|---|---|---|
| deterministic + smoother | 6.3 m | 3.3 m/s | 5/5 | — |
| deterministic, raw | 15.5 m | 2.1 m/s | 5/5 | — |
| **stochastic** (as trained), raw | 21.1 m | 0.69 m/s | 4/5 | 22.0 m |
| deterministic + smoother, **max_speed 1.5** | 16.3 m | 1.2 m/s | 3/5 | 11.3 m |

**Diagnosis: the policy learned to use its exploration noise as a brake.**
Sampled like training (row 3) it drives 20+ m at a sedate 0.7 m/s and
matches the training curves. Its Gaussian *mean* throttle, though, is nearly
floored — deterministic evaluation drives 3–4 m/s into the first hairpin.
The exploration noise (std ≈ 0.88 against a ±1 action range) was doing the
slowing down, so PPO never had to learn a slow mean.

The configurable max-speed cap proved its worth here: capping to 1.5 m/s at
test time (row 4, no retrain) restores deterministic performance to the best
distances at nearly twice the stochastic speed, and cuts collisions to 3/5 —
two of which were "stuck" terminations, not crashes.

## 250k-step run with reverse gear + deterministic selection (checkpoints_v4)

Changes for this run, addressing both v3 findings and the stuck-in-hairpins
behavior:

- **Reverse gear**: action speed now spans `[-0.5, max_speed]` m/s — with a
  forward-only range, a car nosed against a bale was stuck forever. Reward
  still pays forward progress only, so reverse is a recovery move.
- **`env.max_speed` 4.0 → 2.0** so the deterministic mean must learn real
  speed control instead of leaning on exploration noise as a brake.
- **DeterministicEvalCallback** in train.py: every ~8k steps it runs 3
  deterministic episodes and keeps `best_model.zip` by mean distance — the
  deployment metric the v3 training curves masked.
- **Scripted stuck recovery in run_policy.py** (any checkpoint): <0.15 m of
  motion over 3 s triggers 1.5 s of reverse steering toward the obstacle
  (nose swings away), then hands back to the policy.

The run crashed once at 139k steps (Gazebo's pose stream died after ~5.5 h;
a single-instance longevity issue, not a training bug) and resumed cleanly
from the checkpoint; the callback reloads the previous best on resume so a
restart cannot overwrite a better model. Late-run evals oscillated 14–19 m
while the ~197k-step checkpoint held 23.2 m — best-model selection earned
its keep.

**best_model.zip, deterministic + smoother, 5 episodes:**

| | v3 (best config: cap 1.5) | v4 best_model |
|---|---|---|
| mean distance | 16.3 m | **20.5 m** |
| mean speed | 1.2 m/s | **1.62 m/s** |
| collision rate | 3/5 | **0.6 (2 of 3 non-clean were "stuck", not crashes)** |
| steering jerk (rad/tick) | 0.068 | **0.054** |

Farther, faster, smoother, and fewer crashes — natively deterministic, no
test-time cap tricks. The two "stuck" terminations end the *evaluation*
episode by rule, but in deployment `run_policy.py`'s recovery (and the
policy's own reverse gear) backs out of exactly that pose and keeps driving.

## Planned-line racing, and the sim bug it uncovered

The course is static and already parsed from the SDF, and a QuestNav pose
will be available on the Orin, so perception-driven driving is the wrong tool
for a fastest-lap objective. Two new pieces plan on the geometry instead:

- **`course_path.py`** — offline planner, run once per course change. Builds
  a 5 cm occupancy grid of bale OBBs inflated by the car's half-width,
  skeletonizes the *corridor component* (the spawn's connected component, so
  the open field outside the walls is excluded), strips dead-end spurs to
  leave the closed loop, then a CasADi elastic-band NLP straightens the line
  within each point's free disc, and a friction-circle + forward/backward
  pass gives a minimum-time speed profile. Output: 110.1 m loop, planned
  flying lap 28.4 s at mu=0.6.
- **`path_racer.py`** — runtime tracker. A CasADi MPC (kinematic bicycle,
  1 s horizon, ~20 ms/solve) tracks a time-parameterized reference so
  hairpin braking enters the horizon a second early; pure pursuit is the
  per-tick fallback, and a stuck-recovery reverse sits on top. Pose source is
  a parameter: `--pose-msg tf` for the sim's ground-truth bridge today,
  `--pose-msg odom` for QuestNav's `nav_msgs/Odometry` on the Orin (the
  republisher must put the Quest pose in the course frame).

**The blocker: the simulated vehicle cannot turn as tightly as every
controller assumes.** The racer stuck in hairpins with *0.45 m of clearance
on all sides*, commanding 0.59 m/s while sitting still at 6 cm cross-track
error — ruling out both collision and tracking error. An open-field steering
probe (teleport to open ground, hold a steering angle, measure the arc)
showed the simulated car turning at roughly *twice* the kinematic radius and
losing most of its commanded speed at large steering angles:

| steering | model radius | measured |
|---|---|---|
| 0.20 rad | 1.60 m | 2.47 m |
| 0.30 rad | 1.05 m | 1.84 m |
| 0.39 rad | 0.79 m | **1.50 m** |

The planned racing line needs 1.3 m at the hairpin apexes. At a real 1.5 m
minimum radius **the course is not drivable in a single sweep**, which is
exactly why every controller — PPO policies included — wedges in the
hairpins. The policies were never failing to *decide* to turn; they were
commanding turns the vehicle model cannot execute.

Two suspects were found in `speed_course.sdf`, both real modelling errors:

1. **Wheel friction**: every wheel carries
   `<mu>50</mu><mu2>1</mu2><fdir1>0 0 1</fdir1>`. `mu=50` is ~50x rubber's
   real grip, and `fdir1` — the first friction direction — points straight
   *up*, perpendicular to the contact patch, which is meaningless for a
   wheel and leaves ODE resolving near-infinite grip against tire scrub.
2. **Drive/steer joint conflict**: the AckermannSteering plugin lists both
   front and rear wheel joints as driven. It applies one velocity per side,
   but a steered front wheel needs a different rotation rate than the rear,
   so the two fight each other. Real, but — as an A/B later showed — not the
   dominant effect; see "4WD vs rear-wheel drive" below.

### The measurements were corrupted, and one conclusion was wrong

A first pass patched the friction, saw the radius improve, then found the car
at y = -27.9 m on a course spanning +/-6.9 m and concluded the change had
destabilized the vehicle; `speed_course.sdf` was reverted on that basis.

**That conclusion was wrong.** The pose topic was carrying *four different
cars' poses in rotation* — three orphaned `gz sim` servers, 19 hours old,
survivors of the v4 training runs that repeated cleanup passes had missed
(the greps matched the launch wrappers, not a bare `gz sim`). The README
warns about exactly this: several servers on one topic "produce poses that
jump between worlds, which looks like wild physics rather than the
process-management problem it is." The y = -27.9 reading was almost certainly
another instance's car, not a launch. **Before trusting any sim measurement,
check `pgrep -af "gz sim"` returns exactly one server.**

`vehicle_calibration.py` exists so this cannot recur silently. It waits for
genuine stationarity, brakes actively (neutral is a 0.1 m/s^2 coast, ~15 s
from test speed), differences over a ~0.1 s stride so 60 Hz pose jitter is
not read as metres per second, watches z and tilt, and reports how many
samples it rejected. Several of its own bugs had to be fixed first — an
unsigned-speed brake that drove the car backwards forever, and a stationarity
threshold below the vehicle's 4 mm/sample resting jitter.

### Measured before and after, on a single clean sim

| | stock model | **calibrated** | real Slash target |
|---|---|---|---|
| full-lock radius | ~1.50 m | **0.95 m** | ~1.00 m |
| speed held through turns | lost ~75% | **1.51-1.55 m/s** (cmd 1.5) | — |
| stable at rest | no — oscillates in place | **yes** | — |
| understeer ratio | — | **1.09 (0.99 at 10 deg -> 1.24 at full lock)** | 1.2-1.4 |
| hairpin (needs 1.30 m) | not drivable | **drivable** | — |

The stock model could not even be measured at 15 deg and above: the vehicle
oscillates at rest at ~0.27 m/s apparent, never settling. That instability is
itself a symptom of `mu=50` against a stiff contact.

The calibrated model is **applied** to `speed_course.sdf` via
`vehicle_calibration.py patch`: friction `mu=1.2 / mu2=1.0` with the bogus
`fdir1` removed, chassis mass 3.5 -> 1.7 kg (2.28 kg total, matching the real
truck) with inertia recomputed, and track width 0.290 -> 0.296 m. Understeer growing with steering angle is what a real
vehicle does, and 0.95 m at full lock lands right on the ~1.00 m estimate for
a real Slash.

### 4WD vs rear-wheel drive

The first calibrated patch also dropped the front wheels from the driven
joints, on the theory that the plugin's one-velocity-per-side command made
the steered fronts fight the rears. The vehicle is a **Slash 4X4 Ultimate**,
which is four-wheel drive, so that was a fidelity regression — worth keeping
only if 4WD measurably failed. It does not:

| steering | rear-drive R (ratio) | 4WD R (ratio) |
|---|---|---|
| 10 deg | 1.82 m (0.99) | 1.93 m (1.05) |
| 15 deg | 1.24 m (1.03) | 1.31 m (1.08) |
| 20 deg | 0.98 m (1.10) | 1.03 m (1.15) |
| full lock | 0.95 m (1.24) | 0.97 m (1.27) |
| overall understeer | 1.09 | **1.14** |

Both hold 1.5 m/s through every turn with zero rejected samples. 4WD is if
anything the better model: 1.14 sits closer to the 1.2-1.4 band a real truck
shows, and 4WD understeering slightly more than rear-drive is correct
behavior. **Four-wheel drive is restored** in the committed SDF; `--rear-wheel-drive`
remains on the patch tool for a 2WD Slash or to re-isolate the effect.

The lesson for the earlier bundle: friction was the dominant error by a wide
margin, and the joint conflict — real as it is — contributes little once `mu`
is sane. Bundling four changes and validating only the bundle hid that.

### What this fixed, and what it did not

With the corrected vehicle the racer now rails the top straight at 4.0 m/s at
2-7 cm cross-track error, **carries speed through the first hairpin, and
wraps past the start of the loop** — none of which it had ever done. It still
does not complete a full lap: it wedges later in the course, roughly 24 times
per 200 s.

So the vehicle model was a real and necessary fix, but it was not the only
problem. The remaining one was in the tracker — see below.

## The tracker, finished

The telemetry named the failure precisely: the MPC commanded `v_cmd=0.59`
where the car needs about 0.9 m/s to rotate at all, sitting still at full
lock with 6 cm of cross-track error. Its cost function weights position error
heavily and has no term expressing "below this speed the vehicle cannot
turn", so falling behind in a curve made slowing down look optimal — which
made it less able to turn, which made it fall further behind. A wedge is the
fixed point of that loop.

A flat `min_speed` floor was not enough (self-test improved, laps did not).
The fix that worked is a **curvature-dependent floor**: the minimum speed
rises with how hard the path ahead turns, from 0.9 m/s on a straight to
1.4 m/s at full lock, evaluated over the whole horizon so it is up *before*
the hairpin rather than once already in it. Calibration measured a clean
1.5 m/s at every steering angle, so the turning floor is known-achievable.

Two supporting changes: recovery now hands back deliberately (clears the
stuck history, drops the MPC warm start — a plan from before the reverse —
and re-acquires the path index globally instead of from a window centred on
where the car got stuck); and `course_path.py` gained a `--min-radius` guard
so the planner cannot draw an apex tighter than the vehicle can hold. The
guard turned out not to bind on this course: the line's tightest radius is
1.31 m against a measured capability of 0.97 m, 35% margin. The plan was
never infeasible — worth knowing, since it rules the planner out as a
suspect for any future sticking.

**Result: six consecutive laps, zero recoveries, zero stuck events.**

| lap | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| time (s) | 31.22 | 30.15 | 30.25 | 30.30 | **29.95** | 30.95 |
| avg speed (m/s) | 3.53 | 3.65 | 3.64 | 3.63 | 3.68 | 3.56 |

Best lap **29.95 s against the planner's 28.4 s optimum — within 5.5%** — on
a 110.1 m loop, with a spread of about a second across six laps. An earlier
run stuck repeatedly after four clean laps; that did not reproduce, so it is
logged as unexplained rather than fixed.

## Sanity-check against the real Traxxas Slash

Published Slash 4X4 specs vs. what `speed_course.sdf` models:

| | real Slash 4X4 | SDF | verdict |
|---|---|---|---|
| wheelbase | 324 mm | 0.324 | correct |
| body L x W | 568 x 296 mm | 0.55 x 0.30 collision box | correct |
| track width | 296 mm | 0.290 | 6 mm narrow, negligible |
| mass | 2.28 kg | 4.08 kg total | **79% heavy** |
| chassis inertia | — | 0.08 / 0.12 / 0.16 | sane for that box |

Mass does not change the friction-limited cornering radius (lateral
acceleration is `mu*g` regardless), so it is a secondary issue — but it does
change contact resolution and how ESC torque becomes acceleration, and it is
worth correcting to 2.3 kg while the model is open.

**Steering angle and minimum radius.** Traxxas does not publish steering
throw or turning circle, so from the bicycle model at the confirmed 324 mm
wheelbase, with a 1.2-1.4x understeer allowance for tire slip and imperfect
Ackermann:

| steering | kinematic R | realistic R | turning circle |
|---|---|---|---|
| 22.9 deg (SDF's 0.40 rad) | 0.77 m | 0.92-1.07 m | ~2.0 m |
| 28 deg | 0.61 m | 0.73-0.85 m | ~1.6 m |
| 30 deg | 0.56 m | 0.67-0.79 m | ~1.5 m |

The hairpin apexes need **1.30 m**. So a real Slash clears them comfortably
even at the conservative 22.9 deg throw, while the simulated car's measured
1.50 m does not. That confirms the course *is* drivable and the simulator is
what is wrong — and it corrects the earlier target: the sim should reproduce
~0.9-1.1 m at full lock, not the idealized 0.77 m, because the real truck
understeers too. Worth 10 minutes with the physical car: set full lock and
measure the wheel angle with a phone inclinometer, then drive a full-lock
circle and measure its diameter. Those two numbers pin `steering_limit` and
the understeer factor exactly.

**Steering rate — and a correction.** The Traxxas 2075 servo is specified at
0.17 s/60 deg at 6V, i.e. 6.16 rad/s at the servo horn. An RC steering
linkage delivers roughly 50-75% of that at the road wheel, so **3.1-4.6
rad/s**, or lock-to-lock in about 0.22-0.26 s. `config.yaml` had
`max_steering_rate: 8.0` — raised earlier to stop the CasADi smoother
lagging the policy — which is roughly **twice what the servo can physically
deliver**. Reset to 3.5 rad/s. The original 4.0 was very nearly right; the
smoother's lag was a symptom of the vehicle-model bug, and papering over it
with an impossible slew rate was treating the symptom.

Note the sim errs the other way on rate: the AckermannSteering plugin has a
`<steering_limit>` but no steering *rate* limit, so simulated steering snaps
to angle faster than any servo could. Adding a rate limit matching the servo
would close a real sim-to-real gap.

## v5: the vehicle fix transforms RL, and exposes a new mismatch

Retrained on the corrected vehicle with the v4 recipe unchanged (reverse
gear, `max_speed` 2.0, deterministic eval selection), 250k steps over two
chunks — the sim died at 177k with `teleport failed: Service call timed out`
and `train_resilient.sh` resumed it automatically, the failure that cost v4
five hours of manual rescue.

Same algorithm, same reward, same hyperparameters; only the vehicle changed:

| | v4 (broken vehicle) | **v5 (fixed vehicle)** |
|---|---|---|
| deterministic distance | 20.5 m | **130.4 m** |
| collisions (5 episodes) | 3/5 | **0/5** |
| episode reward | ~130 | ~900 |

130.4 m exceeds the 110.1 m lap, so the policy drives more than a full lap
without touching a bale. This is the clearest confirmation that the vehicle
model was the blocker for RL as well: the policies were never failing to
decide to turn.

**But v5 is not deployable, and the reason is a mistake introduced here.**
Evaluated through the CasADi smoother it scores 44.3 m and crashes every
episode; evaluated raw it scores 130.4 m and crashes none. The cause is the
`max_steering_rate` correction from 8.0 to the servo-realistic 3.5 rad/s:
`env.py` never rate-limited steering, so the policy learned to flick the
wheels instantaneously, and the smoother then enforces a limit it has never
experienced. The same train/deploy mismatch this report has flagged twice
already — committing un-executable motion — this time introduced by making
*half* the stack more realistic.

Fixed by applying the servo slew inside `env.py`, with `env.max_steering_rate`
and `smoother.max_steering_rate` both 3.5 in `config.yaml` and a metadata
note so the limit travels with the checkpoint.

## v6, and why the smoother should not sit in front of a policy

Retrained with the servo slew enforced during training. It reached v5's peak
(130.4 m deterministic) in **57k steps rather than 164k**, despite solving the
harder rate-limited problem, and was stopped at 140k of 250k once the
evaluations plateaued at the 60 s x 2.0 m/s episode ceiling.

The decisive test — the same checkpoint with and without the smoother:

| | v5 raw | v5 smoothed | **v6 raw** | v6 smoothed |
|---|---|---|---|---|
| mean distance | 130.4 m | 44.3 m | **120.5 m** | 71.1 m |
| collisions | 0/5 | 5/5 | **1/5** | 4/5 |
| clean-episode distance | 130.4 m | — | **130.2 m** | 137.3 m |
| steer jerk | 0.329 | 0.194 | **0.320** | 0.195 |

The slew fix narrowed the gap (66% loss -> 41%) but did not close it, and the
jerk column says why. The raw policy commands 0.320 rad/tick = **3.2 rad/s of
steering against the servo's 3.5 rad/s limit** — it is already executable.
The smoother halves that to 1.95 rad/s, because it does not merely clamp: it
solves a cost-weighted optimal-control problem whose `w_steer_rate` and
`w_accel` terms lag the reference even *within* the limits. That lag is
dynamics the policy never trained against.

### Retuning the smoother, and the decision

Before dropping the smoother it was worth asking whether its tuning was the
problem rather than its presence. Its tracking weights were low relative to
its change-penalties, so it filtered when it only needed to limit; raising
tracking ~3 orders of magnitude turns it into a constraint projector that
reproduces a feasible reference and bends only an infeasible one.

| v6 best_model | raw | old smoother | **retuned smoother** |
|---|---|---|---|
| mean distance | 120.5 m | 71.1 m | **91.5 m** |
| collisions | 1/5 | 4/5 | **2/5** |
| clean-episode distance | 130.2 m | 137.3 m | **137.7 m** |
| steering rate | 3.20 rad/s | 1.95 rad/s | **2.48 rad/s** |

Retuning recovered most of the loss: +29% distance and half the collisions.
Pushing the weights a further 10x (`w_speed` 200, `w_steer` 800) changed
nothing measurable — 91.06 m against 91.49 m, identical jerk and collision
rate — which says the weights have **saturated**. The residual gap to raw is
structural, not a tuning problem: the smoother brakes across its 0.8 s
horizon where the env's training-time clamp acted greedily on the current
step. Closing it properly means training with the smoother in the loop so
the policy adapts to its dynamics (IPOPT costs ~15 ms against a 100 ms step,
so this is affordable) — deferred, as it needs a retrain.

So the smoother is **not** used in front of the policy. Even retuned as
tightly as its weights allow, it costs 24% of the distance and doubles the
collision rate, because it solves a different problem than the one the policy
was trained against. A v6-or-later policy has the actuator limits baked into
training — servo slew, traction clamp, friction circle — so its raw commands
are executable by construction, and the smoother can only subtract.

The smoother keeps its place in front of `path_racer.py`, whose reference is
a *plan* with no notion of actuator limits and which genuinely needs them
imposed. The retuned weights are kept for that use, and `--smoother` remains
available on `run_policy.py` / `evaluate.py` for pre-v6 checkpoints that were
trained without env-side limits and do need the filtering.

`evaluate.py` now defaults to raw as well, so the measured configuration is
the one that actually ships.

**Deployable configuration: `checkpoints_v6/best_model.zip`, raw.** Two
independent 5-episode runs, on a 110.1 m lap:

| run | mean distance | collisions | clean-episode distance |
|---|---|---|---|
| 1 | 120.5 m | 1/5 | 130.2 m |
| 2 | 112.1 m | 2/5 | 129.5 m |

Mean distance and collision count vary run to run — start poses are
randomized, and five episodes is a small sample — so quote this as roughly
**112-120 m with 1-2 collisions in 5**. The clean-episode figure is the
stable one (130.2 / 129.5 m): when the policy completes an episode it
reliably covers about 130 m, more than a full lap. Anyone reporting a single
number should use the range, not the better run.

## Planner/tracker vs RL, for a fastest lap

| | best lap | notes |
|---|---|---|
| planner + MPC tracker | **29.95 s** | 6 consecutive laps, 0 recoveries, 3.6 m/s avg |
| v5 RL policy | ~55 s equivalent | capped at `max_speed` 2.0, saturates the 60 s episode |

For a static course with a QuestNav pose, the planned-line racer is the
faster and more diagnosable option by a wide margin. RL's value is as a
fallback when the map or pose cannot be trusted — which argues for keeping
it trained against the ZED-style observation rather than pushing its lap time.

## Recommended next steps

1. ~~Fix the vehicle model~~ — **done and validated**: 0.95 m at full lock,
   speed held through turns, stable at rest. Confirm with
   `python vehicle_calibration.py measure` (every row should read `ok`).
   Re-measure against the physical car when convenient and re-patch with
   `--full-lock-deg` / `--turning-circle-m`; the numbers currently in the
   model are estimates from published specs plus a 1.2-1.4x understeer band.
2. **Finish the tracker** — the live blocker. The car clears the first
   hairpin and wraps the loop start, then wedges. Next diagnostics: plot
   cross-track and commanded vs measured speed against arc length to find
   where the MPC's solution stops being executable; check whether the
   `min_speed` floor needs to be curvature-dependent (higher in hairpins);
   and confirm the recovery reverse is not fighting the tracker on re-entry.
   The geometry is now known to be drivable, so this is tuning, not a
   feasibility question.
3. **Then revisit RL.** Every checkpoint to date was trained against the
   mis-modelled vehicle. Once the model is fixed, v4's recipe (reverse gear,
   `max_speed` 2.0, deterministic eval selection) is the one to retrain
   with, followed by a speed curriculum toward 4.0 via `--resume-from`.
4. **Sim longevity**: Gazebo's pose stream dies after ~5 h of continuous
   training; an auto-resume wrapper (relaunch from the latest checkpoint on
   pose timeout) would make overnight runs unattended.
5. The `--stochastic` flag on `test_policy.sh` remains the quick check for
   noise-as-brake regressions: a large stochastic-vs-deterministic spread
   means the mean action is degenerate, not that training failed.

## Known limits

- Observations still come from analytic ray-casts (now ZED-corrupted); a real
  depth pipeline is still required before this drives the physical car.
- Existing `checkpoints/final_model.zip` predates the 110° FOV change and the
  traction clamp — retrain before comparing numbers.
- Traction is modelled as command shaping; Gazebo's contact physics are not
  changed, so the sim itself will still grip better than a real floor.
