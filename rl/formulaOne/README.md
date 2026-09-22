# formulaOne — two laps of the Speed Course, as fast as the rules allow

A PPO policy that drives the Gazebo Speed Course and the real Slash from the
same file. It trains against a numpy model of the measured car (~15k
steps/s), not against Gazebo (~10 steps/s), and deploys as a 35→128→128→2 MLP
that runs on the Orin with numpy alone — **no torch on the car**.

| | |
|---|---|
| Course | 110.2 m closed loop, 0.92 m corridor, two 1.5 m hairpins |
| Target | 2 laps, no contact, no grazing, zero cross-track error, smooth steering, **then stop** |
| Speed cap | **2.5 m/s** in the hairpins, **5.2 m/s** everywhere else |
| Coast-feasible optimum (braking only) | 55.5 s for two laps |
| Scripted baseline (the floor) | 66.5 s race · 0.137 m clearance · 0.031 m mean CTE · 19% under randomisation |

**The car has no brakes.** Deceleration is coast drag only — 9.3 m of track to
go from 5.2 to 2.5 m/s. Every hairpin has to be set up nearly ten metres
early. That, not cornering, is what the policy is really learning.

**And it does not start turning when you turn the wheel.** `yaw_response_tau`
is 0.34 s — measured and fitted, see **The chassis lag** below. At 5 m/s that
is 1.7 m of travel still going straight, inside a
0.92 m corridor. Together the two make this a course about *anticipation*:
every input has to be right about where the car will be half a second from
now.

## Speed limits are structural, not a penalty

The throttle action is a *fraction of the local cap*, so no action — in or out
of distribution — can command more than 2.5 m/s in a hairpin or 5.2 m/s on a
straight. The node clamps it again on the last line before the wire. The
overspeed cost below is therefore about *arriving* too fast, which is a
braking problem, decided ten metres earlier.

## Rewards and costs

Rate terms are per second, scaled by the control period, so changing
`control_hz` does not silently reweight anything.

| Term | Weight | Applies | For |
|---|---|---|---|
| progress | **+6.0** / m | centerline advanced | *as fast as possible* |
| time | −4.0 / s | while racing; off during the stop | *as fast as possible* |
| overspeed | −20.0 / (m/s) − 15.0 / (m/s)² | **above** `progress`, so overspeed never pays for itself | rules |
| graze | −15.0 / s at zero clearance | quadratic inside 0.12 m of body clearance | *without touching the bales* |
| crash | −40.0 | terminal. Modest on purpose: two clean laps are worth ~1400, so a crash halfway already forfeits ~700 of future progress — that opportunity cost is the real deterrent | *without touching the bales* |
| **lateral** | **−9.0 / s at 0.20 m** | quadratic in cross-track error | *cross-track error to be zero* |
| **steer rate** | −0.04 / (Δcmd)²·s | first difference of the command | *steering should not be jittery* |
| **steer jerk** | **−0.15 / (Δ²cmd)²·s** | second difference — the sign changes, which is what jitter *is* | *steering should not be jittery* |
| lap | +25.0 | per completed lap | |
| **lap improvement** | **+2.0 / s faster than the previous lap**, capped at 6 s | paid on the lap that beat the one before it | *more reward if it beats its previous lap times* |
| finish | +60.0 | both laps done — crossing the line, not ending the run | |
| **stop** | **+80.0** | at rest, still inside the corridor | *stop after two complete laps* |
| stall | −20.0 | terminal, below 0.25 m/s for 2.5 s while racing | |

All of it is in `config.yaml` under `reward:`, computed in `reward.py`.

Three of those are worth more than their weight:

**Stopping is a phase, not a flag.** Crossing the line at 5 m/s with no brakes
leaves ~15 m of coasting, all of it inside a 0.92 m corridor. So the run does
not end at the line: the throttle is forced to zero, the steering stays live,
and the episode ends when the car is *at rest*. Progress and time stop paying
during it, so nothing about the stop distorts the race that preceded it. (The
node does the same thing on the car — it used to cut steering at the line,
which coasts straight into the bales it was about to turn away from.)

**Jitter is the second difference.** A penalty on Δcmd alone cannot tell sawing
from turning: hard into a hairpin at 20 Hz is a long run of same-signed steps
and scores exactly like the same steps alternating. Jitter is a *sign change*,
so Δ²cmd carries the larger weight and Δcmd stays only to stop the policy
answering it with a permanent fast ramp.

Its *weight* is set by a different argument, and `reward_probe.py` is what
found it. The sampled command is `ff + 0.5·(μ + σ·ε)` with σ ≈ 0.3, so ε alone
contributes `E[jerk²] = 6·(0.5σ)² = 0.135` every tick whatever the policy
does. This term started at 0.60, which taxes that noise at **32 reward/second
— more than progress pays** — and PPO's only reply is to drive the steering σ
to zero. That exact failure has happened here before (σ 0.208 → 0.146, field
finish 77% → 41%). What survives the noise is the part that matters:
`E[jerk²]` separates into `(the policy's own jerk)² + 1.5σ²`, and the second
term does not depend on the mean — so the gradient shaping the *deployed*
(deterministic) policy is present at any weight, and the weight only decides
how hard the term also squeezes exploration. 0.15 costs ~8/s against
progress's ~18/s.

**Beating the previous lap has to stay cheap.** The bonus pays for
(previous lap − this lap), and the cheapest way to make that number big is to
throw away the *first* lap. At +2.0/s against a time penalty of −4.0/s, giving
away a second of lap 1 costs 4.0 and buys 2.0 — the exploit always loses, and
only a genuinely quicker second lap is worth anything. The policy can see its
own pace (`observation.py` feeds it seconds up or down on the previous lap),
because a reward for something the policy cannot observe is not a reward, it
is noise.

## Run it

```bash
./setup.sh                              # venv + CPU torch + self-test, once
./train.sh --dir runs/v1                # train, export, score, plot
./validate.sh --policy runs/v1/policy.npz    # Gazebo + RViz
```

`train.sh` leaves `runs/v1/policy.npz` (the deployment artefact, ~168 KB) and
`runs/v1/report.png` (line, speed against the cap, clearance).

The same budget inside Gazebo is about two weeks. If you already have a venv
with torch and stable-baselines3, point at it with `VENV=/path/to/.venv
./train.sh` instead of running `setup.sh`.

**After changing anything in `plant.py`, in this order:**

```bash
python3 selftest.py                     # does the model still agree with the car
python3 sweep_prior.py --write          # the prior's gains are plant-dependent
python3 reward_probe.py                 # are the terms still worth what you think
./train.sh --dir runs/v2
```

Skipping the middle two is how you spend half an hour of training on a policy
fighting a prior that was tuned for a different car.

### Validate a checkpoint — this is the one that launches RViz

```bash
./validate.sh --policy runs/v1/policy.npz   # Gazebo speed course + RViz
./validate.sh --baseline                    # scripted driver, no checkpoint needed
./validate.sh --loopback --baseline         # no Gazebo either: ROS plumbing only
./validate.sh --gui                         # Gazebo's own window as well
./validate.sh --sensors                     # start on the real visual signal
```

RViz shows the centerline, the speed cap coloured red→green, the car, and a
live readout of `speed / cap`, clearance and lap.

> Runs on `ROS_DOMAIN_ID=77` and `GZ_PARTITION=formula_one` by default. Another
> simulator on the machine publishes the same topics and two stacks that can
> see each other interleave **silently** — this bit us during development.
>
> **`teleport_api` is NOT domain-scoped.** It binds a fixed port (9003) over
> HTTP, which `ROS_DOMAIN_ID` and `GZ_PARTITION` do not isolate, so two
> simulations on one machine share it and a teleport meant for yours will move
> the other one's car. `measure_turn_radius.py` refuses to post to it until it
> has seen a pose on its own domain; anything else that teleports should do the
> same, or set `CFR_TELEPORT_PORT`.
>
> **Kill process GROUPS, not processes.** `ros2 launch` forks Gazebo, the
> bridge, `sim_vehicle`, `lap_counter` and the randomizer. Killing the launch
> alone orphans them and they keep running — a second `sim_vehicle` then
> consumes `/drive_cmd` and drives `/sim/cmd_vel` alongside the first, into one
> Gazebo, with no error anywhere. `validate.sh` starts children with `setsid`
> and signals the group; check with `ros2 node list --no-daemon` (the daemon
> caches dead nodes).

### Score without the simulator

```bash
python3 evaluate.py runs/v1/policy.npz --plot report.png
python3 evaluate.py --baseline                # the floor to beat
python3 selftest.py                           # track + plant + env, ~5 s
python3 node_selftest.py                      # the ROS node, closed loop, ~60 s
```

`evaluate.py` prints two rows. **`field`** is the one that predicts the day:
256 runs with dead time, drag, steering authority, slew, **chassis lag** and
localisation all redrawn. A policy that does not comfortably beat the
scripted baseline there has not earned a run on the car.

Note what `finish` counts: two laps **and at rest**. A car still doing 5 m/s
as it crosses the line has not completed a run on a course with no run-off,
and counting it as one is how you ship a policy that cannot stop.

The gap between those two rows is the point of the whole design. The scripted
baseline is a fixed set of gains and a fixed speed profile: it is tuned for
one car and falls apart on a redrawn one, and it has to give away a sixth of
its speed everywhere to survive the chassis lag at the two or three places
that lag actually bites. A policy sees the observation and can spend that
margin where it is needed instead of everywhere.

## The chassis lag: what was missing

This is the one thing that had kept every policy trained here from finishing
in Gazebo, and it is worth reading before changing anything in `plant.py`.

**The symptom.** Neither the trained policy nor the *scripted* baseline could
get round in Gazebo, while both completed two clean laps against the model.
Both beached entering the same hairpin. A driver with no learned component
failing identically in both directions of a fix is not a training problem, so
the model was wrong — the question was where.

**What was already modelled, and was right.** `measure_turn_radius.py`
(teleport to open ground, drive a constant command, fit a circle) found the
simulated car turning a uniform **1.10x** wider than a kinematic bicycle
predicts, flat across both directions and 1.5–3.0 m/s. That is `tire_scrub`,
and a re-measurement confirms it: steady-state yaw rate now matches Gazebo
within 2%.

**What was missing.** Matching the radius a car *settles* into says nothing
about how it *gets* there. `measure_step_steer.py` steps the steering command
from straight-line travel and records the yaw-rate transient:

| speed | cmd | Gazebo t₅₀ | old model t₅₀ | extra delay |
|---|---|---|---|---|
| 2.5 m/s | +0.55 | 0.59 s | 0.37 s | **+0.22 s** |
| 2.5 m/s | −0.55 | 0.67 s | 0.32 s | **+0.34 s** |
| 2.5 m/s | −0.30 | 0.71 s | 0.26 s | **+0.45 s** |
| 4.5 m/s | −0.20 | 0.81 s | 0.24 s | **+0.57 s** |
| 1.5 m/s | +0.55 | 0.60 s | 0.37 s | **+0.23 s** |

Gazebo takes roughly twice as long to build yaw rate as the model did, at
every speed tested. **Speed-independent**, so it is not tire relaxation length
(σ/v); it is the yaw inertia and tire force build-up of a car that has to be
pushed into rotating.

**The fit.** `fit_yaw_lag.py` replays each recorded command schedule through
`plant.py` over a grid of time constants and scores mean *relative* yaw-rate
error (relative, so one step to full lock does not buy the fit — see the note
in the file):

| model | error |
|---|---|
| no lag (what the model was) | 0.236 |
| blame the servo (`steering_tau` 0.16) | 0.161 |
| **blame the chassis (`yaw_response_tau` 0.34)** | **0.115** |

A first-order lag on the yaw rate halves the error, and beats putting the same
delay in the servo — where it would have been cheaper, since the servo was
already in the model. Per-trace optima run 0.19–0.59 s, which is what
`yaw_tau_scale: [0.55, 1.75]` randomises over.

**The confirmation, which is the part that matters.** With 0.34 s in
`plant.py`, the scripted baseline — no learned component, nothing retrained —
**beaches at ~21 m against the model**, which is within a metre or two of where
that same driver has always beached in Gazebo (19.8 m, 19.95 m, 25 m across
three runs). The offline model reproduces the real failure for the first time.
That is the gap closing: not a policy that scores better, a *model that is
wrong in the same way the simulator is*.

**What it cost everywhere else.** A 0.34 s lag invalidates anything tuned
without it:

* The steering prior extrapolated heading at a *constant* yaw rate. Through a
  chicane that predicts the car still rotating left while its wheels have
  already gone hard right, so the feedback demands the opposite lock, and it
  saturated for eight ticks together. It now relaxes toward the yaw rate the
  command already in flight implies (a Smith predictor — the command was
  issued a tick ago, so using it is not an algebraic loop), and its
  feedforward carries a `lead_scale` share of the exact lag inverse,
  `k + tau*v*dk/ds`.
* The prior's gains were re-tuned from scratch — `sweep_prior.py`, 432
  settings.
* The coast-feasible profile is feasible for *braking* and not for *turning*:
  a car that cannot start rotating for 0.34 s cannot hold it. The scripted
  floor now drives `baseline_speed_scale` of it, and the measured trade is
  tabulated in `BaselineDriver`.

**Still open:** the model now predicts the Gazebo failure, and the policy
trained against it has to be run in Gazebo to confirm it predicts the
*success* too. `./validate.sh` is that test.

## On the car

```bash
# use_cmd_vel:=false matters -- cmd_vel_to_drive republishes DriveCommand on a
# timer whether or not anything is feeding it, so leaving it up puts a second
# publisher on /drive_cmd and the Arduino acts on whichever arrived last.
ros2 launch cfr_arduino_bridge arduino_bridge.launch.py use_cmd_vel:=false
ros2 launch rl/formulaOne/formula_one.launch.py \
     policy:=runs/v1/policy.npz use_sim_time:=false rviz:=false
```

Same node, same topics, same `policy.npz` as the simulator. Three things make
that true:

- **It publishes `DriveCommand`, not a `Twist`.** `cmd_vel_to_drive` divides by
  a symmetric ±0.40 rad steering limit, but the measured car reaches 0.512 rad
  left and 0.382 rad right. Going through it throws away a third of the lock
  one way and invents it the other.
- **It anchors on the start signal.** The ZED's map origin is wherever the
  camera booted, so on `go` the node latches the pose it is sitting at and
  pins it to the known start pose — the same trick `lap_counter_node` uses.
- **It trained on error, not on truth.** Every episode redraws dead time,
  drag, acceleration, slew, steering gain/asymmetry/trim, understeer, **chassis
  yaw lag**, pose noise, latency, drift, start offset, control jitter and
  command dropout (`randomize:` in `config.yaml`). The policy only ever sees a
  delayed, noisy, drifting pose and a tachometer that is blind below 0.3 m/s.
- **It stops the same way it was trained to.** At the lap target the node
  drops the throttle to zero and *keeps steering* until the car is at rest,
  then logs the race time and the distance past the line. It used to set a
  flag and command neutral on both channels, which on a car with no brakes
  coasts fifteen metres with the wheels straight — into whatever the car was
  about to turn away from.

Everything under `plant:` is the real car, from `config/vehicle.yaml` and
`config/arduino_bridge.yaml`. Change it there first.

## Files

| | |
|---|---|
| `config.yaml` | every knob: track, plant, reward, randomisation, PPO |
| `track.py` | centerline, speed cap, bale distance field (cached) |
| `plant.py` | batched model of bridge + Arduino + ESC + car |
| `env.py` | the vectorised env; true pose for scoring, sensed pose for the policy |
| `reward.py` · `observation.py` | the cost terms; the one observation builder both sides import |
| `baseline.py` | scripted driver — the floor, and the fallback that needs no checkpoint |
| `train.py` · `evaluate.py` · `export_policy.py` | PPO; offline scoring; SB3 → `.npz` (the export checks itself against torch and refuses to write a mismatch) |
| `formula_one_node.py` | the ROS driver, identical in Gazebo and on the car |
| `ros_loopback.py` | a car on ROS topics without Gazebo, for testing the node |
| `selftest.py` · `node_selftest.py` | run both before a training run and before the car |
| `measure_turn_radius.py` · `measure_step_steer.py` | the two Gazebo experiments the plant is fitted to: the radius it settles into, and how long it takes to get there |
| `fit_yaw_lag.py` | turns the step-steer traces into `yaw_response_tau`, and scores the alternative explanations beside it |
| `sweep_prior.py` | re-tunes the steering prior's five gains against the model; the prior is most of the steering, so this is not optional after a plant change |
| `reward_probe.py` | what every reward term is actually worth over a real run, and at the four states that decide a lap |
