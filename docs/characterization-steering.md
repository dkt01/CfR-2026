# Steering and Turn-In — the three re-runs

Three runs that close the three lateral gaps
[characterization-results.md](characterization-results.md) left open. They are
all Session B work: one place, one afternoon, about 10 minutes of driving
between them.

| | Run | Answers | Blocked on |
| --- | --- | --- | --- |
| S1 | `step_steer_fine` | the real turn-in lag, at the commands the policy issues | nothing, but check the camera rate first |
| S2 | `steer_authority_fast` | the steering map, including its endpoints | nothing |
| S3 | `skidpad_chicane` | understeer at cornering speed, and `tire_scrub` | **A6** (below) |

Read [characterization.md](characterization.md) first for how a run works —
arming, the E-Stop interlock, the abort envelope. Nothing about that changes
here.

---

## Why these three and not a general re-run

Each one replaces a number that the model currently leans on and cannot
justify.

**S1 — turn-in lag.** `plant.yaw_response_tau` is 0.34 s and it is fitted to
ten traces on **one simulated car**. It is also the term that decides where a
policy starts turning into a hairpin, which is where the scripted baseline and
two trained policies all beached. The original `step_steer` stepped to ±0.50
at 2.0 and 3.0 m/s and returned a time constant of 1.76 s — roughly nine times
the truth, thrown out in §4 — because the response was over before the second
independent camera sample arrived. `step_steer_fine` steps to ±0.25 and ±0.35,
the band the policy actually trims in, at 2.5 m/s, which is hairpin approach
speed.

**S2 — the steering map.** Every steering command in the system goes through
`effective_angle_table`, and its endpoints (0.512 left, 0.382 right) are a
straight line drawn through the half-lock values and doubled. `max_angle_left`
and `max_angle_right` are still tagged `guess`, the table the original run
produced was non-monotonic — full right lock reading *less* angle than half
right — and §6 names the cause: it ran at 0.6 m/s, the one speed at which both
speed sensors fail at once. `delta_eff = atan(L·yaw_rate/v)` divides by that
speed, so a fabricated `v` is an error on every row.

**S3 — scrub at cornering speed.** `tire_scrub: 1.10` is a Gazebo number.
Read the blocker below before running it: on its own, S3 cannot measure scrub
at all.

---

## Two things to settle before driving

### A. A6 — the static steering map. S3 is worthless without it.

**S3 cannot measure `tire_scrub` on its own, and running it expecting a number
is the trap here.**

What a skidpad observes is the radius. What the analysis reports is
`delta_eff = atan(L/R)` — the angle a bicycle would need to hold that radius.
Scrub is precisely the gap between that and where the wheels are *actually*
pointed. An analysis that derives the angle from the radius has divided the
scrub out before it starts, and will report 1.00 whatever the tires do. That
is why `measure_turn_radius.py` works in Gazebo and does not transfer: there
the commanded wheel angle is known exactly, because the simulator was told it.

The missing half is **A6**, already in the campaign and never run: the car on a
level floor, photographed from directly overhead against a grid, steering swept
−1 to +1 in 0.1 steps, each point approached from both directions. No driving,
no E-Stop, no space, about an hour. It gives the geometric wheel angle
`delta_w(command)`, and then S3 becomes:

```
tire_scrub = tan(delta_w from A6) / tan(zero-a_y effective angle from S3)
```

where the denominator is the intercept `analyze_run.py` now prints for each
ladder — the arcs extrapolated back to zero lateral acceleration, which removes
the speed-dependent part and leaves everything that is flat in speed.

A6 is worth doing for S2 as well. It is an independent read on where the
linkage stops, and `center_offset` and `backlash` — both still `guess` — fall
out of the same photographs.

### B. Check what the camera is actually publishing

This is the whole of S1. On the car:

```bash
ros2 topic hz /zed/zed_node/pose
ros2 topic hz /zed/zed_node/odom
ros2 topic hz /zed/zed_node/imu/data
ros2 param list /zed/zed_node | grep -i -e rate -e resolution
```

**Rate matters less than which estimator you use, and that is worth knowing
before you spend an afternoon on it.** Planting a known 0.34 s lag, sampling
it, and putting it back through `fit_yaw_lag.py`:

```
  pose Hz   hold 1.4 s   hold 2.4 s   hold 3.0 s        steps (15 Hz, 2.4 s hold)
       10         0.34         0.34         0.35          1 ->  0.33 s
       15         0.34         0.34         0.35          2 ->  0.34 s
       30         0.35         0.34         0.34          4 ->  0.34 s
       60         0.34         0.34         0.34          8 ->  0.34 s
      200         0.35         0.34         0.34         12 ->  0.34 s
```

It recovers the planted value from 10 Hz up, because it replays the command
schedule through `plant.py` and scores the whole transient — it never has to
resolve the knee. What produced the 1.76 s was a different estimator:
`analyze_run.py`'s `fit_first_order`, a log-linearisation over the 10–90% band,
which does need distinct samples in exactly that knee and had four of them.
The original run was not beaten by its camera alone; it was beaten by reading
that camera with the wrong tool, and §4 caught it only afterwards.

So: **record at the highest rate you can get, but do not skip the run over it.**
The rate buys three things the table above cannot show — the per-trace spread
that is the only honest error bar, the check that the response is
first-order-*shaped* at all rather than fitted as though it were, and margin
against a real car that need not match the model's structure the way a planted
one does. It is also free. What the table does say is that if the camera turns
out to be stuck at 15 Hz, the run is still worth driving.

**Raising `control_rate_hz` does not help** — `telemetry.csv` resamples
whatever message arrived last, so a faster grid over a slow camera is more rows
carrying the same values. The rate that matters is the camera's:

```bash
ros2 param get /zed/zed_node general.grab_frame_rate     # names vary by wrapper
ros2 param get /zed/zed_node general.pub_frame_rate      # revision - check both
```

The ZED 2i runs 60 fps at HD720 and 100 at VGA, against 30 at HD1080. Drop the
resolution if that is what it takes; a step-steer run needs pose timing, not
pixels. Put the settings in a YAML and pass it:

```bash
ros2 launch cfr_arduino_bridge characterize.launch.py \
    profile:=step_steer_fine zed_params:=/path/to/fast.yaml
```

**Do not pass `zed/config/cfr_zed2i.yaml` for S1.** That is the race
configuration and it turns on `area_memory`, so the SDK rewrites
`/zed/zed_node/pose` when it closes a loop — and a closure inside a 2.4 s step
lands as a jump in yaw that a derivative reads as the car snapping sideways.
`~/odom` is raw visual odometry and is never corrected, which is why it is what
the fit runs on and `~/pose` is the cross-check.

**What the runner now records.** Alongside the 50 Hz `telemetry.csv`, a run
directory carries `pose.csv` (one row per message from both `~/odom` and
`~/pose`, with the message's own header stamp) and `imu.csv` (one row per IMU
message; gyro z is a direct yaw-rate measurement at 200–400 Hz, the highest
resolution on the car and the best answer to "when did it start rotating").
`telemetry.csv` gained an `odom_stamp` column so a repeated sample is visible
as a repeat. **The runner prints the rate it actually got when the run ends,
and warns below 25 Hz.** Read that line before packing up — a second run costs
ten minutes, a second trip costs a day.

---

## The runs

Rehearse the whole sequence against Gazebo first. It catches a typo in a step
list, a limit sized wrong, an argument that does not do what it says:

```bash
ros2 launch cfr_arduino_bridge characterize.launch.py \
    profile:=step_steer_fine use_sim:=true require_estop_cycle:=false
```

Never pass `require_estop_cycle:=false` on the real car.

### Space

`max_distance` is the radius of a clear bubble around the arm point, not path
length. Every one of these steers both ways, so both sides have to be clear.

| Run | Bubble | Nominal peak | Driving time |
| --- | --- | --- | --- |
| `step_steer_fine` | 24 m | 14 m, or 21 m if full lock is really 0.40 rad | 47 s |
| `steer_authority_fast` | 26 m | 17 m, or 24 m if full lock is really 0.40 rad | 183 s |
| `skidpad_chicane` | 20 m | 15 m | 148 s |

The second number for S1 and S2 is the point of running them: the bubble is
sized for the pessimistic map, because the map is what is being measured.

### Order

```bash
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=zed_static
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=steer_authority_fast
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=skidpad_chicane
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=step_steer_fine label:=a
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=step_steer_fine label:=b
```

`zed_static` first, while the cones go out — it never arms, so leave the E-Stop
asserted. It is 5 minutes for a fresh read on the camera's noise floor on
today's surface, and every number below is divided by something it measures.

S2 before S3 because S3's ladders are chosen around the map S2 produces, and
because S2 is the one that cannot be rescued by reanalysis.

S1 twice. It is 47 s of driving, the per-trace spread across repeats is the
only honest error bar on a fitted lag, and `fit_yaw_lag.py` reports it.

### Between runs

Read pack volts off the E-Stop TUI before and after each one and write them on
the field card. The steering asymmetry is a ~34% effect and a sagging pack
moves the servo; S2 alternates left and right for exactly that reason, but the
record is what lets it be checked rather than assumed.

---

## Analysis

```bash
./scripts/sync_runs.sh
./scripts/analyze_run.py runs/<run> --mass 4.7 --speed-source wheel_rpm
```

**Pass `--speed-source wheel_rpm` explicitly for S2 and S3.** The default
`auto` keeps ZED odometry unless it contains physically impossible speeds,
which is the right rule at 3 m/s and the wrong one at 1.5: below about 1 m/s
"physically possible" stops meaning "right". The tachometer is reliable above
1 m/s and every arc in both profiles is. Run it both ways and compare — if they
disagree, the ZED is wrong.

The lateral analyzers changed with these profiles:

- both now take speed through `_speed_column` instead of reading `odom_vx`
  directly, which is the channel §0 found fabricating 4.6% of its samples in
  motion and the reason the first map came out non-monotonic;
- `analyze_steer_authority` **says so** when the map it produced is
  non-monotonic, rather than leaving it to be noticed in the plot two days
  later;
- `analyze_skidpad` groups by (side, command) rather than by side — the fit's
  intercept *is* the kinematic angle for that command, so pooling two commands
  asks one intercept to describe both — reports that intercept for the scrub
  calculation, and warns when an arc reached the grip limit and does not belong
  in a gradient;
- `analyze_skidpad` **corrects the sign of the understeer gradient.** The
  standard form is `delta_wheel = L/R + K·a_y` and `fits.fit_understeer` fits
  exactly that, but what it was handed is `delta_eff = atan(L/R)` — the other
  side of the same equation. At fixed command an understeering car runs wider
  as it speeds up, so `delta_eff` *falls* and the fitted slope is −K. Feeding a
  car with a true K of +0.007 through the old path returns −0.0068. The +0.007
  that reached §4 had its sign restored by hand; the value
  `vehicle_patch.yaml` carried did not;
- `analyze_step_steer` reports how many *distinct* camera samples each step
  got, from `odom_stamp`, refuses to fit below 25 Hz, and refuses a fitted tau
  longer than a fifth of its own hold — a first-order lag that has not
  substantially completed inside the step is a ramp being read as an
  exponential, which is the 1.76 s artifact's actual mechanism and survives any
  sample rate.

### S1 into `yaw_response_tau`

`analyze_run.py`'s first-order fit is a sanity check, not the answer. The
answer comes from the same fitter the Gazebo number came from, so the two are
comparable:

```bash
cd rl/formulaOne
python3 npz_from_run.py ~/runs/<step_steer_fine run> --out /tmp/car --source odom
python3 fit_yaw_lag.py /tmp/car/*.npz
python3 npz_from_run.py ~/runs/<run> --out /tmp/car_imu --source imu   # cross-check
```

`fit_yaw_lag.py` scores three explanations against each other — no lag, lag on
the servo, lag on the chassis — and prints the per-trace spread. Take the
spread seriously: if the per-trace optima are scattered, a single first-order
lag is the wrong *shape* and no value of it is the answer.

Then `randomize.yaw_tau_scale` has to move with it. It is currently
[0.6, 1.6] around a Gazebo fit, deliberately wider than that fit's own
confidence because ten traces on one simulated car is not a survey. Measured
traces on the real car are a better centre and a reason to narrow it —
but only as far as the real spread, and the argument in `config.yaml` for why
an over-wide range costs lap time for nothing still holds.

### S2 into the steering map

`steering.effective_angle_table` in `vehicle.yaml` and the copies in
`arduino_bridge.yaml` (two of them) and `rl/formulaOne/config.yaml`.
`scripts/check_steering_consistency.py` keeps them in step — run it.

Three things to check before anything is written down:

1. **Monotonic.** The analyzer now says if it is not. A map that is not
   monotonic is not a map.
2. **Where it stops.** If 0.85 and 1.00 come back equal, the linkage is
   against its stop at 0.85 and the top 15% of command is dead. The simulated
   car does exactly that and nothing has ever confirmed it on the real one.
3. **Whether the 2.0 m/s pass sits on the 1.5 m/s line.** If it sits below,
   what is being measured is understeer and belongs in S3's gradient instead.

Only then promote `max_angle_left` / `max_angle_right` from `guess`. Note that
`generate_vehicle_model.py` builds the Gazebo world's `<steering_limit>` from
those values, so they are not documentation — changing them changes what the
simulated car can do, and `randomize.steer_limit: [0.38, 0.55]` exists only
because nobody knew which end was real.

### S3 into `tire_scrub` and the gradient

Per side and per command:

```
tire_scrub = tan(delta_w from A6) / tan(intercept from analyze_run.py)
```

Expect the two commands on one side to agree. If they do not, scrub is not a
flat multiplier on this car and `plant.py`'s single `tire_scrub` is the wrong
shape — which is worth knowing, since it is applied as a second multiplier on
the effective wheelbase exactly the way the `K·v²` term is.

### While you are in there: `tire_scrub` is currently double-counted

Not a result of these runs — a consequence of what they measure, and worth
deciding before the numbers land.

`plant.py` computes the radius as

```
R_model = (L + K·v²) · tire_scrub / tan( steering_angle_points(command) )
```

and `steering_angle_points` is `[-0.382, -0.191, 0, 0.256, 0.512]`, whose
measured entries came from §4 — that is, from `atan(L/R)` on the car's own
skidpad arcs. So `tan(table)` is `L/R_car` by construction, and

```
R_model = tire_scrub · R_car
```

With `tire_scrub: 1.10`, **`plant.py` models a car that turns 10% wider than
the one the table was measured on.** The scrub is in the table already; the
multiplier applies it a second time.

This is defensible as things stand, because 1.10 was fitted so that `plant.py`
matches *Gazebo*, which is where policies get validated — and because the
table's ±1.0 endpoints are extrapolated anyway, so it has never been a
description of the real car. It stops being defensible the moment S2 supplies
real endpoints.

A6 + S3 is what makes the two terms mean what their names say: geometric wheel
angles from A6 in `steering_angle_points`, measured scrub in `tire_scrub`.
Then the table is the linkage, the multiplier is the tires, `scrub_scale`
randomises across the gap between the car and Gazebo instead of hiding it, and
the answer to "does the real car scrub like the simulated one" is a number
rather than a category error. Decide which of the two you want before writing
anything into `config.yaml`, because applying S2 and S3 results to the current
arrangement without deciding gets you neither.

### Back to the gradient

Drop the 2.4 m/s rungs from the gradient fit. They are the deliberate grip
probe and the analyzer will say so. They are not wasted: the peak lateral
acceleration any of them reaches is a free lower bound on `mu_lateral`, which
is currently ≥0.55 g from a sweep that stayed below the limit on purpose.

Do **not** run `apply_vehicle_patch.py` blindly, for the reason §8 already
gives and for the sign issue above in any run analyzed before today.

---

## What these three still do not give you

- **Pose drift**, and it is deliberate. `pose_drift_m_per_lap: [0.0, 0.12]` is
  an assumption, and `measure_feasibility.py` found drift the single dominant
  limiter on the speed/reliability frontier at −0.50 correlation, nearly double
  the next axis. It is still not worth measuring yet: the drift on a temporary
  mount is not the drift you will have, so it waits for the ZED to be where it
  will live. `zed_static` at the top of each session is the cheap standing
  proxy until then.
- **Inertia.** S1 validates `izz` in the loop rather than measuring it; A3
  stays `estimated` whatever comes back.
- **Anything at race speed.** Every arc here is capped under 0.45 g against a
  0.55 g measured lower bound, on purpose. A hairpin at 2.5 m/s and 1.48 m is
  0.43 g — the car races at the edge of the only grip number anyone has, and
  raising that bound needs a run that is willing to slide.
- **The real ZED in motion.** §0's 25 m pose jumps are still unmodelled
  anywhere, and the simulator remains optimistic about odometry in exactly the
  way that cost the first campaign its results.
