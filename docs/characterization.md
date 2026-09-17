# Robot Characterization

How to measure the car so the Gazebo simulation can be a twin of it rather than
a plausible-looking stand-in.

Everything here runs offline. The car records locally, the analysis needs
nothing but a stock Python 3, and no step requires reflashing the Arduino.
Print [field-card.md](field-card.md) and take it with you.

---

## Why

Closed-loop velocity control was tuned **with the car on blocks**. The simulator
runs the real autonomy stack, but the car it simulates is largely invented:

| What the simulator believes | Where it came from |
| --- | --- |
| 4.08 kg, inertia 0.08/0.12/0.16 | round numbers |
| centre of mass at ground level | no `<inertial><pose>` was ever set |
| `mu = 50` on the ground and every wheel | tire slip is physically impossible |
| full lock is 0.40 rad | **never measured, and it scales every curve the car drives** |
| coast decelerates at 0.1 m/s² forward, 0.3 reverse | a fudge for a Gazebo contact artifact, per the config's own comment |
| speed reaches its target in one tick | no ESC deadband, no lag, no controller dynamics |
| wheels are rigidly bolted to the chassis | no suspension DOF at all - a wheel follows the ground exactly, which matters everywhere but is a lie on the obstacle course's potholes, gravel and ramps |

`config/vehicle.yaml` now holds all of it in one place, and every entry carries a
`provenance` tag: `measured`, `estimated`, or `guess`. The campaign's job is to
turn `guess` into `measured`. Run `scripts/generate_vehicle_model.py` at any time
to see what is left.

---

## Before you start

Build once, with a network:

```bash
./scripts/build.sh --test
```

Check the E-Stop, the XBee pair and the pack. You do **not** need the laptop
networked after this point.

### How a run works

```bash
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=coastdown
```

That is the whole interface. The launch picks a run directory, points the
Arduino serial traces into it, brings up the bridge and the ZED, and starts the
runner. Then:

1. The runner waits for the Arduino link.
2. **It asks you to assert E-Stop.** Press the button.
3. **Then clear it, with Auto Arm set.** That is the start trigger.
4. Three second countdown, then it drives the profile.
5. Straight-line profiles reverse themselves back to where they started.
6. It stops, writes the result, and the launch exits.

The arming sequence is deliberately awkward. It can only be completed by someone
holding a working, connected E-Stop, which is exactly the precondition worth
enforcing before a car drives itself.

**To abort at any point: assert E-Stop.** The runner notices, commands zero, and
closes the run cleanly. The E-Stop is the safety device; the runner has no
safety authority of its own and is subject to it, to the Arduino's 200 ms
watchdog, and to the `AUTO_ARMED -> AUTO_ACTIVE` handshake like any other client.

### When Wi-Fi drops

It will, at range. It does not matter. The run is autonomous once armed and
records to the Jetson. Watch the **E-Stop TUI** instead of the laptop — the XBee
link is independent of Wi-Fi and shows mode, pack voltage, spur RPM and speed at
20 Hz. Read pack volts off it before and after each run.

### Safety envelope

Every profile declares its own limits, and the runner aborts on any of them:

- `max_distance` from the arm point
- `max_duration`
- E-Stop, lost Arduino link, leaving `AUTO_ACTIVE`, or stale odometry

An abort is graceful: it commands zero and writes the run out. A partial run is
still analysable and the report says it was partial.

---

## Session A — Static (garage, ~2 h, no driving)

No E-Stop, no space, no launch file. Write results straight into
`config/vehicle.yaml` and set each `provenance` to `measured`.

### A1 — Mass and horizontal centre of mass

Car race-ready: pack, Jetson, ZED, Arduino, all wiring.

1. Total mass on the scale.
2. Front axle and rear axle separately, car level, both axles at the same
   height. `W_front + W_rear` must come back to the total — if it doesn't, find
   out why before going further. In the vehicle frame (origin midway between
   axles, x forward): `cg_x = wheelbase * (W_front / W_total - 0.5)`.
3. Left and right sides for `cg_y = track * (W_left / W_total - 0.5)`.

### A2 — Centre of mass height

**Block the shocks solid first** (zip-ties or spacers). Suspension travel during
the tilt biases the answer, and this is the whole reason the measurement is
fiddly enough to be worth doing carefully.

Raise the rear axle by `H` (about 0.15 m), keep the front on the scale, re-weigh:

```
tan(theta) = H / sqrt(wheelbase^2 - H^2)
cg_z = wheel_radius + (dW_front * wheelbase) / (W_total * tan(theta))
```

Scale readings can be in kg throughout; `g` cancels.

### A3 — Inertia (estimated, not measured)

A bifilar pendulum was considered and rejected as too fiddly for the value. Build
a parts list instead — chassis and electronics, pack, Jetson, ZED, motor, wheels
— with each mass and its position in the vehicle frame, and sum the
contributions. Keep the working; put the result in `vehicle.yaml` tagged
`estimated`.

`izz` is the one that matters and it is **validated in the loop by B3**, not
here. If simulated yaw response later disagrees with the car, come back to this
before blaming the tires.

### A4 — Geometry

Wheelbase, front track, rear track (they may differ), ride height, chassis
envelope, ZED and Jetson mount positions, and the mass of one wheel-and-hub.

### A5 — Effective rolling radius

Not a 20 m drive — no long marked course needed.

At race weight, mark the tire and the ground, roll the car **exactly ten wheel
revolutions** in a straight line, and tape the distance (~3.6 m).
`tire.diameter = distance / (10 * pi)`. About 1% accuracy in ten minutes.

Then pin the ratio independently: rotate the spur gear 28.5 turns by hand and
confirm the wheels turn 10 times.

> This calibrates the **product** `tire_diameter / spur_to_wheel_ratio`, which is
> exactly what the code uses. The two cannot be separated by this test and do not
> need to be — the hand count fixes the ratio, which fixes the diameter.

### A6 — Static steering map

Diagnostic for backlash and Ackermann error; **B1 is the authority** on the angle
the car actually achieves.

Bench only, wheels clear or on a smooth floor:

```bash
ros2 launch cfr_arduino_bridge arduino_bridge.launch.py use_cmd_vel:=false
ros2 param set /arduino_bridge require_auto_active false   # BENCH ONLY
ros2 topic pub -r 20 /drive_cmd cfr_interfaces/msg/DriveCommand \
  '{auto_ready: true, steering: 0.5, velocity: 0.0}'
```

Photograph the front wheels from directly overhead against a grid, sweeping
steering from −1 to +1 in 0.1 steps. **Approach each point from both
directions** — the difference is backlash, and it is invisible if you only sweep
one way.

Record: the command that produces zero angle (`center_offset`), the hysteresis
(`backlash`), and the left/right difference at matched commands (Ackermann
error). Note that the firmware's auto-mode centre is 1504 µs, not 1500.

### A7 — Servo slew rate

Film one full-lock-to-full-lock step at 240 fps and count frames. Once is enough:
the servo runs on regulated 5 V, so this does not vary with pack state.

Also film a small step for the time constant.

### A8 — Suspension

**Do this after A1** (needs corner weight) and with the car at full race weight,
tires on. Every shock is stock, collars run to **maximum preload** to hold ride
height under the electronics payload - confirm that is still true before
measuring anything (a collar can walk loose).

1. **Confirm preload.** Photograph all four collars at their topmost
   (stiffest) thread position. This is `suspension.preload`, and it is the one
   entry in that section that gets `measured` for free - it is a setting, not
   something inferred from a reading.
2. **Static sag.** With the car resting normally, mark the shock shaft at the
   collar (a rubber O-ring works). Lift the corner until the wheel just clears
   the ground (unloaded) and mark the shaft again. The gap between marks is
   the sag that `geometry.ride_height` already bakes in - it should be small,
   because that is what maximum preload is for. Repeat all four corners.
3. **Bump and droop travel.** From the unloaded mark in step 2, compress the
   suspension by hand to full bump (shock bottoms or the arm hits its stop)
   and measure wheel travel; then extend to full droop (shock reaches max
   length or the droop limiter engages) and measure that too. These are
   `suspension.travel_bump` and `suspension.travel_droop`, measured from
   **ride height**, not from the unloaded mark - subtract the step 2 sag from
   the bump number and add it to the droop number.
4. **Spring rate.** Add a known mass (a bag of hardware on the scale from A1
   works) centred over one corner and re-measure the shaft position from step
   2's marks. `spring_rate = added_mass * 9.81 / compression`. Repeat on a
   second corner as a check; front and rear should agree if the shocks really
   are identical, which is the whole premise of treating this as one value
   rather than four.
5. **Damping, by bounce decay.** Push one corner down by hand about 20-30 mm
   and release cleanly (no residual push or hold). Film at 240 fps, as in A7.
   Read the peak-to-peak amplitude of at least three successive oscillations
   and compute the logarithmic decrement `delta = ln(x1 / x2)` between
   consecutive peaks; damping ratio `zeta = delta / sqrt(4*pi^2 + delta^2)`,
   and `damping = 2 * zeta * sqrt(spring_rate * corner_mass)` where
   `corner_mass` is the sprung mass over that corner (roughly
   `mass.total * axle_weight_fraction / 2`, from A1's axle weighing). If the
   corner does not oscillate at all (overdamped), say so instead of forcing a
   number - that is itself useful information about the stock shock oil.

Update `suspension.*` in `vehicle.yaml` and set each `provenance` to
`measured`, same as everywhere else in Session A. Until this is done the
simulator has no suspension travel that means anything - it will still move,
because a `guess` is a real number, but every one of `spring_rate`, `damping`,
`travel_bump` and `travel_droop` is a placeholder chosen for plausibility, not
because anyone rolled the car over a bump and measured what it did. The
obstacle course's potholes and gravel section are exactly where this shows up:
`generate_vehicle_model.py` puts the same suspension model into
`worlds/obstacle_course.sdf` as `worlds/speed_course.sdf`, so a bad guess here
is wrong everywhere the car meets an uneven surface, not just there.

---

## Session B — Parking lot (~1.5 h)

Everything here stays in one place. Radius is computed from telemetry
(`R = v / yaw_rate`), so nothing needs marking on the ground.

| Order | Command | Why |
| --- | --- | --- |
| 1 | `profile:=zed_static` | Free — run it while the cones go out |
| 2 | `profile:=steer_authority` | **The most valuable run in the campaign** |
| 3 | `profile:=skidpad` | Understeer, plus a free lower bound on lateral grip |
| 4 | `profile:=step_steer` | Validates the A3 inertia estimate |

```bash
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=zed_static
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=steer_authority
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=skidpad
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=step_steer
```

`zed_static` never arms — the car cannot move, so leave the E-Stop asserted for
it. Paved surfaces are uniform and low-texture, which is a known stressor for
visual-inertial odometry, so do not skip it assuming the answer is fine.

---

## Session C — Bike path (~2 h, 60 m of clear straight)

Every profile here is out-and-back. That is not only about walking less:
averaging the two directions cancels the path's grade, which it certainly has.

| Order | Command | Why |
| --- | --- | --- |
| 1 | `profile:=pulse_staircase speed_slew_rate:=50.0` | Open-loop pulse-to-speed map |
| 2 | `profile:=coastdown speed_slew_rate:=50.0` | **The most valuable longitudinal run** |
| 3 | `profile:=brake_sweep` | Braking authority |
| 4 | `profile:=tune_profile` | Closed-loop scoring — see below |

```bash
ros2 launch cfr_arduino_bridge characterize.launch.py \
    profile:=pulse_staircase speed_slew_rate:=50.0
```

> **Why `speed_slew_rate:=50.0`.** The production 2.0 m/s² takes 1.6 s to reach
> 3.2 m/s, which would swamp a plant time constant near 0.5 s and make every step
> response measure the rate limiter instead of the car. Raise it for the step and
> coastdown work; **leave it alone for `tune_profile`**, which scores the loop as
> it will actually be flown.

### Watch out

- `pulse_staircase` is the greedy one. A leg at `speed_ks=105` plus its coast can
  reach 40 m. If the path is shorter than 60 m, drop the last two steps or
  shorten their holds before running it.
- `brake_sweep` can make the ESC drive away backwards: it reads a reverse-side
  pulse as a brake only until it sees neutral. The firmware locks braking out
  when this happens, but keep a thumb on the E-Stop.

### Closed-loop retune

Feedforward first, from C1's fit — `analyze_run.py` prints the suggested `kS` and
`kV` directly. Then sweep `kP` and `kI` across runs, one number at a time:

```bash
ros2 launch cfr_arduino_bridge characterize.launch.py \
    profile:=tune_profile gains:="speed_kp=24.0 speed_ki=10.0" label:=kp24ki10
```

The overrides are recorded in `metadata.yaml`, so a run directory always says
which gains produced it. `analyze_run.py` emits a row in the same format as the
"Bench Tuning" table in the top level README, so the ground results sit directly
beside the on-blocks ones.

Copy the winning gains into `config/arduino_bridge.yaml`, and ideally into the
firmware's compiled-in `SpeedGains` defaults (rate gains multiplied by 2.1).

---

## What comes back for free

These need no dedicated run — every profile logs them:

- **Pack voltage sensitivity.** Millivolts are in every `D,` line at 20 Hz.
  Regress speed against voltage across the session as the battery drains.
- **Tachometer health.** Merged-pulse and rejected-edge counters are in the `D,`
  line. Bench was ~5% merged; the report prints what it is under real vibration.
- **Step response and dead time.** From the transitions in C1 and C4.
- **ZED scale.** A5 pins the tire radius, which makes RPM-integrated distance
  absolute, which makes the ZED's scale error fall out of any straight leg. The
  report prints the ratio; 1.000 means they agree.

---

## Analysis

Back with a network, or not — it needs neither:

```bash
./scripts/sync_runs.sh                       # pull runs off the Orin
./scripts/analyze_run.py runs/<run> --mass 4.7
./scripts/apply_vehicle_patch.py runs/<run>  # fold results into vehicle.yaml
./scripts/generate_vehicle_model.py          # push them into the Gazebo world
```

`analyze_run.py` writes `report.md`, `vehicle_patch.yaml` and SVG plots next to
the run. `apply_vehicle_patch.py` edits `vehicle.yaml` as text so its comments
survive, and stamps each value it touches with the run that justifies it and the
date. A number in that file should always be able to answer "says who?".

`--mass` is required by the coastdown fit and comes from A1.

---

## A run directory

```
20260915T141203Z_coastdown/
  telemetry.csv     one row per 50 Hz tick - the primary artifact
  arduino_rx.log    host-timestamped D, and T, lines from the firmware
  arduino_tx.log    host-timestamped frames written to the firmware
  metadata.yaml     profile, gains, git SHA, battery, result
  runner.log        what the runner said while it ran
  bag/              rosbag2, raw backup
  report.md         written by analyze_run.py
```

`telemetry.csv` is the primary one, not the bag: it needs no `rosbag2_py`, opens
in any spreadsheet, and survives being emailed.

---

## Acceptance

Once `vehicle.yaml` is measured and the simulator consumes it, run the same
manoeuvres on the car and in the simulation and compare:

| | Manoeuvre | Metric | Target |
| --- | --- | --- | --- |
| M1 | 0 → 3.2 m/s, hold, coast to stop | speed RMSE, stop distance | ≤5% steady, ≤20% τ, ≤10% distance |
| M2 | 2 m/s, step steer, hold 3 s | yaw-rate RMSE, achieved radius | ≤10% radius |
| M3 | The L-path (1.524 m, −90°, 0.610 m) | final pose error | ≤0.15 m, ≤5° |

These targets are a **proposal, not a derivation**. The first campaign sets the
real baseline; record what was achieved beside them here.

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| "still waiting for E-Stop to be ASSERTED" | The runner needs to *see* it asserted before it will accept it being cleared. Press the button. |
| "E-Stop is clear but auto is not armed" | Set Auto Arm on the controller. |
| "waiting for AUTO_ACTIVE" | The Arduino only leaves `AUTO_ARMED` with steering centred and a zero target; the runner is already holding both, so check the mode on the E-Stop TUI. |
| "Arduino never acknowledged the profile gains" | Gains frames are resent for up to `gains_timeout`. A persistent failure means a bad serial link — check `invalid_frames` in the `D,` lines. |
| "profile gains refused" | The bridge rejected a value the firmware would not accept. The message names the parameter. |
| "exceeded max_distance" | The profile outran the space. Shorten its holds or raise the limit, deliberately. |
| `dropping malformed status frame` | Firmware and Jetson packages are from different protocol versions. Flash and build them together. |
| Analysis says "this fit needs the vehicle mass" | Pass `--mass`, from A1. |
