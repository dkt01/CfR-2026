# formulaTwo: three laps of the Speed Course, on the map *and* on what the ZED sees

formulaOne/v12 drives the Speed Course at about 32 s/lap in Gazebo, from a map
read against the ZED pose. formulaTwo has the same goal, **three laps then
stop**, at v12's pace or better. It is built to hold up when the car, the
course and the camera are not the ones in the simulator.

| | |
|---|---|
| Goal | 3 laps, no contact, on the centerline, smooth steering, **then stop** |
| Speed rules | 2.5 m/s in the hairpins, 5.2 m/s elsewhere; the car has no brakes |
| Coast-feasible bound | 27.8 s/lap |
| Scripted baseline, nominal car | 3 laps every time, 30.5 s/lap, 5.8 cm clearance |

## Result: `bestModel/f2_v2_40M/policy.npz`

Offline, 3 laps **and at rest**, from the grid. Each car counts once. "Randomization"
is the strength of every randomized range at once: plant, bales, pose and
camera. Full numbers are in `bestModel/f2_v2_40M/graded_eval.txt`.

| | v12 (map only) | **formulaTwo f2_v2 @ 40M** |
|---|---|---|
| nominal car | 0% (touches a bale at 89 m) | **100%**, min clearance **+7.4 cm** |
| 25% randomization | 41% | **100%**, worst clearance +4.1 cm |
| 50% | 12.5% | **91%** |
| 75% | 1.6% | **48%** |
| full | 0.8% | **15%** |
| mean cross-track error, nominal | 0.053 m | **0.048 m** |
| lap times (standing / flying / flying) | 36.85 / 35.34 / 35.32 s | 36.80 / 35.40 / 35.37 s |
| stop | 13.5 m past the line | 13.3 m past the line |

The final checkpoint was picked on this test from 18M, 35M, 36M, 39M and
40M. The late checkpoints all score within a few points of each other, and
all beat 18M (100/100/80/37.5/9.4%). 40M has the most clearance and the
lowest CTE.

**CORRECTION: formulaTwo is about 2.2 s/lap SLOWER than v12, not equal.**
The table above ran v12 inside formulaTwo's env, which maps v12's throttle
onto formulaTwo's speed floor. That floor is clipped for the lowest-drag car
(`floor_drag_scale: 0.75`), so every car lifts earlier into every hairpin.
Driven at the floor alone, flying laps are:

| `floor_drag_scale` | 1.00 (v12) | 0.90 | 0.85 | 0.75 (formulaTwo) |
|---|---|---|---|---|
| flying lap | 33.45 s | 34.05 s | 34.60 s | 35.65 s |

Both policies hold the throttle at the floor on 100% of ticks, so the floor
IS the pace. In Gazebo, v12 on its own floor ran 32.35 / 31.35 s; formulaTwo's
first lap was 36.15 s. Getting v12's pace back means v12's floor
(`floor_drag_scale` 1.0), with the drag randomization narrowed so that floor
stays feasible, and a retrain.

**Gazebo (`./validate.sh --check`, 2026-09-25): FAIL.**
- Perception matches: Gazebo's depth scan agrees with the training renderer
  to a median ratio of 0.994.
- Driving does not. At the hairpin into the start line, the policy was on
  full right lock at the 2.05 m/s floor and ran 0.32 m wide. It hit a bale
  and was tripped onto its roof, and only then did the scan go invalid.
- v12 passes the same Gazebo build.

**Full-strength randomization is harsher than the car.** At full strength
even the scripted prior at a 1.4 m/s floor finishes 0%. Every failure is at
the 20 m chicane, on draws with slow steering (yaw lag ×1.4–1.6, servo τ
0.065–0.08 s). Those ranges are v12's own. `steering_tau` is still an
unmeasured guess: formulaOne config, experiment A7. Measuring it is the
cheapest way to find out which row of this table the real car is on.

**How it trained.**
- f2_v1: started as v12 with a 1.5M-step critic warm-up. Nominal completions
  from 7M.
- f2_v2: resumed from f2_v1 at 12M with the throttle-edge and std reset.
  Checkpoints selected on a half-strength field eval. The half-strength
  finish rate climbed from ~48% to ~90% as the learning rate annealed over
  the last ~8M steps (209 min wall clock for the 28M).

## What is new, and why each choice was made

**1. The bales are not where the map says (`world.py`).** Every episode draws a
displacement field per car: a long-wavelength *layout* warp (up to 10 cm, the
course laid out a little differently from the drawing), bale-length *jitter*
(up to 3.5 cm) and a bale *scale* change (−1.5 to +2 cm on every face). The
plant drives, collides and is scored in the warped world. The policy's map
features read the nominal map from a noisy, drifting pose. The gap between
them is exactly what the depth channel has to close, and it also stands in
for this design's biggest real-world risk: pose drift.

**2. Ground friction, mass, motor latency (`plant.py`).** Friction is a
lateral-grip cap, `|v·r| ≤ μg`, with μ ∈ [0.60, 1.10]. The skidpad *held*
0.55 g and the course needs 0.43 g, so every draw can make every corner. Mass
∈ [0.90, 1.15] scales drag deceleration and acceleration as 1/m. Motor lag
τ ∈ [0, 0.12] s, and a throttle-only delay of up to 60 ms on top of the shared
dead time. These sit alongside all of v12's draws.

**3. The ZED 2i, artifacts included (`perception.py`).** A pinhole render of the
course as the simulated ZED sees it (110° HFOV, 0.20 m up, 0.315 m forward,
10–15 Hz, 40–120 ms latency). Then the camera's failure modes, applied to the
pixels:

- range noise σ = s0 + s2·Z²
- flying pixels at depth edges
- random pixel dropouts
- blank blobs (glare, shadow)
- a narrower real lens
- mount pitch, yaw and height error
- the 0.3 m minimum range
- dropped frames

**4. Virtual LiDAR, not an 84×84 CNN.** The camera sits at 0.20 m and every
bale top is at 0.356 m, so every near-horizon ray ends on the first bale it
meets. The course is 2.5-D, and a depth image of it carries one number per
column. The image is back-projected, points 0.07–0.33 m above the ground are
kept, and the nearest per column is taken. The result is 64 log-encoded beams,
**stacked over the last 4 frames** (~0.33 s at 12 Hz) plus the newest frame's
age. The reduction is the same code on both sides: `Camera.sample_depth`
reads the real ZED image on the same grid (every 10th column, every 5th row).
There is no GPU here, and a CNN would have cost about 10× the samples per
second to learn something the geometry already gives us.

**5. The expert reward (`reward.py`).** The simulator knows the exact
centerline of the course as built, so the reward is progress along it, gated
by how well the car sits on it:

```
reward += progress · metres · exp(−e_y²/2·0.20²) · exp(−e_ψ²/2·0.35²)
```

Both `e_y` and `e_ψ` come from the **true** pose against the **true**
centerline, which the policy never sees. It drives on a noisy map pose and a
depth scan. Because the gate multiplies progress, precision is priced in
proportion to speed and is bounded, so a corner's unavoidable error can't
outbid the lap (formulaOne's quadratic-lateral lesson). Everything else is
formulaOne's, tuned there for measured reasons: time 3.0, overspeed, graze,
crash, smoothness, bonuses and the lap-improvement shaping.

**6. Asymmetric actor-critic, started as v12 (`train.py`).**

- The actor's first 35 inputs are v12's, bit for bit. `selftest` and a
  trajectory comparison check this, and the trajectories are identical.
- The actor starts as v12's weights, with zero weights on the depth inputs.
  At step 0 it *is* v12. Its steering std is reset from v12's collapsed 0.067
  to 0.2, so it can explore.
- The critic trains alone for 1.5M steps before the actor may move.
- The critic also reads 27 privileged values: true CTE and heading, true
  clearance, map error, pose error, and the car's draw (μ, mass, lags, …).
  Only the actor ships.
- Randomization strength ramps from 0.4 to 1.0 over the first 40% of the run.

**7. Three laps and stop.** This is formulaOne's stopping phase with
`laps: 3`: throttle forced to zero after the line, steering live, and the run
ends at rest. `finish` counts only runs that finish at rest.

## Two things found on the way, both fixed

- **Contact is now scored against the real car.** formulaOne padded the
  footprint to 0.58×0.32. The Gazebo model is a 0.55×0.30 chassis whose tyres
  stand 12.5 mm proud of it at ±0.1625 m. formulaTwo tests the chassis corners
  and the tyre faces. This is also why v12 "crashes" offline and still gets
  round in Gazebo: it is within millimetres everywhere.
- **v12's speed floor was infeasible for low-drag cars.** The floor is clipped
  by the coast-feasible profile of the *nominal* car. A car with 15%+ less
  drag (for example, heavier) is then forced up to 0.35 m/s over a hairpin
  cap, with no action that avoids it and no brakes. The floor is now clipped
  against the lowest drag the randomization draws (`floor_drag_scale`). It is
  only a minimum, so faster cars still carry their speed.

## Run it

```bash
python3 selftest.py                          # ~3 min: world, camera, v12 match, feasibility
./train.sh --dir runs/f2_v1                  # self-test, train, export, score
python3 evaluate.py bestModel/f2_v2_40M/policy.npz --plot report.png   # graded rows by default
python3 evaluate.py ../formulaOne/bestModel/v12/policy.npz   # v12 on the same course, map only
```

Use formulaOne's venv (`../formulaOne/.venv`). Training uses 8 env worker
processes (`--workers`) and evaluates every 1M steps. `nominal` is how fast it
is; `field` (everything redrawn) is whether it will get round on the day.

## Files

| | |
|---|---|
| `config.yaml` | every knob; every random range is `[lo, hi, nominal]` |
| `world.py` | the as-built course: warp, true clearance, true Frenet frame, ray caster |
| `perception.py` | ZED render, artifacts, depth → virtual LiDAR, frame stack, and the car-side sampler |
| `plant.py` | formulaOne's measured plant + friction, mass, motor lag, throttle delay |
| `env.py` | the vectorised env: truth / sensed pose / camera pose, privileged state |
| `reward.py` | the centerline-gated expert reward |
| `train.py` · `vec.py` | PPO with the asymmetric policy and v12 warm start; multi-process env |
| `export_policy.py` | actor only → `.npz` (loads with formulaOne's `NumpyPolicy`), self-checked against torch |
| `evaluate.py` · `selftest.py` | offline score; pre-training gates |
| `track.py` · `observation.py` · `baseline.py` · `policy.py` | from formulaOne (floor clip changed in `observation.speed_floor`) |

## Driving it: `formula_two_node.py`

This is formulaOne's node, plus the ZED depth stream and a watchdog on it.
Anchoring, the rule cap enforced on the wire, the stopping phase and
`~/telemetry` are all unchanged.

```bash
# Gazebo -- sensors:=true is what makes the rgbd camera exist.  Without it
# there is no depth and the car will not move.
ros2 launch cfr_arduino_bridge speed_course.launch.py sensors:=true laps:=3 \
     path_follower:=false cmd_vel_to_drive:=false
ros2 launch rl/formulaTwo/formula_two.launch.py
# The car WAITS FOR THE START SIGNAL.  Turn it green (web viewer "Set signal: Go", or):
ros2 service call /obstacle_randomizer/start_signal std_srvs/srv/SetBool "{data: true}"

# The car (E-stop in hand, see formulaOne/DEPLOY.md).  Terminal 1, then 2:
~/software/scripts/launch.sh --no-cmd-vel
~/software/scripts/launchFormulaTwo.sh          # preflight checks, then asks for GO
~/software/scripts/launchFormulaTwo.sh -s 0.5   # speed_scale (default 0.3); -n prints the command
```

**Depth.**
- The node reads `/zed/zed_node/depth/depth_registered`: 32FC1 metres, or
  16UC1 millimetres in OpenNI mode.
- The camera info comes from `/zed/zed_node/depth/camera_info` on the car and
  Gazebo's `.../left/image_rect_color/camera_info` in simulation.
  `camera_info_topic:=auto` picks between them.
- Every image goes through the training code: sample the grid, reduce to the
  64-beam scan, encode, then push onto the 4-frame stack. Frames faster than
  15 Hz are thinned out, because the policy trained on 10–15 Hz.

**Without depth the policy crashes**, 100% of runs, even with a perfect map
(measured). So the watchdog works in stages:

- It **does not move until a depth frame has arrived**, whatever the start
  signal says.
- **At 0.3 s without a fresh frame, the network is taken off the wheel**
  (`depth_hold_after`). The centerline prior steers at the speed floor. If
  depth returns before 1 s, the stack is refilled and the policy resumes.
- **At 1.0 s, depth is LOST** (`depth_timeout`). It is also lost when more than
  60% of beams are invalid for 3 frames in a row (training peaks at 45%).
- On loss, `depth_fallback:=`
  - `stop` (default): throttle zero, the prior steers, and the car coasts to
    rest. The run is over.
  - `map`: race on without the network, prior steering at the floor.

**Why two stages: crashes after depth freezes mid-run** (sim, 64 runs each).

| | stop at 0.3 s | stop at 1.0 s, network driving | **hold at 0.3 s, stop at 1.0 s** |
|---|---|---|---|
| nominal car | 0% | 0% | **0%** |
| 25% randomization | 1.6% | 43.8% | **4.7%** |
| 50% randomization | 29.7% | 70.3% | **34.4%** |

A 1 s timeout tolerates depth hiccups without ending the run. Letting the
network drive for that second on a frozen scan it never trained on is what
costs; holding it off costs almost nothing.

Racing on after loss (`map`) finished 3 laps on 100% of nominal cars but only
30–37% under randomization. The watchdog does not rescue a policy that is
merely confused by bad but valid-looking depth.

`/formula_one/depth_status` reports the state every tick (ok / waiting / hold
/ stale / invalid, the reason, and any fallback). The telemetry `driver` field
reads `depth_hold`, `fallback_stop` or `fallback_map` when one is active.
`observation` carries the full 292-wide input, scan stack included. The
depth image itself is not in `record_run.py`'s topic list, so the telemetry
is what the Run Lab has.

**`validate.sh`** runs all of this in Gazebo with RViz. The car waits for
the start signal: turn it green yourself, or use `--check` (unattended), which
turns it green through the same service and reports PASS/FAIL. It prints the
verdict from `run_monitor.py`: contact from the true pose and real footprint,
whether contact or depth loss came first, and rollovers. `--manual-start`
bypasses the signal. `--loopback` runs it without Gazebo.

**The ROS name is `formula_one`** (`node_name:=`). `record_run.py` and the
Run Lab key on `/formula_one/telemetry`, and this keeps both working
unchanged. The manual start service is `/formula_one/manual_start`, as
before. Never run the two drivers together: both publish `/drive_cmd`.

**Test it before Gazebo or the car:**

```bash
source /opt/ros/jazzy/setup.bash && source ../../install/setup.bash
python3 node_selftest.py      # ~8 min, real time, on ROS_DOMAIN_ID 78
```

This runs the node against `ros_loopback.py`, which is the plant plus a
rendered 640×360 depth image and `CameraInfo`, with depth faults injected.
There are six scenarios: normal 3 laps, depth stops, all-NaN depth, a 0.6 s
depth blip (the hold engages, the run continues), no depth ever, and the
`map` fallback.

**Onto the Orin.** `jetson/scripts/syncSoftware.sh` deploys both drivers:

```bash
jetson/scripts/syncSoftware.sh --build                          # v12 + f2_v2_40M
jetson/scripts/syncSoftware.sh --build --f2-policy <run>        # another formulaTwo run
jetson/scripts/syncSoftware.sh --build --no-f1                  # formulaTwo only
```

This produces `~/software/formulaTwo/`: the code, the chosen run's
`policy.npz` and `config.yaml` at its top level (which the launch file then
defaults to), and the track cache. It also makes the `~/jetson` symlink both
drivers need.

**Before trusting it on the car:**

- **Pitch.** Check that the scan sees bales and not ground:
  `/formula_one/depth_status` should report a low invalid fraction with the
  car parked in the corridor. The scan believes the camera is level and 0.20 m
  up (`camera_height`, `camera_pitch`). Training covered ±0.75° of pitch error.
- **Frame rate.** Depth should be published at ≥10 Hz; `pub_frame_rate` is 12
  in `cfr_zed2i.yaml`.
