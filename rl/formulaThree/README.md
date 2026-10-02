# formulaThree

PPO for **three clean Speed Course laps and a controlled stop**, using ZED
registered depth, ZED pose and wheel-encoder speed. This is a new training
stack, not a validated faster checkpoint. Train it and compare independent
Gazebo verdicts before drawing conclusions about lap time.

## What changed

- The actor receives 36 pose/map/motion features, four 64-column depth scans,
  and depth age: 293 inputs. It uses the known centerline anchored to the ZED
  pose at the start signal. Depth is sampled with CameraInfo intrinsics,
  leveled with camera attitude, reduced to bale-height ranges and encoded by
  the same code in training and deployment. This is depth-image input with
  geometric preprocessing, not a raw-image CNN.
- The critic additionally receives 27 simulation-only features. Those inputs
  are masked out of the actor and removed from the exported numpy network.
- Throttle action **−1 coasts, 0 follows the coast-feasible reference, +1
  requests the local cap**. Unlike formulaTwo, the reference is not a minimum
  speed. The policy can lift early or recover from tracking error. Steering
  remains a learned residual on the centerline prior. Commands are capped at
  2.5 m/s in hairpins and 5.2 m/s elsewhere; actual overspeed is also scored.
- Collision checks cover the chassis and tire perimeters at every integration
  substep. Clearance subtracts allowances for perimeter sampling, SDF
  interpolation and movement between samples. An estimated clearance at or
  below 6 cm during the race caps the car at a walking-pace crawl (steering
  stays live) for up to `recovery_timeout_s`, so it can be driven clear and
  keep racing instead of ending the episode on first contact; a deeper
  penetration, a contact that never clears, or any contact during the
  post-finish stop still ends the episode as a crash. A separate quadratic
  penalty starts at 16 cm. These are conservative estimates, not a
  collision-free guarantee.
- Steering adds a geometric correction from the newest depth scan: the
  midpoint between the nearest bale left and right of a forward deadzone,
  folded in the same way as the centerline prior. Reads what the camera
  currently sees rather than the pose-anchored centerline, so it keeps
  centering through pose drift and is what mainly steers a car clear during
  recovery. Gated on clearance (`depth_center.touch_margin_m`, defaults to
  `collision.safety_margin`), not on range to the nearest bale: the corridor
  is narrow enough (~0.92 m) that a bale sits within any sane range almost
  everywhere, and a driven apex legitimately runs a few cm of clearance, so
  gating on range alone fought the racing line for the whole lap rather than
  just steering out of a graze.
- Reward favors centerline progress, penalizes time, overspeed, close passes
  and steering oscillation. Success bonuses cannot be collected on a crash or
  a bale-contact tick. There is no previous-lap improvement bonus to reward a
  slow first lap.
- Checkpoint selection ranks nominal clean-finish rate, then full-randomization
  clean-finish rate, then clean lap time. Partial progress breaks ties when
  nothing finishes. Every evaluation reuses held-out seeds. Half-strength
  randomization is diagnostic only; training reward does not choose the model.

The plant, depth renderer, centerline prior and ROS integration originate in
formulaTwo. They are kept local so this racer deploys independently, including
when `--no-f1 --no-f2` is used. Its action semantics are different: **do not
resume formulaOne/formulaTwo checkpoints**. A policy is bound to its effective
runtime configuration by a digest, checked by both the evaluator and driver.

## Train and evaluate

From `rl/formulaThree`:

```bash
./setup.sh
./train.sh --dir runs/f3_v1
# Resume only a formulaThree checkpoint with matching environment settings:
./train.sh --dir runs/f3_v2 --resume runs/f3_v1/best_model.zip --steps 10000000

.venv/bin/python evaluate.py runs/f3_v1/policy.npz \
  --episodes 256 --json runs/f3_v1/evaluation.json --plot runs/f3_v1/report.png
.venv/bin/python reward_probe.py
.venv/bin/python selftest.py
```

Training uses the fast numpy vehicle/depth model, not Gazebo. The default
budget is 40 million steps; allow several hours and measure local throughput
before choosing a budget. `VENV=../formulaOne/.venv ./train.sh ...` reuses the
existing training environment. Regression tests run natively. The trainer
writes `config.yaml`, `history.json`, best/last/periodic SB3 checkpoints; the
wrapper exports and checks `policy.npz` against torch before writing it.

`evaluate.py` loads the config beside the policy. Finish means all laps,
clearance held and at rest, including the coast-down after the line. JSON
contains every outcome, including failures. Report finish rates and clearance
alongside times; a faster crash is not an improvement. Offline measurements
are conditional on the vehicle model, which still needs the high-speed
Gazebo yaw-response fit described in formulaTwo's README.

## Validate in Gazebo

With native ROS 2 Jazzy and Gazebo installed:

```bash
./validate.sh --policy runs/f3_v1/policy.npz --check 300 --no-web
./validate.sh --baseline --check 300 --no-web
```

The script sources ROS and the repository's installed workspace, renders the
ZED and exercises the visual start signal. Build first if needed:
`ROS2_WS="$PWD/../.." ../../jetson/scripts/build.sh`. Set `ROS2_WS` to use a
different native workspace. For RViz/browser use:

```bash
./validate.sh --policy runs/f3_v1/policy.npz
# Optional plumbing-only run, not a Gazebo performance result:
./validate.sh --policy runs/f3_v1/policy.npz --loopback --check 300 --no-web
python3 node_selftest.py --policy runs/f3_v1/policy.npz
```

The default ROS domain is 83 and Gazebo partition is `formula_three`.
`--check` fails for missing monitor data, any contact, clearance ≤6 cm,
overspeed >0.15 m/s, excessive roll, depth loss, pose jumps, clock reversal,
incomplete laps or a timed-out stop. It writes
`/tmp/formula_three_monitor.json.verdict.json`, with the monitor trace and
command CSV beside it. Keep these artifacts with the evaluated checkpoint for comparison.

Compare formulaOne and formulaTwo using their own configs, action mappings
and validation scripts, on the same world. Record standing and flying laps,
finish/stop, minimum clearance and depth health. Their old PASS thresholds
are weaker, so an old PASS alone does not prove it meets formulaThree's
clearance criterion.

## Deploy

After a clean Gazebo result, from the repository root:

```bash
jetson/scripts/syncSoftware.sh --build --f3-policy f3_v1 --no-f1 --no-f2
```

Sync is opt-in (`--f3-policy RUN` or `F3_RUN`), copies the selected policy and
its config together, and uses the existing sync/build flow. On the Orin:

```bash
~/software/scripts/launch.sh --no-cmd-vel
~/software/scripts/launchFormulaThree.sh --speed-scale 0.3
```

Launching arms real actuators: be at the car with the E-stop remote in hand
and confirm immediately before launching. The wrapper retains the existing
preflight and GO confirmation. It starts with recording enabled. Only one
driver may publish `/drive_cmd`; telemetry remains `/formula_one/telemetry`
for Run Lab compatibility. Depth loss defaults to coasting to rest with prior
steering. Neither training nor this implementation launches the physical car.
