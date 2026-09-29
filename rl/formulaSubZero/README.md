# FormulaSubZero

FormulaSubZero drives the **Speed Course** for three laps and coasts to a stop. Its objective is the shortest three-lap time that clears every bale without entering the 10 cm graze band.

The planner reads the committed course map and tests candidate lateral offsets against the mapped chassis and tire envelope. A dynamic program favors clearance and a smooth line. Localized line corrections follow repeated clearance measurements at the start-line exit, middle hairpin, and approach corner; all path parameters are in `config.yaml`. The speed plan is a backward coast-feasibility pass, then a configurable fraction of that profile. The car has no active brakes, so the plan lifts before hairpins.

A CasADi MPC uses the measured yaw lag and command delay to trim the existing FormulaTwo centerline steering prior. Its trim is bounded by `max_steer_trim` on the mapped line and increases when a live obstacle shifts the local path; if the solve fails, it returns to the prior and reduces commanded speed. The FormulaTwo ROS boundary supplies ZED pose, wheel-encoder speed, the start signal, speed caps, three-lap counting, coast-to-stop steering, `/drive_cmd`, and `/formula_one/telemetry` for Run Lab.

Each registered ZED depth frame supplies ground-plane bale points to an 8 m local planner. When those points agree with the map, it follows the measured racing line; when they differ, it searches lateral offsets against both the mapped body envelope and the observed obstacles. If depth is missing, the corridor closes, or the pose jumps, it commands a coast to rest. Speed is limited by the depth-verified stopping distance as well as the mapped hairpin profile.

At the start, the controller matches the visible bales and the measured depth to the start signal against the course map. This lets it estimate an offset start position and heading before moving, rather than assuming the exact simulator spawn pose. The start-signal map coordinate and the physical ZED left-lens offset in `config.yaml` must be checked on the real course. The visible signal pixel falls about 14 cm from the configured mount-center Y coordinate in Gazebo, so the signal corrects only along-track position; visible bales establish lateral position and heading. ZED pose drift after launch remains a calibration risk; a single start alignment does not correct it throughout the run.

## Gazebo result

The depth-aware controller finished and stopped after three nominal rendered-depth Gazebo laps in 134.70 s (2026-09-29). Ground-truth chassis and tire clearance stayed at or above 11.4 cm across 4,188 pose samples: zero grazes, zero contacts, zero depth holds, and no pose-stream contamination. A displaced-start loopback run (+0.5 m along, +0.12 m lateral, +8° heading) also finished three laps in 123.60 s with 11.6 cm minimum clearance and no grazes. The Gazebo margin is 1.4 cm beyond the 10 cm graze boundary. An offset-start loopback with displaced bales (`--world-scale 1`) has failed the criterion with contact at station 72.5 m. The live planner currently keeps the mapped line for depth points within 8 cm of mapped bale surfaces, so small layout changes can consume that margin. The physical bale layout, ZED mount, and start-signal target still need measurement before treating this as a real-course driver.

## Validate in Gazebo

From this directory, with the ROS workspace built:

```bash
./setup.sh
./validate.sh --check 420
```

`validate.sh` runs the speed-course Gazebo stack, waits for the pose and depth streams, releases the visual start signal, and monitors ground-truth chassis and tire clearance. PASS requires three laps, a confirmed stop, **zero contact and zero samples inside the 10 cm graze band**, and an uncontaminated pose stream. It writes `/tmp/formula_sub_zero_<ROS_DOMAIN_ID>/driver.log`, `monitor.json`, and trace and command CSVs beside the monitor. `--loopback` is available for ROS plumbing. Add `--start-along 0.5 --start-lateral 0.12 --start-heading-deg 8` to exercise a displaced start and the depth-based signal anchor. `--world-scale 1` moves the loopback bales with a deterministic layout field while leaving the driver's map fixed. Use Gazebo for the clearance verdict. For an interactive run, use `./validate.sh --gui` or omit `--check` and open the browser viewer printed by the harness.

If startup stops at `waiting for the pose stream`, the harness now exits within 60 wall-clock seconds or as soon as Gazebo or the simulated vehicle exits. It prints the failure and the simulator log path (`/tmp/formula_sub_zero_<ROS_DOMAIN_ID>/sim.log`). An intermittent Gazebo startup crash can be retried after the harness finishes cleanup.

Tune `formula_sub_zero.speed_fraction` only after a clean three-lap verdict. Increase it in small steps and compare total time, minimum clearance, grazing samples, and contact samples. The path biases are map specific; rerun the verdict when the course map changes.

## Deploy to the Orin

From the repository root:

```bash
jetson/scripts/syncSoftware.sh --fsz --build
```

This syncs `jetson/`, FormulaTwo's shared runtime, and FormulaSubZero, then builds the ROS workspace on the Orin. On the Orin, install CasADi in a virtual environment that can read ROS Python packages:

```bash
~/software/formulaSubZero/setup.sh
~/software/scripts/launchFormulaSubZero.sh -n
```

The dry run prints the driver command. For a real run, start the bridge and ZED with `~/software/scripts/launch.sh --no-cmd-vel` in one terminal, then run `~/software/scripts/launchFormulaSubZero.sh` in a second. The wrapper checks for a live bridge and pose, and refuses a second `/drive_cmd` publisher. It starts at `speed_scale=0.3` as a physical-launch fail-safe, records telemetry, and waits for the start signal. The scale multiplies the controller's speed command; it is separate from the mapped speed profile. Launching arms the actuators: be at the car with the E-Stop remote in hand and confirm immediately before the foreground launch.

The controller needs CasADi and IPOPT at runtime. CasADi publishes Linux ARM64 wheels; `setup.sh` verifies both CasADi and `rclpy` in the created environment.
