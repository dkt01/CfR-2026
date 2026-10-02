# FormulaSubZero

FormulaSubZero's default real-car mode follows the visible Speed Course bale corridor continuously. It fits the visible bale-wall centers at several forward distances and steers into the resulting bend. If only one wall is visible, it holds an estimated half-width from that wall. It does not require a bale-map start match or stop after a lap count.

The corridor uses a 4–40 cm above-ground depth band to tolerate bale-height and mount variation. The controller still waits for the start signal (or the manual-start service), a fresh ZED pose, and depth before commanding motion. When no wall is visible, the corridor is too narrow for the car plus clearance margin, or depth becomes stale, it coasts. Steering is bounded and slew-limited; speed is bounded by bend curvature, visible forward stopping room, and the plant's low-speed coast deceleration. The default corridor target is 1.5 m/s before the launch wrapper's `speed_scale=0.3` fail-safe multiplier (about 0.45 m/s on the command wire).

The previous mapped racing-line and CasADi MPC implementation remains selectable with `formula_sub_zero.navigation: map_mpc`. It aligns to the map at the start and has a three-lap stop. In default `corridor` mode, the map is still loaded for shared ROS telemetry, but it does not choose the steering or speed. ZED depth supplies the bale walls directly in the car frame; VIO is used for pose health and telemetry, not for the steering target. This avoids relying on a precise initial map pose, but the follower still needs enough visible bale geometry to identify at least one wall.

## Stall recovery

In the optional `map_mpc` mode, if a forward command produces less than 15 cm of ZED pose movement over two seconds, FormulaSubZero checks a 35 cm reverse arc plus 35 cm of coast margin against the mapped full-body bale clearance. It chooses steering that reduces error to the track center, backs at a commanded 0.45 m/s, waits for the car to stop, then follows the depth planner and MPC at no more than 0.8 m/s for 60 cm before restoring the mapped speed. The launch `speed_scale` also applies to reverse. A blocked corridor near the nose can trigger the same recovery. Recovery gives up after two attempts, a reverse timeout, or no map-cleared reverse path; it holds neutral and logs the reason.

The ZED looks forward and cannot verify the space behind the car. This bounded reverse depends on the mapped bale positions and current ZED pose; a moved bale behind the car is an unobserved hazard. Validate it in simulation and with a supervised low-speed car trial before using it for racing.

## Corridor SIL failure and repair

The first full-scale corridor SIL run grazed at station 103.13 m and contacted a bale at 103.21 m. Its original single midpoint missed the sharp bend, so steering stayed near zero as lateral error grew. Depth was marked lost just before contact; the tachometer's low-speed floor made the node declare a stop while the car still coasted. The follower now fits the wall shape by forward slice, slows for bends and single-wall uncertainty, and uses pose speed plus a sustained-stop check while coasting. These changes have passed the standalone geometry checks but still need a fresh Gazebo clearance verdict.

## Previous mapped-MPC Gazebo result

The earlier mapped-MPC controller finished and stopped after three nominal rendered-depth Gazebo laps in 134.70 s (2026-09-29). Ground-truth chassis and tire clearance stayed at or above 11.4 cm across 4,188 pose samples: zero grazes, zero contacts, zero depth holds, and no pose-stream contamination. A displaced-start loopback run (+0.5 m along, +0.12 m lateral, +8° heading) also finished three laps in 123.60 s with 11.6 cm minimum clearance and no grazes. The Gazebo margin is 1.4 cm beyond the 10 cm graze boundary. An offset-start loopback with displaced bales (`--world-scale 1`) has failed the criterion with contact at station 72.5 m. That mapped planner keeps the line for depth points within 8 cm of mapped bale surfaces, so small layout changes can consume that margin. The physical bale layout, ZED mount, and start-signal target still need measurement before treating this as a real-course driver.

## Validate in Gazebo

From this directory, with the ROS workspace built:

```bash
./selftest.py
./validate.sh --check 60
```

`selftest.py` checks corridor steering and missing-wall behavior, plus the optional mapped planner and MPC safeguards without ROS or Gazebo. `validate.sh` runs the speed-course Gazebo stack at full command scale and watches 60 seconds of continuous corridor driving by default. The real-car launcher keeps its 0.3 fail-safe scale. PASS requires at least 10 m of travel, motion during the final 10 seconds, zero contacts or graze-band samples, no lost depth, and an uncontaminated pose stream. It writes `/tmp/formula_sub_zero_<ROS_DOMAIN_ID>/driver.log`, `monitor.json`, and trace and command CSVs. Use `--loopback` for ROS plumbing, or `--gui` for interactive Gazebo. The previous three-lap verdict applies only to `map_mpc` and does not establish clearance for the new corridor follower.

If startup stops at `waiting for the pose stream`, the harness now exits within 60 wall-clock seconds or as soon as Gazebo or the simulated vehicle exits. It prints the failure and the simulator log path (`/tmp/formula_sub_zero_<ROS_DOMAIN_ID>/sim.log`). An intermittent Gazebo startup crash can be retried after the harness finishes cleanup.

Tune `formula_sub_zero.corridor.speed_mps` only after a clean timed Gazebo verdict, comparing distance, minimum clearance, grazing samples, and contacts. The `map_mpc` path biases are map specific; its startup checks remain active when that mode is selected.

## Deploy to the Orin

From the repository root:

```bash
jetson/scripts/syncSoftware.sh --fsz --build
```

This syncs `jetson/`, FormulaTwo's shared runtime, and FormulaSubZero, then builds the ROS workspace on the Orin. On the Orin, the default corridor follower can use ROS system Python; `setup.sh` is needed only for the optional mapped MPC. To prepare that environment or print the driver command:

```bash
~/software/formulaSubZero/setup.sh
~/software/scripts/launchFormulaSubZero.sh -n
```

The dry run prints the driver command. For a real run, start the bridge and ZED with `~/software/scripts/launch.sh --no-cmd-vel` in one terminal, then run `~/software/scripts/launchFormulaSubZero.sh` in a second. FormulaSubZero starts the visual signal detector on the real ZED color image; Gazebo supplies its own detector. The wrapper checks for a live bridge and pose, and refuses a second `/drive_cmd` publisher. It starts at `speed_scale=0.3` as a physical-launch fail-safe, records telemetry, and waits for the start signal. The scale multiplies the controller's speed command; it is separate from the mapped speed profile. Launching arms the actuators: be at the car with the E-Stop remote in hand and confirm immediately before the foreground launch.

A manual start sets GO without needing a signal pixel or bale-map alignment. The automatic start still uses `/start_signal_detector/go`; if it does not arrive, check `/start_signal_detector/state` and the ZED color image.

Only the optional `map_mpc` controller needs CasADi and IPOPT at runtime. `setup.sh` installs CasADi 3.8.1 or newer and verifies the solver and ROS Python imports. For an offline install, copy a compatible Linux ARM64 wheel to the Orin and pass its path to `setup.sh`; it installs the wheel with package indexes disabled. CasADi 3.8.0 imported on the Orin but crashed while solving with IPOPT, so use 3.8.1 or newer.
