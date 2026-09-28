# Obstacle Racer Numba–Gazebo parity audit

## Follow-up, 2026-09-28 (machine idle, no training running)

Three of the findings below changed once they were measured differently.
The Gazebo side was fixed. The numpy plant is unchanged. The last two rows
were added after v9's refinement run (v8 140M → 200M).

| Check | Evidence | Change |
| --- | --- | --- |
| Cloud delivery | The slow camera was DDS, not rendering. With best-effort subscribers an idle machine still got 6.8 Hz raw and 3.2 Hz processed clouds, p90 gap 0.66 s. With reliable depth-1 subscribers (cloud_segmentation already used them), both read 12.0 Hz in sim time. Gazebo's Sensors system holds each step until the ZED renders, so the sim runs at RTF ≈0.47 instead of dropping frames. A best-effort reader loses most of a multi-MB cloud's UDP fragments. A lower `real_time_factor` gave *fewer* frames per sim second. | `obstacle_racer_node` subscribes reliable, depth 1 (`cloud_reliable`, default true), and so does `gazebo_check`. In the v8 Gazebo runs, stale ticks fell from 38–44% to 0%. On the car, the ZED wrapper publishes reliable, so the same subscription applies. |
| Yaw lag | `validate.sh --step-steer` holds a speed for 2.5 s, then steps the steering, and aligns both traces to the command. Over 16 passes (1–2.5 m/s, ±0.3 and ±0.6), the fitted `yaw_response_tau` has a median of 0.357 s against the nominal 0.340 s. The 0.103 s fit below came from starts at rest, where speed error dominates. Gazebo's lag depends on step size: ±0.3 fits 0.41–0.53 s, −0.6 about 0.28 s, and +0.6 0.05–0.18 s. Steady yaw gain is 0.93–1.09 of the plant's. | The plant nominal and randomization stay as they are. A single first-order lag cannot match this amplitude dependence, and the real car remains unmeasured. |
| Reverse | `sim_vehicle_node` now follows the firmware's SpeedController. A target against the last-driven direction counts as zero until the tach has read stopped for `direction_settle` (0.1 s). The tach sign comes from that direction. `validate.sh --reverse` from 1, 2 and 3 m/s: Gazebo and plant stop at 1.60/1.65, 2.87/2.90 and 3.94/3.95 s, and reach −0.3 m/s at 1.80/1.80, 3.07/3.05 and 4.25/4.15 s. Both settle at −1.00 m/s. | Fixed in Gazebo. |
| Grade | `sim_vehicle_node` set the wheel speed, and Gazebo's unlimited-torque wheels held it. A coasting car therefore braked perfectly down the helix: v8 fell from 1.8 to 0.9 m/s where plant.py gained speed. The node now applies plant.py's climbing term, g·tan(pitch), while rolling. It can stop the car but not roll it back through zero. Pitch comes from `pose` (`/zed/zed_node/pose` in simulation.launch.py). With `--surfaces`, ramp and deck passes at 1–2 m/s match within 0.06–0.11 m/s speed RMS and ≤4 cm end position. Helix passes match within 0.5 cm height RMS and 1° roll RMS. | Fixed in Gazebo. Launches without a pose remap (characterize, training) see zero pitch, as before. |
| v8 in Gazebo | `bestModel/v8` (140M), seeds 201/202/208/218 × 2 starts, 150 s: 0/8 laps with 0% stale. Six runs wedged at the helix exit against the tunnel-mouth wall (6.80, 1.46), and two rolled at the helix entry. The numpy plant is also blocked at that pose. The numpy sim finishes 10.9% of 64 starts on the same seeds, and 25 of those runs end in the Wide Section. | Now a closed-loop policy-robustness gap at the helix, not a sensor-timing artifact. |

| Contact stall | In the six wedged v8/v9 runs, the tach read 0.7–1.0 m/s for 5 s while the car sat still. `sim_vehicle_node` reported its own wheel speed, and Gazebo's velocity-driven wheels spun against the wall. plant.py zeroes the speed on contact, so in numpy the tach reads stopped and v9 backs off (its numpy trace hits the same wall at 9.8 s and reverses out). | `sim_vehicle_node` now takes the ground speed from `pose`. When the car makes under `contact_stall_ratio` (0.3) of the wheel speed for `contact_stall_time` (0.1 s), the wheels and tach drop to the ground speed, as a stalled motor would. With 0.25 s the tach averaged 0.4 m/s through the wedge and v9 kept pushing. The reverse pass is unchanged with the stall on or off. |
| v9 in Gazebo | `runs/v9` best (198M), same 8 starts. Without the stall: 0/8, 6 wedged at (6.80, 1.46), 2 rolled at the helix entry (`validate_20260928_110734.json`). With it: 0/8, but 7 backed off the helix exit and drove the tunnel. 4 of those ended at the tunnel's south end near (7.1, −7.9). The two traces checked there turned tighter than numpy at full lock, clipped the inside corner near (6.8, −8.7) and stopped facing a wall. The other 3 ended in the tunnel at y −3 to −6, after 50–135 s. 1 rolled at the helix entry (`validate_20260928_114452.json`). In numpy, v9 finishes 28.1% of 64 starts on these seeds, against v8's 10.9% (`numba_20260928_055148.json`). | The next gap is at full-lock steering at low speed, where Gazebo turns faster than the plant (see Yaw lag). Rollovers at the helix entry are still open. |

Still open: contact (plant.py slides along walls, Gazebo wedges and can
roll), yaw at full lock and low speed, the ramp crest at 3 m/s, and the
car-wash lip at 2–3 m/s.

---

2026-09-27. Policy: the main repository's `runs/v8/best_model.zip` at 122,000,896
training steps, exported to `runs/v8/policy.npz`. The checkpoint's saved
`config.yaml` has the same settings as this checkout. The export's largest
Torch versus NumPy action difference over its recurrent probe was 5.09e-7.

## Findings

| Check | Evidence | Consequence |
| --- | --- | --- |
| Numba baseline | The checkpoint record reports 19/64 held-out finishes (29.7%) and 90.6% helix section clears. An independent, repeatable single-thread exported-policy evaluation on seeds 201, 202, 208, 218, and 5001 finished 4/25 starts (16%); one ended in a helix crash. | V8 is not yet a reliably finishing policy, even in its training simulator. The default four Gazebo validation seeds (201, 202, 208, 218) are especially hard in the saved training record. |
| Rendered helix scan | On seed 201, five upper-to-lower helix poses had 92.2% scan-bin agreement overall. The lowest pose had 75%; one bin read open at 6 m in Gazebo but a 0.62 m wall in Numba. The follow-the-gap steering prior still differed by only 0.009 normalized command there. | Perception geometry is broadly close at rest, but the lower join and moving-camera latency still need coverage. |
| Open-floor yaw | From rest, a 0.55 steering command and 2 m/s target over 1.2 s yawed Gazebo 34.4° and Numba 21.5°, with a 0.22 m endpoint difference. At 1 m/s the turns were 22.9° and 17.5°. | The trained plant underpredicts early left turning. This is a command-to-motion mismatch independent of helix geometry. |
| Helix yaw | At three helix positions and a 2 m/s target, Gazebo turned 34–35° and Numba 21.6° in 1.2 s. Endpoint yaw errors were 12.6–13.7°; height RMS was 0.3–0.4 cm and speed RMS 0.19 m/s. | Early yaw is the largest measured motion gap on the helix. The short passes do not establish contact parity. |
| Yaw-lag fit | Fitting only the open-floor passes gives `yaw_response_tau=0.103 s`, versus the current 0.340 s. Open-floor yaw RMS fell 4.82° → 1.09°; on separate helix passes, 2 m/s yaw RMS fell 6.34° → 1.24°. Individual fits were 0.146 s at 1 m/s and 0.080 s at 2 m/s. | A single smaller lag explains most of this Gazebo gap, but the speed dependence and real-car lag need measurement before changing the training nominal. Current randomization spans 0.204–0.544 s, so the fitted Gazebo value is outside it. |
| Yaw sensitivity | Replaying v8 in Numba on the same 25 initial poses with the fitted yaw nominal produced 5/25 finishes versus 4/25 at the trained nominal. Two former finishes failed, and three former failures finished; one of the lost finishes crashed on the helix. | Correcting this one motion parameter does not make v8 reliable. The changed episode paths are sensitive to yaw dynamics, so yaw parity still matters. This is a sensitivity probe, not a Gazebo calibration or retraining result. |
| Cloud delivery | Gazebo's camera SDF requests 12 Hz. With the driver absent, three 10–28 s probes observed 6.0–6.5 Hz on the raw ROS camera topic and 2.0–3.4 Hz on the processed topic. Processed-cloud arrival gaps had 0.58–0.85 s p90, although capture-to-receipt age for delivered frames was only about 0.07–0.08 s p90. With the driver running, a 30 s simulated run yielded about 31 observed new processed clouds, a 0.81 s median gap, and 1.14 s cloud-age p90. Pose age p99 stayed below 0.04 s. The driver holds zero velocity when cloud age exceeds 0.5 s. Its stale state occupied 44% of a 150 s run, and 39% and 38% of two 30 s runs with and without the checker's own cloud subscriber. | Training was running concurrently and contended for resources, artificially reducing camera FPS. These probes describe the contended setup, not an intrinsic camera or Gazebo throughput limit. They do not justify changing camera settings, processing, or `cloud_timeout`. Stale-input stops also confound the full policy run. |
| Full Gazebo lap | On seed 5001, one v8 run took about 109 s to traverse the helix and timed out at 150 s after the tunnel, 0/3 hoops. The trace shows repeated forward and reverse commands near the helix rails and 44.8% stale ticks there. | This is a failure example under resource contention, not evidence of a Gazebo transfer failure rate or a measured yaw or contact effect. |
| Reverse | `plant.py` and the Arduino delay a direction change until the tachometer has read stopped. `sim_vehicle_node.cpp` slews a signed target through zero and has no corresponding direction-wait state. | Recovery timing in Gazebo, Numba, and the real controller is not the same. This is a code-level discrepancy, not yet measured by a paired reverse pass. |

The Numba course grid comes from the same SDF collision and visual geometry as
Gazebo, but Numba resolves chassis contact by moving the body back and sliding
it along an axis. Gazebo uses wheel joints and ODE contacts. Matching static
geometry therefore does not establish contact dynamics, especially at the
helix rail and deck seams.

These Gazebo measurements used the headless Docker renderer. Concurrent
training reduced camera FPS during the timing probes. The full policy result
is one seed and one start, so it is a failure example rather than a Gazebo
finish-rate estimate. Numba and Gazebo start offsets were drawn independently;
their aggregate outcomes can be compared, not their individual trajectories.
The raw records are in ignored `runs/gazebo/` files:
`seg_gap_20260927_221858.json`, `surfaces_20260927_222232.json`,
`surfaces_20260927_222612.json`, `validate_20260927_223942.json`,
`validate_20260927_224913.json`, `validate_20260927_225300.json`,
`numba_20260927_175750.json`, `numba_20260927_175845.json`,
`numba_20260927_181318.json` (fitted yaw nominal),
`cadence_20260927_230504.json`, `cadence_20260927_230745.json`, and
`cadence_20260927_231121.json`.

## What must pass before a transfer claim

1. Re-evaluate the exported policy on more held-out layouts and start offsets
   in Numba, then compare matched layout seeds and starts in Gazebo under an
   unloaded setup. Report finish rates and failure regions with intervals.
   The 4/25 result comes from a small, selected sample, but the checkpoint's
   19/64 held-out finishes also leave reliability unproven.
2. Repeat open-floor yaw passes at several speeds in both steering directions.
   Timestamp the command and align the pose traces to it before fitting lag:
   the current checker starts its trace clock at the first pose sample after
   the command. Confirm any fit on the helix and in real-car step-steer logs.
   The 0.103 s fit is a Gazebo diagnostic, not a real-car measurement; the
   real-car step response was undersampled for a precise time constant.
3. Run paired forward-to-reverse passes and rail contacts. Match time to zero,
   reverse onset, reported tachometer sign, actual travel direction, impact
   speed loss, and whether the vehicle can back off a rail. The tachometer is
   direction-blind, so its reported sign alone cannot establish direction.
   Update the Numba or Gazebo model only against those traces, with the Arduino
   behavior as the car reference.
4. Repeat the lower helix scan check across poses and layouts. Compare policy
   actions as well as the steering prior where bins disagree; the small prior
   difference at one pose does not establish policy insensitivity.
5. If comparing full Gazebo laps, first stop concurrent training and measure
   capture-to-policy age and stale fraction under the intended workload.
   Treat the current FPS measurements as confounded; they call for no camera
   or timeout changes.
6. Once simulator results support it, use a low-speed real-car run log to check
   sensor age, yaw response, steering authority, tachometer sign, and helix
   entry. Real-car actuation requires a person at the bench with the E-Stop
   remote immediately before launch.

## Reproducing the measurements

Run `validate.sh` in the built sim container. These write traces under
`runs/gazebo/` (ignored by Git):

```bash
./validate.sh --seg-gap --seeds 201 --poses 1 --helix-poses 5
./validate.sh --cadence --seconds 20
./validate.sh --flat-surfaces
./validate.sh --helix-surfaces
./validate.sh --policy runs/v8/policy.npz --seeds 5001 --starts 1 --timeout 150
python3 parity_eval.py runs/v8/policy.npz --seeds 201,202,208,218,5001 --starts 5
python3 fit_yaw_parity.py runs/gazebo/surfaces_20260927_222612.json --check runs/gazebo/surfaces_20260927_222232.json
```

The Gazebo validator now saves policy actions, commands, tachometer speed,
pose, scan, and stale state every 0.2 s. It subscribes to the cloud only for
`seg-gap` or when `--monitor-cloud` is requested, so ordinary policy validation
does not add another full-cloud consumer.
