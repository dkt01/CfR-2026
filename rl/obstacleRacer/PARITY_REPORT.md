# Obstacle Racer Numba–Gazebo parity audit

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
