# How the capture works, and the traps it avoids

## Pieces

| File | Runs | Does |
|---|---|---|
| `scripts/video.sh` | host | Parses options, exports .zip policies, calls rollout.py, manages the `cfr-video` container, launches the stack and bridge, runs capture.py, encodes, and copies out. |
| `scripts/rollout.py` | host (`rl/obstacleRacer/.venv`) | Rolls out `policy.npz` in the course's numpy sim and saves `track.npz` (t, pose, v, vcmd, steer, progress) plus `track.npz.json` (outcome, dt, labels). |
| `scripts/capture.py` | container | `patch-world` edits the installed world. `gazebo` and `replay` film, writing `frames/`, `list.txt` and `summary.json`. |

## Conventions that matter

- **Pitch sign.** `plant.py` (obstacleRacer) stores pitch nose-UP positive.
  Gazebo, SDF and ROS use nose-DOWN positive. rollout.py negates it once, so a
  track's pose is already Gazebo convention. Roll matches in both. If you
  replay plant state by other means, flip pitch too; `gazebo_check.py` does
  the same.
- **Frames.** Both numpy sims work in Gazebo world coordinates: obstacleRacer
  directly, and formulaOne's track is built from the world's path. So a
  replay pose goes straight into `set_pose`. The Speed Course is flat
  (z, pitch and roll are 0).
- **Cameras.** The chase camera is its own static model, re-posed each frame
  from the car pose with smoothed yaw and z. A camera fixed to the car hides
  pitch and roll, because the car stays still in frame and the world tilts.
  The ZED view is body-fixed on purpose, as on the car.

## Replay mechanics

Gazebo is paused. The real car is moved to (80, 80). A static `ghost` model
carries the car's visuals and a ZED camera. For each track step:
`set_pose(ghost)`, `set_pose(chasecam)`, then `ControlWorld multi_step = dt/1 ms`.
Cameras run at 1/dt Hz, so each step renders exactly one frame. It waits for
a newer chase stamp and re-steps if one was dropped. All of this goes through
`ros_gz_bridge` services (ControlWorld, SetEntityPose, SpawnEntity,
DeleteEntity); the image has no gz python bindings.

## Gazebo-mode mechanics

After setup, the capture pauses the world and runs it in 10 ms steps
(`--step-ms`). Before each step it moves the chase camera (and, on the Speed
Course, the film ZED) to the car and waits for the move to land. The car's
pose is carried forward from its 30 Hz stamp to mid-step with odometry speed
and yaw rate. Wall time is paced to `--rtf`, because the driver and segmenter
run in wall time and need that slack to keep up with sim time. The driver only
sees sim time, so for it this is the same as the world file's real-time cap.

The first version moved the cameras asynchronously while the world ran free.
Each move landed a varying few ms late, so the car crept across the frame and
snapped back ("rubber-banding"): ±15% in the car's apparent size at 5 m/s.

## Traps (each cost a failed run once)

- **Best-effort image subscriptions stall replays.** A 1.5 MB image is
  fragmented, and under load best-effort drops it. Each pose renders once, so
  the wait never ends. Subscribe reliably (depth 10); capture.py does.
- **Unused rendered sensors block every step.** Gazebo's Sensors system holds
  the step until due sensors render. The car's rgbd ZED (depth + cloud) made
  replays about 30× slower. So replay and Speed Course runs launch
  `sensors:=false` and patch the Sensors plugin into the world themselves.
- **Never send a partial `gz.msgs.Physics` to `set_physics`.** It reset
  other physics fields: the car floated up upside down. Cap the real-time
  factor in the world file instead (`patch-world --rtf`).
- **CPU starvation trips driver watchdogs.** See Runtime in SKILL.md: cap RTF.
  Measured on 2026-09-26 with a 4 s sim window: a best-effort Python
  subscriber to `cloud_registered` got 15 of 48 clouds (gaps up to 1.2 s),
  and a reliable one got all 48. Stepping and free running dropped the same.
- **Stopping a launch:** `docker restart` is the only clean way. Killing
  `ros2 launch` orphans gz and the nodes, and `pkill -f` inside
  `docker exec bash -c "..."` matches (and kills) its own shell.
- **Git Bash:** `docker cp` fails on Windows paths, and MSYS rewrites
  container paths. video.sh sets `MSYS_NO_PATHCONV=1` and copies files with
  `docker exec -i ... cat >`.
- **The obstacle start light** often isn't detected in sim. The capture turns
  the signal green, then falls back to `/obstacle_racer/manual_start` after
  10 s. formulaOne uses `/formula_one/manual_start` directly.

## Extending

A new course needs an entry in `capture.COURSES` (world name, SDF, chase
distance), bounds in `course_bounds`, a launch branch in video.sh, and a
rollout function in rollout.py. A new driver for an existing course needs
its Gazebo launch line in video.sh and, for replay, a rollout function that
yields the same track fields.
