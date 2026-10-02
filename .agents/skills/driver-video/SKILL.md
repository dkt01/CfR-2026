---
name: driver-video
description: Make an MP4 video of a CfR-2026 driver (an obstacleRacer or formulaOne policy, any run or checkpoint) driving the Obstacle Course or the Speed Course, filmed in Gazebo with a chase camera, the ZED's view, a top-down minimap and telemetry. It either runs the real stack under Gazebo physics, or replays a lap from the driver's numpy (numba) training sim through Gazebo's renderer. Use this whenever the user asks for a video, clip, recording, movie, animation or "show me" of a policy, driver, RL run or checkpoint driving a course, wants to see how a driver behaves or where it fails, wants footage of a successful lap, or asks to compare what the numpy sim and Gazebo do with the same policy. Prefer it over hand-rolling camera models, gz services and ffmpeg.
---

# Driver video

`scripts/video.sh` does the whole job on the host. It exports the policy if
needed, rolls it out in the numpy sim (replay mode), brings up a dedicated sim
container (`cfr-video`, separate from sim-launch's `cfr-sim`), films, encodes
and copies the MP4 out:

```bash
S=.agents/skills/driver-video/scripts
$S/video.sh gazebo --course obstacle --policy rl/obstacleRacer/runs/v6/policy.npz --seed 104
$S/video.sh replay --course obstacle --policy rl/obstacleRacer/runs/v6/best_model.zip --seed 104
$S/video.sh gazebo --course speed    --policy rl/formulaOne/bestModel/v12/policy.npz
$S/video.sh replay --course speed    --policy rl/formulaOne/bestModel/v12/policy.npz --pick any
$S/video.sh gazebo --course obstacle --policy rl/obstacleRacer/runs/v8/policy.npz --segmentation
$S/video.sh stop
```

It runs for many minutes, so start it with `run_in_background` and wait for
the completion notice. The output lands at `<policy dir>/video/<run>_<policy>[_seed<N>]_<mode>.mp4`
(or `--out`), with a `_summary.json` beside it. Run `video.sh` with no
arguments, or read its header, for every option. Add `--segmentation` to either
mode on either course to show the ZED point cloud classified by the same
compiled segmenter used on the car. It adds sensor rendering and processing
work, so use it when the user asks to see what the car perceives.

## Pick the mode from what the user wants to see

The two modes answer different questions, and the video says on screen which
one it is. Never let a replay pass for Gazebo.

- **gazebo**: "how does it really drive". This is the real ROS stack (driver
  node, segmenter, lap counter) under Gazebo's contacts and tires, the check
  the car cares about. The run ends on the lap counter's done, a rollover, no
  progress for 12 s, or `--timeout`. It may well fail: the numpy sims and
  Gazebo still disagree in places. For example, v6 wedges on the Obstacle
  Course helix in Gazebo but laps it in numpy. That failure is often exactly
  what the user needs to see.
- **replay**: "show a successful lap", or "what did it learn". It rolls out
  `--episodes` starts in the training sim, keeps the one `--pick` asks for
  (fastest finish by default), and re-poses a ghost of the car frame by frame
  with Gazebo paused. Gazebo's physics plays no part. It exits non-zero if no
  episode finished. Then try more episodes, another seed, `--pick any` (the
  longest failed run), or gazebo mode.

If the user asks for "a successful run" and doesn't name a mode, the numpy sim
is usually the only place a lap exists. Make the replay and tell them plainly
that it is numpy physics, not Gazebo's. If they asked for Gazebo physics and
it fails, show that video and say where and how it failed. Don't swap in a
replay unasked.

## Choose the driver and layout

- **Obstacle Course** (`rl/obstacleRacer`, one lap, three hoops). `runs/<v>/best_model.zip`
  is the checkpoint `train.py` kept for held-out finishes. Rank runs by the
  best entry in `runs/<v>/history.json`: `train.finish`, then `train.progress_m`.
  `train.finish_by_seed` says which layout seeds that checkpoint finishes most
  often; pass one as `--seed` for a video likely to complete. The policy's
  `config.yaml` must be the one it was trained with. The node refuses a
  mismatch, and v6 and v7 observations differ.
- **Speed Course** (`rl/formulaOne`, two laps then stop). Committed policies
  are in `bestModel/<v>/`. `evaluate.py <policy.npz> --config <config.yaml>`
  shows whether any numpy episode finishes. v12's never did (it grazes a bale
  at about 6 s, which the numpy sim counts as a crash), so its replay needs
  `--pick any` and shows only that. Gazebo mode is the way to film v12.
- bale_follower was trained in Gazebo and has no numpy sim. It isn't covered.

Paths can be relative to the repo. `runs/` is gitignored, so from a worktree
the script also looks in the main checkout. A `.zip` is exported with the
course's `export_policy.py` first. The host python is
`rl/obstacleRacer/.venv` (override with `PYTHON=`), which covers both courses'
numpy sims.

## What the video shows

The default frame is 1280×720:
- **Left:** a chase camera that follows the car's heading and height with the
  horizon kept level, so the car's own pitch and roll are visible on the
  ramps, helix and bank.
- **Top right:** the ZED's view.
- **Below that:** an overhead photo of the course, taken at the start, with
  the path so far drawn on it.
- **Telemetry:** time, speed, commanded speed or the speed cap, steering, and
  hoops or laps.
- **Caption:** driver, course, and which physics was used.

With `--segmentation`, the frame widens to 1600×720. The ZED view and a
class-colored point cloud view sit side by side. Its legend identifies ground,
obstacle, hoop, car wash, and overhead. Dark pixels had no classified point.
In Gazebo mode on the Obstacle Course this uses the same noisy ZED cloud the
driver receives; in Speed Course Gazebo mode and replay it uses the filming
camera's RGB-D cloud at the ZED mount. Replay still uses numpy physics.

Playback is real time in sim time, whatever the real-time factor was. Gazebo
mode trims the wait for the start.

## Runtime and load

- Gazebo mode runs at `--rtf 0.1` by default: a 60 s lap takes about 10–15
  min. A training run usually holds the CPU, and at full speed the driver's
  0.5 s stale-sensor watchdog ("pose or cloud stale -- holding at zero speed"
  in `/work/sim.log`) keeps stopping the car and corrupts the run. On an idle
  machine `--rtf 0.3` is fine.
- A few holds remain even at 0.1 on the Obstacle Course. obstacle_racer_node
  reads the full `cloud_registered` cloud best-effort, and under load it
  receives only about 4 of the 12 clouds per sim second. Any gap over 0.5 s
  trips the watchdog. That is the driver as it really runs in Gazebo, not the
  capture, so count the holds (`grep -c stale /work/sim.log`, with `--keep`)
  and mention them when a run stutters.
- Replay renders at about 2–3 frames/s: roughly 8 min for a 65 s obstacle
  lap. `--limit 120` gives a quick 6 s look first.
- The first run creates the container and builds the workspace, which takes a
  few extra minutes. Later runs reuse it. `--rebuild` after changing
  `jetson/` code, since the container has its own install tree. The script
  stops the container afterwards unless `--keep`.

Before handing the video over, look at a frame or two:
`ffmpeg -ss <t> -i video.mp4 -frames:v 1 f.jpg`, then Read the jpg. Report the
outcome from `_summary.json` in your own words: finish time, or where and how
it failed.

## If something goes wrong

`references/internals.md` explains how the capture works and the traps it
avoids. Read it before changing the scripts, or when a run stalls or looks
wrong. The logs are in the container: `docker exec cfr-video tail /work/sim.log`
(and `/work/bridge.log`, `/work/driver.log`).
