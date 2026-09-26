# Bale-Following RL

A PPO policy that drives the simulated Slash around the speed course using
bale observations. It can replace `path_follower_node` in simulation.

## How it plugs into the existing stack

The policy publishes `geometry_msgs/Twist` on `/cmd_vel`, using the existing
drive chain. Episode resets use `teleport_api.py` and Gazebo WorldControl.

## Two things the simulation forces on this design

**Pose comes from `/world/<world>/dynamic_pose/info`, not `/zed/zed_node/odom`.**
The Ackermann plugin's wheel odometry ignores teleports. After a reset,
`training.launch.py` instead bridges Gazebo's ground-truth pose. The bridge
drops entity names; index 0 is the Slash, the world's only dynamic model.

**Episodes follow wall time.** WorldControl `multi_step` has corrupted the
Gazebo server heap in testing, so the environment sleeps between steps. It
sends `pause: false` once at startup; otherwise `/clock` stays frozen and
drive commands appear stale.

## Observations

`config.yaml` selects `scan_source: cloud`, which projects the simulated ZED
point cloud into range bins through the same scan code used on the car. The
analytic option ray-casts against bale boxes parsed from the SDF; it is useful
for comparisons but gives the policy exact geometry unavailable on the car.
Collision checks use bale footprints without a Gazebo contact sensor.

## A second objective: the lap racer

This README describes the corridor-following policy. A parallel stack in the
same directory (`lap_env.py`, `lap_reward.py`, `lap_track.py`, `train_lap.py`,
`config_lap.yaml`) optimises lap time round the planned loop instead --
signed arc length against the clock, body-to-bale clearance rather than a
ray fan from the car's centre, the servo's rate as the action, and a recovery
mode the policy is expected to reverse out of. See
[LAP_RACER.md](LAP_RACER.md). Nothing below changes.

## Setup

```bash
cd rl/bale_follower
python3 -m venv --system-site-packages .venv   # --system-site-packages gives us rclpy
source .venv/bin/activate
pip install -r requirements.txt
```

`--system-site-packages` is required: `rclpy` comes from `/opt/ros/jazzy`, so
source the ROS environment before activating the venv.

For a CUDA machine, install the matching `torch` wheel instead of the default
CPU one.

## One-command train / test

```bash
cd rl/bale_follower
./launch_training.sh --total-timesteps 100000   # sim up -> PPO -> sim down
./test_policy.sh --episodes 5                   # sim up -> metrics -> sim down
./test_policy.sh --smoother                     # A/B: route through the CasADi smoother
```

Both wrappers start `training.launch.py` themselves, wait for the teleport
API and pose bridge to be ready, and tear the stack down on exit. Set
`CFR_USE_RUNNING_SIM=1` to run against a simulation you started yourself.
Extra arguments pass through to `train.py` / `evaluate.py`; `--max-speed`
and `--traction` on `test_policy.sh` override the trained caps.

`evaluate.py` publishes the policy's commands directly, matching what
`run_policy.py` deploys. `env.py` enforces the vehicle's real limits during
training -- servo steering slew, traction clamp and friction circle, all from
`config.yaml` -- so a v6-or-later policy cannot learn commands the drivetrain
will not deliver, and needs no filtering downstream. `--smoother` routes
through `casadi_smoother.py` instead, which is what pre-v6 checkpoints
require and what `path_racer.py` uses for its planned reference.

The observation passes through a simulated ZED 2i model (`zed_sim.py`): 110
degree FOV, range-squared stereo noise, dropout -- see "Observations" below
for why the underlying ranges are still analytic.

## Train (manual, two terminals)

Start the simulation:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash                       # from the repo root
ros2 launch cfr_arduino_bridge training.launch.py
```

Run exactly one instance. Several Gazebo servers publishing to the same
topics produce poses that jump between worlds, which looks like wild physics
rather than the process-management problem it is.

Then, in another terminal:

```bash
cd rl/bale_follower
source /opt/ros/jazzy/setup.bash && source .venv/bin/activate
python train.py --total-timesteps 20000
```

Start small. Confirm the loop runs end to end and writes checkpoints before
committing to a long run.

## Current models

Two independent drivers share this directory. They solve different problems
and the right one depends on whether the course geometry can be trusted.

**`path_racer.py` — planned-line racer. Use this for a fast lap.**
`course_path.py` plans a racing line and minimum-time speed profile offline
from the SDF; a CasADi MPC tracks it from a pose estimate. Measured **29.95 s
best lap, six consecutive laps, zero recoveries**. Needs the map and a pose
(sim ground truth today, QuestNav on the Orin via `--pose-msg odom`).

```bash
python course_path.py --plot          # regenerate the line (only after a course change)
python path_racer.py                  # race it
```

**`checkpoints_v6/best_model.zip` — RL policy. Use this as the fallback.**
Drives from a simulated ZED forward scan with no map and no global pose, so
it still works when the geometry or localization cannot be trusted. Measured
**112-120 m mean over two runs, ~130 m on clean episodes, 1-2 collisions in
5** (the lap is 110.1 m); roughly half the racer's speed, capped at 2.0 m/s.

Run it raw — no CasADi smoother. v6 trains against the actuator limits, so
its commands are already executable, and the smoother only adds lag the
policy never saw (91.5 m and 2/5 collisions with it, versus 120.5 m and 1/5
without). `--smoother` exists for pre-v6 checkpoints, which do need it.

```bash
python run_policy.py --checkpoint checkpoints_v6/best_model.zip
```

v6 is trained against the real actuator limits — servo steering slew
(3.5 rad/s), traction clamp and friction circle — so its commands are
executable on hardware. Earlier checkpoints are not: v1-v4 were trained
against a mis-modelled vehicle, and v5 additionally assumed instantaneous
steering. **Do not deploy anything before v6.**

## Watch it drive: validate.sh

```bash
cd rl/bale_follower
./validate.sh
```

Opens three terminal windows that together replace the manual setup from
"Run a trained policy" below:

1. **Simulation stack** -- `simulation.launch.py websocket:=true`: the full
   stack with camera bridges plus the gzweb WebSocket server on port 9002.
   Without that flag the browser viewer sits at "Simulation disconnected".
2. **Glue** -- the two things that launch file lacks for RL: the ground-truth
   pose bridge the policy reads, and
   `ros2 param set /path_follower keep_auto_active_when_idle false` (retried
   until the node is up) so path_follower stops publishing idle zeros that
   fight the policy for `/cmd_vel`.
3. **Policy** -- waits for the pose topic, then runs `run_policy.py` with the
   default demo cap of **1.5 m/s** -- the measured best deterministic
   configuration for the v3 checkpoint (see REPORT.md); at the trained
   4.0 m/s cap the raw mean action floors the throttle and crashes.

Then open the viewer: `cd web/gzweb-viewer && npm run dev` and browse to
<http://localhost:5173>.

Options:

```bash
./validate.sh --max-speed 2.0        # extra args pass through to run_policy.py
./validate.sh --no-smoother          # raw policy commands, no CasADi filtering
DEMO_CHECKPOINT=checkpoints_v2/final_model.zip ./validate.sh
```

Stopping: close window 1 (it owns the Gazebo server) or Ctrl-C in it; the
other windows notice and exit. Each window pauses with a "press enter to
close" prompt after its process exits so errors stay readable.

Prerequisites and refusals:

- workspace built (`colcon build`), venv set up (see Setup), and the
  checkpoint present -- checked up front with a clear error for each.
- **refuses to start if a Gazebo server is already running** (`pgrep -f "gz
  sim"`): duplicate servers publish to the same topics and corrupt each
  other. Stop the old one (`pkill -f "gz sim"`) and rerun. To demo against a
  sim you started yourself, skip window 1 and run the roles directly:
  `./validate.sh _glue` and `./validate.sh _policy` in two terminals.
- needs a graphical session; it picks the first of gnome-terminal,
  x-terminal-emulator, konsole, xterm. (Snap-leaked `GTK_PATH` /
  `LD_LIBRARY_PATH` from VS Code terminals are scrubbed automatically --
  they otherwise crash gnome-terminal with a GLIBC symbol error.)

Note: the runtime `param set` in window 2 relies on the parameter callback
added to `path_follower_node.cpp`; on a workspace built before that change,
rebuild `cfr_arduino_bridge` or the node keeps publishing zeros regardless.

## Run a trained policy (manual)

```bash
ros2 launch cfr_arduino_bridge training.launch.py
python run_policy.py --checkpoint checkpoints/final_model.zip
```

To watch it against the full stack (cameras, web viewer) use
`simulation.launch.py` instead -- but that launch has no ground-truth pose
bridge, so add one:
`/world/cfr_speed_course/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V`.

`path_follower_node` also publishes to `/cmd_vel` -- including zeros at 20 Hz
while idle -- so set `keep_auto_active_when_idle` to false, or it will fight
the policy for control:

```bash
ros2 param set /path_follower keep_auto_active_when_idle false
```

## Verifying the pieces

Most of this needs no training at all:

```bash
python bale_geometry.py --plot     # parsed bales + course polyline + spawn pose
python reward.py                   # reward favors progress over collision
```

Then with a simulation running, check episode mechanics before spending PPO
wall-clock time -- `env.reset()` should teleport the car and return an
observation, and a few hundred random-action steps should produce finite
rewards and detect collisions when the car inevitably hits a bale.

## Design notes

- **Actions**: normalized to `[-1, 1]` in both components and mapped to real
  units inside the env. Keeping the range symmetric matters: SB3's policy is
  a zero-centered Gaussian, so a speed mapping whose zero action means
  standstill clips half the distribution and the car never explores moving.
  Speed spans `[-reverse_speed, max_speed]` linearly: slow reverse exists so
  the policy can unstick itself when nosed against a bale in a hairpin
  (forward-only made that pose terminal), while the reward still pays
  forward progress only. `run_policy.py` additionally carries a scripted
  reverse-out recovery (`--no-recovery` disables) for checkpoints trained
  without reverse. Steering is converted to a yaw rate through the same bicycle model
  `cmd_vel_to_drive_node` inverts (`wheelbase` 0.324 m, `max_steering_angle`
  0.40 rad); those constants are mirrored in `env.py` and must stay in sync
  with `jetson/cfr_arduino_bridge/config/arduino_bridge.yaml`.
- **Observations**: normalized lidar bins plus speed and yaw rate. Course
  progress is deliberately *not* observed -- it is reward-only, so the policy
  never sees privileged information a real onboard sensor could not provide.
- **Reward**: distance travelled in the direction the car was facing, minus a
  penalty for coming within `safe_clearance` of a wall, minus a centering
  penalty for imbalance between the nearest bale left and right (both walls
  used, side distances capped at 1.5 m so open hairpins aren't punished),
  minus steering jerk, with a large penalty and episode termination on
  collision. The centering term is reward-only ground truth, like progress.
- **There is no course centerline, and progress is not measured against one.**
  The bale order in `speed_course.sdf` is DXF drawing order, not a traversal
  path: consecutive indices jump up to 26 m between disjoint wall segments,
  and `jetson/scripts/generate_speed_course.py` discards the source outline.
  The course is a corridor, so "go forward without touching a wall" is a
  sufficient objective -- the walls themselves prevent cutting corners.
- **The course** is a serpentine: three concentric bale walls forming two
  drivable corridors roughly 0.95 m wide, joined by hairpins at each end. The
  car is 0.55 x 0.30 m, so clearances are tight; `safe_clearance` has to stay
  well under half a corridor width or every pose is penalized.

## Future work

- **Parallel environments.** Training runs a single environment because one
  Gazebo instance is the bottleneck. Going wider means N independent Gazebo
  instances on separate `GZ_PARTITION` / ROS domain IDs and teleport ports.
- **Reward refinement.** Offsetting the target line inward from the bales, as
  described above.
