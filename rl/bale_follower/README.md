# Bale-Following RL

A reinforcement-learning policy that drives the simulated Slash around the
speed course by reacting to the bales that line it. Trained with PPO against
the existing Gazebo simulation, and runnable in-sim as a drop-in alternative
to `path_follower_node`.

## How it plugs into the existing stack

The policy publishes `geometry_msgs/Twist` on `/cmd_vel` -- exactly where
`path_follower_node` publishes -- so the rest of the chain
(`cmd_vel_to_drive_node` -> `sim_vehicle_node` -> Gazebo's `AckermannSteering`
plugin) is untouched. Episode resets reuse `teleport_api.py`'s HTTP endpoint
and the `/world/cfr_speed_course/control` WorldControl service that the
browser viewer already uses for its reset button.

No changes to the C++ nodes, `package.xml`, or the colcon build.

## Two things the simulation forces on this design

**Pose comes from `/world/<world>/dynamic_pose/info`, not `/zed/zed_node/odom`.**
The Ackermann plugin dead-reckons odometry from wheel rotation, so it ignores
teleports completely -- after an episode reset it reports a pose with no
relation to where the car actually is. `training.launch.py` bridges Gazebo's
ground-truth pose instead. The bridge drops entity names, so the env reads
index 0 of the pose array, which is the Slash: it is the only dynamic model
in the world, and Gazebo publishes a model before its links.

**Episodes are paced against the wall clock, not stepped.** Driving Gazebo
deterministically through WorldControl `multi_step` corrupts the server's
heap (`malloc(): unaligned fastbin chunk detected`) and leaves the process
alive with dead service threads -- every later service call times out. So the
env sleeps between steps instead, which caps throughput at roughly
`control_hz` steps per second. The one WorldControl call that remains is a
single `pause: false` at startup: the headless server comes up paused even
with `-r`, and while paused `/clock` never advances, so `cmd_vel_to_drive`
treats every command as stale and holds the car at neutral.

## Observations are analytic, not from the ZED point cloud

**The simulated ZED produces no depth data today.** `speed_course.sdf` has no
`gz-sim-sensors-system` plugin and no `<sensor>` element -- the ZED2i on the
vehicle is cosmetic geometry, deliberately left that way pending a working
GPU-backed EGL/OpenGL context. `simulation.launch.py` bridges
`/zed/zed_node/point_cloud/cloud_registered`, but nothing publishes it.

So the "virtual lidar" observation is computed **analytically**: ray-cast from
the car's odometry pose against the 202 bale boxes whose exact poses and
dimensions are parsed straight out of the SDF. Same conceptual signal (range
to nearest bale per angular bin), no rendering dependency, deterministic and
fast. Collision detection works the same way -- the car's chassis footprint
tested against bale footprints with a separating-axis check -- so no Gazebo
contact sensor is needed either.

This is a real sim-to-real gap, and it matters: **the policy is trained on
ground-truth geometry, not noisy depth data.** Before this could run on the
physical car, the observation source has to be swapped for something derived
from the actual ZED point cloud, and the policy retrained or fine-tuned
against it. That is not a small step.

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
./test_policy.sh --no-smoother                  # raw-policy A/B baseline
```

Both wrappers start `training.launch.py` themselves, wait for the teleport
API and pose bridge to be ready, and tear the stack down on exit. Set
`CFR_USE_RUNNING_SIM=1` to run against a simulation you started yourself.
Extra arguments pass through to `train.py` / `evaluate.py`; `--max-speed`
and `--traction` on `test_policy.sh` override the trained caps.

`evaluate.py` routes the policy through the CasADi command smoother
(`casadi_smoother.py`): a short-horizon optimizer that tracks the policy's
(speed, steering) target subject to a friction-circle traction model, so the
car brakes before corners it cannot carry at speed instead of understeering
through them. Training applies the same limits greedily in `env.py`
(`traction` in `config.yaml`), so the policy never learns commands the
drivetrain won't deliver. The observation passes through a simulated ZED 2i
model (`zed_sim.py`): 110 degree FOV, range-squared stereo noise, dropout --
see "Observations" below for why the underlying ranges are still analytic.

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

## Watch it drive: demo.sh

```bash
cd rl/bale_follower
./demo.sh
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
./demo.sh --max-speed 2.0        # extra args pass through to run_policy.py
./demo.sh --no-smoother          # raw policy commands, no CasADi filtering
DEMO_CHECKPOINT=checkpoints_v2/final_model.zip ./demo.sh
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
  `./demo.sh _glue` and `./demo.sh _policy` in two terminals.
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

- **Real depth pipeline.** To make the simulated ZED actually produce a point
  cloud: add a `gz-sim-sensors-system` world plugin with `<render_engine>ogre2`,
  add an `rgbd_camera` sensor under the ZED2i link with ZED-like intrinsics
  (~110 deg HFOV), and give Gazebo an EGL context (a real GPU, or `Xvfb` plus
  software rasterization). Verify with
  `ros2 topic hz /zed/zed_node/point_cloud/cloud_registered`. Then replace the
  analytic lidar in `bale_geometry.py` with a point-cloud-to-range-bin
  projector.
- **Parallel environments.** Training runs a single environment because one
  Gazebo instance is the bottleneck. Going wider means N independent Gazebo
  instances on separate `GZ_PARTITION` / ROS domain IDs and teleport ports.
- **Reward refinement.** Offsetting the target line inward from the bales, as
  described above.
