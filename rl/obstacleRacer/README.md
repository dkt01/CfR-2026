# Obstacle Racer

An RL driver for the Obstacle Course. It drives one lap through all three
hoops, as fast as it can. It is trained in numpy, the way `rl/formulaOne`
trains the Speed Course, and Gazebo only checks the result.

## What the policy gets and gives

It gets only what the car can measure:
- the segmented ZED cloud, as a 36-bin blocking scan plus the nearest hoop
  and car-wash gates
- tachometer speed
- yaw rate
- its previous action

Two frames are stacked. `observation.py` builds this, for both training and
`obstacle_racer_node.py` on the car.

It gives a `DriveCommand`:
- steering = a map-free follow-the-gap prior + the policy's residual (full authority)
- speed from -1 to 3.5 m/s. The car has no brakes, so speed is limited by
  sight distance.

The policy (`ppo_policy.py`) keeps its Gaussian's mean inside the action range
(tanh) and its spread inside `train.log_std_range` (a sigmoid, so the spread
can always move). In v2 the unbounded mean ran to -6 at the bank, so every
sample clipped to a full stop and the slow crawl round it was never tried.
PPO sees rewards times `train.reward_scale` (0.1): unscaled, v3's value net
saturated in its first 2M steps and never learned.

The reward is privileged, computed in simulation only (`reward.py`):
- progress along a hand-placed centerline, per layout
- a time cost
- a bonus for each hoop and for finishing, and a potential-based nudge
  toward each hoop's center over the last 2.5 m before it (it hands back
  what it gave as the car crosses, so only threading pays)
- a cost for speed lost to contact; touching is not the end of a run
- terminal penalties for hard crashes (over 1.5 m/s lost in one step), hoop
  misses, leaving the course and, worst of all, stopping or circling. A stop
  within 3 s of touching something is "pinned" and costs a crash, so a gentle
  touch never costs more than a hard hit
- costs for grazing obstacles, steering chatter and speed-command chatter.
  These are charged on the sampled action, exploration noise included, so
  `reward.py` checks that noise at the starting std costs under half the time
  cost; at v3's weights it cost ~20/s and the policy held full lock to escape

A run that never reaches the hoops is neither paid nor charged for them: the
miss penalty needs the car at or past a hoop.

## How the course is modeled

`world.py` reads the Gazebo world's two geometry sources.

Collision primitives are the physics. They are tagged support or obstacle, by
name:
- ramps, the 24 helix segments and the 8.5° bank are tilted boxes
- the pothole board is boxes leaving 18 mm recesses, with bump cylinders
- the hoop and car-wash posts are cylinders
- the bank's plywood wall is boxes

Visual meshes are what the camera sees: hoop top bars, car-wash ribbons and
arches, the tunnel roof.

`course_model.py` bakes both into 2 cm grids. A static grid covers the whole
course, and a per-layout window covers the buckets, hoops and entrance gap.

`plant.py` is formulaOne's actuation chain plus a sprung body. It has heave,
pitch and roll on the Gazebo car's corner springs, rigid tires on the real
surfaces, and wheels that can leave the ground. Traction is mu times load. So
the car pitches up the ramp, goes light over the deck crest, rolls on the
bank, and bounces through the potholes.

Reverse follows the Arduino and last week's characterization
(`docs/characterization-results.md`). A reverse target while rolling forward
only coasts, because there is no braking. The car drives backwards 0.5 s
after it drops under the tachometer's 0.3 m/s floor, on the same drag curve.
Below about 1 m/s the target the car chases wanders (throttle dither), so a
slow crawl is not a precise tool. The speed the policy sees is signed by the
controller's direction, as `ArduinoStatus.speed` is.

A car that touches an obstacle is put back where it was clear and slides
along it, keeping the share of its speed the slide carries; head-on, it
stops and has to back off.

`sensor.py` ray-marches the visual grid from wherever the body put the camera,
following the surface out from under the car the way the segmenter does.

## Layouts and starts

Training uses 10 randomizer seeds (101–110). Four more seeds (201, 202, 208,
218) are held out, one per entrance slot. They come from
`obstacle_randomizer_node`'s own draw (`obstacle_layout_draw.py`), so a seed
means the same course in Gazebo.

Episodes start either:
- in the start box, (−0.7, 0) ± 0.1 m in x and y and ± 5° in heading, from
  rest. Evaluation always starts here.
- dealt part way round, with the same noise, the helix included: 35% of
  these start 0.5-6 m before a spot where a recent training episode failed,
  45% start 0.5-4 m before an obstacle picked uniformly from all nine (so the
  hoops, car wash and buckets get practiced however rarely the policy reaches
  them from the start box), and the rest anywhere.

Each eval also deals the car 2 m before every obstacle on the held-out
layouts and records whether it gets through; the dashboard shows that per
obstacle next to how often training met and failed at it.

Every episode is one full lap from its own start point. Progress wraps round
the loop at the timing line, all three hoops must be threaded during the
episode, and the lap bonus is paid only when the car gets back past the
point it started from (from the start box, that is over the timing line).

## Running

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe numpy scipy pyyaml matplotlib gymnasium stable-baselines3 numba torch
.venv/Scripts/python.exe selftest.py
.venv/Scripts/python.exe train.py --dir runs/v1
../../.agents/skills/rl-train/scripts/rl.sh tui --dir rl/obstacleRacer/runs/v1
.venv/Scripts/python.exe export_policy.py runs/v1/best_model.zip -o runs/v1/policy.npz
```

In the sim container, with the workspace built:

```bash
./validate.sh --seg-gap      # sensor model vs the real segmenter
./validate.sh --surfaces     # plant pitch/roll/z vs Gazebo on ramps, potholes, bank, gravel
./validate.sh --policy runs/v1/policy.npz   # held-out layouts x 5 noisy starts
```

To watch a run the way it happens on race day (the driver waits for the start
signal and takes the throttle off when lap_counter reports the lap):

```bash
# Gazebo: course, ZED, start signal detector, lap counter (1 lap), driver
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch rl/obstacleRacer/obstacle_racer_sim.launch.py \
    policy:=runs/v4/policy.npz green_after:=60
# or turn the signal green yourself:
ros2 service call /obstacle_randomizer/start_signal std_srvs/srv/SetBool "{data: true}"

# Send it to the Orin: the code to ~/software/obstacleRacer, and runs/v4's
# policy.npz (export_policy.py first) and config.yaml to its top level
jetson/scripts/syncSoftware.sh --racer-policy v4 --build

# The car, with ~/software/scripts/launch.sh --no-cmd-vel already up and the
# E-Stop in hand: start signal detector, lap counter, driver, recorder
ros2 launch ~/software/obstacleRacer/obstacle_racer_car.launch.py speed_scale:=0.3
```

Without `--racer-policy` the sync sends the code and the tree's config.yaml
only, which is enough for `driver:=prior`.

`obstacle_racer.launch.py` is the driver alone, for validate.sh.  The car
launch records with `--cloud-hz 0`: record_run.py otherwise drops the ZED cloud
to 1 Hz, and this driver steers from the cloud.

Other checks:

| Command | Checks |
|---|---|
| `python3 reward.py` | Episode-level incentives |
| `python3 centerline.py` | Every layout's line is clear of the walls |
| `python3 layouts.py --check` | Exported layouts match the randomizer |
