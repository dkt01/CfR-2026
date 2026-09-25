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
- speed from 0 to 3.5 m/s. The car has no brakes, so speed is limited by
  sight distance.

The reward is privileged, computed in simulation only (`reward.py`):
- progress along a hand-placed centerline, per layout
- a time cost
- a bonus for each hoop and for finishing
- terminal penalties for crashes, hoop misses, stalls and leaving the course
- costs for grazing obstacles and steering chatter

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
- dealt part way round, with the same noise, the helix included. Half of
  these start 0.5-6 m before a spot where a recent training episode failed,
  so the obstacles the policy cannot yet do get practiced. Progress and the
  finish bonus count from the episode's own start point.

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

Other checks:

| Command | Checks |
|---|---|
| `python3 reward.py` | Episode-level incentives |
| `python3 centerline.py` | Every layout's line is clear of the walls |
| `python3 layouts.py --check` | Exported layouts match the randomizer |
