# Deploying the formulaOne policy to the Orin

How to get a trained Speed Course policy from this repo onto the car and
driving. Written for whoever is doing the deployment, at the bench, with the
car in front of them.

---

## 0. Before you start — read this part

**Launching the driver arms the actuators on a physical vehicle.** The last
command in this document makes the car drive. Every time:

- Have the **E-stop remote in your hand** before you run it, not beside you.
- Have the car **on blocks or in a clear run-off area** for the first launch
  of any new policy.
- **Never** run the launch step unattended, over a flaky SSH link, or in the
  background.
- Start with `speed_scale:=0.3` (see step 6). A policy that is going to do
  something surprising does it just as clearly at a third of the speed.

The car has **no brakes**. Deceleration is coast drag only — from 5.2 m/s it
needs about 9.3 m to reach 2.5 m/s and roughly 15 m to stop. Plan the run-off
accordingly.

---

## 1. What actually runs on the car

| | |
|---|---|
| Driver | `formula_one_node.py`, the same file the simulator runs |
| Brain | one `policy.npz` — a 35→128→128→2 MLP, ~170 KB |
| Maths | **numpy only**. No torch, no scipy, no gymnasium on the car |
| Inputs | `/zed/zed_node/pose` (map-frame x, y, yaw) and `/arduino_bridge/status` (speed) |
| Output | `/drive_cmd` (`cfr_interfaces/DriveCommand`) — **never** a Twist |

The policy has **no camera or point-cloud input**. It drives from a known map
read against the ZED's corrected pose. That means **everything rides on
localisation**: if the pose drifts, the car will drive into a bale with
complete confidence. This is the single biggest risk on the real car and it is
not exercised by any simulator test, because there the pose is ground truth.

Python files the node needs beside it: `track.py`, `observation.py`,
`policy.py`, `baseline.py`, plus `config.yaml` and the policy `.npz`.

---

## 2. Choose the policy, and check it before it leaves your machine

```bash
cd rl/formulaOne
.venv/bin/python evaluate.py runs/<run>/policy.npz     # offline score
./validate.sh --check 450 --policy runs/<run>/policy.npz   # must print PASS
```

**Do not deploy a policy that has not printed `PASS` in Gazebo.** The offline
evaluator scores it against the model it was trained on; only `validate.sh`
scores it against real contacts, real tire friction and the real nodes.

Note the observation width, because it has to match the config you ship:

```bash
.venv/bin/python -c "from policy import NumpyPolicy; \
  print(NumpyPolicy.load('runs/<run>/policy.npz').obs_dim)"
```

35 means `observe_accel: false` in `config.yaml`; 36 means `true`. **A
mismatch is a hard failure at the first tick** — the node will not start. Each
run directory carries the `config.yaml` it was trained with; if in doubt ship
`runs/<run>/config.yaml` rather than the one at the top of the tree.

---

## 3. Orin layout

The node resolves the course files **relative to the repo root**, which it
takes as two directories above itself (`repo_root` parameter). So mirror the
repo's own shape on the Orin and everything resolves with no arguments:

```
~/cfr/
  jetson/               <- synced by syncSoftware.sh
    cfr_arduino_bridge/
      worlds/speed_course.sdf          <- the bale positions
      config/speed_course_path.json    <- the centerline
  rl/formulaOne/        <- copied in step 5
```

`~/ros2_ws` stays the colcon workspace, as it is today.

---

## 4. Sync and build the ROS side

From the dev machine, in the repo root:

```bash
jetson/scripts/syncSoftware.sh --dir ~/cfr/jetson --dry-run    # look first
jetson/scripts/syncSoftware.sh --dir ~/cfr/jetson --build
```

Defaults are `ORIN_HOST=tejam@192.168.55.1` and `ORIN_WS=~/ros2_ws`; override
with `--host` / `-w` or the `ORIN_HOST` / `ORIN_WS` environment variables.
Source only — the host is x86_64 and the Orin is aarch64, so no build
artifacts are transferred.

This builds `cfr_interfaces` (which defines `DriveCommand` and
`ArduinoStatus`) and `cfr_arduino_bridge`. **The policy node needs
`cfr_interfaces` to exist, so this step is not optional even though the node
itself is plain Python.**

---

## 5. Copy the policy stack

`syncSoftware.sh` only syncs `jetson/`. The policy lives outside it and has to
be copied separately:

```bash
rsync -av --exclude '__pycache__' --exclude '.venv' --exclude 'runs' \
      rl/formulaOne/ tejam@192.168.55.1:~/cfr/rl/formulaOne/

# and just the one policy you chose
rsync -av rl/formulaOne/runs/<run>/policy.npz \
          rl/formulaOne/runs/<run>/config.yaml \
          tejam@192.168.55.1:~/cfr/rl/formulaOne/runs/<run>/
```

**Optionally copy the track cache too.** `track.py` builds a distance field
over every bale on first use and caches it; shipping the cache skips that work
on the Orin:

```bash
rsync -av rl/formulaOne/.cache/ tejam@192.168.55.1:~/cfr/rl/formulaOne/.cache/
```

The cache key is a hash of the world SDF and the track settings, so a stale
cache is ignored rather than used — it cannot give you the wrong course.

---

## 6. Run it

Two terminals on the Orin. **E-stop in hand.**

**Terminal 1 — the Arduino bridge:**

```bash
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch cfr_arduino_bridge arduino_bridge.launch.py use_cmd_vel:=false
```

`use_cmd_vel:=false` **matters.** `cmd_vel_to_drive_node` republishes
`DriveCommand` on a timer whether or not anything is feeding it, so leaving it
up puts a second publisher on `/drive_cmd` and the Arduino acts on whichever
message arrived last. The car then drives on a mixture of the policy and a
stream of neutral commands, with no obvious point of failure. It just drives
badly.

**Terminal 2 — the policy, first run, at a third speed:**

```bash
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch ~/cfr/rl/formulaOne/formula_one.launch.py \
     policy:=$HOME/cfr/rl/formulaOne/runs/<run>/policy.npz \
     config:=$HOME/cfr/rl/formulaOne/runs/<run>/config.yaml \
     use_sim_time:=false rviz:=false speed_scale:=0.3
```

- `use_sim_time:=false` — **required on the car.** Left at its `true` default
  the node waits on a `/clock` that never comes.
- `speed_scale` multiplies every speed command. Work up 0.3 → 0.5 → 1.0 over
  separate runs, checking the line each time.
- `laps:=N` overrides the two laps in the config.

The car waits for the start signal and will not move until it gets one. Either
show it the real green signal, or release it by hand:

```bash
ros2 service call /formula_one/manual_start std_srvs/srv/SetBool "{data: true}"
```

---

## 7. What a good run looks like

The node prints a telemetry line every 3 seconds:

```
station   23.3 m  lap 1/2  2.93/5.20 m/s  cmd +0.04  clear +0.268 m  cte +0.060 m
lap 2 of 2  49.70 s, +1.75 s on the last
FINISHED 2 laps in 101.15 s (220.4 m) -- coasting to a stop
STOPPED after 3.65 s and 5.1 m past the line
```

Watch three things:

- **`station` increasing.** Frozen station with a non-zero speed means the car
  is wedged or the pose stream has died — not that it is driving slowly.
- **`clear`** — body clearance to the nearest bale, *computed from the map*.
  Below about 0.12 m it is in the graze band. If `clear` looks healthy while
  the car is visibly close to a bale, **the pose has drifted** — stop the run.
- **`cte`** — cross-track error. Steadily growing is drift; oscillating is the
  controller.

After the last lap the throttle goes to zero and **the steering stays live**
while the car coasts to rest. That is deliberate: with no brakes it coasts
~15 m, all of it still inside the corridor, and freezing the wheels at the
line drives it into whatever it was turning away from.

---

## 8. Stopping

- **E-stop.** Always available, always the right answer if unsure.
- Ctrl-C in terminal 2. The node commands neutral with `auto_ready=false` on
  the way out.
- Without killing the node:
  ```bash
  ros2 service call /formula_one/manual_start std_srvs/srv/SetBool "{data: false}"
  ```

---

## 9. Troubleshooting

| Symptom | Cause |
|---|---|
| Car sits at the line, log says "Waiting for the start signal" | No start signal. Call `manual_start` (§6). |
| Node dies on first tick with `AttributeError` | Stale `.py` files on the Orin. Re-run step 5 — a partial copy leaves the node importing a mix of versions. |
| Node refuses to load the policy, complains about width | Observation width mismatch. Ship the run's own `config.yaml` (§2). |
| Car moves but wanders / drives at a bale | Localisation. Check `/zed/zed_node/pose` is being published and that the ZED has a map. This is the failure mode this design is most exposed to. |
| Car twitches or fights itself | A second publisher on `/drive_cmd`. Check `use_cmd_vel:=false`, and `ros2 topic info /drive_cmd --verbose` should show exactly one publisher. |
| "No such file" for the world or centerline | `repo_root` is wrong. It defaults to two directories above `formula_one_node.py`; pass `repo_root:=$HOME/cfr` explicitly if your layout differs from §3. |
| Everything starts but nothing moves and no telemetry appears | The node returns early before logging when it is finished, manually stopped, or the pose is stale. Check the pose topic first. |

Useful checks on the Orin:

```bash
ros2 topic hz /zed/zed_node/pose          # localisation alive?
ros2 topic hz /arduino_bridge/status      # speed feedback alive?
ros2 topic info /drive_cmd --verbose      # exactly ONE publisher
```

---

## 10. Falling back

The scripted driver needs no checkpoint and is a useful sanity check that the
plumbing, the map and the localisation are all good before you blame a policy:

```bash
ros2 launch ~/cfr/rl/formulaOne/formula_one.launch.py \
     driver:=baseline use_sim_time:=false rviz:=false speed_scale:=0.3
```

To go back to a previous policy, point `policy:=` at its `.npz` — nothing else
on the Orin has to change, as long as the observation width matches (§2).
