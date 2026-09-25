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

`syncSoftware.sh` (§4) produces this, and the node resolves everything in it
with no extra arguments:

```
~/software/                <- jetson/: the ROS packages and scripts
  cfr_arduino_bridge/
    worlds/speed_course.sdf          <- the bale positions
    config/speed_course_path.json    <- the centerline
  scripts/launch.sh, record_run.py, ...
  formulaOne/              <- rl/formulaOne/: the driver
    policy.npz             <- the chosen policy (v12 by default)
    config.yaml            <- THAT policy's config, not the tree's
    .cache/                <- the track cache
~/jetson -> ~/software     <- symlink, made by the sync
```

The symlink is there because the node finds the course files, and the launch
file finds `record_run.py`, as `<two directories above itself>/jetson/...`,
which is the repo's own layout. For `~/software/formulaOne` that is
`~/jetson`. The sync never replaces a real `~/jetson` directory; it warns
instead.

`~/ros2_ws` stays the colcon workspace, as it is today.

---

## 4. Sync, copy the policy and build

From the dev machine, in the repo root:

```bash
jetson/scripts/syncSoftware.sh --dry-run    # look first
jetson/scripts/syncSoftware.sh --build
```

One command does all of it:

- syncs `jetson/` to `~/software` and builds `cfr_interfaces` (which defines
  `DriveCommand` and `ArduinoStatus`) and `cfr_arduino_bridge`. **The policy
  node needs `cfr_interfaces` to exist, so the build is not optional even
  though the node itself is plain Python.**
- syncs `rl/formulaOne/` to `~/software/formulaOne`, leaving out `runs/`,
  `bestModel/`, `.venv` and the tree's own `config.yaml`
- copies the chosen policy's `policy.npz` and `config.yaml` to the top of
  `~/software/formulaOne`. The default is **v12**; choose another with
  `--policy <run>` (or `F1_RUN=<run>`). It is taken from
  `rl/formulaOne/bestModel/<run>/`, which is committed, or else from
  `runs/<run>/`. A run that has neither file stops the sync before anything is
  copied.
- copies the track cache. `track.py` builds a distance field over every bale
  on first use; the cache saves the Orin that work. Its key is a hash of the
  world SDF and the track settings, so a stale cache is ignored rather than
  used.
- makes the `~/jetson` symlink (§3)

The policy's own `config.yaml` replaces the tree's on purpose: a policy must
drive with the config it was trained under (§2). `--no-f1` syncs `jetson/`
alone.

Defaults are `ORIN_HOST=tejam@192.168.55.1` and `ORIN_WS=~/ros2_ws`; override
with `--host` / `-w` or the `ORIN_HOST` / `ORIN_WS` environment variables.
Source only — the host is x86_64 and the Orin is aarch64, so no build
artifacts are transferred.

---

## 5. Run it

Two terminals on the Orin. **E-stop in hand.**

**Terminal 1 — the Arduino bridge and the ZED:**

```bash
~/software/scripts/launch.sh --no-cmd-vel
```

This brings up the bridge and the ZED together, and starts the ZED with the
race configuration, `jetson/cfr_arduino_bridge/config/cfr_zed2i.yaml`. That
file pins the grab rate, tracking mode, loop closure and 2D mode the pose
depends on. **Do not start the ZED with a bare `ros2 launch zed_wrapper ...`**:
without the file the wrapper runs on whatever defaults its revision ships, and
the policy drives on that pose. Check it took before the first run:

```bash
ros2 param get /zed/zed_node pos_tracking.pos_tracking_mode   # GEN_3
ros2 topic hz /zed/zed_node/pose                              # ~60 Hz
```

`--no-cmd-vel` (`use_cmd_vel:=false` on the bridge launch) **matters.** `cmd_vel_to_drive_node` republishes
`DriveCommand` on a timer whether or not anything is feeding it, so leaving it
up puts a second publisher on `/drive_cmd` and the Arduino acts on whichever
message arrived last. The car then drives on a mixture of the policy and a
stream of neutral commands, with no obvious point of failure. It just drives
badly.

**Terminal 2 — the policy, first run, at a third speed:**

```bash
source ~/software/scripts/setEnv.sh
ros2 launch ~/software/formulaOne/formula_one.launch.py \
     policy:=$HOME/software/formulaOne/policy.npz \
     config:=$HOME/software/formulaOne/config.yaml \
     record_label:=v12 use_sim_time:=false rviz:=false speed_scale:=0.3
```

- `use_sim_time:=false` — **required on the car.** Left at its `true` default
  the node waits on a `/clock` that never comes.
- `speed_scale` multiplies every speed command. Work up 0.3 → 0.5 → 1.0 over
  separate runs, checking the line each time.
- `laps:=N` overrides the two laps in the config.
- `record_label` names the recording after the policy. Without it the run is
  labelled after the policy's directory, which is now always `formulaOne`.
  Set it to whatever `--policy` you synced.
- **The run is recorded automatically** (`record:=auto` records whenever
  `use_sim_time:=false`) into `~/cfr_runs/<UTC>_f1_<label>/`: the bag, the exact
  policy and config, parameter dumps, ZED area memory, ROS logs and
  tegrastats. `record_label:=<name>` names it. Camera frames are
  recorded at 12 fps and JPEG quality 50. Instead of the live point cloud, the
  ZED builds a spatial map during the run and the **finished map** is saved in
  the bag at the end. `record_args:="--no-map"` skips the map,
  `--cloud-hz 1` adds the live cloud at 1 Hz, `--svo` adds a ZED SVO. See §10.

The car waits for the start signal and will not move until it gets one. Either
show it the real green signal, or release it by hand:

```bash
ros2 service call /formula_one/manual_start std_srvs/srv/SetBool "{data: true}"
```

---

## 6. What a good run looks like

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

## 7. Stopping

- **E-stop.** Always available, always the right answer if unsure.
- Ctrl-C in terminal 2. The node commands neutral with `auto_ready=false` on
  the way out.
- Without killing the node:
  ```bash
  ros2 service call /formula_one/manual_start std_srvs/srv/SetBool "{data: false}"
  ```

---

## 8. Troubleshooting

| Symptom | Cause |
|---|---|
| Car sits at the line, log says "Waiting for the start signal" | No start signal. Call `manual_start` (§5). |
| Node dies on first tick with `AttributeError` | Stale `.py` files on the Orin. Re-run `syncSoftware.sh` (§4) — a partial copy leaves the node importing a mix of versions. |
| Node refuses to load the policy, complains about width | Observation width mismatch. Ship the run's own `config.yaml` (§2). |
| Car moves but wanders / drives at a bale | Localisation. Check `/zed/zed_node/pose` is being published and that the ZED has a map. This is the failure mode this design is most exposed to. |
| Car twitches or fights itself | A second publisher on `/drive_cmd`. Check `use_cmd_vel:=false`, and `ros2 topic info /drive_cmd --verbose` should show exactly one publisher. |
| "No such file" for the world or centerline | The `~/jetson -> ~/software` symlink is missing, or `~/jetson` is a real directory. Re-run `syncSoftware.sh` and read its warning (§3). |
| `record_run: only N GB free` and no recording | The Orin disk is nearly full. Pull runs in the Run Lab, then delete them from the Orin there (only offered after a verified copy). |
| `WARNING: could not lower the ZED cloud rate` | The ZED node did not accept `depth.point_cloud_freq`, so the recorder left the cloud out rather than fill the disk. Everything else is recorded. |
| Everything starts but nothing moves and no telemetry appears | The node returns early before logging when it is finished, manually stopped, or the pose is stale. Check the pose topic first. |

Useful checks on the Orin:

```bash
ros2 topic hz /zed/zed_node/pose          # localisation alive?  ~60 Hz
ros2 topic echo /zed/zed_node/pose/status # tracking state; loop closures show here
ros2 topic hz /arduino_bridge/status      # speed feedback alive?
ros2 topic info /drive_cmd --verbose      # exactly ONE publisher
```

---

## 9. Falling back

The scripted driver needs no checkpoint and is a useful sanity check that the
plumbing, the map and the localisation are all good before you blame a policy:

```bash
ros2 launch ~/software/formulaOne/formula_one.launch.py \
     driver:=baseline use_sim_time:=false rviz:=false speed_scale:=0.3
```

To go back to a previous policy, re-run `jetson/scripts/syncSoftware.sh
--policy <run>`. It replaces `policy.npz` and `config.yaml` together, so the
policy and its config always match (§2).

---

## 10. After the run: pull it and read it

Plug the Orin into the laptop's USB-C port and start the Run Lab on the laptop:

```bash
web/run-lab/run.sh                # http://localhost:8765
```

**Car** lists every run on the Orin. **Pull** copies one, verifies it
byte-for-byte and processes it. The run's pages then answer, in order: did it
finish and what failed (Overview), where on the course (Track & sections,
Replay), and why: the car against the plant (Vehicle model), the policy's own
behaviour (RL policy), and the pose it was driving on (ZED & localisation).
**Check ZED & localisation before trusting any clearance**, because clearances
come from the pose, exactly as the driver's do. Replay opens the run in Rerun:
the course and car in 3D, the ZED clouds and map, the camera, and every
channel, on the same timeline.

The recorder needs `cfr_interfaces` built with `DriverTelemetry` (step 4 does
this). Without it the car still drives, but the node warns that `~/telemetry`
is off, and the analysis loses the driver's own view (actions, prior, anchor).
Details: [web/run-lab/README.md](../../web/run-lab/README.md).

