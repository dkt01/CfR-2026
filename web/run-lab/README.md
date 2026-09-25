# Run Lab

Record a drive on the Orin, pull it to a laptop over USB-C, and find out what
worked and what didn't: the verdict, where on the course it happened, and
whether the problem was the policy, the car, or the localisation.

The work is split between two tools. **The Run Lab does the analysis** Rerun
can't do: pulling from the Orin, verdicts, laps and sections, the car against
the plant model and the policy's behaviour. **[Rerun](https://rerun.io)
does the viewing**: the course and car in 3D, ZED clouds and the map they
build, the camera, every channel against time, and the logs. Rerun is embedded
in the Replay page and shares the Run Lab's timeline in both directions.

```bash
web/run-lab/run.sh            # → http://localhost:8765
```

The first start sets up its own `.venv` and builds the UI (needs `npm` once).
After that it starts in about a second. Reading bags needs no ROS, because it uses
[`rosbags`](https://pypi.org/project/rosbags/).

---

## 1. Recording on the car

`formula_one.launch.py` records by itself. With `record:=auto`, the default, it
starts [`jetson/scripts/record_run.py`](../../jetson/scripts/record_run.py)
whenever `use_sim_time:=false`, which means on the car. A real run can't be
driven unrecorded by accident.

```bash
ros2 launch ~/cfr/rl/formulaOne/formula_one.launch.py \
     policy:=... config:=... use_sim_time:=false rviz:=false speed_scale:=0.3 \
     record_label:=v12_first_car_run            # optional name
     record_args:="--svo --map"                  # optional heavier ZED capture
```

To record anything else (the wall follower, a manual RC drive), run the
recorder on its own in a third terminal and stop it with Ctrl-C:

```bash
~/cfr/jetson/scripts/record_run.py --label rc_test --notes "new tires"
```

Each run becomes one folder in `~/cfr_runs/<UTC>_<label>/`:

| | |
|---|---|
| `bag/` | MCAP rosbag. See the topic list below |
| `metadata.yaml` | label, git SHA, driver, speed scale, SHA-256 of the policy and config, recorder options, a result summary |
| `policy/` | the exact `policy.npz` and `config.yaml` that drove |
| `params/` | `ros2 param dump` of the ZED, bridge, driver, lap counter, start detector |
| `zed/` | the ZED area memory (`.area`), plus `run.svo2` with `--svo` |
| `logs/` | every ROS log file written during the run, and `tegrastats.log` (CPU, GPU, temperatures) |
| `status.json` | live progress while recording. The Car page shows it |
| `recorder.log` | what the recorder did, including anything it could not do |

**Recorded topics.** `/rosout` (every log line of every node), `/tf*`,
`/drive_cmd`, `/arduino_bridge/status`, `/formula_one/telemetry` (every
control tick: station, CTE, map clearance, cap, prior, raw action and the full
observation), `/lap_counter/*`, `/start_signal_detector/state|go`, the ZED pose,
pose status, odom, IMU, path, health, the rectified camera (compressed), the
point cloud, and `mapping/fused_cloud` with `--map`.

**The point cloud is rate-limited.** The ZED 2i's registered cloud is about 3.7 MB
per frame, around 55 MB/s at full rate. The recorder lowers the ZED's own
`depth.point_cloud_freq` to `--cloud-hz` (default 1 Hz) for the run and
restores it afterwards. **If the ZED won't take the parameter, the cloud is left
out of the bag** rather than filling the Orin's disk mid-drive. `recorder.log`
says which happened. `--cloud-any-rate` overrides this (for simulation).
`--cloud-hz 0` records no cloud.

**Other options.** `--no-images`, `--map` (ZED spatial mapping on; costs Orin
GPU), `--svo` (full SVO2 for offline ZED processing), `--no-area`, `--compress`
(zstd chunks; costs CPU), `--min-free-gb` (default 3: below this it refuses to
start), `--extra-topic /foo`.

**Stopping.** Ctrl-C on the launch stops the recorder too. It closes the bag,
restores the cloud rate, saves area memory and copies the logs, allowing up to
45 s before the launch escalates. Each step is independent, so one failing
does not cost the others. A run killed hard still leaves a readable bag,
because MCAP is written incrementally.

`~/cfr/jetson/scripts/sync_runs.sh` still works as a no-UI alternative for
pulling runs.

## 2. Pulling runs off the car

Plug the Orin into the laptop's USB-C port. The Orin appears at `192.168.55.1`.
Open the **Car** page:

- connection state, the Orin's free disk, and live progress of any run still
  recording
- every run on the car, with **Pull**. Pulling rsyncs the run, **verifies the
  copy byte-for-byte against the Orin**, then queues it for processing
- **Delete from the Orin** appears only after a verified pull, and asks for
  confirmation first. Nothing is ever deleted from the car otherwise

SSH must be key-based (`ssh-copy-id tejam@192.168.55.1` once), because a
password prompt can't be answered from a web page. Over Wi-Fi, change the host
on the page or start with `ORIN_HOST=tejam@orin.local`.

Runs land in `<repo>/runs/`, which is already in `.gitignore` (override with
`CFR_RUNS_LOCAL`). **Link folder** on the Runs page adds a run copied some
other way.

## 3. Reading a run

Processing reads the bag once, in about 20 s for a two-lap run with clouds. The
timeline at the bottom is shared by every page: click an event, a verdict, a
chart or the map, and everything moves there. Space plays and pauses, the arrow
keys step, and shift-arrow steps 5 s.

| Page | Answers |
|---|---|
| **Overview** | Did it finish? What worked, what didn't (each item jumps to its moment), laps, and the event list: contacts, grazes, stalls, pose jumps, link drops, errors |
| **Track & sections** | The driven path on the surveyed course, coloured by speed, speed vs cap, clearance, CTE, steering or lap. Section table per lap, worst clearance flagged |
| **Replay (Rerun)** | The run in the embedded Rerun viewer. It fills the window (or goes fullscreen from its top bar) and holds the course with the car (a fixed Course view, and a Follow car view that tracks it; the *Follow car* toggle opens on that one), the ZED cloud at each moment plus the map they accumulate (and the ZED's own spatial map if recorded), the full camera frame, every channel in grouped plots, and the events and `/rosout` logs. It shares the Run Lab's timeline both ways. Also: open the same recording, or the raw bag, in the native Rerun app |
| **Vehicle model** | The car against `plant.py`: speed envelope, speed loop (command → target → measured), coast-down, steering authority left/right, and the **command-to-yaw lag** the policy was trained with |
| **RL policy** | Speed, clearance, CTE and steering per lap against station; section × lap clearance matrix; action saturation; prior vs residual |
| **ZED & localisation** | Pose health: rate, gaps, jumps, odom divergence, tracking status, pose age at the driver, and the driver's position belief against the analysis |
| **System health** | Battery, Arduino link, E-stops, modes, Orin CPU/GPU/temperature, and every topic's rate against what it should be |
| **Files** | Everything in the run, downloadable; the whole run as one `.tar`; the commands to open it in Rerun or `ros2 bag play` |

**Report** in the header downloads a Markdown summary for sharing.

### How the numbers are made

- **Track frame.** The driver latches the ZED map pose at the start signal and
  pins it to the surveyed start. The analyser fits the map→track transform to
  the driver's own `~/telemetry`, so clearances are the ones the driver
  believed. It fits one transform per anchor if the driver re-anchored mid-run
  (a second `manual_start`). Without telemetry it latches the way the driver
  does. In simulation it uses Gazebo ground truth.
- **Clearance** is body-to-bale off the map and the pose, using the same distance
  field the policy trained on (`rl/formulaOne/track.py`). If the pose is wrong,
  the clearance is wrong. On the car, check **ZED & localisation** first.
- **Steering → yaw lag** fits `measured yaw rate ≈ lag(gain × plant kinematic
  yaw rate)` over every moving sample. It is compared with the plant's dead time,
  servo lag and `yaw_response_tau`. `server/selftest.py` holds it to cases with
  a known answer. On a run through `ros_loopback` (which *is* the plant) it
  recovers 0.53 s against the plant's 0.58 s.
- **Sim runs** (a `/clock` in the bag) are put on sim time, so lap times are
  correct even when Gazebo runs slower than real time.

## 4. Rerun

Processing writes `analysis/recording.rrd` (see `server/rerun_export.py`),
already in the **track frame** and on the Run Lab's timeline, so the same
number means the same moment in both tools:

| Entity | |
|---|---|
| `world/course/*` | bales, the centerline tinted by cap zone, the start line |
| `world/car`, `world/path`, `world/events` | the car over time, its path coloured by speed, contacts, grazes, stalls and pose jumps |
| `world/cloud/frame`, `map`, `zed_map` | the ZED cloud at each moment, the voxel map accumulated from it, the ZED's own spatial map |
| `camera` | the rectified camera, 5 Hz |
| `metrics/<group>/*` | speed, steering, yaw rate against the plant, clearance, line, acceleration, actions, progress, localisation, battery |
| `events`, `rosout` | the Run Lab's events and verdicts, and every ROS log line |

The saved layout opens paused at the playhead. Rearrange it freely; Rerun
remembers your layout per recording. For big runs the native viewer is faster
than the embedded one (**Open in Rerun app**, or
`web/run-lab/.venv/bin/rerun runs/<run>/analysis/recording.rrd`).
`rerun runs/<run>/bag/*.mcap` opens the raw bag as recorded, in the map frame
with no analysis.

`rerun-sdk` (Python) and `@rerun-io/web-viewer` (npm) are pinned to the same
version, because the embedded viewer reads what the SDK writes. Upgrade them
together. Rerun collects anonymous usage statistics by default;
`web/run-lab/.venv/bin/rerun analytics disable` turns that off on this machine.

## Development

```bash
cd web/run-lab/frontend && npm run dev          # UI on :5180, proxies /api to :8765
web/run-lab/.venv/bin/python web/run-lab/server/selftest.py
web/run-lab/.venv/bin/python web/run-lab/server/analyze.py runs/<run>   # headless
```

`server/` is FastAPI (`app.py`), the bag reader (`bagio.py`), the analysis
(`analyze.py`), the Rerun recording (`rerun_export.py`), course geometry via
the driver's own `track.py` (`course.py`), and Orin transfer
(`orin.py`). `frontend/` is Vite with plain JS, uPlot
for the analysis charts and the Rerun web viewer. Bump
`analyze.VERSION` when outputs change shape, and the UI will offer to
re-process older runs.
