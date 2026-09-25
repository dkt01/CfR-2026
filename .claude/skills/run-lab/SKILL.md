---
name: run-lab
description: Record, pull, process and read CfR-2026 drive logs with the logging tools -- jetson/scripts/record_run.py on the Orin (MCAP bag + metadata + params + ZED area memory under ~/cfr_runs), the Run Lab (web/run-lab: pull from the Orin, analyze.py verdicts/laps/sections/events, Rerun .rrd export and embedded Rerun replay) and /formula_one/telemetry (DriverTelemetry). Use this whenever the user asks to record a run or drive, log a test, pull/sync/download runs off the car, process or analyze a bag, asks "how did the run go", "why did it crash/stall/not finish", wants lap times, clearance, CTE, verdicts or events from a real or simulated drive, wants to open or replay a run in Rerun, or start the Run Lab -- even if they just say "the last run" or "the logs". Prefer it over hand-parsing rosbags, ros2 bag info, or dumping summary.json.
---

# Run Lab: record, pull, process, read

Four stages. Know which one the user is at before acting:

| Stage | Where | Tool |
|---|---|---|
| Record | Orin | `formula_one.launch.py` (auto) or `jetson/scripts/record_run.py` |
| Pull | laptop | Run Lab **Car** page, or `jetson/scripts/sync_runs.sh` |
| Process | laptop | `web/run-lab/server/analyze.py <run>` (or the UI's Process button) |
| Read | laptop | `scripts/summarize_run.py`, Run Lab pages, Rerun |

Full reference: `web/run-lab/README.md`. Read it when a question goes past
what is here (page-by-page contents, how the track-frame fit works, Rerun
entity paths).

## 1. Record (on the Orin)

`rl/formulaOne/formula_one.launch.py` records by itself: `record:=auto`
(default) starts `record_run.py` whenever `use_sim_time:=false`. Extras:
`record_label:=<name>`, `record_args:="--svo --map"`. `record:=true` records
in simulation too; `record:=false` never records.

Anything else (wall follower, manual RC drive, a sim run you want analyzed):
run the recorder alone in its own terminal, stop with Ctrl-C.

```bash
~/cfr/jetson/scripts/record_run.py --label rc_test --notes "new tires"
~/cfr/jetson/scripts/record_run.py --label sim_check --cloud-any-rate   # simulation
```

Useful flags: `--driver`, `--speed-scale`, `--surface`, `--notes` (all land
in `metadata.yaml`, and later filter/label runs); `--cloud-hz N` (default 1,
0 = no cloud); `--no-images`; `--map`, `--svo` (heavier ZED capture);
`--compress`; `--min-free-gb` (default 3 -- below it the recorder refuses to
start); `--extra-topic /foo` (repeatable).

Things that bite:

- **Starting the driver drives the car.** Launching `formula_one.launch.py`
  with `use_sim_time:=false` or `launch.sh` arms real actuators. Hand the
  user the command; never run it yourself -- same rule as the
  `deploy-to-car` skill. Running `record_run.py` alone only records and is
  fine to start over ssh if the user asks.
- **Point cloud silently absent** is by design: if the ZED refuses the
  `depth.point_cloud_freq` change the cloud is left out rather than filling
  the disk at ~55 MB/s. `recorder.log` says so. In sim pass `--cloud-any-rate`.
- **Stop with Ctrl-C / SIGTERM, not kill -9.** A clean stop closes the bag,
  restores the cloud rate, saves area memory, copies ROS logs and finalizes
  `metadata.yaml` (up to ~45 s). SIGKILL leaves a readable MCAP but no
  finished metadata and no restored ZED rate.
- A run still in progress has a `RECORDING` file and a live `status.json`
  (lap, station, speed, battery, min clearance, bag bytes, free disk) --
  that is the cheap way to check on a recording over ssh.

Run layout: `~/cfr_runs/<UTC>_<label>/` with `bag/` (MCAP),
`metadata.yaml`, `policy/` (exact `policy.npz` + `config.yaml`), `params/`,
`zed/`, `logs/` (ROS logs, `tegrastats.log`), `recorder.log`, `bag.log`.

## 2. Pull to the laptop

Runs land in `<repo>/runs/` (gitignored; `CFR_RUNS_LOCAL` overrides).

- Run Lab **Car** page: Pull rsyncs, verifies byte-for-byte, then queues
  processing. Delete-from-Orin only appears after a verified pull.
- No UI: `jetson/scripts/sync_runs.sh --list`, then `sync_runs.sh` (add
  `--host tejam@192.168.0.167` over Wi-Fi; USB-C is `tejam@192.168.55.1`,
  the script's default is `orin.local`).

Never pass `--purge` or delete runs from the Orin unless the user asked for
that specific deletion -- a run is a trip to the track and cannot be
re-recorded. SSH must be key-based for the Run Lab to pull.

`runs/` also holds older **characterization** runs (`*_skidpad`,
`*_step_steer`, ... with `telemetry.csv`, `report.md`). Those come from
`characterize.launch.py`, not `record_run.py`; `jetson/scripts/analyze_run.py`
and `jetson/scripts/characterize/` handle them. `record_run.py` runs have
`kind: drive` in `metadata.yaml`; characterization runs have no `kind`.
`analyze.py` will still process a characterization bag, but with no ZED pose
or telemetry it can only report mode/E-stop/stall events -- use the
characterization tools for those.

## 3. Process

Headless, which is what you want from the terminal:

```bash
web/run-lab/.venv/bin/python web/run-lab/server/analyze.py runs/<run>
web/run-lab/.venv/bin/python web/run-lab/server/analyze.py runs/<run> --no-clouds --no-images   # fast
```

It reads the bag with `rosbags` (no ROS needed) and writes
`runs/<run>/analysis/`: `summary.json`, `series.json` (every channel on one
20 Hz timeline), `course.json`, `logs.json` (`/rosout`), `recording.rrd`.
About 20 s for a two-lap run with clouds; `--no-clouds --no-images` when you
only need numbers. It writes into `analysis.tmp/` and swaps at the end, so a
crash never leaves a half-written `analysis/`.

The venv comes from `web/run-lab/run.sh` on first start (it also builds the
UI, needing `npm` once). If `.venv` is missing and you only want to process,
create it yourself: `python -m venv web/run-lab/.venv` then install
`web/run-lab/requirements.txt` into it.

On Windows the venv's Python is `web/run-lab/.venv/Scripts/python`, not
`.venv/bin/python`; `analyze.py` runs fine natively there (verified with
Windows Python 3.14). `run.sh` itself hard-codes `.venv/bin`, so the UI
server wants Linux/WSL -- and WSL's system Python here is 3.8, too old for
the pinned `rerun-sdk`. For headless processing on Windows, use the Windows
venv. `run.sh` unsets `PYTHONPATH` so ROS's site-packages cannot shadow the
venv; do the same if you call the venv Python from a ROS-sourced shell.

Reprocess when `summary.json`'s `meta.version` is older than
`analyze.VERSION` (the UI offers this) or after changing `analyze.py`.

## 4. Read

Start with the bundled summarizer, not `cat summary.json` -- the JSON is
large and mostly nested stats you don't need for the first answer:

```bash
python .claude/skills/run-lab/scripts/summarize_run.py runs/<run>
python .claude/skills/run-lab/scripts/summarize_run.py runs/<run> --sections --logs WARN
python .claude/skills/run-lab/scripts/summarize_run.py runs/<a> runs/<b>     # compare
```

Standard library only; any Python works. It prints metadata, KPIs, verdicts
(worked / failed, each with its timeline second), laps, events (contacts,
grazes, stalls, pose jumps, link drops, errors, with station), and optionally
sections and `/rosout`.

Then drill in by domain -- keys in `summary.json`: `speed`, `steering`
(incl. command-to-yaw lag vs the plant), `policy` (clearance, CTE,
saturation, prior vs residual), `localization` (pose rate, gaps, jumps, odom
divergence), `system` (battery, link, E-stops, tegrastats), `topic_health`,
`frame`. For a time slice, index `series.json`'s `columns` (all same length,
`hz` = 20, `columns.t` the timeline) around an event's `t`.

How to reason about a result:

- **Check localization before trusting clearance.** Clearance and CTE are
  computed off the map and the pose. A pose jump or a bad track-frame fit
  (`frame.fit_rms_m` large, `anchors` > 1 meaning the driver re-anchored)
  makes every downstream number wrong. Look at `localization` and pose-jump
  events first when a run "hit a bale" that the numbers say it cleared, or
  vice versa.
- **Telemetry is what the driver believed.** `/formula_one/telemetry`
  (`jetson/cfr_interfaces/msg/DriverTelemetry.msg`) is published every tick,
  so a gap in it means the node died or stalled, not a stale pose (that is
  `state == 4`). `speed_from_tach == false` means the tachometer went stale.
- **Sim vs car.** A bag with `/clock` is put on sim time (`meta.simulation`),
  so lap times stay correct when Gazebo runs slower than real time.
- Report findings with the timeline second and station, so the user can jump
  there in the Run Lab or Rerun.

## Viewing and replay (hand to the user)

- Run Lab UI: `web/run-lab/run.sh` then http://localhost:8765 (`--lan` binds
  0.0.0.0 with no auth -- trusted networks only). It is a long-running
  server; start it in the background if the user wants it, and say so.
- Rerun: `web/run-lab/.venv/bin/rerun runs/<run>/analysis/recording.rrd`
  (track frame, with analysis) or `rerun runs/<run>/bag/*.mcap` (raw, map
  frame). `rerun-sdk` and `@rerun-io/web-viewer` are pinned to the same
  version; upgrade them together.
- The Replay page embeds the same recording (full-window, fullscreen from
  the viewer's top bar); its *Follow car* toggle opens the 3D view that
  tracks the car. There is no Gazebo/RViz replay any more: for a live ROS
  graph, `ros2 bag play runs/<run>/bag`.

## Changing the tools

- New recorded topic: add it to `RECORD_TOPICS` in `record_run.py`
  (anchored regex fragments, one family per line).
- New driver field: `DriverTelemetry.msg`, then rebuild `cfr_interfaces`
  (`run-tests` skill). `bagio.py` registers the repo's `.msg` files, so
  older bags still decode.
- Analysis output changes shape: bump `analyze.VERSION`, and run
  `web/run-lab/.venv/bin/python web/run-lab/server/selftest.py`.
