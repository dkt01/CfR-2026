---
name: rl-train
description: Start, watch and end a reinforcement-learning training run for the CfR-2026 bale-following policy (rl/bale_follower, PPO via stable-baselines3) on the speed or obstacle course, inside the Docker sim container, and judge from the deterministic-eval trend whether the run is still improving or has plateaued and should be stopped early. Use this whenever the user asks to train, retrain, resume, launch a PPO/RL run, check on training, ask "is it still learning", "how's the run going", "should we stop it", wants a checkpoint evaluated, or asks about training on the obstacle course -- and prefer it over reconstructing launch_training.sh / train_resilient.sh / train_curriculum.sh invocations or tailing logs by hand.
---

# CfR-2026 RL training runs

Wraps the whole life of a training run -- container, venv, workspace build,
launch, progress judgement, teardown -- into `scripts/rl.sh`, so a run that
costs five to nine hours of wall clock is started correctly the first time and
is not left burning hours after it has stopped learning.

It drives the repo's own wrappers (`train_resilient.sh`, `test_policy.sh`)
rather than reimplementing them. Their chunked auto-resume exists because
Gazebo's pose stream dies after roughly five hours of continuous training; that
logic is worth more than a tidier command line.

```bash
.agents/skills/rl-train/scripts/rl.sh setup                      # once per machine
.agents/skills/rl-train/scripts/rl.sh start --steps 200000 --dir checkpoints_v10
.agents/skills/rl-train/scripts/rl.sh status --dir checkpoints_v10
.agents/skills/rl-train/scripts/rl.sh logs --lines 60
.agents/skills/rl-train/scripts/rl.sh eval --checkpoint checkpoints_v10/best_model.zip
.agents/skills/rl-train/scripts/rl.sh stop
```

## First run on a machine

`setup` creates the `cfr-rl` container from the same image the simulator uses,
installs `python3-venv`/`pip` (the image ships neither), builds the venv with
`--system-site-packages` so `rclpy` resolves, installs the CPU torch wheel plus
`requirements.txt`, and builds the ROS workspace into `/repo` with
`ROS2_WS=/repo`. Expect ten minutes or so, mostly pip.

Two choices in there are load-bearing:

- **The venv is a named Docker volume** mounted over
  `rl/bale_follower/.venv`, not a directory in the Windows bind mount. pip
  unpacks tens of thousands of files and doing that through the mount is slow
  both on install and on every later import. The volume also survives
  `docker rm`.
- **The workspace builds into `/repo`, not `~/ros2_ws`.** `_sim_stack.sh` and
  `train_resilient.sh` both source `$REPO_ROOT/install/setup.bash` and refuse
  to run without it, so building where they look keeps them usable unmodified.
  `install/`, `build/` and `log/` are already gitignored.

This container is deliberately separate from `cfr-sim` (the `sim-launch`
skill's viewer container) and runs on its own `ROS_DOMAIN_ID` and
`GZ_PARTITION`. Two Gazebo servers that can see each other publish onto one
pose topic and silently corrupt a run's measurements.

## Starting a run

```bash
rl.sh start --steps 200000 --dir checkpoints_v10           # speed course
rl.sh start --resume-from checkpoints_v9/best_model.zip --steps 100000 --dir checkpoints_v10
rl.sh start --curriculum                                   # the three-stage speed ladder
```

`--sensors` is decided from `config.yaml`: `scan_source: cloud` needs the
rendered ZED and its bridged point cloud, and starting without it leaves the
env waiting for a cloud nothing publishes. Override with `--sensors` /
`--no-sensors` only when deliberately A/B-ing the observation source.

`--curriculum` runs `train_curriculum.sh`, which **rewrites `env.max_speed` in
`config.yaml` in place** at each stage and names its own checkpoint
directories. Say so before starting it; the file is left at the last stage's
value.

**Tell the user how long it will take, up front.** The env paces itself against
the wall clock at `control_hz` steps per second (10 Hz today), so:

| steps | sensors off | sensors on (RTF ~0.63) |
|---|---|---|
| 50k | ~1.4 h | ~2.2 h |
| 200k | ~5.5 h | ~8.8 h |
| 250k | ~7 h | ~11 h |

plus the deterministic eval episodes, which run every 16 rollouts (~8k steps)
and cost up to three minutes each. `start` prints this estimate itself.

## Watching it, and knowing when it is done

`rl.sh status` prints run liveness, the Gazebo server count, and then
`progress.py`, which reads the deterministic-eval lines out of the chunk logs,
stitches them across resumes (SB3 restarts its own step counter on every
resume, so raw per-chunk numbers are not comparable), and returns one of four
verdicts: `too-early`, `improving`, `plateau`, `regressing`.

Check roughly every 30-45 minutes — evals arrive about every 15 minutes, so
anything tighter reports the same numbers twice. Each time, report: latest eval
distance, best so far and how long ago, the slope in metres per 10k steps, and
any restart lines. Keep it to a few lines; the user is waiting out hours, not
reading a report.

**Only the deterministic eval counts as progress.** `ep_rew_mean` is the
stochastic sampled policy; deployment runs the Gaussian mean, and the two come
apart — the v3 run had a healthy training curve while the mean action floored
the throttle and crashed within seconds. `progress.py` prints `ep_rew_mean` as
context and never judges on it.

On `plateau`, bring it to the user with the evidence and a recommendation
rather than either killing the run or letting it grind on:

- **Stopping is cheap and safe.** `best_model.zip` is written by the eval
  callback the moment a new best appears, and the step checkpoints are on disk.
  Nothing is lost but the remaining steps.
- **Stopping is often not the best move.** A plateau at a `max_speed` ceiling
  usually means the ceiling is the binding constraint, not the policy — the
  next curriculum stage (resume from this best with a higher `max_speed`) is
  the productive step. A plateau well below the ceiling, with collisions still
  in the eval episodes, is a reward-shaping problem: hand it to the
  `rl-reward` skill rather than spending more steps.
- **`regressing` is different.** Catastrophic forgetting or a sim that has gone
  bad. Check the restart lines and the gz server count before concluding
  anything about the policy.

Ask before stopping. It is the user's run and their hours.

## Evaluating the result

```bash
rl.sh eval --checkpoint checkpoints_v10/best_model.zip --episodes 5
```

This runs `test_policy.sh`, which brings up its own sim and tears it down.
It refuses while training is running: that Gazebo server owns the topics and a
second one corrupts both. Evaluation uses the reward and env settings recorded
in the checkpoint's `.json` metadata, not the current `config.yaml`, so a
checkpoint stays comparable to what it was trained under.

Five episodes is a small sample with randomized starts. Quote the range and the
clean-episode distance, not the best single number — REPORT.md's v6 entry is
the precedent (112-120 m mean, ~130 m clean).

## The obstacle course

`start --course obstacle` runs `course_preflight.py` first and refuses if the
stack cannot actually train there. As of this writing five checks fail, which
is the honest state of the code rather than a limitation of this skill:

1. `bale_geometry.parse_bales` looks up `course_bales/link[@name='bales']`;
   the obstacle world names that link `link`, so it raises `ValueError`.
2. `training.launch.py` hard-codes the pose bridge to
   `/world/cfr_speed_course/dynamic_pose/info`. On another world the ROS topic
   exists but never publishes, and the env dies on its odom timeout.
3. `config.yaml`'s `env.world_name` drives `_unpause_world`; the wrong name
   leaves the server paused, `/clock` frozen and the car at neutral — which
   looks exactly like a dead policy.
4. `env._on_pose` reads `msg.transforms[0]` and assumes that is the car. In
   `obstacle_course.sdf`, `start_signal_arms` is declared before `slash`, so
   index 0 is the start signal.
5. `training.launch.py` starts `teleport_api.py` without `CFR_SIM_WORLD`, and
   that script defaults to `cfr_speed_course`, so every episode reset would
   address a world that is not loaded and silently do nothing.
   `simulation.launch.py` already passes it through `additional_env`.
6. `_sim_stack.sh` and `train_resilient.sh` never pass `world:=`, so they load
   the speed course whatever was asked for.

Plus two warnings the preflight cannot gate on. The buckets, hoops, tunnel,
bridge, ramps and gravel are STL meshes that the analytic scan and the
separating-axis collision check never see, so a policy trained there today
would learn to drive through them. And `training.launch.py` never starts
`obstacle_randomizer_node` — which keeps static geometry parsing valid, at the
cost of a policy that can memorize one layout. The obstacle course is also a
gated, three-dimensional task — "forward progress in a corridor" is not the
objective, so it needs reward work (the `rl-reward` skill) as well as geometry.

`--force` starts anyway and fails at the first of those; useful only for
confirming a fix. The checks read the repo live, so when support lands they
pass on their own and the gate opens with no edit here.

## Failure modes worth recognizing

- **Chunk died, run resumed.** Normal after ~5 h; `progress.py` lists it under
  `restart:`. Several restarts inside an hour is not normal — check the gz
  server count and `resilient_sim.log`.
- **More than one gz server.** Duplicate servers publish onto the same pose
  topic. Stop everything (`rl.sh stop`) and restart; do not trust measurements
  taken while it was true.
- **No eval lines after 30+ minutes.** The callback needs ~8k steps. If the
  newest checkpoint is also stale, the run is wedged rather than slow.
- **Distances stuck near zero from the start.** Usually the car never moved:
  wrong world name (paused world), missing point cloud with `scan_source:
  cloud`, or `path_follower` publishing idle zeros onto `/cmd_vel` — the last
  one cannot happen under `training.launch.py`, which is why training uses it
  instead of `simulation.launch.py`.

## Notes

- Nothing here shows the car in the browser. The viewer stack is the
  `sim-launch` skill's container, and training runs headless in its own. To
  watch a finished policy drive, evaluate it there instead.
- `rl_run.log`, `train_resilient.log`, `train_chunk_*.log` and the checkpoint
  directories all live under `rl/bale_follower/` on the host (gitignored), so
  `progress.py` can be run directly against them without Docker.
- Training has no GPU here. It is not the bottleneck: the wall-clock pacing
  caps throughput at `control_hz` steps per second whatever the network runs
  on. `rl.sh setup --gpu` skips the CPU torch wheel if that ever changes.
