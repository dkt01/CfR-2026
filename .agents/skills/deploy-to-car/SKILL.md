---
name: deploy-to-car
description: Sync jetson/ source to the Orin, build it there, and get the exact command to launch the actuator link + ZED camera on the real car. Use this whenever the user asks to deploy, sync, push, or update software on the car/Orin/robot, or wants to run/launch/test something on the real hardware as opposed to the simulator. Chains the manual sync-then-build-then-launch sequence people otherwise reconstruct from jetson/scripts/syncSoftware.sh's flags each time. Does NOT itself run launch.sh on the Orin -- launching arms real actuators on a physical vehicle, so that step always needs the user physically at the bench with the E-stop remote in hand, confirmed immediately before running, never run unattended or in the background.
---

# Deploy to the Orin

Two safe, automatable steps (sync + remote build) plus one that stays manual
on purpose (launching real hardware).

## 1. Sync and build

```bash
.agents/skills/deploy-to-car/scripts/deploy.sh sync                # sync + colcon build on the Orin
.agents/skills/deploy-to-car/scripts/deploy.sh sync --test          # also colcon test there
.agents/skills/deploy-to-car/scripts/deploy.sh sync --dry-run       # show what would transfer, change nothing
.agents/skills/deploy-to-car/scripts/deploy.sh sync --host tejam@192.168.0.167  # e.g. over Wi-Fi instead of USB-Ethernet
```

`--host` without a `user@` prefix (e.g. `--host 192.168.0.167`) has the
user defaulted for you (from `ORIN_HOST`, normally `tejam`) rather than
failing -- a bare IP used to make the reachability check below fail while
blaming "is the Orin powered on", when the real problem was ssh trying to
authenticate as the local machine's own user.

This is a thin wrapper around `jetson/scripts/syncSoftware.sh --build`, which
already does the real work (rsync with the right excludes, then `ssh` +
`colcon build` on the Orin, cleaning `build/`/`install/` first because the
Jetson's clock can lag files synced from the dev host and otherwise Make
keeps a stale binary). The wrapper adds one thing: a 5-second reachability
check before handing off, because `syncSoftware.sh`'s own `ssh` calls have no
connect timeout and hang for a long time if the Orin isn't actually on the
network (not powered on, USB-Ethernet not plugged in).

If the user hasn't said whether to run the on-Orin test suite too, `--test`
is worth asking about rather than assuming -- it roughly doubles the time
this takes.

**If `run-tests` (the Docker-based CI-equivalent suite) hasn't been run
recently on this change, suggest running it first.** It's faster to catch a
real bug there than after a round-trip sync to the robot.

## 2. Launching on the car

```bash
.agents/skills/deploy-to-car/scripts/deploy.sh launch-cmd [launch.sh args...]
```

This only **prints** the `ssh -t ... launch.sh ...` command -- it does not
run it. `launch.sh` starts real actuators on a vehicle that can physically
move (its own preflight output says as much: "keep the offboard remote in
hand and the wheels clear"). That's a decision that needs the user actually
present, not something to run in the background the way `sim-launch` runs
the simulator. Hand them the printed command (or run it in the foreground,
attached, only after they've explicitly confirmed *right then* that they're
at the bench and ready) -- never detach it, and don't treat an earlier
approval to deploy as approval to also launch.

Common args to pass through: `--fake-arduino` (no Arduino attached --
useful for confirming the sync worked without arming anything real),
`--no-zed`, `--device /dev/ttyACM1`, `--rosboard`. For the full flag list,
ssh in and run `./software/scripts/launch.sh -h` directly.

## Notes

- Host/dir/workspace defaults (`tejam@192.168.55.1`, `~/software`,
  `~/ros2_ws`) come from `syncSoftware.sh`'s own `ORIN_HOST`/`ORIN_DIR`/
  `ORIN_WS` env vars -- override with `--host`, or export those vars before
  calling `sync`.
- This only reaches the Orin over its direct network (USB-Ethernet at
  `192.168.55.1` by default, or whatever `--host` points at) -- it can't be
  exercised from a dev machine that isn't actually connected to the robot.
- `stop.sh` on the Orin stops whatever `launch.sh` started (and anything
  else ROS/Gazebo-shaped it finds) if a session needs to be torn down
  without physical access to Ctrl-C the terminal.
