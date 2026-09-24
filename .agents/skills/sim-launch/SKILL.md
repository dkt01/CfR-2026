---
name: sim-launch
description: Launch the CfR-2026 Gazebo simulation (speed or obstacle course) in its Docker container and start the gzweb browser viewer, so the user can see the course and watch the vehicle. Use this whenever the user asks to launch, start, run, or open the simulation/sim/Gazebo/course/viewer for this project, wants to "see" a course change or obstacle-course fix in the browser, or asks to stop/restart/tear down the simulation. Covers the full sequence by itself: starting or reusing the Docker container, installing ros-jazzy-gz-launch-vendor if missing, building the ROS workspace, launching with the websocket bridge, verifying the ports actually bound (not just that Docker published them), and starting the vite dev server for the browser viewer. Prefer this over running the individual docker/ros2 launch/npm commands by hand.
---

# CfR-2026 simulation launcher

Wraps the manual sequence for standing up the Gazebo sim + browser viewer into
one idempotent script, `scripts/sim.sh`, so this doesn't need to be
reconstructed step by step each time (Docker image, container name, port
numbers, the gz-launch-vendor install, the docker-proxy false-positive on the
websocket port, and so on all live in the script instead of in memory).

## Starting the sim

If the user hasn't said which course, ask -- `speed` or `obstacle` -- rather
than guessing; they load different worlds and the viewer URL differs
(`?course=obstacle` vs. no query param).

```bash
.agents/skills/sim-launch/scripts/sim.sh start --course obstacle
.agents/skills/sim-launch/scripts/sim.sh start --course speed --gui       # local GUI window (needs an authorized display)
.agents/skills/sim-launch/scripts/sim.sh start --course obstacle --sensors # renders the ZED camera, ~5-12 Hz headless
.agents/skills/sim-launch/scripts/sim.sh start --course obstacle --laps 1  # override the default lap count
```

This prints the viewer URL when it's done, e.g.
`http://localhost:5173/?course=obstacle`. Hand that URL to the user (or open
it if there's a browser tool available) rather than just saying "it's
running."

**Idempotent by design.** Calling `start` again while the sim is already up
does not relaunch it -- it just confirms the container, ports and viewer are
alive and reprints the URL. This matters because the running sim holds state
(randomized bucket/hoop layout, the vehicle's driven pose) that a careless
relaunch would silently throw away. If the user wants a genuinely fresh
world, or wants different flags (a different course, `--gui`, `--sensors`),
run `stop` first, then `start` again.

## Stopping

```bash
.agents/skills/sim-launch/scripts/sim.sh stop
```

Restarts the Docker container (cleanly kills Gazebo and every ROS node the
launch started) and kills the tracked viewer dev server. The container
itself is left in place so the next `start` doesn't need to recreate it or
reinstall `gz-launch-vendor`.

## Checking state

```bash
.agents/skills/sim-launch/scripts/sim.sh status
```

Reports whether the container/simulation/viewer are up, and whether the
websocket and teleport ports are genuinely listening inside the container --
not just whether Docker has them published, which can look open on the host
even when nothing inside is serving them.

## Things worth knowing before debugging by hand

- **The two Docker images this needs are already local**
  (`unfrobotics/docker-ros2-jazzy-gz-rviz2:latest` for the sim,
  `ros:jazzy-ros-base` for CI-equivalent test runs -- this skill only touches
  the former). If `start` fails at the `docker run` step, check `docker
  images` before assuming a pull is needed.
- **`docker exec` does not source ROS.** `bash -lc` skips the `.bashrc` line
  that does it, which is why every exec in the script explicitly sources
  `/opt/ros/jazzy/setup.bash` (and the workspace overlay) first.
- **A live TCP connect to a published port is not proof the process behind
  it is up.** `docker-proxy` binds the host side of `-p 9002:9002` regardless
  of whether anything inside the container is listening, so `status` and the
  launch's own readiness check read the container's `/proc/net/tcp` instead
  of trusting a host-side connect.
- **The obstacle course and speed course are different worlds with different
  layout files** -- `obstacle_course.launch.py` vs. `speed_course.launch.py`,
  wrapping the shared `simulation.launch.py`. Passing the wrong `?course=`
  query param to the viewer still says "Live simulation connected" (the
  websocket server doesn't care which world it's bridging), it just never
  moves the drawn vehicle, which is a confusing failure mode if it comes up.
- If `npm run dev` itself is missing dependencies, `cd web/gzweb-viewer &&
  npm install` first -- the script assumes `node_modules` already exists,
  which it normally does since it's committed to not require setup on a
  fresh clone... unless `node_modules` was gitignored and this is a truly
  fresh checkout, in which case run `npm install` once.
