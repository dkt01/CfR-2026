# CfR Gazebo Browser Viewer

Browser viewer built with `gzweb` 3. It renders a course's world file,
including the bales, obstacle meshes and Slash model, then updates the Slash
pose from Gazebo's live dynamic-pose transport topic.

## Courses

The speed course is the default; add `?course=obstacle` for the obstacle
course:

| Course | URL |
| ------ | --- |
| Speed | `http://localhost:5173/` |
| Obstacle | `http://localhost:5173/?course=obstacle` |

Select the course running in Gazebo. A mismatched page can show "Live
simulation connected" while its robot stays still because topic names include
the world name.

`src/main.js` patches gzweb 3.0.2 to load binary STL meshes over HTTP; without
it the obstacle meshes silently disappear.

## Dependency overrides

gzweb 3.0.2 requests dependency versions with published advisories. Its
external imports allow `package.json` overrides to select newer versions:

| Package | gzweb asks for | We resolve |
| --- | --- | --- |
| `protobufjs` | `^6.11.3` | `^7.6.6`, the same copy `src/main.js` already used |
| `fast-xml-parser` | `^4.1.3` | `^5.11.1`, parses both world files byte-identically |
| `three-nebula` | `^10.0.3` | `^11.1.2`, which dropped the vulnerable `uuid` dependency outright |

Remove the overrides when gzweb updates its dependencies.

## Run

For the three-lap wall follower, install dependencies and start the simulation
and viewer in a ROS 2 environment:

```bash
cd web/gzweb-viewer && npm ci && cd ../..
bash ./jetson/scripts/launch_wall_web.sh
```

Open `http://localhost:5173/`. **Set signal** starts or stops the follower;
the lap counter stops it after three laps. `npm` and the built ROS 2 workspace
must be available in the shell.

To run the viewer alone with [nvm](https://github.com/nvm-sh/nvm):

```bash
nvm install 24
nvm use 24
npm install
npm run dev -- --port 5173
```

Open `http://localhost:5173/` on the host. Vite listens on all container
interfaces, so use the Docker host's reachable address if the browser is on a
different machine.

## What the page follows

| | Where it comes from |
| --- | --- |
| The robot | `dynamic_pose/info`, live |
| The start signal's arms | the same stream -- they turn on a joint, so Gazebo reports them as a moving link |
| Buckets and hoops | sampled from `pose/info` every 3 s over a connection of its own |

Buckets and hoops are absent from the dynamic pose stream. The websocket
server latches the whole-world topic when a client subscribes, so a persistent
connection would miss layout changes. A fresh connection every 3 s gets the
current layout.

## Live Gazebo Transport Bridge

The simulation package includes a `gz-launch7` WebSocket configuration for
Gazebo Harmonic. Start the simulation with the optional bridge enabled:

```bash
ros2 launch cfr_arduino_bridge simulation.launch.py websocket:=true
```

It listens on `ws://localhost:9002` and provides gzweb-compatible Gazebo
Transport data. The page reports `Live simulation connected` when it is
receiving the Slash pose stream.

The plugin is not in the simulation image. It ships as a binary package now,
so it no longer has to be built from source:

```bash
sudo apt install ros-jazzy-gz-launch-vendor
```

Without it `gz launch` is not a command, the websocket process exits 255 at
startup and the rest of the simulation comes up normally -- so the symptom is
a viewer stuck on `Connecting to simulation` rather than anything obviously
missing.

Simulation launch also starts an owned HTTP teleport bridge on port `9003`.
The viewer sends JSON to this bridge; the bridge issues Gazebo's native
`set_pose` request, so teleporting does not depend on WebSocket protobuf
request support.
