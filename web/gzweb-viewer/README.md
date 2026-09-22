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

Every topic name carries the world name, so the page has to be on the same
course the simulation is running. The websocket server does not care which,
so the wrong one still reports `Live simulation connected` -- it just draws a
course whose robot never moves.

The obstacle course's meshes need a patch that `src/main.js` applies at
startup: gzweb 3.0.2 cannot load a binary STL over HTTP and says nothing when
it fails, so without it every mesh silently goes missing and only the
collision primitives are drawn. The comment there has the details.

## Dependency overrides

gzweb 3.0.2 is the latest release and still asks for `protobufjs` 6,
`fast-xml-parser` 4 and `three-nebula` 10, all three of which carry published
advisories -- protobufjs 6 a critical one. Its bundle leaves those imports
external rather than inlining them, so the `overrides` block in
`package.json` is enough to make it resolve the patched majors instead:

| Package | gzweb asks for | We resolve |
| --- | --- | --- |
| `protobufjs` | `^6.11.3` | `^7.6.6`, the same copy `src/main.js` already used |
| `fast-xml-parser` | `^4.1.3` | `^5.11.1`, parses both world files byte-identically |
| `three-nebula` | `^10.0.3` | `^11.1.2`, which dropped the vulnerable `uuid` dependency outright |

Drop the whole block once gzweb bumps these itself; until then removing it
brings the advisories back.

## Run

In environment with [nvm](https://github.com/nvm-sh/nvm) installed:

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

That last one looks wasteful and is the only thing that works. Buckets and
hoops are static models, which Gazebo leaves out of the dynamic pose stream
altogether, and the websocket server latches what it sends on the whole-world
topic when a connection subscribes -- a long-lived subscription reports the
layout that was there when the page opened, for ever, however often it
resubscribes. A connection opened fresh is always current. A layout only
changes when somebody calls the randomiser, so arriving a few seconds later
is not late.

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
