# CfR Gazebo Browser Viewer

Browser viewer built with `gzweb` 3. It renders the simulation's
`speed_course.sdf`, including the course bales and Slash model, then updates
the Slash pose from Gazebo's live dynamic-pose transport topic.

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

## Live Gazebo Transport Bridge

The simulation package includes a `gz-launch7` WebSocket configuration for
Gazebo Harmonic. Start the simulation with the optional bridge enabled:

```bash
ros2 launch cfr_arduino_bridge simulation.launch.py websocket:=true
```

It listens on `ws://localhost:9002` and provides gzweb-compatible Gazebo
Transport data. The `gz-launch7` WebSocket plugin must be built and installed
under `/opt/ros_ws/install`, with `libwebsockets-dev` and
`ros-jazzy-gz-tools-vendor` installed. Start the simulation with
`websocket:=true`; the page reports `Live simulation connected` when it is
receiving the Slash pose stream.

Simulation launch also starts an owned HTTP teleport bridge on port `9003`.
The viewer sends JSON to this bridge; the bridge issues Gazebo's native
`set_pose` request, so teleporting does not depend on WebSocket protobuf
request support.
