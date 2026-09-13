# Jetson Onboard Software

ROS 2 (Jazzy) packages that run on the NVIDIA Jetson Orin Nano Super and drive the
Traxxas Slash 4X4 through the Arduino over the USB serial link described in the
[Onboard Protocol](../README.md) section of the top level README.

| Package | Contents |
| ------- | -------- |
| [`cfr_interfaces`](cfr_interfaces/) | `DriveCommand`, `ArduinoStatus`, `PathSegment` messages; `DrivePath` action |
| [`cfr_arduino_bridge`](cfr_arduino_bridge/) | `arduino_bridge_node`, `cmd_vel_to_drive_node`, `path_follower_node` |

## Gazebo Simulation

Simulation can be run in a container using the [unfrobotics/docker-ros2-jazzy-gz-rviz2:latest](https://github.com/UNF-Robotics/docker-ros2-jazzy-gz-rviz2) image.

`simulation.launch.py` mounts an RGB-D camera pair at the front of the simulated
Slash. It provides ZED-compatible ROS interfaces while the simulator is running:

| Topic | Type | Source |
| ----- | ---- | ------ |
| `/zed/zed_node/odom` | `nav_msgs/Odometry` | Gazebo vehicle odometry |
| `/zed/zed_node/left/image_rect_color` | `sensor_msgs/Image` | simulated left color camera |
| `/zed/zed_node/left/image_rect_color/camera_info` | `sensor_msgs/CameraInfo` | simulated left camera calibration |
| `/zed/zed_node/depth/depth_registered` | `sensor_msgs/Image` | simulated depth camera |
| `/zed/zed_node/depth/depth_registered/camera_info` | `sensor_msgs/CameraInfo` | simulated depth camera calibration |
| `/zed/zed_node/point_cloud/cloud_registered` | `sensor_msgs/PointCloud2` | simulated registered depth point cloud |

The camera sensors use a $110^\circ$ horizontal field of view, $640 \times 360$
resolution, 15 Hz update rate, and a 0.2 m to 20 m depth range.

The default headless world leaves Gazebo's rendered sensor system disabled so it
can run on systems without an EGL/OpenGL context; `/zed/zed_node/odom` remains
available for autonomy and the visual ZED 2i mount remains on the vehicle. The
image, depth, and point-cloud bridges require a GPU-capable container with the
NVIDIA OpenGL/EGL libraries exposed before enabling `gz-sim-sensors-system` in
the world.

## Nodes

### `arduino_bridge_node`

Owns `/dev/ttyACM0` and is the only thing allowed to write to the Arduino.

| Interface | Type | Direction |
| --------- | ---- | --------- |
| `~/drive_cmd` | `cfr_interfaces/DriveCommand` | subscribed |
| `~/status` | `cfr_interfaces/ArduinoStatus` | published |

Behaviour:

* Transmits a frame every cycle at `tx_rate_hz` (50 Hz default). The Arduino
  reverts to neutral after `200 ms` without a valid frame, so the transmit timer
  runs regardless of whether the autonomy stack is producing commands.
* Holds steering centered and the speed target at zero until the Arduino
  reports `AUTO_ACTIVE`. The firmware only makes the `AUTO_ARMED -> AUTO_ACTIVE`
  transition while `AUTO_READY` is set **and** steering sits inside the
  `127 +/- 5` deadband with a zero speed target, so sending real commands early
  would deadlock the handshake. Set `require_auto_active: false` for bench
  testing without the arming sequence.
* The speed target reverts to zero when `~/drive_cmd` goes stale
  (`command_timeout`), when the status link drops (`link_timeout`), or when the
  Arduino reports E-Stop.
* Clamps `DriveCommand.velocity` to `max_speed` (m/s), rate limits it with
  `speed_slew_rate` (m/s per second), and converts it into the target spur RPM
  tracked by the Arduino's closed loop speed controller.
* Owns that controller's gains as `speed_*` parameters and keeps the Arduino in
  step with them, including after an Arduino reset. See
  [Speed controller tuning](#speed-controller-tuning).
* Reopens the port automatically if the Arduino is unplugged or reset, waiting
  `boot_delay` seconds after each open for the Uno bootloader.

### `cmd_vel_to_drive_node`

Translates `geometry_msgs/Twist` on `cmd_vel` into a `DriveCommand`: `linear.x`
passes straight through as the velocity target (clamped to `max_speed`), and
steering comes from the bicycle model, `delta = atan(wheelbase * yaw_rate / speed)`,
normalized by `max_steering_angle`. Republishes at a fixed rate so the bridge
always has a fresh command. Sets `auto_ready` while `cmd_vel` is fresh.

Skip this node entirely if the autonomy stack publishes `DriveCommand` directly:

```bash
ros2 launch cfr_arduino_bridge arduino_bridge.launch.py use_cmd_vel:=false
```

### `path_follower_node`

Drives a fixed sequence of straight/turn segments -- e.g. an L-shape -- using
odometry for closed-loop feedback, and publishes `cmd_vel` like a human
operator would. Sits upstream of `cmd_vel_to_drive_node` and knows nothing
about the vehicle's wheelbase or the wire protocol.

| Interface | Type | Direction |
| --------- | ---- | --------- |
| `~/odom` | `nav_msgs/Odometry` | subscribed (remapped to `/zed/zed_node/odom`) |
| `cmd_vel` | `geometry_msgs/Twist` | published (remapped to `/cmd_vel` by default) |
| `~/drive_path` | `cfr_interfaces/action/DrivePath` | action server |

The vehicle is Ackermann and cannot rotate in place, so a `PathSegment` is one
of two kinds, and a turn is physically driven as an arc rather than a spin:

* `STRAIGHT` -- drive `distance` metres, holding the heading measured at the
  start of the segment with a P controller.
* `TURN` -- drive at `turn_speed` while yawing at `turn_rate` until heading has
  rotated by `turn_angle` radians (positive is left, REP-103). Segments must
  satisfy `|turn_angle| <= pi`; split larger turns into multiple segments.

Both segment types decelerate as they approach their target and reject goals
with an empty segment list. The goal aborts if odometry is stale when it
starts or goes stale mid-path (`odom_timeout`), or if a segment does not
complete within `max_segment_duration` (wheel slip, stuck odometry). Only one
goal runs at a time; a second is rejected outright rather than queued.

```bash
ros2 launch cfr_arduino_bridge path_follower.launch.py
```

Drive the L-shape from the top of this file (5 ft straight, 90 deg right, 2 ft
straight; feet converted to metres):

```bash
ros2 action send_goal /path_follower/drive_path cfr_interfaces/action/DrivePath \
  "{segments: [
     {type: 0, distance: 1.524},
     {type: 1, turn_angle: -1.5708},
     {type: 0, distance: 0.610}
  ]}" --feedback
```

`--feedback` prints `current_segment` / `segment_progress` as it runs. Ctrl-C on
that `send_goal` process sends a cancel request, which brings the car to a stop
within one control tick.

#### `path_tui.py`

[`scripts/path_tui.py`](scripts/path_tui.py) is a terminal UI for building a
segment list and sending it, instead of hand-writing the `send_goal` YAML
above. Needs the workspace sourced first:

```bash
source ~/ros2_ws/install/setup.bash
~/software/scripts/path_tui.py                              # default action name
~/software/scripts/path_tui.py --action /other_ns/drive_path
```

| Key | Action |
| --- | ------ |
| `s` | add a `STRAIGHT` segment (prompts for distance in feet, `+` forward / `-` reverse) |
| `t` | add a `TURN` segment (prompts for angle in degrees, `+` left / `-` right) |
| `d` | delete the last segment |
| `c` | clear all segments |
| `r` | reset local odometry to the robot's current pose; available only when no path is running |
| `g` / Enter | send the goal and switch to a live progress view |
| `x` | cancel while executing |
| `q` | quit (cancels first if a goal is executing) |

Distance and angle are entered in feet/degrees for readability and converted
to the action's metres/radians internally. Segments are kept after a run
completes so a failed or canceled path can be resent as-is. Ctrl-C at any
point cancels an in-flight goal before exiting, the same as `q` -- closing the
TUI should stop the car, not abandon it mid-path.

While a path executes, the TUI shows local current pose, the active segment's
target pose, pose error, controller velocity/yaw-rate command, and the velocity
target and normalized steering from `/drive_cmd`. A turn has a target heading but not a
target position because its radius is determined by the vehicle steering
geometry, so its position target and error are shown as `n/a`.

`STRAIGHT` segments are green and `TURN` segments are yellow, in both the
segment list and the live progress bar during execution; the result screen is
green on success, red on failure/rejection. Falls back to plain text on a
terminal without color support.

For a single command that brings up the actuator link, ZED, and
`path_follower_node` and then drops straight into the TUI, use
[`scripts/launchPathFollowingTUI.sh`](scripts/launchPathFollowingTUI.sh):

```bash
~/software/scripts/launchPathFollowingTUI.sh
~/software/scripts/launchPathFollowingTUI.sh --device /dev/ttyACM1
~/software/scripts/launchPathFollowingTUI.sh --no-stack       # bridge/ZED already up elsewhere
~/software/scripts/launchPathFollowingTUI.sh --fake-arduino   # no Arduino attached
```

It starts `launch.sh` and `path_follower.launch.py` in the background (logs go
to `/tmp/cfr_path_following/*.log`, keeping the TUI's screen clean) and runs
the TUI in the foreground. Quitting the TUI, or Ctrl-C, tears both background
launches down -- the same fate-sharing `launch.sh` itself uses for the bridge
and ZED. Its bridge converter listens only on `/path_follower/cmd_vel`, so
unrelated publishers on global `/cmd_vel` cannot arm or command the car before
a path goal is accepted.

The follower's odometry initialization and reset messages are in
`/tmp/cfr_path_following/path_follower.log` during a TUI session.

## Wire format

Both directions are ASCII, comma separated, newline terminated, at 115200 baud.
The field tables are in the [Onboard Protocol](../README.md#onboard-protocol)
section of the top level README. In brief:

Jetson -> Arduino is a drive command every cycle, always exactly
`C,b,sss,+rrrrr\n` (auto ready, steering byte, signed target spur RPM), plus a
checksummed `G,...` speed gains frame whenever the Arduino is not yet running
the current gains. Every field is fixed width on purpose: the firmware requires
the exact frame length, so a byte lost on the link rejects the frame instead of
shifting a value into a different command.

Arduino -> Jetson is
`estop,auto_arm,manual_start,mode,battery,rpm,target_rpm,throttle_us,gains_seq,`
at 20 Hz. `Deserialize()` requires exactly nine fields, so a firmware from
before closed loop speed control is rejected outright rather than misread.
Flash both sides together. `D,` debug and `T,` controller trace lines share the
link; the bridge skips them and they land only in `rx_trace_path`.

Spur RPM comes from a hall sensor watching a single trigger magnet in the spur
gear, so it counts spur revolutions -- not motor and not wheel revolutions.
`arduino_bridge` converts between it and m/s in both directions using the
`spur_to_wheel_ratio` (default `2.85`, the Slash 4X4 transmission, independent
of pinion) and `tire_diameter` (default `0.1143` m, the nominal 4.5" Traxxas
6764 Gravix 2.8" tire) parameters: `DriveCommand.velocity` into the target RPM,
and the reported RPM into the `wheel_rpm`, `speed` and `target_speed` fields of
`ArduinoStatus`. 1 m/s is about 476 spur RPM. The sensor cannot see direction,
so the sign of `rpm` and `speed` is the Arduino controller's estimate: the
direction it last drove. Speeds assume no wheel slip. Foam tires grow with
speed, so calibrate `tire_diameter` with a measured roll-out if accuracy
matters; it scales commanded and reported speed together.
Zero is ambiguous between stopped, no sensor fitted, and a dead link; check
`link_ok` on `ArduinoStatus` to rule out the last.

## Speed controller tuning

The Arduino runs a feedforward plus PID loop from target spur RPM to throttle
pulse every 20 ms. The control law, speed measurement, direction handling and
braking are described in [Speed Control](../README.md#speed-control). Its gains
are `arduino_bridge` parameters, and they are the only bridge parameters that
take effect when changed at runtime:

| Parameter | Units | Meaning |
| --------- | ----- | ------- |
| `speed_ks` | us | static feedforward, added whenever the target is nonzero |
| `speed_kv` | us per m/s | velocity feedforward |
| `speed_kp` | us per m/s of error | proportional |
| `speed_ki` | us per m/s of error, per second | integral |
| `speed_kd` | us per m/s per second | derivative, on measured speed |
| `speed_i_limit` | us | integrator clamp |
| `speed_output_limit` | us, at most 500 | largest drive offset from the 1500 us neutral pulse |
| `speed_brake_limit` | us, at most 440 | braking effort; `0` disables braking so the car coasts |
| `speed_trace` | bool | have the Arduino emit a `T,` trace line every 20 ms |

"us" is microseconds of throttle pulse. The bridge converts the rate gains into
the firmware's per-1000-spur-RPM units, sends them tagged with a new sequence
number, and resends until the Arduino echoes that number; `gains_applied` on
`ArduinoStatus` is false in between. `ros2 param set` rejects a value the
firmware would not accept.

The defaults were tuned **with the car on blocks** (see the bench notes under
[Speed Control](../README.md#speed-control)). Expect to retune on the ground:
the car adds several times the inertia plus rolling drag, so the feedforward
terms will need to rise and the loop will respond more slowly than on blocks.

A procedure for the ground, with the E-Stop in hand and plenty of room:

1. Log every controller tick to `rx_trace_path`:
   `ros2 param set /arduino_bridge speed_trace true`.
2. Feedforward first. Set `speed_kp` and `speed_ki` to `0`, hold a few constant
   speeds, and adjust `speed_ks` and `speed_kv` until `speed` settles close to
   `target_speed` at each. `speed_ks` mostly decides how slowly the car can
   creep, `speed_kv` the slope above that. On blocks the integral term settled
   near zero at every speed, which is the thing to aim for.
3. Raise `speed_kp` until a step in target settles quickly without ringing,
   then `speed_ki` until steady error disappears. Raise `speed_i_limit` if the
   integrator pins against it on a grade.
4. Only then try `speed_brake_limit` for firmer stops: start small, on flat
   ground, and confirm the car stops rather than rolling back in reverse.
5. Copy the values into `config/arduino_bridge.yaml`, and ideally into the
   compiled-in `SpeedGains` defaults in the firmware (converted to per 1000 RPM
   by multiplying the rate gains by 2.1), then turn the trace back off.

Constant speeds are easiest to hold by publishing `DriveCommand` directly, with
the bridge launched `use_cmd_vel:=false` so nothing else publishes on the topic.
The bridge holds zero until the Arduino is `AUTO_ACTIVE`, so this is safe to
leave running while arming:

```bash
ros2 topic pub -r 20 /drive_cmd cfr_interfaces/msg/DriveCommand \
  '{auto_ready: true, steering: 0.0, velocity: 1.0}'
```

`T,` trace columns are: Arduino milliseconds, tracked RPM, measured RPM, then
feedforward, proportional, integral, derivative and total output in tenths of a
microsecond, the requested pulse, the braking flag, and the dither duty out of
256.

## Deploy from a development host

[`scripts/syncSoftware.sh`](scripts/syncSoftware.sh) rsyncs this directory to
`~/software` on the Orin and can build there over ssh:

```bash
./scripts/syncSoftware.sh --dry-run          # preview the transfer
./scripts/syncSoftware.sh --host orin.local  # sync only
./scripts/syncSoftware.sh --build --test     # sync, then build and test on the Orin
```

The host and destination come from `--host` / `--dir` or the `ORIN_HOST`,
`ORIN_DIR` and `ORIN_WS` environment variables. Build artifacts are never
transferred: a typical dev host is `x86_64` and the Orin is `aarch64`, so
compiled binaries are not portable between them and the Orin always builds its
own. Stale files on the Orin are left alone unless `--delete` is passed.

## Build

On the Orin, from wherever the sources landed:

```bash
~/software/scripts/build.sh          # Release build into ~/ros2_ws
~/software/scripts/build.sh --test   # build, then run the test suites
~/software/scripts/build.sh --clean cfr_arduino_bridge
```

[`build.sh`](scripts/build.sh) resolves the source directory from its own
location, so it works the same at `~/software` or in a clone of the repository.
The workspace defaults to `~/ros2_ws` and can be overridden with `ROS2_WS`.

It wraps these, if you would rather run them directly:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
colcon build --base-paths ~/software --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon test --base-paths ~/software --packages-select cfr_arduino_bridge
colcon test-result --verbose
```

`--base-paths` points colcon at the source directory, so the packages do not
have to be copied or symlinked into the workspace `src/`. When building from a
clone rather than a sync, add `--symlink-install` to pick up edits to the launch
file and config without rebuilding.

`test_protocol` covers the wire format, `test_serial_port` runs the port
against a pseudo terminal, and `test_path_geometry` covers `path_follower_node`'s
control law -- all three pass with no Arduino, camera, or car attached.

Built executables land in `build/cfr_arduino_bridge/bin/` and are installed to
both `install/cfr_arduino_bridge/bin/` and `install/cfr_arduino_bridge/lib/cfr_arduino_bridge/`.
The `lib/<package>/` copy is the one `ros2 run` and launch files resolve, so it
cannot be dropped; `bin/` is there for running a node directly:

```bash
./install/cfr_arduino_bridge/bin/arduino_bridge_node --ros-args -p device:=/dev/ttyACM0
```

Serial access without root:

```bash
sudo usermod -aG dialout $USER   # log out and back in
```

`ModemManager` will probe a freshly enumerated `/dev/ttyACM*` and corrupt the
first second of traffic. Either uninstall it or exclude the Arduino:

```bash
sudo systemctl disable --now ModemManager
```

## Run

[`launch.sh`](scripts/launch.sh) is the onboard bringup: the actuator link and
the ZED camera together.

```bash
~/software/scripts/launch.sh                       # bridge + ZED
~/software/scripts/launch.sh --no-zed              # actuator link only
~/software/scripts/launch.sh --rosboard            # also serve rosboard on :8888
~/software/scripts/launch.sh --device /dev/ttyACM1
~/software/scripts/launch.sh --no-cmd-vel          # autonomy publishes DriveCommand directly
~/software/scripts/launch.sh max_speed:=1.0        # extra args pass to the bridge launch
```

The bridge starts first so the Arduino is receiving neutral commands while the
camera initializes. The two share a fate: if either exits, the other is torn
down, rather than leaving a live actuator link with dead perception. Ctrl-C
stops everything.

Preflight covers the three things that usually go wrong: the device is missing,
the user is not in `dialout`, or `ModemManager` is probing the port.
`--skip-checks` bypasses them. The ZED node comes from `zed_wrapper`, built per
[../zed/README.md](../zed/README.md); `--no-zed` runs without it.

### Gazebo speed-course simulation

The Gazebo launch replaces the USB Arduino and ZED camera with simulation. It
runs the existing `cmd_vel_to_drive_node` and `path_follower_node` unchanged:
the simulator consumes `/drive_cmd`, publishes simulated Arduino status on
`/arduino_bridge/status`, and bridges Gazebo's collision-aware odometry to
`/zed/zed_node/odom`.

Install Gazebo Harmonic plus its ROS 2 Jazzy integration on the development
machine:

```bash
sudo apt install gz-harmonic ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge
```

Build, source, then start the simulation:

```bash
~/software/scripts/build.sh
source "${ROS2_WS:-$HOME/ros2_ws}/install/setup.bash"
ros2 launch cfr_arduino_bridge simulation.launch.py
```

`ROS2_WS` must name the workspace built by `build.sh` (it defaults to
`~/ros2_ws`). Source that workspace's `install/setup.bash`, rather than an
unrelated ROS overlay, before sending `DrivePath` goals.

The launch defaults to headless Gazebo, suitable for containers. To use GUI,
pass `gui:=true` from an authorized desktop X session; `DISPLAY` alone is not
enough because the container must also have permission to open that display.

`speed_course.sdf` is a 44.7 m by 34.5 m field scaled from the supplied speed
course SVG's 2640 by 2040 drawing area. It contains a basic 0.324 m-wheelbase
Slash model, collision bales, and the course perimeter/chicanes. Gazebo is the
odometry source, so collisions and vehicle motion feed the same closed-loop
path follower used on the car. Send `/path_follower/drive_path` goals with the
same command shown above. No Arduino, ZED, XBee, or physical E-Stop is present;
simulation status is always link-healthy and must never be treated as a safety
test.

### No Arduino attached

```bash
~/software/scripts/launch.sh --fake-arduino --no-zed
```

Runs [`scripts/fake_arduino.py`](scripts/fake_arduino.py) -- a PTY that speaks
just enough of the onboard protocol to unblock `arduino_bridge_node`'s
`AUTO_ARMED -> AUTO_ACTIVE` handshake (`auto_ready=1` with both axes centered)
-- and points the bridge at it instead of a real device. **This does not
simulate the offboard XBee/RC/E-Stop link**, which the real sketch's mode state
machine is also gated on (`offboardTimedOut` forces `Mode::ESTOP` regardless of
what the Jetson sends) and which has no code in this repo yet. Use it to
exercise message flow through `arduino_bridge_node` /
`cmd_vel_to_drive_node` / `path_follower_node` with no hardware attached, not
as a stand-in for a safety-validated bench session -- there are no real
actuators and no real E-Stop behind it. Can also be run standalone:

```bash
python3 scripts/fake_arduino.py                                # creates /tmp/fake_arduino
~/software/scripts/launch.sh --device /tmp/fake_arduino --skip-checks
```

Underneath it is just:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cfr_arduino_bridge arduino_bridge.launch.py device:=/dev/ttyACM0 &
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i &
```

Watch what the Arduino reports:

```bash
ros2 topic echo /arduino_bridge/status
```

Drive it by hand (wheels off the ground, offboard E-Stop within reach):

```bash
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.5}, angular: {z: 0.3}}'
```

## Bench checklist

1. Car on a stand, wheels clear, offboard E-Stop in hand.
2. Launch the bridge and confirm `/arduino_bridge/status` shows `link_ok: true`
   and `gains_applied: true`. A bridge log of "dropping malformed status frame"
   means the firmware and the Jetson packages are from different protocol
   versions; flash and build them together.
3. Arm autonomy offboard and confirm mode goes `1 -> 3` (`RC_ARMED` to
   `AUTO_ARMED`), then `-> 4` (`AUTO_ACTIVE`) once commands start flowing.
4. Publish a small `cmd_vel` and confirm the steering servo and wheels respond
   in the expected directions, and that `speed` follows `target_speed`. Flip
   `invert_steering` / `invert_speed` if not.
5. Kill the publisher and confirm the car coasts to a stop: `target_speed` and
   `throttle_us` return to `0` and `1500` within `command_timeout`.
6. For `path_follower_node`: confirm `ros2 topic echo /zed/zed_node/odom` is
   publishing, then send a short single `STRAIGHT` segment on a stand and
   check the wheels turn the right way for the whole segment (not just at the
   start) before ever sending a `TURN` segment or running on the ground.
