# Jetson Onboard Software

ROS 2 (Jazzy) packages that run on the NVIDIA Jetson Orin Nano Super and drive the
Traxxas Slash 4X4 through the Arduino over the USB serial link described in the
[Onboard Protocol](../README.md) section of the top level README.

| Package | Contents |
| ------- | -------- |
| [`cfr_interfaces`](cfr_interfaces/) | `DriveCommand`, `ArduinoStatus`, `PathSegment`, `StartSignal` messages; `DrivePath` action |
| [`cfr_arduino_bridge`](cfr_arduino_bridge/) | `arduino_bridge_node`, `cmd_vel_to_drive_node`, `path_follower_node`, `start_signal_detector_node`, `obstacle_randomizer_node` |

## Gazebo Simulation

Two courses are simulated, each with its own launch file. Both run in a
container built from the
[unfrobotics/docker-ros2-jazzy-gz-rviz2:latest](https://github.com/UNF-Robotics/docker-ros2-jazzy-gz-rviz2)
image.

```bash
ros2 launch cfr_arduino_bridge speed_course.launch.py      # 135 ft x 47 ft oval
ros2 launch cfr_arduino_bridge obstacle_course.launch.py   # 65 ft x 48 ft, 11 sections
```

Both accept `gui:=true` (needs an authorized host display),
`websocket:=true` (see [the browser viewer](../web/gzweb-viewer/README.md),
which takes `?course=obstacle` to match)
and `sensors:=true` (below). Both are thin wrappers over
`simulation.launch.py`, which still works directly with `world:=<path>` if
you want a world neither names.

In both worlds the car starts on the start/finish line at the origin, facing
`+x` the way it drives away from the line, so `path_follower_node` goals read
the same as they would on the real course.

### The obstacle course

The course is generated from the same site-layout DXF as the speed course, by
[`scripts/generate_obstacle_course.py`](scripts/generate_obstacle_course.py).
Its eleven sections run start lane, ramp up, flat bridge, helical ramp down,
tunnel, narrow path, gravel box, potholes, buckets, hoops, car wash, and a
banked turn, walled throughout with the 121 straw bales the drawing places.

Obstacle *shapes* come from the team's CAD, tessellated into the STL visuals
in [`cfr_arduino_bridge/meshes/`](cfr_arduino_bridge/meshes/). Collision is
primitives, sized from the DXF: the meshes are decimated hard enough that
driving on them would catch a wheel on a triangle edge. Two consequences
worth knowing:

* The whole ramp/bridge/helix structure is primitives for both, because the
  car drives on it and because the CAD and DXF disagree by about 5% on the
  helix radius. The DXF wins there -- the bale walls around it were drawn to
  the DXF -- and it reproduces the drawing's annotated 4 ft centreline
  radius, 41.5 in width and 11% grade.
* The car wash's hanging ribbons are drawn but have no collision. They are
  streamer weight and could not deflect the car, so simulating forty contacts
  against them would cost time and change nothing.

The gravel box is a friction feature rather than a geometric one. Gazebo has
no granular physics, so the box gets a low-friction lid -- `mu` 0.35 against
the course floor's 50 -- and a scatter of 90 pebbles for the suspension to
find. Both wheel slip and an uneven surface, which is what the section tests.

### Variable elements

Three things change between runs on the real course, and
`obstacle_randomizer_node` moves all three in a running simulation, so a
layout can be re-drawn without restarting Gazebo.

The node runs on **both** courses, because both start the same way: on the
visual signal. The speed course has no buckets or hoops, so there its layout
carries only the signal block and `randomize` and `reset` say as much rather
than pretending to shuffle something.

| Service | Type | Effect |
| ------- | ---- | ------ |
| `/obstacle_randomizer/randomize` | `std_srvs/Trigger` | draw a new bucket and hoop layout |
| `/obstacle_randomizer/reset` | `std_srvs/Trigger` | restore the layout the drawing shows, and red |
| `/obstacle_randomizer/start_signal` | `std_srvs/SetBool` | `true` shows green, `false` red |

```bash
ros2 service call /obstacle_randomizer/randomize std_srvs/srv/Trigger
ros2 service call /obstacle_randomizer/start_signal std_srvs/srv/SetBool "{data: true}"
```

`/obstacle_randomizer/start_signal_green` (`std_msgs/Bool`, transient local)
carries the signal state, so a detector can be scored against what the signal
is actually showing.

**Buckets.** Two to nine, at least 3 ft between centres and 2.5 ft off the
bale walls, both measured off the drawing. Those two numbers are also why the
drawing's "placed so a path exists around and between buckets" needs no
reachability check: 3 ft between centres leaves a 0.62 m gap between two
0.29 m buckets, and the same off the walls, and the car is 0.30 m wide.
Keeping the spacing is keeping the path. Nine models exist from the start --
Gazebo will not spawn a static model on demand -- and a draw stands the ones
it does not use off the course, south of the bale walls.

Set `bucket_count` to pin the count, or `seed` to repeat a draw:

```bash
ros2 param set /obstacle_randomizer bucket_count 9
ros2 param set /obstacle_randomizer seed 12345
```

**Hoops.** Three, each sliding laterally along the dashed line the drawing
puts it on, clamped by half a base length so a hoop cannot end up half
outside the bales.

**Start signal, on both courses.** Red and green arms 90 degrees apart on a
common pivot 32 in up; whichever is horizontal stands out past the sky blue
board and is the one the car sees. The arms ride a revolute joint whose limits
are those two positions, and it rests at the red one, so **a freshly loaded
world always shows red** without anything having to command it.

The service **turns** the arm rather than snapping it: 90 degrees in one
second, matching the signal on the course and giving a detector the part-way
arm it will have to cope with. The randomiser ramps the joint setpoint at
25 Hz over `/start_signal/arm`, which `simulation.launch.py` bridges to
Gazebo's joint-position controller; the controller follows the ramp, so the
rate is what the node says it is. `signal_sweep_rate` (degrees per second,
default 90) changes it, and 0 commands the far end directly for a scripted
test that does not want to wait.

Through the camera, one sweep looks like this -- the counts
`start_signal_detector_node` reports, from
[`scripts/check_start_signal.py`](scripts/check_start_signal.py) against the
speed course:

```
   t(s)   red px   green px   reads
   0.00     133         0     red
   0.33     132        10     red
   0.46     131        42     red
   0.53     123        61     red
   0.66      84        94     green
   0.73      58       113     green   <- ~/go latches here
   0.99       0       133     green
```

The two arms are 90 degrees apart on one pivot, so through the turn they
trade projected area rather than both disappearing: the total stays near 133
px and the verdict is whichever count is ahead. With a tighter saturation
floor than the detector's the crossover becomes a hole instead -- both arms
wash out around 45 degrees and a frame or two reads as neither -- so the
detector has to cope with both shapes, and does: an unconfirmed frame holds
its counters rather than resetting them.

Stepping the model's pose would have kept one mechanism for everything and
was tried first. It does not work: each `set_pose` is a `gz service`
subprocess costing about 340 ms, so a one-second sweep fits three poses and
arrives as a stutter. Publishing a setpoint costs nothing.
[`scripts/start_signal.py`](scripts/start_signal.py) builds the models, and
both course generators call it, so the two courses cannot drift apart.

The signal is not in the DXF -- only the CAD has one -- so each course picks
its own spot, a single `SIGNAL_POSITION` constant in its generator. Both sit
about 4 m ahead of the car at a bearing of 20 degrees, near enough to read and
far enough off to the side to be outside the lane, with the sight line from
the 8 in camera clearing the 14 in bale wall between the two:

| Course | Position | Bearing | Range | Clears the wall by |
| ------ | -------- | ------- | ----- | ------------------ |
| Obstacle | (3.40, 1.40) | 20.3 deg | 4.04 m | 24 mm |
| Speed | (16.00, 3.40) | 19.6 deg | 4.07 m | 54 mm |

At that range the arm is about 15 x 12 px of saturated red or green in a
640 x 360 frame. [`scripts/check_signal_sightline.py`](scripts/check_signal_sightline.py)
re-derives all of it from the generated worlds and fails if a nudged constant
puts the signal behind a bale or outside the camera's field:

```bash
./scripts/check_signal_sightline.py
```

Bounds for everything the randomiser moves come from the layout file beside
each world --
[`obstacle_course_layout.yaml`](cfr_arduino_bridge/config/obstacle_course_layout.yaml)
and
[`speed_course_layout.yaml`](cfr_arduino_bridge/config/speed_course_layout.yaml)
-- which the generators write, so the randomiser and the world cannot drift
apart.

Two things the drawing calls variable are **not** randomised: the pothole
bumps, which are placed as drawn because their matching holes are cut into
the board mesh and cannot move with them, and the bucket section's entrance,
which would mean moving bale walls.

### Cameras

`simulation.launch.py` mounts an RGB-D camera at the front of the simulated
Slash and publishes it under ZED-compatible names:

| Topic | Type | Source |
| ----- | ---- | ------ |
| `/zed/zed_node/odom` | `nav_msgs/Odometry` | Gazebo vehicle odometry |
| `/zed/zed_node/left/image_rect_color` | `sensor_msgs/Image` | simulated left color camera |
| `/zed/zed_node/left/image_rect_color/camera_info` | `sensor_msgs/CameraInfo` | simulated camera calibration |
| `/zed/zed_node/depth/depth_registered` | `sensor_msgs/Image` | simulated depth camera |
| `/zed/zed_node/point_cloud/cloud_registered` | `sensor_msgs/PointCloud2` | simulated registered depth point cloud |

$110^\circ$ horizontal field of view, $640 \times 360$, 15 Hz, 0.2 m to 20 m.
Colour and depth come from one `rgbd_camera` sensor, so they share one
calibration and there is one `camera_info` rather than two.

Rendered sensors need a render context, so they live in a second world file
and are off by default:

```bash
ros2 launch cfr_arduino_bridge obstacle_course.launch.py sensors:=true
```

Gazebo ignores its own default server config once a world declares plugins,
so the sensors system has to be written into the world file -- there is no
launch argument for it. Rather than keep a second copy of each world, both
worlds carry two `<!-- cfr:sensors-... -->` markers and `simulation.launch.py`
fills them in, writing the result to `/tmp/cfr_sim/`. Without `sensors:=true`
the world is used exactly as committed, comments and all.

A second copy would have been the simpler mechanism, but the speed course's
world is maintained by hand, and a derived copy of a hand-maintained file goes
stale the first time somebody edits one and not the other.
`/zed/zed_node/odom` is published either way; without `sensors:=true` only the
images are missing.

A GPU is not required. `LIBGL_ALWAYS_SOFTWARE=1` renders the sensors world on
llvmpipe at roughly 5 Hz against the 15 Hz the sensor asks for, which is
enough to check that the car sees the start signal turn green:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch cfr_arduino_bridge obstacle_course.launch.py sensors:=true
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch cfr_arduino_bridge speed_course.launch.py sensors:=true
```

### Regenerating the course

Neither step is needed to run the simulation -- the worlds and meshes are
committed. Both are needed when the drawing or the CAD changes.

```bash
# Obstacle course world and the randomiser's bounds, from the site-layout DXF.
./scripts/generate_obstacle_course.py "2026 course designs v08 ... site layout 2.dxf"

# Speed course start signal and its layout file.  The DXF argument is optional
# and only the straw-bale block needs it -- see the note below.
./scripts/generate_speed_course.py

# STL visuals, from the assembled obstacle CAD.  Needs Docker; builds its own
# image, because the OpenCASCADE bindings want a Python neither host has.
./scripts/convert_obstacle_meshes.sh "~/Downloads/All Obstacles.step"
```

**The speed course's bales do not currently regenerate.**
`generate_speed_course.py` picks the Speed Course out of the drawing by an x
range, and in "site layout 2" the two courses are separated by y instead, so
it finds 82 bales there rather than 202. Selecting by y does find 202 -- but a
different 202, up to 39 m away and rotated a quarter turn from the ones
committed in the world, so the committed Speed Course came from a revision
again different from either. Until that is reconciled the DXF argument is
optional and the bale block is only rewritten when one is passed, so
regenerating the start signal cannot silently replace the course.

`convert_obstacle_meshes.sh` wants the *assembled* STEP export, not the
per-part directory: the individual files carry no assembly relationships, so
a car wash or a start signal cannot be put back together from them.

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
* Holds both axes neutral until the Arduino reports `AUTO_ACTIVE`. The firmware
  only makes the `AUTO_ARMED -> AUTO_ACTIVE` transition while `AUTO_READY` is set
  **and** both axis commands sit inside the `127 +/- 5` deadband, so sending real
  commands early would deadlock the handshake. Set `require_auto_active: false`
  for bench testing without the arming sequence.
* Commands revert to neutral when `~/drive_cmd` goes stale (`command_timeout`),
  when the status link drops (`link_timeout`), or when the Arduino reports
  E-Stop.
* Scales throttle by `max_throttle` (0.25 by default) and rate limits it with
  `throttle_slew_per_s`.
* Reopens the port automatically if the Arduino is unplugged or reset, waiting
  `boot_delay` seconds after each open for the Uno bootloader.

### `cmd_vel_to_drive_node`

Translates `geometry_msgs/Twist` on `cmd_vel` into a normalized `DriveCommand`
using the bicycle model, `delta = atan(wheelbase * yaw_rate / speed)`, and
republishes at a fixed rate so the bridge always has a fresh command. Sets
`auto_ready` while `cmd_vel` is fresh.

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
target pose, pose error, controller velocity/yaw-rate command, and normalized
throttle/steering from `/drive_cmd`. A turn has a target heading but not a
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

### `start_signal_detector_node`

Watches the camera for the start signal -- a red arm that turns to a green one
on a common pivot, in about a second -- and latches it, so the driver has one
thing to wait on and no camera code of its own.

| Interface | Type | Direction |
| --------- | ---- | --------- |
| `image` | `sensor_msgs/Image` | subscribed (remapped to `/zed/zed_node/left/image_rect_color`) |
| `~/go` | `std_msgs/Bool` | published, transient local, latched |
| `~/state` | `cfr_interfaces/StartSignal` | published once per frame |
| `~/reset` | `std_srvs/Trigger` | wait for another start |
| `~/debug_image` | `sensor_msgs/Image` | published while `debug_image` is true |

`/start_signal_detector/go` is **the trigger**. It is published false once at
startup and true once the start is confirmed, on a transient local publisher,
so a driver launched after the signal turned still receives it -- and one
launched before gets the false, which is the difference between "not yet" and
"no detector running". Waiting on it is one subscription:

```bash
ros2 topic echo /start_signal_detector/go
ros2 service call /start_signal_detector/reset std_srvs/srv/Trigger   # next run
```

`/start_signal_detector/state` is the running commentary -- what this frame
shows, how many pixels of each colour, and where. A driver that must stop on a
red flag mid-run watches that rather than `~/go`, because `~/go` deliberately
stays up once a run has started: a single mis-hued frame must not be able to
retract a start that has already happened.

A start is a red arm that *becomes* green, not merely green in frame. Four
things stand between "something coloured" and releasing the car, because at
the 4 m both courses stand the signal at the arm is only about 15 x 12 px of a
640 x 360 frame:

* **Above the horizon only.** The arm is 32 in up and the camera 8 in up, so
  the arm is above the camera's horizon from anywhere on the course -- and for
  a level camera that horizon is the middle row of the image. `region`
  searches the top half, which is also all of the dark green ground plane
  excluded.
* **Hue bands, not channel comparisons.** The bands stop short of every other
  saturated hue either course puts in frame: the straw bales at 36 degrees,
  the ground at 105, the signal's own sky blue board at 197, the car wash's
  blue ribbons at 212. The arms themselves render at 1.5 and 130.
* **A cluster, not a count.** The threshold applies to the densest
  `cluster_window` box, so scattered matches across the region never add up to
  an arm the way a raw pixel count would.
* **Green where red was.** Both arms turn about one pivot, so the transition
  happens in one place, within `max_transition_distance`. This is what stops
  the car wash's twenty red ribbons -- the same red as the arms -- from
  pairing with a green somewhere else in frame.

Then `confirm_frames` frames of it, which at the camera's 15 Hz costs 133 ms.
Measured against both courses, `~/go` latches about 0.7 s into the 1 s sweep,
on 90 to 120 px of green; see the sweep table above.

Tuning is live: `ros2 param set` on any threshold rebuilds the classifier
without disturbing the latch, and a value that does not make sense is refused
with a reason rather than quietly clamped.

```bash
ros2 launch cfr_arduino_bridge start_signal.launch.py debug:=true
ros2 param set /start_signal_detector min_saturation 0.35
ros2 run rqt_image_view rqt_image_view /start_signal_detector/debug_image
```

`debug:=true` publishes each frame with the region and the winning cluster
drawn on it, which is how the bands get moved to fit the real signal. The
defaults live in
[`config/arduino_bridge.yaml`](cfr_arduino_bridge/config/arduino_bridge.yaml)
with a note on each.

The decision itself is in
[`start_signal_detector.py`](cfr_arduino_bridge/src/start_signal_detector.py),
free of ROS like `path_geometry` on the C++ side, and covered by
`test_start_signal_detector` against synthetic frames built from the worlds'
own colours. Nothing in a synthetic frame can show that Gazebo renders those
colours where the geometry says it will, so
[`scripts/check_start_signal.py`](scripts/check_start_signal.py) drives a
running simulation -- red, turn it green, wait for the trigger -- and prints
the frames it took:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch cfr_arduino_bridge speed_course.launch.py sensors:=true
./scripts/check_start_signal.py            # exits non-zero if a start is missed
```

Both course launches start the detector themselves with `sensors:=true`; on
the car it comes up with `start_signal.launch.py` alongside the ZED. It needs
no GPU: llvmpipe renders the camera at about 5 Hz, which still puts three or
four frames inside the turn.

## Wire format

Both directions are ASCII, comma separated, newline terminated at 115200 baud.

Jetson -> Arduino, always exactly `b,nnn,nnn\n`:

| Field | Description | Range |
| ----- | ----------- | ----- |
| 0 | Auto Ready | `0` or `1` |
| 1 | Steering command | `[0,255]`, `0` full left, `127` center, `255` full right |
| 2 | Throttle command | `[0,255]`, `0` full reverse, `127` neutral, `255` full forward |

Integer fields are zero padded to three digits on purpose: the firmware's
`FromJetson::deSerialize()` rejects payloads shorter than 6 characters, so an
unpadded frame such as `1,0,0` would be dropped silently. Padding also keeps
every frame a constant 10 bytes.

Arduino -> Jetson matches the `ToJetson` struct, with a trailing comma before the
newline:

| Field | Description | Range |
| ----- | ----------- | ----- |
| 0 | E-Stop State | `0` or `1`, `1` is active |
| 1 | Auto Arm | `0` or `1` |
| 2 | Manual Start | `0` or `1` |
| 3 | Mode | `[0,4]`, see the run mode table in the top level README |
| 4 | Battery Level | `[0,255]`, `0` empty, `255` full |
| 5 | Spur RPM | `[0,65535]` spur gear revolutions per minute |

Unlike the command direction these are *not* zero padded, so field widths vary
and frames run 12 to 19 bytes. `Deserialize()` requires exactly six fields,
which means a firmware predating the RPM field is rejected outright rather than
parsed with a stale speed. Flash both sides together.

Spur RPM comes from a hall sensor watching a single trigger magnet in the spur
gear, so it counts spur revolutions -- not motor and not wheel revolutions.
`arduino_bridge` converts it into the `wheel_rpm` and `speed` (m/s) fields of
`ArduinoStatus` using the `spur_to_wheel_ratio` (default `2.85`, the Slash 4X4
transmission, independent of pinion) and `tire_diameter` (default `0.1143` m,
the nominal 4.5" Traxxas 6764 Gravix 2.8" tire) parameters. `speed` is a
magnitude, since the sensor cannot see direction, and assumes no wheel slip.
Foam tires grow with speed, so calibrate `tire_diameter` with a measured
roll-out if accuracy matters.
Zero is ambiguous between stopped, no sensor fitted, and a dead link; check
`link_ok` on `ArduinoStatus` to rule out the last.

The firmware measures it by polling a pin change flag rather than taking an
interrupt, which cannot lose an edge but can merge two that fall inside one
loop stall. Expect a slight undercount at full throttle rather than a clean
signal; see the Onboard I/O section of the top level README.

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
against a pseudo terminal, `test_path_geometry` covers `path_follower_node`'s
control law, and `test_start_signal_detector` covers
`start_signal_detector_node`'s colour decision against synthetic frames -- all
of them pass with no Arduino, camera, or car attached.

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
~/software/scripts/launch.sh max_throttle:=0.15    # extra args pass to the bridge launch
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
2. Launch the bridge and confirm `/arduino_bridge/status` shows `link_ok: true`.
3. Arm autonomy offboard and confirm mode goes `1 -> 3` (`RC_ARMED` to
   `AUTO_ARMED`), then `-> 4` (`AUTO_ACTIVE`) once commands start flowing.
4. Publish a small `cmd_vel` and confirm the steering servo and ESC respond in
   the expected directions. Flip `invert_steering` / `invert_throttle` if not.
5. Kill the publisher and confirm the car returns to neutral within
   `command_timeout`.
6. For `path_follower_node`: confirm `ros2 topic echo /zed/zed_node/odom` is
   publishing, then send a short single `STRAIGHT` segment on a stand and
   check the wheels turn the right way for the whole segment (not just at the
   start) before ever sending a `TURN` segment or running on the ground.
