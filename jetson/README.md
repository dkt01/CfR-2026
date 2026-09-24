# Jetson Onboard Software

ROS 2 (Jazzy) packages that run on the NVIDIA Jetson Orin Nano Super and drive the
Traxxas Slash 4X4 through the Arduino over the USB serial link described in the
[Onboard Protocol](../README.md) section of the top level README.

| Package | Contents |
| ------- | -------- |
| [`cfr_interfaces`](cfr_interfaces/) | `DriveCommand`, `ArduinoStatus`, `PathSegment`, `StartSignal`, `LapCount` messages; `DrivePath` action |
| [`cfr_arduino_bridge`](cfr_arduino_bridge/) | `arduino_bridge_node`, `cmd_vel_to_drive_node`, `path_follower_node`, `start_signal_detector_node`, `lap_counter_node`, `obstacle_randomizer_node` |

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
  the DXF -- and it reproduces the drawing's annotated 4 ft centerline
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

**Buckets.** Two to nine, at least 3 ft between centers and 2.5 ft off the
bale walls, both measured off the drawing. Those two numbers are also why the
drawing's "placed so a path exists around and between buckets" needs no
reachability check: 3 ft between centers leaves a 0.62 m gap between two
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
common pivot 32 in up; whichever is horizontal stands out past the oasis blue
board and is the one the car sees. The arms are painted Rust-Oleum 2X Satin
Poppy Red and Leafy Green, and the board Rust-Oleum Satin Oasis Blue. The arms
ride a revolute joint whose limits
are those two positions, and it rests at the red one, so **a freshly loaded
world always shows red** without anything having to command it.

The service **turns** the arm rather than snapping it: 90 degrees in one
second, matching the signal on the course and giving a detector the part-way
arm it will have to cope with. The randomizer ramps the joint setpoint at
25 Hz over `/start_signal/arm`, which `simulation.launch.py` bridges to
Gazebo's joint-position controller; the controller follows the ramp, so the
rate is what the node says it is. `signal_sweep_rate` (degrees per second,
default 90) changes it, and 0 commands the far end directly for a scripted
test that does not want to wait.

Through the camera, one sweep looks like this -- the counts
`start_signal_detector_node` reports, from
[`scripts/check_start_signal.py`](scripts/check_start_signal.py) against the
obstacle course -- the speed course reads the same, since both stand the
signal 8 ft down a 32 in lane and its range differs by a centimeter:

```
   t(s)   red px   green px   reads
   0.00     249         0     red
   0.33     243        25     red
   0.59     226       155     red
   0.66     208       185     red
   0.73     182       208     green   <- the arms cross over
   0.79     137       230     green   <- ~/go latches here
   1.06       5       246     green
```

The two arms are 90 degrees apart on one pivot, so through the turn they
trade projected area rather than both disappearing: the total stays near 250
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

The drawing places the signal, and places it the same way on both courses:
**three bales down the wall from the start line** -- annotated "approximately
8 ft" -- **with the bales moved so it stands in line with the inner edge of the
bale border**. So neither generator chooses a spot any more;
`start_signal.position()` derives one from each course's start line, heading
and lane edge, and the two come out within a centimeter of each other from the
driver's seat:

| Course | Position | Yaw | Down the lane | Bearing | Range | Clears the wall by |
| ------ | -------- | --- | ------------- | ------- | ----- | ------------------ |
| Obstacle | (2.44, 0.81) | -90 deg | 8.0 ft | 16.1 deg | 2.94 m | 152 mm |
| Speed | (17.01, 3.89) | +90 deg | 8.0 ft | 17.2 deg | 2.95 m | 172 mm |

The board is 32 in wide and its arms sweep in its plane, so it stands **square
across the lane**, facing back up it: width across the lane, 4 in of depth
along it. Square, and not aimed at the point the car waits at -- the board
sits off to the side of a 32 in lane, so a car on the centerline is a good
16 degrees off the perpendicular, and aiming at it would cant the board, its
arms and its footprint by that much. The sign on the course stands square to
the path, and an 8 ft signal reads the same either way.

Centered on the lane edge, half the board would be in the path, so it stands
half a width outboard: inner end flush with the edge -- to within 2 mm, all of
it the wall's own placement -- and body where the wall was. That is what "the
bales moved" means, and `start_signal.clear_bales()` does it, sliding the one
bale the board displaces along its own wall until it clears (0.17 m on the
obstacle course, 0.25 m on the speed course) and leaving the board's 4 in of
depth standing in the gap.

Bearing and range are to the middle of the board; the arm itself hangs nearer
the lane, about 11 degrees off the axis, where it is about 21 x 16 px of
saturated red or green in a 640 x 360 frame. Moving in from the old 4 m spot
outside the wall also bought a much better sight line: the ray from the 8 in
camera used to pass 24 mm over the 14 in bale wall, and now clears it by
150 mm and more.

[`scripts/check_signal_sightline.py`](scripts/check_signal_sightline.py)
re-derives all of it from the generated worlds -- the distance down the lane,
the board's alignment with the border, that it stands square to the path, that
no bale is left inside the board, the camera's field of view and the sight
line -- and fails if a nudged constant breaks one:

```bash
./scripts/check_signal_sightline.py
```

```
obstacle_course.sdf      ok    8.0 ft down the lane, board -2 mm off the border's edge, -0.0 deg off square, bearing +16.1 deg, range 2.94 m, clears bales by 152 mm
speed_course.sdf         ok    8.0 ft down the lane, board +2 mm off the border's edge, +0.1 deg off square, bearing +17.2 deg, range 2.95 m, clears bales by 172 mm
```

Bounds for everything the randomizer moves come from the layout file beside
each world --
[`obstacle_course_layout.yaml`](cfr_arduino_bridge/config/obstacle_course_layout.yaml)
and
[`speed_course_layout.yaml`](cfr_arduino_bridge/config/speed_course_layout.yaml)
-- which the generators write, so the randomizer and the world cannot drift
apart.

Two things the drawing calls variable are **not** randomized: the pothole
bumps, which are placed as drawn because their matching holes are cut into
the board mesh and cannot move with them, and the bucket section's entrance,
which would mean moving bale walls.

### Cameras

`simulation.launch.py` mounts an RGB-D camera at the front of the simulated
Slash and publishes it under ZED-compatible names:

| Topic | Type | Source |
| ----- | ---- | ------ |
| `/zed/zed_node/odom` | `nav_msgs/Odometry` | Gazebo vehicle odometry |
| `/zed/zed_node/pose` | `geometry_msgs/PoseStamped` | Gazebo ground truth, standing in for the ZED's map frame pose |
| `/zed/zed_node/left/image_rect_color` | `sensor_msgs/Image` | simulated left color camera |
| `/zed/zed_node/left/image_rect_color/camera_info` | `sensor_msgs/CameraInfo` | simulated camera calibration |
| `/zed/zed_node/depth/depth_registered` | `sensor_msgs/Image` | simulated depth camera |
| `/zed/zed_node/point_cloud/cloud_registered` | `sensor_msgs/PointCloud2` | simulated registered depth point cloud |

$110^\circ$ horizontal field of view, $640 \times 360$, 15 Hz, 0.2 m to 20 m.
Color and depth come from one `rgbd_camera` sensor, so they share one
calibration and there is one `camera_info` rather than two. The published
point cloud passes through `zed_cloud_noise_node.py`: forward range gets
Gaussian noise with sigma `0.01 + 0.008 * range²` metres, with 3% independent
invalid pixels. Its XYZ values stay on the original camera rays, and color is
preserved. This is a stereo-like approximation, not a calibration from real
ZED point clouds; the bridged depth image remains Gazebo's ideal depth.

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

### Point cloud segmentation

`src/cloud_segmentation.py` classifies every point of the ZED cloud as
ground, obstacle, hoop, car wash or overhead. Ground includes the ramps, deck,
helix, bank, pothole board and gravel; hoops and the car wash are things to
drive through; overhead covers the tunnel roof and anything else above the
car with open floor seen under it. It reads only the cloud and the camera's
pitch and roll, so the same code runs on the car. `scan_from_segmentation`
reduces the result to the same bearing-binned scan
`rl/bale_follower/cloud_scan.py` produces, with only blocking points in it: a
gate's posts block, its bar and ribbons do not.

With `sensors:=true`, `cloud_segmentation_node.py` republishes the cloud
colored by class on `/zed/segmented/points`, bridged to Gazebo transport.
The browser viewer's **Color by class** button switches the robot view to it.

`test/test_cloud_segmentation.py` scores the segmenter against 62 rendered
views of both courses (`test/segmentation_scenarios.py` lists them). Each
fixture is a depth image plus a per-pixel label saying which part of the
course the pixel shows. A Gazebo segmentation camera rendered beside the ZED
produces the labels. The camera is on a static rig, not the car, so a view
on the 19% ramp does not roll away while it renders. The rig's pitch and roll
come from dropping the car's four wheels onto the course's surfaces, which is
how views like "one tire in a pothole recess" are posed. To re-render after a
course change, run this with the repository mounted read-write (about ten
minutes on llvmpipe):

```bash
docker run --rm -v "$PWD:/repo" -w /repo unfrobotics/docker-ros2-jazzy-gz-rviz2:latest \
    bash -c 'source /opt/ros/jazzy/setup.bash && LIBGL_ALWAYS_SOFTWARE=1 \
        python3 jetson/scripts/capture_segmentation_fixtures.py --preview /tmp/preview'
```

`python3 test/test_cloud_segmentation.py --report` prints the per-view table
the thresholds were set against. The views the segmenter still gets wrong are
listed in `KNOWN_LIMITATIONS` in the test as strict expected failures, each
with its reason. They are all the car wash, seen from inside it or from 2 m
or more out.

### Regenerating the course

Neither step is needed to run the simulation -- the worlds and meshes are
committed. Both are needed when the drawing or the CAD changes.

```bash
# Obstacle course world and the randomizer's bounds, from the site-layout DXF.
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

Behavior:

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

* `STRAIGHT` -- drive `distance` meters, holding the heading measured at the
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
straight; feet converted to meters):

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
to the action's meters/radians internally. Segments are kept after a run
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
shows, how many pixels of each color, and where. A driver that must stop on a
red flag mid-run watches that rather than `~/go`, because `~/go` deliberately
stays up once a run has started: a single mis-hued frame must not be able to
retract a start that has already happened.

`armed` in that message is the field to watch while the car sits at the line.
The detector finds the signal *first* -- a place in the image that holds red
long enough to be it -- and only then waits for that place to turn green.
Until it is armed no amount of green will start the run, so `armed` false at
the line is the thing to notice before the flag drops rather than after.

#### Outdoors

The course is outside, the run may be at any hour, and there will be people
about in shirts of every color. A start is a red arm that *becomes* green
**in one place**, and at the 8 ft both courses stand the signal at that arm is
only about 21 x 16 px of a 640 x 360 frame, so none of the following is
optional:

* **Above the horizon only.** The arm is 32 in up and the camera 8 in up, so
  the arm is above the camera's horizon from anywhere on the course -- and for
  a level camera that horizon is the middle row of the image. `region`
  searches the top half, which is also all the ground excluded. Narrowing it
  around where the signal actually lands is the cheapest way to make a busy
  course easier.
* **Chroma, not brightness.** Exposure scales all three channels, so it leaves
  hue and saturation alone and takes chroma; glare behind the signal and
  clipping in the sun *add* to all three, so they leave hue and chroma alone
  and take saturation. So `min_chroma` -- how far from gray a pixel is -- is
  the floor that does the work, and `min_saturation` is only there to reject
  gray. Measured, the arm is still read at a fifth of the chroma it renders
  with and at a fifth of its saturation; see the lighting check below.
* **Hue bands drawn for a field, not a renderer.** Red stops at 16 degrees,
  leaving 4 degrees to skin at 20 to 35, straw and dry grass at 30 to 50, and
  the orange of cones and barrels; it reaches back to 338 instead, because
  open shade is lit blue and takes red towards magenta. Poppy Red is warm
  enough to drift past a tighter band under direct-sun clipping, which is why
  16 rather than 12. Green starts at 110, right at the edge of turf and
  foliage at 80 to 110, and stops short of the board's oasis blue at 198 and
  the car wash's blue ribbons at 212. The arms render at 5 and 114 -- Leafy
  Green sits close enough to real foliage that size and place, not hue, are
  what keep foliage out here.
* **Arm-sized, not merely the right color.** The count applies to the densest
  `cluster_window` box, and a candidate whose color keeps going outside that
  box is thrown out on its size: `max_spread`. A shirt on somebody 5 m away
  scores 3.8 against an arm's 1.0, so hue cannot save it -- which is what
  keeps a red shirt, a tent, a hedge or a hillside out.
* **Several candidates, so nothing can hide the signal.** A red shirt is
  bigger than the arm, so reporting only the densest cluster would report the
  person and never see the signal behind them. The best few places of each
  color are searched, and a blob too big to be an arm is skipped without
  using one of the slots.
* **Places, not colors.** The latch keeps a site for each place where
  arm-sized signal color keeps turning up and counts red and green frames
  per site, so a start is one site going from red to green within
  `max_transition_distance` -- both arms turn about one pivot, so a real
  transition happens in one place. A marshal in a red shirt has their own
  site and can do nothing from it but stand there; the car wash's twenty red
  ribbons likewise. A site nothing has been seen at for `forget_frames` is
  dropped, which is also what stops a red object removed from a spot pairing
  with a green one put there later.
* **Weaker evidence once it is found.** The best site is fed back as a box to
  look harder inside, where the color floors relax by `focus_relaxation`.
  That is what reads a backlit arm after the sun has come round behind it
  mid-wait. The whole region is still searched at full strength as well, so
  the prior can only add candidates, never hide them.

Then `arm_frames` frames of red to find it and `confirm_frames` of green to
call it. The asymmetry is deliberate: the car stands at the line for as long
as it takes, so 5 frames of red costs nothing, while the green has to be
caught inside the second the arm takes to turn. Measured against both
courses, `~/go` latches one frame after the arms cross over: 0.79 s into the
1 s sweep on 228 px of green, or 0.86 s when the software renderer drops a
frame.

Tuning is live: `ros2 param set` on any threshold rebuilds the classifier
without disturbing the latch, and a value that does not make sense is refused
with a reason rather than quietly clamped.

```bash
ros2 launch cfr_arduino_bridge start_signal.launch.py debug:=true
ros2 param set /start_signal_detector min_saturation 0.25
ros2 run rqt_image_view rqt_image_view /start_signal_detector/debug_image
```

`debug:=true` publishes each frame with the region, the winning cluster and
the box the detector is watching drawn on it, which is how the bands get moved
to fit the real signal: if that white box is not on the signal, nothing else
in the frame matters. The
defaults live in
[`config/arduino_bridge.yaml`](cfr_arduino_bridge/config/arduino_bridge.yaml)
with a note on each.

The decision itself is in
[`start_signal_detector.py`](cfr_arduino_bridge/src/start_signal_detector.py),
free of ROS like `path_geometry` on the C++ side, and covered by
`test_start_signal_detector` against synthetic frames built from the worlds'
own colors. Nothing in a synthetic frame can show that Gazebo renders those
colors where the geometry says it will, so
[`scripts/check_start_signal.py`](scripts/check_start_signal.py) drives a
running simulation -- red, turn it green, wait for the trigger -- and prints
the frames it took:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch cfr_arduino_bridge speed_course.launch.py sensors:=true
./scripts/check_start_signal.py            # exits non-zero if a start is missed
```

Nor can either of those show what the light will do to it, so
[`scripts/check_signal_lighting.py`](scripts/check_signal_lighting.py) takes
one real frame with the signal red and one with it green and replays them
through the decision under light they were not taken in. The cases are
derived from the arm's own measured color rather than picked -- exposure
solved for the chroma it would leave, glare solved for the saturation it would
leave -- so they mean the same thing against a dim rendering as against a
signal in daylight. Then it puts people in frame and checks both that they do
not stop a start and that they cannot cause one:

```bash
./scripts/check_signal_lighting.py                  # against a running sim
./scripts/check_signal_lighting.py --spin-by-hand   # on the car, at the course
```

Against the obstacle course, where the arm renders at a chroma of 0.27 and a
value of 0.36:

```
   case                          expect  result
   chroma 0.40 (as rendered)      start   start
   chroma 0.20                    start   start
   chroma 0.10                    start   start
   chroma 0.05                    start   start     <- a fifth of the light
   chroma 0.02                       --      no
   glare to saturation 0.50       start   start
   glare to saturation 0.30       start   start
   glare to saturation 0.20       start   start
   glare to saturation 0.15          --   start     <- all but grayed out
   glare to saturation 0.10          --      no
   people in frame                start   start
   a shirt changing color            no      no
   green from the start              no      no
   the signal never turning          no      no
```

The rows with no expectation are past what the detector claims and are
measured for the record: knowing the cliff is at a chroma of 0.02 and a
saturation of 0.10 is what says how much room a threshold has before it
matters. Past that the frame no longer holds the answer, and the fix is a
lens hood or an exposure setting rather than a band.

Both course launches start the detector themselves with `sensors:=true`; on
the car it comes up with `start_signal.launch.py` alongside the ZED. It needs
no GPU: llvmpipe renders the camera at about 5 Hz, which still puts three or
four frames inside the turn.

### `lap_counter_node`

Counts crossings of the start/finish line and latches `~/done` once the course
has been run: three laps of the speed course, two of the obstacle course. A
driver subscribes to `~/done` and stops the car; nothing does yet.

| Interface | Type | Notes |
| --------- | ---- | ----- |
| `pose` | `geometry_msgs/PoseStamped` | subscribed (remapped to `/zed/zed_node/pose`) |
| `status` | `cfr_interfaces/ArduinoStatus` | subscribed (remapped to `/arduino_bridge/status`) |
| `go` | `std_msgs/Bool` | subscribed (remapped to `/start_signal_detector/go`), transient local |
| `~/count` | `cfr_interfaces/LapCount` | published per pose sample |
| `~/done` | `std_msgs/Bool` | published latched, transient local, on change |
| `~/reset` | `std_srvs/Trigger` | service, re-arm for another run |

The pose is the ZED's **map** frame topic, not `~/odom`. The SDK applies loop
closure to that one and deliberately never to odometry, and three laps of the
speed course is about 300 m of travel returning to the same spot, which raw
dead reckoning will not hold. `zed/config/cfr_zed2i.yaml` pins `area_memory`
on rather than leaving it to whatever the installed wrapper defaults to;
confirm on the car with

```bash
ros2 param get /zed/zed_node pos_tracking.area_memory
```

Note that the same setting makes `~/odom` jump as well -- the wrapper's
`reset_odom_with_loop_closure` defaults to true -- so nothing should treat
that topic as continuous.

There is no map of the course and the line is not published anywhere at run
time, but both courses park the car 0.70 m behind it, on the lane centerline,
pointed down the lane. So the counter latches the pose the car held at the
start and works relative to that: the line is the plane `line_offset` ahead.
The car crosses it on the way out, which arms the counter rather than scoring
-- three laps is four crossings in all.

What stops the oval's far side counting is **heading**, not distance. The
return leg passes through the plane of the line too, 14 m out and travelling
the opposite way; at the line the car travels the way the run started. That
holds for any start/finish line on any closed course, where "within a few
metres of where we started" is a claim about how wide this particular course
is -- and the course built on the day will not match the drawing. So the
counter holds no model of the course's shape at all. `lateral_gate` is
available as a backstop and is off by default.

Counting is suspended unless the Arduino reports `AUTO_ACTIVE` with no e-stop.
The rules allow an e-stop to lift the car past an obstacle or off the course,
and when counting resumes the motion baseline is re-seeded, so the
displacement cannot read as driving. If the car was set down more than
`carry_tolerance` from where it stopped, the distance it had driven is thrown
away too -- otherwise a car lifted back behind the line would score on the way
over it using travel banked before the stop, a lap it never completed. An
e-stop that does not move the car keeps its lap, so a pause costs nothing.

A loop closure is the opposite case and is handled differently. It shows up as
a pose step no ground vehicle could drive, and is reported and kept out of the
distance travelled, but it re-seeds nothing -- the correction moves the
estimate towards truth, and the latched reference is in the same corrected
frame. A carry is the car really moving; a closure is an estimate improving.

Every pass through the line is logged, counted or not, with the gate that
rejected it:

```
lap 2 counted   heading +0.0 deg  lateral +0.00 m  travelled 114.0 m
crossing rejected (heading)   heading +178.4 deg  lateral -13.9 m  travelled 48.1 m
carried 40.03 m while stopped; lap distance restarted
loop closure   jump 1.50 m  at s=0.8 d=0.0
```

That is how the gates get tuned against the real course rather than the
idealized one. Every gate is a reason to reject, so a gate set too tight means
a missed lap and a car that keeps driving, never one that stops early --
watch `rejected` on `~/count` during practice runs.

On the car, alongside the bridge and the ZED:

```bash
ros2 launch cfr_arduino_bridge lap_counter.launch.py laps:=3
ros2 launch cfr_arduino_bridge lap_counter.launch.py free_run:=true
```

`free_run:=true` arms on the first pose instead of the start signal and counts
without waiting for `AUTO_ACTIVE`, which is what makes the counter usable from
`path_tui.py` with nothing else running. Both course launches set the right
lap target themselves.

`scripts/check_lap_counter.py` drives a synthetic run -- three laps, an e-stop
and a carry over the line, and a loop-closure jump -- at a running node and
checks what it reports. It needs no simulator and no car, and covers the
wiring the unit tests cannot:

```bash
ros2 run cfr_arduino_bridge lap_counter_node.py --ros-args     -r pose:=/check/pose -r status:=/check/status -r go:=/check/go
python3 scripts/check_lap_counter.py
```

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
are `arduino_bridge` parameters that take effect when changed at runtime:

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

Three more bridge parameters are runtime-settable, for characterization runs
that sweep them between steps and cannot afford a restart (a restart costs the
Arduino handshake and, mid-run, the run directory):

| Parameter | Units | Meaning |
| --------- | ----- | ------- |
| `speed_slew_rate` | m/s per second | `0` disables rate limiting. Raise it for step-response work: the production `2.0` takes 1.6 s to reach 3.2 m/s and would swamp a plant time constant near 0.5 s |
| `spur_to_wheel_ratio` | - | must stay positive |
| `tire_diameter` | m | must stay positive |

The drivetrain pair is part of the gain conversion - the firmware works per 1000
spur RPM - so changing either revalidates the gains against the new scaling and
resends them. Everything else is launch-time only on purpose: the device, the
safety clamps and the arming policy should not move under a car that is already
armed.

"us" is microseconds of throttle pulse. The bridge converts the rate gains into
the firmware's per-1000-spur-RPM units, sends them tagged with a new sequence
number, and resends until the Arduino echoes that number; `gains_applied` on
`ArduinoStatus` is false in between. `ros2 param set` rejects a value the
firmware would not accept.

The defaults were tuned **with the car on blocks** (see the bench notes under
[Speed Control](../README.md#speed-control)). Expect to retune on the ground:
the car adds several times the inertia plus rolling drag, so the feedforward
terms will need to rise and the loop will respond more slowly than on blocks.

The ground retune is now a characterization profile rather than a manual
procedure - see [Characterization](#characterization). `tune_profile` drives the
same speed sequence these gains were scored against on blocks, and
`analyze_run.py` prints a row in the same format as the table below, so the
ground numbers sit directly beside the bench ones. Sweep gains across runs:

```bash
ros2 launch cfr_arduino_bridge characterize.launch.py \
    profile:=tune_profile gains:="speed_kp=24.0 speed_ki=10.0" label:=kp24ki10
```

The manual procedure, still valid if you would rather drive it by hand, with the
E-Stop in hand and plenty of room:

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

## Characterization

[`docs/characterization.md`](../docs/characterization.md) is the procedure for
measuring the car so the Gazebo simulation can be a twin of it, and
[`docs/field-card.md`](../docs/field-card.md) is the printable one-page version
to take to the test site.

Everything about it is built for testing away from a network: the car records
locally because Wi-Fi drops out at range, the analysis needs nothing but a stock
Python 3, and **no step requires reflashing the Arduino** - the whole campaign
runs through the runtime `speed_*` parameters and the firmware's existing `D,`
debug line.

```bash
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=coastdown
```

One command, one profile name. The launch picks a run directory under
`~/cfr_runs`, points the Arduino serial traces into it, and starts
`maneuver_runner_node.py`, which **waits for E-Stop to be asserted and then
cleared** before it moves the car. That sequence can only be completed by
someone holding a working, connected E-Stop, which is the precondition worth
enforcing before a car drives itself. Re-asserting E-Stop aborts the run.

Straight-line profiles reverse themselves back to the start under odometry, so
nobody walks the length of a bike path after every run - and averaging the two
directions cancels the path's grade into the bargain.

| Profile | Session | Measures |
| ------- | ------- | -------- |
| `zed_static` | parking lot | ZED odometry noise and drift; never arms, the car cannot move |
| `steer_authority` | parking lot | effective steering angle per command, and minimum turn radius |
| `skidpad` | parking lot | understeer gradient, plus a lower bound on lateral grip |
| `step_steer` | parking lot | yaw response, validating the inertia estimate |
| `pulse_staircase` | 60 m straight | open-loop throttle pulse to ground speed |
| `coastdown` | 60 m straight | rolling, viscous and aero resistance |
| `brake_sweep` | 60 m straight | braking authority against `speed_brake_limit` |
| `tune_profile` | 60 m straight | closed-loop scoring, matching the on-blocks table |

Rehearse the procedure against Gazebo before the car ever sees it:
`use_sim:=true require_estop_cycle:=false` swaps in `sim_vehicle_node` (bringing
up its own world), exercising arming, gains handshake, limits and output end to
end - but with no E-Stop interlock and no plant to characterize. See
[Dry run in simulation](../docs/characterization.md#dry-run-in-simulation).

Afterwards:

```bash
./scripts/sync_runs.sh                        # pull runs off the Orin
./scripts/analyze_run.py runs/<run> --mass 4.7
./scripts/apply_vehicle_patch.py runs/<run>   # fold results into vehicle.yaml
./scripts/generate_vehicle_model.py           # push them into the Gazebo world
```

### `config/vehicle.yaml`

The single source of truth for vehicle geometry, mass, actuator and sensor
behaviour. Before it existed the wheelbase was a bare literal in five places,
the steering limit in five more, and the Gazebo wheel radius disagreed with the
drivetrain tire diameter by 3.6% without anything noticing.

Every entry carries a **provenance** tag - `measured`, `estimated` or `guess` -
so an unmeasured number is visible rather than implied. `apply_vehicle_patch.py`
stamps each value it updates with the run that justifies it and the date; a
number in that file should always be able to answer "says who?".

`scripts/generate_vehicle_model.py` writes the `<model name="slash">` block of
`worlds/speed_course.sdf` **and** `worlds/obstacle_course.sdf` from it (each
world keeps its own spawn point; everything else about the car is shared), and
`--check` runs in the test suite, so a `vehicle.yaml` edit that nobody
regenerated fails the build instead of quietly leaving the simulator
describing a different car.

The vehicle includes a suspension DOF per corner (a damped, sprung prismatic
joint between the chassis and each wheel's upright) so the obstacle course's
potholes, gravel and ramps actually excite the chassis instead of being
transmitted straight through rigidly-mounted wheels. The rates describe the
stock Slash 4X4 Ultimate hardware - GTR long shocks on #7444 springs at the
front, XX-long on the stiffer #7446 at the rear, all four collars at maximum
preload - derived from the car's own measured mass and ride height rather than
from a catalog rate Traxxas does not publish. They are tagged `estimated`;
characterization's A8 measures them directly, and `vehicle.yaml` writes out
every assumption A8 has to check. See
[docs/characterization.md](../docs/characterization.md).

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
`start_signal_detector_node`'s color decision against synthetic frames -- all
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

### Physical speed-course wall follower

On the Orin, after building this package and `zed_wrapper`, run the full
hardware stack from one launch file:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cfr_arduino_bridge left_wall_robot.launch.py device:=/dev/ttyACM0
```

This starts the Arduino bridge, `cmd_vel_to_drive_node`, ZED 2i with the
loop-closure settings, start-signal detector, three-lap counter, and one wall
follower. The follower uses the ZED registered point cloud rather than the
simulator's bale map. It waits for the visual start, sends zero velocity if
the cloud becomes stale, and stops when the counter completes three laps.
The initial cloud controller is capped at 0.6 m/s and needs a controlled
on-robot trial before relying on it for a full course run.

The ZED color and cloud topics can be overridden with `image_topic:=` and
`cloud_topic:=`. The usual Arduino auto-active and E-Stop interlocks still
apply. Use this launch by itself; `scripts/launch.sh` would start a second
bridge and camera.


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

For a three-lap autonomous rehearsal with the simple left-wall guide:

```bash
ros2 launch cfr_arduino_bridge left_wall_speed_course.launch.py
```

This launches the rendered camera, waits until the start-signal detector has
seen red, turns the simulated signal green, and drives only after the detector
publishes `/start_signal_detector/go`. It stops when `/lap_counter/done` is
latched after three laps. The guide follows the bale-derived speed-course
centerline and applies a small left-wall distance correction; it uses the
simulator's map pose and course geometry, so it is intended for simulation.
Software rendering is enabled for headless runs and can be slow.

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
