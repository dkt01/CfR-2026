# Jetson Onboard Software

ROS 2 (Jazzy) packages that run on the NVIDIA Jetson Orin Nano Super and drive the
Traxxas Slash 4X4 through the Arduino over the USB serial link described in the
[Onboard Protocol](../README.md) section of the top level README.

| Package | Contents |
| ------- | -------- |
| [`cfr_interfaces`](cfr_interfaces/) | `DriveCommand` and `ArduinoStatus` message definitions |
| [`cfr_arduino_bridge`](cfr_arduino_bridge/) | `arduino_bridge_node` and `cmd_vel_to_drive_node` |

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

Arduino -> Jetson matches the `ToJetson` struct: E-Stop, Auto Arm, Manual Start,
Mode `[0,4]`, Battery Level `[0,255]`, with a trailing comma before the newline.

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

`test_protocol` covers the wire format and `test_serial_port` runs the port
against a pseudo terminal, so both pass with no Arduino attached.

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
