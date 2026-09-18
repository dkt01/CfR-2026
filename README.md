# CfR-2026
The Command for Racing team repository for the 2026 Cat Technology DIY Robot Challenge

## Commissioning

* XBee setup and pairing the two modules is covered in this [SparkFun Guide](https://learn.sparkfun.com/tutorials/xbee-shield-hookup-guide)

## Run Modes

The robot may be in one of the following states:

| Mode        | Description                                                                                                                |
| ----------- | -------------------------------------------------------------------------------------------------------------------------- |
| E-Stop      | E-Stop active, no RC or Auto                                                                                               |
| RC Armed    | RC requested, but conditions not met to begin RC.  All joysticks must move to neutral position before RC control is active |
| RC Active   | RC control                                                                                                                 |
| Auto Armed  | Auto mode active awaiting start signal                                                                                     |
| Auto Active | Autonomous running                                                                                                         |
| Fault       | Cannot run                                                                                                                 |

## RC & E-Stop Protocol

The robot features an offboard E-Stop interface with remote control.  The offboard interface is run on a Raspberry Pi and the onboard system runs on an Arduino Uno.  Both communicate wirelessly using an [XBee Pro S1](https://cdn-shop.adafruit.com/datasheets/Xbee%20series%201%20DS.pdf) radio pair.  The Offboard system is designed to interface with the course E-Stop system through an RJ-45 connector.  RC is controlled using a PS3 controller wired to the offboard Raspberry Pi.

Messages transmitted each direction are packed binary carried as the payload of an XBee API frame.  Fields are fixed offset and fixed width; 16-bit fields are little endian.  The API frame supplies its own length and checksum, so the payload carries no framing or integrity fields of its own — only a format tag, so that a version-skewed firmware/offboard pair fails closed instead of misreading a payload.  A receiver rejects any frame whose tag, length, or reserved bits do not match.

This link was originally ASCII comma-separated.  It was packed because of the Arduino's interrupt budget rather than for bandwidth: `SoftwareSerial` disables interrupts for a full byte time (~174 µs at 57600 baud) and works through a frame's bytes back to back, so the ~50 byte ASCII command frame suppressed interrupts for nearly 9 ms in one stretch, twenty times a second.  The hardware USART's receive buffer is two bytes deep and a byte arrives every 10 µs at 1 Mbaud, so every command frame overran the USB link and corrupted whichever Jetson command was in flight.  Eight bytes cuts that window by more than six times.

### Onboard -> Offboard

Tag `0x51`, 11 bytes.

| Byte  | Description     | Data Type     | Data Range  | Notes                                                                    |
| ----- | --------------- | ------------- | ----------- | ------------------------------------------------------------------------ |
| 0     | Format Tag      | Integer       | `0x51`      | Rejected if it does not match.                                           |
| 1     | Auto Mode       | Enum          | [0,4]       | `0` E-Stop, `1` RC Armed, `2` RC Active, `3` Auto Armed, `4` Auto Active |
| 2     | Battery Level   | Integer       | [0,255]     | `0` is empty battery, `255` is full battery.                             |
| 3-4   | Battery Voltage | Integer (LE)  | [0,65535]   | Pack millivolts as measured onboard.  Sent alongside the scaled level because the level cannot be inverted once its endpoints move. |
| 5-6   | Steering Output | Integer (LE)  | [1000,2000] | Final steering PWM pulse width in microseconds.                          |
| 7-8   | Throttle Output | Integer (LE)  | [1000,2000] | Final throttle PWM pulse width in microseconds.                          |
| 9-10  | Spur RPM        | Integer (LE)  | [0,65535]   | Spur gear revolutions per minute.  `0` also means stopped or no sensor.   |

### Offboard -> Onboard

Tag `0xC1`, 8 bytes.  Booleans are packed into three flag bytes, bit 0 first.

| Byte  | Description | Data Type | Data Range | Notes                        |
| ----- | ----------- | --------- | ---------- | ---------------------------- |
| 0     | Format Tag  | Integer   | `0xC1`     | Rejected if it does not match. |
| 1     | Flags A     | Bitfield  |            | See below.                   |
| 2     | Flags B     | Bitfield  |            | See below.                   |
| 3     | Flags C     | Bitfield  |            | See below.  Bits 5-7 reserved, must be zero. |
| 4     | RC - Left Joystick X  | Integer | [0,255] | `0` is full left, `255` is full right |
| 5     | RC - Left Joystick Y  | Integer | [0,255] | `0` is full up, `255` is full down    |
| 6     | RC - Right Joystick X | Integer | [0,255] | `0` is full left, `255` is full right |
| 7     | RC - Right Joystick Y | Integer | [0,255] | `0` is full up, `255` is full down    |

Every bit below is `1` for active/pressed.

| Bit | Flags A (byte 1)      | Flags B (byte 2) | Flags C (byte 3) |
| --- | --------------------- | ---------------- | ---------------- |
| 0   | E-Stop State          | RC - L1          | RC - PS          |
| 1   | Auto Arm              | RC - R1          | RC - DPad Up     |
| 2   | Manual Start          | RC - L2          | RC - DPad Right  |
| 3   | RC Controller Present | RC - R2          | RC - DPad Down   |
| 4   | RC - X                | RC - L3          | RC - DPad Left   |
| 5   | RC - O                | RC - R3          | *reserved*       |
| 6   | RC - Square           | RC - Select      | *reserved*       |
| 7   | RC - Triangle         | RC - Start       | *reserved*       |

E-Stop State `1` indicates E-Stop active.  Auto Arm `1` indicates autonomous mode active, `0` indicates RC only.  Manual Start `1` indicates the robot should start autonomous driving without a visual start signal.  RC Controller Present `1` indicates a gamepad is connected.

The bit assignments are defined in `CMD_FLAGS_A`/`B`/`C` in `src/rpi_rc_estop/estop_core.py` and in `FromOffboard::deSerialize` in `src/arduino_rcm/arduino_rcm.ino`.  Changing one without the other is the failure this table exists to prevent.

## Onboard Protocol

The robot features an Arduino that controls the car's actuators and an NVIDIA Jetson Orin Nano Super.  The two components communicate over a USB serial interface at 115200 Baud.

Messages transmitted each direction are ASCII serial where fields are comma-separated and messages are separated by a new line character (`\n`).

The rate is deliberately modest.  `SoftwareSerial` holds interrupts off for about 174 µs per XBee byte, while at 1000000 Baud a Jetson byte arrives every 10 µs, so each blackout could overrun the Uno's two byte receive buffer: on the bench 14% of Jetson frames arrived with bytes missing.  At 115200 Baud a byte takes 868 µs and the loss fell below 1%.  Both Jetson frames are also fixed width and the Arduino requires the exact length, so a byte that does go missing rejects the frame instead of changing what it says.

### Arduino -> Jetson

Every field is followed by a comma, including the last.

| Field Index | Description     | Data Type      | Data Range     | Notes                                                                           |
| ----------- | --------------- | -------------- | -------------- | ------------------------------------------------------------------------------- |
| 0           | E-Stop State    | Boolean        | `0` or `1`     | `1` indicates E-Stop active                                                     |
| 1           | Auto Arm        | Boolean        | `0` or `1`     | `1` indicates autonomous mode active.  `0` indicates RC only                    |
| 2           | Manual Start    | Boolean        | `0` or `1`     | `1` indicates robot should start autonomous driving without visual start signal |
| 3           | Auto Mode       | Enum           | [0,4]          | `0` E-Stop, `1` RC Armed, `2` RC Active, `3` Auto Armed, `4` Auto Active        |
| 4           | Battery Level   | Integer        | [0,255]        | `0` is empty battery, `255` is full battery.                                    |
| 5           | Spur RPM        | Signed Integer | [-20000,20000] | Spur gear revolutions per minute.  The sensor cannot see direction, so the sign is the speed controller's estimate: the direction it last drove, positive when unknown.  `0` also means stopped or no sensor. |
| 6           | Target RPM      | Signed Integer | [-20000,20000] | Spur RPM the speed controller is tracking.  Held at `0` while it stops the car ahead of a reversal. |
| 7           | Throttle Output | Integer        | [1000,2000]    | Requested throttle pulse in microseconds, before low speed dithering.           |
| 8           | Gains Sequence  | Integer        | [0,255]        | Sequence number of the speed gains in use.  `0` is the firmware's compiled-in defaults. |

`Deserialize()` on the Jetson requires exactly nine fields, so firmware predating closed loop speed control is rejected outright.  Flash both sides together.

The same link carries diagnostic lines that the Jetson skips: a `D,` debug line at 20 Hz, and a `T,` speed controller trace every 20 ms while the trace flag of the gains frame is set.  Their fields are listed beside the `snprintf` calls that produce them in `src/arduino_rcm/arduino_rcm.ino`.

### Jetson -> Arduino

#### Drive Command

Sent at 50 Hz, always exactly `C,b,sss,+rrrrr`.  Only this frame feeds the Arduino's 200 ms watchdog.

| Field Index | Description    | Data Type         | Data Range      | Notes                                                           |
| ----------- | -------------- | ----------------- | --------------- | --------------------------------------------------------------- |
| 0           | Frame Type     | Character         | `C`             |                                                                 |
| 1           | Auto Ready     | Boolean           | `0` or `1`      | `1` indicates auto control requested                            |
| 2           | Steering Angle | Integer, 3 digits | [0,255]         | `0` is full right, `255` is full left                           |
| 3           | Target RPM     | Sign and 5 digits | [-20000,+20000] | Spur RPM for the speed controller, positive forward.  `+00000` stops. |

The Arduino enters Auto Active from Auto Armed only while Auto Ready is `1`, steering is within `127 ± 5`, and the target is `+00000`.

The target is spur RPM rather than a ground speed so the drivetrain geometry lives in one place.  The Jetson converts from m/s with its `spur_to_wheel_ratio` and `tire_diameter` parameters; see `jetson/README.md`.

#### Speed Gains

Sent by the Jetson at startup and whenever its gain parameters change, and repeated until the Arduino reports the frame's sequence number back in Gains Sequence.  Always exactly `G,qqq,t,`, then eight `±nnnnnnn,` fields, then a two digit checksum: 82 bytes before the newline.  Values are signed thousandths, so `+0009500` is 9.5.  Speed is in spur RPM and the output is a throttle pulse offset from 1500 µs.

| Field Index | Description  | Units                                | Notes                                                     |
| ----------- | ------------ | ------------------------------------ | --------------------------------------------------------- |
| 0           | Frame Type   | `G`                                  |                                                           |
| 1           | Sequence     | [0,255], 3 digits                    | Echoed in Gains Sequence once applied.  The Jetson never sends `000`. |
| 2           | Trace        | `0` or `1`                           | `1` enables the `T,` trace line                           |
| 3           | kS           | µs                                   | Static feedforward, added whenever the target is nonzero  |
| 4           | kV           | µs per 1000 RPM                      | Velocity feedforward                                      |
| 5           | kP           | µs per 1000 RPM of error             |                                                           |
| 6           | kI           | µs per 1000 RPM of error, per second |                                                           |
| 7           | kD           | µs per 1000 RPM per second           | Acts on measured speed, not error                         |
| 8           | I Limit      | µs, non-negative                     | Integrator clamp                                          |
| 9           | Output Limit | µs, [0,500]                          | Largest drive offset                                      |
| 10          | Brake Limit  | µs, [0,440]                          | Braking effort past the ESC's brake threshold.  `0` coasts instead of braking. |
| 11          | Checksum     | 2 uppercase hex digits               | 8-bit sum of every preceding byte                         |

Unlike a drive command, which is replaced 50 times a second, a corrupted gain would persist, which is why only this frame carries a checksum.  A frame that fails any check leaves the gains in use untouched, and a gains frame never feeds the watchdog.

## Speed Control

In Auto Active the Arduino closes the loop from the target spur RPM to the throttle pulse once per 20 ms PWM frame, in `SpeedController` in `src/arduino_rcm/arduino_rcm.ino`:

```text
output = kS + kV * |target| + kP * error + I - kD * d(measured)/dt     [us]
I      = I + kI * error * dt, clamped to +/- I Limit
pulse  = 1500 +/- output                                               [us]
```

with the rate gains per 1000 RPM.  Feedforward carries most of the output and the PID terms trim it, so set kS and kV first.  The integrator is held while the output is pinned at a limit, and cleared whenever the target is zero.  The controller resets every time Auto Active is left, and RC mode remains open loop.

**Speed measurement.**  The tachometer timestamps every revolution and averages the whole revolutions inside a 100 ms window, but always at least one.  A revolution still in progress that has already outlasted the last measured period caps the estimate, so a stopping wheel reads low promptly, and 400 ms without a revolution reads as stopped.  That puts the slowest speed the controller can see at 150 spur RPM, about 0.3 m/s; slower targets run on feedforward alone.  At 1500 spur RPM (3.2 m/s) a new timestamp arrives every 40 ms.  A pin change with no change in level is counted as a whole pulse swallowed by a loop stall, which on the bench happened to roughly 5% of revolutions, and a revolution shorter than 3 ms is rejected as noise.

**Actuator.**  The output passes through the existing low speed dithering, so the region inside the ESC's ±50 µs deadband is the bottom of the controller's range rather than a dead zone.  Reverse is now dithered the same way as forward.

**Direction.**  The sensor cannot see direction, so the controller takes the direction it last drove as the direction of travel and changes it only after the wheels have read stopped for a further 100 ms.  Until then a reversal is tracked as a zero target, so the car coasts (or brakes) to a stop first.

**Braking.**  Off by default: with Brake Limit `0`, a lower target or a stop coasts.  The VXL-3S reads a reverse side pulse as a brake only until it sees neutral, and after that the same pulse drives in reverse, so braking is never dithered.  When Brake Limit is nonzero and the car is moving forward, an output below -5 µs sends a steady pulse starting 60 µs below neutral, until the output returns to zero or the wheels stop.  If speed climbs 300 RPM above its lowest point since braking began, the ESC has gone to reverse drive and the direction-blind sensor would read that as more forward speed, so braking locks out until the wheels stop.  Reverse never brakes, because its far side of neutral is forward drive.

### Bench Characterization

Open loop through the dithering, car on blocks, 11.8 V pack:

| Throttle Pulse | Steady Spur RPM       | Notes                                                 |
| -------------- | --------------------- | ----------------------------------------------------- |
| 1500-1525 µs   | 0                     | Too little dither duty to break static friction       |
| 1530-1550 µs   | 225-2300, ~105 RPM/µs | Dithered and roughly linear                           |
| 1555-1585 µs   | ~2900                 | Flat: the ESC's own minimum steady speed              |
| 1590-1600 µs   | 3100-3550, ~50 RPM/µs |                                                       |
| 1470, 1460 µs  | 240, 1070 in reverse  | From a standstill, reverse dithering mirrors forward  |

Steps behave like a first order lag with a 450-650 ms time constant and little dead time.  Coasting down at neutral is slower, 700-1000 ms.  A steady 1440 µs pulse from forward speed braked to a stop and held there, while a dithered 1470 µs braked and then drove away in reverse, which is what shaped the braking rules above.

### Bench Tuning

The compiled-in defaults, and the matching per m/s defaults in `jetson/cfr_arduino_bridge/config/arduino_bridge.yaml`, are kS 28 µs; kV 9.5, kP 16, kI 10 and kD 0 per 1000 RPM; I Limit 60 µs, Output Limit 128 µs and Brake Limit 0.  kS and kV come from the characterization above.  kP and kI came from sweeping them with the car on blocks over a path following-like profile of 0, 1.5, 3.2, 0.8, 0, -1.0 and 0 m/s, each held 3-4 s, scored in spur RPM:

| kP  | kI  | Mean Abs. Error | Settled Mean Abs. Error | Settled Std. Dev. | Time to 90% of Step |
| --- | --- | --------------- | ----------------------- | ----------------- | ------------------- |
| 16  | 0   | 132             | 100                     | 161               | 0.48 s              |
| 12  | 10  | 126             | 96                      | 103               | 0.46 s              |
| 16  | 10  | 121             | 95                      | 106               | 0.44 s              |
| 16  | 20  | 143             | 131                     | 193               | 0.44 s              |
| 24  | 10  | 124             | 84                      | 98                | 0.40 s              |

kP 24 was marginally sharper on blocks, but kP 16 leaves more margin for a loaded car.  Larger kI winds up during the ~0.4 s after a start before the tachometer has timed a revolution, and then overshoots.  I Limit has to cover the ESC's plateau: holding 3000 RPM takes about 30 µs more than the straight line feedforward predicts.

Most of the remaining error is ripple at low speed, where the dithering drives the free wheels in bursts: a standard deviation of roughly 200-350 RPM around targets of 150-500 RPM, though the means land close.  The car's own inertia on the ground should smooth much of that out.

With those gains, still on blocks:

| Check                                   | Result                                                                                   |
| --------------------------------------- | ---------------------------------------------------------------------------------------- |
| Compiled-in defaults, no gains frame    | 1000, 2000 and -800 RPM targets settled at 1052, 2081 and -812 RPM                       |
| Jetson drive commands stop              | Neutral pulse 208 ms later, from the 200 ms watchdog                                     |
| XBee E-Stop                             | Neutral pulse and E-Stop mode within 81 ms                                               |
| Auto Ready cleared                      | Neutral pulse within one 20 ms control tick                                              |
| +1200 to -1200 RPM                      | No reverse side pulse until the wheels read stopped; reverse began 1.15 s after the command and settled at -1240 RPM |
| Stop from 2500 RPM, coasting            | Reads zero after 2.2 s                                                                   |
| Stop from 2500 RPM, Brake Limit 40 µs   | Steady 1400 µs brake, reads zero after 1.7 s and stays there                             |

None of this says much about the loaded car.  Retune on the ground before relying on it.  The ground retune, and the rest of the work needed to make the Gazebo simulator a twin of the car rather than a plausible-looking stand-in, is [docs/characterization.md](docs/characterization.md); [docs/field-card.md](docs/field-card.md) is the printable version to take to the test site.

The whole campaign runs through the runtime `speed_*` parameters and the `D,` debug line this firmware already emits, so **no step requires reflashing the Arduino** — which matters, because the sequences that need sweeping gains are the ones run furthest from a bench.

## Onboard I/O

The Arduino Uno carries the [FlippinDisaster shield](https://github.com/dkt01/FlippinDisaster/tree/master/Shield) and a [SparkFun XBee Shield](https://www.sparkfun.com/products/12847) with its switch in the `DLINE` position.

| Pin    | Assignment                       | Notes                                                                        |
| ------ | -------------------------------- | ---------------------------------------------------------------------------- |
| D0/D1  | USB serial to the Jetson         | Hardware USART, 115200 baud.                                                 |
| D2/D3  | XBee                             | `SoftwareSerial`, 57600 baud, via the XBee shield's `DLINE` switch position.  |
| D5     | *unused*                         | Shield `FR` header.  See the note below before reusing it.                    |
| D8     | RPM sensor input                 | Shield prototyping area or the Uno header.                                    |
| D9     | Steering PWM                     | Timer1 `OC1A`, shield `RL` header.                                            |
| D10    | Throttle PWM                     | Timer1 `OC1B`, shield `FL` header.                                            |
| A0     | Battery voltage divider          | ADC with the internal 1.1 V reference.                                        |

Timer allocation is fully committed and worth stating explicitly, because it constrains what can be added later:

* **Timer0** — `millis()`/`micros()`, used throughout.
* **Timer1** — servo PWM in Fast PWM mode 14, using `ICR1` as `TOP` and both compare registers as outputs.  `ICR1` must be `TOP` for a 50 Hz frame at usable resolution; the fixed-`TOP` modes cap at 1023 counts, which forces either a 244 Hz frame or ~62 steps across the 1000-2000 µs range.
* **Timer2** — free, but has no usable external clock input on an Uno (`TOSC1`/`TOSC2` are the 16 MHz crystal pins).

Consequently the ATmega328P has **no free hardware pulse counter** in this configuration.  `T1` (pin 5) and `T0` (pin 4) are the only counter inputs and both timers behind them are spoken for, and `ICP1` input capture (pin 8) needs `ICR1`.  Adding one would require moving the servo PWM off-chip, for example to an I²C PWM controller.

### RPM sensor

A [Traxxas 6520/6522 RPM sensor](https://www.traxxas.com/products/parts/6520) — a hall switch reading a trigger magnet in the spur gear — wired to **D8**.  Its output is open collector. It pulls the line low and otherwise leaves it floating so the firmware enables D8's internal pull-up; without a pull-up no edge is ever latched and RPM reads 0.  One magnet gives one pulse per spur revolution, so the reported figure is spur RPM, not motor or wheel RPM.  The Slash 4X4 reduces spur to wheel 2.85:1 (the manual's final ratio is spur/pinion × 2.85, so the fitted 9T pinion does not enter into it), and the Traxxas 6764 Gravix 2.8″ tire is a nominal 4.5″ (114.3 mm) outer diameter, giving `wheel RPM = spur RPM / 2.85` and `speed (m/s) = wheel RPM × π × 0.1143 / 60`.  The Jetson bridge publishes both (see `jetson/README.md`) and the TUI displays them; the firmware stays on raw spur RPM.  A stock Slash 4x4 spurs at roughly 12,000 RPM flat out, around 200 Hz.

The firmware does **not** install an interrupt handler for it.  Instead the pin's `PCINT` mask bit is set while its group enable (`PCIE0`) is left clear, so the hardware latches `PCIF0` on every edge without dispatching a vector.  A hardware flag is unaffected by `cli()`, so an edge arriving during a `SoftwareSerial` blackout waits for the main loop instead of being lost.  The trade-off is that `PCIF0` is a single bit: one poll observes *at least one* edge rather than a count.  The firmware reads the pin level alongside the flag, so a poll that finds the level unchanged counts the whole pulse that fit inside a loop stall, and only that revolution's timestamp is late.  The hall pulse is only about 10% of a revolution, so the blocking XBee transmits do swallow one now and then: about 5% of revolutions on the bench.

Pin 5 is the tempting choice — it is broken out on the shield's `FR` header *and* it is `T1`, the hardware counter input — but it does not work for either purpose here:

* `T1` is Timer1's external clock source, and Timer1 generates the servo PWM.
* Pin 5 is in PORTD, whose `PCINT2` group belongs to `SoftwareSerial`'s receive vector.  Adding a mask bit there would run that ISR on every tach edge, and it clears `PCIF2` before the main loop could read it.

**Wiring note:** the shield ties a 180 Ω resistor and an indicator LED to ground on each servo header net, including `FR`/D5.  That is fine for an output but not for an input — an open-collector sensor with a 10 kΩ pull-up settles near 2.05 V against it, below the AVR's 3.0 V V<sub>IH</sub>, so the pin would never read high.  If a servo header is ever reused as an input, lift its resistor or LED first.  Sensor lines also run alongside the ESC, so a 1 kΩ series resistor with 1 nF to ground at the Arduino end is worthwhile; a polled latch counts noise as faithfully as it counts pulses.

### Battery voltage

A resistor divider from the pack to **A0**: 100 kΩ from the pack to the node, 9.09 kΩ from the node to ground, both 1%, with 100 nF across the bottom leg.  The divider ratio is 0.083325 and the Thévenin source impedance 8.3 kΩ, just inside the ADC's 10 kΩ guideline.  Quiescent draw is 119 µA.  Sense at the pack terminals, not downstream of the ESC, and share a ground with the Arduino.

The ADC uses the **internal 1.1 V bandgap**, not `AVCC`.  `AVCC` is the board's 5 V rail and moves several percent with USB and regulator load, which is worth hundreds of millivolts referred back to the pack.  1.1 V / 1024 counts is 1.0742 mV per count at the pin, or 12.89 mV per count at the pack.

The bandgap is only specified to 1.0-1.2 V, so **one-time calibration is required per board**: read a known pack voltage with a meter, compare against the millivolts reported in the debug frame, and scale `BATTERY_UV_PER_COUNT` in `src/arduino_rcm/arduino_rcm.ino` by `meter / reported`.

Conversions are started and collected by polling, so the ADC never blocks and never takes an interrupt.  Sixty-four samples are averaged and passed through an exponential filter with roughly a 0.85 s time constant, which keeps throttle sag from being reported as a discharged pack.  Because the reference selection is held in `ADMUX` for the whole run, **`analogRead()` must not be called anywhere in the sketch** — the Arduino core resets the reference to `AVCC`, which would silently scale every reading by 4.5×.

`BATTERY_LEVEL` maps 10.0 V to `0` and 12.6 V to `255` (a 3S lithium polymer pack, 4.2 V per cell charged and a 3.33 V per cell floor), which works out to 10.2 mV per code.

## Simulation

Both competition courses are simulated in Gazebo Harmonic, one launch file
each:

```bash
ros2 launch cfr_arduino_bridge speed_course.launch.py      # 135 ft x 47 ft oval
ros2 launch cfr_arduino_bridge obstacle_course.launch.py   # 65 ft x 48 ft, 11 sections
```

Both are built from the site-layout DXF, and the obstacle course's obstacles
from the team's CAD. Both start on the same visual signal: it loads showing
red, and `obstacle_randomizer_node` turns it to green at 90 degrees a second
while the simulation runs -- along with the obstacle course's buckets and
hoops -- so a layout can be re-drawn and a start signalled without a restart.

`start_signal_detector_node` watches the camera for that turn and latches it
on `/start_signal_detector/go`, which is what an autonomous run waits on in
place of the Arduino's Manual Start bit. It runs against the simulated camera
with `sensors:=true` and against the ZED on the car. The course is outdoors
and there will be people about in every color, so it finds the signal first
-- a place in the image that holds red long enough to be it, of about the size
an arm is -- and only then waits for that place to turn green; `armed` on
`/start_signal_detector/state` says whether it has.

See [jetson/README.md](jetson/README.md#gazebo-simulation) for the launch
arguments, the randomizer's services, and how to regenerate the worlds and
meshes when the drawing or the CAD changes.

## Documentation

* [Characterization procedure](docs/characterization.md) and its [printable field card](docs/field-card.md)
* [Traxxas Slash 4X4 VXL Ultimate](https://traxxas.com/media/productattach/C-68277-4/2/68277-4-OM-EN-R01.pdf)
* [Traxxas VXL-3S ESC](https://traxxas.com/media/productattach/3350R/8/KC2014-R02-3355R-VXL-3s-Installation%20Instruction_160217-ML_WEB_EN.pdf)
* [ATmega328P datasheet](https://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-7810-Automotive-Microcontrollers-ATmega328P_Datasheet.pdf) — timer, pin change interrupt, and ADC chapters
* [SparkFun XBee Shield hookup guide](https://learn.sparkfun.com/tutorials/xbee-shield-hookup-guide)

### Gotchas

* The VXL-3S ESC has a 50us deadband around 1500us pulse length.
* The VXL-3S ESC does not power on when the Arduino is already sending signals.  One work around is to power on the autonomy layer after powering on the ESC.
