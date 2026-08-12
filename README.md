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

This link was ASCII comma-separated through mid-2026.  It was packed because of the Arduino's interrupt budget rather than for bandwidth: `SoftwareSerial` disables interrupts for a full byte time (~174 µs at 57600 baud) and works through a frame's bytes back to back, so the ~50 byte ASCII command frame suppressed interrupts for nearly 9 ms in one stretch, twenty times a second.  The hardware USART's receive buffer is two bytes deep and a byte arrives every 10 µs at 1 Mbaud, so every command frame overran the USB link and corrupted whichever Jetson command was in flight.  Eight bytes cuts that window by more than six times.

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

The robot features an Arduino that controls the car's actuators and an NVIDIA Jetson Orin Nano Super.  The two components communicate over a USB serial interface at 1000000 Baud.

Messages transmitted each direction are ASCII serial where fields are comma-separated and messages are separated by a new line character (`\n`).

### Arduino -> Jetson

| Field Index | Description   | Data Type | Data Range | Notes                                                                           |
| ----------- | ------------- | --------- | ---------- | ------------------------------------------------------------------------------- |
| 0           | E-Stop State  | Boolean   | `0` or `1` | `1` indicates E-Stop active                                                     |
| 1           | Auto Arm      | Boolean   | `0` or `1` | `1` indicates autonomous mode active.  `0` indicates RC only                    |
| 2           | Manual Start  | Boolean   | `0` or `1` | `1` indicates robot should start autonomous driving without visual start signal |
| 3           | Auto Mode     | Enum      | [0,4]      | `0` E-Stop, `1` RC Armed, `2` RC Active, `3` Auto Armed, `4` Auto Active        |
| 4           | Battery Level | Integer   | [0,255]    | `0` is empty battery, `255` is full battery.                                    |
| 5           | Spur RPM      | Integer   | [0,65535]  | Spur gear revolutions per minute.  `0` also means stopped or no sensor.          |

`Deserialize()` on the Jetson requires exactly six fields, so firmware predating the RPM field is rejected outright rather than parsed with a stale speed.  Flash both sides together.

### Jetson -> Arduino

| Field Index | Description    | Data Type | Data Range | Notes                                      |
| ----------- | -------------- | --------- | ---------- | ------------------------------------------ |
| 0           | Auto Ready     | Boolean   | `0` or `1` | `1` indicates auto control requested       |
| 1           | Steering Angle | Integer   | [0,255]    | `0` is full right, `255` is full left      |
| 2           | Velocity       | Integer   | [0,255]    | `0` is full reverse, `255` is full forward |

## Onboard I/O

The Arduino Uno carries the [FlippinDisaster shield](https://github.com/dkt01/FlippinDisaster/tree/master/Shield) and a [SparkFun XBee Shield](https://www.sparkfun.com/products/12847) with its switch in the `DLINE` position.

| Pin    | Assignment                       | Notes                                                                        |
| ------ | -------------------------------- | ---------------------------------------------------------------------------- |
| D0/D1  | USB serial to the Jetson         | Hardware USART, 1000000 baud.                                                |
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

A [Traxxas 6520/6522 RPM sensor](https://www.traxxas.com/products/parts/6520) — a hall switch reading a trigger magnet in the spur gear — wired to **D8**.  One magnet gives one pulse per spur revolution, so the reported figure is spur RPM, not motor or wheel RPM.  A stock Slash 4x4 spurs at roughly 12,000 RPM flat out, around 200 Hz.

The firmware does **not** install an interrupt handler for it.  Instead the pin's `PCINT` mask bit is set while its group enable (`PCIE0`) is left clear, so the hardware latches `PCIF0` on every edge without dispatching a vector.  A hardware flag is unaffected by `cli()`, so an edge arriving during a `SoftwareSerial` blackout waits for the main loop instead of being lost.  The trade-off is that `PCIF0` is a single bit: one poll observes *at least one* edge rather than a count, so sustained loop stalls longer than half the pulse interval undercount.  Polling runs at loop rate and the only stalls that come close are the blocking XBee transmits.

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

## Documentation

* [Traxxas Slash 4X4 VXL Ultimate](https://traxxas.com/media/productattach/C-68277-4/2/68277-4-OM-EN-R01.pdf)
* [Traxxas VXL-3S ESC](https://traxxas.com/media/productattach/3350R/8/KC2014-R02-3355R-VXL-3s-Installation%20Instruction_160217-ML_WEB_EN.pdf)
* [ATmega328P datasheet](https://ww1.microchip.com/downloads/en/DeviceDoc/Atmel-7810-Automotive-Microcontrollers-ATmega328P_Datasheet.pdf) — timer, pin change interrupt, and ADC chapters
* [SparkFun XBee Shield hookup guide](https://learn.sparkfun.com/tutorials/xbee-shield-hookup-guide)
