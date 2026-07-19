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

Messages transmitted each direction are ASCII serial where fields are comma-separated and messages are separated by a new line character (`\n`).

### Onboard -> Offboard

| Field Index | Description   | Data Type | Data Range | Notes                                                                    |
| ----------- | ------------- | --------- | ---------- | ------------------------------------------------------------------------ |
| 0           | Auto Mode     | Enum      | [0,4]      | `0` E-Stop, `1` RC Armed, `2` RC Active, `3` Auto Armed, `4` Auto Active |
| 1           | Battery Level | Integer   | [0,255]    | `0` is empty battery, `255` is full battery.                             |

### Offboard -> Onboard

| Field Index | Description           | Data Type | Data Range | Notes                                                                           |
| ----------- | --------------------- | --------- | ---------- | ------------------------------------------------------------------------------- |
| 0           | E-Stop State          | Boolean   | `0` or `1` | `1` indicates E-Stop active                                                     |
| 1           | Auto Arm              | Boolean   | `0` or `1` | `1` indicates autonomous mode active.  `0` indicates RC only                    |
| 2           | Manual Start          | Boolean   | `0` or `1` | `1` indicates robot should start autonomous driving without visual start signal |
| 3           | RC Contorller Present | Boolean   | `0` or `1` | `1` indicates gamepad is connected                                              |
| 4           | RC - Left Joystick X  | Integer   | [0,255]    | `0` is full left, `255` is full right                                           |
| 5           | RC - Left Joystick Y  | Integer   | [0,255]    | `0` is full up, `255` is full down                                              |
| 6           | RC - Right Joystick X | Integer   | [0,255]    | `0` is full left, `255` is full right                                           |
| 7           | RC - Right Joystick Y | Integer   | [0,255]    | `0` is full up, `255` is full down                                              |
| 8           | RC - X                | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 9           | RC - ○                | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 10          | RC - □                | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 11          | RC - △                | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 12          | RC - L1               | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 13          | RC - R1               | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 14          | RC - L2               | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 15          | RC - R2               | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 16          | RC - L3               | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 17          | RC - R3               | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 18          | RC - Select           | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 19          | RC - Start            | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 20          | RC - PS               | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 21          | RC - DPad Up          | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 22          | RC - DPad Right       | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 23          | RC - DPad Down        | Boolean   | `0` or `1` | `1` indicates pressed                                                           |
| 24          | RC - DPad Left        | Boolean   | `0` or `1` | `1` indicates pressed                                                           |

## Documentation

* [Traxxas Slash 4X4 VXL Ultimate](https://traxxas.com/media/productattach/C-68277-4/2/68277-4-OM-EN-R01.pdf)
* [Traxxas VXL-3S ESC](https://traxxas.com/media/productattach/3350R/8/KC2014-R02-3355R-VXL-3s-Installation%20Instruction_160217-ML_WEB_EN.pdf)
