"""Platform-independent E-Stop/RC logic shared between the Raspberry Pi GPIO
box (estop.py) and the laptop terminal emulator (estop_tui.py).
"""

import dataclasses
import os
import platform
import struct
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum

import serial
import serial.tools.list_ports
from xbee import XBee
from xbee.thread.base import ThreadQuitException

try:
    import evdev

    EVDEV_SUPPORTED = True
except ImportError:
    evdev = None
    EVDEV_SUPPORTED = False

try:
    import pygame

    PYGAME_SUPPORTED = True
except ImportError:
    pygame = None
    PYGAME_SUPPORTED = False

ROBOT_ADDR = b"\x00\x01"
TX_OPT = b"\x00"

# Offboard link wire format.  Must stay byte-for-byte identical to the constants
# and bit assignments in src/arduino_rcm/arduino_rcm.ino.
#
# Both directions are packed binary rather than ASCII.  The reason is the
# Arduino's interrupt budget, not bandwidth: SoftwareSerial disables interrupts
# for a whole byte time while receiving, so the old ~50 byte ASCII command frame
# blacked out the firmware's interrupts for nearly 9ms at a time and overran the
# USB link to the Jetson.  Every byte removed here is ~174us of blackout removed
# onboard.
#
# The one byte tag makes a version mismatch between this script and the firmware
# fail closed instead of decoding stale ASCII as flag bits.
CMD_TAG = 0xC1
CMD_STRUCT = struct.Struct("<BBBBBBBB")  # tag, flagsA, flagsB, flagsC, 4 axes
STATUS_TAG = 0x51
# tag, mode, battery level, battery mV, steering us, throttle us, rpm
STATUS_STRUCT = struct.Struct("<BBBHHHH")

# Bit position of each boolean within its flag byte.  Order is arbitrary but
# fixed; the firmware indexes the same way.
CMD_FLAGS_A = (
    "ESTOP",
    "AUTO_ARM",
    "MANUAL_START",
    "RC_PRESENT",
    "BUTTON_X",
    "BUTTON_O",
    "BUTTON_SQUARE",
    "BUTTON_TRIANGLE",
)
CMD_FLAGS_B = (
    "BUTTON_L1",
    "BUTTON_R1",
    "BUTTON_L2",
    "BUTTON_R2",
    "BUTTON_L3",
    "BUTTON_R3",
    "BUTTON_SELECT",
    "BUTTON_START",
)
CMD_FLAGS_C = (
    "BUTTON_PS",
    "BUTTON_UP",
    "BUTTON_RIGHT",
    "BUTTON_DOWN",
    "BUTTON_LEFT",
)
CMD_AXES = ("AXIS_LX", "AXIS_LY", "AXIS_RX", "AXIS_RY")

TIMEOUT_CONTROLLER = 0.2
# This drives only the TUI link indicator. Keep it longer than the Arduino's
# command failsafe because duplex XBee traffic can lose short bursts of feedback.
TIMEOUT_COMMS = 0.5
CYCLE_RATE_WINDOW = 5.0
# How often to re-scan for a device while none is connected; doesn't need to be
# fast since a human plugging in a controller/XBee won't notice ~1s of latency.
DISCOVERY_INTERVAL = 1.0
SERIAL_TRACE_ENV = "ESTOP_SERIAL_TRACE"
# SERIAL_TRACE_DEFAULT = "/tmp/xbee-serial.log"
SERIAL_TRACE_DEFAULT = None

# Candidate serial adapters preferred during auto-discovery, most specific first.
SERIAL_PORT_PREFERRED_TOKENS = ("xbee", "ftdi", "cp210", "usb")


class LEDMode(Enum):
    OFF = 0
    ON = 1
    BLINK = 2


@dataclass
class LEDState:
    estop: LEDMode = LEDMode.OFF
    estop_sense: LEDMode = LEDMode.OFF
    estop_override: LEDMode = LEDMode.OFF
    auto_arm: LEDMode = LEDMode.OFF
    comms: LEDMode = LEDMode.OFF
    mode_stop: LEDMode = LEDMode.OFF
    mode_rc: LEDMode = LEDMode.OFF
    battery: LEDMode = LEDMode.OFF
    mode_auto: LEDMode = LEDMode.OFF


@dataclass
class InputState:
    estop: bool = False
    estop_sense: bool = False
    estop_override: bool = False
    auto_arm: bool = False
    manual_start: bool = False


AXIS_ZERO = 127
PRESSURE_ZERO = 0
ACCEL_ZERO = 0
GYRO_ZERO = 0


@dataclass
class ControllerState:
    BUTTON_SELECT: bool = False
    BUTTON_L3: bool = False
    BUTTON_R3: bool = False
    BUTTON_START: bool = False
    BUTTON_UP: bool = False
    BUTTON_RIGHT: bool = False
    BUTTON_DOWN: bool = False
    BUTTON_LEFT: bool = False
    BUTTON_L2: bool = False
    BUTTON_R2: bool = False
    BUTTON_L1: bool = False
    BUTTON_R1: bool = False
    BUTTON_TRIANGLE: bool = False
    BUTTON_O: bool = False
    BUTTON_X: bool = False
    BUTTON_SQUARE: bool = False
    BUTTON_PS: bool = False
    AXIS_LX: int = AXIS_ZERO
    AXIS_LY: int = AXIS_ZERO
    AXIS_RX: int = AXIS_ZERO
    AXIS_RY: int = AXIS_ZERO
    PRESSURE_UP: int = PRESSURE_ZERO
    PRESSURE_RIGHT: int = PRESSURE_ZERO
    PRESSURE_DOWN: int = PRESSURE_ZERO
    PRESSURE_LEFT: int = PRESSURE_ZERO
    PRESSURE_L2: int = PRESSURE_ZERO
    PRESSURE_R2: int = PRESSURE_ZERO
    PRESSURE_L1: int = PRESSURE_ZERO
    PRESSURE_R1: int = PRESSURE_ZERO
    PRESSURE_TRIANGLE: int = PRESSURE_ZERO
    PRESSURE_O: int = PRESSURE_ZERO
    PRESSURE_X: int = PRESSURE_ZERO
    PRESSURE_SQUARE: int = PRESSURE_ZERO
    ACCEL_X: int = ACCEL_ZERO
    ACCEL_Y: int = ACCEL_ZERO
    ACCEL_Z: int = ACCEL_ZERO
    GYRO_Z: int = GYRO_ZERO
    comms_ok: bool = False
    device_name: str = ""


class AutoMode(Enum):
    UNKNOWN = -1
    ESTOP = 0
    RC_ARMED = 1
    RC_ACTIVE = 2
    AUTO_ARMED = 3
    AUTO_ACTIVE = 4


@dataclass
class RobotState:
    auto_mode: AutoMode = AutoMode.UNKNOWN
    battery_level: int = 0
    # Raw pack millivolts as measured by the Arduino, before scaling to
    # battery_level.  Reported separately because the scaled level cannot be
    # inverted once its endpoints change.
    battery_mv: int = 0
    steering_output_us: int = 1500
    throttle_output_us: int = 1500
    # Spur gear RPM from the hall sensor.  One trigger magnet, so this is spur
    # revolutions rather than motor or wheel revolutions.
    rpm: int = 0
    comms_ok: bool = False
    comm_port: str = ""
    tx_ack: bool = False
    tx_status: int = 255
    tx_message: bytes = b""


KEYCODE_MAP = {
    0: "AXIS_LX",
    1: "AXIS_LY",
    2: "AXIS_RX",
    5: "AXIS_RY",
    44: "PRESSURE_UP",
    45: "PRESSURE_RIGHT",
    46: "PRESSURE_DOWN",
    47: "PRESSURE_LEFT",
    48: "PRESSURE_L2",
    49: "PRESSURE_R2",
    50: "PRESSURE_L1",
    51: "PRESSURE_R1",
    52: "PRESSURE_TRIANGLE",
    53: "PRESSURE_O",
    54: "PRESSURE_X",
    55: "PRESSURE_SQUARE",
    59: "ACCEL_X",
    60: "ACCEL_Y",
    61: "ACCEL_Z",
    62: "GYRO_Z",
    288: "BUTTON_SELECT",
    289: "BUTTON_L3",
    290: "BUTTON_R3",
    291: "BUTTON_START",
    292: "BUTTON_UP",
    293: "BUTTON_RIGHT",
    294: "BUTTON_DOWN",
    295: "BUTTON_LEFT",
    296: "BUTTON_L2",
    297: "BUTTON_R2",
    298: "BUTTON_L1",
    299: "BUTTON_R1",
    300: "BUTTON_TRIANGLE",
    301: "BUTTON_O",
    302: "BUTTON_X",
    303: "BUTTON_SQUARE",
    704: "BUTTON_PS",
}

SERIAL_BUTTONS = [
    "BUTTON_SELECT",
    "BUTTON_L3",
    "BUTTON_R3",
    "BUTTON_START",
    "BUTTON_UP",
    "BUTTON_RIGHT",
    "BUTTON_DOWN",
    "BUTTON_LEFT",
    "BUTTON_L2",
    "BUTTON_R2",
    "BUTTON_L1",
    "BUTTON_R1",
    "BUTTON_TRIANGLE",
    "BUTTON_O",
    "BUTTON_X",
    "BUTTON_SQUARE",
    "BUTTON_PS",
]
SERIAL_AXES = ["AXIS_LX", "AXIS_LY", "AXIS_RX", "AXIS_RY"]

# Raw pygame joystick indices for the connected PS3 controller. Index 2 is the
# shared trigger axis, so it must not be serialized as the right stick X axis.
PYGAME_AXIS_MAP = {
    0: "AXIS_LX",
    1: "AXIS_LY",
    3: "AXIS_RX",
    4: "AXIS_RY",
}
PYGAME_BUTTON_MAP = {
    0: "BUTTON_X",
    1: "BUTTON_O",
    2: "BUTTON_TRIANGLE",
    3: "BUTTON_SQUARE",
    4: "BUTTON_L1",
    5: "BUTTON_R1",
    6: "BUTTON_L2",
    7: "BUTTON_R2",
    8: "BUTTON_SELECT",
    9: "BUTTON_START",
    10: "BUTTON_PS",
    11: "BUTTON_L3",
    12: "BUTTON_R3",
    13: "BUTTON_UP",
    14: "BUTTON_DOWN",
    15: "BUTTON_LEFT",
    16: "BUTTON_RIGHT",
}
# Windows' XInput compatibility layer enumerates the PS3 controller as an Xbox
# controller, remapping axis/button indices and reporting L2/R2 as separate
# trigger axes instead of buttons.
PYGAME_AXIS_MAP_WINDOWS = {
    0: "AXIS_LX",
    1: "AXIS_LY",
    2: "AXIS_RX",
    3: "AXIS_RY",
}
PYGAME_BUTTON_MAP_WINDOWS = {
    0: "BUTTON_X",
    1: "BUTTON_O",
    2: "BUTTON_SQUARE",
    3: "BUTTON_TRIANGLE",
    4: "BUTTON_L1",
    5: "BUTTON_R1",
    6: "BUTTON_SELECT",
    7: "BUTTON_START",
    8: "BUTTON_L3",
    9: "BUTTON_R3",
    10: "BUTTON_PS",
}
PYGAME_TRIGGER_AXIS_L2_WINDOWS = 4
PYGAME_TRIGGER_AXIS_R2_WINDOWS = 5
# Digital press threshold on the 0-255 scaled trigger pressure.
PYGAME_TRIGGER_PRESSED_THRESHOLD_WINDOWS = 32
PYGAME_HAT_INDEX = 0

IS_WINDOWS = platform.system() == "Windows"
XBOX_CONTROLLER_NAME_TOKENS = ("xbox", "xinput", "x-box")
PS_CONTROLLER_NAME_TOKENS = ("playstation", "dualshock", "sony", "ps3", "ps4", "ps5")


def _useWindowsControllerMap(deviceName: str) -> bool:
    """Decide whether to use the Windows/Xbox-style axis+button layout.

    Applies whenever the reported device name says Xbox (the case where Windows
    has remapped a PS3 pad), or on Windows when the name is inconclusive (some
    drivers don't report a usable name at all).
    """
    name = (deviceName or "").lower()
    if any(token in name for token in XBOX_CONTROLLER_NAME_TOKENS):
        return True
    if any(token in name for token in PS_CONTROLLER_NAME_TOKENS):
        return False
    return IS_WINDOWS


class TimestampedSerial:
    def __init__(self, serial_port, trace_path: str | None):
        self.serial_port = serial_port
        self.trace_file = open(trace_path, "a", buffering=1) if trace_path else None
        self.trace_mutex = threading.Lock()

    def read(self, size=1):
        start_timestamp = time.monotonic_ns()
        data = self.serial_port.read(size)
        end_timestamp = time.monotonic_ns()
        if self.trace_file:
            with self.trace_mutex:
                self.trace_file.write(
                    f"start={start_timestamp} end={end_timestamp} "
                    f"duration_ns={end_timestamp - start_timestamp} "
                    f"requested={size} returned={len(data)} data={data.hex()}\n"
                )
        return data

    def close(self):
        try:
            self.serial_port.close()
        finally:
            if self.trace_file:
                self.trace_file.close()

    def __getattr__(self, name):
        return getattr(self.serial_port, name)


class CycleRateMonitor:
    def __init__(self, window: float):
        self.window = window
        self.cycles: dict[str, deque[float]] = {}
        self.durations: dict[str, deque[tuple[float, float]]] = {}
        self.mutex = threading.Lock()

    def record(self, name: str):
        now = time.monotonic()
        cutoff = now - self.window
        with self.mutex:
            timestamps = self.cycles.setdefault(name, deque())
            timestamps.append(now)
            while timestamps[0] < cutoff:
                timestamps.popleft()

    def record_duration(self, name: str, duration: float):
        now = time.monotonic()
        cutoff = now - self.window
        with self.mutex:
            durations = self.durations.setdefault(name, deque())
            durations.append((now, duration))
            while durations[0][0] < cutoff:
                durations.popleft()

    def rates(self) -> dict[str, float]:
        now = time.monotonic()
        cutoff = now - self.window
        with self.mutex:
            for timestamps in self.cycles.values():
                while timestamps and timestamps[0] < cutoff:
                    timestamps.popleft()
            result = {}
            for name, timestamps in self.cycles.items():
                if len(timestamps) < 2:
                    result[name] = 0.0
                    continue
                # Divide by the span actually covered so far, not the full
                # window, so the estimate doesn't ramp up from 0 at startup.
                # len(timestamps) - 1 is the number of complete intervals
                # spanned, which is the unbiased estimator for small counts.
                span = min(self.window, now - timestamps[0])
                result[name] = (len(timestamps) - 1) / span
            return result

    def duration_stats(self) -> dict[str, tuple[float, float]]:
        now = time.monotonic()
        cutoff = now - self.window
        with self.mutex:
            for durations in self.durations.values():
                while durations and durations[0][0] < cutoff:
                    durations.popleft()
            return {
                name: (
                    sum(duration for _, duration in durations) / len(durations),
                    max(duration for _, duration in durations),
                )
                for name, durations in self.durations.items()
                if durations
            }


def compute_led_state(
    robotState: RobotState,
    inputState: InputState,
    controllerState: ControllerState,
) -> LEDState:
    """Pure decision logic for LED states; callers apply the result to GPIO/TUI output."""
    newLedState = LEDState()

    if robotState.auto_mode == AutoMode.ESTOP:
        newLedState.mode_stop = LEDMode.ON
    elif robotState.auto_mode == AutoMode.RC_ARMED:
        newLedState.mode_rc = LEDMode.ON
    elif robotState.auto_mode == AutoMode.RC_ACTIVE:
        newLedState.mode_rc = LEDMode.BLINK
    elif robotState.auto_mode == AutoMode.AUTO_ARMED:
        newLedState.mode_auto = LEDMode.ON
    elif robotState.auto_mode == AutoMode.AUTO_ACTIVE:
        newLedState.mode_auto = LEDMode.BLINK
    else:
        # Invalid
        newLedState.mode_stop = LEDMode.BLINK
        newLedState.mode_rc = LEDMode.BLINK
        newLedState.mode_auto = LEDMode.BLINK

    if robotState.battery_level < 64:
        newLedState.battery = LEDMode.OFF
    elif robotState.battery_level < 128:
        newLedState.battery = LEDMode.BLINK
    else:
        newLedState.battery = LEDMode.ON

    if robotState.comms_ok:
        newLedState.comms = LEDMode.ON
    else:
        newLedState.comms = LEDMode.BLINK

    if inputState.estop:
        newLedState.estop = LEDMode.ON

    if inputState.estop_sense:
        newLedState.estop_sense = LEDMode.ON

    if inputState.estop_override:
        newLedState.estop_override = LEDMode.ON
        if newLedState.estop == LEDMode.ON:
            newLedState.estop = LEDMode.BLINK

    if inputState.auto_arm:
        newLedState.auto_arm = LEDMode.ON

    if not controllerState.comms_ok and newLedState.mode_rc != LEDMode.OFF:
        newLedState.mode_stop = LEDMode.BLINK

    return newLedState


def controllerControlEvdev(
    controllerState: ControllerState,
    controllerMutex: threading.Lock,
    runEvent: threading.Event,
    cycleMonitor: CycleRateMonitor,
):
    if not EVDEV_SUPPORTED:
        print("evdev not installed; controller input disabled")
        while runEvent.is_set():
            time.sleep(0.1)
        return

    latestEvent = time.monotonic()
    ps = None
    # Outer loop handles disconnected controller
    while runEvent.is_set():
        cycleMonitor.record("controller-discovery")
        devices = evdev.list_devices()

        if len(devices) > 0:
            print("New controller @", devices[0])
            ps = evdev.InputDevice(devices[0])
            with controllerMutex:
                controllerState.device_name = devices[0]

        # Inner loop handles normal input events
        while runEvent.is_set() and len(evdev.list_devices()) > 0:
            cycleMonitor.record("controller-input")
            try:
                with controllerMutex:
                    for event in ps.read():
                        if (
                            event.type != evdev.ecodes.EV_SYN
                            and event.code in KEYCODE_MAP
                        ):
                            fieldName = KEYCODE_MAP[event.code]
                            value = event.value
                            if fieldName in SERIAL_AXES:
                                axisInfo = ps.absinfo(event.code)
                                value = _scaleEvdevAxis(
                                    event.value, axisInfo.min, axisInfo.max
                                )
                            setattr(controllerState, fieldName, value)
                latestEvent = time.monotonic()
                controllerState.comms_ok = True
            except OSError:
                pass

            if time.monotonic() - latestEvent >= TIMEOUT_CONTROLLER:
                # print("Timeout", time.monotonic() - latestEvent)
                with controllerMutex:
                    _zeroControllerState(controllerState)  # Zero out for safety

            time.sleep(0.02)  # Limit loop rate to 50Hz

        with controllerMutex:
            _zeroControllerState(controllerState)  # Zero out for safety

        time.sleep(
            DISCOVERY_INTERVAL
        )  # Only need to notice a reconnect, not react instantly


def _zeroControllerState(state: ControllerState) -> None:
    blank = ControllerState()
    for field in dataclasses.fields(state):
        setattr(state, field.name, getattr(blank, field.name))


def _scalePygameAxis(value: float) -> int:
    rawValue = int(round((max(-1.0, min(1.0, value)) + 1.0) * 127.5))
    return _applyAxisDeadband(rawValue)


def _applyAxisDeadband(rawValue: int) -> int:
    if 122 <= rawValue <= 133:
        return AXIS_ZERO
    if rawValue < 122:
        return int(round(rawValue * 126 / 121))
    return min(255, int(round(128 + (rawValue - 134) * 127 / 121)))


def _scaleEvdevAxis(value: int, minimum: int, maximum: int) -> int:
    if maximum <= minimum:
        return AXIS_ZERO
    rawValue = round((value - minimum) * 255 / (maximum - minimum))
    return _applyAxisDeadband(max(0, min(255, rawValue)))


def _scalePygamePressure(value: float) -> int:
    return int(round((max(-1.0, min(1.0, value)) + 1.0) / 2.0 * 255))


def _applyWindowsTriggerAxis(
    joystick,
    axisIndex: int,
    controllerState: ControllerState,
    pressureField: str,
    buttonField: str,
) -> None:
    """Windows reports L2/R2 as separate full-range trigger axes rather than buttons."""
    if axisIndex >= joystick.get_numaxes():
        return
    pressure = _scalePygamePressure(joystick.get_axis(axisIndex))
    setattr(controllerState, pressureField, pressure)
    setattr(
        controllerState,
        buttonField,
        pressure > PYGAME_TRIGGER_PRESSED_THRESHOLD_WINDOWS,
    )


def _applyPygameState(
    joystick, controllerState: ControllerState, useWindowsMap: bool
) -> None:
    axisMap = PYGAME_AXIS_MAP_WINDOWS if useWindowsMap else PYGAME_AXIS_MAP
    buttonMap = PYGAME_BUTTON_MAP_WINDOWS if useWindowsMap else PYGAME_BUTTON_MAP

    numAxes = joystick.get_numaxes()
    for axisIndex, fieldName in axisMap.items():
        if axisIndex < numAxes:
            setattr(
                controllerState,
                fieldName,
                _scalePygameAxis(joystick.get_axis(axisIndex)),
            )

    numButtons = joystick.get_numbuttons()
    for buttonIndex, fieldName in buttonMap.items():
        if buttonIndex >= numButtons:
            continue
        pressed = bool(joystick.get_button(buttonIndex))
        setattr(controllerState, fieldName, pressed)
        pressureField = "PRESSURE_" + fieldName.split("_", 1)[1]
        if hasattr(controllerState, pressureField):
            setattr(controllerState, pressureField, 255 if pressed else 0)

    if useWindowsMap:
        _applyWindowsTriggerAxis(
            joystick,
            PYGAME_TRIGGER_AXIS_L2_WINDOWS,
            controllerState,
            "PRESSURE_L2",
            "BUTTON_L2",
        )
        _applyWindowsTriggerAxis(
            joystick,
            PYGAME_TRIGGER_AXIS_R2_WINDOWS,
            controllerState,
            "PRESSURE_R2",
            "BUTTON_R2",
        )

    # On the Windows/Xbox layout the dpad is always reported via a hat; on the
    # default (PS3) layout some drivers instead expose it as buttons 13-16, so
    # only fall back to the hat there when the button count says it's absent.
    useHatForDpad = useWindowsMap or numButtons <= 13
    if useHatForDpad and joystick.get_numhats() > PYGAME_HAT_INDEX:
        hatX, hatY = joystick.get_hat(PYGAME_HAT_INDEX)
        controllerState.PRESSURE_UP = 255 if hatY > 0 else 0
        controllerState.PRESSURE_DOWN = 255 if hatY < 0 else 0
        controllerState.PRESSURE_LEFT = 255 if hatX < 0 else 0
        controllerState.PRESSURE_RIGHT = 255 if hatX > 0 else 0

    controllerState.comms_ok = True


def controllerControlPygame(
    controllerState: ControllerState,
    controllerMutex: threading.Lock,
    runEvent: threading.Event,
    cycleMonitor: CycleRateMonitor,
):
    if not PYGAME_SUPPORTED:
        print("pygame not installed; controller input disabled")
        while runEvent.is_set():
            time.sleep(0.1)
        return

    pygame.init()
    pygame.joystick.init()
    latestEvent = time.monotonic()

    while runEvent.is_set():
        cycleMonitor.record("controller-discovery")
        pygame.joystick.quit()
        pygame.joystick.init()

        joystick = None
        useWindowsMap = False
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print("New controller @", joystick.get_name())
            useWindowsMap = _useWindowsControllerMap(joystick.get_name())
            with controllerMutex:
                controllerState.device_name = joystick.get_name()

        while (
            runEvent.is_set()
            and joystick is not None
            and pygame.joystick.get_count() > 0
        ):
            cycleMonitor.record("controller-input")
            try:
                pygame.event.pump()
                with controllerMutex:
                    _applyPygameState(joystick, controllerState, useWindowsMap)
                latestEvent = time.monotonic()
            except pygame.error:
                break

            if time.monotonic() - latestEvent >= TIMEOUT_CONTROLLER:
                with controllerMutex:
                    _zeroControllerState(controllerState)

            time.sleep(0.02)  # Limit loop rate to 50Hz

        with controllerMutex:
            _zeroControllerState(controllerState)

        time.sleep(
            DISCOVERY_INTERVAL
        )  # Only need to notice a reconnect, not react instantly


def _packFlags(source, names: tuple) -> int:
    """Fold a run of named booleans into one byte, bit 0 first."""
    field = 0
    for bit, name in enumerate(names):
        if getattr(source, name):
            field |= 1 << bit
    return field


def serializeState(
    controllerState: ControllerState,
    controllerStateLock: threading.Lock,
    inputState: InputState,
    inputStateLock: threading.Lock,
) -> bytes:
    with inputStateLock:
        estop = False
        if not inputState.estop_override:
            estop = inputState.estop
        # The E-Stop and arming bits live in flag byte A alongside controller
        # buttons, so they are collected here and merged below rather than
        # packed independently.
        inputBits = (
            (1 << CMD_FLAGS_A.index("ESTOP") if estop else 0)
            | (1 << CMD_FLAGS_A.index("AUTO_ARM") if inputState.auto_arm else 0)
            | (1 << CMD_FLAGS_A.index("MANUAL_START") if inputState.manual_start else 0)
        )

    with controllerStateLock:
        # ControllerState carries comms_ok; the wire calls it RC_PRESENT.
        controllerBits = 0
        if controllerState.comms_ok:
            controllerBits |= 1 << CMD_FLAGS_A.index("RC_PRESENT")
        for bit, name in enumerate(CMD_FLAGS_A):
            if name in ("ESTOP", "AUTO_ARM", "MANUAL_START", "RC_PRESENT"):
                continue
            if getattr(controllerState, name):
                controllerBits |= 1 << bit
        flagsB = _packFlags(controllerState, CMD_FLAGS_B)
        flagsC = _packFlags(controllerState, CMD_FLAGS_C)
        axes = [getattr(controllerState, name) & 0xFF for name in CMD_AXES]

    return CMD_STRUCT.pack(CMD_TAG, inputBits | controllerBits, flagsB, flagsC, *axes)


def deserializeState(
    message: bytes,
    robotState: RobotState,
    robotStateLock: threading.Lock,
    recvTime: list[float],
):
    try:
        if len(message) != STATUS_STRUCT.size or message[0] != STATUS_TAG:
            return False
        (
            _tag,
            mode,
            batteryLevel,
            batteryMv,
            steeringUs,
            throttleUs,
            rpm,
        ) = STATUS_STRUCT.unpack(message)
        # AutoMode() raises on an out-of-range value, which is the intended
        # rejection path for a frame that passed the tag check but is garbage.
        autoMode = AutoMode(mode)
        with robotStateLock:
            robotState.auto_mode = autoMode
            robotState.battery_level = batteryLevel
            robotState.battery_mv = batteryMv
            robotState.steering_output_us = steeringUs
            robotState.throttle_output_us = throttleUs
            robotState.rpm = rpm
            recvTime[0] = time.monotonic()
            robotState.comms_ok = True
        return True
    except Exception:
        return False


# Receive data from robot
def receiveData(
    xbee: XBee,
    recvTime: list[float],
    robotState: RobotState,
    robotStateMutex: threading.Lock,
    runEvent: threading.Event,
    cycleMonitor: CycleRateMonitor,
):
    oldCB = xbee._callback
    oldTC = xbee._thread_continue
    xbee._callback = True
    # XBee tests this attribute directly rather than calling it.  A lambda is
    # always truthy, so it prevents wait_read_frame() from seeing shutdown.
    xbee._thread_continue = True
    try:
        while runEvent.is_set():
            waitStart = time.monotonic()
            data = xbee.wait_read_frame()
            cycleMonitor.record("comm-receive")
            cycleMonitor.record_duration(
                "comm-receive-wait", time.monotonic() - waitStart
            )
            try:
                if data["id"] == "tx_status":
                    with robotStateMutex:
                        robotState.tx_status = data["status"][0]
                        robotState.tx_ack = robotState.tx_status == 0
                    cycleMonitor.record("comm-tx-ack")
                elif deserializeState(
                    data["rf_data"], robotState, robotStateMutex, recvTime
                ):
                    cycleMonitor.record("comm-receive-valid")
                else:
                    cycleMonitor.record("comm-receive-invalid")
            except Exception:
                cycleMonitor.record("comm-receive-invalid")
    except ThreadQuitException:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        # This prevents weirdness during xbee shutdown
        xbee._thread_continue = oldTC
        xbee._callback = oldCB


def discover_serial_port() -> str | None:
    """Find a likely XBee serial adapter across Linux (/dev/ttyUSB*, /dev/ttyACM*)
    and Windows (COMx) by preferring known adapter descriptions/hwids, falling
    back to the first available port."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None

    for port in ports:
        haystack = f"{port.description} {port.hwid}".lower()
        if any(token in haystack for token in SERIAL_PORT_PREFERRED_TOKENS):
            return port.device

    return ports[0].device


def commControl(
    robotState: RobotState,
    robotStateMutex: threading.Lock,
    controllerState: ControllerState,
    controllerStateMutex: threading.Lock,
    inputState: InputState,
    inputStateMutex: threading.Lock,
    runEvent: threading.Event,
    cycleMonitor: CycleRateMonitor,
):
    m_ser = None
    m_xbee = None
    # Needs to be list so we can pass by reference to the rx thread
    latestReceiveTime = [time.monotonic()]
    thread_read = None
    runReadEvent = threading.Event()
    runReadEvent.clear()

    while runEvent.is_set():
        cycleMonitor.record("comm-discovery")
        portPath = discover_serial_port()
        if portPath is None:
            time.sleep(
                DISCOVERY_INTERVAL
            )  # Only need to notice a reconnect, not react instantly
            continue
        try:
            print("New XBee @", portPath)
            serial_trace_path = os.getenv(SERIAL_TRACE_ENV, SERIAL_TRACE_DEFAULT)
            m_ser = TimestampedSerial(
                serial.Serial(portPath, baudrate=57600, timeout=0.1),
                serial_trace_path,
            )
            if serial_trace_path:
                print("XBee serial trace @", serial_trace_path)
            m_xbee = XBee(m_ser, escaped=True)
            with robotStateMutex:
                robotState.comm_port = portPath

            runReadEvent.set()
            thread_read = threading.Thread(
                target=receiveData,
                args=(
                    m_xbee,
                    latestReceiveTime,
                    robotState,
                    robotStateMutex,
                    runReadEvent,
                    cycleMonitor,
                ),
            )
            thread_read.start()

            while runEvent.is_set():
                startTime = time.monotonic()
                message = serializeState(
                    controllerState, controllerStateMutex, inputState, inputStateMutex
                )
                with robotStateMutex:
                    robotState.tx_message = message
                sendStart = time.monotonic()
                m_xbee.send(
                    "tx",
                    dest_addr=ROBOT_ADDR,
                    frame_id=b"\x01",
                    options=TX_OPT,
                    data=message,
                )
                cycleMonitor.record("comm-send")
                cycleMonitor.record_duration("comm-send", time.monotonic() - sendStart)
                endTime = time.monotonic()
                if endTime - latestReceiveTime[0] >= TIMEOUT_COMMS:
                    with robotStateMutex:
                        robotState.comms_ok = False
                delayTime = max(0.0, 0.05 - (endTime - startTime))
                time.sleep(delayTime)  # Limit loop rate to 20Hz
        except Exception:
            traceback.print_exc()
        finally:
            # xbee.halt() assumes internal threading state that is
            # never set up when we drive wait_read_frame() manually
            # (it raises AttributeError on self._thread), so stop the
            # reader thread ourselves instead.
            runReadEvent.clear()
            if m_xbee is not None:
                # wait_read_frame() polls this boolean and raises
                # ThreadQuitException.  Do this before closing the serial port
                # so the reader never races a closed file descriptor.
                m_xbee._thread_continue = False
            if thread_read is not None:
                thread_read.join()
                thread_read = None
            if m_ser is not None:
                m_ser.close()
                m_ser = None
            m_xbee = None
            with robotStateMutex:
                robotState.comm_port = ""
            time.sleep(
                DISCOVERY_INTERVAL
            )  # Only need to notice a reconnect, not react instantly


def cycleRateControl(cycleMonitor: CycleRateMonitor, runEvent: threading.Event):
    while runEvent.is_set():
        time.sleep(1.0)
        rates = cycleMonitor.rates()
        durations = cycleMonitor.duration_stats()
        durationText = " ".join(
            f"{name}=avg:{average * 1000:.1f}ms,max:{maximum * 1000:.1f}ms"
            for name, (average, maximum) in sorted(durations.items())
        )
        print(
            "Cycle rates (Hz):",
            " ".join(f"{name}={rate:.1f}" for name, rate in sorted(rates.items())),
        )
        print("Cycle durations:", durationText)
