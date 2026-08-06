"""Platform-independent E-Stop/RC logic shared between the Raspberry Pi GPIO
box (estop.py) and the laptop terminal emulator (estop_tui.py).
"""

import dataclasses
import os
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
    battery_level: int = 255
    comms_ok: bool = False
    comm_port: str = ""
    tx_ack: bool = False
    tx_status: int = 255
    tx_message: str = ""


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
PYGAME_HAT_INDEX = 0


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
                            setattr(
                                controllerState, KEYCODE_MAP[event.code], event.value
                            )
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
    return int(round((max(-1.0, min(1.0, value)) + 1.0) * 127.5))


def _scalePygamePressure(value: float) -> int:
    return int(round((max(-1.0, min(1.0, value)) + 1.0) / 2.0 * 255))


def _applyPygameState(joystick, controllerState: ControllerState) -> None:
    numAxes = joystick.get_numaxes()
    for axisIndex, fieldName in PYGAME_AXIS_MAP.items():
        if axisIndex < numAxes:
            setattr(
                controllerState,
                fieldName,
                _scalePygameAxis(joystick.get_axis(axisIndex)),
            )

    numButtons = joystick.get_numbuttons()
    for buttonIndex, fieldName in PYGAME_BUTTON_MAP.items():
        if buttonIndex >= numButtons:
            continue
        pressed = bool(joystick.get_button(buttonIndex))
        setattr(controllerState, fieldName, pressed)
        pressureField = "PRESSURE_" + fieldName.split("_", 1)[1]
        if hasattr(controllerState, pressureField):
            setattr(controllerState, pressureField, 255 if pressed else 0)

    if numButtons <= 13 and joystick.get_numhats() > PYGAME_HAT_INDEX:
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
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print("New controller @", joystick.get_name())
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
                    _applyPygameState(joystick, controllerState)
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


def serializeState(
    controllerState: ControllerState,
    controllerStateLock: threading.Lock,
    inputState: InputState,
    inputStateLock: threading.Lock,
):
    inputStateString = ""
    with inputStateLock:
        estop = False
        if not inputState.estop_override:
            estop = inputState.estop
        inputStateString = ",".join(
            [
                str(int(estop)),
                str(int(inputState.auto_arm)),
                str(int(inputState.manual_start)),
            ]
        )
    controllerStateString = ""
    with controllerStateLock:
        controllerStateString = ",".join(
            [
                str(int(controllerState.comms_ok)),
                str(controllerState.AXIS_LX),
                str(controllerState.AXIS_LY),
                str(controllerState.AXIS_RX),
                str(controllerState.AXIS_RY),
                str(int(controllerState.BUTTON_X)),
                str(int(controllerState.BUTTON_O)),
                str(int(controllerState.BUTTON_SQUARE)),
                str(int(controllerState.BUTTON_TRIANGLE)),
                str(int(controllerState.BUTTON_L1)),
                str(int(controllerState.BUTTON_R1)),
                str(int(controllerState.BUTTON_L2)),
                str(int(controllerState.BUTTON_R2)),
                str(int(controllerState.BUTTON_L3)),
                str(int(controllerState.BUTTON_R3)),
                str(int(controllerState.BUTTON_SELECT)),
                str(int(controllerState.BUTTON_START)),
                str(int(controllerState.BUTTON_PS)),
                str(int(controllerState.BUTTON_UP)),
                str(int(controllerState.BUTTON_RIGHT)),
                str(int(controllerState.BUTTON_DOWN)),
                str(int(controllerState.BUTTON_LEFT)),
            ]
        )

    return ",".join([inputStateString, controllerStateString]) + "\n"


def deserializeState(
    message: bytes,
    robotState: RobotState,
    robotStateLock: threading.Lock,
    recvTime: list[float],
):
    try:
        # Make sure to strip trailing commas
        parts = message.decode("utf-8").strip().strip(",").split(",")
        if len(parts) != 2:
            raise ValueError("Invalid received message length")
        with robotStateLock:
            robotState.auto_mode = AutoMode(int(parts[0]))
            robotState.battery_level = int(parts[1])
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
    xbee._thread_continue = lambda: runEvent.is_set()
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
                serial.Serial(portPath, baudrate=38400, timeout=0.1),
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
            if thread_read is not None:
                thread_read.join(timeout=1.0)
                if thread_read.is_alive():
                    print("Warning: xbee read thread did not exit cleanly")
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
