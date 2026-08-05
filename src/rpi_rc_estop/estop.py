#!/usr/bin/env python3

import atexit
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Final

import evdev
import lgpio
import serial
from xbee import XBee
from xbee.thread.base import ThreadQuitException

ROBOT_ADDR = "\x00\x01"
TX_OPT = "\x01"

TIMEOUT_CONTROLLER = 0.2
TIMEOUT_COMMS = 0.2
CYCLE_RATE_WINDOW = 5.0
SERIAL_TRACE_ENV = "ESTOP_SERIAL_TRACE"
# SERIAL_TRACE_DEFAULT = "/tmp/xbee-serial.log"
SERIAL_TRACE_DEFAULT = None


# key: physical pin number, value: GPIO number
# https://www.raspberry-pi-geek.com/howto/GPIO-Pinout-Rasp-Pi-1-Rev1-and-Rev2
RpiB2PinMap = {
    3: 2,
    5: 3,
    7: 4,
    8: 14,
    10: 15,
    11: 17,
    12: 18,
    13: 27,
    15: 22,
    16: 23,
    18: 24,
    19: 10,
    21: 9,
    22: 25,
    23: 11,
    24: 8,
    26: 7,
}

LED_ESTOP: Final = RpiB2PinMap[8]
LED_ESTOP_SENSE: Final = RpiB2PinMap[10]
LED_ESTOP_OVERRIDE: Final = RpiB2PinMap[13]
LED_AUTO_ARM: Final = RpiB2PinMap[16]
LED_COMMS: Final = RpiB2PinMap[18]
LED_MODE_STOP: Final = RpiB2PinMap[19]
LED_MODE_RC: Final = RpiB2PinMap[21]
LED_BATTERY: Final = RpiB2PinMap[22]
LED_MODE_AUTO: Final = RpiB2PinMap[23]
INPUT_ESTOP: Final = RpiB2PinMap[5]
INPUT_ESTOP_SENSE: Final = RpiB2PinMap[7]
INPUT_ESTOP_OVERRIDE: Final = RpiB2PinMap[11]
INPUT_AUTO_ARM: Final = RpiB2PinMap[15]
INPUT_MANUAL_START: Final = RpiB2PinMap[12]

ACTIVE_HIGH_LEDS = [
    LED_ESTOP,
    LED_ESTOP_SENSE,
    LED_COMMS,
    LED_MODE_STOP,
    LED_MODE_RC,
    LED_BATTERY,
    LED_MODE_AUTO,
]
ACTIVE_LOW_LEDS = [LED_ESTOP_OVERRIDE, LED_AUTO_ARM]
ALL_LEDS = ACTIVE_HIGH_LEDS + ACTIVE_LOW_LEDS
PULL_UP_INPUTS = [INPUT_ESTOP, INPUT_ESTOP_SENSE, INPUT_MANUAL_START]
PULL_DOWN_INPUTS = [INPUT_ESTOP_OVERRIDE, INPUT_AUTO_ARM]

# key: physical pin number, value: GPIO number
# https://www.raspberry-pi-geek.com/howto/GPIO-Pinout-Rasp-Pi-1-Rev1-and-Rev2
RpiB2PinMap = {
    3: 2,
    5: 3,
    7: 4,
    8: 14,
    10: 15,
    11: 17,
    12: 18,
    13: 27,
    15: 22,
    16: 23,
    18: 24,
    19: 10,
    21: 9,
    22: 25,
    23: 11,
    24: 8,
    26: 7,
}


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


m_gpio = lgpio.gpiochip_open(0)
m_inputState = InputState()
m_ledState = LEDState()
m_controllerState = ControllerState()
m_robotState = RobotState()

m_inputMutex = threading.Lock()
m_ledMutex = threading.Lock()
m_controllerMutex = threading.Lock()
m_robotMutex = threading.Lock()


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
            return {
                name: len(timestamps) / self.window
                for name, timestamps in self.cycles.items()
            }

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


def deluminate(leds):
    for led in leds:
        lgpio.gpio_write(m_gpio, led, 0)


def ledControl(
    robotState: RobotState,
    robotStateMutex: threading.Lock,
    inputState: InputState,
    inputStateMutex: threading.Lock,
    controllerState: ControllerState,
    controllerStateMutex: threading.Lock,
    ledState: LEDState,
    ledStateMutex: threading.Lock,
    runEvent: threading.Event,
    cycleMonitor: CycleRateMonitor,
):
    loopCount = 0

    while runEvent.is_set():
        cycleMonitor.record("led")
        # Update LED states first
        newLedState = LEDState()

        with robotStateMutex:
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

            if robotState.battery_level < 128:
                newLedState.battery = LEDMode.BLINK
            elif robotState.battery_level < 64:
                newLedState.battery = LEDMode.OFF
            else:
                newLedState.battery = LEDMode.ON

            if robotState.comms_ok:
                newLedState.comms = LEDMode.ON
            else:
                newLedState.comms = LEDMode.BLINK
            # print("Robot state:", robotState)

        with inputStateMutex:
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

            # print("Input state:", inputState)

        with controllerStateMutex:
            if not controllerState.comms_ok and newLedState.mode_rc != LEDMode.OFF:
                newLedState.mode_stop = LEDMode.BLINK
            # print("Controller state:", controllerState)

        # Set LED illuminations based on LED state
        with ledStateMutex:
            ledState = newLedState
            for led_label in ledState.__dataclass_fields__.keys():
                constant_name = f"LED_{led_label.upper()}"
                led = globals().get(constant_name)
                state = getattr(ledState, led_label)
                if state == LEDMode.ON:
                    lgpio.gpio_write(m_gpio, led, 1)
                elif state == LEDMode.OFF:
                    lgpio.gpio_write(m_gpio, led, 0)
                elif state == LEDMode.BLINK:
                    lgpio.gpio_write(m_gpio, led, loopCount % 2 == 0)
            # print(ledState)

        loopCount += 1
        time.sleep(0.25)


def controllerControl(
    controllerState: ControllerState,
    controllerMutex: threading.Lock,
    runEvent: threading.Event,
    cycleMonitor: CycleRateMonitor,
):
    latestEvent = time.monotonic()
    ps = None
    # Outer loop handles disconnected controller
    while runEvent.is_set():
        cycleMonitor.record("controller-discovery")
        devices = evdev.list_devices()

        if len(devices) > 0:
            print("New controller @", devices[0])
            ps = evdev.InputDevice(devices[0])

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
                    controllerState = ControllerState()  # Zero out for safety

            time.sleep(0.02)  # Limit loop rate to 50Hz

        controllerState = ControllerState()  # Zero out for safety

        time.sleep(0.1)  # Limit discovery loop rate to 10Hz


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
    message: str, robotState: RobotState, robotStateLock: threading.Lock
):
    try:
        # Make sure to strip trailing commas
        parts = message.decode("utf-8").strip().strip(",").split(",")
        if len(parts) != 2:
            raise ValueError("Invalid received message length")
        with robotStateLock:
            robotState.auto_mode = AutoMode(int(parts[0]))
            robotState.battery_level = int(parts[1])
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
                if deserializeState(data["rf_data"], robotState, robotStateMutex):
                    cycleMonitor.record("comm-receive-valid")
                    recvTime[0] = time.monotonic()
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
        serDevs = [f for f in os.listdir("/dev") if "ttyUSB" in f]
        if len(serDevs) == 0:
            time.sleep(0.1)  # Limit discovery loop rate to 10Hz
            continue
        try:
            print("New XBee @", serDevs[0])
            serial_trace_path = os.getenv(SERIAL_TRACE_ENV, SERIAL_TRACE_DEFAULT)
            m_ser = TimestampedSerial(
                serial.Serial("/dev/" + serDevs[0], baudrate=38400, timeout=0.1),
                serial_trace_path,
            )
            if serial_trace_path:
                print("XBee serial trace @", serial_trace_path)
            m_xbee = XBee(m_ser, escaped=True)

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
                sendStart = time.monotonic()
                m_xbee.send(
                    "tx",
                    dest_addr=ROBOT_ADDR,
                    frame_id="\x00",
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
            time.sleep(0.1)  # Limit discovery loop rate to 10Hz


def inputControl(
    inputState: InputState,
    inputMutex: threading.Lock,
    runEvent: threading.Event,
    cycleMonitor: CycleRateMonitor,
):
    while runEvent.is_set():
        cycleMonitor.record("input")
        with inputMutex:
            inputState.estop = lgpio.gpio_read(m_gpio, INPUT_ESTOP)
            inputState.estop_sense = lgpio.gpio_read(m_gpio, INPUT_ESTOP_SENSE)
            inputState.estop_override = lgpio.gpio_read(m_gpio, INPUT_ESTOP_OVERRIDE)
            inputState.auto_arm = lgpio.gpio_read(m_gpio, INPUT_AUTO_ARM)
            inputState.manual_start = lgpio.gpio_read(m_gpio, INPUT_MANUAL_START)

        time.sleep(0.05)  # Limit loop rate to 20Hz


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


cycleMonitor = CycleRateMonitor(CYCLE_RATE_WINDOW)
runEvent = threading.Event()
runEvent.set()

latestPin = 0
try:
    for led in ACTIVE_HIGH_LEDS:
        latestPin = led
        lgpio.gpio_claim_output(m_gpio, led)

    for led in ACTIVE_LOW_LEDS:
        latestPin = led
        lgpio.gpio_claim_output(m_gpio, led, lFlags=lgpio.SET_ACTIVE_LOW)

    for input in PULL_UP_INPUTS:
        latestPin = input
        lgpio.gpio_claim_input(m_gpio, input, lFlags=lgpio.SET_PULL_UP)

    for input in PULL_DOWN_INPUTS:
        latestPin = input
        lgpio.gpio_claim_input(
            m_gpio, input, lFlags=lgpio.SET_PULL_DOWN | lgpio.SET_ACTIVE_LOW
        )
except lgpio.error as e:
    print("Exception:", e)
    print("pin:", latestPin)


atexit.register(deluminate, ALL_LEDS)

thread_ledControl = threading.Thread(
    target=ledControl,
    args=(
        m_robotState,
        m_robotMutex,
        m_inputState,
        m_inputMutex,
        m_controllerState,
        m_controllerMutex,
        m_ledState,
        m_ledMutex,
        runEvent,
        cycleMonitor,
    ),
)
thread_controllerControl = threading.Thread(
    target=controllerControl,
    args=(m_controllerState, m_controllerMutex, runEvent, cycleMonitor),
)
thread_commControl = threading.Thread(
    target=commControl,
    args=(
        m_robotState,
        m_robotMutex,
        m_controllerState,
        m_controllerMutex,
        m_inputState,
        m_inputMutex,
        runEvent,
        cycleMonitor,
    ),
)
thread_inputControl = threading.Thread(
    target=inputControl, args=(m_inputState, m_inputMutex, runEvent, cycleMonitor)
)
thread_cycleRateControl = threading.Thread(
    target=cycleRateControl, args=(cycleMonitor, runEvent)
)

thread_ledControl.start()
thread_controllerControl.start()
thread_commControl.start()
thread_inputControl.start()
thread_cycleRateControl.start()

try:
    while True:
        cycleMonitor.record("main")
        time.sleep(0.1)
except KeyboardInterrupt:
    runEvent.clear()
    thread_cycleRateControl.join()
    thread_commControl.join()
    thread_controllerControl.join()
    thread_ledControl.join()
    thread_inputControl.join()

if threading.active_count() > 1:
    print("STILL RUNNING THREADS:", threading.enumerate())
