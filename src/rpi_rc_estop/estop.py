#!/usr/bin/env python3

import atexit
import copy
import math
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Final

import evdev
import lgpio
import serial
from xbee import XBee

ROBOT_ADDR = "\x00\x01"
TX_OPT = "\x01"

TIMEOUT_CONTROLLER = 0.2
TIMEOUT_COMMS = 0.2

LED_ESTOP: Final = 8
LED_ESTOP_SENSE: Final = 10
LED_ESTOP_OVERRIDE: Final = 13
LED_AUTO_ARM: Final = 16
LED_COMMS: Final = 18
LED_MODE_STOP: Final = 19
LED_MODE_RC: Final = 21
LED_BATTERY: Final = 22
LED_MODE_AUTO: Final = 23
INPUT_ESTOP: Final = 5
INPUT_ESTOP_SENSE: Final = 7
INPUT_ESTOP_OVERRIDE: Final = 11
INPUT_AUTO_ARM: Final = 15
INPUT_MANUAL_START: Final = 12

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


def deluminate(leds):
    for led in leds:
        lgpio.gpio_write(m_gpio, led, 0)


def ledControl(
    robotState,
    robotStateMutex,
    inputState,
    inputStateMutex,
    controllerState,
    controllerStateMutex,
    ledState,
    ledStateMutex,
    runEvent,
):
    loopCount = 0

    while runEvent.is_set():
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

        with inputStateMutex:
            if inputState.estop:
                newLedState.estop = LEDMode.ON

            if inputState.estop_sense:
                newLedState.estop_sense = LEDMode.ON

            if inputState.estop_override:
                newLedState.estop_override = LEDMode.ON

            if inputState.auto_arm:
                newLedState.auto_arm = LEDMode.ON

        with controllerStateMutex:
            if not controllerState.comms_ok and newLedState.mode_rc != LEDMode.OFF:
                newLedState.mode_stop = LEDMode.BLINK

        # Set LED illuminations based on LED state
        with ledStateMutex:
            for led in ALL_LEDS:
                state = ledState.__getattribute__(f"led_{led}")
                if state == LEDMode.ON:
                    lgpio.gpio_write(m_gpio, led, 1)
                elif state == LEDMode.OFF:
                    lgpio.gpio_write(m_gpio, led, 0)
                elif state == LEDMode.BLINK:
                    lgpio.gpio_write(m_gpio, led, loopCount % 2 == 0)

        loopCount += 1
        time.sleep(0.25)


# TODO: Implementation
def controllerControl(controllerQueue, controllerState, runEvent):
    latestEvent = time.clock()
    latestMode = None
    ps = None
    # Outer loop handles disconnected controller
    while runEvent.is_set():
        devices = evdev.list_devices()

        if len(devices) > 0:
            print("New controller @", devices[0])
            ps = evdev.InputDevice(devices[0])

        # Inner loop handles normal input events
        while runEvent.is_set() and len(evdev.list_devices()) > 0:
            try:
                for event in ps.read():
                    if event.type != evdev.ecodes.EV_SYN and event.code in KEYCODE_MAP:
                        controllerState[KEYCODE_MAP[event.code]] = event.value

                latestEvent = time.clock()
            except OSError:
                pass

            newMode = LED_ON
            if time.clock() - latestEvent >= TIMEOUT_CONTROLLER:
                # print("Timeout", time.clock() - latestEvent
                controllerState.update(
                    copy.deepcopy(CONTROLLER_ZERO)
                )  # Zero out for safety
                newMode = LED_BLINK

            if newMode != latestMode:
                controllerQueue.put(newMode)
                latestMode = newMode

            time.sleep(0.02)  # Limit loop rate to 50Hz

        newMode = LED_OFF
        controllerState.update(copy.deepcopy(CONTROLLER_ZERO))  # Zero out for safety

        if newMode != latestMode:
            # print("LOST_CONTROLLER")
            controllerQueue.put(newMode)
            latestMode = newMode

        time.sleep(0.1)  # Limit discovery loop rate to 10Hz


# TODO: Implementation
# Receive Acks from Arduino
def receiveData(xbee, ackTime, runEvent):
    oldCB = xbee._callback
    oldTC = xbee._thread_continue
    xbee._callback = True
    xbee._thread_continue = lambda: runEvent.is_set()
    try:
        while runEvent.is_set():
            data = xbee.wait_read_frame()
            ackTime[0] = time.clock()
    except Exception:
        pass  # Exception will be raised on event interrupt
    finally:
        # This prevents weirdness during xbee shutdown
        xbee._thread_continue = oldTC
        xbee._callback = oldCB


# TODO: Implementation
def serializeState(controllerState):
    buttonBytes = "\x00" * int(math.ceil(float(len(SERIAL_BUTTONS)) / 8.0))
    axisBytes = "\x00" * len(SERIAL_AXES)

    buttonBytes = list(buttonBytes)
    axisBytes = list(axisBytes)

    for btn in range(len(SERIAL_BUTTONS)):
        byteNum = int(math.floor(btn / 8.0))
        bitNum = btn % 8
        val = controllerState[SERIAL_BUTTONS[btn]]
        buttonBytes[byteNum] = chr(ord(buttonBytes[byteNum]) | (val << bitNum))

    for axs in range(len(SERIAL_AXES)):
        val = controllerState[SERIAL_AXES[axs]]
        axisBytes[axs] = chr(val)

    buttonBytes = "".join(buttonBytes)
    axisBytes = "".join(axisBytes)

    return buttonBytes + axisBytes


# TODO: Implementation
def commControl(robotState, robotStateMutex, runEvent):
    m_ser = None
    m_xbee = None
    latestMode = LED_OFF
    latestAck = [0]
    commsQueue.put(latestMode)
    thread_read = None
    runReadEvent = threading.Event()
    runReadEvent.clear()

    while runEvent.is_set():
        serDevs = [f for f in os.listdir("/dev") if "ttyUSB" in f]
        if len(serDevs) == 0:
            time.sleep(0.1)  # Limit discovery loop rate to 10Hz
            continue
        try:
            print("New XBee @", serDevs[0])
            m_ser = serial.Serial("/dev/" + serDevs[0])
            m_xbee = XBee(m_ser, escaped=True)

            runReadEvent.set()
            thread_read = threading.Thread(
                target=receiveData, args=(m_xbee, latestAck, runReadEvent)
            )
            thread_read.start()

            while runEvent.is_set():
                message = serializeState(controllerState)
                m_xbee.send(
                    "tx",
                    dest_addr=ROBOT_ADDR,
                    frame_id="\x00",
                    options=TX_OPT,
                    data=message,
                )

                newMode = LED_ON
                if time.clock() - latestAck[0] >= TIMEOUT_COMMS:
                    newMode = LED_BLINK

                if newMode != latestMode:
                    commsQueue.put(newMode)
                    latestMode = newMode

                time.sleep(0.05)  # Limit loop rate to 20Hz
            runReadEvent.clear()
            m_xbee.halt()
        except Exception:
            # traceback.print_exc()
            newMode = LED_OFF
            if newMode != latestMode:
                commsQueue.put(newMode)
                latestMode = newMode
            if thread_read != None:
                runReadEvent.clear()
            # if(m_xbee != None and m_xbee.isAlive):
            #     m_xbee.halt()
            time.sleep(0.1)  # Limit discovery loop rate to 10Hz


def inputControl(inputState, inputMutex, runEvent):
    while runEvent.is_set():
        with inputMutex:
            inputState.estop = m_gpio.gpio_read(m_gpio, INPUT_ESTOP)
            inputState.estop_sense = m_gpio.gpio_read(m_gpio, INPUT_ESTOP_SENSE)
            inputState.estop_override = m_gpio.gpio_read(m_gpio, INPUT_ESTOP_OVERRIDE)
            inputState.auto_arm = m_gpio.gpio_read(m_gpio, INPUT_AUTO_ARM)
            inputState.manual_start = m_gpio.gpio_read(m_gpio, INPUT_MANUAL_START)

        time.sleep(0.05)  # Limit loop rate to 20Hz


runEvent = threading.Event()
runEvent.set()

for led in ACTIVE_HIGH_LEDS:
    m_gpio.gpio_claim_output(m_gpio, led)

for led in ACTIVE_LOW_LEDS:
    m_gpio.gpio_claim_output(m_gpio, led, lFlags=lgpio.SET_ACTIVE_LOW)

for input in PULL_UP_INPUTS:
    m_gpio.gpio_claim_input(m_gpio, input, lFlags=lgpio.SET_PULL_UP)

for input in PULL_DOWN_INPUTS:
    m_gpio.gpio_claim_input(
        m_gpio, input, lFlags=lgpio.SET_PULL_DOWN | lgpio.SET_ACTIVE_LOW
    )

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
    ),
)
thread_controllerControl = threading.Thread(
    target=controllerControl, args=(m_controllerQueue, m_controllerState, runEvent)
)
thread_commControl = threading.Thread(
    target=commControl, args=(m_robotState, m_robotMutex, runEvent)
)
thread_inputControl = threading.Thread(
    target=inputControl, args=(m_inputState, m_inputMutex, runEvent)
)

thread_ledControl.start()
thread_controllerControl.start()
thread_commControl.start()
thread_inputControl.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    runEvent.clear()
    thread_commControl.join()
    thread_controllerControl.join()
    thread_ledControl.join()
    thread_inputControl.join()

if threading.active_count() > 1:
    print("STILL RUNNING THREADS:", threading.enumerate())
