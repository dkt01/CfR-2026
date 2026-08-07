#!/usr/bin/env python3

import atexit
import threading
import time
from typing import Final

import lgpio

from estop_core import (
    CYCLE_RATE_WINDOW,
    ControllerState,
    CycleRateMonitor,
    InputState,
    LEDMode,
    LEDState,
    RobotState,
    commControl,
    compute_led_state,
    controllerControlEvdev,
    cycleRateControl,
)

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
LED_MODE_AUTO: Final = RpiB2PinMap[23]
LED_BATTERY: Final = RpiB2PinMap[22]
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

m_gpio = lgpio.gpiochip_open(0)
m_inputState = InputState()
m_ledState = LEDState()
m_controllerState = ControllerState()
m_robotState = RobotState()

m_inputMutex = threading.Lock()
m_ledMutex = threading.Lock()
m_controllerMutex = threading.Lock()
m_robotMutex = threading.Lock()


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

        with robotStateMutex, inputStateMutex, controllerStateMutex:
            newLedState = compute_led_state(robotState, inputState, controllerState)

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

        loopCount += 1
        time.sleep(0.25)


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
    target=controllerControlEvdev,
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
