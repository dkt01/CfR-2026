#!/usr/bin/env python3
"""Laptop terminal emulator for the E-Stop/RC offboard interface.

Emulates the GPIO LEDs and switch inputs of estop.py in a terminal UI, reusing
estop_core for the XBee comms, gamepad polling, and wire protocol so the robot
sees no difference between the physical box and this emulator.
"""

import threading
import time

import readchar
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

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
    controllerControlPygame,
)

# Emulates a momentary/held key via a refresh timeout, mirroring TIMEOUT_CONTROLLER,
# since terminal input has no key-up event to detect release directly.
MANUAL_START_TIMEOUT = 0.2
# 6Hz is comfortably above the Nyquist rate for a 2Hz blink (4 transitions/sec)
# while cutting full-screen redraw CPU cost vs. the previous 10Hz.
REFRESH_HZ = 6
RENDER_INTERVAL = 1 / REFRESH_HZ
BLINK_HZ = 2.0
TUI_TITLE = "CfR E-Stop / RC Emulator"

LED_COLORS = {
    "estop": "red",
    "estop_sense": "yellow",
    "estop_override": "red",
    "auto_arm": "blue",
    "comms": "white",
    "mode_stop": "red",
    "mode_rc": "green",
    "mode_auto": "blue",
    "battery": "white",
}

KEY_BINDINGS_HELP = (
    "space=SET ESTOP  c=CLEAR ESTOP  backspace=TOGGLE SENSE  `=TOGGLE OVERRIDE  "
    "a=TOGGLE AUTO ARM  enter=MANUAL START  q=QUIT"
)


def _blink_on() -> bool:
    # Two transitions per cycle, so this flips at 2x BLINK_HZ.
    return int(time.monotonic() * BLINK_HZ * 2) % 2 == 0


def render_leds(ledState: LEDState) -> Table:
    table = Table(title="LEDs", show_header=False, box=None)
    for name, color in LED_COLORS.items():
        mode = getattr(ledState, name)
        if mode == LEDMode.ON:
            glyph = "\u25cf"  # ●
        elif mode == LEDMode.BLINK:
            glyph = "\u25cf" if _blink_on() else "\u25cb"  # ●/○
        else:
            glyph = "\u25cb"  # ○
        table.add_row(name.replace("_", " ").upper(), f"[{color}]{glyph}[/{color}]")
    return table


def render_inputs(inputState: InputState) -> Table:
    table = Table(title="Inputs", show_header=False, box=None)

    def toggleSwitch(value: bool, color: str) -> str:
        return (
            f"[{color}]\u25a0 ON[/{color}]" if value else "[grey58]\u25a1 off[/grey58]"
        )

    table.add_row("ESTOP", toggleSwitch(inputState.estop, "red"))
    table.add_row("ESTOP OVERRIDE", toggleSwitch(inputState.estop_override, "red"))
    table.add_row("AUTO ARM", toggleSwitch(inputState.auto_arm, "blue"))

    # Physically a sensed cable connection, not a switch, so it gets a distinct glyph.
    senseText = (
        "[green]\u2b23 CONNECTED[/green]"
        if inputState.estop_sense
        else "[red]\u2715 UNPLUGGED[/red]"
    )
    table.add_row("ESTOP SENSE", senseText)

    # Push button: only ever shown highlighted while transiently active.
    manualStartText = (
        "[yellow]\u25cf PRESSED[/yellow]"
        if inputState.manual_start
        else "[grey58]\u25cb idle[/grey58]"
    )
    table.add_row("MANUAL START", manualStartText)
    return table


def render_status(robotState: RobotState, controllerState: ControllerState) -> Table:
    table = Table(title="Status", show_header=False, box=None)

    def connection(ok: bool) -> str:
        return "[green]CONNECTED[/green]" if ok else "[red]DISCONNECTED[/red]"

    table.add_row("XBEE", connection(robotState.comms_ok))
    table.add_row("", f"[grey58]{robotState.comm_port or '-'}[/grey58]")
    ackText = "ACK" if robotState.tx_ack else f"NACK ({robotState.tx_status:#04x})"
    ackColor = "green" if robotState.tx_ack else "red"
    table.add_row("TX", f"[{ackColor}]{ackText}[/{ackColor}]")
    table.add_row("TX DATA", repr(robotState.tx_message))
    table.add_row("STEERING OUT", f"{robotState.steering_output_us} us")
    table.add_row("THROTTLE OUT", f"{robotState.throttle_output_us} us")
    table.add_row("CONTROLLER", connection(controllerState.comms_ok))
    table.add_row("", f"[grey58]{controllerState.device_name or '-'}[/grey58]")
    return table


def render_cycle_rates(cycleMonitor: CycleRateMonitor) -> Table:
    table = Table(title="Cycle Rates (Hz)", show_header=False, box=None)
    rates = cycleMonitor.rates()
    for name, rate in sorted(rates.items()):
        table.add_row(name, f"{rate:.1f}")
    return table


def render(
    robotState: RobotState,
    inputState: InputState,
    controllerState: ControllerState,
    cycleMonitor: CycleRateMonitor,
) -> Panel:
    ledState = compute_led_state(robotState, inputState, controllerState)
    grid = Table.grid(padding=(0, 4))
    grid.add_row(
        render_leds(ledState),
        render_inputs(inputState),
        render_status(robotState, controllerState),
        render_cycle_rates(cycleMonitor),
    )
    return Panel(grid, title=TUI_TITLE, subtitle=KEY_BINDINGS_HELP)


def render_quit_confirm() -> Panel:
    return Panel(
        "Quit? [bold]y[/bold] = yes, any other key = cancel",
        title="Confirm Exit",
        style="yellow",
    )


def keyboardControl(
    inputState: InputState,
    inputMutex: threading.Lock,
    lastEnterTime: list[float],
    runEvent: threading.Event,
    quitConfirmEvent: threading.Event,
    cycleMonitor: CycleRateMonitor,
):
    # This thread is the sole reader of stdin for the process's lifetime; the
    # quit confirmation is answered through this same loop (see below) rather
    # than a second blocking input() call, to avoid two threads fighting over
    # terminal raw/cooked mode at once.
    awaitingQuitConfirm = False

    while runEvent.is_set():
        key = readchar.readkey()
        cycleMonitor.record("keyboard")

        if awaitingQuitConfirm:
            awaitingQuitConfirm = False
            quitConfirmEvent.clear()
            if key in ("y", "Y"):
                runEvent.clear()
                return
            continue

        if key == readchar.key.SPACE:
            with inputMutex:
                inputState.estop = True
        elif key in ("c", "C"):
            with inputMutex:
                inputState.estop = False
        elif key == readchar.key.BACKSPACE:
            with inputMutex:
                inputState.estop_sense = not inputState.estop_sense
        elif key == "`":
            with inputMutex:
                inputState.estop_override = not inputState.estop_override
        elif key in ("a", "A"):
            with inputMutex:
                inputState.auto_arm = not inputState.auto_arm
        elif key == readchar.key.ENTER:
            lastEnterTime[0] = time.monotonic()
            with inputMutex:
                inputState.manual_start = True
        elif key in ("q", "Q"):
            awaitingQuitConfirm = True
            quitConfirmEvent.set()


def renderLoop(
    robotState: RobotState,
    robotStateMutex: threading.Lock,
    inputState: InputState,
    inputStateMutex: threading.Lock,
    controllerState: ControllerState,
    controllerStateMutex: threading.Lock,
    lastEnterTime: list[float],
    runEvent: threading.Event,
    quitConfirmEvent: threading.Event,
    cycleMonitor: CycleRateMonitor,
):
    console = Console()
    console.set_window_title(TUI_TITLE)

    # Starts False so a not-yet-discovered controller at launch doesn't read as
    # a disconnect and slam the estop on before the operator has done anything.
    controllerWasConnected = False

    try:
        with Live(console=console, screen=True, refresh_per_second=REFRESH_HZ) as live:
            while runEvent.is_set():
                cycleMonitor.record("render")
                with inputStateMutex:
                    if (
                        inputState.manual_start
                        and time.monotonic() - lastEnterTime[0] >= MANUAL_START_TIMEOUT
                    ):
                        inputState.manual_start = False

                with controllerStateMutex:
                    controllerConnected = controllerState.comms_ok
                if controllerWasConnected and not controllerConnected:
                    with inputStateMutex:
                        inputState.estop = True
                controllerWasConnected = controllerConnected

                if quitConfirmEvent.is_set():
                    panel = render_quit_confirm()
                else:
                    with robotStateMutex, inputStateMutex, controllerStateMutex:
                        panel = render(
                            robotState, inputState, controllerState, cycleMonitor
                        )

                live.update(panel)
                time.sleep(RENDER_INTERVAL)
    finally:
        console.set_window_title("")


def main():
    inputState = InputState()
    inputState.estop = True
    inputState.estop_sense = True
    controllerState = ControllerState()
    robotState = RobotState()

    inputMutex = threading.Lock()
    controllerMutex = threading.Lock()
    robotMutex = threading.Lock()

    lastEnterTime = [0.0]
    cycleMonitor = CycleRateMonitor(CYCLE_RATE_WINDOW)
    runEvent = threading.Event()
    runEvent.set()
    quitConfirmEvent = threading.Event()

    threadController = threading.Thread(
        target=controllerControlPygame,
        args=(controllerState, controllerMutex, runEvent, cycleMonitor),
    )
    threadComm = threading.Thread(
        target=commControl,
        args=(
            robotState,
            robotMutex,
            controllerState,
            controllerMutex,
            inputState,
            inputMutex,
            runEvent,
            cycleMonitor,
        ),
    )
    threadKeyboard = threading.Thread(
        target=keyboardControl,
        args=(
            inputState,
            inputMutex,
            lastEnterTime,
            runEvent,
            quitConfirmEvent,
            cycleMonitor,
        ),
    )

    threadController.start()
    threadComm.start()
    threadKeyboard.start()

    renderLoop(
        robotState,
        robotMutex,
        inputState,
        inputMutex,
        controllerState,
        controllerMutex,
        lastEnterTime,
        runEvent,
        quitConfirmEvent,
        cycleMonitor,
    )

    threadComm.join()
    threadController.join()
    threadKeyboard.join(timeout=1.0)

    if threading.active_count() > 1:
        print("STILL RUNNING THREADS:", threading.enumerate())


if __name__ == "__main__":
    main()
