#!/usr/bin/env python3
"""Check that start_signal_detector_node really sees the signal turn, in sim.

The unit tests cover the decision against synthetic frames.  What they cannot
cover is the half of this that lives in the world: that Gazebo renders the arms
in the hues the bands are drawn around, at the pixel the geometry predicts, and
that the detector latches inside the second the arm takes to turn.  So this
drives a running simulation -- red, turn it green, wait for the trigger -- and
prints the frames it took to get there.

Needs a course up with the camera rendered and the randomizer running, which
is what either course launch file gives:

    LIBGL_ALWAYS_SOFTWARE=1 ros2 launch cfr_arduino_bridge speed_course.launch.py \\
        sensors:=true
    ./scripts/check_start_signal.py

Exits non-zero if the detector misses the turn, calls it before the signal
moves, or never sees a frame -- so it can be run after touching either the
thresholds or the worlds.
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from cfr_interfaces.msg import StartSignal

DETECTOR = "/start_signal_detector"
RANDOMIZER = "/obstacle_randomizer"

STATE_NAMES = {
    StartSignal.UNKNOWN: "unknown",
    StartSignal.RED: "red",
    StartSignal.GREEN: "green",
}


def stamp_seconds(header) -> float:
    return header.stamp.sec + header.stamp.nanosec / 1e9


class Checker(Node):
    def __init__(self) -> None:
        super().__init__("check_start_signal")
        self.states: list[StartSignal] = []
        self.images = 0
        self.go = None

        self.create_subscription(
            StartSignal, f"{DETECTOR}/state", self.states.append, 10
        )
        self.create_subscription(
            Bool,
            f"{DETECTOR}/go",
            self.on_go,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # Subscribed only to tell "the camera is not publishing" apart from
        # "the detector is not running", which are different things to fix.
        self.create_subscription(
            Image,
            "/zed/zed_node/left/image_rect_color",
            self.on_image,
            qos_profile_sensor_data,
        )
        self.reset = self.create_client(Trigger, f"{DETECTOR}/reset")
        self.signal = self.create_client(SetBool, f"{RANDOMIZER}/start_signal")

    def on_go(self, message: Bool) -> None:
        self.go = message.data

    def on_image(self, _message: Image) -> None:
        self.images += 1

    # ------------------------------------------------------------------ waits

    def wait(self, predicate, timeout: float, description: str) -> bool:
        """Spin until `predicate` holds, and say what was missing if it does not."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        print(f"    timed out after {timeout:.0f} s waiting for {description}")
        return False

    def call(self, client, request, timeout: float):
        """A service call that keeps spinning, since the sweep takes a second."""
        if not client.wait_for_service(timeout_sec=timeout):
            print(f"    {client.srv_name} is not there")
            return None
        future = client.call_async(request)
        if not self.wait(future.done, timeout, f"{client.srv_name} to answer"):
            return None
        return future.result()

    def latest(self) -> StartSignal | None:
        return self.states[-1] if self.states else None

    def settled(self, state: int, frames: int = 2) -> bool:
        """Whether the last `frames` states all read `state`."""
        if len(self.states) < frames:
            return False
        return all(message.state == state for message in self.states[-frames:])

    def watching(self) -> bool:
        """Whether the detector has found the signal and is reading it as red.

        Not merely "a frame read red": the detector holds out for a place in
        the image that has been red for arm_frames before it will take it for
        the signal, and only then can a green start the run.  Turning the
        signal before that is a test of nothing, so this is what the check
        waits on.
        """
        latest = self.latest()
        return latest is not None and latest.armed and self.settled(StartSignal.RED)


def report(states: list[StartSignal], since: float) -> None:
    """The frame-by-frame table, in the shape the README quotes it in."""
    print(
        f"    {'t(s)':>8} {'state':>8} {'red px':>8} {'green px':>9} {'armed':>6}  go"
    )
    for message in states:
        elapsed = stamp_seconds(message.header) - since
        print(
            f"    {elapsed:8.2f} {STATE_NAMES[message.state]:>8} "
            f"{message.red_pixels:8d} {message.green_pixels:9d} "
            f"{'yes' if message.armed else 'no':>6}"
            f"  {'GO' if message.go else ''}"
        )


def round_trip(checker: Checker, timeout: float) -> bool:
    """One red-to-green start, from a detector that has been reset."""
    if not checker.call(checker.reset, Trigger.Request(), timeout):
        return False
    if not checker.call(checker.signal, SetBool.Request(data=False), timeout):
        return False

    checker.states.clear()
    if not checker.wait(
        checker.watching, timeout, "the detector to find the signal reading red"
    ):
        if not checker.images:
            print("    no camera frames at all -- is sensors:=true?")
        elif checker.latest() is not None:
            print(f"    the detector is seeing {STATE_NAMES[checker.latest().state]}")
            report(checker.states[-6:], stamp_seconds(checker.states[0].header))
        return False
    red_frames = len(checker.states)

    if checker.go:
        print("    ~/go was already up before the signal turned")
        return False

    began = stamp_seconds(checker.states[-1].header)
    latest = checker.latest()
    print(
        f"    signal found at ({latest.lock_x:.0f}, {latest.lock_y:.0f}) and read "
        f"red after {red_frames} frames; turning it green"
    )
    future = checker.signal.call_async(SetBool.Request(data=True))
    started = checker.wait(lambda: checker.go, timeout, "~/go")
    checker.wait(future.done, timeout, "the randomizer to finish the sweep")

    report(checker.states[max(0, red_frames - 2) :], began)
    if not started:
        return False

    latched = next(message for message in checker.states if message.go)
    print(
        f"    go {stamp_seconds(latched.header) - began:.2f} s after the sweep was "
        f"commanded, at ({latched.x:.0f}, {latched.y:.0f}) "
        f"on {latched.green_pixels} px of green"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="starts to check; more than one proves ~/reset works (default 2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds to wait on any one step; a software-rendered world is "
        "slow (default 60)",
    )
    arguments = parser.parse_args()

    rclpy.init()
    checker = Checker()
    try:
        passed = 0
        for round_number in range(1, arguments.rounds + 1):
            print(f"round {round_number} of {arguments.rounds}")
            passed += round_trip(checker, arguments.timeout)
        print(
            f"\n{passed} of {arguments.rounds} starts detected, "
            f"from {checker.images} camera frames"
        )
        return 0 if passed == arguments.rounds else 1
    finally:
        checker.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
