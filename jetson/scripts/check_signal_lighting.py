#!/usr/bin/env python3
"""Measure how much weather the start signal detector will survive.

The course is outdoors and the run may be at any hour, so the question this
answers is not "does the detector work" -- `check_start_signal.py` answers
that -- but "how far from the light it was tuned in does it still work, and
what is in frame that could start the run without the signal".

It takes two real frames from the camera, one with the signal red and one with
it green, and replays them through the decision offline under light they were
not taken in.  The cases are derived from the arm's own measured color rather
than picked: exposure is solved for the chroma it would leave in the arm, and
glare for the saturation it would leave, so the same cases mean the same thing
against a dim rendering and against a signal in daylight.  Then it puts people
in frame -- patches of the signal's own red and green, sized as somebody a few
meters away -- and checks both that they do not stop a start and that they
cannot cause one.

Run it against a simulation, where it measures the detector against a rendered
signal:

    LIBGL_ALWAYS_SOFTWARE=1 ros2 launch cfr_arduino_bridge obstacle_course.launch.py \\
        sensors:=true
    ./scripts/check_signal_lighting.py

or against the car on the course, where it measures it against the real
signal in the real light, which is the number that matters:

    ./scripts/check_signal_lighting.py --spin-by-hand

Exits non-zero if a start that has to work does not, or if anything that must
not start the run does.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool

RANDOMIZER = "/obstacle_randomizer"
IMAGE_TOPIC = "/zed/zed_node/left/image_rect_color"


def decision_module():
    """The detector's ROS-free decision, from the workspace or the source tree.

    Imported by path rather than as a package because that is how the node
    imports it too: it is installed beside the node in lib/, not onto the
    Python path.
    """
    candidates = []
    try:
        from ament_index_python.packages import get_package_prefix

        candidates.append(
            Path(get_package_prefix("cfr_arduino_bridge"))
            / "lib"
            / "cfr_arduino_bridge"
            / "start_signal_detector.py"
        )
    except Exception:  # noqa: BLE001 -- not built, or not sourced
        pass
    candidates.append(
        Path(__file__).parents[1]
        / "cfr_arduino_bridge"
        / "src"
        / "start_signal_detector.py"
    )
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("start_signal_detector", path)
            module = importlib.util.module_from_spec(spec)
            # Registered before it is executed, because dataclasses looks its
            # own module up by name while building each class.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise SystemExit(f"cannot find start_signal_detector.py; looked in {candidates}")


sd = decision_module()


# ----------------------------------------------------------------- the light


def under(frame: np.ndarray, gain: float = 1.0, glare: float = 0.0) -> np.ndarray:
    """A frame under different light: scaled, then washed with flat light.

    The two effects are the two that cost the detector anything, and they cost
    it different things.  `gain` is how much light there is -- cloud, dusk,
    direct sun, and whatever the camera's exposure does about them.  It scales
    all three channels, so it leaves hue and saturation alone and takes
    chroma.  `glare` is light *added* to every channel: veiling glare from a
    sun behind the signal, and the same arithmetic as a sensor clipping.  It
    leaves hue and chroma alone and takes saturation.  Both in 0..1.
    """
    return np.clip(frame.astype(np.float32) * gain + glare * 255.0, 0, 255).astype(
        np.uint8
    )


def arm_colour(frame: np.ndarray, observation, limits) -> tuple[float, float]:
    """The arm's own (chroma, value), from the pixels the detector matched.

    Measured rather than assumed, because it is what the cases below are
    derived from: how dark the light can get is a question about the arm's
    chroma, and how much glare it can take is a question about its
    saturation, and neither means anything in the abstract.
    """
    x, y = (int(value) for value in observation.position)
    half = limits.cluster_window // 2
    patch = frame[
        max(0, y - half) : y + half + 1,
        max(0, x - half) : x + half + 1,
    ]
    planes = limits.mask(
        limits.red if observation.state == sd.RED else limits.green, sd.hsv(patch)
    )
    _hue, saturation, value = sd.hsv(patch)
    lit = (saturation * value)[planes], value[planes]
    return float(np.median(lit[0])), float(np.median(lit[1]))


# What a start has to survive, in the two quantities weather moves.  The arm
# renders at a chroma of about 0.2 in the simulator and a bright signal in
# daylight is better than that, so a tenth of its chroma is a dark frame
# indeed -- and a fifth of its saturation is a signal being read through
# glare that has all but grayed it out.  Below these the check goes on
# measuring but stops requiring, because at some point the frame no longer
# holds the answer and the fix is a lens hood, not a threshold.
REQUIRED_CHROMA = 0.05
REQUIRED_SATURATION = 0.20

# Chroma left in the arm, as the light drops.  Each becomes an exposure.
BRIGHTNESS = [0.40, 0.20, 0.10, 0.05, 0.02, 0.01]

# Saturation left in the arm, as glare washes it out.  Each becomes an
# exposure and a wash together, because that is what a camera looking into
# the sun does: it meters the blazing sky, stops down until the frame is no
# longer blown out, and the glare fills the arm's shadow back in.  Washing a
# dim arm without stopping down would only clip the whole frame to white.
WASH = [0.50, 0.30, 0.20, 0.15, 0.10, 0.05]

# The mid-tone a camera's exposure lands the arm on in those cases.  Bright
# enough to be the frame a camera would actually deliver, and low enough that
# the glare needed to gray the arm out does not clip it.
EXPOSED_VALUE = 0.6


def lights(chroma: float, value: float) -> list[tuple[str, float, float, float, bool]]:
    """The cases to try, as (label, gain, glare, what it measures, required).

    Both are solved for rather than guessed at.  Chroma scales with exposure,
    so the gain that leaves a given chroma is a division.  For the glare
    cases, the exposure and the wash are solved together to land the arm on
    `EXPOSED_VALUE` with the saturation asked for: chroma survives the glare
    and scales with the gain, so `gain = saturation * value / chroma` and the
    glare is whatever is left to make up the mid-tone.
    """
    cases = []
    for target in BRIGHTNESS:
        gain = target / chroma
        cases.append(
            (
                f"chroma {target:.2f}" + (" (as rendered)" if gain >= 1.0 else ""),
                gain,
                0.0,
                target,
                target >= REQUIRED_CHROMA,
            )
        )
    for target in WASH:
        gain = target * EXPOSED_VALUE / chroma
        glare = EXPOSED_VALUE - gain * value
        if glare <= 0:
            continue  # the arm is already less saturated than that
        cases.append(
            (
                f"glare to saturation {target:.2f}",
                gain,
                glare,
                target,
                target >= REQUIRED_SATURATION,
            )
        )
    return cases


# A person a few meters away, in a shirt each of the signal's colors.  Both
# are further from the camera than the arm and still several times its size,
# which is what the detector throws them out on.
SHIRT_RED = (180, 30, 40)
SHIRT_GREEN = (40, 170, 90)
SHIRT_SIZE = (46, 64)


def paste(frame: np.ndarray, colour, centre, size=SHIRT_SIZE) -> np.ndarray:
    frame = np.array(frame, dtype=np.uint8)
    height, width = frame.shape[:2]
    x0 = max(0, min(width - size[0], centre[0] - size[0] // 2))
    y0 = max(0, min(height - size[1], centre[1] - size[1] // 2))
    frame[y0 : y0 + size[1], x0 : x0 + size[0]] = colour
    return frame


# ------------------------------------------------------------------ the camera


class Camera(Node):
    """One camera subscription and the randomizer's signal service."""

    def __init__(self, image_topic: str) -> None:
        super().__init__("check_signal_lighting")
        self.frame: np.ndarray | None = None
        self.frames = 0
        self.create_subscription(
            Image, image_topic, self.on_image, qos_profile_sensor_data
        )
        self.signal = self.create_client(SetBool, f"{RANDOMIZER}/start_signal")

    def on_image(self, message: Image) -> None:
        try:
            frame = sd.decode(
                message.encoding,
                message.width,
                message.height,
                message.step,
                message.data,
            )
        except sd.ImageFormatError as error:
            print(f"    cannot read this camera: {error}")
            return
        # A copy: decode returns a view onto the message's own buffer.
        self.frame = np.array(frame)
        self.frames += 1

    def grab(self, timeout: float) -> np.ndarray | None:
        """The next frame to arrive, so it is one taken after the arm moved."""
        self.frame = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.frame is not None:
                return self.frame
        return None

    def turn(self, green: bool, timeout: float) -> bool:
        """Turn the simulated signal, and wait out the sweep."""
        if not self.signal.wait_for_service(timeout_sec=timeout):
            print(f"    {self.signal.srv_name} is not there")
            return False
        future = self.signal.call_async(SetBool.Request(data=green))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                return True
        return False


# ----------------------------------------------------------------- the replay


def start(frames, thresholds=None, region=None) -> tuple[bool, list]:
    """Replay a sequence through a whole detector: did it start, and what did it see."""
    unit = sd.Detector(sd.Classifier(thresholds, region))
    observations = [unit.process(frame) for frame in frames]
    return unit.go, observations


def waiting_then_turning(red: np.ndarray, green: np.ndarray) -> list[np.ndarray]:
    """The frames of a start: the wait at the line, then the arm turned.

    Long enough to arm the detector several times over, so a case that fails
    has failed on the color and not on the count.
    """
    return [red] * 10 + [green] * 4


def measure(red: np.ndarray, green: np.ndarray, position, chroma, value) -> list:
    """Every case, as (label, whether it must start, whether it did, frames)."""
    cases: list[tuple] = []

    for label, gain, glare, _target, required in lights(chroma, value):
        went, observations = start(
            waiting_then_turning(under(red, gain, glare), under(green, gain, glare))
        )
        cases.append((label, True if required else None, went, observations))

    # People in frame, on both sides of the signal and well clear of it.
    beside = (int(position[0]) - 170, max(40, int(position[1]) - 60))
    other = (int(position[0]) + 170, max(40, int(position[1]) - 60))

    def crowded(frame):
        return paste(paste(frame, SHIRT_RED, beside), SHIRT_GREEN, other)

    went, observations = start(waiting_then_turning(crowded(red), crowded(green)))
    at_arm = observations[-1].position is not None and (
        max(
            abs(observations[-1].position[0] - position[0]),
            abs(observations[-1].position[1] - position[1]),
        )
        < 60
    )
    cases.append(("people in frame", True, went and at_arm, observations))

    # ...and the same people with nothing to start: the signal stays red
    # while a red shirt is replaced by a green one in the same place.
    went, observations = start(
        [paste(red, SHIRT_RED, beside)] * 10 + [paste(red, SHIRT_GREEN, beside)] * 10
    )
    cases.append(("a shirt changing color", False, went, observations))

    # Green with no red before it is not a transition, whoever is in frame.
    went, observations = start([crowded(green)] * 20)
    cases.append(("green from the start", False, went, observations))

    # The course with the signal red throughout, which is the whole of a run
    # that must not start.
    went, observations = start([red] * 30)
    cases.append(("the signal never turning", False, went, observations))
    return cases


def report(cases) -> bool:
    """The table, and whether every case that had to go one way did.

    A case with no expectation is one past the point the detector claims to
    work, and is printed for the record: it is where the cliff is, which is
    worth knowing before somebody moves a threshold towards it.
    """
    print(
        f"    {'case':28s} {'expect':>7} {'result':>7} "
        f"{'red px':>7} {'green px':>9}  note"
    )
    passed = True
    for label, expected, went, observations in cases:
        last = observations[-1]
        good = expected is None or went == expected
        passed = passed and good
        if expected is None:
            wanted = "--"
        else:
            wanted = "start" if expected else "no"
        print(
            f"    {label:28s} {wanted:>7} {'start' if went else 'no':>7} "
            f"{last.red.pixels:7d} {last.green.pixels:9d}"
            f"  {'' if good else 'FAIL -- '}{last}"
        )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=IMAGE_TOPIC,
        help="camera topic to take the two frames from",
    )
    parser.add_argument(
        "--spin-by-hand",
        action="store_true",
        help="on the car: prompt for the signal to be turned, rather than "
        "asking the randomizer to turn it",
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
    camera = Camera(arguments.image)
    try:
        for colour in ("red", "green"):
            if arguments.spin_by_hand:
                input(f"    turn the signal {colour} and press return: ")
            elif not camera.turn(colour == "green", arguments.timeout):
                print("    could not turn the signal")
                return 1
            frame = camera.grab(arguments.timeout)
            if frame is None:
                print(f"    no camera frames on {arguments.image}")
                return 1
            if colour == "red":
                red = frame
            else:
                green = frame

        limits = sd.Thresholds()
        observation = sd.Classifier(limits).classify(red)
        if observation.state != sd.RED:
            print(f"    the frame taken with the signal red reads {observation}")
            return 1
        position = observation.position
        chroma, value = arm_colour(red, observation, limits)
        print(
            f"    signal read at ({position[0]:.0f}, {position[1]:.0f}) on "
            f"{observation.red.pixels} px of red, in {red.shape[1]}x{red.shape[0]} "
            f"frames"
        )
        print(
            f"    the arm's own color there: chroma {chroma:.3f}, value "
            f"{value:.2f}, saturation {chroma / value:.2f} -- every case below "
            f"is derived from those\n"
        )

        cases = measure(red, green, position, chroma, value)
        passed = report(cases)
        required = [case for case in cases if case[1] is not None]
        print(
            f"\n{sum(went == expected for _l, expected, went, _o in required)} of "
            f"{len(required)} required cases as expected, "
            f"{len(cases) - len(required)} more measured for the record"
        )
        return 0 if passed else 1
    finally:
        camera.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
