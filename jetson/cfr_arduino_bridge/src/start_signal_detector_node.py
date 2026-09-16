#!/usr/bin/env python3
"""Watch the camera for the course's start signal and say when to go.

A run starts on a visual signal: a red arm turns to a green one, 90 degrees
apart on a common pivot, in about a second.  This node watches the left colour
image for that transition and latches it, so the driver has one thing to wait
on and no camera code of its own.

| Interface | Type | Direction |
| --------- | ---- | --------- |
| `image` | `sensor_msgs/Image` | subscribed (remapped to the ZED's left colour topic) |
| `~/go` | `std_msgs/Bool` | published, transient local, latched |
| `~/state` | `cfr_interfaces/StartSignal` | published once per frame |
| `~/reset` | `std_srvs/Trigger` | wait for another start |
| `~/debug_image` | `sensor_msgs/Image` | published while `debug_image` is true |

`~/go` is the trigger.  It is published once as false at startup and once as
true when the start is confirmed, on a transient local publisher, so a driver
that comes up after the signal has already turned still receives it -- and one
that comes up before gets the false, which is the difference between "not yet"
and "no detector running".

`~/state` is the running commentary: what this frame shows, how many pixels of
each colour, where, and whether the signal has been found at all.  A driver
that has to stop on a red flag mid-run watches that rather than `~/go`,
because `~/go` deliberately stays up once a run has started; a momentarily
mis-hued frame must not be able to retract a start that has already happened.

`armed` in that message is the field to watch while the car waits at the line.
The course is outdoors and the light is whatever the day gives, so the
detector finds the signal first -- a place in the image that holds red long
enough to be it -- and only then waits for that place to turn green.  Until it
is armed, no amount of green will start the run, and that is the state worth
knowing about before the flag drops rather than after.  `start_signal_detector`
explains what else is in frame outdoors and what stops a red shirt being taken
for the signal.

The decision itself is in `start_signal_detector.py`, which is free of ROS and
tested against synthetic frames.  This file is the wiring: parameters, one
subscription, and the logging that makes a false start diagnosable afterwards.

Everything is tunable while running -- `ros2 param set` on any of the
thresholds rebuilds the classifier without disturbing the latch, so the bands
can be moved with a live camera and the `~/debug_image` in front of you.
"""

from __future__ import annotations

import sys
from pathlib import Path

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from cfr_interfaces.msg import StartSignal

# Installed alongside this file, and also its neighbour in the source tree, so
# an interpreter started on either finds it on sys.path already.  Added
# explicitly regardless, because a launch file that runs this through a
# wrapper need not leave it there.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import start_signal_detector as sd  # noqa: E402  (needs the path above)

if (sd.UNKNOWN, sd.RED, sd.GREEN) != (
    StartSignal.UNKNOWN,
    StartSignal.RED,
    StartSignal.GREEN,
):
    raise RuntimeError(
        "start_signal_detector's state constants no longer match "
        "cfr_interfaces/StartSignal; one of the two has been renumbered"
    )

# Parameters that may be changed while the node runs.  Anything else -- the
# topic it subscribes to, chiefly -- takes a restart.
TUNABLE = (
    "region",
    "red_hue",
    "green_hue",
    "min_saturation",
    "min_chroma",
    "min_value",
    "min_pixels",
    "cluster_window",
    "max_spread",
    "candidates",
    "focus_relaxation",
    "confirm_frames",
    "arm_frames",
    "require_red_first",
    "max_transition_distance",
    "forget_frames",
)


class StartSignalDetector(Node):
    def __init__(self) -> None:
        super().__init__("start_signal_detector")

        defaults = sd.Thresholds()
        region = sd.DEFAULT_REGION
        self.declare_parameter(
            "region", [region.x_min, region.y_min, region.x_max, region.y_max]
        )
        self.declare_parameter("red_hue", [defaults.red.low, defaults.red.high])
        self.declare_parameter("green_hue", [defaults.green.low, defaults.green.high])
        self.declare_parameter("min_saturation", defaults.min_saturation)
        self.declare_parameter("min_chroma", defaults.min_chroma)
        self.declare_parameter("min_value", defaults.min_value)
        self.declare_parameter("min_pixels", defaults.min_pixels)
        self.declare_parameter("cluster_window", defaults.cluster_window)
        self.declare_parameter("max_spread", defaults.max_spread)
        self.declare_parameter("candidates", defaults.candidates)
        self.declare_parameter("focus_relaxation", defaults.focus_relaxation)
        latch = sd.StartLatch()
        self.declare_parameter("confirm_frames", latch.confirm_frames)
        self.declare_parameter("arm_frames", latch.arm_frames)
        self.declare_parameter("require_red_first", latch.require_red_first)
        self.declare_parameter("max_transition_distance", latch.max_transition_distance)
        self.declare_parameter("forget_frames", latch.forget_frames)
        # Wall time without a frame before this starts complaining.  Not a
        # failure of the detector, but the thing most likely to be wrong when
        # a run does not start: the camera is not publishing.
        self.declare_parameter("image_timeout", 3.0)
        self.declare_parameter("debug_image", False)
        self.declare_parameter("debug_image_period", 0.2)

        self.detector = sd.Detector(
            self.build_classifier(),
            sd.StartLatch(
                confirm_frames=int(self.get_parameter("confirm_frames").value),
                arm_frames=int(self.get_parameter("arm_frames").value),
                require_red_first=bool(self.get_parameter("require_red_first").value),
                max_transition_distance=float(
                    self.get_parameter("max_transition_distance").value
                ),
                forget_frames=int(self.get_parameter("forget_frames").value),
            ),
        )
        self.add_on_set_parameters_callback(self.on_parameters)

        self.state_publisher = self.create_publisher(StartSignal, "~/state", 10)
        # Transient local and depth 1: the last value is the whole message, and
        # a driver launched after the signal turned still has to hear about it.
        self.go_publisher = self.create_publisher(
            Bool,
            "~/go",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.debug_publisher = self.create_publisher(Image, "~/debug_image", 1)
        self.create_service(Trigger, "~/reset", self.on_reset)

        # Best effort, to match both publishers this ever subscribes to: the
        # ZED wrapper and ros_gz_bridge.  A reliable subscription would simply
        # never connect to a best effort camera.
        self.create_subscription(Image, "image", self.on_image, qos_profile_sensor_data)

        self.state = sd.UNKNOWN
        # Whether the start has been announced on ~/go.  Separate from the
        # latch itself, which is up from the moment it decides; this is what
        # keeps the announcement to once per start.
        self.announced = False
        # Whether the detector has found the signal, to log the edge.
        self.was_armed = False
        self.last_image = None
        self.last_debug = 0.0
        self.publish_go(False)
        self.create_timer(1.0, self.on_watchdog)
        self.get_logger().info(
            f"watching {self.resolve_topic_name('image')} for the start signal; "
            f"{self.describe()}"
        )

    # ----------------------------------------------------------- parameters

    def parameter(self, name: str, pending: dict | None = None):
        if pending and name in pending:
            return pending[name]
        return self.get_parameter(name).value

    def numbers(self, name: str, count: int, pending: dict | None = None):
        """A list parameter of exactly `count` numbers, or a clear complaint."""
        values = [float(value) for value in self.parameter(name, pending)]
        if len(values) != count:
            raise ValueError(
                f"{name} takes {count} numbers, not the {len(values)} given"
            )
        return values

    def build_classifier(self, pending: dict | None = None) -> sd.Classifier:
        """A classifier built from the current parameters.

        Raises rather than clamping: a region or a hue band that does not make
        sense is a mistake to report, not something to quietly correct into a
        detector that looks in the wrong place.
        """
        region = sd.Region(*self.numbers("region", 4, pending))
        thresholds = sd.Thresholds(
            red=sd.HueBand(*self.numbers("red_hue", 2, pending)),
            green=sd.HueBand(*self.numbers("green_hue", 2, pending)),
            min_saturation=float(self.parameter("min_saturation", pending)),
            min_chroma=float(self.parameter("min_chroma", pending)),
            min_value=float(self.parameter("min_value", pending)),
            min_pixels=int(self.parameter("min_pixels", pending)),
            cluster_window=int(self.parameter("cluster_window", pending)),
            max_spread=float(self.parameter("max_spread", pending)),
            candidates=int(self.parameter("candidates", pending)),
            focus_relaxation=float(self.parameter("focus_relaxation", pending)),
        )
        if thresholds.min_pixels < 1:
            raise ValueError("min_pixels must be at least 1")
        if thresholds.cluster_window < 1:
            raise ValueError("cluster_window must be at least 1 pixel")
        if thresholds.max_spread < 1.0:
            raise ValueError(
                "max_spread must be at least 1.0, which is an arm that fills "
                "its window and nothing outside it"
            )
        if thresholds.candidates < 1:
            raise ValueError("candidates must be at least 1")
        if not 0.0 < thresholds.focus_relaxation <= 1.0:
            raise ValueError(
                "focus_relaxation is the fraction of the colour floors that "
                "applies inside the box around the signal, so it is over 0 "
                "and at most 1"
            )
        return sd.Classifier(thresholds, region)

    def on_parameters(self, parameters) -> SetParametersResult:
        """Retune while running, without losing the latch."""
        pending = {
            parameter.name: parameter.value
            for parameter in parameters
            if parameter.name in TUNABLE
        }
        if not pending:
            return SetParametersResult(successful=True)
        try:
            classifier = self.build_classifier(pending)
        except (TypeError, ValueError) as error:
            self.get_logger().warning(f"rejected {', '.join(pending)}: {error}")
            return SetParametersResult(successful=False, reason=str(error))

        self.detector.classifier = classifier
        latch = self.detector.latch
        latch.confirm_frames = max(1, int(self.parameter("confirm_frames", pending)))
        latch.arm_frames = max(1, int(self.parameter("arm_frames", pending)))
        latch.require_red_first = bool(self.parameter("require_red_first", pending))
        latch.max_transition_distance = float(
            self.parameter("max_transition_distance", pending)
        )
        latch.forget_frames = max(1, int(self.parameter("forget_frames", pending)))
        self.get_logger().info(f"retuned: {self.describe()}")
        return SetParametersResult(successful=True)

    def describe(self) -> str:
        thresholds = self.classifier.thresholds
        region = self.classifier.region
        return (
            f"red hue {thresholds.red.low:.0f}..{thresholds.red.high:.0f}, "
            f"green hue {thresholds.green.low:.0f}..{thresholds.green.high:.0f}, "
            f"saturation over {thresholds.min_saturation:.2f} "
            f"({thresholds.focus_relaxation:.2f} of it once the signal is found), "
            f"{thresholds.min_pixels} px in a {thresholds.cluster_window} px window "
            f"spreading no more than {thresholds.max_spread:.1f}x, "
            f"{self.latch.arm_frames} frames to arm and "
            f"{self.latch.confirm_frames} to confirm, "
            f"region x {region.x_min:.2f}..{region.x_max:.2f} "
            f"y {region.y_min:.2f}..{region.y_max:.2f}"
        )

    @property
    def classifier(self) -> sd.Classifier:
        return self.detector.classifier

    @property
    def latch(self) -> sd.StartLatch:
        return self.detector.latch

    # ---------------------------------------------------------------- images

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
            # Throttled: an unreadable encoding is unreadable at 15 Hz.
            self.get_logger().error(
                f"cannot use this camera: {error}", throttle_duration_sec=5.0
            )
            return

        self.last_image = self.get_clock().now()
        # The focus the latch hands back steers the next frame's search, so
        # the box the classifier was given is read before the frame is folded
        # in and is the one the debug image draws.
        focus = self.latch.focus()
        observation = self.detector.process(frame)
        started = self.latch.go
        self.publish_state(message, observation)

        if observation.state != self.state:
            self.state = observation.state
            self.get_logger().info(f"signal reads {observation}")
        # Arming is the half of this that goes wrong quietly: a detector that
        # has never found the signal will not start the run whatever the
        # marshal does with it, and says nothing unless it says this.
        if self.latch.armed != self.was_armed:
            self.was_armed = self.latch.armed
            site = self.latch.signal
            if self.was_armed and site is not None:
                self.get_logger().info(
                    f"signal found at ({site.x:.0f}, {site.y:.0f}) after "
                    f"{site.red_frames} frames of red; watching there for the turn"
                )
            else:
                self.get_logger().warning(
                    "lost the signal; searching the whole region again"
                )
        # Every frame, not only on a change: a detector that has been looking
        # at a green it will not accept -- because nothing showed red first,
        # or because it is green somewhere else in frame -- is the case where
        # the car sits still and nobody knows why.  Throttled, since the
        # answer does not change at 15 Hz.
        rejection = self.latch.rejection(observation)
        if rejection:
            self.get_logger().warning(
                f"not starting: {rejection}", throttle_duration_sec=10.0
            )
        if started and not self.announced:
            self.announced = True
            stamp = message.header.stamp
            self.get_logger().info(
                f"START -- green confirmed {observation}, on the frame stamped "
                f"{stamp.sec}.{stamp.nanosec // 1000000:03d}"
            )
            self.publish_go(True)

        self.publish_debug(message, frame, observation, focus)

    def publish_state(self, image: Image, observation: sd.Observation) -> None:
        message = StartSignal()
        # The camera's own stamp, so a detection can be lined up against the
        # frame that caused it rather than against when this got to it.
        message.header = image.header
        message.state = observation.state
        message.go = self.latch.go
        message.red_pixels = min(observation.red.pixels, 65535)
        message.green_pixels = min(observation.green.pixels, 65535)
        message.red_total = min(observation.red.total, 65535)
        message.green_total = min(observation.green.total, 65535)
        position = observation.position or (-1.0, -1.0)
        message.x, message.y = float(position[0]), float(position[1])
        message.armed = self.latch.armed
        site = self.latch.signal
        lock = site.position if site is not None else (-1.0, -1.0)
        message.lock_x, message.lock_y = float(lock[0]), float(lock[1])
        self.state_publisher.publish(message)

    def publish_debug(self, image: Image, frame, observation, focus) -> None:
        """The frame with the region and the cluster drawn on, for tuning."""
        if not self.get_parameter("debug_image").value:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        period = float(self.get_parameter("debug_image_period").value)
        if now - self.last_debug < period:
            return
        self.last_debug = now

        marked = sd.annotate(frame, observation, self.classifier.region, focus)
        message = Image()
        message.header = image.header
        message.height, message.width = marked.shape[:2]
        message.encoding = "rgb8"
        message.is_bigendian = 0
        message.step = message.width * 3
        message.data = marked.tobytes()
        self.debug_publisher.publish(message)

    # ------------------------------------------------------------- reporting

    def publish_go(self, started: bool) -> None:
        message = Bool()
        message.data = started
        self.go_publisher.publish(message)

    def on_reset(self, _request, response):
        self.detector.reset()
        self.state = sd.UNKNOWN
        self.announced = False
        self.was_armed = False
        self.publish_go(False)
        response.success = True
        response.message = "waiting for red, then green"
        self.get_logger().info(f"reset: {response.message}")
        return response

    def on_watchdog(self) -> None:
        """Say so when no frames are arriving, which is the usual reason."""
        timeout = float(self.get_parameter("image_timeout").value)
        topic = self.resolve_topic_name("image")
        if self.last_image is None:
            self.get_logger().warning(
                f"no camera frames on {topic} yet; the simulation needs "
                "sensors:=true, and the car needs the ZED running",
                throttle_duration_sec=10.0,
            )
            return
        idle = (self.get_clock().now() - self.last_image).nanoseconds / 1e9
        if idle > timeout:
            self.get_logger().warning(
                f"no camera frames on {topic} for {idle:.1f} s",
                throttle_duration_sec=10.0,
            )


def main() -> None:
    rclpy.init()
    node = StartSignalDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C, or the launch tearing the stack down around us.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
