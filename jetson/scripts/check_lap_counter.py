#!/usr/bin/env python3
"""Drive a synthetic run at lap_counter_node and check what it reports.

The unit tests cover the counting itself against traces built in Python; what
they cannot cover is the wiring -- the remappings, the QoS on the latched
topics, the Arduino status gate, and the message the driver will actually
subscribe to.  This does that, with no Gazebo and no car: it publishes the
pose, the status and the start signal itself, then watches `~/count` and
`~/done`.

    source ~/ros2_ws/install/setup.bash
    ros2 run cfr_arduino_bridge lap_counter_node.py --ros-args \
        -r pose:=/check/pose -r status:=/check/status -r go:=/check/go \
        -p require_auto_active:=true
    python3 jetson/scripts/check_lap_counter.py

It runs the speed course's shape: out across the line, three laps of
a 41 x 14 m oval, with an e-stop and a carry over the line partway through the
second lap and a loop-closure jump on the third.  A correct node reports three
laps, latches ~/done once, and counts neither the carry nor the jump.
"""

from __future__ import annotations

import math
import sys

import rclpy
from cfr_interfaces.msg import ArduinoStatus, LapCount
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool

LANE_LENGTH = 41.0
LANE_WIDTH = 14.0
BEHIND = -2.0
# Replay rate, not vehicle speed: the gates are geometric, so the trace can be
# pushed through as fast as the graph will take it.  STEP is what matters --
# it has to stay well under the node's max_step, or every sample would read as
# a loop closure.
RATE_HZ = 200.0
STEP = 0.12  # m between samples; the ZED's 30 Hz at 3.6 m/s


def leg(start, end, step=STEP):
    dx, dy = end[0] - start[0], end[1] - start[1]
    span = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    count = max(int(span / step), 1)
    return [
        (start[0] + dx * i / count, start[1] + dy * i / count, yaw)
        for i in range(1, count + 1)
    ]


def rejoin(end=(LANE_LENGTH, 0.0)):
    return (
        leg((LANE_LENGTH, 0.0), (LANE_LENGTH, -LANE_WIDTH))
        + leg((LANE_LENGTH, -LANE_WIDTH), (BEHIND, -LANE_WIDTH))
        + leg((BEHIND, -LANE_WIDTH), (BEHIND, 0.0))
        + leg((BEHIND, 0.0), end)
    )


def build_run():
    """Poses, each tagged with whether the car is under autonomy."""
    samples = [(pose, True) for pose in leg((0.0, 0.0), (LANE_LENGTH, 0.0))]
    samples += [(pose, True) for pose in rejoin()]  # lap 1

    # Lap 2, interrupted: e-stop on the back straight, carried over the line
    # and set down past it, then re-armed.  None of that may score.
    samples += [
        (pose, True) for pose in leg((LANE_LENGTH, 0.0), (LANE_LENGTH, -LANE_WIDTH))
    ]
    samples += [(pose, False) for pose in leg((LANE_LENGTH, -LANE_WIDTH), (-1.0, 0.0))]
    samples += [((3.0, 0.0, 0.0), False)] * 5
    samples += [(pose, True) for pose in leg((3.0, 0.0), (LANE_LENGTH, 0.0))]
    samples += [(pose, True) for pose in rejoin()]  # lap 2

    # Lap 3, with a loop closure a couple of metres short of the line.
    third = rejoin(end=(-0.6, 0.0))
    samples += [(pose, True) for pose in third]
    samples += [((0.9, 0.0, 0.0), True)]  # the jump, across the line
    samples += [(pose, True) for pose in leg((0.9, 0.0), (10.0, 0.0))]
    return samples


class Check(Node):
    def __init__(self) -> None:
        super().__init__("check_lap_counter")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pose_publisher = self.create_publisher(PoseStamped, "/check/pose", 10)
        self.status_publisher = self.create_publisher(
            ArduinoStatus, "/check/status", 10
        )
        self.go_publisher = self.create_publisher(Bool, "/check/go", latched)
        self.create_subscription(LapCount, "/lap_counter/count", self.on_count, 10)
        self.create_subscription(Bool, "/lap_counter/done", self.on_done, latched)

        self.samples = build_run()
        self.index = 0
        self.last: LapCount | None = None
        self.done_messages: list[bool] = []
        self.lap_marks: list[tuple[int, float]] = []
        self.started = False
        self.create_timer(1.0 / RATE_HZ, self.tick)

    def on_count(self, message: LapCount) -> None:
        if self.last is not None and message.laps > self.last.laps:
            self.lap_marks.append((message.laps, message.lap_distance))
            print(
                f"  lap {message.laps} at along_track={message.along_track:+.2f} "
                f"cross_track={message.cross_track:+.2f}"
            )
        self.last = message

    def on_done(self, message: Bool) -> None:
        self.done_messages.append(message.data)

    def tick(self) -> None:
        if not self.started:
            # Wait for the node to be listening before streaming anything.
            # Publishing into an unmatched subscription drops the first poses,
            # and the node would arm metres down the lane -- which puts the
            # line somewhere it is not and makes this check meaningless.
            if (
                self.pose_publisher.get_subscription_count() == 0
                or self.status_publisher.get_subscription_count() == 0
                or self.go_publisher.get_subscription_count() == 0
            ):
                return
            self.go_publisher.publish(Bool(data=True))
            self.started = True
            print(f"replaying {len(self.samples)} samples at {RATE_HZ:.0f} Hz")
            return

        if self.index >= len(self.samples):
            self.report()
            raise SystemExit(0 if self.passed() else 1)

        if self.index == 0:
            # One tick of grace so the latched go arrives before the pose that
            # arms the counter.
            self.index = 1
            (x, y, yaw), counting = self.samples[0]
        else:
            (x, y, yaw), counting = self.samples[self.index]
            self.index += 1
        now = self.get_clock().now().to_msg()

        status = ArduinoStatus()
        status.header.stamp = now
        status.header.frame_id = "base_link"
        status.link_ok = True
        status.estop = not counting
        status.mode = (
            ArduinoStatus.MODE_AUTO_ACTIVE if counting else ArduinoStatus.MODE_ESTOP
        )
        self.status_publisher.publish(status)

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        self.pose_publisher.publish(pose)

    # ------------------------------------------------------------- verdict

    def passed(self) -> bool:
        return not self.failures()

    def failures(self) -> list[str]:
        problems = []
        if self.last is None:
            return ["no LapCount received; check the remappings"]
        if self.last.laps != 3:
            problems.append(f"expected 3 laps, got {self.last.laps}")
        if not self.last.done:
            problems.append("done never latched")
        if self.done_messages[-1:] != [True]:
            problems.append(f"done topic ended at {self.done_messages[-1:]}")
        if self.done_messages.count(True) != 1:
            problems.append(f"done published {self.done_messages.count(True)} times")
        if self.last.target != 3:
            problems.append(f"target reported as {self.last.target}, not 3")
        if self.last.loop_closures < 1:
            problems.append("the loop-closure jump was not noticed")
        return problems

    def report(self) -> None:
        last = self.last
        print()
        if last is not None:
            print(
                f"laps={last.laps}/{last.target} done={last.done} "
                f"rejected={last.rejected} loop_closures={last.loop_closures}"
            )
        problems = self.failures()
        if problems:
            print("FAIL")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("PASS -- three laps, one latched done, carry and jump ignored")


def main() -> None:
    rclpy.init()
    node = Check()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except SystemExit as exit_code:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(exit_code.code)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
