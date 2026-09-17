#!/usr/bin/env python3
"""Count laps of the start/finish line, and latch when the run is done.

Run alongside the bridge and the ZED:

    ~/software/scripts/launch.sh                                # bridge + ZED
    ros2 launch cfr_arduino_bridge lap_counter.launch.py laps:=3

In the Gazebo courses this comes up already, with the lap target the course
calls for -- three for the speed course, two for the obstacle course.

| Interface  | Type                        | Direction                     |
| ---------- | --------------------------- | ----------------------------- |
| `pose`     | `geometry_msgs/PoseStamped` | subscribed (`/zed/.../pose`)  |
| `status`   | `cfr_interfaces/ArduinoStatus` | subscribed (run mode)      |
| `go`       | `std_msgs/Bool`             | subscribed (the start signal) |
| `~/count`  | `cfr_interfaces/LapCount`   | published every pose sample   |
| `~/done`   | `std_msgs/Bool`             | published latched, on change  |
| `~/reset`  | `std_srvs/Trigger`          | service, re-arm for a re-run  |

The pose is the ZED's **map** frame pose, not `~/odom`: the SDK applies loop
closure to that one and deliberately never to odometry, and three laps of the
speed course is ~300 m of travel returning to the same spot.  See
`zed/config/cfr_zed2i.yaml`, which pins `area_memory` on rather than leaving
it to whatever the installed wrapper defaults to.

The counting itself is in `lap_counter.py`, which is free of ROS and tested
against synthetic traces.  This file is the wiring: parameters, the gate that
stops an e-stopped car scoring, and the logging that makes a miscount
diagnosable afterwards.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import rclpy
from cfr_interfaces.msg import ArduinoStatus, LapCount
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lap_counter as lc  # noqa: E402

# The states are spelled out in both the message and the ROS-free module, so
# that module can be tested without a built workspace.  Fail loudly here if
# the two ever drift apart.
assert lc.STATE_IDLE == LapCount.STATE_IDLE
assert lc.STATE_ARMED == LapCount.STATE_ARMED
assert lc.STATE_RUNNING == LapCount.STATE_RUNNING
assert lc.STATE_DONE == LapCount.STATE_DONE

# Parameters that may be retuned while the node runs.  The geometry can move
# between practice runs; target_laps is here too because a heat can be
# re-flighted for a different number of laps.
TUNABLE = (
    "target_laps",
    "line_offset",
    "heading_tolerance",
    "min_lap_distance",
    "lateral_gate",
    "max_step",
    "carry_tolerance",
)


class LapCounter(Node):
    def __init__(self) -> None:
        super().__init__("lap_counter")

        defaults = lc.Geometry()
        self.declare_parameter("target_laps", 3)
        self.declare_parameter("line_offset", defaults.line_offset)
        self.declare_parameter("heading_tolerance", defaults.heading_tolerance)
        self.declare_parameter("min_lap_distance", defaults.min_lap_distance)
        self.declare_parameter("lateral_gate", defaults.lateral_gate)
        self.declare_parameter("max_step", defaults.max_step)
        self.declare_parameter("carry_tolerance", defaults.carry_tolerance)
        self.declare_parameter("require_go", True)
        self.declare_parameter("require_auto_active", True)
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("status_timeout", 1.0)

        self.tracker = lc.LapTracker(self.target_laps(), self.build_geometry())
        self.add_on_set_parameters_callback(self.on_parameters)

        self.go = False
        self.status: ArduinoStatus | None = None
        self.status_time = None
        self.pose_time = None
        self.was_counting = False
        self.last_crossing = LapCount().last_crossing
        self.done_published: bool | None = None
        self.warned_no_pose = False

        self.count_publisher = self.create_publisher(LapCount, "~/count", 10)
        # Transient local and depth 1, matching the start signal's ~/go: the
        # last value is the whole message, and a driver that starts late still
        # has to hear that the run is over.
        self.done_publisher = self.create_publisher(
            Bool,
            "~/done",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_service(Trigger, "~/reset", self.on_reset)

        self.create_subscription(PoseStamped, "pose", self.on_pose, 10)
        self.create_subscription(ArduinoStatus, "status", self.on_status, 10)
        # Transient local, to match the detector's latched publisher.
        self.create_subscription(
            Bool,
            "go",
            self.on_go,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        self.create_timer(1.0, self.on_watchdog)
        self.publish_done(False)

        geometry = self.tracker.geometry
        self.get_logger().info(
            f"counting {self.tracker.target_laps} laps; line "
            f"{geometry.line_offset:.2f} m ahead of the start pose, heading "
            f"within {math.degrees(geometry.heading_tolerance):.0f} deg, laps "
            f"at least {geometry.min_lap_distance:.0f} m"
        )
        self.get_logger().info(f"pose from {self.resolve_topic_name('pose')}")

    # ------------------------------------------------------------ parameters

    def parameter(self, name, pending=None):
        if pending is not None and name in pending:
            return pending[name]
        return self.get_parameter(name).value

    def target_laps(self, pending=None) -> int:
        return int(self.parameter("target_laps", pending))

    def build_geometry(self, pending=None) -> lc.Geometry:
        return lc.Geometry(
            line_offset=float(self.parameter("line_offset", pending)),
            heading_tolerance=float(self.parameter("heading_tolerance", pending)),
            min_lap_distance=float(self.parameter("min_lap_distance", pending)),
            lateral_gate=float(self.parameter("lateral_gate", pending)),
            max_step=float(self.parameter("max_step", pending)),
            carry_tolerance=float(self.parameter("carry_tolerance", pending)),
        )

    def on_parameters(self, parameters) -> SetParametersResult:
        """Retune while running, without losing the count or the latch."""
        pending = {
            parameter.name: parameter.value
            for parameter in parameters
            if parameter.name in TUNABLE
        }
        if not pending:
            return SetParametersResult(successful=True)
        try:
            geometry = self.build_geometry(pending)
            target = self.target_laps(pending)
            if target < 1:
                raise ValueError("target_laps must be at least 1")
        except (TypeError, ValueError) as error:
            self.get_logger().warning(f"rejected {', '.join(pending)}: {error}")
            return SetParametersResult(successful=False, reason=str(error))
        self.tracker.geometry = geometry
        self.tracker.target_laps = target
        self.get_logger().info(f"retuned {', '.join(sorted(pending))}")
        return SetParametersResult(successful=True)

    # ---------------------------------------------------------------- inputs

    def on_status(self, message: ArduinoStatus) -> None:
        self.status = message
        self.status_time = self.get_clock().now()

    def on_go(self, message: Bool) -> None:
        if message.data and not self.go:
            self.get_logger().info("start signal latched; arming on the next pose")
        self.go = message.data

    def on_reset(self, request, response):
        del request
        self.tracker.reset()
        self.go = False
        self.was_counting = False
        self.last_crossing = LapCount().last_crossing
        self.publish_done(False)
        self.get_logger().info("reset; waiting for the start signal again")
        response.success = True
        response.message = "lap counter reset"
        return response

    def counting_now(self) -> bool:
        """Is the car actually running, rather than stopped or being moved?

        The rules allow an e-stop to lift the car past an obstacle or off the
        course, so a crossing that happens then is not a lap.  Fails closed:
        no status, or stale status, means not counting.
        """
        if not self.get_parameter("require_auto_active").value:
            return True
        if self.status is None or self.status_time is None:
            return False
        age = (self.get_clock().now() - self.status_time).nanoseconds * 1e-9
        if age > self.get_parameter("status_timeout").value:
            return False
        return (
            self.status.link_ok
            and not self.status.estop
            and self.status.mode == ArduinoStatus.MODE_AUTO_ACTIVE
        )

    # ----------------------------------------------------------------- laps

    def on_pose(self, message: PoseStamped) -> None:
        self.pose_time = self.get_clock().now()
        self.warned_no_pose = False

        orientation = message.pose.orientation
        pose = lc.Pose2D(
            x=message.pose.position.x,
            y=message.pose.position.y,
            yaw=lc.yaw_from_quaternion(
                orientation.w, orientation.x, orientation.y, orientation.z
            ),
        )

        counting = self.counting_now()

        if not self.tracker.armed:
            if self.get_parameter("require_go").value and not self.go:
                self.publish_count(message, self.tracker.update(pose, False), False)
                return
            self.tracker.arm(pose)
            self.get_logger().info(
                f"armed at x={pose.x:.2f} y={pose.y:.2f} yaw={pose.yaw:.3f}; the "
                f"line is {self.tracker.geometry.line_offset:.2f} m ahead"
            )

        if counting and not self.was_counting:
            # Back under autonomy.  Whatever happened while it was not -- a
            # carry past an obstacle, most likely -- must not read as driving.
            self.tracker.resume()
            # Quiet on the first pose of a run, when there is nothing to resume.
            if self.tracker.state != lc.STATE_ARMED:
                self.get_logger().info("counting resumed; motion baseline re-seeded")
        elif self.was_counting and not counting:
            self.get_logger().info("counting suspended; not in auto, or e-stopped")
        self.was_counting = counting

        observation = self.tracker.update(pose, counting)

        if observation.carried > 0.0:
            self.get_logger().info(
                f"carried {observation.carried:.2f} m while stopped; lap "
                f"distance restarted"
            )
        if observation.loop_closure > 0.0:
            self.get_logger().info(
                f"loop closure   jump {observation.loop_closure:.2f} m  at "
                f"s={observation.along_track:.1f} d={observation.cross_track:.1f}"
            )
        if observation.crossing is not None:
            self.log_crossing(observation.crossing, observation.laps)
            if observation.crossing.counted:
                self.last_crossing = message.header.stamp

        self.publish_count(message, observation, counting)
        self.publish_done(observation.done)

    def log_crossing(self, crossing, laps: int) -> None:
        """Report every pass through the line, counted or not.

        The physical course will not match the idealized one, so these lines
        are how the gates get tuned: they say which gate rejected a crossing
        and by how much it missed.
        """
        where = (
            f"heading {math.degrees(crossing.heading_error):+.1f} deg  "
            f"lateral {crossing.cross_track:+.2f} m  "
            f"travelled {crossing.travelled:.1f} m"
        )
        if crossing.verdict == lc.COUNTED:
            self.get_logger().info(f"lap {laps} counted   {where}")
        elif crossing.verdict == lc.DEPARTURE:
            self.get_logger().info(f"left the line, run under way   {where}")
        else:
            self.get_logger().info(f"crossing rejected ({crossing.verdict})   {where}")

    def on_watchdog(self) -> None:
        if self.pose_time is None:
            if not self.warned_no_pose:
                self.warned_no_pose = True
                self.get_logger().warning(
                    f"no pose yet on {self.resolve_topic_name('pose')}"
                )
            return
        age = (self.get_clock().now() - self.pose_time).nanoseconds * 1e-9
        if age > self.get_parameter("pose_timeout").value and not self.warned_no_pose:
            self.warned_no_pose = True
            self.get_logger().warning(f"pose stale for {age:.1f} s; not counting")

    # ---------------------------------------------------------------- output

    def publish_count(self, source: PoseStamped, observation, counting: bool) -> None:
        message = LapCount()
        # The pose's own stamp, so a crossing can be lined up against the
        # sample that caused it rather than against when this got to it.
        message.header = source.header
        message.state = observation.state
        message.laps = observation.laps
        message.target = self.tracker.target_laps
        message.done = observation.done
        message.counting = counting
        message.along_track = float(observation.along_track)
        message.cross_track = float(observation.cross_track)
        message.heading_error = float(observation.heading_error)
        message.lap_distance = float(observation.lap_distance)
        message.rejected = observation.rejected
        message.loop_closures = observation.loop_closures
        message.last_crossing = self.last_crossing
        self.count_publisher.publish(message)

    def publish_done(self, done: bool) -> None:
        if done == self.done_published:
            return
        self.done_published = done
        message = Bool()
        message.data = done
        self.done_publisher.publish(message)
        if done:
            self.get_logger().info(
                f"DONE -- {self.tracker.laps} laps complete, stop the car"
            )


def main() -> None:
    rclpy.init()
    node = LapCounter()
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
