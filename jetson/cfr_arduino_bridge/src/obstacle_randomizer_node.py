#!/usr/bin/env python3
"""Re-lay the obstacle course's variable elements, and work the start signal.

Three things about the Obstacle Course change between runs: where the buckets
stand, where the hoops sit along their lines, and whether the start signal is
showing red or green.  All three are moved here rather than baked into the
world, so a layout can be re-drawn without restarting Gazebo.

The buckets and hoops are separate static models, moved through Gazebo's
``set_pose``, which is why they are separate models at all.

The signal is not, because it has to *turn*: 90 degrees a second, like the one
on the course, so a detector gets the part-way-round arm it will have to cope
with.  Its arms ride a revolute joint and this node ramps the joint setpoint
across a bridged topic.  Stepping a model pose would have kept one mechanism
for everything, but each ``set_pose`` is a ``gz service`` subprocess costing
about 340 ms, so a one second sweep fits three poses and arrives as a stutter.

| Service | Type | Effect |
| ------- | ---- | ------ |
| ``~/randomize`` | ``std_srvs/Trigger`` | draw a new bucket and hoop layout |
| ``~/reset`` | ``std_srvs/Trigger`` | restore the layout the drawing shows |
| ``~/start_signal`` | ``std_srvs/SetBool`` | ``true`` shows green, ``false`` red |

``~/start_signal_green`` (``std_msgs/Bool``, transient local) carries the
signal state, so a detector can be scored against what the signal is actually
showing.

Bounds come from ``config/obstacle_course_layout.yaml``, which
``generate_obstacle_course.py`` writes alongside the world.  Nothing here
knows the course geometry; if the DXF moves a section, regenerating fixes
both at once.
"""

from __future__ import annotations

import math
import random
import subprocess
import time
import traceback

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float64
from std_srvs.srv import SetBool, Trigger

# Bucket sizes, for reporting how much room a draw actually left.
BUCKET_DIAMETER = 0.29
VEHICLE_WIDTH = 0.30

# How fast the signal arm turns, in degrees per second, and how often the
# sweep sends a new setpoint.  A quarter turn at 90 deg/s takes the second the
# real signal takes; 40 ms between setpoints puts 25 of them in that second,
# comfortably more than the 15 Hz camera can resolve.
SIGNAL_SWEEP_RATE = 90.0
# How many times to ask Gazebo to move a model before giving up.  See set_pose.
SET_POSE_ATTEMPTS = 3

SIGNAL_STEP_SECONDS = 0.04
# How much longer than the sweep itself the node will wait in wall time before
# giving up and commanding the far end.  A world running at a tenth of real
# time still finishes; a paused one does not hang the service call.
SIGNAL_WALL_LIMIT = 20.0


class GazeboError(RuntimeError):
    pass


class ObstacleRandomizer(Node):
    def __init__(self) -> None:
        super().__init__(
            "obstacle_randomizer", automatically_declare_parameters_from_overrides=True
        )
        self.declare_parameter_if_missing("world", "cfr_obstacle_course")
        self.declare_parameter_if_missing("bucket_count", 0)
        self.declare_parameter_if_missing("seed", -1)
        self.declare_parameter_if_missing("set_pose_timeout", 2.0)
        self.declare_parameter_if_missing("signal_sweep_rate", SIGNAL_SWEEP_RATE)

        self.world = self.get_parameter("world").value
        self.random = random.Random()

        self.signal_publisher = self.create_publisher(
            Bool,
            "~/start_signal_green",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        # Bridged to the world's joint controller by simulation.launch.py.
        # Latching, so the setpoint survives the bridge coming up late.
        self.arm_publisher = self.create_publisher(
            Float64,
            self.signal_param("topic"),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        # Reentrant, and spun by a multi-threaded executor below: a sweep
        # paces itself off the simulation clock, and the clock only advances
        # while the node is free to take /clock messages.  On the default
        # executor the service callback blocks that, sim time stops, and the
        # sweep waits out its whole wall-clock allowance.
        group = ReentrantCallbackGroup()
        self.create_service(
            Trigger, "~/randomize", self.on_randomize, callback_group=group
        )
        self.create_service(Trigger, "~/reset", self.on_reset, callback_group=group)
        self.create_service(
            SetBool, "~/start_signal", self.on_start_signal, callback_group=group
        )

        # A run starts on red: the arm joint rests at its lower limit, which is
        # the angle the red arm is modeled at, so the red arm is already the
        # one lying across the board when Gazebo finishes loading.  Commanding
        # it anyway costs nothing and means the joint is held there rather than
        # merely left there; the state topic tells a detector that subscribes
        # later.
        self.arm_angle = float(self.signal_param("red_angle"))
        self.command_arm(self.arm_angle)
        self.publish_signal(False)
        movable = f"{len(self.hoop_names())} hoops, up to {self.bucket_limit()} buckets"
        if self.has_gap_bales():
            movable += f", 1 of {len(self.gap_bale_names())} wall bales parked"
        if (
            not self.hoop_names()
            and not self.has_buckets()
            and not self.has_gap_bales()
        ):
            movable = "start signal only"
        self.get_logger().info(f"Ready on world '{self.world}': {movable}")

    def declare_parameter_if_missing(self, name: str, default) -> None:
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    # ---------------------------------------------------------------- params

    def has_buckets(self) -> bool:
        """Whether this course has a bucket section at all.

        The Speed Course shares this node for its start signal but varies
        nothing else, so its layout file carries only the signal block and
        both collections come back empty.
        """
        return self.has_parameter("buckets.count_max")

    def bucket_param(self, name: str):
        return self.get_parameter(f"buckets.{name}").value

    def bucket_limit(self) -> int:
        return int(self.bucket_param("count_max")) if self.has_buckets() else 0

    def hoop_names(self) -> list[str]:
        if not self.has_parameter("hoops.names"):
            return []
        return list(self.get_parameter("hoops.names").value)

    def hoop_param(self, hoop: str, name: str):
        return self.get_parameter(f"hoops.{hoop}.{name}").value

    def signal_param(self, name: str):
        return self.get_parameter(f"start_signal.{name}").value

    def has_gap_bales(self) -> bool:
        """Whether this course has the four-bale wall gap to randomize.

        The Speed Course shares this node too and carries neither block, so
        this comes back empty there the same way ``has_buckets`` does.
        """
        return self.has_parameter("gap_bales.names")

    def gap_bale_names(self) -> list[str]:
        if not self.has_gap_bales():
            return []
        return list(self.get_parameter("gap_bales.names").value)

    def gap_bale_param(self, bale: str, name: str):
        return self.get_parameter(f"gap_bales.{bale}.{name}").value

    # ------------------------------------------------------------ gz set_pose

    def set_pose(self, model: str, x: float, y: float, z: float, yaw=0.0):
        """Move one model.  Everything this moves stands upright, so yaw only."""
        half_yaw = yaw / 2.0
        qw, qx, qy, qz = math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)
        request = (
            f'name: "{model}" '
            f"position {{ x: {x:.9g} y: {y:.9g} z: {z:.9g} }} "
            f"orientation {{ x: {qx:.17g} y: {qy:.17g} z: {qz:.17g} w: {qw:.17g} }}"
        )
        timeout = float(self.get_parameter("set_pose_timeout").value)
        command = [
            "gz",
            "service",
            "-s",
            f"/world/{self.world}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(int(timeout * 1000)),
            "--req",
            request,
        ]
        # Retried, because this times out now and then and it is worth nothing
        # when it does.  Most of a `gz service` call is the CLI discovering the
        # service afresh, and under load that occasionally overruns -- about
        # one call in thirty, measured, which is one randomize in three across
        # the dozen models a draw moves.  A second attempt has always found it.
        for attempt in range(SET_POSE_ATTEMPTS):
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout + 1.0,
                    check=False,
                )
            except FileNotFoundError as error:
                raise GazeboError("the Gazebo CLI is not on PATH") from error
            except subprocess.TimeoutExpired:
                failure = f"set_pose timed out moving {model}"
            else:
                if result.returncode == 0 and "data: true" in result.stdout:
                    return
                detail = result.stderr.strip() or result.stdout.strip() or "no reply"
                failure = f"Gazebo refused to move {model}: {detail}"
                if "timed out" not in detail:
                    # A refusal is an answer; asking again will get the same one.
                    raise GazeboError(failure)
            if attempt + 1 < SET_POSE_ATTEMPTS:
                self.get_logger().debug(f"{failure} -- retrying")
        raise GazeboError(f"{failure} (after {SET_POSE_ATTEMPTS} attempts)")

    # ----------------------------------------------------------------- layout

    def draw_buckets(self) -> list[tuple[float, float]]:
        """Bucket positions honoring the drawing's spacing and clearance.

        The drawing asks for buckets "placed so a path exists around and
        between buckets".  That does not need a separate reachability check:
        3 ft between centers leaves a 0.62 m gap between two 0.29 m buckets,
        and the same clearance off the walls, both of which a 0.30 m car fits
        through.  Keeping the spacing is keeping the path.
        """
        if not self.has_buckets():
            return []
        low = list(self.bucket_param("region_min"))
        high = list(self.bucket_param("region_max"))
        clearance = float(self.bucket_param("wall_clearance"))
        spacing = float(self.bucket_param("min_spacing"))
        low = [value + clearance for value in low]
        high = [value - clearance for value in high]
        if low[0] > high[0] or low[1] > high[1]:
            raise GazeboError("bucket clearance leaves no room in the section")

        count = int(self.get_parameter("bucket_count").value)
        if count <= 0:
            count = self.random.randint(
                int(self.bucket_param("count_min")), int(self.bucket_param("count_max"))
            )
        count = max(
            int(self.bucket_param("count_min")),
            min(int(self.bucket_param("count_max")), count),
        )

        # Rejection sampling first, restarted rather than relaxed: a greedy
        # fill can paint itself into a corner, and starting over is simpler
        # than backtracking at this size.
        for _ in range(200):
            placed: list[tuple[float, float]] = []
            for _ in range(count * 200):
                if len(placed) == count:
                    break
                candidate = (
                    self.random.uniform(low[0], high[0]),
                    self.random.uniform(low[1], high[1]),
                )
                if all(math.dist(candidate, other) >= spacing for other in placed):
                    placed.append(candidate)
            if len(placed) == count:
                return placed

        # The high counts do not fall out of rejection sampling.  Nine buckets
        # 3 ft apart fit the section, but only in very nearly a 3x3 grid, and
        # the chance of stumbling on that by drawing points uniformly is not
        # worth waiting for.  So lay out a grid and jitter it by whatever slack
        # the spacing leaves: still a different layout every time, just not a
        # uniform one, which is the honest trade at the top of the range.
        return self.grid_layout(low, high, count, spacing)

    def grid_layout(self, low, high, count: int, spacing: float):
        span = (high[0] - low[0], high[1] - low[1])
        best = None
        for columns in range(1, count + 1):
            rows = math.ceil(count / columns)
            pitch = tuple(
                span[axis] / (steps - 1) if steps > 1 else float("inf")
                for axis, steps in ((0, columns), (1, rows))
            )
            if min(pitch) < spacing:
                continue
            slack = min(pitch[0] - spacing, pitch[1] - spacing)
            if best is None or slack > best[0]:
                best = (slack, columns, rows, pitch)
        if best is None:
            raise GazeboError(
                f"{count} buckets will not fit {spacing:.2f} m apart in the section"
            )

        _, columns, rows, pitch = best
        cells = [(column, row) for column in range(columns) for row in range(rows)]

        placed = []
        for cell in self.random.sample(cells, count):
            position = []
            for axis, (index, steps) in enumerate(
                ((cell[0], columns), (cell[1], rows))
            ):
                if steps == 1:
                    # A single row or column is free to sit anywhere: nothing
                    # on this axis constrains it.
                    position.append(self.random.uniform(low[axis], high[axis]))
                    continue
                jitter = (pitch[axis] - spacing) / 2
                centre = low[axis] + index * pitch[axis]
                offset = self.random.uniform(-jitter, jitter)
                position.append(min(high[axis], max(low[axis], centre + offset)))
            placed.append(tuple(position))
        return placed

    def draw_hoops(self) -> dict[str, tuple[float, float]]:
        """A position for each hoop along the line the drawing puts it on.

        The line spans the full width of the corridor, so the ends are clamped
        by half the hoop's base -- a hoop centered on the very end would stand
        half outside the bales.
        """
        positions = {}
        for hoop in self.hoop_names():
            start = list(self.hoop_param(hoop, "from"))
            end = list(self.hoop_param(hoop, "to"))
            span = math.dist(start, end)
            margin = float(self.hoop_param(hoop, "base_length")) / 2.0
            if span <= 2 * margin:
                positions[hoop] = (
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2,
                )
                continue
            fraction = self.random.uniform(margin / span, 1.0 - margin / span)
            positions[hoop] = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
        return positions

    def draw_gap_bale(self) -> str:
        """Which of the four wall bales stands off-course this draw.

        Exactly one is always parked -- the wall is four bale-widths tall and
        the gap it leaves is what the vehicle drives through -- so this picks
        one name rather than a count or a set.
        """
        return self.random.choice(self.gap_bale_names())

    def apply_gap_bale(self, parked: str) -> None:
        parking = list(self.get_parameter("gap_bales.parking").value)
        for bale in self.gap_bale_names():
            if bale == parked:
                x, y = parking
                yaw = 0.0
            else:
                x, y = self.gap_bale_param(bale, "position")
                yaw = float(self.gap_bale_param(bale, "yaw"))
            self.set_pose(bale, x, y, 0.0, yaw=yaw)

    def apply(self, buckets, hoops) -> None:
        # Guarded rather than relying on bucket_limit() being 0: the parking
        # parameters are read before the loop, and on a course with no bucket
        # section reading them at all is what goes wrong.
        if self.has_buckets():
            parking = list(self.bucket_param("parking"))
            pitch = float(self.bucket_param("parking_pitch"))
            for index in range(self.bucket_limit()):
                if index < len(buckets):
                    x, y = buckets[index]
                else:
                    x, y = parking[0] - pitch * index, parking[1]
                self.set_pose(f"bucket_{index}", x, y, 0.0)
        for hoop, (x, y) in hoops.items():
            self.set_pose(hoop, x, y, 0.0, yaw=float(self.hoop_param(hoop, "yaw")))

    # --------------------------------------------------------------- services

    def failed(self, response, error: Exception):
        """Answer a service call that could not do what was asked.

        Deliberately catches everything.  A callback that raises never sends a
        reply at all, and the caller sits waiting for one that is not coming --
        which is exactly how a missing parameter on a course with no bucket
        section hid as an intermittent timeout.
        """
        response.success = False
        response.message = (
            str(error)
            if isinstance(error, GazeboError)
            else f"{type(error).__name__}: {error}"
        )
        self.get_logger().error(response.message)
        self.get_logger().debug(traceback.format_exc())
        return response

    def reseed(self) -> int:
        seed = int(self.get_parameter("seed").value)
        if seed < 0:
            seed = random.SystemRandom().randrange(1 << 30)
        self.random = random.Random(seed)
        return seed

    def on_randomize(self, _request, response):
        try:
            seed = self.reseed()
            buckets = self.draw_buckets()
            hoops = self.draw_hoops()
            gap_bale = self.draw_gap_bale() if self.has_gap_bales() else None
            self.apply(buckets, hoops)
            if gap_bale is not None:
                self.apply_gap_bale(gap_bale)
        except Exception as error:  # noqa: BLE001 - a service has to answer
            return self.failed(response, error)
        response.success = True
        if not buckets and not hoops and gap_bale is None:
            response.message = "this course varies nothing but the start signal"
        else:
            parts = []
            if buckets:
                closest = min(
                    math.dist(a, b)
                    for index, a in enumerate(buckets)
                    for b in buckets[index + 1 :]
                )
                parts.append(
                    f"{len(buckets)} buckets, closest pair {closest:.2f} m apart "
                    f"({closest - BUCKET_DIAMETER:.2f} m gap, car is "
                    f"{VEHICLE_WIDTH:.2f} m wide)"
                )
            if hoops:
                parts.append(f"{len(hoops)} hoops repositioned")
            if gap_bale is not None:
                parts.append(f"wall gap at {gap_bale}")
            response.message = f"seed {seed}: " + "; ".join(parts)
        self.get_logger().info(response.message)
        return response

    def on_reset(self, _request, response):
        try:
            nominal = []
            if self.has_buckets():
                nominal = [
                    tuple(self.bucket_param(f"nominal.bucket_{index}"))
                    for index in range(int(self.bucket_param("default_count")))
                ]
            hoops = {
                hoop: tuple(self.hoop_param(hoop, "nominal"))
                for hoop in self.hoop_names()
            }
            self.apply(nominal, hoops)
            if self.has_gap_bales():
                self.apply_gap_bale(self.get_parameter("gap_bales.default_gap").value)
            self.show_signal(False)
        except Exception as error:  # noqa: BLE001 - a service has to answer
            return self.failed(response, error)
        response.success = True
        response.message = (
            f"restored the drawn layout: {len(nominal)} buckets, red"
            if nominal
            else "start signal back to red"
        )
        self.get_logger().info(response.message)
        return response

    def on_start_signal(self, request, response):
        try:
            self.show_signal(request.data)
        except Exception as error:  # noqa: BLE001 - a service has to answer
            return self.failed(response, error)
        response.success = True
        response.message = "green" if request.data else "red"
        self.get_logger().info(f"start signal now showing {response.message}")
        return response

    def command_arm(self, angle: float) -> None:
        message = Float64()
        message.data = angle
        self.arm_publisher.publish(message)

    def show_signal(self, green: bool) -> None:
        """Turn the arm pair to red or green, sweeping rather than jumping.

        The signal on the course takes about a second to turn its quarter
        turn, and a detector has to cope with the arm part way round, so the
        arm is turned rather than teleported.

        The setpoint is ramped here and followed by the joint controller in
        the world, which is what keeps the rate honest.  Stepping the model's
        pose with set_pose was the obvious alternative and does not work: each
        call is a `gz service` subprocess costing about 340 ms, so a one second
        sweep fits three poses and arrives as a stutter.  Publishing a setpoint
        costs nothing, so the ramp runs at `SIGNAL_STEP_SECONDS` throughout.

        The ramp is paced by the simulation's clock, not the wall's.  A
        headless world runs at well under real time, so a sweep timed with
        `time.monotonic` lands in the world at whatever the real time factor
        happens to be -- about 160 deg/s here, not the 90 asked for -- and a
        detector under test would see a signal the course never shows.  Wall
        time is still what the loop sleeps on, and still bounds it, since a
        paused world would otherwise never let the sweep finish.

        Setting `signal_sweep_rate` to 0 commands the far end directly, which
        is easier to script a test around.
        """
        target = float(self.signal_param("green_angle" if green else "red_angle"))
        start = self.arm_angle
        rate = math.radians(float(self.get_parameter("signal_sweep_rate").value))
        duration = abs(target - start) / rate if rate > 0 else 0.0

        def sim_now() -> float:
            return self.get_clock().now().nanoseconds / 1e9

        began, wall_began = sim_now(), time.monotonic()
        deadline = wall_began + duration * SIGNAL_WALL_LIMIT + 1.0
        steps = 0
        while True:
            elapsed = sim_now() - began
            overrun = time.monotonic() > deadline
            fraction = 1.0 if duration <= 0 or overrun else min(1.0, elapsed / duration)
            # Kept up to date as we go, so an interrupted sweep leaves the node
            # believing where the arm actually got to.
            self.arm_angle = start + (target - start) * fraction
            self.command_arm(self.arm_angle)
            steps += 1
            if fraction >= 1.0:
                break
            time.sleep(SIGNAL_STEP_SECONDS)

        swept = math.degrees(abs(target - start))
        taken = sim_now() - began
        if swept:
            self.get_logger().info(
                f"signal arm turned {swept:.0f} deg in {taken:.2f} s of sim time "
                f"({swept / taken:.0f} deg/s) over {steps} setpoints"
                + (" -- cut short, the world is not keeping up" if overrun else "")
            )

        # Published once the arm has arrived: this is the ground truth for
        # what the signal *is* showing, and part way round it is showing
        # neither.  A detector may well call it sooner.
        self.publish_signal(green)

    def publish_signal(self, green: bool) -> None:
        message = Bool()
        message.data = green
        self.signal_publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = ObstacleRandomizer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C, or the launch tearing the stack down around us.  Both are
        # how this node is meant to stop, so neither is worth a traceback.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
