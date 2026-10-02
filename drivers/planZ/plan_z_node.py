#!/usr/bin/env python3
"""Plan Z: the backup driver for both courses, in Gazebo and on the car.

No policy and no map of the obstacles: planner.py steers from the segmented
ZED cloud, with the course's route (routes/*.yaml) as a hint of which way it
goes next.  One node, one code path, both places and both courses:

    /zed/zed_node/point_cloud/cloud_registered   PointCloud2, camera frame
    /zed/zed_node/pose           PoseStamped: position and yaw (the route
                                 hint, and the memory of what the camera has
                                 passed), pitch and roll if there is no IMU
    /zed/zed_node/imu/data       Imu: pitch and roll to level the cloud.  The
                                 car's pose is flat (cfr_zed2i.yaml
                                 two_d_mode), so on the car this is what
                                 keeps a ramp from reading as a wall
    /arduino_bridge/status       ArduinoStatus: tachometer speed, and the
                                 Manual Start bit, a run trigger
    /start_signal_detector/go    latched Bool, the other run trigger
    /lap_counter/done            latched Bool, the end of the run
    /drive_cmd                   DriveCommand out
    /left_wall_follower/manual_go  latched Bool out on a manual start, so
                                 lap_counter arms as it does on the signal

The cloud goes through the compiled segmenter (cloud_segmentation.py, the
one cloud_segmentation_node runs), then `scan_from_segmentation` and the
segmenter's gates -- the same path rl/obstacleRacer drives on.

    ros2 launch drivers/planZ/plan_z.launch.py course:=obstacle

Its ROS name is `obstacle_racer` on the Obstacle Course and `formula_one` on
the Speed Course (plan_z.launch.py sets it), so record_run.py, the Run Lab
and rl/obstacleRacer/gazebo_check.py work with it unchanged; the telemetry's
`driver` field says plan_z.

Every knob in config.yaml can be set at launch: `knobs:="steer_trim=0.02
sections.helical_ramp=0.8"`.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from cfr_interfaces.msg import ArduinoStatus, DriveCommand

try:
    from cfr_interfaces.msg import DriverTelemetry
except ImportError:  # pragma: no cover - depends on the installed workspace
    DriverTelemetry = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from planner import Planner, PoseDrift  # noqa: E402
from route import Route, load_config  # noqa: E402

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
TACH_FLOOR = 0.30  # m/s: ArduinoStatus.speed cannot resolve below this


def _segmenter():
    """cloud_segmentation.py from the installed cfr_arduino_bridge."""
    try:
        from ament_index_python.packages import get_package_prefix

        sys.path.insert(
            0,
            str(
                Path(get_package_prefix("cfr_arduino_bridge"))
                / "lib"
                / "cfr_arduino_bridge"
            ),
        )
    except Exception:  # noqa: BLE001 - fall back to the source tree
        sys.path.insert(
            0, str(HERE.parents[1] / "jetson" / "cfr_arduino_bridge" / "src")
        )
    import cloud_segmentation

    return cloud_segmentation


def attitude(q):
    """(pitch nose-down +, roll left-up +, yaw) from a REP-103 orientation."""
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    roll = math.atan2(
        2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    )
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return pitch, roll, yaw


class PlanZ(Node):
    def __init__(self):
        super().__init__("plan_z")
        self.declare_parameter("course", "obstacle")  # obstacle | speed | a route file
        self.declare_parameter("config", str(HERE / "config.yaml"))
        self.declare_parameter("knobs", "")  # "name=value name=value"
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("cloud_timeout", 0.5)
        self.declare_parameter("yaw_rate_filter", 0.5)
        # Reliable, depth 1, as cloud_segmentation reads the same topic: a
        # whole cloud is several MB of UDP fragments, and best effort drops
        # most of them between processes.
        self.declare_parameter("cloud_reliable", True)
        # Start on the Arduino's Manual Start bit as well as the signal.
        self.declare_parameter("arduino_manual_start", True)

        overrides = dict(pair.split("=", 1) for pair in self.param("knobs").split())
        self.k = load_config(self.param("config"), overrides)
        self.route = (
            Route(self.param("course"), self.k) if self.k["route_hint"] else None
        )
        self.planner = Planner(self.k, self.route)
        self.seg = _segmenter()
        self.drift = PoseDrift(
            float(self.k["inject_pose_drift"]),
            math.radians(float(self.k["inject_yaw_drift_deg"])),
        )
        self.dt = 1.0 / float(self.k["control_hz"])

        self.poses = deque(
            maxlen=120
        )  # (stamp, chassis pose) for the cloud's capture time
        self.pose = None
        self.pose_time = None
        self.level = None  # (pitch, roll), from the pose
        self.imu_level = None
        self.imu_time = None
        self.prev_yaw = None
        self.prev_stamp = None
        self.yaw_rate = 0.0
        self.last_steer = 0.0
        self.cloud_time = None
        self.status = None
        self.go = False
        self.done = False
        self.started_at = None
        self.prev_manual_bit = None  # first status seeds it: edges only
        self.level_said = None

        self.create_subscription(
            PointCloud2,
            "/zed/zed_node/point_cloud/cloud_registered",
            self.on_cloud,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            if self.param("cloud_reliable")
            else qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
        )
        self.create_subscription(
            Imu, "/zed/zed_node/imu/data", self.on_imu, qos_profile_sensor_data
        )
        self.create_subscription(
            ArduinoStatus, "/arduino_bridge/status", self.on_status, 10
        )
        self.create_subscription(Bool, "/start_signal_detector/go", self.on_go, LATCHED)
        self.create_subscription(Bool, "/lap_counter/done", self.on_done, LATCHED)
        self.create_service(SetBool, "~/manual_start", self.on_manual)
        self.manual_go = self.create_publisher(
            Bool, "/left_wall_follower/manual_go", LATCHED
        )
        self.drive = self.create_publisher(
            DriveCommand, "/drive_cmd", qos_profile_sensor_data
        )
        self.telemetry = (
            self.create_publisher(DriverTelemetry, "~/telemetry", 50)
            if DriverTelemetry
            else None
        )
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(
            f"plan_z ready: course {self.param('course')}, speed_scale "
            f"{self.k['speed_scale']}, v_max {self.k['v_max']} m/s"
            + (f", knobs {overrides}" if overrides else "")
            + ".  Waiting for the start signal."
        )

    def param(self, name):
        return self.get_parameter(name).value

    def now(self):
        """Seconds on the node's clock: sim time in Gazebo.  Staleness is
        judged on it, not on the wall clock -- Gazebo rendering the ZED runs
        at a fraction of real time."""
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------- inputs

    def on_pose(self, msg):
        pitch, roll, yaw = attitude(msg.pose.orientation)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_yaw is not None and stamp > self.prev_stamp:
            dyaw = (yaw - self.prev_yaw + math.pi) % (2 * math.pi) - math.pi
            rate = dyaw / (stamp - self.prev_stamp)
            a = float(self.param("yaw_rate_filter"))
            self.yaw_rate = (1 - a) * self.yaw_rate + a * max(-8.0, min(8.0, rate))
        self.prev_yaw, self.prev_stamp = yaw, stamp
        # The chassis center: the pose's own point may be the camera.
        ahead = float(self.k["pose_forward"])
        pose = self.drift(
            (
                msg.pose.position.x - ahead * math.cos(yaw),
                msg.pose.position.y - ahead * math.sin(yaw),
                yaw,
            )
        )
        self.pose = pose
        self.level = (pitch, roll)
        self.pose_time = self.now()
        self.poses.append((stamp, pose))

    def on_imu(self, msg):
        pitch, roll, _ = attitude(msg.orientation)
        self.imu_level = (pitch, roll)
        self.imu_time = self.now()

    def leveling(self):
        """(pitch, roll) to level the cloud with, and where it came from."""
        source = str(self.k["level_source"])
        fresh = self.imu_time is not None and self.now() - self.imu_time < 0.2
        if source in ("auto", "imu") and fresh:
            return self.imu_level, "imu"
        if source in ("auto", "pose") and self.level is not None:
            return self.level, "pose"
        return (0.0, 0.0), "none"

    def on_cloud(self, msg):
        if self.pose is None:
            return
        pts = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        stride = max(1, int(self.k["cloud_stride"]))
        pts = np.asarray(pts, dtype=np.float64)[::stride]
        (pitch, roll), source = self.leveling()
        if source != self.level_said:
            self.level_said = source
            log = self.get_logger().info if source == "imu" else self.get_logger().warn
            log(
                {
                    "imu": "leveling the cloud on the ZED IMU",
                    "pose": "leveling the cloud on the pose's pitch and roll (no IMU) -- "
                    "flat on the car, where the pose is 2D",
                    "none": "no attitude -- the cloud is taken as level",
                }[source]
            )
        seg = self.seg.segment(
            pts,
            pitch + math.radians(float(self.k["camera_pitch_deg"])),
            roll + math.radians(float(self.k["camera_roll_deg"])),
        )
        scan = self.seg.scan_from_segmentation(
            seg,
            int(self.k["scan_bins"]),
            float(self.k["fov_deg"]),
            float(self.k["max_range"]),
            float(self.k["min_range"]),
        )
        gates = [
            (g.kind, g.center[0], g.center[1], -g.axis[1], g.axis[0], 0.5 * g.span)
            for g in seg.gates
        ]
        # The pose the frame was taken at: by its stamp if the pose stream
        # reaches back that far, else a typical latency ago.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if (
            not self.poses
            or stamp < self.poses[0][0]
            or stamp > self.poses[-1][0] + 0.5
        ):
            stamp = self.poses[-1][0] - float(self.k["cloud_latency_s"])
        pose = min(self.poses, key=lambda entry: abs(entry[0] - stamp))[1]
        self.cloud_time = self.now()
        if self.go:
            self.planner.observe(
                np.asarray(scan, dtype=np.float64), gates, pose, self.cloud_time
            )

    def on_status(self, msg):
        self.status = msg
        pressed = bool(msg.link_ok and msg.manual_start)
        rising = (
            self.prev_manual_bit is not None and pressed and not self.prev_manual_bit
        )
        self.prev_manual_bit = pressed
        if rising and self.param("arduino_manual_start") and not self.done:
            self.begin("Arduino Manual Start")

    def begin(self, source, manual=True):
        """Start a run now, unless one is already under way."""
        if self.go or self.pose is None:
            return
        self.get_logger().info(f"{source} -- going")
        self.go = True
        self.done = False
        self.started_at = self.now()
        # The route is tied to the pose here: the car is on the start line.
        self.planner.reset(self.pose, self.started_at)
        if manual:
            self.manual_go.publish(Bool(data=True))

    def on_go(self, msg):
        if msg.data:
            self.begin("GREEN", manual=False)

    def on_done(self, msg):
        if msg.data and not self.done:
            self.get_logger().info("lap_counter reports done -- throttle off")
        self.done = self.done or bool(msg.data)

    def on_manual(self, request, response):
        if request.data:
            self.begin("manual start service")
        else:
            self.go = False
            self.started_at = None
            self.manual_go.publish(Bool(data=False))
        response.success = True
        response.message = "manual start" if request.data else "manual stop"
        return response

    # ---------------------------------------------------------------- loop

    def send(self, steering, velocity):
        msg = DriveCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        # Held true from startup: the Arduino only leaves AUTO_ARMED on a
        # ready command with neutral steering and zero speed.
        msg.auto_ready = True
        msg.steering = float(np.clip(steering, -1.0, 1.0))
        msg.velocity = float(velocity)
        self.drive.publish(msg)

    def tick(self):
        now = self.now()
        blind = self.pose_time is None or now - self.pose_time > self.param(
            "pose_timeout"
        )
        stale = (
            blind
            or self.cloud_time is None
            or now - self.cloud_time > self.param("cloud_timeout")
        )
        speed = float(self.status.speed) if self.status is not None else 0.0
        if abs(speed) < TACH_FLOOR:
            speed = 0.0
        # With the cloud late but the pose live, the throttle comes off and
        # the wheel goes on being steered along the path in hand, against
        # the walls remembered: the car has no brakes, and a wheel put
        # straight in a bend at speed is a car in the bales a second later.
        # With no pose there is nothing to steer by: the wheel is held.
        steer, velocity = self.last_steer if self.go else 0.0, 0.0
        if self.go and not blind:
            steer, velocity = self.planner.step(self.pose, speed, self.yaw_rate, now)
            self.last_steer = steer
        # Steering stays live after the finish, while the car coasts.
        velocity_out = velocity if (self.go and not self.done and not stale) else 0.0
        self.send(steer, velocity_out)
        if stale and self.go and not self.done:
            pose_age = (
                now - self.pose_time if self.pose_time is not None else float("inf")
            )
            cloud_age = (
                now - self.cloud_time if self.cloud_time is not None else float("inf")
            )
            self.get_logger().warn(
                f"pose age {pose_age:.2f} s, cloud age {cloud_age:.2f} s -- throttle off",
                throttle_duration_sec=2.0,
            )
        if self.go and not stale:
            info = self.planner.info
            section = self.route.section[self.route.i] if self.route else "-"
            shift = self.route.align_shift if self.route else (0.0, 0.0)
            turn = math.degrees(self.route.align_yaw) if self.route else 0.0
            self.get_logger().info(
                f"{section}  {speed:.2f} m/s  cmd {velocity_out:.2f}  steer {steer:+.2f}  "
                f"{info.get('mode', '-')}  offset {info.get('offset', 0.0):+.2f}  "
                f"trim {self.planner.trim_est:+.3f}  lag {self.planner.tau:.2f}  "
                f"camera {math.degrees(self.planner.cam_yaw_est):+.1f}  "
                f"route {shift[0]:+.2f} {shift[1]:+.2f} {turn:+.1f}  "
                f"hoops {len(self.planner.hoops)}  reversals {len(self.planner.events)}",
                throttle_duration_sec=1.0,
            )

        if self.telemetry is not None:
            t = DriverTelemetry()
            t.header.stamp = self.get_clock().now().to_msg()
            t.state = (
                DriverTelemetry.STATE_STALE
                if (stale and self.go)
                else DriverTelemetry.STATE_STOPPING
                if self.done
                else DriverTelemetry.STATE_RUNNING
                if self.go
                else DriverTelemetry.STATE_WAITING
            )
            t.driver = "plan_z"
            t.race_time = float(now - self.started_at) if self.started_at else 0.0
            if self.pose is not None:
                t.x, t.y, t.yaw = (float(v) for v in self.pose)
            if self.route is not None:
                t.station = float(self.route.i * self.route.spacing)
                t.v_cap = float(self.route.v[self.route.i])
            info = self.planner.info
            t.cross_track = float(info.get("offset", 0.0))
            t.heading_error = float(info.get("turn", 0.0))
            t.speed = speed
            t.yaw_rate = float(self.yaw_rate)
            t.action = [float(info.get("want", 0.0)), float(info.get("free", 0.0))]
            t.steer_ff = float(info.get("bend", 0.0))
            t.steer_cmd = float(steer)
            t.velocity_cmd = float(velocity_out)
            t.speed_scale = float(self.k["speed_scale"])
            t.pose_age = float(now - self.pose_time) if self.pose_time else -1.0
            t.speed_from_tach = self.status is not None
            # The scan the planner last saw first, as rl/obstacleRacer's
            # telemetry carries it, for gazebo_check.py and the Run Lab.
            scan = self.planner.scan
            t.observation = [float(v) for v in scan] if scan is not None else []
            self.telemetry.publish(t)


def main():
    rclpy.init()
    node = PlanZ()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.send(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
