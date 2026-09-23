#!/usr/bin/env python3
"""Drive the Speed Course with a trained policy -- in Gazebo and on the car.

ONE node, ONE code path, both places.  What it subscribes to and what it
publishes exist identically on the simulator and on the Orin:

    /zed/zed_node/pose          PoseStamped.  On the car this is the ZED's
                                map-frame pose, which the SDK corrects on loop
                                closure; simulation.launch.py bridges Gazebo's
                                ground truth onto the same topic.
    /arduino_bridge/status      ArduinoStatus.  Ground speed off the spur
                                tachometer, from the Arduino or from
                                sim_vehicle_node.
    /start_signal_detector/go   latched Bool, the run trigger.
    /drive_cmd                  DriveCommand, consumed by arduino_bridge_node
                                on the car and sim_vehicle_node in Gazebo.

DriveCommand rather than a Twist on /cmd_vel, deliberately.  cmd_vel_to_drive
inverts a bicycle model and then divides by a SYMMETRIC +/-0.40 rad steering
limit, but the measured car reaches 0.512 rad left and 0.382 rad right.  Going
through it would throw away a third of the lock on one side and invent lock on
the other.  The policy was trained on the normalized command itself, against
the measured table, so it publishes the normalized command itself.

Localisation, and why a latched anchor:  the ZED's map origin is wherever the
camera booted, so the pose stream has no fixed relation to the course.  On the
`go` signal this node latches the pose it is sitting at and pins it to the
known start pose on the track, exactly as lap_counter_node does with its own
reference.  Everything after that is track-relative.  The car will not be
standing EXACTLY on the surveyed start pose, which is why training randomises
a +/-0.16 m lateral and +/-0.09 rad heading error into every episode.

Safety: the policy's speed is clamped to the rule cap AGAIN here, on the last
line before the wire.  The action scaling already makes that structurally
true, but the limits are a rule we race to, and a rule is worth enforcing
where it can be read off one line of code.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Bool, ColorRGBA
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

from cfr_interfaces.msg import ArduinoStatus, DriveCommand

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track as track_mod  # noqa: E402
from baseline import BaselineDriver  # noqa: E402
from observation import ObservationBuilder, scale_action  # noqa: E402
from policy import NumpyPolicy  # noqa: E402

HERE = Path(__file__).resolve().parent
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_of(pose):
    q = pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))


class FormulaOne(Node):
    def __init__(self, **kwargs):
        super().__init__("formula_one", **kwargs)
        self.declare_parameter("policy", str(HERE / "runs/v1/policy.npz"))
        # "baseline" runs the scripted driver from baseline.py instead of a
        # network.  It takes the same observation and emits the same action,
        # so it exercises every line of this node -- anchoring, the cap, the
        # DriveCommand -- against Gazebo or the car BEFORE a policy exists.
        # When something goes wrong on the day, it is also the fallback that
        # needs no checkpoint.
        self.declare_parameter("driver", "policy")
        self.declare_parameter("config", str(HERE / "config.yaml"))
        self.declare_parameter("repo_root", str(HERE.parents[1]))
        self.declare_parameter("laps", 0)          # 0 = take it from config
        self.declare_parameter("telemetry_period", 3.0)   # s between log lines
        self.declare_parameter("anchor", "signal")  # signal | world
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("status_timeout", 1.0)
        self.declare_parameter("track_frame", "map")
        self.declare_parameter("publish_markers", True)
        # Scales the commanded speed, after the rule cap.  Two uses: a first
        # run on the real car at half pace, and -- with driver:=baseline -- a
        # controlled sweep for finding the speed at which Gazebo stops
        # matching plant.py, which a crashing driver cannot measure.
        self.declare_parameter("speed_scale", 1.0)
        # A pose step larger than this is not the car driving.  1.0 m matches
        # lap_counter's max_step: at 8 m/s and the ZED's 30 Hz a real step is
        # 0.27 m.  Below the second threshold it is treated as the SDK closing
        # a loop (re-localise, keep the run); above it as a teleport or reset
        # (re-localise and start the run over).
        self.declare_parameter("relocalize_step", 1.0)
        self.declare_parameter("reset_run_step", 3.0)

        self.cfg = yaml.safe_load(Path(self.param("config")).read_text())
        self.track = track_mod.build(self.cfg, Path(self.param("repo_root")))
        self.obs = ObservationBuilder(self.track, self.cfg, 1)
        self.mode = self.param("driver")
        if self.mode == "baseline":
            self.policy = None
            self.scripted = BaselineDriver(self.track, self.cfg)
            self.scripted.reset(1)
        else:
            path = Path(self.param("policy"))
            if not path.exists():
                raise SystemExit(
                    f"no policy at {path}.\n"
                    "Train and export one (./train.sh), or run the scripted "
                    "driver with  -p driver:=baseline  /  validate.sh --baseline"
                )
            self.policy = NumpyPolicy.load(path)
            self.scripted = None
        self.laps_target = int(self.param("laps")) or int(self.cfg["env"]["laps"])
        self.residual = float(self.cfg["env"]["steer_residual"])
        self.yaw_filter = float(self.cfg["env"]["yaw_rate_filter"])
        # Defaulted for configs saved before this existed -- same reason as
        # env.py's copy, and the two must agree or the policy sees a
        # differently-filtered input on the car than it trained on.
        self.accel_filter = float(self.cfg["env"].get("speed_rate_filter", 0.25))
        self.control_period = 1.0 / float(self.cfg["env"]["control_hz"])
        self.frame = self.param("track_frame")

        self.start_index = int(
            np.clip(np.searchsorted(self.track.s, self.track.start_station),
                    0, len(self.track.s) - 1)
        )

        self.pose = None
        self.pose_time = None
        self.prev_pose = None
        self.status = None
        self.status_time = None
        self.go = False
        self.manual_stop = False
        self.finished = False
        # STOPPING IS A PHASE, NOT A FLAG.  The car has no brakes: from 5.2
        # m/s it coasts about 15 m before it is at rest, and that 15 m is
        # still inside a 0.92 m corridor.  Cutting steering at the line -- as
        # this node used to -- coasts the car straight into the bales it was
        # about to turn away from.  So the throttle goes to zero and the
        # steering keeps running until the car has actually stopped.
        self.stopping = False
        self.stopping_since = None
        self.stop_speed = float(self.cfg["env"]["stop_speed"])
        self.stop_timeout = float(self.cfg["env"]["stop_timeout_s"])
        self.race_time = None
        self.run_started_at = None
        self.lap_started_at = None
        self.last_lap_time = 0.0
        self.anchor = None          # (dx, dy, dyaw) once latched
        self.prev_action = np.zeros((1, 2))
        self.last_steer = np.zeros(1)
        # Yaw rate for the steering prior.  Differenced from the pose and
        # smoothed exactly as env.py does it, because the prior is the same
        # code and a rate estimated differently here is a policy driving on an
        # input it never saw in training.
        self.prev_track_yaw = None
        self.prev_yaw_stamp = None
        self.yaw_rate = np.zeros(1)
        # Achieved dv/dt, the same channel env.py feeds -- see observation.py
        # for why the policy needs it.  Differenced off the POSE stamps, so it
        # sits on the same clock as the yaw rate rather than on wall time.
        self.speed_rate = np.zeros(1)
        self.prev_obs_speed = None
        self.prev_speed_stamp = None
        self.last_track_xy = None
        self.station = None
        self.distance = 0.0
        self.laps_done = 0
        self.last_report = 0.0

        # BEST_EFFORT deliberately.  A best-effort subscriber accepts a
        # reliable publisher, but a reliable subscriber silently receives
        # NOTHING from a best-effort one -- and the pose publisher is reliable
        # in Gazebo and best-effort in some ZED wrapper builds.  Subscribing
        # this way is the only setting that works against both.
        self.create_subscription(PoseStamped, "/zed/zed_node/pose", self.on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(ArduinoStatus, "/arduino_bridge/status",
                                 self.on_status, 10)
        self.create_subscription(Bool, "/start_signal_detector/go", self.on_go, LATCHED)
        self.create_subscription(Bool, "/lap_counter/done", self.on_done, LATCHED)
        self.create_service(SetBool, "~/manual_start", self.on_manual)

        self.drive = self.create_publisher(DriveCommand, "/drive_cmd",
                                           qos_profile_sensor_data)
        self.markers = self.create_publisher(MarkerArray, "~/markers", LATCHED)
        self.path_pub = self.create_publisher(PathMsg, "~/centerline", LATCHED)
        self.car_pub = self.create_publisher(PoseStamped, "~/car", 10)
        if self.param("publish_markers"):
            self.publish_track()

        self.create_timer(self.control_period, self.tick)
        self.get_logger().info(
            f"formula_one ready: {self.laps_target} laps, "
            f"cap {self.cfg['track']['v_hairpin']}/{self.cfg['track']['v_straight']} m/s, "
            f"driver {self.mode}"
            + ("" if self.scripted else f" ({Path(self.param('policy')).name})")
            + ". Waiting for the start signal."
        )

    def param(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------ callbacks

    def on_pose(self, msg):
        self.prev_pose = self.pose
        self.pose = msg
        self.pose_time = self.get_clock().now()

    def on_status(self, msg):
        self.status = msg
        self.status_time = self.get_clock().now()

    def on_go(self, msg):
        if msg.data and not self.go:
            self.get_logger().info("GREEN -- anchoring and going")
        self.go = bool(msg.data)

    def on_done(self, msg):
        # The official counter is allowed to end the run even if our own
        # station bookkeeping disagrees with it.
        if msg.data and not self.finished and not self.stopping:
            self.get_logger().info("lap_counter reports done -- coasting to a stop")
            self.begin_stopping()

    def on_manual(self, request, response):
        self.go = bool(request.data)
        self.manual_stop = not request.data
        self.stopping = False
        self.stopping_since = None
        self.race_time = None
        self.run_started_at = None
        self.lap_started_at = None
        self.last_lap_time = 0.0
        if request.data:
            self.finished = False
            self.anchor = None
        response.success = True
        response.message = "manual start" if request.data else "manual stop"

        return response

    # ------------------------------------------------------------- localise

    def latch(self, x, y, yaw):
        i = self.start_index
        yaw_start = math.atan2(self.track.ty[i], self.track.tx[i])
        self.start_pose = (float(self.track.x[i]), float(self.track.y[i]), yaw_start)
        if self.param("anchor") == "world":
            self.anchor = (0.0, 0.0, 0.0, 0.0)
            self.get_logger().info("anchor=world: pose is taken as track coordinates")
            return
        self.anchor = (x, y, yaw, yaw_start - yaw)
        self.get_logger().info(
            f"anchored: ZED ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f} deg) "
            f"-> track ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f}, "
            f"{math.degrees(yaw_start):.1f} deg)"
        )

    def to_track(self, x, y, yaw):
        if self.param("anchor") == "world" or self.anchor is None:
            return x, y, yaw
        x_l, y_l, _, dyaw = self.anchor
        c, s = math.cos(dyaw), math.sin(dyaw)
        dx, dy = x - x_l, y - y_l
        return (
            self.start_pose[0] + c * dx - s * dy,
            self.start_pose[1] + s * dx + c * dy,
            math.atan2(math.sin(yaw + dyaw), math.cos(yaw + dyaw)),
        )

    def measured_speed(self):
        """Tachometer when it is fresh, differenced pose when it is not.

        The training env zeroes anything below 0.3 m/s because the spur
        tachometer has one magnet and a 0.4 s stall timeout and genuinely
        cannot see a crawl.  The same floor is applied here so the policy
        reads the same instrument it learned on.
        """
        now = self.get_clock().now()
        fresh = self.status_time is not None and (
            (now - self.status_time).nanoseconds * 1e-9 < self.param("status_timeout")
        )
        if fresh and self.status is not None and self.status.link_ok:
            speed = abs(float(self.status.speed))
        elif self.prev_pose is not None:
            dt = max(
                (rclpy.time.Time.from_msg(self.pose.header.stamp)
                 - rclpy.time.Time.from_msg(self.prev_pose.header.stamp)
                 ).nanoseconds * 1e-9, 1e-3)
            speed = math.hypot(
                self.pose.pose.position.x - self.prev_pose.pose.position.x,
                self.pose.pose.position.y - self.prev_pose.pose.position.y,
            ) / dt
        else:
            speed = 0.0
        return 0.0 if speed < 0.3 else speed

    # ------------------------------------------------------------- the loop

    def send(self, steering, velocity, ready=True):
        msg = DriveCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        # auto_ready must be true, with neutral steering and zero speed,
        # before the Arduino will leave AUTO_ARMED -- so it is held true from
        # the moment this node comes up, not from the moment it starts driving.
        msg.auto_ready = bool(ready)
        msg.steering = float(np.clip(steering, -1.0, 1.0))
        msg.velocity = float(velocity)
        self.drive.publish(msg)

    def begin_stopping(self):
        """Throttle off, steering live, until the car is actually at rest."""
        self.stopping = True
        self.stopping_since = self.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        now = self.get_clock().now()
        stale = self.pose_time is None or (
            (now - self.pose_time).nanoseconds * 1e-9 > self.param("pose_timeout")
        )
        if self.finished or self.manual_stop or not self.go or stale:
            if stale and self.go and not self.finished:
                self.get_logger().warn("pose is stale -- commanding neutral",
                                       throttle_duration_sec=2.0)
            self.send(0.0, 0.0)
            return

        x_raw = self.pose.pose.position.x
        y_raw = self.pose.pose.position.y
        yaw_raw = yaw_of(self.pose.pose)
        if self.anchor is None:
            self.latch(x_raw, y_raw, yaw_raw)
            self.obs.set_station(np.array([True]),
                                 np.array([self.track.start_station]))
            self.station = self.track.start_station
            self.prev_track_yaw = None
            self.prev_yaw_stamp = None
            self.yaw_rate = np.zeros(1)

        x, y, yaw = self.to_track(x_raw, y_raw, yaw_raw)
        speed = self.measured_speed()

        # Re-localise after a jump.  The station is tracked incrementally off
        # a hint, which is right while the car drives and wrong the moment it
        # is teleported: the hint stays where the car used to be, the search
        # window never reaches the new position, and every command after that
        # is computed for a place the car is not.  The symptom is a car that
        # ignores the course and steers into the bales, which is exactly what
        # a reset used to produce.
        if self.last_track_xy is not None:
            step = math.hypot(x - self.last_track_xy[0], y - self.last_track_xy[1])
            if step > float(self.param("relocalize_step")):
                idx, station = self.track.locate(
                    np.array([x]), np.array([y]), np.array([yaw])
                )
                self.obs.set_station(np.array([True]), station)
                self.station = float(station[0])
                self.prev_track_yaw = None
                self.prev_yaw_stamp = None
                self.yaw_rate = np.zeros(1)
                self.speed_rate = np.zeros(1)
                self.prev_obs_speed = None
                self.prev_speed_stamp = None
                self.prev_action = np.zeros((1, 2))
                self.last_steer = np.zeros(1)
                if self.scripted is not None:
                    self.scripted.reset(1)
                if step > float(self.param("reset_run_step")):
                    self.distance = 0.0
                    self.laps_done = 0
                    self.finished = False
                    self.stopping = False
                    self.stopping_since = None
                    self.race_time = None
                    self.run_started_at = None
                    self.lap_started_at = None
                    self.last_lap_time = 0.0
                    self.get_logger().info(
                        f"moved {step:.1f} m -- treating as a reset, "
                        f"restarting at station {self.station:.1f} m"
                    )
                else:
                    self.get_logger().warn(
                        f"pose jumped {step:.2f} m -- re-localised to station "
                        f"{self.station:.1f} m, run continues"
                    )
        self.last_track_xy = (x, y)

        # Yaw rate over the interval the two poses ACTUALLY span, taken from
        # their stamps -- not over the control period.
        #
        # The pose stream and the control loop do not run at the same rate and
        # are not meant to: Gazebo's PosePublisher is 30 Hz, the ZED is 15-30
        # Hz, and this node controls at 20 Hz to match training.  So the two
        # poses a tick differences are 33 or 67 ms apart, alternating, never
        # the 50 ms a fixed divisor assumes.  Dividing by the control period
        # puts a +/-33% oscillation straight into the yaw rate, and the
        # steering prior extrapolates that over a 0.24 s horizon.  In training
        # the pose updates exactly once per step, so the error does not exist
        # there and the policy has never seen it.
        stamp = rclpy.time.Time.from_msg(self.pose.header.stamp).nanoseconds * 1e-9
        if self.prev_track_yaw is None or self.prev_yaw_stamp is None:
            rate = float(self.yaw_rate[0])
        elif stamp <= self.prev_yaw_stamp:
            # Same pose as last tick: nothing new to difference, so hold the
            # estimate rather than reporting a spurious zero.
            rate = float(self.yaw_rate[0])
        else:
            step = math.atan2(math.sin(yaw - self.prev_track_yaw),
                              math.cos(yaw - self.prev_track_yaw))
            # A car at full lock and full speed turns at under 6 rad/s;
            # anything past that is a pose jump, not a yaw rate.
            rate = max(-8.0, min(8.0, step / (stamp - self.prev_yaw_stamp)))
            self.prev_track_yaw = yaw
            self.prev_yaw_stamp = stamp
        if self.prev_track_yaw is None:
            self.prev_track_yaw = yaw
            self.prev_yaw_stamp = stamp
        self.yaw_rate = (1 - self.yaw_filter) * self.yaw_rate + self.yaw_filter * rate

        stamp_s = rclpy.time.Time.from_msg(self.pose.header.stamp).nanoseconds * 1e-9
        if self.prev_obs_speed is None or self.prev_speed_stamp is None \
                or stamp_s <= self.prev_speed_stamp:
            self.prev_obs_speed, self.prev_speed_stamp = speed, stamp_s
        else:
            raw = (speed - self.prev_obs_speed) / (stamp_s - self.prev_speed_stamp)
            raw = max(-20.0, min(20.0, raw))
            self.speed_rate = ((1 - self.accel_filter) * self.speed_rate
                               + self.accel_filter * raw)
            self.prev_obs_speed, self.prev_speed_stamp = speed, stamp_s

        clock_s = now.nanoseconds * 1e-9
        if self.run_started_at is None:
            self.run_started_at = clock_s
            self.lap_started_at = clock_s
        lap_progress = float(np.clip(
            (self.distance - self.laps_done * self.track.length)
            / self.track.length, 0.0, 1.0))
        lap_state = np.array([[lap_progress,
                               clock_s - self.lap_started_at,
                               self.last_lap_time]])
        obs, frame = self.obs.compute(
            np.array([x]), np.array([y]), np.array([yaw]),
            np.array([speed]), self.yaw_rate, self.speed_rate, self.prev_action,
            self.last_steer, lap_state,
        )
        v_cap = frame["v_cap"]
        if self.scripted is not None:
            action = self.scripted.act(frame["station"], np.array([speed]), v_cap,
                                       frame["v_floor"])
        else:
            action = self.policy.act(obs)
        steer, velocity = scale_action(action, v_cap, frame["steer_ff"],
                                       self.residual, frame["v_floor"])
        # Belt and braces: the rule limit, enforced again on the way out.
        velocity = np.minimum(velocity, v_cap) * float(self.param("speed_scale"))
        self.prev_action = np.clip(action, -1.0, 1.0)
        # The prior predicts forward through the command already in flight, so
        # it has to be the REALISED command -- after scale_action folded in
        # the residual -- not the network's raw output.
        self.last_steer = np.asarray(steer, dtype=float).reshape(1)

        advance = (frame["station"][0] - self.station + self.track.length / 2) \
            % self.track.length - self.track.length / 2
        if abs(advance) < 2.0:
            self.distance += advance
        self.station = float(frame["station"][0])
        laps = int(max(self.distance, 0.0) // self.track.length)
        if laps > self.laps_done:
            self.laps_done = laps
            lap_time = clock_s - self.lap_started_at
            delta = (f", {self.last_lap_time - lap_time:+.2f} s on the last"
                     if self.last_lap_time > 0.0 else "")
            self.get_logger().info(
                f"lap {laps} of {self.laps_target}  {lap_time:.2f} s{delta}")
            self.last_lap_time = lap_time
            self.lap_started_at = clock_s
        if not self.stopping and self.distance >= self.laps_target * self.track.length:
            self.race_time = clock_s - (self.run_started_at or clock_s)
            self.get_logger().info(
                f"FINISHED {self.laps_target} laps in {self.race_time:.2f} s "
                f"({self.distance:.1f} m) -- coasting to a stop")
            self.begin_stopping()

        if self.stopping:
            # Throttle off, steering still live.  The car is coasting and is
            # still in the corridor, so it still has to be driven.
            since = clock_s - (self.stopping_since or clock_s)
            if speed <= self.stop_speed or since >= self.stop_timeout:
                self.finished = True
                self.get_logger().info(
                    f"STOPPED after {since:.2f} s and "
                    f"{self.distance - self.laps_target * self.track.length:.1f} m "
                    f"past the line"
                    + ("" if speed <= self.stop_speed else "  (TIMED OUT, still moving)"))
                self.send(0.0, 0.0)
                return
            velocity = np.zeros_like(velocity)

        self.send(steer[0], velocity[0])
        self.publish_car(x, y, yaw, frame, speed, velocity[0])
        # A throttled line of telemetry, so a run that goes wrong says where
        # and in what state.  Without it a car wedged against a bale is
        # indistinguishable in the log from a car that never started: both
        # are silence after the last lap message.
        self.get_logger().info(
            f"station {frame['station'][0]:6.1f} m  lap {self.laps_done}/"
            f"{self.laps_target}  {speed:4.2f}/{frame['v_cap'][0]:.2f} m/s  "
            f"cmd {steer[0]:+.2f}  clear {frame['clearance'][0]:+.3f} m  "
            f"cte {frame['lateral'][0]:+.3f} m",
            throttle_duration_sec=float(self.param("telemetry_period")),
        )

    # -------------------------------------------------------------- display

    def publish_track(self):
        t = self.track
        path = PathMsg()
        path.header.frame_id = self.frame
        for i in range(0, len(t.s), 4):
            p = PoseStamped()
            p.header.frame_id = self.frame
            p.pose.position.x = float(t.x[i])
            p.pose.position.y = float(t.y[i])
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        self.path_pub.publish(path)

        array = MarkerArray()
        cap = Marker()
        cap.header.frame_id = self.frame
        cap.ns, cap.id, cap.type, cap.action = "speed_cap", 0, Marker.POINTS, Marker.ADD
        cap.scale.x = cap.scale.y = 0.10
        cap.pose.orientation.w = 1.0
        lo = float(self.cfg["track"]["v_hairpin"])
        hi = float(self.cfg["track"]["v_straight"])
        for i in range(0, len(t.s), 2):
            cap.points.append(Point(x=float(t.x[i]), y=float(t.y[i]), z=0.02))
            f = (t.v_cap[i] - lo) / max(hi - lo, 1e-6)
            cap.colors.append(ColorRGBA(r=float(1 - f), g=float(f), b=0.1, a=1.0))
        array.markers.append(cap)
        self.markers.publish(array)

    def publish_car(self, x, y, yaw, frame, speed, commanded):
        pose = PoseStamped()
        pose.header.frame_id = self.frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        self.car_pub.publish(pose)

        if not self.param("publish_markers"):
            return
        text = Marker()
        text.header.frame_id = self.frame
        text.ns, text.id, text.type, text.action = "hud", 1, Marker.TEXT_VIEW_FACING, Marker.ADD
        text.pose.position.x, text.pose.position.y, text.pose.position.z = 20.0, 0.0, 2.0
        text.pose.orientation.w = 1.0
        text.scale.z = 0.7
        over = speed - float(frame["v_cap"][0])
        text.color = ColorRGBA(
            r=1.0 if over > 0.05 else 0.2, g=0.2 if over > 0.05 else 1.0, b=0.2, a=1.0
        )
        text.text = (
            f"lap {self.laps_done}/{self.laps_target}   "
            f"{speed:.2f} / {frame['v_cap'][0]:.2f} m/s   "
            f"cmd {commanded:.2f}   "
            f"clearance {frame['clearance'][0]:.2f} m   "
            f"station {frame['station'][0]:.1f} m"
        )
        self.markers.publish(MarkerArray(markers=[text]))


def main():
    rclpy.init()
    node = FormulaOne()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.send(0.0, 0.0, ready=False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
