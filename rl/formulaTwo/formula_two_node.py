#!/usr/bin/env python3
"""Drive the Speed Course with a formulaTwo policy -- in Gazebo and on the car.

formulaOne's node (formula_one_node.py), plus the ZED depth stream and a
watchdog on it.  Everything formulaOne's node does -- the latched start
anchor, the tachometer speed, the rule cap enforced on the wire, the stopping
phase after the last lap, ~/telemetry -- is unchanged here.  The comments that
explain those are kept because they are still the reasons.

    /zed/zed_node/pose                    PoseStamped, map-frame pose
    /arduino_bridge/status                ArduinoStatus, tachometer speed
    /start_signal_detector/go             latched Bool, the run trigger
    /zed/zed_node/depth/depth_registered  Image, depth in metres (32FC1) or mm
                                          (16UC1).  Gazebo's rgbd_camera is
                                          bridged onto the same name.
    <camera_info_topic>                   CameraInfo for that depth image
    /drive_cmd                            DriveCommand out

THE DEPTH PATH IS THE TRAINING CODE.  Each image is sampled on the canonical
grid (perception.Camera.sample_depth), reduced to the 64-beam virtual LiDAR
(depth_to_scan), encoded and pushed onto the same 4-frame ScanStack env.py
uses.  Frames closer together than `depth_min_period` are dropped, because
the policy was trained on a 10-15 Hz camera, and a stack of four frames taken
33 ms apart is not the 0.33 s of history it learned from.

THE POLICY CANNOT DRIVE WITHOUT DEPTH.  Measured in simulation: a stale,
all-invalid or never-filled scan crashes the 40M policy on 100% of runs, even
on the nominal car with a perfect map -- a frozen scan is an input it never
saw (frame age in training: p99 0.22 s, max 0.46 s).  So:

  * it does not move until a real depth frame has arrived, go or no go;
  * past `depth_hold_after` (0.3 s) without a fresh frame the network is
    taken off the wheel -- the centerline prior steers at the speed floor --
    and the policy resumes, on a refilled stack, if depth returns in time.
    Letting the network drive on a frozen scan instead is what costs: stopping
    at 1.0 s with the network still driving crashed 44% of runs at 25%
    randomisation, against 4.7% with the hold (1.6% stopping at 0.3 s);
  * depth is LOST when the newest frame is older than `depth_timeout`
    (1.0 s), or when more than `depth_invalid_fraction` (0.6) of the beams
    are invalid for `depth_invalid_frames` (3) frames in a row -- normal
    frames peak at 45% under full training randomisation;
  * on loss, `depth_fallback` decides:
      stop  (default)  throttle to zero, the centerline prior keeps steering,
                       the car coasts to rest and the run is over.  Measured:
                       100% at rest with no contact on the nominal car,
                       95-100% under randomisation.
      map              race on without the network: prior steering at the
                       speed floor.  Measured: 3 laps 100% on the nominal car,
                       30-37% under randomisation.  If depth comes back for
                       `depth_recover_frames` good frames, the stack is
                       refilled with the fresh frame and the policy resumes.
    Either way ~/depth_status says why, and the telemetry `driver` field
    reads `fallback_stop` / `fallback_map` so the Run Lab shows it.

The node name in code is formula_two.  formula_two.launch.py runs it as
`formula_one` by default, because jetson/scripts/record_run.py and the Run Lab
key on /formula_one/telemetry.
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
from sensor_msgs.msg import CameraInfo, Image, Imu
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import Bool, ColorRGBA, String
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

from cfr_interfaces.msg import ArduinoStatus, DriveCommand

# ~/telemetry is what the Run Lab (web/run-lab) judges a run by.  Optional so a
# car whose cfr_interfaces predates the message still drives -- it just says,
# once, that the run it is about to make will be hard to analyse.
try:
    from cfr_interfaces.msg import DriverTelemetry
except ImportError:  # pragma: no cover - depends on the installed workspace
    DriverTelemetry = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track as track_mod  # noqa: E402
from baseline import BaselineDriver  # noqa: E402
from observation import ObservationBuilder, scale_action  # noqa: E402
from perception import INVALID, Camera, ScanStack  # noqa: E402
from policy import NumpyPolicy  # noqa: E402

HERE = Path(__file__).resolve().parent
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_of(pose):
    q = pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))


class FormulaTwo(Node):
    def __init__(self, **kwargs):
        super().__init__("formula_two", **kwargs)
        deployed = HERE / "policy.npz"  # a synced Orin tree; see the launch file
        self.declare_parameter(
            "policy",
            str(deployed if deployed.exists() else HERE / "bestModel/f2_v2_40M/policy.npz"),
        )
        # "baseline" runs the scripted driver from baseline.py instead of a
        # network.  It takes the same observation and emits the same action,
        # so it exercises every line of this node -- anchoring, the cap, the
        # DriveCommand -- against Gazebo or the car BEFORE a policy exists.
        # When something goes wrong on the day, it is also the fallback that
        # needs no checkpoint.
        self.declare_parameter("driver", "policy")
        self.declare_parameter("config", str(HERE / "config.yaml"))
        self.declare_parameter("repo_root", str(HERE.parents[1]))
        self.declare_parameter("laps", 0)  # 0 = take it from config
        self.declare_parameter("telemetry_period", 3.0)  # s between log lines
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
        # --- depth
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("camera_info_topic", "/zed/zed_node/depth/camera_info")
        # Two stages.  Past `depth_hold_after` the network is taken off the
        # wheel (prior steering at the floor) and waits; past `depth_timeout`
        # depth is LOST and `depth_fallback` ends or continues the run.
        self.declare_parameter("depth_hold_after", 0.3)
        self.declare_parameter("depth_timeout", 1.0)
        self.declare_parameter("depth_invalid_fraction", 0.6)
        self.declare_parameter("depth_invalid_frames", 3)
        self.declare_parameter("depth_recover_frames", 5)
        self.declare_parameter("depth_fallback", "stop")  # stop | map
        # Trained on 10-15 Hz.  A faster stream is thinned to <= 15 Hz.
        self.declare_parameter("depth_min_period", 1.0 / 15.0)
        # auto matches the publisher's reliability -- see `subscribe_depth`.
        self.declare_parameter("depth_qos", "auto")  # auto | reliable | best_effort
        # What the car believes about its camera: height above the ground and
        # nose-down pitch (rad).  Training believed 0.20 m and level; the real
        # mount error is what `sensor_noise.pitch_err_deg` trained through.
        self.declare_parameter("camera_height", -1.0)  # <0 = from config
        # Static mount pitch (rad, + nose down) added to the POSE's attitude.
        # Not to the IMU's: the ZED IMU already measures the camera itself.
        self.declare_parameter("camera_pitch", 0.0)
        # LEVEL THE SCAN by the measured attitude -- see perception.depth_to_scan
        # for the Gazebo run that lost 75% of its beams to a 5 deg roll.  The
        # ZED IMU (gravity-referenced, imu_fusion) is preferred; the pose's
        # own roll and pitch are the fallback, which in Gazebo is ground truth
        # and on the car is flat (two_d_mode) -- so on the car, no IMU means
        # an unlevelled scan, and the node says so.
        self.declare_parameter("imu_topic", "/zed/zed_node/imu/data")
        self.declare_parameter("level_scan", True)

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
        self.camera = Camera(self.cfg)
        self.scans = ScanStack(1, self.camera.stack, self.camera.width)
        want = self.obs.obs_dim + self.camera.stack * self.camera.width + 1
        if self.policy is not None and self.policy.obs_dim != want:
            raise SystemExit(
                f"{path} takes {self.policy.obs_dim} inputs; this config builds "
                f"{want} ({self.obs.obs_dim} map + depth stack).  Ship the policy "
                "with the config.yaml it was trained under."
            )
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
            np.clip(
                np.searchsorted(self.track.s, self.track.start_station),
                0,
                len(self.track.s) - 1,
            )
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
        self.anchor = None  # (dx, dy, dyaw) once latched
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

        # Depth.  `depth_stamp` is the capture time of the newest frame on the
        # stack; its age is what the policy's last input channel reads.
        self.intrinsics = None  # (fx, fy, cx, cy, width, height)
        self.depth_stamp = None
        self.depth_last_accept = None
        self.depth_frames = 0
        self.depth_bad_run = 0  # consecutive mostly-invalid frames
        self.depth_good_run = 0  # consecutive good frames, for recovery
        self.depth_invalid = 0.0  # fraction of invalid beams, newest frame
        self.fallback = None  # None | "stop" | "map"
        self.fallback_reason = ""
        self.depth_state = "waiting"
        self.holding = False
        self.hold_count = 0
        cam_h = float(self.param("camera_height"))
        self.camera_height = np.array([cam_h if cam_h > 0 else self.camera.z])
        self.camera_pitch = float(self.param("camera_pitch"))
        self.imu = None
        self.imu_time = None
        self.attitude_source = None
        self.roll_pitch = (0.0, 0.0)

        # BEST_EFFORT deliberately.  A best-effort subscriber accepts a
        # reliable publisher, but a reliable subscriber silently receives
        # NOTHING from a best-effort one -- and the pose publisher is reliable
        # in Gazebo and best-effort in some ZED wrapper builds.  Subscribing
        # this way is the only setting that works against both.
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
        )
        self.create_subscription(
            ArduinoStatus, "/arduino_bridge/status", self.on_status, 10
        )
        self.create_subscription(Bool, "/start_signal_detector/go", self.on_go, LATCHED)
        self.create_subscription(Bool, "/lap_counter/done", self.on_done, LATCHED)
        # The depth subscription is made once a publisher exists, so that its
        # QoS can match it -- see `subscribe_depth`.
        self.depth_sub = None
        self.depth_sub_timer = self.create_timer(0.5, self.subscribe_depth)
        self.create_subscription(
            Imu, self.param("imu_topic"), self.on_imu, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            self.param("camera_info_topic"),
            self.on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_service(SetBool, "~/manual_start", self.on_manual)

        self.drive = self.create_publisher(
            DriveCommand, "/drive_cmd", qos_profile_sensor_data
        )
        self.markers = self.create_publisher(MarkerArray, "~/markers", LATCHED)
        self.path_pub = self.create_publisher(PathMsg, "~/centerline", LATCHED)
        self.car_pub = self.create_publisher(PoseStamped, "~/car", 10)
        self.depth_pub = self.create_publisher(String, "~/depth_status", 10)
        self.telemetry_pub = (
            self.create_publisher(DriverTelemetry, "~/telemetry", 50)
            if DriverTelemetry is not None
            else None
        )
        if self.telemetry_pub is None:
            self.get_logger().warn(
                "cfr_interfaces has no DriverTelemetry -- ~/telemetry is OFF and "
                "this run will be hard to analyse.  Rebuild cfr_interfaces."
            )
        self.speed_from_tach = False
        self.relocalized = False
        if self.param("publish_markers"):
            self.publish_track()

        self.create_timer(self.control_period, self.tick)
        self.get_logger().info(
            f"formula_two ready: {self.laps_target} laps, "
            f"cap {self.cfg['track']['v_hairpin']}/{self.cfg['track']['v_straight']} m/s, "
            f"driver {self.mode}"
            + ("" if self.scripted else f" ({Path(self.param('policy')).name})")
            + f", depth fallback {self.param('depth_fallback')}"
            + ". Waiting for depth and the start signal."
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
            self.fallback = None
            self.fallback_reason = ""
        response.success = True
        response.message = "manual start" if request.data else "manual stop"

        return response

    # ---------------------------------------------------------------- depth

    def subscribe_depth(self):
        """Subscribe to depth RELIABLE when the publisher allows it.

        Measured across processes with a 640x360 float image (~900 KB): a
        best-effort subscription received 8.2 of 15 frames/s with gaps up to
        533 ms -- one lost UDP fragment loses the frame -- and tripped the
        depth hold 62 times in one run.  Reliable received all 15, worst gap
        78 ms.  But a reliable subscription receives NOTHING from a
        best-effort publisher (formulaOne subscribes to pose best-effort for
        that reason), so the choice has to follow the publisher: reliable when
        every publisher is, best-effort otherwise, with a warning.
        """
        topic = self.param("depth_topic")
        want = str(self.param("depth_qos"))
        pubs = self.get_publishers_info_by_topic(topic)
        if want == "auto" and not pubs:
            self.get_logger().info(
                f"no publisher on {topic} yet", throttle_duration_sec=10.0
            )
            return
        if want == "auto":
            reliable = all(
                p.qos_profile.reliability == ReliabilityPolicy.RELIABLE for p in pubs
            )
        else:
            reliable = want == "reliable"
        qos = (
            QoSProfile(
                depth=2,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            )
            if reliable
            else qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(Image, topic, self.on_depth, qos)
        self.depth_sub_timer.cancel()
        log = self.get_logger().info if reliable else self.get_logger().warn
        log(
            f"depth on {topic}: {'RELIABLE' if reliable else 'BEST EFFORT'}"
            + (
                ""
                if reliable
                else " -- the publisher offers nothing better; large frames will "
                "drop and the depth hold will trip.  Set the publisher's QoS to "
                "reliable."
            )
        )

    def on_imu(self, msg):
        self.imu = msg
        self.imu_time = self.get_clock().now()

    @staticmethod
    def roll_pitch_of(q):
        roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x**2 + q.y**2))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
        return roll, pitch

    def attitude(self):
        """(roll, pitch) of the camera against gravity, and where it came from."""
        if not self.param("level_scan"):
            return (0.0, self.camera_pitch), "off"
        now = self.get_clock().now()
        if self.imu_time is not None and (now - self.imu_time).nanoseconds < 0.2e9:
            return self.roll_pitch_of(self.imu.orientation), "imu"
        if self.pose is not None:
            roll, pitch = self.roll_pitch_of(self.pose.pose.orientation)
            return (roll, pitch + self.camera_pitch), "pose"
        return (0.0, self.camera_pitch), "none"

    def on_camera_info(self, msg):
        k = msg.k
        if k[0] <= 0.0:
            return
        new = (float(k[0]), float(k[4]), float(k[2]), float(k[5]), msg.width, msg.height)
        if self.intrinsics is None:
            self.get_logger().info(
                f"depth camera: {msg.width}x{msg.height}, fx {k[0]:.1f}, "
                f"hfov {math.degrees(2 * math.atan(msg.width / (2 * k[0]))):.1f} deg"
            )
        self.intrinsics = new

    @staticmethod
    def decode_depth(msg):
        """Image -> (H, W) float metres, NaN where there is no depth."""
        if msg.encoding == "32FC1":
            row = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.step // 4
            )
            depth = row[:, : msg.width].astype(np.float64)
        elif msg.encoding in ("16UC1", "mono16"):
            row = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.step // 2
            )
            depth = row[:, : msg.width].astype(np.float64) / 1000.0
            depth[depth <= 0.0] = np.nan
        else:
            raise ValueError(f"unsupported depth encoding {msg.encoding!r}")
        if msg.is_bigendian != (sys.byteorder == "big"):
            depth = depth.byteswap()
        depth[~np.isfinite(depth)] = np.nan
        return depth

    def on_depth(self, msg):
        """One depth image -> one frame on the stack, exactly as env.py does it."""
        if self.intrinsics is None:
            self.get_logger().warn(
                "depth arriving but no camera_info yet on "
                f"{self.param('camera_info_topic')} -- ignoring it",
                throttle_duration_sec=5.0,
            )
            return
        stamp = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        # 20 ms of slack: a 15 Hz stream arrives 66.6 +/- a few ms apart, and
        # without it jitter dropped 46% of those frames (measured), which is a
        # 10 Hz stack with 133 ms holes.  A 30 Hz stream is still halved.
        if (
            self.depth_last_accept is not None
            and stamp - self.depth_last_accept
            < float(self.param("depth_min_period")) - 0.02
            and stamp >= self.depth_last_accept
        ):
            return
        try:
            depth = self.decode_depth(msg)
        except ValueError as err:
            self.get_logger().error(str(err), throttle_duration_sec=5.0)
            return
        fx, fy, cx, cy, w, h = self.intrinsics
        # camera_info may describe a different resolution from the depth
        # image (the ZED can publish depth downscaled); scale K to the image.
        if (w, h) != (msg.width, msg.height) and w and h:
            sx, sy = msg.width / w, msg.height / h
            fx, cx, fy, cy = fx * sx, (cx + 0.5) * sx - 0.5, fy * sy, (cy + 0.5) * sy - 0.5
        grid = self.camera.sample_depth(depth, fx, fy, cx, cy)
        (roll, pitch), source = self.attitude()
        if source != self.attitude_source:
            self.attitude_source = source
            note = {
                "imu": "levelling the scan on the ZED IMU",
                "pose": "levelling the scan on the pose's roll/pitch (no IMU) -- "
                "ground truth in Gazebo, but FLAT on the car in two_d_mode",
                "none": "no attitude yet -- scan assumed level",
                "off": "level_scan is off -- scan assumed level",
            }[source]
            (self.get_logger().info if source == "imu" else self.get_logger().warn)(note)
        self.roll_pitch = (roll, pitch)
        scan = self.camera.depth_to_scan(
            grid, self.camera_height, np.array([pitch]), np.array([roll])
        )
        encoded = self.camera.encode(scan).astype(np.float32)
        self.depth_invalid = float(np.mean(encoded <= INVALID + 1e-6))
        bad = self.depth_invalid > float(self.param("depth_invalid_fraction"))
        self.depth_bad_run = self.depth_bad_run + 1 if bad else 0
        self.depth_good_run = 0 if bad else self.depth_good_run + 1

        # A first frame, or the first good one after an outage, REFILLS the
        # stack -- the same as a reset in env.py.  Pushing it onto frames that
        # are seconds old would hand the policy a history that never happened.
        restart = self.depth_frames == 0 or (
            self.depth_state in ("hold", "stale", "invalid") and not bad
        )
        if restart:
            self.scans.reset(np.array([True]), encoded)
        else:
            self.scans.push(np.array([True]), encoded, 0.0)
        self.depth_frames += 1
        self.depth_stamp = stamp
        self.depth_last_accept = stamp

    def depth_health(self, now_s):
        """ok | waiting | stale | invalid, and why."""
        if self.depth_stamp is None:
            return "waiting", "no depth frame yet"
        age = now_s - self.depth_stamp
        if age > float(self.param("depth_timeout")):
            return "stale", f"newest depth frame is {age:.2f} s old"
        if age > float(self.param("depth_hold_after")):
            return "hold", f"newest depth frame is {age:.2f} s old -- network off"
        if self.depth_bad_run >= int(self.param("depth_invalid_frames")):
            return "invalid", (
                f"{100 * self.depth_invalid:.0f}% of beams invalid for "
                f"{self.depth_bad_run} frames, {self.attitude_text()}"
            )
        return "ok", f"{age * 1000:.0f} ms, {100 * self.depth_invalid:.0f}% invalid"

    def attitude_text(self):
        roll, pitch = self.roll_pitch
        return (
            f"roll {math.degrees(roll):+.1f} pitch {math.degrees(pitch):+.1f} deg "
            f"({self.attitude_source})"
        )

    def depth_features(self, now_s):
        """(1, stack*W + 1): the stack and the newest frame's age, as trained."""
        self.scans.age[:] = max(now_s - (self.depth_stamp or now_s), 0.0)
        return self.scans.features()

    def check_depth(self, why):
        """Enter or leave the depth fallback.  Called once per driving tick."""
        mode = str(self.param("depth_fallback"))
        if self.depth_state == "hold" and not self.holding:
            self.holding = True
            self.hold_count += 1
            self.get_logger().warn(
                f"depth late ({why}) -- prior steering at the floor until it "
                f"returns or turns {float(self.param('depth_timeout')):.1f} s old"
            )
        elif self.depth_state == "ok" and self.holding:
            self.holding = False
            self.get_logger().info("depth back -- policy resumes")
        if self.depth_state in ("stale", "invalid") and self.fallback is None:
            self.holding = False
            self.fallback = "map" if mode == "map" else "stop"
            self.fallback_reason = why
            if self.fallback == "stop":
                self.get_logger().error(
                    f"DEPTH LOST ({why}) -- throttle off, prior steering, "
                    "coasting to rest.  The run is over."
                )
                if not self.stopping:
                    self.begin_stopping()
            else:
                self.get_logger().error(
                    f"DEPTH LOST ({why}) -- racing on the map alone at the "
                    "speed floor until depth recovers"
                )
        elif (
            self.fallback == "map"
            and self.depth_state == "ok"
            and self.depth_good_run >= int(self.param("depth_recover_frames"))
        ):
            self.get_logger().info(
                f"depth back ({self.depth_good_run} good frames) -- policy resumes"
            )
            self.fallback = None
            self.fallback_reason = ""

    def publish_depth_status(self, state, why):
        text = f"{state}: {why} | {self.attitude_text()}"
        if self.fallback:
            text += f" | FALLBACK {self.fallback} ({self.fallback_reason})"
        self.depth_pub.publish(String(data=text))

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
        self.speed_from_tach = bool(
            fresh and self.status is not None and self.status.link_ok
        )
        if self.speed_from_tach:
            speed = abs(float(self.status.speed))
        elif self.prev_pose is not None:
            dt = max(
                (
                    rclpy.time.Time.from_msg(self.pose.header.stamp)
                    - rclpy.time.Time.from_msg(self.prev_pose.header.stamp)
                ).nanoseconds
                * 1e-9,
                1e-3,
            )
            speed = (
                math.hypot(
                    self.pose.pose.position.x - self.prev_pose.pose.position.x,
                    self.pose.pose.position.y - self.prev_pose.pose.position.y,
                )
                / dt
            )
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
        now_s = now.nanoseconds * 1e-9
        self.depth_state, depth_why = self.depth_health(now_s)
        self.publish_depth_status(self.depth_state, depth_why)
        # The network cannot drive without a depth frame, and a car that has
        # been anchored and released on a map alone is the policy driving
        # blind.  Hold neutral -- and do not anchor -- until one arrives.
        no_depth = self.policy is not None and self.depth_stamp is None
        if self.finished or self.manual_stop or not self.go or stale or no_depth:
            if stale and self.go and not self.finished:
                self.get_logger().warn(
                    "pose is stale -- commanding neutral", throttle_duration_sec=2.0
                )
            elif no_depth and self.go and not self.finished:
                self.get_logger().warn(
                    f"GO received but no depth yet on {self.param('depth_topic')} "
                    "-- holding neutral",
                    throttle_duration_sec=2.0,
                )
            self.send(0.0, 0.0)
            if self.finished:
                state = DriverTelemetry.STATE_FINISHED if DriverTelemetry else 0
            elif self.manual_stop:
                state = DriverTelemetry.STATE_MANUAL_STOP if DriverTelemetry else 0
            elif not self.go:
                state = DriverTelemetry.STATE_WAITING if DriverTelemetry else 0
            else:
                state = DriverTelemetry.STATE_STALE if DriverTelemetry else 0
            self.publish_telemetry(state, now)
            return

        x_raw = self.pose.pose.position.x
        y_raw = self.pose.pose.position.y
        yaw_raw = yaw_of(self.pose.pose)
        if self.anchor is None:
            self.latch(x_raw, y_raw, yaw_raw)
            self.obs.set_station(np.array([True]), np.array([self.track.start_station]))
            self.station = self.track.start_station
            self.prev_track_yaw = None
            self.prev_yaw_stamp = None
            self.yaw_rate = np.zeros(1)

        x, y, yaw = self.to_track(x_raw, y_raw, yaw_raw)
        speed = self.measured_speed()
        self.relocalized = False

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
                self.relocalized = True
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
            step = math.atan2(
                math.sin(yaw - self.prev_track_yaw), math.cos(yaw - self.prev_track_yaw)
            )
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
        if (
            self.prev_obs_speed is None
            or self.prev_speed_stamp is None
            or stamp_s <= self.prev_speed_stamp
        ):
            self.prev_obs_speed, self.prev_speed_stamp = speed, stamp_s
        else:
            raw = (speed - self.prev_obs_speed) / (stamp_s - self.prev_speed_stamp)
            raw = max(-20.0, min(20.0, raw))
            self.speed_rate = (
                1 - self.accel_filter
            ) * self.speed_rate + self.accel_filter * raw
            self.prev_obs_speed, self.prev_speed_stamp = speed, stamp_s

        clock_s = now.nanoseconds * 1e-9
        if self.run_started_at is None:
            self.run_started_at = clock_s
            self.lap_started_at = clock_s
        lap_progress = float(
            np.clip(
                (self.distance - self.laps_done * self.track.length)
                / self.track.length,
                0.0,
                1.0,
            )
        )
        lap_state = np.array(
            [[lap_progress, clock_s - self.lap_started_at, self.last_lap_time]]
        )
        obs, frame = self.obs.compute(
            np.array([x]),
            np.array([y]),
            np.array([yaw]),
            np.array([speed]),
            self.yaw_rate,
            self.speed_rate,
            self.prev_action,
            self.last_steer,
            lap_state,
        )
        v_cap = frame["v_cap"]
        if self.policy is not None:
            obs = np.concatenate([obs, self.depth_features(now_s)], axis=1)
            self.check_depth(depth_why)
        if self.scripted is not None:
            action = self.scripted.act(
                frame["station"], np.array([speed]), v_cap, frame["v_floor"]
            )
        elif self.fallback or self.depth_state != "ok":
            # Without trustworthy depth the network's output is not a
            # decision, it is an accident.  The centerline prior steers (zero
            # residual) and the throttle sits at the floor -- which the
            # stopping phase below turns into zero for the `stop` fallback.
            action = np.array([[0.0, -1.0]])
        else:
            action = self.policy.act(obs)
        steer, velocity = scale_action(
            action, v_cap, frame["steer_ff"], self.residual, frame["v_floor"]
        )
        # Belt and braces: the rule limit, enforced again on the way out.
        velocity = np.minimum(velocity, v_cap) * float(self.param("speed_scale"))
        self.prev_action = np.clip(action, -1.0, 1.0)
        # The prior predicts forward through the command already in flight, so
        # it has to be the REALISED command -- after scale_action folded in
        # the residual -- not the network's raw output.
        self.last_steer = np.asarray(steer, dtype=float).reshape(1)

        advance = (
            frame["station"][0] - self.station + self.track.length / 2
        ) % self.track.length - self.track.length / 2
        if abs(advance) < 2.0:
            self.distance += advance
        self.station = float(frame["station"][0])
        laps = int(max(self.distance, 0.0) // self.track.length)
        if laps > self.laps_done:
            self.laps_done = laps
            lap_time = clock_s - self.lap_started_at
            delta = (
                f", {self.last_lap_time - lap_time:+.2f} s on the last"
                if self.last_lap_time > 0.0
                else ""
            )
            self.get_logger().info(
                f"lap {laps} of {self.laps_target}  {lap_time:.2f} s{delta}"
            )
            self.last_lap_time = lap_time
            self.lap_started_at = clock_s
        if not self.stopping and self.distance >= self.laps_target * self.track.length:
            self.race_time = clock_s - (self.run_started_at or clock_s)
            self.get_logger().info(
                f"FINISHED {self.laps_target} laps in {self.race_time:.2f} s "
                f"({self.distance:.1f} m) -- coasting to a stop"
            )
            self.begin_stopping()

        if self.stopping:
            # Throttle off, steering still live.  The car is coasting and is
            # still in the corridor, so it still has to be driven.
            since = clock_s - (self.stopping_since or clock_s)
            if speed <= self.stop_speed or since >= self.stop_timeout:
                self.finished = True
                where = (
                    f"at {self.distance:.1f} m, lap {self.laps_done}/{self.laps_target} "
                    f"-- RUN ABANDONED, depth lost ({self.fallback_reason})"
                    if self.fallback == "stop"
                    else f"{self.distance - self.laps_target * self.track.length:.1f} m "
                    "past the line"
                )
                self.get_logger().info(
                    f"STOPPED after {since:.2f} s and {where}"
                    + (
                        ""
                        if speed <= self.stop_speed
                        else "  (TIMED OUT, still moving)"
                    )
                )
                self.send(0.0, 0.0)
                self.publish_telemetry(
                    DriverTelemetry.STATE_FINISHED if DriverTelemetry else 0, now
                )
                return
            velocity = np.zeros_like(velocity)

        self.send(steer[0], velocity[0])
        self.publish_car(x, y, yaw, frame, speed, velocity[0])
        self.publish_telemetry(
            (
                DriverTelemetry.STATE_STOPPING
                if self.stopping
                else DriverTelemetry.STATE_RUNNING
            )
            if DriverTelemetry
            else 0,
            now,
            pose=(x, y, yaw),
            frame=frame,
            speed=speed,
            action=action,
            obs=obs,
            steer=float(steer[0]),
            velocity=float(velocity[0]),
            lap_clock=clock_s,
        )
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

    # ------------------------------------------------------------ telemetry

    def publish_telemetry(
        self,
        state,
        now,
        pose=None,
        frame=None,
        speed=0.0,
        action=None,
        obs=None,
        steer=0.0,
        velocity=0.0,
        lap_clock=None,
    ):
        """One DriverTelemetry per tick, whatever the tick decided.

        On the idle branches only the header, state and health fields mean
        anything; the rest stay zero.  The analyser keys on `state`, so a zero
        there is never mistaken for the car sitting at station 0.
        """
        if self.telemetry_pub is None:
            return
        msg = DriverTelemetry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame
        msg.state = int(state)
        msg.driver = (
            f"fallback_{self.fallback}"
            if self.fallback
            else ("depth_hold" if self.holding else str(self.mode))
        )
        msg.laps_target = int(self.laps_target)
        msg.lap = int(self.laps_done)
        msg.distance = float(self.distance)
        msg.last_lap_time = float(self.last_lap_time)
        msg.speed_scale = float(self.param("speed_scale"))
        msg.pose_age = (
            float((now - self.pose_time).nanoseconds * 1e-9)
            if self.pose_time is not None
            else float("inf")
        )
        msg.speed_from_tach = bool(self.speed_from_tach)
        msg.relocalized = bool(self.relocalized)
        msg.yaw_rate = float(self.yaw_rate[0])
        msg.speed_rate = float(self.speed_rate[0])
        if self.station is not None:
            msg.station = float(self.station)
        if lap_clock is not None:
            msg.lap_time = float(lap_clock - (self.lap_started_at or lap_clock))
            msg.race_time = float(lap_clock - (self.run_started_at or lap_clock))
        if pose is not None:
            msg.x, msg.y, msg.yaw = (float(v) for v in pose)
        if frame is not None:
            msg.station = float(frame["station"][0])
            msg.cross_track = float(frame["lateral"][0])
            msg.heading_error = float(frame["psi"][0])
            msg.clearance = float(frame["clearance"][0])
            msg.v_cap = float(frame["v_cap"][0])
            msg.v_floor = float(frame["v_floor"][0])
            msg.steer_ff = float(np.asarray(frame["steer_ff"]).reshape(-1)[0])
        msg.speed = float(speed)
        if action is not None:
            msg.action = [float(a) for a in np.asarray(action).reshape(-1)]
        if obs is not None:
            msg.observation = [float(o) for o in np.asarray(obs).reshape(-1)]
        msg.steer_cmd = float(steer)
        msg.velocity_cmd = float(velocity)
        self.telemetry_pub.publish(msg)

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
        text.ns, text.id, text.type, text.action = (
            "hud",
            1,
            Marker.TEXT_VIEW_FACING,
            Marker.ADD,
        )
        text.pose.position.x, text.pose.position.y, text.pose.position.z = (
            20.0,
            0.0,
            2.0,
        )
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
    node = FormulaTwo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl-C under `ros2 launch` can shut the context down before this
        # runs; the neutral command is best effort, and the bridge's
        # command_timeout (0.2 s, arduino_bridge.yaml) zeroing the speed is
        # the guarantee.
        try:
            node.send(0.0, 0.0, ready=False)
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
