#!/usr/bin/env python3
"""Drive the Obstacle Course with a trained policy -- in Gazebo and on the car.

One node, one code path, both places.  Everything it reads exists identically
on the Orin and in the simulator:

    /zed/zed_node/point_cloud/cloud_registered   PointCloud2, camera frame
    /zed/zed_node/pose           PoseStamped.  Read ONLY for pitch and roll
                                 (to level the cloud, as cloud_segmentation_node
                                 does) and for yaw rate, which it also
                                 integrates into the heading since the start.
                                 No position is used: the policy has no map.
    /arduino_bridge/status       ArduinoStatus: tachometer speed, and the
                                 Manual Start bit (manual_start), a run trigger
    /start_signal_detector/go    latched Bool, the other run trigger
    /lap_counter/done            latched Bool, the end of the run
    /drive_cmd                   DriveCommand out, as rl/formulaOne sends it
    /left_wall_follower/manual_go  latched Bool out on a manual start (the
                                 Arduino bit or ~/manual_start), so
                                 lap_counter arms as it does on the signal

A run starts on whichever comes first: the detector seeing red turn green,
or the Arduino's Manual Start bit going from 0 to 1 (README "Manual Start":
start without the visual signal).  Only a rising edge counts -- a bit already
set when the node comes up does not launch the car (sim_vehicle holds it at
1), and it takes a fresh press.  arduino_manual_start:=false ignores it.

The cloud goes through the compiled segmenter (cloud_segmentation.py, the one
cloud_segmentation_node runs and the fixtures score), then
`scan_from_segmentation` and the segmenter's gates, then observation.py -- the
module the trainer built every training observation with.

    ros2 launch rl/obstacleRacer/obstacle_racer.launch.py policy:=runs/v1/policy.npz

driver:=prior drives the map-free steering prior alone at a fixed speed, for
checking the whole chain before a policy exists.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
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
import observation as O  # noqa: E402
from policy import NumpyPolicy  # noqa: E402

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


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


def attitude(pose):
    """(pitch nose-down +, roll left-up +, yaw) from a REP-103 orientation."""
    q = pose.orientation
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    roll = math.atan2(
        2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    )
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return pitch, roll, yaw


class ObstacleRacer(Node):
    def __init__(self):
        super().__init__("obstacle_racer")
        self.declare_parameter("policy", str(HERE / "runs/v1/policy.npz"))
        self.declare_parameter("config", str(HERE / "config.yaml"))
        self.declare_parameter("driver", "policy")  # policy | prior
        self.declare_parameter("prior_speed", 1.0)  # m/s, driver:=prior
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("cloud_timeout", 0.5)
        self.declare_parameter(
            "cloud_stride", 2
        )  # every other point, as the node's viewer copy
        self.declare_parameter("yaw_rate_filter", 0.5)
        # Start on the Arduino's Manual Start bit as well as the signal.
        self.declare_parameter("arduino_manual_start", True)

        self.cfg = yaml.safe_load(Path(self.param("config")).read_text())
        self.seg = _segmenter()
        self.mode = self.param("driver")
        self.policy = (
            NumpyPolicy.load(self.param("policy")) if self.mode == "policy" else None
        )
        want = O.obs_dim(self.cfg)
        if self.policy is not None and self.policy.obs_dim != want:
            raise SystemExit(
                f"{self.param('policy')} takes {self.policy.obs_dim} inputs; observation.py "
                f"builds {want} under {self.param('config')}.  Ship the policy's own "
                "config.yaml: it was trained on a different observation."
            )
        self.stack = O.Stack(1, O.frame_offsets(self.cfg), O.frame_dim(self.cfg))
        self.memory = O.Memory(1, self.cfg)
        # Heading since the start box: every run starts there pointing down
        # the lane, heading 0 in the course frame training uses.
        self.heading = O.Heading(1, self.cfg)
        # True until the first tick of a run: the stack, the memory and the
        # prior then start from that frame, as an episode does in training.
        self.fresh = True

        self.dt = 1.0 / float(self.cfg["env"]["control_hz"])
        self.scan = np.full((1, O.SCAN_BINS), float(self.cfg["sensor"]["max_range"]))
        self.gate = np.zeros((1, O.GATE_DIM))
        self.cloud_time = None
        self.pose = None
        self.pose_time = None
        self.prev_yaw = None
        self.prev_stamp = None
        self.yaw_rate = 0.0
        self.status = None
        self.go = False
        self.done = False
        self.started_at = None
        self.prev_manual_bit = None  # first status seeds it: edges only
        self.prev_action = np.zeros((1, 2))
        self.prior = np.zeros(1)

        self.create_subscription(
            PointCloud2,
            "/zed/zed_node/point_cloud/cloud_registered",
            self.on_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
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
            f"obstacle_racer ready: driver {self.mode}, v_cap {self.cfg['env']['v_cap']} m/s. "
            "Waiting for the start signal."
        )

    def param(self, name):
        return self.get_parameter(name).value

    def now(self):
        """Seconds on the node's clock: sim time in Gazebo, as formula_one_node.

        Staleness is judged on it, not on the wall clock -- Gazebo rendering
        the ZED under load runs at a fraction of real time (0.19 measured), so
        a cloud arriving every 0.3 s of sim time is 1.6 s apart on the wall,
        and a wall-clock timeout held the car at zero speed forever.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------- inputs

    def on_pose(self, msg):
        pitch, roll, yaw = attitude(msg.pose)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Yaw rate over the interval the two poses actually span, from their
        # stamps: the pose stream and the 20 Hz loop do not share a rate.
        if self.prev_yaw is not None and stamp > self.prev_stamp:
            dyaw = (yaw - self.prev_yaw + math.pi) % (2 * math.pi) - math.pi
            rate = dyaw / (stamp - self.prev_stamp)
            a = float(self.param("yaw_rate_filter"))
            self.yaw_rate = (1 - a) * self.yaw_rate + a * max(-8.0, min(8.0, rate))
        self.prev_yaw, self.prev_stamp = yaw, stamp
        self.pose = (pitch, roll, yaw)
        self.pose_time = self.now()

    def on_cloud(self, msg):
        if self.pose is None:
            return
        pts = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        stride = max(1, int(self.param("cloud_stride")))
        pts = np.asarray(pts, dtype=np.float64)[::stride]
        pitch, roll, _ = self.pose
        seg = self.seg.segment(pts, pitch, roll)
        s = self.cfg["sensor"]
        scan = self.seg.scan_from_segmentation(
            seg,
            O.SCAN_BINS,
            float(s["fov_deg"]),
            float(s["max_range"]),
            float(s["min_range"]),
        )
        self.scan = np.asarray(scan, dtype=np.float64)[None, :]
        gates = [
            (g.kind, g.center[0], g.center[1], g.axis[0], g.axis[1]) for g in seg.gates
        ]
        self.gate = O.gate_features(gates, self.cfg)[None, :]
        self.cloud_time = self.now()

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
        if self.go:
            return
        self.get_logger().info(f"{source} -- going")
        self.go = True
        self.done = False
        self.started_at = self.now()
        self.fresh = True
        if manual:
            # lap_counter arms on the detector's go or on this topic.
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
            self.fresh = True
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
        stale = (
            self.pose_time is None
            or now - self.pose_time > self.param("pose_timeout")
            or self.cloud_time is None
            or now - self.cloud_time > self.param("cloud_timeout")
        )
        speed = float(self.status.speed) if self.status is not None else 0.0
        speed_meas = O.tach(np.array([speed]))
        yaw_rate = np.array([self.yaw_rate])

        fresh = np.array([self.fresh])
        if self.fresh:
            self.prev_action = np.zeros((1, 2))
            self.heading.reset(np.array([0]), 0.0)
            if self.policy is not None:
                self.policy.reset(1)  # the LSTM's memory starts with the run
        raw = O.prior_steer(self.scan, self.gate, yaw_rate, self.cfg)
        self.prior = O.smooth_prior(self.prior, raw, fresh, self.cfg)
        memory = self.memory.update(speed_meas, fresh=fresh)
        heading = self.heading.update(yaw_rate)
        frame = O.frame(
            self.scan,
            self.gate,
            speed_meas,
            yaw_rate,
            self.prev_action,
            self.prior,
            self.cfg,
            memory,
            heading,
        )
        if self.fresh:
            self.stack.reset(np.array([0]), frame)
            self.fresh = False
            obs = self.stack.obs
        else:
            obs = self.stack.push(frame)

        if self.policy is not None:
            action = self.policy.act(obs)
        else:
            action = np.array(
                [
                    [
                        0.0,
                        float(
                            O.speed_to_action(
                                float(self.param("prior_speed")), self.cfg
                            )
                        ),
                    ]
                ]
            )
        steer, velocity = O.action_to_command(action, self.prior, self.cfg)
        self.prev_action = action

        driving = self.go and not self.done and not stale
        v_cap = float(self.cfg["env"]["v_cap"])
        velocity_out = (
            min(float(velocity[0]) * float(self.param("speed_scale")), v_cap)
            if driving
            else 0.0
        )
        # Steering stays live after the finish, while the car coasts.
        self.send(steer[0] if (self.go and not stale) else 0.0, velocity_out)
        if stale and self.go and not self.done:
            self.get_logger().warn(
                "pose or cloud stale -- holding at zero speed",
                throttle_duration_sec=2.0,
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
            t.driver = self.mode
            t.race_time = float(now - self.started_at) if self.started_at else 0.0
            t.yaw = float(self.pose[2]) if self.pose else 0.0
            t.speed = float(speed_meas[0])
            t.v_cap = v_cap
            t.yaw_rate = float(self.yaw_rate)
            t.action = [float(a) for a in action[0]]
            t.steer_ff = float(self.prior[0])
            t.steer_cmd = float(steer[0])
            t.velocity_cmd = float(velocity_out)
            t.speed_scale = float(self.param("speed_scale"))
            t.pose_age = float(now - self.pose_time) if self.pose_time else -1.0
            t.speed_from_tach = self.status is not None
            t.observation = [float(v) for v in obs[0]]
            self.telemetry.publish(t)


def main():
    rclpy.init()
    node = ObstacleRacer()
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
