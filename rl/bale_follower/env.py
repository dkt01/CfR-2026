"""Gymnasium environment for RL-based bale following in the Gazebo sim.

Wraps the existing training.launch.py stack: publishes geometry_msgs/Twist
on /cmd_vel (consumed by cmd_vel_to_drive_node -> sim_vehicle_node -> Gazebo's
AckermannSteering plugin, unchanged), and reads ground-truth world poses from
/world/<world>/dynamic_pose/info. Episode resets reuse teleport_api.py's HTTP
endpoint.

The pose stream is the vehicle's true world pose, deliberately not
/zed/zed_node/odom: the Ackermann plugin dead-reckons odometry from wheel
rotation, so it ignores teleports entirely and reports a pose that drifts
further from reality with every episode reset.

Steps are paced against the wall clock rather than stepped explicitly.
Driving Gazebo through WorldControl `multi_step` corrupts the server's heap
(`malloc(): unaligned fastbin chunk detected`) and leaves it alive but with
dead service threads, so training runs at real time instead. That caps
throughput at control_hz steps per second -- the main cost of this approach.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from dataclasses import dataclass

import gymnasium
import numpy as np
import requests
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from rosgraph_msgs.msg import Clock

import bale_geometry
import zed_sim
from course_progress import CourseProgress
from reward import RewardConfig, compute_reward
from cloud_scan import points_from_pointcloud2, scan_from_points

WHEELBASE = 0.324
MAX_STEERING_ANGLE = (
    0.40  # rad, matches jetson/cfr_arduino_bridge/config/arduino_bridge.yaml
)
GRAVITY = 9.81


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    stamp: float  # wall clock, for "has a fresh pose arrived yet"
    sim_stamp: float = 0.0  # simulation clock, for anything measuring motion


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _unpause_world(world: str, attempts: int = 3) -> bool:
    """Start the world running.

    The headless server comes up paused even when launched with `-r`, and
    while it is paused /clock never advances, so the control nodes treat every
    command as stale and hold the vehicle at neutral. Called once at startup:
    hammering WorldControl per step corrupts the server's heap.
    """
    command = [
        "gz",
        "service",
        "-s",
        f"/world/{world}/control",
        "--reqtype",
        "gz.msgs.WorldControl",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "3000",
        "--req",
        "pause: false",
    ]
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=6.0, check=False
            )
            if result.returncode == 0 and "data: true" in result.stdout:
                return True
        except subprocess.TimeoutExpired:
            pass
        if attempt + 1 < attempts:
            time.sleep(0.3)
    return False


class BaleFollowerEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        sdf_path: str,
        teleport_url: str = "http://localhost:9003/api/sim/teleport",
        world_name: str = "cfr_speed_course",
        num_lidar_bins: int = 36,
        lidar_fov_deg: float = 180.0,
        lidar_max_range: float = 6.0,
        control_hz: float = 10.0,
        max_speed: float = 3.5,
        reverse_speed: float = 0.0,
        episode_time_limit_s: float = 120.0,
        progress_window_s: float = 5.0,
        min_progress_speed: float = 1.0,
        randomize_start: bool = True,
        reward_config: RewardConfig | None = None,
        odom_timeout_s: float = 2.0,
        teleport_timeout_s: float = 150.0,
        traction: float = 0.6,
        max_steering_rate: float = 3.5,
        course_path_file: str | None = None,
        scan_source: str = "analytic",
        cloud_topic: str = "/zed/zed_node/point_cloud/cloud_registered",
        zed_config: zed_sim.ZedSimConfig | None = None,
    ) -> None:
        super().__init__()
        self.traction = traction
        self.max_steering_rate = max_steering_rate
        self.zed_config = zed_config or zed_sim.ZedSimConfig(enabled=False)
        # When the ZED model is on, the observation FOV is the camera's FOV --
        # the policy must not be trained on rays a real ZED 2i cannot see.
        if self.zed_config.enabled:
            lidar_fov_deg = min(lidar_fov_deg, self.zed_config.hfov_deg)
        self.world_name = world_name
        self.teleport_url = teleport_url
        self.num_lidar_bins = num_lidar_bins
        self.lidar_fov_deg = lidar_fov_deg
        self.lidar_max_range = lidar_max_range
        self.control_hz = control_hz
        self.max_speed = max_speed
        self.reverse_speed = reverse_speed
        self.obs_speed_scale = max_speed
        self.episode_time_limit_s = episode_time_limit_s
        self.progress_window_s = progress_window_s
        self.min_progress_speed = min_progress_speed
        self.randomize_start = randomize_start
        self.reward_config = reward_config or RewardConfig()
        self.odom_timeout_s = odom_timeout_s
        self.teleport_timeout_s = teleport_timeout_s

        self.bales = bale_geometry.parse_bales(sdf_path)
        self.spawn_pose = bale_geometry.parse_vehicle_spawn(sdf_path)
        # Arc-length progress along the planned centerline. Privileged, and
        # deliberately reward-only: it never enters the observation.
        self.course = (
            CourseProgress(course_path_file) if course_path_file else CourseProgress()
        )

        # Symmetric and normalized, as SB3 expects: its policy is a
        # zero-centered Gaussian, so an asymmetric range like [0, max_speed]
        # clips half the distribution to a standstill and the car never
        # explores moving at all. Mapped to real units in `step`.
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        obs_dim = num_lidar_bins + 2
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._pose_lock = threading.Lock()
        self._latest_pose: Pose2D | None = None
        self._episode_step = 0
        self._episode_time = 0.0
        self._prev_pose: Pose2D | None = None
        self._prev_angular_z = 0.0
        self._prev_steer_fraction = 0.0
        self._progress_window: list[float] = []
        self._cmd_speed = 0.0
        self._cmd_steer_fraction = 0.0
        self._rng = np.random.default_rng()

        # "cloud" runs the observation through the same point-cloud path the
        # robot will use (cloud_scan.scan_from_points), so what the policy
        # learns on is what a real ZED can produce. "analytic" ray-casts the
        # known bale geometry: faster, but it is the input that made earlier
        # checkpoints undeployable.
        self.scan_source = scan_source
        self.cloud_topic = cloud_topic
        self._latest_cloud = None
        self._cloud_lock = threading.Lock()
        # Camera tilt, for levelling the cloud before its height band is
        # applied. The ZED is bolted to the chassis with no relative rotation
        # (<pose>0.315 0 0.20 0 0 0</pose>), so chassis tilt is camera tilt.
        self._tilt = (0.0, 0.0)
        self._cloud_hits = 0
        self._cloud_misses = 0

        self._closed = False
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node("bale_follower_rl_env")
        self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel", 10)
        self._node.create_subscription(
            TFMessage, f"/world/{world_name}/dynamic_pose/info", self._on_pose, 10
        )
        # Simulation clock. The pose messages themselves carry header.stamp 0
        # (the ros_gz bridge does not fill it in -- measured), so /clock is the
        # only source of sim time, and sim time is what motion has to be
        # measured against once rendering pulls the real-time factor below 1.
        self._sim_time = 0.0
        self._node.create_subscription(
            Clock,
            "/clock",
            lambda m: setattr(self, "_sim_time", m.clock.sec + m.clock.nanosec * 1e-9),
            10,
        )
        if self.scan_source == "cloud":
            qos = rclpy.qos.QoSProfile(depth=1)
            qos.reliability = rclpy.qos.ReliabilityPolicy.BEST_EFFORT
            self._node.create_subscription(
                PointCloud2, self.cloud_topic, self._on_cloud, qos
            )
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        if not _unpause_world(world_name):
            raise RuntimeError(f"could not start world '{world_name}' running")

    def _on_pose(self, msg: TFMessage) -> None:
        # Gazebo publishes the model's world pose first, then its links'
        # poses relative to the model, and the bridge drops entity names --
        # hence the positional access rather than a lookup by name.
        #
        # index 0 is the Slash. It is no longer the only dynamic model: the
        # start signal added a second one (verified -- transforms[0] reads the
        # SDF spawn pose 20.15, 4.76 while transforms[1] sits at 17.01, 3.89),
        # so this depends on the Slash being declared first in the world.
        if not msg.transforms:
            return
        transform = msg.transforms[0].transform
        q = transform.rotation
        pose = Pose2D(
            x=transform.translation.x,
            y=transform.translation.y,
            yaw=_yaw_from_quaternion(q.x, q.y, q.z, q.w),
            stamp=time.monotonic(),
            sim_stamp=self._sim_time,
        )
        # Nose-down-positive pitch and roll, matching cloud_scan's convention
        # (verified: a +theta rotation about +y extracts as +theta). Without
        # these the height band stops rejecting the ground the moment the car
        # pitches, and 3 degrees is enough to fill every bin with phantom
        # returns at 2.5 m.
        sin_pitch = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
        pitch = math.asin(sin_pitch)
        roll = math.atan2(
            2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        )
        with self._pose_lock:
            self._latest_pose = pose
            self._tilt = (pitch, roll)

    def _on_cloud(self, msg: PointCloud2) -> None:
        with self._cloud_lock:
            self._latest_cloud = msg

    def _cloud_scan(self) -> np.ndarray | None:
        """Latest camera scan, or None if no cloud has arrived yet."""
        with self._cloud_lock:
            msg = self._latest_cloud
        if msg is None:
            return None
        with self._pose_lock:
            pitch, roll = self._tilt
        return scan_from_points(
            points_from_pointcloud2(msg),
            self.num_lidar_bins,
            self.lidar_fov_deg,
            self.lidar_max_range,
            pitch=pitch,
            roll=roll,
        )

    def _wait_for_pose(self, since: float) -> Pose2D:
        deadline = time.monotonic() + self.odom_timeout_s
        while time.monotonic() < deadline:
            with self._pose_lock:
                pose = self._latest_pose
            if pose is not None and pose.stamp > since:
                return pose
            time.sleep(0.005)
        raise TimeoutError(
            f"no pose received on /world/{self.world_name}/dynamic_pose/info"
        )

    def _settle(self, seconds: float) -> None:
        """Hold zero velocity so the car stops before the next episode starts."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._cmd_pub.publish(Twist())
            time.sleep(0.02)

    def _teleport(
        self, x: float, y: float, heading_deg: float, attempts: int = 3
    ) -> None:
        # teleport_api shells out to `gz service`, which can time out right
        # after a world reset while Gazebo is still settling. This client
        # timeout has to sit ABOVE teleport_api's own GZ_SERVICE_TIMEOUT_MS,
        # or the HTTP request gives up first and the server's patience is
        # wasted -- which is what killed two chunks of the 300k run.
        message = None
        for attempt in range(attempts):
            try:
                response = requests.post(
                    self.teleport_url,
                    json={"x": x, "y": y, "heading": heading_deg},
                    timeout=self.teleport_timeout_s,
                )
                result = response.json()
                if response.ok and result.get("success"):
                    return
                message = result.get("message")
            except requests.RequestException as error:
                message = str(error)
            if attempt + 1 < attempts:
                time.sleep(0.2)
        raise RuntimeError(f"teleport failed: {message}")

    def _advance_sim(self) -> None:
        time.sleep(1.0 / self.control_hz)

    def decode_action(self, action: np.ndarray) -> tuple[float, float]:
        """Map a normalized action in [-1, 1]^2 to (speed m/s, steering fraction).

        Speed spans [-reverse_speed, max_speed] linearly. Reverse exists so
        the policy can unstick itself when nosed against a bale in a hairpin
        -- with a forward-only range that pose is terminal. It is kept slow
        (a fraction of forward speed) and forward progress is what the reward
        pays, so reversing is a recovery move, not a strategy.
        """
        fraction = (float(np.clip(action[0], -1.0, 1.0)) + 1.0) / 2.0
        speed = -self.reverse_speed + fraction * (self.max_speed + self.reverse_speed)
        steer_fraction = float(np.clip(action[1], -1.0, 1.0))
        return speed, steer_fraction

    def encode_action(self, speed: float, steer_fraction: float) -> np.ndarray:
        """Inverse of decode_action, for re-injecting externally filtered commands."""
        fraction = (speed + self.reverse_speed) / (self.max_speed + self.reverse_speed)
        return np.array([2.0 * fraction - 1.0, steer_fraction], dtype=np.float32)

    def _apply_traction(self, speed: float, steer_fraction: float) -> float:
        """Limit the commanded speed to what the tires can transmit.

        Two constraints from a friction coefficient `traction` (mu):
        longitudinal accel is slewed to a_max = mu*g, and the kinematic
        bicycle's lateral acceleration v^2 tan(delta)/L is capped at the same
        a_max, so a full-lock command at speed is slowed rather than executed.
        These are the same limits the CasADi smoother enforces predictively at
        deployment; applying them (greedily) during training keeps the policy
        from learning commands the drivetrain will never deliver.
        """
        a_max = self.traction * GRAVITY
        dt = 1.0 / self.control_hz
        speed = self._cmd_speed + min(
            max(speed - self._cmd_speed, -a_max * dt), a_max * dt
        )
        tan_delta = abs(math.tan(steer_fraction * MAX_STEERING_ANGLE))
        if tan_delta > 1e-6:
            grip_speed = math.sqrt(a_max * WHEELBASE / tan_delta)
            speed = min(max(speed, -grip_speed), grip_speed)
        self._cmd_speed = max(-self.reverse_speed, min(speed, self.max_speed))
        return self._cmd_speed

    def _pick_start_pose(self) -> tuple[float, float, float]:
        """Start anywhere on the planned centerline, facing along it.

        Sampling the whole loop rather than jittering the SDF spawn matters
        now that episodes are cut short for slow progress: starting every
        episode in the same place would teach the policy only the first few
        metres of course, because that is all a short episode ever sees.

        An earlier version could not do this -- it had only the bale list,
        whose indices are DXF drawing order, so interpolating between
        neighbours landed inside bales. The centerline removes that problem:
        every sample on it is drivable by construction.
        """
        if not self.randomize_start:
            return self.spawn_pose

        for _ in range(20):
            index = int(self._rng.integers(len(self.course.x)))
            yaw = float(self.course.heading[index])
            # Small lateral jitter so the policy sees off-line recoveries,
            # not just the perfect line.
            offset = float(self._rng.uniform(-0.15, 0.15))
            candidate = (
                float(self.course.x[index]) - offset * math.sin(yaw),
                float(self.course.y[index]) + offset * math.cos(yaw),
                yaw + float(self._rng.uniform(-0.15, 0.15)),
            )
            if not bale_geometry.check_collision(self.bales, *candidate):
                return candidate
        return self.spawn_pose

    def _build_observation(
        self,
        pose: Pose2D,
        speed: float,
        yaw_rate: float,
        scan: np.ndarray | None = None,
    ) -> np.ndarray:
        if scan is None and self.scan_source == "cloud":
            scan = self._cloud_scan()
            # Falling back to the analytic scan here would silently train a
            # policy that never saw a point cloud, which is the exact failure
            # this whole path exists to avoid -- and it would look like a
            # normal run for all fifteen hours of it. Say so, once.
            if scan is None:
                # Consecutive, not cumulative: scattered misses across a long
                # run are normal (the camera publishes at 15 Hz against a
                # 10 Hz control loop), and aborting on their sum would kill a
                # healthy run hours in. A sustained run of them is the real
                # signal that the camera is not there.
                self._cloud_misses += 1
                if self._cloud_misses == 1:
                    print(
                        "WARNING: scan_source=cloud but no cloud yet; using analytic scan",
                        flush=True,
                    )
                elif self._cloud_misses >= 200:
                    raise RuntimeError(
                        f"scan_source=cloud but {self.cloud_topic} has produced nothing "
                        "in 200 steps -- launch with sensors:=true (CFR_SENSORS=1)"
                    )
            else:
                self._cloud_hits += 1
                self._cloud_misses = 0
        if scan is None:
            scan = bale_geometry.lidar_scan(
                self.bales,
                pose.x,
                pose.y,
                pose.yaw,
                self.num_lidar_bins,
                self.lidar_fov_deg,
                self.lidar_max_range,
            )
        # Only the observation is corrupted; collision checks and the reward's
        # min_clearance stay on the ground-truth scan, the same split a real
        # robot has between what it senses and what physically happens.
        scan = zed_sim.apply(
            scan.copy(), self.zed_config, self.lidar_max_range, self._rng
        )
        normalized_scan = (scan / self.lidar_max_range).astype(np.float32)
        normalized_speed = np.clip(speed / self.obs_speed_scale, 0.0, 1.0)
        normalized_yaw_rate = np.clip((yaw_rate + 1.0) / 2.0, 0.0, 1.0)
        return np.concatenate(
            [normalized_scan, [normalized_speed, normalized_yaw_rate]]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # No WorldControl reset here -- teleporting and then holding zero
        # velocity repositions the car without touching the service that
        # destabilizes the server.
        self._settle(0.3)
        x, y, yaw = self._pick_start_pose()
        self._teleport(x, y, math.degrees(yaw))
        self._settle(0.3)

        before = time.monotonic()
        pose = self._wait_for_pose(since=before)

        self._episode_step = 0
        self._episode_time = 0.0
        self._prev_pose = pose
        self._prev_angular_z = 0.0
        self._prev_steer_fraction = 0.0
        self._progress_window = []
        self._cmd_speed = 0.0
        self._cmd_steer_fraction = 0.0
        # Re-acquire the loop from the pose the car actually landed at, not
        # the one requested: the teleport settles and can shift it slightly.
        self.course.reset(pose.x, pose.y, pose.yaw)

        observation = self._build_observation(pose, speed=0.0, yaw_rate=0.0)
        return observation, {}

    def step(self, action: np.ndarray):
        speed, steer_fraction = self.decode_action(action)
        # Steering slew, matching the physical servo (Traxxas 2075: 0.17 s/60
        # deg at the horn, ~3.5 rad/s at the road wheel). Without this the
        # policy learns to flick the wheels instantaneously and then fails the
        # moment a rate-limited controller sits in front of it: the v5 policy
        # scored 130 m raw but 44 m with the CasADi smoother, crashing every
        # episode, purely because it had never trained against a limit the
        # real hardware always imposes.
        max_delta_step = self.max_steering_rate / self.control_hz / MAX_STEERING_ANGLE
        steer_fraction = float(
            np.clip(
                steer_fraction,
                self._cmd_steer_fraction - max_delta_step,
                self._cmd_steer_fraction + max_delta_step,
            )
        )
        self._cmd_steer_fraction = steer_fraction
        speed = self._apply_traction(speed, steer_fraction)
        angular_z = 0.0
        if abs(speed) > 1e-3:
            # Signed bicycle model: reversing with the wheels turned swings
            # the nose the other way, exactly as the real car does.
            angular_z = (speed / WHEELBASE) * math.tan(
                steer_fraction * MAX_STEERING_ANGLE
            )

        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = angular_z
        self._cmd_pub.publish(twist)

        before = time.monotonic()
        self._advance_sim()
        pose = self._wait_for_pose(since=before)

        collided = bale_geometry.check_collision(self.bales, pose.x, pose.y, pose.yaw)
        scan = bale_geometry.lidar_scan(
            self.bales,
            pose.x,
            pose.y,
            pose.yaw,
            self.num_lidar_bins,
            self.lidar_fov_deg,
            self.lidar_max_range,
        )
        course_s, arc_progress, lateral_error = self.course.update(pose.x, pose.y)

        # The pose stream carries no twist, and the Ackermann plugin's
        # odometry is dead-reckoned (it ignores teleports), so velocities come
        # from differencing ground-truth poses.
        #
        # dt comes from the SIMULATION clock, not 1/control_hz. The step loop
        # sleeps on the wall clock, so those agree only while the simulator
        # keeps up. Rendering the ZED drops the real-time factor to ~0.63,
        # and assuming 0.1 s of sim time then under-reports every speed by
        # ~37% -- corrupting both the speed the policy observes and the yaw
        # rate the smoothness reward is computed from, exactly when the camera
        # is switched on.
        dt = pose.sim_stamp - self._prev_pose.sim_stamp
        if not (1e-4 < dt < 1.0):
            dt = 1.0 / self.control_hz
        measured_speed = (
            math.hypot(pose.x - self._prev_pose.x, pose.y - self._prev_pose.y) / dt
        )
        measured_yaw_rate = _wrap_to_pi(pose.yaw - self._prev_pose.yaw) / dt

        result = compute_reward(
            self.reward_config,
            arc_progress=arc_progress,
            min_clearance=float(scan.min()),
            angular_z=measured_yaw_rate,
            prev_angular_z=self._prev_angular_z,
            collided=collided,
            steer_fraction=self._cmd_steer_fraction,
            prev_steer_fraction=self._prev_steer_fraction,
        )

        self._prev_pose = pose
        self._prev_angular_z = measured_yaw_rate
        self._prev_steer_fraction = self._cmd_steer_fraction
        self._episode_step += 1
        self._episode_time += dt

        # Progress/time ratio, measured along the lap rather than along the
        # car's nose: circling, sawing in place and running the loop backwards
        # all fail it, and none of them used to. Cutting these episodes early
        # is most of the point -- a 120 s episode spent crawling is 1200 steps
        # of rollout that taught the policy nothing.
        window_steps = max(1, round(self.progress_window_s * self.control_hz))
        self._progress_window.append(arc_progress)
        self._progress_window = self._progress_window[-window_steps:]
        too_slow = (
            len(self._progress_window) == window_steps
            and sum(self._progress_window)
            < self.min_progress_speed * self.progress_window_s
        )

        terminated = collided
        truncated = self._episode_time >= self.episode_time_limit_s or too_slow

        observation = self._build_observation(
            pose, measured_speed, measured_yaw_rate, scan
        )
        info = {
            "collided": collided,
            "too_slow": too_slow,
            "arc_progress": arc_progress,
            "course_s": course_s,
            "lap_distance": self.course.travelled,
            "laps": self.course.laps,
            "elapsed_s": self._episode_time,
            "lateral_error": lateral_error,
            "min_clearance": float(scan.min()),
            "speed": measured_speed,
        }
        return observation, result.total, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown()
        self._spin_thread.join(timeout=2.0)
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
