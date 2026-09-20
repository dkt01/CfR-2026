"""Gymnasium environment for RL-based driving on the Gazebo Obstacle Course.

Sibling to env.py's BaleFollowerEnv, not a subclass: the two courses do not
share enough to make inheritance the right shape.

The Obstacle Course is not a bale corridor. BaleFollowerEnv's perception and
collision ground truth (bale_geometry.py) is a 2D top-down OBB model, which
is exactly right for a course that is walls in a plane -- and exactly wrong
here. The ramps, tunnel, helix and bank are 3D: a top-down box model cannot
tell "the car is driving up the ramp it is meant to climb" from "the car has
hit a wall", because both look like the car overlapping a box in plan view.
So this env does not parse course geometry at all. It perceives the same way
the real car will -- through the ZED's simulated point cloud
(`cloud_scan.py`), height-gated against the camera's own pitch/roll each
step, which handles a slope by construction instead of needing to know one is
there. This is scan_source="cloud" from zed_sim's "future sim-to-real pass"
comment in config.yaml, made mandatory here rather than optional. Running
with `sensors:=true` (rendering the ZED) is therefore required, not
cosmetic -- see obstacle_course.launch.py.

Collision has no separate ground truth either, for the same reason
bale_geometry-style checks do not generalize: it is the raw (pre-noise) cloud
scan's minimum range crossing `collision_clearance`. That threshold has not
been validated against real sensor noise -- see ObstacleRewardConfig's
docstring in obstacle_reward.py before trusting it for anything but a first
pass.

Course layout (which buckets stand where, where each hoop sits along its
line) is randomized through `obstacle_randomizer_node`'s services, cycled
through a small pool of precomputed seeds rather than drawn fresh every
episode: drawing a fresh layout means ~10 `gz service` calls at roughly
340 ms each (see obstacle_randomizer_node.py), so re-randomizing every reset
would spend more wall-clock moving buckets than driving. Holding each of a
handful of seeds for several consecutive episodes keeps that cost bounded
while still preventing the policy from overfitting to one arrangement -- see
`randomize_layout`/`layout_seeds`/`episodes_per_layout` below.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass

import gymnasium
import numpy as np
import rclpy
import requests
from cfr_interfaces.msg import HoopStatus
from geometry_msgs.msg import PoseStamped, Twist
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

import bale_geometry
import zed_sim
from cloud_scan import points_from_pointcloud2, scan_from_points
from env import (
    MAX_STEERING_ANGLE,
    WHEELBASE,
    _unpause_world,
    _wrap_to_pi,
    _yaw_from_quaternion,
)
from obstacle_course_path import CourseProgress
from obstacle_reward import ObstacleRewardConfig, compute_reward

GRAVITY = 9.81


def _forward_travel(
    prev_x: float, prev_y: float, prev_yaw: float, x: float, y: float
) -> float:
    """Displacement projected onto the heading the step started with.

    Local rather than imported: `reward.forward_progress` is removed on the
    branch that reworks the Speed Course, and this is the only use left.
    Note what it is and is not -- it is longitudinal travel, which the car's
    own tachometer measures, so it may inform the observation. It is *not*
    course progress; see obstacle_course_path.py.
    """
    return (x - prev_x) * math.cos(prev_yaw) + (y - prev_y) * math.sin(prev_yaw)


# Full scale for the observed yaw rate, in rad/s. The car cannot exceed
# roughly 2.8 rad/s -- grip caps it to ~2.1 m/s at full lock, and
# (2.1 / WHEELBASE) * tan(MAX_STEERING_ANGLE) is 2.77 -- so this covers the
# whole achievable range with a little headroom. The previous +-1.0 rad/s
# saturated through every tight corner, hiding exactly the information the
# policy needed there.
YAW_RATE_SCALE = 3.0


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    stamp: float
    sim_stamp: float = 0.0
    # Only used to tell the bridge deck from the tunnel running underneath
    # it, which are the same (x, y) -- see obstacle_course_path.Z_WEIGHT.
    z: float = 0.0


class ObstacleCourseEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        sdf_path: str,
        teleport_url: str = "http://localhost:9003/api/sim/teleport",
        world_name: str = "cfr_obstacle_course",
        pose_topic: str = "/zed/zed_node/pose",
        num_lidar_bins: int = 36,
        lidar_fov_deg: float = 110.0,
        lidar_max_range: float = 6.0,
        control_hz: float = 10.0,
        max_speed: float = 3.5,
        reverse_speed: float = 0.5,
        episode_time_limit_s: float = 90.0,
        stuck_window_s: float = 5.0,
        stuck_distance: float = 0.5,
        randomize_start: bool = False,
        reward_config: ObstacleRewardConfig | None = None,
        odom_timeout_s: float = 2.0,
        traction: float = 0.6,
        max_steering_rate: float = 3.5,
        scan_source: str = "cloud",
        cloud_topic: str = "/zed/zed_node/point_cloud/cloud_registered",
        zed_config: zed_sim.ZedSimConfig | None = None,
        randomize_layout: bool = True,
        layout_seeds: list[int] | None = None,
        episodes_per_layout: int = 5,
        randomizer_node: str = "/obstacle_randomizer",
        hoop_monitor_node: str = "/hoop_monitor",
        randomize_timeout_s: float = 60.0,
        teleport_timeout_s: float = 5.0,
        lap_finish_tolerance_m: float = 0.5,
        frame_stack: int = 4,
        start_anywhere_prob: float = 0.8,
        start_anywhere_margin_m: float = 8.0,
        ground_step: float | None = 0.15,
    ) -> None:
        super().__init__()
        self.traction = traction
        self.max_steering_rate = max_steering_rate
        self.zed_config = zed_config or zed_sim.ZedSimConfig(enabled=True)
        if self.zed_config.enabled:
            lidar_fov_deg = min(lidar_fov_deg, self.zed_config.hfov_deg)
        self.world_name = world_name
        self.teleport_url = teleport_url
        self.teleport_timeout_s = teleport_timeout_s
        self.num_lidar_bins = num_lidar_bins
        self.lidar_fov_deg = lidar_fov_deg
        self.lidar_max_range = lidar_max_range
        self.control_hz = control_hz
        self.max_speed = max_speed
        self.reverse_speed = reverse_speed
        self.obs_speed_scale = max_speed
        self.episode_time_limit_s = episode_time_limit_s
        self.stuck_window_s = stuck_window_s
        self.stuck_distance = stuck_distance
        self.randomize_start = randomize_start
        self.reward_config = reward_config or ObstacleRewardConfig()
        self.odom_timeout_s = odom_timeout_s
        self.lap_finish_tolerance_m = lap_finish_tolerance_m
        self.start_anywhere_prob = start_anywhere_prob
        self.start_anywhere_margin_m = start_anywhere_margin_m
        # Without this the ramp reads as a wall 2.18 m ahead and the car is
        # penalised for climbing it -- see cloud_scan._scan_tracking_the_ground.
        self.ground_step = ground_step
        self._start_s = 0.0

        # Generic XML lookup by model name -- no course-shape assumptions --
        # so this works against the Obstacle Course's "slash" model exactly
        # like it does against the Speed Course's.
        self.spawn_pose = bale_geometry.parse_vehicle_spawn(sdf_path, "slash")

        self.scan_source = scan_source
        if self.scan_source != "cloud":
            print(
                f"WARNING: scan_source={self.scan_source!r} on the Obstacle Course "
                "has no course geometry to fall back on and will report a "
                "permanently clear scan. This is only useful for smoke-testing "
                "the episode loop -- real training needs scan_source: cloud "
                "and sensors:=true.",
                flush=True,
            )
        self.cloud_topic = cloud_topic
        self._latest_cloud = None
        self._cloud_lock = threading.Lock()
        self._tilt = (0.0, 0.0)
        self._cloud_hits = 0
        self._cloud_misses = 0

        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Stacked frames, because one is not a state. A single forward depth
        # profile cannot tell which way the car is moving through it, and
        # plenty of spots on this course look alike from a 110 degree scan --
        # an open bucket room especially. With a feed-forward policy the only
        # way to recover that is to show it several frames at once.
        self.frame_stack = max(1, frame_stack)
        self._frame_dim = num_lidar_bins + 2
        self._frames: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.frame_stack
        )
        self.observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._frame_dim * self.frame_stack,),
            dtype=np.float32,
        )

        self._pose_lock = threading.Lock()
        self._latest_pose: Pose2D | None = None
        self._episode_step = 0
        self._episode_time = 0.0
        self._prev_pose: Pose2D | None = None
        self._prev_angular_z = 0.0
        self._prev_steer_fraction = 0.0
        # (course progress, sim dt) per step, trimmed to stuck_window_s.
        self._stuck_window_travel: collections.deque[tuple[float, float]] = (
            collections.deque()
        )
        self._cmd_speed = 0.0
        self._cmd_steer_fraction = 0.0
        self._rng = np.random.default_rng()

        # Course progress, for the reward only -- never for the observation.
        # See obstacle_course_path.py on why measuring progress against the
        # course rather than against the car's own heading is the whole point.
        self._course = CourseProgress()
        self._prev_s = 0.0
        self._lap_completed = False
        # Verified once here rather than sampled blind every reset -- see
        # CourseProgress.safe_start_arcs for what "safe" is checked against.
        self._start_arcs = (
            self._course.safe_start_arcs(
                sdf_path, finish_margin_m=start_anywhere_margin_m
            )
            if start_anywhere_prob > 0.0
            else []
        )

        # Course layout randomization -- see the module docstring for why
        # this cycles a small pool of seeds instead of redrawing every reset.
        self.randomize_layout = randomize_layout
        self.layout_seeds = list(layout_seeds) if layout_seeds else None
        self.episodes_per_layout = max(1, episodes_per_layout)
        self._episode_count = 0
        self._last_layout_index: int | None = None
        self.randomize_timeout_s = randomize_timeout_s

        # Per-run hoop tracking, latest-known from hoop_monitor_node's
        # ~/status. Reset every episode via its ~/reset service (independent
        # of whether the layout itself was re-randomized this episode) so
        # "missed a hoop" always means "missed one THIS episode."
        self._hoop_status: HoopStatus | None = None
        self._hoop_status_lock = threading.Lock()
        self._prev_any_missed = False
        self._prev_passed_count = 0

        self._closed = False
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node("obstacle_course_rl_env")
        self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel", 10)
        self._node.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)
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
        self._node.create_subscription(
            HoopStatus,
            f"{hoop_monitor_node}/status",
            self._on_hoop_status,
            10,
        )

        self._randomize_client = self._node.create_client(
            Trigger, f"{randomizer_node}/randomize"
        )
        self._randomizer_params_client = self._node.create_client(
            SetParameters, f"{randomizer_node}/set_parameters"
        )
        self._hoop_reset_client = self._node.create_client(
            Trigger, f"{hoop_monitor_node}/reset"
        )

        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        if not _unpause_world(world_name):
            raise RuntimeError(f"could not start world '{world_name}' running")

    # ------------------------------------------------------------- callbacks

    def _on_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        pose = Pose2D(
            x=p.x,
            y=p.y,
            yaw=_yaw_from_quaternion(q.x, q.y, q.z, q.w),
            stamp=time.monotonic(),
            sim_stamp=self._sim_time,
            z=p.z,
        )
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

    def _on_hoop_status(self, msg: HoopStatus) -> None:
        with self._hoop_status_lock:
            self._hoop_status = msg

    # ------------------------------------------------------------ perception

    def _raw_cloud_scan(self) -> np.ndarray | None:
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
            ground_step=self.ground_step,
        )

    def _ground_truth_scan(self) -> np.ndarray:
        """Pre-noise range scan, for collision/reward math -- never the observation."""
        if self.scan_source != "cloud":
            return np.full(self.num_lidar_bins, self.lidar_max_range, dtype=np.float32)
        scan = self._raw_cloud_scan()
        if scan is None:
            self._cloud_misses += 1
            if self._cloud_misses == 1:
                print(
                    "WARNING: scan_source=cloud but no cloud yet; "
                    "reporting a clear scan until one arrives",
                    flush=True,
                )
            elif self._cloud_misses >= 200:
                raise RuntimeError(
                    f"scan_source=cloud but {self.cloud_topic} has produced "
                    "nothing in 200 steps -- launch with sensors:=true "
                    "(CFR_SENSORS=1)"
                )
            return np.full(self.num_lidar_bins, self.lidar_max_range, dtype=np.float32)
        self._cloud_hits += 1
        self._cloud_misses = 0
        return scan

    def _build_observation(
        self, scan: np.ndarray, speed: float, yaw_rate: float
    ) -> np.ndarray:
        noisy = zed_sim.apply(
            scan.copy(), self.zed_config, self.lidar_max_range, self._rng
        )
        normalized_scan = (noisy / self.lidar_max_range).astype(np.float32)
        # Both of these are signed and centred on 0.5. Speed used to be a
        # magnitude clipped at zero, which made reversing indistinguishable
        # from standing still, and yaw rate used to saturate at +-1 rad/s,
        # which is most of a tight corner (see YAW_RATE_SCALE).
        normalized_speed = np.clip((speed / self.obs_speed_scale + 1.0) / 2.0, 0.0, 1.0)
        normalized_yaw_rate = np.clip((yaw_rate / YAW_RATE_SCALE + 1.0) / 2.0, 0.0, 1.0)
        frame = np.concatenate(
            [normalized_scan, [normalized_speed, normalized_yaw_rate]]
        ).astype(np.float32)
        if not self._frames:
            # First frame of an episode: repeat it, so the stack never
            # contains anything from the episode before.
            self._frames.extend([frame] * self.frame_stack)
        else:
            self._frames.append(frame)
        return np.concatenate(self._frames).astype(np.float32)

    # ----------------------------------------------------------- ROS helpers

    def _call_service(self, client, request, timeout_s: float = 8.0):
        """Call a service and poll for its result.

        Not `rclpy.spin_until_future_complete`: the node is already spinning
        continuously on `self._spin_thread`, and spinning it again from here
        would be a second spinner on the same node. That background thread is
        the one that actually completes the future; this just waits for it.
        """
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"service {client.srv_name} not available")
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{client.srv_name} timed out")
            time.sleep(0.02)
        return future.result()

    def _set_randomizer_seed(self, seed: int) -> None:
        request = SetParameters.Request()
        parameter = Parameter()
        parameter.name = "seed"
        parameter.value = ParameterValue(
            type=ParameterType.PARAMETER_INTEGER, integer_value=seed
        )
        request.parameters = [parameter]
        self._call_service(self._randomizer_params_client, request)

    def _maybe_randomize_layout(self) -> None:
        """Draw a new course layout if this episode's dwell period calls for one.

        `layout_seeds` set: cycles that pool, holding each seed for
        `episodes_per_layout` consecutive episodes -- a handful of distinct,
        repeatable layouts rather than one redrawn every reset. Unset: draws
        a fresh layout every `episodes_per_layout` episodes (seed=-1, which
        obstacle_randomizer_node reads as "draw a new one").
        """
        if not self.randomize_layout:
            return
        if self.layout_seeds:
            index = (self._episode_count // self.episodes_per_layout) % len(
                self.layout_seeds
            )
            if index == self._last_layout_index:
                return
            self._set_randomizer_seed(self.layout_seeds[index])
            self._last_layout_index = index
        elif self._episode_count % self.episodes_per_layout != 0:
            return
        else:
            self._set_randomizer_seed(-1)
        # Longer than _call_service's default: this one triggers roughly a
        # dozen sequential `gz service` calls inside obstacle_randomizer_node
        # (observed ~340 ms each), so its own latency already approaches the
        # 8 s default under a quiet host -- and a second training container
        # or anything else competing for CPU pushes it well past that
        # (observed: timed out at 8 s with another training run's Gazebo
        # server also active on the same host).
        result = self._call_service(
            self._randomize_client,
            Trigger.Request(),
            timeout_s=self.randomize_timeout_s,
        )
        self._node.get_logger().info(f"course layout: {result.message}")

    # ------------------------------------------------------------ mechanics

    def _settle(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._cmd_pub.publish(Twist())
            time.sleep(0.02)

    def _teleport(
        self,
        x: float,
        y: float,
        heading_deg: float,
        ground_z: float = 0.0,
        attempts: int = 3,
    ) -> None:
        message = None
        for attempt in range(attempts):
            try:
                response = requests.post(
                    self.teleport_url,
                    json={"x": x, "y": y, "heading": heading_deg, "z": ground_z},
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

    def _wait_for_pose(self, since: float) -> Pose2D:
        deadline = time.monotonic() + self.odom_timeout_s
        while time.monotonic() < deadline:
            with self._pose_lock:
                pose = self._latest_pose
            if pose is not None and pose.stamp > since:
                return pose
            time.sleep(0.005)
        raise TimeoutError("no pose received on the ZED map-frame pose topic")

    def decode_action(self, action: np.ndarray) -> tuple[float, float]:
        fraction = (float(np.clip(action[0], -1.0, 1.0)) + 1.0) / 2.0
        speed = -self.reverse_speed + fraction * (self.max_speed + self.reverse_speed)
        steer_fraction = float(np.clip(action[1], -1.0, 1.0))
        return speed, steer_fraction

    def _apply_traction(self, speed: float, steer_fraction: float) -> float:
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
        """Jitter around the SDF spawn pose.

        Unlike BaleFollowerEnv's `_pick_start_pose`, this has no course
        geometry to validate a candidate against -- see the module docstring
        for why. `randomize_start` therefore defaults to False here; turning
        it on trusts that a small jitter this close to the start line cannot
        land inside anything.
        """
        if not self.randomize_start:
            return self.spawn_pose
        x, y, yaw = self.spawn_pose
        return (
            x + self._rng.uniform(-0.3, 0.3),
            y + self._rng.uniform(-0.15, 0.15),
            yaw + self._rng.uniform(-0.1, 0.1),
        )

    def _pick_start_s(self) -> float:
        """Where round the lap this episode begins.

        The course is ~75 m of tunnel, gravel, bank, buckets and hoops, and
        a run that always starts on the line only ever sees the first few
        metres of it: the car has to solve section N before it can even
        observe section N+1. At the ~350 episodes a 300k-step run affords,
        that is not enough to reach the back half of the course, let alone
        learn it.

        Starting most episodes somewhere along the centerline turns that
        sequential problem into a parallel one -- every section gets
        practised from the first rollout. The remaining `1 - prob` start on
        the line so the policy still trains on the real thing, and
        evaluation always does (see `reset`'s `start_s` option).

        Only the ground-level span is drawn from: the ramp/deck/helix is a
        narrow raised deck with a drop either side, and the teleport places
        the car at a fixed height, so a spawn there would drop it off the
        edge. The car still drives that section -- it just is not dealt into
        the middle of it.
        """
        if not self._start_arcs or self._rng.random() >= self.start_anywhere_prob:
            return 0.0
        return float(self._start_arcs[self._rng.integers(len(self._start_arcs))])

    # ----------------------------------------------------------------- gym

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._maybe_randomize_layout()
        self._episode_count += 1

        self._settle(0.3)
        start_s = (options or {}).get("start_s")
        if start_s is None:
            start_s = self._pick_start_s()
        ground_z = 0.0
        if start_s > 0.0:
            x, y, ground_z, yaw = self._course.pose_at(start_s)
        else:
            x, y, yaw = self._pick_start_pose()
        self._teleport(x, y, math.degrees(yaw), ground_z)
        self._settle(0.3)

        before = time.monotonic()
        pose = self._wait_for_pose(since=before)

        # Clear hoop_monitor_node AFTER the teleport, never before it.
        # Teleporting is a jump, and a jump across a hoop's plane outside
        # its gate is exactly what that node is built to report as a miss.
        # Resetting first therefore charges the new episode for the *move
        # into* its own start pose. Harmless while every episode started on
        # the line, 0.7 m from the spawn and nowhere near a hoop; fatal once
        # starts are dealt round the course, where the jump routinely flies
        # past one. It read as hoop_miss_rate 0.75 in episodes that had
        # travelled 0.4 m -- the outcome counters are what made it visible.
        # Cheap (no gz service calls, just clears that node's Python state).
        self._call_service(self._hoop_reset_client, Trigger.Request())
        with self._hoop_status_lock:
            self._hoop_status = None
        self._prev_any_missed = False
        self._prev_passed_count = 0

        self._episode_step = 0
        self._episode_time = 0.0
        self._prev_pose = pose
        self._prev_angular_z = 0.0
        self._prev_steer_fraction = 0.0
        self._stuck_window_travel.clear()
        self._cmd_speed = 0.0
        self._cmd_steer_fraction = 0.0

        # Seed the projection window at the dealt start, not at the line:
        # the search is deliberately local (see CourseProgress) so it cannot
        # find the car 40 m away on its own.
        self._course.reset(start_s)
        self._prev_s = self._course.update(pose.x, pose.y, pose.z)
        self._start_s = self._prev_s
        self._lap_completed = False
        self._frames.clear()

        scan = self._ground_truth_scan()
        observation = self._build_observation(scan, speed=0.0, yaw_rate=0.0)
        return observation, {}

    def step(self, action: np.ndarray):
        speed, steer_fraction = self.decode_action(action)
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

        scan = self._ground_truth_scan()
        min_clearance = float(scan.min())
        collided = min_clearance < self.reward_config.collision_clearance

        # Longitudinal displacement, signed: what the car's own tachometer
        # plus direction estimate would report, so it is fair game for the
        # observation. Distinct from progress_s below, which is measured
        # against the course and is reward-only.
        travel = _forward_travel(
            self._prev_pose.x, self._prev_pose.y, self._prev_pose.yaw, pose.x, pose.y
        )

        dt = pose.sim_stamp - self._prev_pose.sim_stamp
        if not (1e-4 < dt < 1.0):
            dt = 1.0 / self.control_hz
        measured_speed = travel / dt
        measured_yaw_rate = _wrap_to_pi(pose.yaw - self._prev_pose.yaw) / dt

        s_now = self._course.update(pose.x, pose.y, pose.z)
        progress_s = s_now - self._prev_s
        self._prev_s = s_now
        if s_now >= self._course.lap_length - self.lap_finish_tolerance_m:
            self._lap_completed = True

        with self._hoop_status_lock:
            hoop_status = self._hoop_status
        hoop_missed_now = False
        hoops_passed_now = 0
        if hoop_status is not None:
            any_missed = bool(hoop_status.any_missed)
            passed_count = sum(1 for value in hoop_status.passed if value)
            hoop_missed_now = any_missed and not self._prev_any_missed
            hoops_passed_now = max(0, passed_count - self._prev_passed_count)
            self._prev_any_missed = any_missed
            self._prev_passed_count = passed_count

        # Net progress *round the course* over the window, not distance
        # travelled: the old `abs(...)` here counted spinning on the spot and
        # driving backwards as getting somewhere, so a car doing neither
        # usefully never tripped the detector. This is also the
        # progress-per-time floor -- below stuck_distance per stuck_window_s
        # the episode is not worth finishing.
        # Measured in *sim seconds*, not in steps. A step is only
        # 1/control_hz seconds if the simulator is keeping up with the wall
        # clock, and it is not: dt is measured at 0.058 s against the 0.1 s
        # the loop assumes, because the env paces itself against the wall
        # clock while Gazebo runs freely (see training.launch.py on why it
        # cannot be stepped). Counting steps therefore made the detector
        # 1.7x more aggressive than stuck_window_s says, and its
        # aggressiveness drift with host load, which is not something a
        # termination condition should do.
        self._stuck_window_travel.append((progress_s, dt))
        window_dt = sum(entry[1] for entry in self._stuck_window_travel)
        while (
            len(self._stuck_window_travel) > 1
            and window_dt - self._stuck_window_travel[0][1] >= self.stuck_window_s
        ):
            window_dt -= self._stuck_window_travel.popleft()[1]
        stuck = window_dt >= self.stuck_window_s and (
            sum(entry[0] for entry in self._stuck_window_travel) < self.stuck_distance
        )

        result = compute_reward(
            self.reward_config,
            progress_s=progress_s,
            time_remaining_s=self.episode_time_limit_s - self._episode_time,
            min_clearance=min_clearance,
            angular_z=measured_yaw_rate,
            prev_angular_z=self._prev_angular_z,
            collided=collided,
            hoop_missed=hoop_missed_now,
            hoops_passed_this_step=hoops_passed_now,
            steer_fraction=self._cmd_steer_fraction,
            prev_steer_fraction=self._prev_steer_fraction,
            lap_completed=self._lap_completed,
            stuck=stuck,
        )

        self._prev_pose = pose
        self._prev_angular_z = measured_yaw_rate
        self._prev_steer_fraction = self._cmd_steer_fraction
        self._episode_step += 1
        self._episode_time += dt

        # A missed hoop fails the run outright per the rules, and a finished
        # lap is the goal -- no point spending the rest of the rollout in
        # either case.
        terminated = collided or hoop_missed_now or self._lap_completed
        truncated = self._episode_time >= self.episode_time_limit_s or stuck

        observation = self._build_observation(scan, measured_speed, measured_yaw_rate)
        info = {
            "collided": collided,
            "hoop_missed": hoop_missed_now,
            "hoops_passed": hoops_passed_now,
            "stuck": stuck,
            "lap_completed": self._lap_completed,
            "progress_distance": progress_s,
            "course_s": s_now,
            # Where this episode was dealt in, and how far it has got from
            # there -- with start_anywhere_prob on, raw course_s says more
            # about the draw than about the policy.
            "start_s": self._start_s,
            "course_advance": s_now - self._start_s,
            "min_clearance": min_clearance,
            "speed": measured_speed,
            # Commanded vs achieved, and the sim-time step they are measured
            # over. "the car is barely advancing" has three different causes
            # -- the policy is asking for little, the car is not delivering
            # what is asked, or dt is not what the control loop assumes --
            # and they need different fixes.
            "cmd_speed": speed,
            "dt": dt,
            # Where the return actually went. An episode's total can sit
            # near zero either because nothing happened or because a large
            # progress reward was cancelled by a large penalty, and those
            # want opposite fixes.
            "steer": self._cmd_steer_fraction,
            "yaw_rate": measured_yaw_rate,
            "r_progress": result.progress,
            "r_proximity": result.proximity,
            "r_touch": result.touch,
            "r_smoothness": result.smoothness,
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
