#!/usr/bin/env python3
"""Race the planned course: pure-pursuit tracking of course_path.json.

The course geometry is static and known, so this replaces perception-driven
driving (the RL policy on a simulated ZED scan) with plan + pose tracking:
course_path.py plans the racing line and minimum-time speed profile once,
offline; this node needs only a pose estimate at runtime.

Pose sources, selected with --pose-topic / --pose-msg:
  * default: Gazebo's ground-truth pose bridge (tf2_msgs/TFMessage on
    /world/<world>/dynamic_pose/info), same as the RL stack uses in sim.
  * `--pose-msg odom` for a nav_msgs/Odometry topic -- e.g. the QuestNav
    pose from the Quest 3, republished on the Jetson. The Quest pose must
    already be transformed into the course frame (the frame the SDF bales
    are expressed in); do that calibration in the republisher, not here.

The RL policy remains the fallback for when the geometry is NOT trustworthy;
on the real car the ZED scan should become a safety layer (emergency slow /
stop on unexpected obstacles), which is a subscriber away in this node.

    ros2 launch cfr_arduino_bridge training.launch.py    # or validate.sh's stack
    python path_racer.py                                 # laps + lap times

    DEMO_PROGRAM=path_racer.py ./validate.sh                 # watch in the viewer
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from env import MAX_STEERING_ANGLE, WHEELBASE, _unpause_world, _yaw_from_quaternion
from mpc_tracker import MpcConfig, MpcTracker

GRAVITY = 9.81
DEFAULT_PATH = Path(__file__).resolve().parent / "course_path.json"


class PathRacer(Node):
    def __init__(self, plan: dict, args) -> None:
        super().__init__("path_racer")
        self.xy = np.stack([plan["x"], plan["y"]], axis=1)
        self.v_profile = np.array(plan["v"]) * args.speed_scale
        # Hairpin crawl caps, on top of the friction-circle profile: the
        # profile says what the tires allow, but tracking error plus command
        # lag makes tight-radius sections wedge-prone well below that limit.
        # Empirical from sim runs: r < ~2.2 m needs a crawl.
        # Speed FLOOR in curves, not a cap. Measured on the sim vehicle: at
        # full lock 0.8 m/s yaws 89 deg in 2 s while 0.5 m/s barely rotates at
        # all -- below roughly 0.8 m/s the steering cannot overcome tire
        # scrub, so crawling into a hairpin is what wedges the car. The
        # friction-circle profile stays the upper bound; this is the lower one.
        self.curvature = np.array(plan["curvature"])
        kappa = np.abs(self.curvature)
        self.turn_speed_floor = np.where(kappa > 0.25, 1.3, 0.0)
        self.n = len(self.xy)
        self.lap_length = plan["length_m"]
        self.a_max = plan["traction"] * GRAVITY
        # 20 Hz, because that is what the loop actually sustains. A warm
        # MPC solve is ~11.5 ms, which looks like it fits a 50 Hz budget, but
        # a full tick (reference generation, solve, publish) costs ~34 ms, so
        # requesting 50 Hz yields a jittery ~29 Hz and measurably WORSE
        # driving than a steady 20 Hz: at 4 m/s, 6 clean laps at 20 Hz became
        # 1 lap and 19 stuck events at 50. Regular beats fast.
        #
        # Raising this only helps once the tick itself is cheaper -- a
        # code-generated QP solver in place of IPOPT is the lever, not a
        # bigger number here.
        self.control_hz = float(getattr(args, "control_hz", 20.0))
        self.min_lookahead = 0.6
        self.max_lookahead = 1.8
        self.lookahead_gain = 0.4  # seconds of travel

        self._lock = threading.Lock()
        self._pose = None
        self._index: int | None = None
        self._direction_set = False
        self._cmd_speed = 0.0
        self._lap_progress = 0.0
        self._lap_started = time.monotonic()
        self._lap_count = 0
        self._lap_recoveries = 0
        self.debug = bool(getattr(args, "debug", False))

        # CasADi MPC is the primary tracker: pure pursuit cuts 0.2-0.4 m
        # inside curves (measured in sim), which the hairpins do not forgive.
        # Pure pursuit remains as the per-tick fallback when a solve fails.
        self.mpc = None
        self._prev_pose_speed: tuple[float, float, float] | None = None
        self._measured_speed = 0.0
        if not args.no_mpc:
            self.mpc = MpcTracker(MpcConfig(
                wheelbase=WHEELBASE,
                max_steering_angle=MAX_STEERING_ANGLE,
                traction=plan["traction"],
                max_speed=plan["max_speed"] * args.speed_scale,
            ))

        # Stuck recovery, same shape as run_policy.py's: reverse briefly,
        # steering toward the path side so the nose swings back onto it.
        self.stuck_window_s = 2.0
        self.stuck_distance = 0.12
        self.recovery_duration_s = 1.2
        self._pose_history: list[tuple[float, float, float]] = []
        self._recovery_until = 0.0
        self._recovery_delta = 0.0
        self._no_trigger_until = time.monotonic() + 3.0
        self._rejoining = False

        if args.pose_msg == "tf":
            self.create_subscription(TFMessage, args.pose_topic, self._on_tf, 10)
        else:
            self.create_subscription(Odometry, args.pose_topic, self._on_odom, 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / self.control_hz, self._on_tick)
        self.get_logger().info(
            f"racing {self.lap_length:.1f} m loop, planned flying lap "
            f"{plan['estimated_time_s'] / args.speed_scale:.1f} s; waiting for pose on {args.pose_topic}"
        )

    def _on_tf(self, msg: TFMessage) -> None:
        if not msg.transforms:
            return
        t = msg.transforms[0].transform
        q = t.rotation
        with self._lock:
            self._pose = (t.translation.x, t.translation.y,
                          _yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose
        with self._lock:
            self._pose = (p.position.x, p.position.y,
                          _yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                               p.orientation.z, p.orientation.w))

    def _nearest_index(self, x: float, y: float) -> int:
        """Nearest path sample, searched locally but re-globalized when lost.

        A purely local window drifts onto the wrong arm of a hairpin (the
        return corridor is ~1 m away) and then never recovers -- observed as
        cross-track errors of 5-17 m while the car was actually on the line.
        So: take the local answer, and if it is implausibly far, re-search
        globally.
        """
        if self._index is None:
            return int(np.argmin(np.linalg.norm(self.xy - (x, y), axis=1)))
        window = np.arange(self._index - 30, self._index + 31) % self.n
        local = int(window[np.argmin(np.linalg.norm(self.xy[window] - (x, y), axis=1))])
        if np.linalg.norm(self.xy[local] - (x, y)) > 1.0:
            return int(np.argmin(np.linalg.norm(self.xy - (x, y), axis=1)))
        return local

    def _set_direction(self, index: int, yaw: float) -> None:
        tangent = self.xy[(index + 1) % self.n] - self.xy[index - 1]
        if math.cos(yaw) * tangent[0] + math.sin(yaw) * tangent[1] < 0.0:
            self.xy = self.xy[::-1].copy()
            self.v_profile = self.v_profile[::-1].copy()
            # curvature and the floor derived from it are indexed by the same
            # samples, so they have to be reversed with them. Left unflipped,
            # every curvature lookup returned the mirror-image point of the
            # loop: the speed floor was relaxed in the hairpins (where it is
            # what keeps the car steerable) and raised on the straights.
            self.curvature = self.curvature[::-1].copy()
            self.turn_speed_floor = self.turn_speed_floor[::-1].copy()
            self._index = self.n - 1 - index
            self.get_logger().info("path direction flipped to match initial heading")
        self._direction_set = True

    def _check_stuck(self, now: float, x: float, y: float) -> bool:
        self._pose_history.append((now, x, y))
        while self._pose_history and now - self._pose_history[0][0] > self.stuck_window_s + 0.5:
            self._pose_history.pop(0)
        if now < self._no_trigger_until:
            return False
        oldest = self._pose_history[0]
        return (now - oldest[0] >= self.stuck_window_s
                and math.hypot(x - oldest[1], y - oldest[2]) < self.stuck_distance)

    def _on_tick(self) -> None:
        with self._lock:
            pose = self._pose
        if pose is None:
            return
        x, y, yaw = pose
        now = time.monotonic()

        if now < self._recovery_until:
            twist = Twist()
            twist.linear.x = -0.6
            twist.angular.z = (twist.linear.x / WHEELBASE) * math.tan(self._recovery_delta)
            self._cmd_pub.publish(twist)
            return

        if self._recovery_until and not self._rejoining:
            # Recovery just ended. Hand back deliberately: clear the stuck
            # history, drop the MPC's warm start (it is a plan from before the
            # reverse), and re-acquire the path index globally rather than
            # from a window centred on where we were when we got stuck.
            self._rejoining = True
            self._index = None
            self._measured_speed = 0.0
            self._prev_pose_speed = None
            if self.mpc is not None:
                self.mpc.reset()
            if self.debug:
                print(f"DBG recovery ended at ({x:.2f},{y:.2f}), re-acquiring path",
                      flush=True)

        index = self._nearest_index(x, y)
        if not self._direction_set:
            self._set_direction(index, yaw)
            index = self._nearest_index(x, y)

        if self._check_stuck(now, x, y):
            # Steer toward where the path continues so reversing swings the
            # nose back onto it.
            target = self.xy[(index + 5) % self.n]
            bearing = math.atan2(target[1] - y, target[0] - x) - yaw
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            # Half lock: full lock in a 0.95 m corridor pivots the tail into
            # the opposite wall instead of backing clear.
            #
            # NOTE: the sign here looks wrong against the bicycle model --
            # reversing (v < 0) makes yaw rate (v/L)*tan(delta) swing the nose
            # opposite to forward travel, so this steers the nose AWAY from
            # the path. Inverting it measured worse (150 s lap, 21 stuck vs
            # 31.85 s clean), because turning the nose toward the line throws
            # the tail into the opposite wall of a 0.95 m corridor. Backing
            # out along the way we came in beats aiming the nose. Do not
            # "fix" this sign without a lap-time measurement.
            self._recovery_delta = math.copysign(0.5 * MAX_STEERING_ANGLE, bearing)
            if now - getattr(self, "_last_recovery_end", 0.0) < 5.0:
                self._escalation = min(getattr(self, "_escalation", 1.0) * 1.6, 4.0)
                # Repeated sticking in the same spot: alternate steering so
                # the car wiggles out instead of replaying the same arc.
                if self._escalation > 2.0:
                    self._recovery_delta = -self._recovery_delta
            else:
                self._escalation = 1.0
            self._recovery_until = now + self.recovery_duration_s * self._escalation
            self._last_recovery_end = self._recovery_until
            self._no_trigger_until = self._recovery_until + 2.0
            self._rejoining = False
            self._pose_history.clear()
            self._cmd_speed = 0.0
            self._lap_recoveries += 1
            if self.mpc is not None:
                self.mpc.reset()
            self.get_logger().info("stuck; reversing toward the line")
            return

        # Lap accounting from index progress (handles the wrap).
        if self._index is not None:
            steps = (index - self._index) % self.n
            # Only count small forward hops. A large jump means the index
            # search relocated (recovery, or a re-globalized search), and
            # crediting it as travel is what produced 6 s "laps" on a 110 m
            # loop; a backwards hop is the car reversing.
            if 0 < steps <= 20:
                self._lap_progress += steps * self.lap_length / self.n
                if self._lap_progress >= self.lap_length:
                    self._lap_count += 1
                    lap_time = now - self._lap_started
                    self.get_logger().info(
                        f"LAP {self._lap_count}: {lap_time:.2f} s "
                        f"({self.lap_length / lap_time:.2f} m/s avg, "
                        f"{self._lap_recoveries} recoveries)"
                    )
                    self._lap_progress -= self.lap_length
                    self._lap_started = now
                    self._lap_recoveries = 0
            elif steps > self.n // 2:
                self._lap_progress -= (self.n - steps) * self.lap_length / self.n
        self._index = index

        # Measured speed from pose differencing, for the MPC's initial state
        # (the drivetrain lags, so commanded speed overstates reality).
        if self._prev_pose_speed is not None:
            pt, px, py = self._prev_pose_speed
            if now - pt > 1e-3:
                raw = math.hypot(x - px, y - py) / (now - pt)
                self._measured_speed = 0.6 * self._measured_speed + 0.4 * min(raw, 8.0)
        self._prev_pose_speed = (now, x, y)

        step_len = self.lap_length / self.n
        if self.mpc is not None:
            # Time-parameterized reference: step along the path at profile
            # speed so hairpin braking enters the horizon a second early.
            ref_xy = np.empty((self.mpc.config.horizon, 2))
            ref_v = np.empty(self.mpc.config.horizon)
            travelled = 0.0
            for k in range(self.mpc.config.horizon):
                ref_index = (index + int(travelled / step_len)) % self.n
                speed_k = max(float(self.v_profile[ref_index]),
                              float(self.turn_speed_floor[ref_index]), 0.3)
                travelled += speed_k * self.mpc.config.dt
                point_index = (index + int(travelled / step_len)) % self.n
                ref_xy[k] = self.xy[point_index]
                ref_v[k] = speed_k
            # Worst curvature over the horizon, not at the car: the floor has
            # to be up before the hairpin, not once already in it.
            horizon_end = (index + max(1, int(travelled / step_len))) % self.n
            if horizon_end > index:
                kappa_ahead = float(np.abs(self.curvature[index:horizon_end + 1]).max())
            else:
                kappa_ahead = float(max(np.abs(self.curvature[index:]).max(),
                                        np.abs(self.curvature[:horizon_end + 1]).max()))
            result = self.mpc.solve(x, y, yaw, self._measured_speed,
                                    ref_xy, ref_v, kappa_ahead)
            if result is not None:
                speed_cmd, delta_cmd = result
                self._cmd_speed = speed_cmd
                twist = Twist()
                twist.linear.x = speed_cmd
                if abs(speed_cmd) > 1e-3:
                    twist.angular.z = (speed_cmd / WHEELBASE) * math.tan(delta_cmd)
                self._cmd_pub.publish(twist)
                if self.debug and now - getattr(self, "_last_debug", 0.0) > 0.5:
                    self._last_debug = now
                    crosstrack = float(np.linalg.norm(self.xy[index] - (x, y)))
                    print(f"DBG t={now:.1f} pos=({x:.2f},{y:.2f}) idx={index} "
                          f"xtrack={crosstrack:.2f} v_cmd={speed_cmd:.2f} "
                          f"v_meas={self._measured_speed:.2f} delta={delta_cmd:+.2f} MPC",
                          flush=True)
                return
            # fall through to pure pursuit on solver failure

        # Pure pursuit fallback: chase a point one lookahead ahead.
        # Speed comes from the minimum of the profile over the next ~1.2 s of
        # travel, not the current sample: the drivetrain lags the command, so
        # braking for a hairpin has to start before the hairpin's own samples.
        anticipation = max(1, int(self._cmd_speed * 1.2 / step_len))
        ahead = (index + np.arange(anticipation + 1)) % self.n
        speed_target = float(self.v_profile[ahead].min())
        lookahead = float(np.clip(self.lookahead_gain * self._cmd_speed,
                                  self.min_lookahead, self.max_lookahead))
        steps_ahead = max(1, int(lookahead / (self.lap_length / self.n)))
        target = self.xy[(index + steps_ahead) % self.n]
        alpha = math.atan2(target[1] - y, target[0] - x) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        distance = math.hypot(target[0] - x, target[1] - y)
        delta = math.atan2(2.0 * WHEELBASE * math.sin(alpha), max(distance, 0.3))
        delta = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, delta))

        # Big heading error (rejoining after recovery, standing start facing
        # off-line): slow down until pointed roughly along the path.
        if abs(alpha) > 0.9:
            speed_target = min(speed_target, 0.5)
        elif abs(alpha) > 0.6:
            speed_target = min(speed_target, 0.8)

        # Accel-limited ramp toward the profile -- the profile is a flying
        # lap; this handles the standing start and post-recovery pickup.
        # Measured tick interval, not the nominal period. The timer only
        # achieves ~29 Hz when asked for 50 (the MPC solve plus reference
        # generation costs ~34 ms), so a nominal dt makes the acceleration
        # ramp run at a fraction of real time -- the car accelerates far
        # slower than planned and falls behind its own reference. Clamped
        # because a late tick after a recovery must not produce a huge step.
        dt = min(max(now - getattr(self, "_last_tick", now - 1.0 / self.control_hz),
                     1e-3), 0.2)
        self._last_tick = now
        self._cmd_speed += float(np.clip(speed_target - self._cmd_speed,
                                         -self.a_max * dt, self.a_max * dt))
        # Grip check against the commanded steering angle, same friction
        # circle as everywhere else in this stack.
        tan_d = abs(math.tan(delta))
        if tan_d > 1e-6:
            self._cmd_speed = min(self._cmd_speed,
                                  math.sqrt(self.a_max * WHEELBASE / tan_d))

        twist = Twist()
        twist.linear.x = self._cmd_speed
        if self._cmd_speed > 1e-3:
            twist.angular.z = (self._cmd_speed / WHEELBASE) * math.tan(delta)
        self._cmd_pub.publish(twist)

        if self.debug and now - getattr(self, "_last_debug", 0.0) > 0.5:
            self._last_debug = now
            crosstrack = float(np.linalg.norm(self.xy[index] - (x, y)))
            print(f"DBG t={now:.1f} pos=({x:.2f},{y:.2f}) idx={index} "
                  f"xtrack={crosstrack:.2f} v_cmd={self._cmd_speed:.2f} "
                  f"alpha={alpha:+.2f} delta={delta:+.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--world-name", default="cfr_speed_course")
    parser.add_argument("--pose-topic", default=None,
                        help="pose source; default is the sim ground-truth bridge")
    parser.add_argument("--pose-msg", choices=["tf", "odom"], default="tf",
                        help="tf: tf2_msgs/TFMessage (sim bridge); odom: nav_msgs/Odometry (QuestNav)")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="scale the planned speed profile (e.g. 0.7 to shake down)")
    parser.add_argument("--control-hz", type=float, default=20.0,
                        help="command publish / MPC re-solve rate. 20 Hz is what a full "
                             "tick (~34 ms) actually sustains; asking for more yields "
                             "jitter and drives worse, not faster.")
    parser.add_argument("--debug", action="store_true", help="0.5 s telemetry prints")
    parser.add_argument("--no-mpc", action="store_true",
                        help="pure pursuit only (MPC is the default tracker)")
    args = parser.parse_args()
    if args.pose_topic is None:
        args.pose_topic = f"/world/{args.world_name}/dynamic_pose/info"

    with open(args.path) as handle:
        plan = json.load(handle)

    if args.pose_msg == "tf" and not _unpause_world(args.world_name):
        raise SystemExit(f"could not start world '{args.world_name}' running")

    rclpy.init()
    node = PathRacer(plan, args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._cmd_pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
