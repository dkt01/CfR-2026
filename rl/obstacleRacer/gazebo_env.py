#!/usr/bin/env python3
"""One Gazebo car as a training environment: ObstacleEnv with Gazebo physics.

Runs inside a sim container, next to a running Obstacle Course (sensors on),
and serves one car to a trainer on the host over stdin/stdout
(`gazebo_vec.py` starts it).  Everything the reward and the episode rules
need is ObstacleEnv's own code, unchanged: this file swaps only

  - the plant: each control step publishes the DriveCommand, runs the paused
    world for exactly one control period (ControlWorld multi_step) and reads
    the car back from Gazebo's pose.  sim_vehicle_node does the dead time,
    slew, reverse wait and contact stall, as it does for validate.sh;
  - the sensor: the rendered ZED cloud, through zed_cloud_noise and the
    compiled segmenter, as obstacle_racer_node segments it on the car.  Each
    run draws the camera latency from sensor.latency_s, as training does, and
    the policy sees the newest frame captured at least that long ago;
  - the observation's speed and yaw rate: the sim Arduino's signed
    tachometer and the node's own filtered yaw rate from the pose, with none
    of the numpy env's added noise (Gazebo is its own noise).

Contact, the crash rule and rollovers are judged from the Gazebo pose: the
car touched when the numpy course model puts its body outline within
TOUCH_M of an obstacle, and the speed it lost over that step is `lost`.

Starts are ObstacleEnv's own deal (start box, before recent failures, before
obstacles, stuck poses), from rest: a teleport cannot set a speed.  The
layout changes every `layout_every` episodes; the randomizer takes seconds.

    python3 gazebo_env.py serve      # the host talks to it over stdin/stdout
    python3 gazebo_env.py smoke      # a few episodes of the prior, printed
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
import yaml

# Numba's cache sits beside the sources, on a checkout the Windows trainer
# shares: keep the container's compiled code out of it.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import course_model  # noqa: E402
import gazebo_check  # noqa: E402
import layouts  # noqa: E402
import observation as O  # noqa: E402
import plant as P  # noqa: E402
from env import CLEARANCE_RINGS, ObstacleEnv  # noqa: E402
from gazebo_wire import receive, send  # noqa: E402
from obstacle_racer_node import _segmenter  # noqa: E402

WORLD = "/world/cfr_obstacle_course"
PHYSICS_DT = 0.001  # obstacle_course.sdf max_step_size
# Body outline this near an obstacle in the numpy grid is a touch.  Gazebo's
# wheels stand 1.25 cm proud of the chassis: at 1 cm, v9 stopped against the
# helix exit wall and at the tunnel's south end ended "stall", never "pinned".
TOUCH_M = 0.03
ROLLED_RAD = 0.8  # gazebo_check's rollover verdict


class GazeboLink(gazebo_check.Sim):
    """gazebo_check's handles, plus world stepping and segmented frames."""

    def __init__(self, cfg):
        super().__init__(need_cloud=False)
        from ros_gz_interfaces.srv import ControlWorld

        self.ControlWorld = ControlWorld
        self.cfg = cfg
        self.seg = _segmenter()
        self.frames = collections.deque(maxlen=24)  # (capture stamp, scan, gate)
        self.yaw_rate = 0.0
        self._prev_yaw = None
        self.create_subscription(
            PointCloud2,
            "/zed/zed_node/point_cloud/cloud_registered",
            self.on_frame,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        # gazebo_check reads the pose best effort.  Stepping paused, a dropped
        # pose is never followed by another, so also read it reliably (a
        # repeated stamp changes nothing in on_pose).
        self.create_subscription(
            PoseStamped,
            "/zed/zed_node/pose",
            self.on_pose,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.control_client = self.create_client(ControlWorld, f"{WORLD}/control")
        if not self.control_client.wait_for_service(timeout_sec=60.0):
            raise RuntimeError(f"{WORLD}/control is not bridged")

    def on_pose(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.pose is not None and stamp <= self.pose[0]:
            return
        super().on_pose(msg)
        # obstacle_racer_node's yaw rate: over the stamps the poses span.
        stamp, yaw = self.pose[0], self.pose[4]
        if self._prev_yaw is not None and stamp > self._prev_yaw[0]:
            dyaw = math.remainder(yaw - self._prev_yaw[1], math.tau)
            rate = max(-8.0, min(8.0, dyaw / (stamp - self._prev_yaw[0])))
            self.yaw_rate = 0.5 * self.yaw_rate + 0.5 * rate
        self._prev_yaw = (stamp, yaw)

    def on_frame(self, msg):
        if self.pose is None:
            return
        pts = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        pts = np.asarray(pts, dtype=np.float64)[::2]  # the node's cloud_stride
        _, _, _, _, _, pitch, roll = self.pose
        seg = self.seg.segment(pts, pitch, roll)
        s = self.cfg["sensor"]
        scan = self.seg.scan_from_segmentation(
            seg,
            O.SCAN_BINS,
            float(s["fov_deg"]),
            float(s["max_range"]),
            float(s["min_range"]),
        )
        gates = [
            (g.kind, g.center[0], g.center[1], g.axis[0], g.axis[1]) for g in seg.gates
        ]
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.frames.append(
            (stamp, np.asarray(scan, np.float64), O.gate_features(gates, self.cfg))
        )

    def control(self, pause, steps=0):
        req = self.ControlWorld.Request()
        req.world_control.pause = bool(pause)
        if steps:
            req.world_control.multi_step = int(steps)
        future = self.control_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is None or not future.result().success:
            raise RuntimeError("ControlWorld failed")

    def wait(self, ready, timeout, what):
        end = time.monotonic() + timeout
        while not ready():
            rclpy.spin_once(self, timeout_sec=0.005)
            if time.monotonic() > end:
                raise TimeoutError(what)

    def advance(self, dt, frame_by):
        """Run the paused world for dt of sim time; wait for pose and frames.

        `frame_by`: the capture time the newest frame the policy may see is
        due by, so the step waits until the segmenter has delivered it.
        """
        t_end = self.now() + dt
        self.control(True, round(dt / PHYSICS_DT))
        self.wait(lambda: self.now() >= t_end - 1e-4, 60.0, "clock")
        fresh = lambda: self.pose is not None and self.pose[0] >= t_end - 0.04  # noqa: E731
        try:
            self.wait(fresh, 3.0, "pose")
        except TimeoutError:
            # Still missing: run a few more ms for the next one.
            self.late_poses += 1
            for _ in range(20):
                self.control(True, 5)
                try:
                    self.wait(fresh, 2.0, "pose")
                    break
                except TimeoutError:
                    continue
            else:
                raise
        # A frame is rendered every 1/camera_hz; the one due by frame_by must
        # have come through zed_cloud_noise and the segmenter.
        period = 1.0 / float(self.cfg["sensor"]["camera_hz"])
        try:
            self.wait(
                lambda: self.frames and self.frames[-1][0] >= frame_by - period + 0.002,
                10.0,
                "frame",
            )
        except TimeoutError:
            self.late_frames += 1

    late_frames = 0
    late_poses = 0


class GazeboSensor:
    """The segmented ZED frames, delayed by the run's latency."""

    def __init__(self, link, cfg, rng):
        self.link = link
        self.rng = rng
        self.latency = (
            float(cfg["sensor"]["latency_s"][0]),
            float(cfg["sensor"]["latency_s"][1]),
        )
        self.lat = self.latency[0]
        self.max_range = float(cfg["sensor"]["max_range"])

    def new_run(self):
        self.lat = self.rng.uniform(*self.latency)

    def read(self, lay, state, idx=None, draw_for_all=False):
        cutoff = self.link.now() - self.lat
        pick = None
        for frame in self.link.frames:
            if frame[0] <= cutoff + 1e-6:
                pick = frame
        if pick is None:
            pick = self.link.frames[0] if self.link.frames else None
        if pick is None:
            return np.full((1, O.SCAN_BINS), self.max_range), np.zeros((1, O.GATE_DIM))
        return pick[1][None, :].copy(), pick[2][None, :].copy()


class GazeboPlant(P.Plant):
    """The numpy plant's tables and state layout, with Gazebo moving the car."""

    def __init__(self, cfg, model, rng, link, sensor):
        super().__init__(cfg, model, 1, rng)
        self.link = link
        self.sensor = sensor
        self.seeds = list(model.seeds)
        self.current_seed = None
        self.dt_control = 1.0 / float(cfg["env"]["control_hz"])
        self.skipped = 0

    def _read(self):
        stamp, x, y, z, yaw, pitch_down, roll = self.link.pose
        st = self.state[0]
        prev = (st[P.S_X], st[P.S_Y], self.stamp)
        self.stamp = stamp
        st[P.S_X], st[P.S_Y], st[P.S_Z] = x, y, z
        st[P.S_YAW], st[P.S_PITCH], st[P.S_ROLL] = yaw, -pitch_down, roll
        st[P.S_R] = self.link.yaw_rate
        tach = float(self.link.status.speed) if self.link.status is not None else 0.0
        st[P.S_DIR] = -1.0 if tach < 0 else 1.0
        return prev

    def reset(self, idx, lay, x, y, z, yaw, speed):
        super().reset(idx, lay, x, y, z, yaw, np.zeros(len(idx)))
        # ObstacleEnv re-deals a start that lands inside something, judged
        # on this state: say so before Gazebo is asked to put the car there
        # (a teleport into a post never settles).  After three tries it
        # takes the start anyway, as ObstacleEnv does.
        touching = P.body_contact(
            self.OBS, self.lay[idx], self.state[idx], P.CONTACT_SPACING
        )
        if touching.any() and self.skipped < 3:
            self.skipped += 1
            return
        self.skipped = 0
        link = self.link
        seed = self.seeds[int(lay[0])]
        link.control(False)
        link.command(0.0, 0.0)
        if seed != self.current_seed:
            link.set_layout(seed)
            self.current_seed = seed
        placed = False
        for ground in (max(0.0, float(z[0])), max(0.0, float(z[0])) + 0.15):
            if link.place(float(x[0]), float(y[0]), float(yaw[0]), ground, tries=2):
                placed = True
                break
        if not placed:
            # The episode's bookkeeping assumes the car is where it was
            # dealt; the host restarts this container instead.
            raise RuntimeError(f"could not place the car at ({x[0]:.2f}, {y[0]:.2f})")
        link.control(True)
        link.frames.clear()
        self.sensor.new_run()
        # A frame from the new pose, old enough for this run's latency.
        t0 = link.now()
        for _ in range(40):
            link.command(0.0, 0.0)
            link.advance(self.dt_control, t0)
            if (
                link.frames
                and link.frames[0][0] >= t0
                and link.now() - link.frames[0][0] >= self.sensor.lat
            ):
                break
        self._read()
        self.state[0, P.S_V] = 0.0
        self.state[0, P.S_TARGET] = 0.0

    def push_command(self, steer, speed):
        self.last_command[:, 0] = steer
        self.last_command[:, 1] = speed
        return steer, speed

    def step(self, steer, speed, substeps):
        link = self.link
        v_before = abs(float(self.state[0, P.S_V]))
        link.command(float(steer[0]), float(speed[0]))
        link.advance(self.dt_control, link.now() + self.dt_control - self.sensor.lat)
        px, py, pt = self._read()
        st = self.state[0]
        dx, dy = st[P.S_X] - px, st[P.S_Y] - py
        # Over the poses' own stamps: the pose due by a step's end lands up
        # to 40 ms early, so dividing by the control period read speeds that
        # halved and doubled from step to step, and each halving next to a
        # wall judged a crash (v11's helix "crashes" rolling down the ramp).
        span = self.stamp - pt
        ground = math.hypot(dx, dy) / span if span > 1e-3 else abs(float(st[P.S_V]))
        forward = dx * math.cos(st[P.S_YAW]) + dy * math.sin(st[P.S_YAW])
        st[P.S_V] = ground if forward >= 0 else -ground
        st[P.S_TARGET] = float(speed[0])
        clearance = P.body_clearance(self.OBS, self.lay, self.state, CLEARANCE_RINGS)
        touched = clearance <= TOUCH_M
        lost = np.where(touched, max(0.0, v_before - ground), 0.0)
        rolled = np.array(
            [abs(st[P.S_PITCH]) > ROLLED_RAD or abs(st[P.S_ROLL]) > ROLLED_RAD]
        )
        if self.log is not None:
            # Before ObstacleEnv's auto-reset moves the car: the trace
            # subcommand's record of each step, the terminal one included.
            self.log.append(
                dict(
                    now=link.now(),
                    stamp=float(link.pose[0]),
                    pose=[
                        float(v)
                        for v in st[[P.S_X, P.S_Y, P.S_Z, P.S_YAW, P.S_PITCH, P.S_ROLL]]
                    ],
                    v=float(st[P.S_V]),
                    tach=float(link.status.speed) if link.status is not None else 0.0,
                    yaw_rate=float(st[P.S_R]),
                    cmd=[float(steer[0]), float(speed[0])],
                    clear=float(clearance[0]),
                    lost=float(lost[0]),
                )
            )
        return touched, lost, rolled

    log = None
    stamp = 0.0


class GazeboEnv(ObstacleEnv):
    """ObstacleEnv, one car, in Gazebo.  See the module docstring."""

    def __init__(self, cfg, model, layout_ids, link, seed=0, layout_every=4):
        g = cfg.get("gazebo", {})
        cfg = dict(cfg)
        # Teleports set a pose, never a speed.
        cfg["env"] = dict(cfg["env"], dealt_speed=[0.0, 0.0])
        if "start_box_prob" in g:
            cfg["env"]["start_box_prob"] = float(g["start_box_prob"])
        super().__init__(cfg, model, 1, layout_ids, seed=seed)
        self.link = link
        self.sensor = GazeboSensor(link, cfg, self.rng)
        self.plant = GazeboPlant(cfg, model, self.rng, link, self.sensor)
        self.layout_every = int(g.get("layout_every", layout_every))
        self.episodes = 0
        self.lay_now = None

    def _pick_layouts(self, k):
        if self.lay_now is None or self.episodes % self.layout_every == 0:
            self.lay_now = int(self.layout_ids[self.rng.integers(len(self.layout_ids))])
        self.episodes += 1
        return np.full(k, self.lay_now, np.int64)

    def _dither(self, speed_cmd):
        return np.zeros(self.n)  # the sim Arduino's own, if any

    def _capture_rows(self):
        return np.arange(self.n)

    def _camera(self, scan, gate, capture_rows):
        return scan, gate  # GazeboSensor already delivers the delayed frame

    def _frame(self, scan, gate, idx=None, fresh=None):
        if idx is None:
            idx = np.arange(self.n)
        st = self.plant.state[idx]
        status = self.link.status
        speed = O.tach(np.array([float(status.speed) if status is not None else 0.0]))
        yaw_rate = st[:, P.S_R].copy()
        raw = O.prior_steer(scan, gate, yaw_rate, self.cfg)
        first = fresh[idx] if fresh is not None else None
        prior = O.smooth_prior(self.prior[idx], raw, first, self.cfg)
        self.prior[idx] = prior
        memory = self.memory.update(speed, idx, first)
        heading = self.heading.update(yaw_rate, idx)
        return O.frame(
            scan,
            gate,
            speed,
            yaw_rate,
            self.prev_action[idx],
            prior,
            self.cfg,
            memory,
            heading,
        )

    def _reset_idx(self, idx, attempt=0, lay=None):
        super()._reset_idx(idx, attempt, lay)
        # No heading error from placement: Gazebo put the car exactly there.
        self.heading.reset(idx, self.plant.state[idx, P.S_YAW])
        self.heading_bias[idx] = 0.0


def make_env(cfg, seeds, seed):
    model = course_model.CourseModel(seeds)
    link = GazeboLink(cfg)
    deadline = time.monotonic() + 60.0
    while link.pose is None and time.monotonic() < deadline:
        rclpy.spin_once(link, timeout_sec=0.1)
    if link.pose is None:
        raise RuntimeError("no /zed/zed_node/pose -- is the course running?")
    # Paused from here on except while placing the car: a free-running
    # world renders the ZED and runs physics on cores the trainer needs.
    link.control(True)
    return GazeboEnv(cfg, model, np.arange(len(seeds)), link, seed=seed)


def serve():
    wire_in, wire_out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # anything printed must not corrupt the wire
    hello = receive(wire_in)
    cfg, seeds = hello["cfg"], hello["seeds"]
    rclpy.init()
    env = make_env(cfg, seeds, int(hello.get("seed", 0)))
    send(wire_out, dict(obs_dim=env.obs_dim, act_dim=env.act_dim))
    while True:
        try:
            msg = receive(wire_in)
        except EOFError:
            break
        kind = msg["kind"]
        try:
            if kind == "reset":
                send(wire_out, env.reset())
            elif kind == "step":
                t0 = time.monotonic()
                obs, rew, term, trunc, info = env.step(msg["action"])
                info[0]["_wall"] = time.monotonic() - t0
                info[0]["_late"] = env.link.late_frames
                info[0]["_late_pose"] = env.link.late_poses
                send(wire_out, (obs, rew, term, trunc, info))
            elif kind == "close":
                break
        except Exception:  # noqa: BLE001 - the host restarts the container
            error = traceback.format_exc()
            print(error, flush=True)
            send(wire_out, dict(error=error))
            return
    env.link.control(False)
    rclpy.try_shutdown()


def smoke(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.box:
        cfg.setdefault("gazebo", {})["start_box_prob"] = 1.0
    rclpy.init()
    seeds = [int(s) for s in args.seeds.split(",")]
    env = make_env(cfg, seeds, 0)
    policy = None
    if args.policy:
        from policy import NumpyPolicy

        policy = NumpyPolicy.load(args.policy)
        policy.reset(1)
    obs = env.reset()
    t0, steps, episodes = time.monotonic(), 0, 0
    while episodes < args.episodes:
        if policy is not None:
            action = policy.act(obs)
        else:
            action = np.array([[0.0, float(O.speed_to_action(1.0, cfg))]])
        obs, rew, term, trunc, info = env.step(action)
        steps += 1
        if steps % 100 == 0:
            st = env.plant.state[0]
            print(
                f"  t {env.t[0]:6.1f}  ({st[P.S_X]:6.2f}, {st[P.S_Y]:6.2f})  v {st[P.S_V]:5.2f}  "
                f"s {env.s[0]:5.1f} {env.zone_names[env.zone_of[env.plant.lay[0], env.idx[0]]]:>14s}  {steps / (time.monotonic() - t0):5.1f} steps/s wall, "
                f"late frames {env.link.late_frames}",
                flush=True,
            )
        if term[0] or trunc[0]:
            episodes += 1
            i = info[0]
            print(
                f"episode {episodes}: {i['outcome']} at {i['zone']} after {i['time']:.1f} s, "
                f"{i['dist']:.1f} m, start {i['start']}, return {i['episode']['r']:.1f}",
                flush=True,
            )
            if policy is not None:
                policy.reset(1)
    env.link.control(False)
    rclpy.try_shutdown()


def trace(args):
    """Drive a policy from a fixed arc length before the helix; save every step.

    For comparing with the numpy plant (helix_parity.py): the Gazebo pose,
    speed, command and numpy-grid clearance per control step, and how each
    run ended.
    """
    from policy import NumpyPolicy

    cfg = yaml.safe_load(Path(args.config).read_text())
    rclpy.init()
    seeds = [int(s) for s in args.seeds.split(",")]
    env = make_env(cfg, seeds, args.seed)
    policy = NumpyPolicy.load(args.policy)
    runs = []
    for k, seed in enumerate(seeds):
        line = env.lines.lines[k]
        for rep in range(args.runs):
            env.forced_lay = np.array([k])
            env.forced_s = np.array([line.helix_start_s - args.before])
            env.plant.log = None
            obs = env.reset()
            policy.reset(1)
            env.plant.log = rows = []
            info = [{}]
            while env.t[0] < args.seconds:
                obs, _, term, trunc, info = env.step(policy.act(obs))
                if term[0] or trunc[0]:
                    break
            env.plant.log = None
            end = info[0] if "outcome" in info[0] else {}
            runs.append(
                dict(
                    seed=seed,
                    rep=rep,
                    outcome=end.get("outcome", "cut"),
                    zone=end.get("zone"),
                    dist=end.get("dist"),
                    helix=[line.helix_start_s, line.helix_end_s],
                    rows=rows,
                )
            )
            print(
                f"seed {seed} run {rep}: {runs[-1]['outcome']} at {runs[-1]['zone']} "
                f"after {len(rows) * env.dt:.1f} s, late poses {env.link.late_poses}",
                flush=True,
            )
            Path(args.out).write_text(json.dumps(dict(before=args.before, runs=runs)))
    env.link.control(False)
    rclpy.try_shutdown()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    t = sub.add_parser("trace")
    t.add_argument("--config", default=str(HERE / "config.yaml"))
    t.add_argument("--policy", required=True)
    t.add_argument("--seeds", default="201,202,208,218")
    t.add_argument("--runs", type=int, default=3, help="per seed")
    t.add_argument("--before", type=float, default=2.0, help="m before the helix")
    t.add_argument("--seconds", type=float, default=15.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--config", default=str(HERE / "config.yaml"))
    s.add_argument("--seeds", default=",".join(map(str, layouts.TRAIN_SEEDS[:4])))
    s.add_argument("--episodes", type=int, default=2)
    s.add_argument("--policy", default="")
    s.add_argument("--box", action="store_true", help="start-box starts only")
    args = ap.parse_args()
    if args.cmd == "serve":
        serve()
    elif args.cmd == "trace":
        trace(args)
    else:
        smoke(args)


if __name__ == "__main__":
    main()
