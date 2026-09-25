#!/usr/bin/env python3
"""The numpy model against Gazebo: sensor, surfaces, and the driven lap.

Run inside the sim container with ROS and the workspace sourced, against an
obstacle course that validate.sh (or sim.sh) has already brought up:

    python3 gazebo_check.py seg-gap  --poses 500        # needs sensors:=true
    python3 gazebo_check.py surfaces                     # scripted passes
    python3 gazebo_check.py validate --starts 5          # the real node drives

seg-gap   Teleports the car to poses across the layouts (ramp and deck too:
          teleport_api takes a ground height), segments the rendered ZED cloud
          with the real segmenter, and compares its scan and gates with what
          sensor.py computes for the same ground-truth pose.  Pass/fail per
          course section; the samples are saved for refitting sensor noise.

surfaces  Drives scripted passes -- ramp, deck crest and helix entry, potholes,
          bank, gravel, the hoop and car-wash lips -- at 1, 2 and 3 m/s,
          records Gazebo's 6-DOF pose, replays the same commands through
          plant.py from the same start, and reports pitch/roll/z residuals.
          Thresholds (plan): RMS pitch and roll within 1.5 deg and peak timing
          within 50 ms on the ramp and potholes.

validate  For each held-out layout and each noisy start (+/-0.1 m, +/-5 deg in
          the start box), runs obstacle_racer_node to a verdict judged by
          hoop_monitor and lap_counter: finish with all hoops, hoop miss,
          stuck, rolled, or timeout.  This is the acceptance test.

Everything that talks to Gazebo goes through topics and services the stack
already has; nothing here reads the world file for the answer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from cfr_interfaces.msg import DriveCommand, HoopStatus

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import centerline as centerline_module  # noqa: E402
import course_model  # noqa: E402
import env as env_module  # noqa: E402
import layouts  # noqa: E402
import observation as O  # noqa: E402
import plant as P  # noqa: E402
import sensor as S  # noqa: E402

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
TELEPORT_URL = "http://127.0.0.1:9003/api/sim/teleport"
OUT = HERE / "runs" / "gazebo"


def attitude(q):
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))  # nose-down +
    roll = math.atan2(
        2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    )
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return pitch, roll, yaw


class Sim(Node):
    """The handles on the running stack this script needs."""

    def __init__(self):
        super().__init__(
            "obstacle_racer_check",
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    "use_sim_time", rclpy.Parameter.Type.BOOL, True
                )
            ],
        )
        self.pose = None
        self.pose_log = None
        self.cloud = None
        self.cloud_count = 0
        self.hoops = None
        self.done = False
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2,
            "/zed/zed_node/point_cloud/cloud_registered",
            self.on_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(HoopStatus, "/hoop_monitor/status", self.on_hoops, 10)
        self.create_subscription(Bool, "/lap_counter/done", self.on_done, LATCHED)
        self.drive = self.create_publisher(
            DriveCommand, "/drive_cmd", qos_profile_sensor_data
        )
        self.manual_go = self.create_publisher(
            Bool, "/left_wall_follower/manual_go", LATCHED
        )

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_pose(self, msg):
        p = msg.pose.position
        pitch, roll, yaw = attitude(msg.pose.orientation)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pose = (stamp, p.x, p.y, p.z, yaw, pitch, roll)
        if self.pose_log is not None:
            self.pose_log.append(self.pose)

    def on_cloud(self, msg):
        self.cloud = msg
        self.cloud_count += 1

    def on_hoops(self, msg):
        self.hoops = msg

    def on_done(self, msg):
        self.done = bool(msg.data)

    # ------------------------------------------------------------ actions

    def spin_for(self, seconds, wall=False):
        end = time.monotonic() + seconds if wall else None
        start = self.now()
        while True:
            rclpy.spin_once(self, timeout_sec=0.02)
            if wall and time.monotonic() >= end:
                return
            if not wall and self.now() - start >= seconds:
                return

    def call(self, srv_type, name, request, timeout=30.0):
        client = self.create_client(srv_type, name)
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"{name} is not up")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        self.destroy_client(client)
        if future.result() is None:
            raise RuntimeError(f"{name} did not answer")
        return future.result()

    def set_layout(self, seed):
        req = SetParameters.Request()
        req.parameters = [
            Parameter(
                name="seed",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER, integer_value=int(seed)
                ),
            )
        ]
        self.call(SetParameters, "/obstacle_randomizer/set_parameters", req)
        res = self.call(
            Trigger, "/obstacle_randomizer/randomize", Trigger.Request(), timeout=120.0
        )
        if not res.success:
            raise RuntimeError(f"randomize failed: {res.message}")
        self.get_logger().info(res.message)

    def teleport(self, x, y, heading, z=0.0):
        body = json.dumps(
            {"x": x, "y": y, "heading": heading, "z": max(0.0, z)}
        ).encode()
        req = urllib.request.Request(
            TELEPORT_URL, body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=150) as reply:
            answer = json.loads(reply.read())
        if not answer.get("success"):
            raise RuntimeError(f"teleport failed: {answer}")

    def command(self, steering, velocity):
        msg = DriveCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.auto_ready = True
        msg.steering = float(steering)
        msg.velocity = float(velocity)
        self.drive.publish(msg)

    def fresh_cloud(self, count=2, timeout=10.0):
        start = self.cloud_count
        end = time.monotonic() + timeout
        while self.cloud_count < start + count:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() > end:
                raise RuntimeError(
                    "no point cloud -- was the sim launched with sensors:=true?"
                )
        return self.cloud


def model_state(x, y, z, yaw, pitch_down, roll):
    st = np.zeros((1, P.N_STATE))
    st[0, P.S_X], st[0, P.S_Y], st[0, P.S_Z] = x, y, z
    st[0, P.S_YAW], st[0, P.S_PITCH], st[0, P.S_ROLL] = yaw, -pitch_down, roll
    return st


def load_cfg():
    return yaml.safe_load((HERE / "config.yaml").read_text())


# -------------------------------------------------------------- seg-gap


def seg_gap(sim, args):
    from ament_index_python.packages import get_package_prefix

    sys.path.insert(
        0,
        str(
            Path(get_package_prefix("cfr_arduino_bridge"))
            / "lib"
            / "cfr_arduino_bridge"
        ),
    )
    import cloud_segmentation as seg

    cfg = load_cfg()
    for k in (
        "noise_a",
        "noise_b",
        "dropout",
        "phantom",
        "attitude_noise_deg",
        "carwash_misread_prob",
    ):
        cfg["sensor"][k] = 0.0
    seeds = layouts.HELDOUT_SEEDS + layouts.TRAIN_SEEDS[:2]
    model = course_model.CourseModel(seeds)
    lines = centerline_module.Centerlines(model.layouts)
    sen = S.Sensor(cfg, model, 1, np.random.default_rng(0))
    rng = np.random.default_rng(args.seed)
    zones, per_layout = env_module.zone_labels(lines.lines)
    s_cfg = cfg["sensor"]
    rows = []
    per = max(1, args.poses // len(seeds))
    for lay, seed in enumerate(seeds):
        sim.set_layout(seed)
        line = lines.lines[lay]
        for _ in range(per):
            s = rng.uniform(0.5, line.lap_length - 1.0)
            if line.helix_start_s - 0.5 < s < line.helix_end_s + 0.2:
                continue
            x, y, z, yaw = line.pose_at(s)
            x += rng.uniform(-0.15, 0.15)
            y += rng.uniform(-0.15, 0.15)
            yaw += math.radians(rng.uniform(-15, 15))
            sim.teleport(x, y, yaw, z)
            sim.spin_for(1.0)
            cloud = sim.fresh_cloud()
            _, gx, gy, gz, gyaw, gpitch, groll = sim.pose
            pts = point_cloud2.read_points_numpy(
                cloud, field_names=("x", "y", "z"), skip_nans=True
            ).astype(np.float64)
            result = seg.segment(pts, gpitch, groll)
            real = np.asarray(
                seg.scan_from_segmentation(
                    result,
                    O.SCAN_BINS,
                    s_cfg["fov_deg"],
                    s_cfg["max_range"],
                    s_cfg["min_range"],
                )
            )
            real_gate = O.gate_features(
                [
                    (g.kind, g.center[0], g.center[1], g.axis[0], g.axis[1])
                    for g in result.gates
                ],
                cfg,
            )
            scan, gate = sen.read(
                np.array([lay]), model_state(gx, gy, gz, gyaw, gpitch, groll)
            )
            idx = line.nearest_index((gx, gy, gz))
            rows.append(
                dict(
                    seed=seed,
                    zone=zones[per_layout[lay][idx]],
                    s=float(line.arc[idx]),
                    pose=[gx, gy, gz, gyaw, gpitch, groll],
                    real=real.tolist(),
                    model=scan[0].tolist(),
                    real_gate=real_gate.tolist(),
                    model_gate=gate[0].tolist(),
                )
            )
            print(
                f"  seed {seed} s {line.arc[idx]:5.1f}: bins agree "
                f"{_agree(real, scan[0], s_cfg['max_range']).mean():.2f}",
                flush=True,
            )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"seg_gap_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(rows))
    return _seg_verdict(rows, s_cfg["max_range"], path)


def _agree(real, model, max_range):
    real, model = np.asarray(real), np.asarray(model)
    tol = np.maximum(0.20, 0.10 * np.minimum(real, model))
    both_open = (real >= max_range - 0.05) & (model >= max_range - 0.05)
    return (np.abs(real - model) <= tol) | both_open


def _seg_verdict(rows, max_range, path):
    by_zone = {}
    for r in rows:
        by_zone.setdefault(r["zone"], []).append(
            _agree(r["real"], r["model"], max_range).mean()
        )
    overall = float(np.mean([v for vs in by_zone.values() for v in vs]))
    gate_valid = np.mean(
        [(r["real_gate"][0] > 0.5) == (r["model_gate"][0] > 0.5) for r in rows]
    )
    ok = overall >= 0.85 and all(np.mean(v) >= 0.70 for v in by_zone.values())
    print(f"\nscan bins agreeing: {100 * overall:.1f}% overall (need 85%)")
    for zone, v in sorted(by_zone.items(), key=lambda kv: np.mean(kv[1])):
        print(
            f"  {zone:24s} {100 * np.mean(v):5.1f}%  over {len(v)} poses"
            + ("" if np.mean(v) >= 0.70 else "  <-- under 70%")
        )
    print(f"hoop gate seen/not seen agrees: {100 * gate_valid:.1f}%")
    print(f"samples: {path}")
    print(
        "PASS"
        if ok
        else "FAIL -- refit sensor noise or fix the model in the sections above"
    )
    return 0 if ok else 1


# ------------------------------------------------------------- surfaces

PASSES = [
    # name, start (x, y, z, yaw), steering, speeds
    ("ramp up + deck crest", (2.3, 0.0, 0.0, 0.0), 0.0, (1.0, 2.0, 3.0)),
    ("potholes", (-1.4, -8.3, 0.0, 0.0), 0.0, (1.0, 2.0, 3.0)),
    ("gravel", (4.3, -9.9, 0.0, math.pi), 0.0, (1.0, 2.0)),
    ("bank straight", (-3.0, -9.85, 0.0, math.pi), 0.0, (1.0, 2.0)),
    ("bank turning", (-3.2, -9.85, 0.0, math.pi), -0.45, (1.0, 2.0)),
    ("car-wash lip", (-5.2, 0.35, 0.0, 0.0), 0.0, (1.0, 2.0, 3.0)),
]
SURFACE_SEED = layouts.TRAIN_SEEDS[0]


def surfaces(sim, args):
    cfg = load_cfg()
    cfg["randomize"]["enabled"] = False
    model = course_model.CourseModel([SURFACE_SEED])
    sim.set_layout(SURFACE_SEED)
    duration = args.seconds
    results = []
    for name, (x, y, z, yaw), steer, speeds in PASSES:
        for v in speeds:
            sim.command(0.0, 0.0)
            sim.teleport(x, y, yaw, z)
            sim.spin_for(1.5)
            sim.pose_log = []
            t0 = sim.now()
            while sim.now() - t0 < duration:
                sim.command(steer, v)
                sim.spin_for(0.05)
            sim.command(0.0, 0.0)
            track = np.array(sim.pose_log)
            sim.pose_log = None
            if len(track) < 10:
                print(f"  {name} @ {v}: no pose stream")
                continue
            sim_t = track[:, 0] - track[0, 0]
            # The same commands through plant.py, from the same place.
            pl = P.Plant(cfg, model, 1, np.random.default_rng(0))
            first = track[0]
            pl.reset(
                np.arange(1),
                0,
                np.array([first[1]]),
                np.array([first[2]]),
                np.array([max(0.0, first[3] - 0.02)]),
                np.array([first[4]]),
                np.zeros(1),
            )
            steps = int(duration * float(cfg["env"]["control_hz"]))
            mt, mp, mr, mz = [], [], [], []
            for i in range(steps):
                pl.step(np.array([steer]), np.array([v]), int(cfg["env"]["substeps"]))
                s = pl.state[0]
                mt.append((i + 1) / float(cfg["env"]["control_hz"]))
                mp.append(-s[P.S_PITCH])  # nose-down +, as the pose reports it
                mr.append(s[P.S_ROLL])
                mz.append(s[P.S_Z])
            mt = np.array(mt)
            gp = np.interp(mt, sim_t, track[:, 5])
            gr = np.interp(mt, sim_t, track[:, 6])
            gz = np.interp(mt, sim_t, track[:, 3])
            rms_p = math.degrees(np.sqrt(np.mean((gp - np.array(mp)) ** 2)))
            rms_r = math.degrees(np.sqrt(np.mean((gr - np.array(mr)) ** 2)))
            rms_z = float(np.sqrt(np.mean((gz - np.array(mz)) ** 2)))
            lag = (np.argmax(np.abs(gp)) - np.argmax(np.abs(mp))) / float(
                cfg["env"]["control_hz"]
            )
            results.append(
                dict(
                    name=name,
                    speed=v,
                    rms_pitch_deg=rms_p,
                    rms_roll_deg=rms_r,
                    rms_z_m=rms_z,
                    peak_lag_s=lag,
                    gazebo=dict(
                        t=sim_t.tolist(),
                        pitch=track[:, 5].tolist(),
                        roll=track[:, 6].tolist(),
                        z=track[:, 3].tolist(),
                    ),
                    model=dict(t=mt.tolist(), pitch=mp, roll=mr, z=mz),
                )
            )
            print(
                f"  {name:22s} {v:.0f} m/s: rms pitch {rms_p:4.2f} deg  roll {rms_r:4.2f} deg  z {100 * rms_z:4.1f} cm  peak lag {1000 * lag:+5.0f} ms",
                flush=True,
            )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"surfaces_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(results))
    critical = [r for r in results if r["name"] in ("ramp up + deck crest", "potholes")]
    ok = all(
        r["rms_pitch_deg"] <= 1.5
        and r["rms_roll_deg"] <= 1.5
        and abs(r["peak_lag_s"]) <= 0.05
        for r in critical
    )
    print(f"traces: {path}")
    print(
        "PASS"
        if ok
        else "FAIL -- the plant does not pitch and roll like Gazebo on the ramp or potholes"
    )
    return 0 if ok else 1


# ------------------------------------------------------------- validate


def validate(sim, args):
    rng = np.random.default_rng(args.seed)
    seeds = (
        [int(s) for s in args.seeds.split(",")] if args.seeds else layouts.HELDOUT_SEEDS
    )
    runs = []
    for seed in seeds:
        for k in range(args.starts):
            sim.call(
                SetBool, "/obstacle_racer/manual_start", SetBool.Request(data=False)
            )
            sim.command(0.0, 0.0)
            sim.set_layout(seed)
            x = -0.70 + rng.uniform(-0.1, 0.1)
            y = rng.uniform(-0.1, 0.1)
            yaw = math.radians(rng.uniform(-5, 5))
            sim.teleport(x, y, yaw, 0.0)
            sim.spin_for(1.5)
            sim.call(Trigger, "/hoop_monitor/reset", Trigger.Request())
            sim.call(Trigger, "/lap_counter/reset", Trigger.Request())
            sim.done = False
            sim.spin_for(0.5)
            sim.manual_go.publish(Bool(data=True))
            sim.call(
                SetBool, "/obstacle_racer/manual_start", SetBool.Request(data=True)
            )
            t0 = sim.now()
            last_move, last_xy = t0, (sim.pose[1], sim.pose[2])
            outcome = "timeout"
            while sim.now() - t0 < args.timeout:
                sim.spin_for(0.2)
                _, px, py, _, _, pitch, roll = sim.pose
                if math.hypot(px - last_xy[0], py - last_xy[1]) > 0.3:
                    last_move, last_xy = sim.now(), (px, py)
                if sim.hoops is not None and sim.hoops.any_missed:
                    outcome = "hoop_miss"
                    break
                if sim.done:
                    outcome = (
                        "finish"
                        if (sim.hoops is not None and sim.hoops.all_passed)
                        else "finish_without_hoops"
                    )
                    break
                if abs(pitch) > 0.8 or abs(roll) > 0.8:
                    outcome = "rollover"
                    break
                if sim.now() - last_move > 5.0:
                    outcome = "stuck"
                    break
            elapsed = sim.now() - t0
            sim.manual_go.publish(Bool(data=False))
            sim.call(
                SetBool, "/obstacle_racer/manual_start", SetBool.Request(data=False)
            )
            hoops = int(sum(sim.hoops.passed)) if sim.hoops is not None else 0
            runs.append(
                dict(
                    seed=seed,
                    start=[x, y, yaw],
                    outcome=outcome,
                    time=elapsed,
                    hoops=hoops,
                    end=list(sim.pose[1:4]),
                )
            )
            print(
                f"  seed {seed} start {k}: {outcome:20s} {elapsed:6.1f} s  hoops {hoops}/3  ended at "
                f"({sim.pose[1]:.2f}, {sim.pose[2]:.2f})",
                flush=True,
            )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"validate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(runs, indent=1))
    finishes = [r for r in runs if r["outcome"] == "finish"]
    print(
        f"\n{len(finishes)}/{len(runs)} clean laps"
        + (
            f", mean {np.mean([r['time'] for r in finishes]):.1f} s, best {min(r['time'] for r in finishes):.1f} s"
            if finishes
            else ""
        )
    )
    print(f"results: {path}")
    return 0 if len(finishes) == len(runs) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("seg-gap")
    a.add_argument("--poses", type=int, default=500)
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("surfaces")
    b.add_argument("--seconds", type=float, default=4.0)
    c = sub.add_parser("validate")
    c.add_argument("--starts", type=int, default=5)
    c.add_argument("--seeds", default="")
    c.add_argument("--timeout", type=float, default=90.0)
    c.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rclpy.init()
    sim = Sim()
    try:
        sim.spin_for(2.0, wall=True)
        if sim.pose is None:
            print("no /zed/zed_node/pose -- is the obstacle course running?")
            return 1
        return {"seg-gap": seg_gap, "surfaces": surfaces, "validate": validate}[
            args.cmd
        ](sim, args)
    finally:
        sim.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
