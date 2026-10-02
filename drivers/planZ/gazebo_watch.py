#!/usr/bin/env python3
"""Start Plan Z in a running Gazebo course and say how the run went.

    python3 gazebo_watch.py --course speed --timeout 400 --out /tmp/run.json

Run by validate.sh once the course and the driver are up.  It waits for the
pose and the cloud, releases the car through the driver's manual start, and
watches from outside the driver until lap_counter reports the laps done (and
the car has rolled to a stop), the car is on its side, it has not moved for
`--stuck` seconds, or the time is up.

What it reports is judged on Gazebo's own pose, not on anything the driver
believes: laps and lap times from /lap_counter/count, the body's clearance to
the nearest bale (the box collisions in the course's SDF, so a touch is a
clearance of zero or less), and the driver's reversals.

On the Obstacle Course the bales are judged in plan only, which the ramp and
the deck over the tunnel make meaningless (the car drives over bales there),
so contact is reported but does not fail a run; the run is judged on its
laps and its hoops.  The hoops are judged here, lap by lap, with
hoop_monitor.py's own crossing test on the randomizer's hoop positions and
the layout file's yaws; /hoop_monitor/status, which resolves each hoop once
for the whole run, is reported beside it.

Exit status 0 only for a clean finish: every lap, every hoop, and on the
Speed Course no contact.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from cfr_interfaces.msg import ArduinoStatus, DriverTelemetry, LapCount

try:
    from cfr_interfaces.msg import HoopLayout, HoopStatus
except ImportError:  # pragma: no cover
    HoopLayout = HoopStatus = None

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] / "jetson" / "cfr_arduino_bridge"
WORLDS = PACKAGE / "worlds"
sys.path.insert(0, str(PACKAGE / "src"))
import hoop_monitor  # noqa: E402

NODE = {"obstacle": "obstacle_racer", "speed": "formula_one"}
HALF_LENGTH, HALF_WIDTH = 0.275, 0.1625
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def bales(course):
    """(N, 5): x, y, yaw, half length, half width of every bale collision box."""
    text = (WORLDS / f"{course}_course.sdf").read_text()
    rows = []
    for match in re.finditer(
        r'<collision name="[^"]*bale[^"]*">\s*<pose>([^<]+)</pose>\s*<geometry>\s*<box>\s*<size>([^<]+)</size>',
        text,
    ):
        pose = [float(v) for v in match.group(1).split()]
        size = [float(v) for v in match.group(2).split()]
        rows.append((pose[0], pose[1], pose[5], size[0] / 2, size[1] / 2))
    return np.asarray(rows)


def clearance(boxes, x, y, yaw):
    """m from the body's outline to the nearest bale; <= 0 is contact."""
    if not len(boxes):
        return float("inf")
    near = boxes[np.hypot(boxes[:, 0] - x, boxes[:, 1] - y) < 1.6]
    if not len(near):
        return 1.0
    # The body's outline, sampled, in the world.
    edge = np.linspace(-1.0, 1.0, 9)
    outline = np.concatenate(
        [
            np.c_[HALF_LENGTH * edge, np.full(9, HALF_WIDTH)],
            np.c_[HALF_LENGTH * edge, np.full(9, -HALF_WIDTH)],
            np.c_[np.full(9, HALF_LENGTH), HALF_WIDTH * edge],
            np.c_[np.full(9, -HALF_LENGTH), HALF_WIDTH * edge],
        ]
    )
    c, s = math.cos(yaw), math.sin(yaw)
    wx = x + c * outline[:, 0] - s * outline[:, 1]
    wy = y + s * outline[:, 0] + c * outline[:, 1]
    best = float("inf")
    for bx, by, byaw, hl, hw in near:
        bc, bs = math.cos(byaw), math.sin(byaw)
        lx = bc * (wx - bx) + bs * (wy - by)
        ly = -bs * (wx - bx) + bc * (wy - by)
        out = np.hypot(
            np.maximum(np.abs(lx) - hl, 0.0), np.maximum(np.abs(ly) - hw, 0.0)
        )
        inside = np.minimum(hl - np.abs(lx), hw - np.abs(ly))
        best = min(best, float(np.where(out > 0, out, -inside).min()))
    return best


class Watch(Node):
    def __init__(self, course):
        super().__init__("plan_z_watch")
        self.pose = None
        self.released = False
        self.cloud = False
        self.count = None
        self.done = False
        self.status = None
        self.driver = None
        self.hoops = None
        name = NODE[course]
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2,
            "/zed/zed_node/point_cloud/cloud_registered",
            lambda _: setattr(self, "cloud", True),
            qos_profile_sensor_data,
        )
        self.create_subscription(LapCount, "/lap_counter/count", self.on_count, 10)
        self.create_subscription(Bool, "/lap_counter/done", self.on_done, LATCHED)
        self.create_subscription(
            ArduinoStatus, "/arduino_bridge/status", self.on_status, 10
        )
        self.create_subscription(
            DriverTelemetry, f"/{name}/telemetry", self.on_driver, 50
        )
        self.judge = None
        if HoopStatus is not None:
            self.create_subscription(
                HoopStatus, "/hoop_monitor/status", self.on_hoops, 10
            )
        if HoopLayout is not None and course == "obstacle":
            layout = yaml.safe_load(
                (PACKAGE / "config" / "obstacle_course_layout.yaml").read_text()
            )
            self.hoop_yaw = {
                name: float(spec["yaw"])
                for name, spec in layout["obstacle_randomizer"]["ros__parameters"][
                    "hoops"
                ].items()
                if name != "names"
            }
            self.create_subscription(
                HoopLayout, "/obstacle_randomizer/hoop_layout", self.on_layout, LATCHED
            )
        self.start = self.create_client(SetBool, f"/{name}/manual_start")
        self.manual_go = self.create_publisher(
            Bool, "/left_wall_follower/manual_go", LATCHED
        )

    def on_layout(self, msg):
        """Where the randomizer has stood the hoops: judged afresh from here."""
        self.judge = hoop_monitor.HoopMonitor(
            [
                hoop_monitor.Hoop(name, p.x, p.y, self.hoop_yaw[name])
                for name, p in zip(msg.names, msg.positions)
            ]
        )

    def on_pose(self, msg):
        q = msg.pose.orientation
        self.pose = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            msg.pose.position.x,
            msg.pose.position.y,
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
            math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))),
            math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)),
        )
        if self.judge is not None and self.released:
            self.judge.update(
                hoop_monitor.Pose2D(self.pose[1], self.pose[2], self.pose[3])
            )

    def on_count(self, msg):
        self.count = msg

    def on_done(self, msg):
        self.done = self.done or bool(msg.data)

    def on_status(self, msg):
        self.status = msg

    def on_driver(self, msg):
        self.driver = msg

    def on_hoops(self, msg):
        self.hoops = msg

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def sim_now(self):
        return self.pose[0] if self.pose else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--course", choices=("obstacle", "speed"), required=True)
    ap.add_argument("--timeout", type=float, default=400.0, help="s of sim time")
    ap.add_argument(
        "--stuck", type=float, default=25.0, help="s of sim time without moving"
    )
    ap.add_argument("--wall-timeout", type=float, default=3600.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rclpy.init()
    node = Watch(args.course)
    boxes = bales(args.course)
    print(
        f"waiting for the pose and the cloud ({len(boxes)} bales in the course)...",
        flush=True,
    )
    wall0 = time.time()
    while (node.pose is None or not node.cloud) and time.time() - wall0 < 300:
        node.spin_for(0.5)
    if node.pose is None or not node.cloud:
        print("FAIL -- no pose or no cloud: is the course up with sensors:=true?")
        return 2
    if not node.start.wait_for_service(timeout_sec=120.0):
        print("FAIL -- the driver's manual_start service never came up")
        return 2
    node.spin_for(3.0)
    node.manual_go.publish(Bool(data=True))
    node.start.call_async(SetBool.Request(data=True))
    node.released = True
    t0 = node.sim_now()
    print("released", flush=True)

    trace, laps, lap_times, lap_hoops = [], 0, [], []
    closest, contacts, reversing = float("inf"), 0, 0
    last_move, last_xy = t0, (node.pose[1], node.pose[2])
    lap_start = t0
    outcome = "timeout"
    stopped_since = None
    while (
        node.sim_now() - t0 < args.timeout and time.time() - wall0 < args.wall_timeout
    ):
        node.spin_for(0.2)
        t, x, y, yaw, pitch, roll = node.pose
        gap = clearance(boxes, x, y, yaw)
        closest = min(closest, gap)
        contacts += gap <= 0.0
        cmd = node.driver.velocity_cmd if node.driver else 0.0
        reversing += cmd < 0.0
        d = node.driver
        trace.append(
            (
                round(t - t0, 2),
                round(x, 3),
                round(y, 3),
                round(yaw, 3),
                round(gap, 3),
                round(float(node.status.speed), 2) if node.status else None,
                round(cmd, 2),
                round(d.steer_cmd, 3) if d else None,
                round(d.cross_track, 3) if d else None,
                round(d.heading_error, 3) if d else None,
                round(d.steer_ff, 3) if d else None,
                round(d.action[0], 3) if d and len(d.action) else None,
                round(d.yaw_rate, 3) if d else None,
                [round(v, 2) for v in d.observation]
                if d and len(trace) % 3 == 0
                else None,
            )
        )
        if node.count is not None and node.count.laps > laps:
            laps = int(node.count.laps)
            lap_times.append(round(t - lap_start, 2))
            lap_start = t
            if node.judge is not None:
                # Each lap has to thread them all: judged lap by lap.
                lap_hoops.append(
                    sorted(
                        name for name, state in node.judge.state.items() if state.passed
                    )
                )
                node.judge.reset()
            print(
                f"  lap {laps}: {lap_times[-1]:.1f} s   closest so far {closest:.3f} m",
                flush=True,
            )
        if math.hypot(x - last_xy[0], y - last_xy[1]) > 0.3:
            last_move, last_xy = t, (x, y)
        if abs(pitch) > 0.8 or abs(roll) > 0.8:
            outcome = "rollover"
            break
        if node.done:
            # Run on until it has coasted to rest: no brakes.
            if stopped_since is None and t - last_move > 1.0:
                stopped_since = t
            if stopped_since is not None:
                outcome = "finish"
                break
        elif t - last_move > args.stuck:
            outcome = "stuck"
            break
    hoops, missed = None, []
    clean = outcome == "finish"
    if args.course == "obstacle":
        # Laps and hoops: how many each lap threaded, and which it did not.
        if node.judge is not None:
            names = sorted(node.judge.state)
            if not lap_hoops:
                lap_hoops.append(
                    sorted(n for n, state in node.judge.state.items() if state.passed)
                )
            hoops = [len(passed) for passed in lap_hoops]
            missed = [[n for n in names if n not in passed] for passed in lap_hoops]
            clean = clean and not any(missed)
    else:
        clean = clean and contacts == 0
    result = dict(
        label=args.label,
        course=args.course,
        outcome=outcome,
        clean=clean,
        laps=laps,
        lap_times=lap_times,
        time=round(node.sim_now() - t0, 1),
        closest=round(closest, 3),
        contact_samples=int(contacts),
        reversing_samples=int(reversing),
        hoops=hoops,
        hoops_missed=missed,
        hoop_monitor=int(sum(node.hoops.passed)) if node.hoops is not None else None,
        end=[round(v, 2) for v in node.pose[1:3]],
        wall_s=round(time.time() - wall0),
    )
    print(json.dumps(result))
    if args.out:
        args.out.write_text(json.dumps(dict(result, trace=trace), indent=1))
    node.start.call_async(SetBool.Request(data=False))
    node.spin_for(0.5)
    print("PASS" if clean else f"FAIL -- {outcome}, closest {closest:.3f} m")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
