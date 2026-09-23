#!/usr/bin/env python3
"""At what lateral acceleration does the simulated car go over?

    python3 measure_rollover.py          # needs a running sim

`plant.py` is a kinematic bicycle. It has no roll degree of freedom, no centre
of gravity and no track width, so there is no lateral acceleration at which it
does anything but keep turning. Gazebo's car has all three, and it does go
over: a trained policy was found lying on its roof at station 0.6 m, roll
-180 deg, motionless for 200 s, having commanded full lock.

That is a whole failure mode the trainer cannot see, so it cannot be trained
around -- it has to be fenced off instead, and the fence has to be measured
rather than guessed. This drives a constant steering command on open ground
beside the course at a series of speeds, and reports the lateral acceleration
at which roll first departs from flat.

The number it prints goes in `config.yaml` as `track.rollover_accel`, where
`env.py` treats exceeding it as contact: the car is on its roof, the run is
over, and pretending otherwise would train a policy to drive through a wall
it cannot see.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cfr_interfaces.msg import DriveCommand

HERE = Path(__file__).resolve().parent
# See measure_turn_radius.py: this endpoint is NOT domain-scoped.
TELEPORT_PORT = int(os.environ.get("CFR_TELEPORT_PORT", "9003"))
TELEPORT = f"http://127.0.0.1:{TELEPORT_PORT}/api/sim/teleport"
TICK = 0.02


def teleport(x, y, heading):
    body = json.dumps({"x": x, "y": y, "heading": heading}).encode()
    req = urllib.request.Request(
        TELEPORT, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def rpy(q):
    sinr, cosr = 2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)
    sinp = max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))
    return math.atan2(sinr, cosr), math.asin(sinp)


class Driver(Node):
    def __init__(self):
        super().__init__("measure_rollover")
        self.pub = self.create_publisher(
            DriveCommand, "/drive_cmd", qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
        )
        self.steer = self.speed = 0.0
        self.samples = []
        self.recording = False
        self.create_timer(TICK, self.tick)

    def on_pose(self, m):
        if not self.recording:
            return
        roll, pitch = rpy(m.pose.orientation)
        self.samples.append(
            (time.time(), m.pose.position.x, m.pose.position.y, roll, pitch)
        )

    def tick(self):
        msg = DriveCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.auto_ready = True
        msg.steering = float(self.steer)
        msg.velocity = float(self.speed)
        self.pub.publish(msg)


def spin(node, seconds):
    stop = time.time() + seconds
    while time.time() < stop:
        rclpy.spin_once(node, timeout_sec=0.02)


def one(node, x, y, speed, cmd, settle, hold, reversal=0.0):
    """Accelerate straight, then steer.  Returns (a_y, max roll, v).

    With `reversal` > 0 the lock is held for that long and then swung to the
    OPPOSITE lock in a single tick.  That is the manoeuvre this course forces
    at station 103->107, where curvature goes from +0.659 to -0.630 in four
    metres, and it is the one that rolls cars: a steady corner settles into
    one load transfer, a reversal stacks the second one on top of a body that
    is still rolling back from the first.  Vehicle certification calls it the
    fishhook test, and measuring only the steady case -- which is what the
    first version of this file did -- misses it completely.
    """
    teleport(x, y, 0.0)
    node.steer, node.speed, node.recording, node.samples = 0.0, 0.0, False, []
    spin(node, 1.0)
    node.steer, node.speed = 0.0, speed
    spin(node, settle)  # up to speed, straight

    node.recording, node.samples = True, []
    node.steer = cmd
    if reversal > 0.0:
        spin(node, reversal)
        node.steer = -cmd  # THE REVERSAL
        spin(node, hold)
    else:
        spin(node, hold)
    node.recording = False
    node.steer, node.speed = 0.0, 0.0
    spin(node, 0.5)

    if len(node.samples) < 20:
        return None
    a = np.array([(s[0], s[1], s[2], s[3], s[4]) for s in node.samples])
    t, xs, ys = a[:, 0] - a[0, 0], a[:, 1], a[:, 2]
    roll = np.abs(a[:, 3])
    # Lateral acceleration from the path itself: fit the yaw rate over the
    # settled part of the hold and use a_y = v * omega.  Measured rather than
    # assumed, because the achieved radius is not the commanded one.
    mask = t > (t[-1] * 0.4)
    if mask.sum() < 10:
        return None
    dx, dy = np.gradient(xs[mask], t[mask]), np.gradient(ys[mask], t[mask])
    v = float(np.median(np.hypot(dx, dy)))
    heading = np.unwrap(np.arctan2(dy, dx))
    omega = float(np.median(np.gradient(heading, t[mask])))
    return v * abs(omega), float(np.degrees(roll.max())), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=20.0)
    ap.add_argument("--y", type=float, default=-14.0)
    ap.add_argument("--settle", type=float, default=3.5)
    ap.add_argument("--hold", type=float, default=2.5)
    ap.add_argument(
        "--speeds", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    )
    ap.add_argument(
        "--cmd",
        type=float,
        default=-1.0,
        help="steering command to hold (default full right lock)",
    )
    ap.add_argument(
        "--reversal",
        type=float,
        default=0.0,
        help="hold the lock this long, then swing to the opposite "
        "lock in one tick (0 = steady lock, the old test)",
    )
    args = ap.parse_args()

    yaml.safe_load((HERE / "config.yaml").read_text())
    rclpy.init()
    node = Driver()

    node.recording, node.samples = True, []
    spin(node, 4.0)
    seen, node.samples, node.recording = list(node.samples), [], False
    if not seen:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(
            "  No pose on /zed/zed_node/pose for this ROS_DOMAIN_ID.\n"
            f"  Refusing to POST to port {TELEPORT_PORT}: that endpoint is not\n"
            "  domain-scoped, so it may belong to a different simulation."
        )

    what = (
        f"lock {args.cmd:+.2f} for {args.reversal:.2f}s then REVERSED"
        if args.reversal > 0
        else f"constant lock {args.cmd:+.2f}"
    )
    print(f"\n  {what}, increasing speed\n")
    print(f"  {'speed':>6} {'achieved v':>11} {'lat accel':>11} {'max roll':>10}")
    rolled_at = None
    for speed in args.speeds:
        got = one(
            node, args.x, args.y, speed, args.cmd, args.settle, args.hold, args.reversal
        )
        if got is None:
            print(f"  {speed:6.2f}   too few samples, skipped")
            continue
        a_y, roll_deg, v = got
        flag = ""
        if roll_deg > 45.0:
            flag = "   <- OVER"
            rolled_at = a_y if rolled_at is None else min(rolled_at, a_y)
        elif roll_deg > 15.0:
            flag = "   <- lifting"
        print(
            f"  {speed:6.2f} {v:11.2f} {a_y:9.2f} g={a_y / 9.81:4.2f} "
            f"{roll_deg:9.1f}d{flag}"
        )

    node.destroy_node()
    rclpy.shutdown()
    print()
    if rolled_at:
        print(f"  ROLLS at {rolled_at:.2f} m/s^2 ({rolled_at / 9.81:.2f} g).")
        print("  Put a margin under it in config.yaml:")
        print(f"      track.rollover_accel: {0.85 * rolled_at:.1f}")
    else:
        print(
            "  Never went over in this sweep.  Widen --speeds or try the "
            "other lock (steering is asymmetric)."
        )
    print()


if __name__ == "__main__":
    main()
