#!/usr/bin/env python3
"""Measure Gazebo's actual turning radius and compare it with plant.py.

    python3 measure_turn_radius.py          # needs a running sim

Teleports the car to open ground beside the course, drives a constant
steering command at a constant speed, fits a circle to the path, and prints
the radius the car achieved next to the radius `plant.py` predicts for that
same command and speed.

This is the measurement that decides how to close the sim-to-real gap, and it
cannot be taken on the course itself: the corridor is 0.92 m wide, so any car
that turns differently from the model hits a bale before it has turned far
enough to measure.  Hence the teleport.

It also has to be taken with something OTHER than a driving policy in the
loop.  A policy that crashes gives a handful of seconds of data in a corner
it entered badly, which is how a meaningless ratio gets mistaken for a
finding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import os

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cfr_interfaces.msg import DriveCommand

HERE = Path(__file__).resolve().parent
# teleport_api binds a FIXED port with no domain scoping.  ROS_DOMAIN_ID and
# GZ_PARTITION isolate ROS and gz transport; they do not isolate HTTP.  Two
# simulations on one machine therefore share this endpoint, and a teleport
# meant for yours will silently move someone else's car -- which is exactly
# what happened while this file was being written.
TELEPORT_PORT = int(os.environ.get("CFR_TELEPORT_PORT", "9003"))
TELEPORT = f"http://127.0.0.1:{TELEPORT_PORT}/api/sim/teleport"


def teleport(x, y, heading):
    body = json.dumps({"x": x, "y": y, "heading": heading}).encode()
    req = urllib.request.Request(TELEPORT, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def fit_circle(x, y):
    """Algebraic circle fit; returns radius (inf for a straight line)."""
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx**2 + cy**2
    return math.sqrt(r2) if r2 > 0 else float("inf")


class Driver(Node):
    def __init__(self):
        super().__init__("measure_turn_radius")
        self.pub = self.create_publisher(DriveCommand, "/drive_cmd",
                                         qos_profile_sensor_data)
        self.poses = []
        self.recording = False
        self.create_subscription(PoseStamped, "/zed/zed_node/pose", self.on_pose,
                                 qos_profile_sensor_data)
        self.steer = 0.0
        self.speed = 0.0
        self.create_timer(0.02, self.tick)

    def on_pose(self, msg):
        if self.recording:
            self.poses.append((msg.pose.position.x, msg.pose.position.y))

    def tick(self):
        m = DriveCommand()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "base_link"
        m.auto_ready = True
        m.steering = float(self.steer)
        m.velocity = float(self.speed)
        self.pub.publish(m)


def spin(node, seconds):
    stop = time.time() + seconds
    while time.time() < stop:
        rclpy.spin_once(node, timeout_sec=0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=20.0)
    ap.add_argument("--y", type=float, default=-14.0)
    ap.add_argument("--settle", type=float, default=4.0)
    ap.add_argument("--record", type=float, default=11.0)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    p = cfg["plant"]
    pts = np.asarray(p["steering_command_points"])
    angs = np.asarray(p["steering_angle_points"])

    rclpy.init()
    node = Driver()

    # Refuse to teleport anything until a simulator is confirmed alive on OUR
    # ROS_DOMAIN_ID.  Without this the script will happily drive a simulation
    # belonging to someone else that merely happens to hold the port.
    print(f"  checking for a simulator on ROS_DOMAIN_ID="
          f"{os.environ.get('ROS_DOMAIN_ID', '0')} ...", flush=True)
    seen = []
    node.recording = True
    spin(node, 4.0)
    seen, node.poses, node.recording = list(node.poses), [], False
    if not seen:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(
            "  No pose on /zed/zed_node/pose for this ROS_DOMAIN_ID.\n"
            f"  Refusing to POST to port {TELEPORT_PORT}: that endpoint is not\n"
            "  domain-scoped, so it may belong to a different simulation.\n"
            "  Start the sim on this domain first (./validate.sh ...)."
        )

    print(f"\n  {'cmd':>6} {'speed':>6} {'model R':>9} {'gazebo R':>9} {'ratio':>7}")
    rows = []
    sweep = [(s, 2.0) for s in (1.0, 0.8, 0.6, 0.4, 0.25,
                                -0.25, -0.4, -0.6, -0.8, -1.0)]
    for steer, speed in sweep:
        teleport(args.x, args.y, 0.0)
        node.steer, node.speed, node.recording, node.poses = 0.0, 0.0, False, []
        spin(node, 1.5)
        node.steer, node.speed = steer, speed
        spin(node, args.settle)              # past the dead time, up to speed
        node.recording = True
        spin(node, args.record)
        node.recording = False
        pts_xy = np.array(node.poses)
        node.steer, node.speed = 0.0, 0.0
        spin(node, 0.5)
        if len(pts_xy) < 30:
            print(f"  {steer:6.2f} {speed:6.2f}   too few samples")
            continue
        gz = fit_circle(pts_xy[:, 0], pts_xy[:, 1])
        angle = float(np.interp(steer, pts, angs))
        eff = p["wheelbase"] + p["understeer_gradient"] * speed**2
        model = abs(eff / math.tan(angle))
        rows.append((steer, speed, model, gz))
        print(f"  {steer:6.2f} {speed:6.2f} {model:8.2f}m {gz:8.2f}m "
              f"{gz/model:7.2f}")

    node.destroy_node()
    rclpy.shutdown()
    if rows:
        ratios = np.array([r[3] / r[2] for r in rows])
        print(f"\n  Gazebo turns {np.median(ratios):.2f}x the radius plant.py predicts")
        print(f"  (1.00 = the model is right; >1 = the car understeers more than modelled)")
        tightest = min(r[3] for r in rows if abs(r[0]) == 1.0)
        print(f"  tightest circle Gazebo achieved: {tightest:.2f} m "
              f"-- the course's hairpin needs 1.48 m")


if __name__ == "__main__":
    main()
