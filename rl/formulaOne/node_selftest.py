#!/usr/bin/env python3
"""Closed-loop check of the ROS node itself, with no Gazebo and no checkpoint.

    source /opt/ros/jazzy/setup.bash && source ../../install/setup.bash
    python3 node_selftest.py                 # scripted driver
    python3 node_selftest.py --policy runs/v1/policy.npz

Runs `formula_one_node` against `ros_loopback` in one process and watches the
wire.  What it is actually asserting is that the things which only exist on
the ROS side -- the start latch, the speed source, the lap bookkeeping and the
cap clamp on the way out -- behave the way `env.py` assumed they would.  Those
four are exactly the parts the offline evaluator cannot see, and therefore the
parts most likely to be wrong on the day.

Real time, because the node runs on a wall-clock timer: budget about a minute.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cfr_interfaces.msg import DriveCommand

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track as track_mod  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
failures = []


def check(name, ok, detail=""):
    print(
        f"[{'  ok  ' if ok else ' FAIL '}] {name}" + (f"   {detail}" if detail else "")
    )
    if not ok:
        failures.append(name)


class Monitor(Node):
    """Watches the two topics that matter and records what crossed them."""

    def __init__(self, track, cfg):
        super().__init__("monitor")
        self.track = track
        veh = cfg["vehicle"]
        self.hl, self.hw = veh["length"] / 2, veh["width"] / 2
        self.commands = []
        self.before_go = []
        self.samples = []
        self.hint = np.zeros(1, dtype=np.int64)
        self.pose = None
        self.create_subscription(
            DriveCommand, "/drive_cmd", self.on_cmd, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
        )

    def on_pose(self, msg):
        self.pose = msg

    def on_cmd(self, msg):
        self.commands.append((msg.auto_ready, msg.steering, msg.velocity))
        if self.pose is None:
            return
        x = np.array([self.pose.pose.position.x])
        y = np.array([self.pose.pose.position.y])
        q = self.pose.pose.orientation
        yaw = np.array([np.arctan2(2 * (q.w * q.z), 1 - 2 * q.z**2)])
        self.hint, station, _ = self.track.project(x, y, self.hint)
        clear = self.track.body_clearance(x, y, yaw, self.hl, self.hw)[0]
        self.samples.append(
            (station[0], self.track.v_cap[self.hint][0], msg.velocity, clear)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=140.0)  # two laps at the
    # scripted floor is ~72 s, plus the coast to rest
    ap.add_argument(
        "--domain",
        type=int,
        default=77,
        help="ROS_DOMAIN_ID to run on; must be one nothing else uses",
    )
    args = ap.parse_args()

    # On its own domain, always.  A simulator left running elsewhere on the
    # machine publishes /drive_cmd and /zed/zed_node/pose too, and two stacks
    # that can see each other do not fail loudly -- they interleave, and the
    # car appears to be driven by a policy that is not the one under test.
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    trk = track_mod.build(cfg, HERE.parents[1])

    rclpy.init()
    import formula_one_node
    import ros_loopback

    from rclpy.parameter import Parameter

    overrides = [
        Parameter("driver", value="baseline" if args.policy is None else "policy")
    ]
    if args.policy is not None:
        overrides.append(Parameter("policy", value=str(args.policy)))

    car = ros_loopback.Loopback()
    driver = formula_one_node.FormulaOne(parameter_overrides=overrides)
    monitor = Monitor(trk, cfg)

    ex = SingleThreadedExecutor()
    for n in (car, driver, monitor):
        ex.add_node(n)

    deadline = time.time() + args.seconds
    while time.time() < deadline and not driver.finished:
        ex.spin_once(timeout_sec=0.02)

    laps, distance, finished = driver.laps_done, driver.distance, driver.finished
    stopping, race_time = driver.stopping, driver.race_time
    for n in (car, driver, monitor):
        ex.remove_node(n)
        n.destroy_node()
    rclpy.shutdown()

    print()
    arming = [c for c in monitor.commands[:20]]
    check(
        "arms the Arduino before it drives",
        bool(arming)
        and all(c[0] for c in arming)
        and all(abs(c[2]) < 1e-6 for c in arming[:5]),
        "auto_ready true with zero velocity while waiting for green",
    )
    check(
        "published DriveCommand throughout",
        len(monitor.commands) > 100,
        f"{len(monitor.commands)} commands",
    )

    s = np.array(monitor.samples)
    moving = s[s[:, 2] > 0.1] if len(s) else np.empty((0, 4))
    check("drove", len(moving) > 50, f"{len(moving)} commands with throttle")
    if len(moving):
        excess = (moving[:, 2] - moving[:, 1]).max()
        check(
            "never commanded above the rule cap",
            excess <= 1e-6,
            f"worst {excess:+.4f} m/s over",
        )
        check(
            "steering stayed in range",
            max(abs(c[1]) for c in monitor.commands) <= 1.0 + 1e-6,
        )
        check(
            "never touched a bale",
            s[:, 3].min() > 0.0,
            f"min clearance {s[:, 3].min():.3f} m",
        )
    check(
        "completed the run",
        finished and laps >= cfg["env"]["laps"],
        f"{laps} laps, {distance:.1f} m",
    )
    # The run is two laps AND a stop, and the node has to do the stop the same
    # way the trainer does: throttle to zero, steering still live, until the
    # car is at rest.  Cutting steering at the line -- which is what this node
    # used to do -- coasts 15 m straight into the bales it was turning away
    # from, and `finished` would still have been True.
    check(
        "stopped after the last lap rather than freezing the wheels",
        stopping and race_time is not None,
        f"race {race_time:.1f} s, then coasted "
        f"{distance - cfg['env']['laps'] * trk.length:.1f} m to rest"
        if race_time
        else "never entered the stopping phase",
    )
    if len(moving):
        tail = [c for c in monitor.commands[-40:]]
        check(
            "commanded zero throttle while coasting to a stop",
            all(abs(c[2]) < 1e-6 for c in tail),
            f"last {len(tail)} commands",
        )

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("ROS node drives the course end to end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
