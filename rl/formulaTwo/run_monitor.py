#!/usr/bin/env python3
"""Watch a formulaTwo run from outside the driver, and say whether it was clean.

    python3 run_monitor.py --out /tmp/formula_two_monitor.json

Started by validate.sh beside Gazebo (or the loopback).  The driver's own
`clear` figure is computed from the map and the pose the driver BELIEVES, so
it cannot see its own mistakes.  This one uses the pose topic as truth -- in
Gazebo and the loopback it is ground truth -- and the real footprint (chassis
box plus tyre faces, config.yaml `collision`), which is what actually touches
a bale.  It also counts the depth watchdog's holds and losses off
/formula_one/depth_status.

Rewrites --out once a second, so the verdict survives however the run ends.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track as track_mod  # noqa: E402
from world import World  # noqa: E402

HERE = Path(__file__).resolve().parent


class RunMonitor(Node):
    def __init__(self, cfg, out, status_topic):
        super().__init__("formula_two_monitor")
        self.track = track_mod.build(cfg, HERE.parents[1])
        # The nominal world: in Gazebo the bales are exactly where the SDF says.
        self.world = World(self.track, cfg, 1, np.random.default_rng(0))
        self.world.sample(np.array([True]), enabled=False)
        self.graze = float(cfg["reward"]["graze_margin"])
        self.out = Path(out)
        self.hint = None
        self.state = dict(
            samples=0,
            min_clearance=None,
            min_clearance_station=None,
            grazing_samples=0,
            contact_samples=0,
            max_abs_cte=0.0,
            depth_holds=0,
            depth_lost=False,
            depth_status_last="",
            max_abs_roll_deg=0.0,
            first_contact_time=None,
            first_contact_station=None,
            depth_lost_time=None,
        )
        self.t0 = time.time()
        self._holding = False
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
        )
        self.create_subscription(String, status_topic, self.on_status, 10)
        self.create_timer(1.0, self.write)

    def on_pose(self, msg):
        p = msg.pose
        x, y = np.array([p.position.x]), np.array([p.position.y])
        q = p.orientation
        yaw = np.array(
            [math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))]
        )
        roll = math.degrees(
            math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
        )
        if self.hint is None:
            self.hint, _ = self.track.locate(x, y, yaw)
        self.hint, station, lateral, _ = self.world.frenet(x, y, yaw, self.hint)
        clear = float(self.world.body_clearance(x, y, yaw)[0])
        s = self.state
        s["samples"] += 1
        s["max_abs_cte"] = max(s["max_abs_cte"], abs(float(lateral[0])))
        s["grazing_samples"] += int(clear < self.graze)
        s["contact_samples"] += int(clear <= 0.0)
        s["max_abs_roll_deg"] = max(s["max_abs_roll_deg"], abs(roll))
        if clear <= 0.0 and s["first_contact_time"] is None:
            # Which came first matters: a bale strike tips the car over in
            # Gazebo, and a car on its side sees no bales in the height band,
            # so contact FOLLOWED by depth loss is a driving failure, not a
            # perception one.
            s["first_contact_time"] = round(time.time() - self.t0, 2)
            s["first_contact_station"] = float(station[0])
        if s["min_clearance"] is None or clear < s["min_clearance"]:
            s["min_clearance"] = clear
            s["min_clearance_station"] = float(station[0])

    def on_status(self, msg):
        text = msg.data
        s = self.state
        s["depth_status_last"] = text
        holding = text.startswith("hold")
        if holding and not self._holding:
            s["depth_holds"] += 1
        self._holding = holding
        if "FALLBACK" in text and not s["depth_lost"]:
            s["depth_lost"] = True
            s["depth_lost_time"] = round(time.time() - self.t0, 2)

    def write(self):
        self.state["wall_time"] = time.time()
        self.out.write_text(json.dumps(self.state, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/formula_two_monitor.json")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--status-topic", default="/formula_one/depth_status")
    args, ros_args = ap.parse_known_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rclpy.init(args=ros_args)
    node = RunMonitor(cfg, args.out, args.status_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
