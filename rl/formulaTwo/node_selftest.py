#!/usr/bin/env python3
"""Closed-loop check of formula_two_node, and of its depth watchdog.

    source /opt/ros/jazzy/setup.bash && source ../../install/setup.bash
    python3 node_selftest.py                  # all six scenarios, ~8 min
    python3 node_selftest.py --only drive,stop

Runs the node against ros_loopback (the plant plus a rendered 640x360 ZED
depth image) in one process, on its own ROS domain, in real time, and asserts
on what crossed /drive_cmd:

    drive   depth throughout: arms, never over the cap, never touches a bale,
            3 laps, then coasts to rest with zero throttle
    stop    depth stops 20 s after green: throttle zero within
            depth_timeout + a tick, prior steering, at rest, no contact
    nan     frames keep coming but every pixel is NaN: same, via the
            invalid-beam check
    blip    depth drops out for 0.6 s at 20 s: the network is taken off the
            wheel at 0.3 s, gets it back when depth returns, and the run
            finishes -- no fallback
    none    no depth ever: green is ignored, the car never moves
    map     depth stops, depth_fallback:=map: races on and finishes 3 laps

The loopback's car is the model the policy trained on, so a pass here says
the NODE is right, not that Gazebo or the car will agree.  Run it after
touching formula_two_node.py (formulaOne trap #10: a node that dies on its
first tick looks exactly like a policy that will not move).
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
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data

from cfr_interfaces.msg import DriveCommand

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track as track_mod  # noqa: E402
import yaml  # noqa: E402
from world import World  # noqa: E402

HERE = Path(__file__).resolve().parent
failures = []


def check(name, ok, detail=""):
    print(f"[{'  ok  ' if ok else ' FAIL '}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


class Monitor(Node):
    """Records every DriveCommand with the car's true clearance at that moment."""

    def __init__(self, track, cfg):
        super().__init__("monitor")
        self.track = track
        self.world = World(track, cfg, 1, np.random.default_rng(0))
        self.world.sample(np.array([True]), enabled=False)
        self.log = []  # (t, ready, steer, velocity, clearance, v_cap)
        self.hint = np.zeros(1, dtype=np.int64)
        self.pose = None
        self.create_subscription(DriveCommand, "/drive_cmd", self.on_cmd, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data)

    def on_pose(self, msg):
        self.pose = msg

    def on_cmd(self, msg):
        t = time.time()
        if self.pose is None:
            self.log.append((t, msg.auto_ready, msg.steering, msg.velocity, 9.0, 9.0))
            return
        x = np.array([self.pose.pose.position.x])
        y = np.array([self.pose.pose.position.y])
        q = self.pose.pose.orientation
        yaw = np.array([np.arctan2(2 * q.w * q.z, 1 - 2 * q.z**2)])
        self.hint, _, _ = self.track.project(x, y, self.hint)
        clear = float(self.world.body_clearance(x, y, yaw)[0])
        cap = float(self.track.v_cap[self.hint][0])
        self.log.append((t, msg.auto_ready, msg.steering, msg.velocity, clear, cap))


def scenario(name, cfg, trk, policy, mode, fallback, fail_after, seconds):
    import formula_two_node
    import ros_loopback

    rclpy.init()
    car = ros_loopback.Loopback(
        parameter_overrides=[
            Parameter("depth_mode", value=mode),
            Parameter("depth_fail_after", value=float(fail_after)),
        ]
    )
    driver = formula_two_node.FormulaTwo(
        parameter_overrides=[
            Parameter("policy", value=str(policy)),
            Parameter("depth_fallback", value=fallback),
            Parameter("publish_markers", value=False),
        ]
    )
    monitor = Monitor(trk, cfg)
    ex = SingleThreadedExecutor()
    for n in (car, driver, monitor):
        ex.add_node(n)
    t0 = time.time()
    deadline = t0 + seconds
    while time.time() < deadline and not driver.finished:
        ex.spin_once(timeout_sec=0.01)
    out = dict(
        finished=driver.finished,
        laps=driver.laps_done,
        distance=driver.distance,
        fallback=driver.fallback,
        reason=driver.fallback_reason,
        race_time=driver.race_time,
        frames=driver.depth_frames,
        holds=driver.hold_count,
        go_time=car.go_time,
        speed=float(car.plant.speed[0]),
        log=np.array(monitor.log, dtype=float),
        wall=time.time() - t0,
    )
    for n in (car, driver, monitor):
        ex.remove_node(n)
        n.destroy_node()
    rclpy.shutdown()
    print(
        f"\n--- {name}: {out['wall']:.0f} s, {out['laps']} laps, {out['distance']:.1f} m, "
        f"fallback {out['fallback']} {out['reason']}"
    )
    return out


def common(name, r, cfg):
    log = r["log"]
    moving = log[log[:, 3] > 0.05]
    check(f"{name}: never over the rule cap", len(moving) == 0 or (moving[:, 3] - moving[:, 5]).max() <= 1e-6)
    check(f"{name}: steering in range", np.abs(log[:, 2]).max() <= 1.0 + 1e-6)
    check(f"{name}: never touched a bale", log[:, 4].min() > 0.0, f"min body clearance {log[:, 4].min():+.3f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=HERE / "bestModel/f2_v2_40M/policy.npz")
    ap.add_argument("--only", default="drive,stop,nan,blip,none,map")
    ap.add_argument("--domain", type=int, default=78)
    args = ap.parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    trk = track_mod.build(cfg, HERE.parents[1])
    laps = int(cfg["env"]["laps"])
    timeout = 1.0  # the node's default depth_timeout
    todo = args.only.split(",")

    if "drive" in todo:
        r = scenario("drive", cfg, trk, args.policy, "ok", "stop", 1e9, 150)
        log = r["log"]
        check("drive: arms before it drives", bool(len(log)) and all(log[:10, 1] > 0.5) and np.abs(log[:10, 3]).max() < 1e-6)
        common("drive", r, cfg)
        check("drive: 3 laps and at rest", r["finished"] and r["laps"] >= laps, f"{r['laps']} laps, race {r['race_time'] or float('nan'):.1f} s")
        check("drive: no fallback", r["fallback"] is None, r["reason"])
        check("drive: zero throttle while coasting to rest", np.abs(log[-20:, 3]).max() < 1e-6)

    for name, mode in (("stop", "stop"), ("nan", "nan")):
        if name not in todo:
            continue
        r = scenario(name, cfg, trk, args.policy, mode, "stop", 20.0, 60)
        log = r["log"]
        common(name, r, cfg)
        # The loopback stamps green on the node clock, which is the wall clock
        # here (no sim time), the same clock the monitor logs on.
        after = log[log[:, 0] >= r["go_time"] + 20.0]
        zero = after[np.abs(after[:, 3]) < 1e-6]
        react = zero[0, 0] - (r["go_time"] + 20.0) if len(zero) else float("inf")
        budget = timeout + 0.15 if mode == "stop" else 0.5
        check(f"{name}: throttle cut after depth loss", react <= budget, f"{react:.2f} s (budget {budget:.2f} s)")
        tail = after[after[:, 0] >= zero[0, 0]] if len(zero) else after
        check(f"{name}: throttle stays zero afterwards", len(tail) > 0 and np.abs(tail[:, 3]).max() < 1e-6)
        check(f"{name}: fallback is stop", r["fallback"] == "stop", r["reason"])
        check(f"{name}: came to rest", r["finished"] and r["speed"] < 0.3, f"{r['speed']:.2f} m/s, {r['distance']:.1f} m")

    if "blip" in todo:
        r = scenario("blip", cfg, trk, args.policy, "blip", "stop", 20.0, 150)
        common("blip", r, cfg)
        check("blip: network was taken off the wheel", r["holds"] >= 1, f"{r['holds']} hold(s)")
        check("blip: no fallback", r["fallback"] is None, r["reason"])
        check("blip: still finished 3 laps and stopped", r["finished"] and r["laps"] >= laps, f"{r['laps']} laps, race {r['race_time'] or float('nan'):.1f} s")

    if "none" in todo:
        r = scenario("none", cfg, trk, args.policy, "none", "stop", 0.0, 12)
        log = r["log"]
        check("none: never moved after green", np.abs(log[:, 3]).max() < 1e-6 and r["distance"] < 0.05, f"max velocity cmd {np.abs(log[:, 3]).max():.2f}")
        check("none: saw no depth frames", r["frames"] == 0)

    if "map" in todo:
        r = scenario("map", cfg, trk, args.policy, "stop", "map", 20.0, 170)
        common("map", r, cfg)
        check("map: fell back to the map", r["fallback"] == "map", r["reason"])
        check("map: still finished 3 laps and stopped", r["finished"] and r["laps"] >= laps, f"{r['laps']} laps, race {r['race_time'] or float('nan'):.1f} s")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("formula_two_node: all scenarios passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
