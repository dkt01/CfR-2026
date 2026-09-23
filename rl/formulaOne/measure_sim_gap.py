#!/usr/bin/env python3
"""Compare what Gazebo's car DOES against what plant.py says it will do.

    python3 measure_sim_gap.py --seconds 40

Records /drive_cmd alongside /zed/zed_node/pose while something else drives,
then asks one question at every logged instant: given this steering command
and this speed, plant.py predicts a yaw rate -- did the car deliver it?

That ratio is the sim-to-model gap in the one channel that matters for a
0.92 m corridor.  A ratio near 1.0 means the trainer's model is right and a
failure in Gazebo is about timing or noise.  A ratio well under 1.0 means the
car understeers more than the kinematic bicycle in `plant.py` allows, which no
amount of randomising its PARAMETERS can fix -- it is a missing term, not a
mis-set one.

The course has no open area to run a skidpad in, so this measures in situ, at
the speeds and steering angles actually used, which is where the answer
matters anyway.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cfr_interfaces.msg import DriveCommand

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent


class Recorder(Node):
    def __init__(self):
        super().__init__("measure_sim_gap")
        self.cmd = None
        self.poses = []
        self.cmds = []
        self.create_subscription(
            DriveCommand, "/drive_cmd", self.on_cmd, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
        )

    def on_cmd(self, msg):
        self.cmd = (float(msg.steering), float(msg.velocity), bool(msg.auto_ready))

    def on_pose(self, msg):
        if self.cmd is None or not self.cmd[2]:
            return
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        t = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.poses.append((t, msg.pose.position.x, msg.pose.position.y, yaw))
        self.cmds.append(self.cmd[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    p = cfg["plant"]
    steer_pts = np.asarray(p["steering_command_points"])
    steer_ang = np.asarray(p["steering_angle_points"])

    rclpy.init()
    node = Recorder()
    import time as wall

    stop = wall.time() + args.seconds
    while wall.time() < stop:
        rclpy.spin_once(node, timeout_sec=0.05)
    poses = np.array(node.poses)
    cmds = np.array(node.cmds)
    node.destroy_node()
    rclpy.shutdown()

    if len(poses) < 50:
        raise SystemExit(f"only {len(poses)} samples -- is anything driving?")

    t, x, y, yaw = poses.T
    dt = np.diff(t)
    dyaw = np.unwrap(yaw)
    achieved = np.diff(dyaw) / np.maximum(dt, 1e-6)
    speed = np.hypot(np.diff(x), np.diff(y)) / np.maximum(dt, 1e-6)
    steer = cmds[:-1, 0]

    # plant.py's prediction for the same command and the same measured speed.
    angle = np.interp(steer, steer_pts, steer_ang)
    eff_wheelbase = p["wheelbase"] + p["understeer_gradient"] * speed**2
    predicted = speed * np.tan(angle) / eff_wheelbase

    # Only where the car is actually cornering at speed: a straight line has
    # no yaw rate to get wrong, and a crawl is dominated by pose noise.
    sane = (dt > 1e-3) & (dt < 0.2)
    moving = sane & (speed > 1.5)
    use = moving & (np.abs(predicted) > 0.15)
    print(
        f"\n  {len(poses)} poses | {sane.sum()} sane dt | {moving.sum()} moving "
        f"(>1.5 m/s) | {use.sum()} cornering (|predicted| > 0.15 rad/s)"
    )
    if moving.sum() < 20:
        raise SystemExit(
            "  The car was barely moving for this window -- it had already "
            "beached, or the run had not started. Time the measurement to "
            "begin at the green light."
        )
    if use.sum() < 20:
        raise SystemExit(f"  only {use.sum()} cornering samples; widen the window")
    ratio = achieved[use] / predicted[use]

    print(
        f"\n  {use.sum()} cornering samples, speed {speed[use].min():.1f}"
        f"-{speed[use].max():.1f} m/s\n"
    )
    print("  achieved / predicted yaw rate")
    print(f"    median   {np.median(ratio):.3f}")
    print(f"    mean     {ratio.mean():.3f}")
    print(
        f"    p10-p90  {np.percentile(ratio, 10):.3f} - {np.percentile(ratio, 90):.3f}"
    )
    print()
    for lo, hi in ((1.5, 3.0), (3.0, 4.2), (4.2, 6.0)):
        m = use & (speed > lo) & (speed <= hi)
        if m.sum() > 10:
            r = achieved[m] / predicted[m]
            print(f"    {lo:.1f}-{hi:.1f} m/s: {np.median(r):.3f}  (n={m.sum()})")
    print()
    shortfall = 1.0 - np.median(ratio)
    if shortfall > 0.10:
        print(f"  Gazebo delivers {100 * shortfall:.0f}% LESS yaw rate than plant.py")
        print("  predicts. That is a missing term in the model, not a mis-set")
        print("  parameter -- randomising understeer cannot span it.")
    else:
        print(f"  plant.py matches Gazebo to {100 * abs(shortfall):.0f}% in yaw rate.")
    if args.out:
        np.savez(
            args.out, t=t, x=x, y=y, yaw=yaw, steer=cmds[:, 0], velocity=cmds[:, 1]
        )
        print(f"\n  raw log -> {args.out}")


if __name__ == "__main__":
    main()
