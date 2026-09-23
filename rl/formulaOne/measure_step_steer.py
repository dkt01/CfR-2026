#!/usr/bin/env python3
"""Step-steer transient test: does tire_scrub (calibrated from steady circles)
also describe Gazebo's response during hairpin TURN-IN, or is there extra
transient understeer a flat multiplier misses?

    python3 measure_step_steer.py          # needs a running sim

Drives straight at a hairpin approach speed, then steps the steering command
from 0 to a fixed value in a single tick -- matching how both the scripted
prior and the RL policy actually act at a hairpin entry, there is no ramp on
the command side, only the servo's own lag -- and records the pose trajectory
through the transient.  The identical command trace is replayed through
plant.py's Plant at fine resolution, and the two yaw-rate traces are compared
throughout the transient, not just at steady state:

  * measure_turn_radius.py already established tire_scrub ~= 1.10 STEADY
    STATE (settled circles, several seconds each, see its module docstring).
  * If Gazebo's yaw rate divided by the model's stays close to 1.10 all the
    way through the transient, tire_scrub already explains turn-in and the
    Gazebo regression (fine-tuned policy beaches EARLIER than the pre-fix
    one, see README's "Known gap") has some other cause.
  * If that ratio is well above 1.10 in the first ~0.3-0.5s after the step
    and relaxes toward 1.10 later, the car understeers MORE while turning in
    than once settled, and a flat scrub multiplier undercorrects exactly
    where both post-scrub Gazebo runs failed.

Measured yaw rate is extracted with a Savitzky-Golay filter (smooth + take
the derivative in one pass) rather than a parametric fit, so the comparison
does not presuppose the model's own first-order-lag shape -- if the real car
has a genuinely different transient shape, this will show it instead of
fitting it away.
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
from scipy.signal import savgol_filter

from cfr_interfaces.msg import DriveCommand
from plant import Plant

HERE = Path(__file__).resolve().parent
# See measure_turn_radius.py: this endpoint is NOT domain-scoped.
TELEPORT_PORT = int(os.environ.get("CFR_TELEPORT_PORT", "9003"))
TELEPORT = f"http://127.0.0.1:{TELEPORT_PORT}/api/sim/teleport"

TICK = 0.02  # s, command publish period (matches measure_turn_radius.py)


def teleport(x, y, heading):
    body = json.dumps({"x": x, "y": y, "heading": heading}).encode()
    req = urllib.request.Request(
        TELEPORT, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def yaw_of(msg):
    # This car's own convention (formula_one_node.py): planar quaternion,
    # z = sin(yaw/2), w = cos(yaw/2).
    return 2.0 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w)


class Driver(Node):
    def __init__(self):
        super().__init__("measure_step_steer")
        self.pub = self.create_publisher(
            DriveCommand, "/drive_cmd", qos_profile_sensor_data
        )
        self.samples = []  # (t_wall, yaw)
        self.recording = False
        self.t0 = None
        self.create_subscription(
            PoseStamped, "/zed/zed_node/pose", self.on_pose, qos_profile_sensor_data
        )
        self.steer = 0.0
        self.speed = 0.0
        self.create_timer(TICK, self.tick)

    def on_pose(self, msg):
        if self.recording:
            self.samples.append((time.time(), yaw_of(msg)))

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


def unwrap_rel(t, yaw):
    """Unwrap yaw, return (t, yaw) with yaw[0] == 0."""
    yaw = np.unwrap(yaw)
    return t, yaw - yaw[0]


def smoothed_rate(t, yaw, dt_grid=0.01, win_s=0.15, poly=3):
    """Resample to a uniform grid and take a Savitzky-Golay derivative."""
    t_grid = np.arange(t[0], t[-1], dt_grid)
    yaw_grid = np.interp(t_grid, t, yaw)
    win = int(win_s / dt_grid)
    win += 1 - (win % 2)  # savgol needs an odd window
    win = max(win, poly + 2 + (poly + 2) % 2 + 1)
    rate = savgol_filter(yaw_grid, win, poly, deriv=1, delta=dt_grid)
    return t_grid, rate


def crossing_time(t, rate, frac, steady):
    """First time `rate` (signed like `steady`) crosses frac*steady, t>0 only."""
    target = frac * steady
    post = t > 0
    t, rate = t[post], rate[post]
    if steady >= 0:
        idx = np.argmax(rate >= target)
    else:
        idx = np.argmax(rate <= target)
    if idx == 0 and not (
        (steady >= 0 and rate[0] >= target) or (steady < 0 and rate[0] <= target)
    ):
        return float("nan")
    if idx == 0:
        return t[0]
    r0, r1 = rate[idx - 1], rate[idx]
    t0, t1 = t[idx - 1], t[idx]
    if r1 == r0:
        return t1
    frac_seg = (target - r0) / (r1 - r0)
    return t0 + frac_seg * (t1 - t0)


def model_trace(cfg, speed, step_command, t_pre, t_post):
    """Replay the identical command schedule through plant.py.

    dt is the plant's OWN substep resolution (cfg env control_hz/substeps),
    not an arbitrary finer step: the dead-time history buffer is sized for
    that dt_sub, so a smaller dt here would silently clip the 0.19s command
    dead time down to whatever the (too-short) buffer holds.
    """
    flat = {**cfg, "randomize": {**cfg["randomize"], "enabled": False}}
    plant = Plant(flat, 1, np.random.default_rng(0))
    plant.reset(
        np.array([True]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([speed]),
    )
    dt = plant.dt_sub
    n_pre = int(round(t_pre / dt))
    n_post = int(round(t_post / dt))
    t = np.zeros(n_pre + n_post)
    rate = np.zeros_like(t)
    for i in range(n_pre):
        r = plant.substep(np.array([0.0]), np.array([speed]), np.array([dt]))
        t[i] = (i + 1) * dt - t_pre
        rate[i] = r[0]
    for i in range(n_post):
        r = plant.substep(np.array([step_command]), np.array([speed]), np.array([dt]))
        t[n_pre + i] = (i + 1) * dt
        rate[n_pre + i] = r[0]
    return t, rate


def run_one(node, cfg, x, y, speed, step_command, settle, pre, post):
    teleport(x, y, 0.0)
    node.steer, node.speed, node.recording, node.samples = 0.0, 0.0, False, []
    spin(node, 1.0)
    node.steer, node.speed = 0.0, speed
    spin(node, settle)  # up to speed, straight, steer settled at 0

    node.recording, node.samples = True, []
    spin(node, pre)  # baseline, steer still 0
    step_t = time.time()
    node.steer = step_command  # THE STEP
    spin(node, post)
    node.recording = False
    node.steer, node.speed = 0.0, 0.0
    spin(node, 0.5)

    if len(node.samples) < 30:
        return None
    t = np.array([s[0] - step_t for s in node.samples])
    yaw = np.array([s[1] for s in node.samples])
    order = np.argsort(t)
    t, yaw = t[order], yaw[order]
    return unwrap_rel(t, yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, default=20.0)
    ap.add_argument("--y", type=float, default=-14.0)
    ap.add_argument(
        "--speed", type=float, default=2.5, help="hairpin approach speed, m/s"
    )
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--pre", type=float, default=0.5)
    ap.add_argument("--post", type=float, default=2.0)
    ap.add_argument(
        "--commands",
        type=float,
        nargs="+",
        default=[1.0, -1.0, 0.55, -0.55],
        help="steering commands to step to (0.55 ~= the "
        "tightest hairpin's required angle)",
    )
    ap.add_argument(
        "--save",
        type=Path,
        default=None,
        help="write the raw (t, yaw) traces to an .npz, so a lag "
        "model can be FITTED to them offline instead of "
        "eyeballed off the table below",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())

    rclpy.init()
    node = Driver()

    print(
        f"  checking for a simulator on ROS_DOMAIN_ID="
        f"{os.environ.get('ROS_DOMAIN_ID', '0')} ...",
        flush=True,
    )
    node.recording, node.samples = True, []
    spin(node, 4.0)
    seen, node.samples, node.recording = list(node.samples), [], False
    if not seen:
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(
            "  No pose on /zed/zed_node/pose for this ROS_DOMAIN_ID.\n"
            f"  Refusing to POST to port {TELEPORT_PORT}: that endpoint is not\n"
            "  domain-scoped, so it may belong to a different simulation.\n"
            "  Start the sim on this domain first (./validate.sh ...)."
        )

    print(
        f"\n  step-steer @ {args.speed:.2f} m/s, TICK={TICK * 1000:.0f} ms command rate\n"
    )
    results = []
    raw = {
        "speed": np.array(args.speed),
        "pre": np.array(args.pre),
        "post": np.array(args.post),
    }
    for cmd in args.commands:
        got = run_one(
            node, cfg, args.x, args.y, args.speed, cmd, args.settle, args.pre, args.post
        )
        if got is None:
            print(f"  cmd {cmd:+.2f}   too few samples, skipped")
            continue
        t_r, yaw_r = got
        raw[f"t_{cmd:+.2f}"] = t_r
        raw[f"yaw_{cmd:+.2f}"] = yaw_r
        tg_r, rate_r = smoothed_rate(t_r, yaw_r)

        tg_m, rate_m = model_trace(cfg, args.speed, cmd, args.pre, args.post)
        rate_m_on_grid = np.interp(tg_r, tg_m, rate_m)

        mask_ss = tg_r > (args.post - 0.4)  # last 0.4s = steady state
        ss_r = np.mean(rate_r[mask_ss])
        ss_m = np.mean(rate_m_on_grid[mask_ss])
        ss_ratio = ss_r / ss_m if abs(ss_m) > 1e-6 else float("nan")

        print(
            f"  cmd {cmd:+.2f}   steady yaw rate  gazebo {ss_r:+.3f}  "
            f"model {ss_m:+.3f} rad/s   ratio {ss_ratio:.3f}"
        )
        for frac in (0.2, 0.5):
            tc_r = crossing_time(tg_r, rate_r, frac, ss_r)
            tc_m = crossing_time(tg_m, rate_m, frac, ss_m)
            extra = (
                tc_r - tc_m if np.isfinite(tc_r) and np.isfinite(tc_m) else float("nan")
            )
            print(
                f"    time to {frac * 100:.0f}% of steady:  gazebo {tc_r:.3f}s  "
                f"model {tc_m:.3f}s   EXTRA DELAY {extra:+.3f}s"
            )
        print(f"    {'t (s)':>7} {'gazebo':>9} {'model':>9} {'ratio':>7}")
        checkpoints = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, args.post - 0.2]
        row = []
        for tc in checkpoints:
            if tc <= 0 or tc >= tg_r[-1]:
                continue
            gr = np.interp(tc, tg_r, rate_r)
            mr = np.interp(tc, tg_m, rate_m)
            ratio = gr / mr if abs(mr) > 1e-6 else float("nan")
            row.append((tc, gr, mr, ratio))
            print(f"    {tc:7.2f} {gr:9.3f} {mr:9.3f} {ratio:7.3f}")
        results.append((cmd, ss_ratio, row))
        print()

    node.destroy_node()
    rclpy.shutdown()

    if args.save is not None and len(raw) > 3:
        np.savez(args.save, **raw)
        print(f"  raw traces -> {args.save}\n")

    if not results:
        return
    print("  --- summary ---")
    print(
        f"  steady-state ratio (this run):    "
        f"{np.mean([r[1] for r in results]):.3f}  "
        f"(measure_turn_radius.py found ~1.10 median)"
    )
    early_ratios = []
    for cmd, ss_ratio, row in results:
        for tc, gr, mr, ratio in row:
            if tc <= 0.20 and np.isfinite(ratio):
                early_ratios.append(ratio)
    if early_ratios:
        print(f"  turn-in ratio (t<=0.20s, this run): {np.mean(early_ratios):.3f}")
        print(
            "  turn-in >> steady-state  ==>  extra transient understeer, "
            "tire_scrub undercorrects turn-in"
        )
        print(
            "  turn-in ~= steady-state  ==>  scrub already explains "
            "turn-in; the Gazebo regression has some other cause"
        )


if __name__ == "__main__":
    main()
