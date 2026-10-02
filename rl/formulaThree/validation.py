"""Strict simulation verdict, shared by validate.sh and offline regression checks."""

import json
import re
import sys
import time

import numpy as np
import yaml


def verdict(monitor, log, laps, limits, stop_speed=0.2):
    reasons = []
    if not re.search(rf"FINISHED {laps} laps in [\d.]+ s", log):
        reasons.append("lap target not reached")
    if "STOPPED" not in log or "TIMED OUT" in log:
        reasons.append("stop not confirmed")
    if "DEPTH LOST" in log or "ABANDONED" in log or monitor.get("depth_lost"):
        reasons.append("depth lost or run abandoned")
    clearance = monitor.get("min_clearance")
    if (
        not monitor.get("samples")
        or clearance is None
        or not np.isfinite(clearance)
        or clearance <= limits["min_clearance"]
    ):
        reasons.append("insufficient clearance or missing monitor data")
    for key in ("contact_samples", "pose_jumps", "clock_reversals"):
        if monitor.get(key, 0):
            reasons.append(key)
    if monitor.get("max_abs_roll_deg", 0) > limits["max_roll_deg"]:
        reasons.append("excessive roll")
    if not monitor.get("speed_samples"):
        reasons.append("missing speed measurements")
    elif (
        not np.isfinite(monitor.get("max_overspeed", float("inf")))
        or monitor.get("max_overspeed", float("inf")) > limits["max_overspeed"]
    ):
        reasons.append("speed limit exceeded")
    for key in ("last_pose_wall_time", "last_speed_wall_time"):
        stamp = monitor.get(key)
        if (
            stamp is None
            or not np.isfinite(stamp)
            or not -1 <= time.time() - stamp <= 5
        ):
            reasons.append(f"stale monitor {key}")
    speed = monitor.get("last_speed")
    if speed is None or not np.isfinite(speed) or abs(speed) > stop_speed:
        reasons.append("vehicle not measured at rest")
    return reasons


def main():
    mon_path, log_path, laps, cfg_path = sys.argv[1:]
    try:
        with open(mon_path) as f:
            monitor = json.load(f)
    except (OSError, ValueError):
        monitor = {}
    with open(log_path) as f:
        log = f.read()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    reasons = verdict(
        monitor, log, int(laps), cfg["validation"], cfg["env"]["stop_speed"]
    )
    result = dict(
        passed=not reasons,
        reasons=reasons,
        monitor=monitor,
        lap_times=re.findall(r"lap (\d+) of \d+\s+([\d.]+) s", log),
    )
    with open(mon_path + ".verdict.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return bool(reasons)


if __name__ == "__main__":
    sys.exit(main())
