#!/usr/bin/env python3
"""Drive the lap environment against the real simulator, with no policy.

`lap_env_selftest.py` proves the bookkeeping on a kinematic stand-in. This
proves the same environment against Gazebo -- real contact, real tire
behaviour, a real-time factor below 1, and (with CFR_SENSORS=1) observations
built from the rendered ZED cloud rather than ray-casts.

It answers three questions a training run should not be the first to ask:

  1. does a lap complete at all, and in what time;
  2. does the ZED cloud actually arrive at the control rate;
  3. what does the reward pay, term by term, over a real lap.

The lap time it reports is also the floor a trained policy should beat: pure
pursuit on the planned line with no braking plan and no idea the bales exist.

    ./lap_live_check.sh --laps 2
    CFR_SENSORS=1 ./lap_live_check.sh --laps 2 --scan-source cloud
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from lap_driver import PursuitDriver
from lap_env import LapRacerEnv
from lap_reward import LapRewardConfig
from zed_sim import ZedSimConfig

HERE = Path(__file__).resolve().parent
DEFAULT_SDF = HERE.parents[1] / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config_lap.yaml"))
    parser.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    parser.add_argument("--teleport-url",
                        default="http://localhost:9003/api/sim/teleport")
    parser.add_argument("--laps", type=int, default=2)
    parser.add_argument("--target-speed", type=float, default=3.2)
    parser.add_argument("--lookahead", type=float, default=1.2)
    parser.add_argument("--plan-scale", type=float, default=1.0,
                        help="fraction of the plan's min-time speed to ask for")
    parser.add_argument("--no-plan-speed", action="store_true",
                        help="ignore the plan's speed profile (the driver then "
                             "arrives at hairpins at straight-line speed)")
    parser.add_argument("--scan-source", default="analytic",
                        choices=["analytic", "cloud"])
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    env_config = dict(config["env"])
    env_config.update(scan_source=args.scan_source, randomize_start=False,
                      wedged_start_prob=0.0, max_laps=args.laps,
                      episode_time_limit_s=args.time_limit)

    environment = LapRacerEnv(
        sdf_path=args.sdf_path,
        teleport_url=args.teleport_url,
        lap_reward_config=LapRewardConfig(**config["reward"]),
        zed_config=ZedSimConfig(**config.get("zed_sim", {})),
        **env_config,
    )
    driver = PursuitDriver(environment, args.lookahead, args.target_speed,
                           use_plan_speed=not args.no_plan_speed,
                           plan_scale=args.plan_scale)

    try:
        environment.reset()
        totals: dict[str, float] = {}
        speeds, clearances, rates = [], [], []
        reward_total = 0.0
        steps = 0
        info: dict = {}
        while True:
            pose = environment._prev_pose
            action = driver.action(pose.x, pose.y, pose.yaw)
            _, reward, terminated, truncated, info = environment.step(action)
            reward_total += reward
            for key, value in info["reward_terms"].items():
                totals[key] = totals.get(key, 0.0) + value
            speeds.append(info["speed"])
            clearances.append(info["min_clearance"])
            rates.append(info["steer_rate"])
            steps += 1
            if info["lap_time"] is not None:
                print(f"  lap {info['laps']}: {info['lap_time']:.2f} s", flush=True)
            if terminated or truncated:
                break

        result = {
            "scan_source": args.scan_source,
            "steps": steps,
            "sim_seconds": round(environment._episode_time, 1),
            "laps": info["laps"],
            "lap_times": [round(t, 2) for t in info["lap_times"]],
            "s_progress_m": round(info["s_progress"], 1),
            "mean_speed": round(statistics.fmean(speeds), 2),
            "min_body_clearance_m": round(min(clearances), 3),
            "mean_steer_rate_rad_s": round(statistics.fmean(rates), 3),
            "collided": info["collided"],
            "stuck": info["stuck"],
            "reward_total": round(reward_total, 1),
            "reward_terms": {k: round(v, 1) for k, v in sorted(totals.items())},
            "cloud_frames": environment._cloud_hits,
            "cloud_misses": environment._cloud_misses,
        }
        print(json.dumps(result, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
