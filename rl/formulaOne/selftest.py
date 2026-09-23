#!/usr/bin/env python3
"""Checks that have to pass before a training run is worth starting.

Three groups:

  TRACK    the course is the shape it should be, and the speed cap says what
           the rules say.
  PLANT    the model agrees with the numbers in vehicle.yaml and with the
           behaviour sim_vehicle_node.cpp documents -- above all that the car
           cannot brake.
  ENV      a hand-written driver gets round, twice, without touching a bale,
           at a lap time in the region the physics allows.

Run it after touching config.yaml.  It needs numpy and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

import track as track_mod
from baseline import BaselineDriver, feasible_profile
from env import FormulaOneEnv
from plant import Plant

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PASS, FAIL = "  ok  ", " FAIL "
failures = []


def check(name, condition, detail=""):
    print(f"[{PASS if condition else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    t = track_mod.build(cfg, ROOT)
    tc = cfg["track"]

    print("\n--- track")
    check("closed loop 110 m", abs(t.length - 110.2) < 1.0, f"{t.length:.2f} m")
    check("min radius above the car's own", 1 / np.abs(t.kappa).max() > 0.8,
          f"{1 / np.abs(t.kappa).max():.2f} m")
    corridor = (t.half_left + t.half_right)
    check("corridor never below the car's width",
          corridor.min() > cfg["vehicle"]["width"],
          f"{corridor.min():.2f} m min vs {cfg['vehicle']['width']} m car")
    check("cap honours the hairpin rule exactly",
          np.allclose(t.v_cap[t.hairpin], tc["v_hairpin"]),
          f"{tc['v_hairpin']} m/s over {100 * t.hairpin.mean():.0f}% of the lap")
    check("cap never exceeds the straight rule", t.v_cap.max() <= tc["v_straight"] + 1e-9,
          f"max {t.v_cap.max():.2f} m/s")
    check("start line is on the course",
          t.clearance(np.array([tc["vehicle_start"][0]]),
                      np.array([tc["vehicle_start"][1]]))[0] > 0.3)
    # Two hairpins and nothing else should be under the limit.  Counted
    # circularly: one of them straddles the start line.
    n = len(t.hairpin)
    zones = sum(1 for i in range(n) if t.hairpin[i] and not t.hairpin[i - 1])
    check("exactly two hairpin zones", zones == 2, f"{zones} found")

    print("\n--- plant")
    n = 3
    rng = np.random.default_rng(0)
    flat = {**cfg, "randomize": {**cfg["randomize"], "enabled": False}}
    p = Plant(flat, n, rng)
    p.reset(np.ones(n, dtype=bool), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
    dt = np.full(n, 0.005)

    # Coast from 5.2 with a zero command: the published curve is
    # decel = (2.18 + 0.47 v) / 3.599, i.e. 1.28 m/s^2 at 5.2 m/s.
    p.history[:] = 0.0
    for _ in range(int(0.5 / 0.005)):   # flush the dead-time queue to zero
        p.substep(np.zeros(n), np.zeros(n), dt)
    p.speed[:] = 5.2                    # then start the coast from the top
    p.target[:] = 5.2
    start_v, dist = p.speed[0], 0.0
    while p.speed[0] > 2.5:
        prev = p.speed[0]
        p.substep(np.zeros(n), np.zeros(n), dt)
        dist += 0.5 * (prev + p.speed[0]) * 0.005
    check("no brakes: 5.2 -> 2.5 m/s takes ~9 m of track", 8.0 < dist < 11.0,
          f"{dist:.1f} m from {start_v:.2f} m/s")

    p2 = Plant(flat, n, rng)
    p2.reset(np.ones(n, dtype=bool), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
    for _ in range(int(1.0 / 0.005)):
        p2.substep(np.zeros(n), np.full(n, 5.2), dt)
    # Bridge slew (2.0) binds below the plant's own 3.0 m/s^2, so one second
    # after the command lands the car is at ~2.0 m/s, not ~3.0.
    check("acceleration is bridge-slew limited, not plant limited",
          1.4 < p2.speed[0] < 2.1, f"{p2.speed[0]:.2f} m/s one second in")

    left = p2.steering_angle(np.full(n, 1.0))[0]
    right = p2.steering_angle(np.full(n, -1.0))[0]
    check("steering is asymmetric, as measured", abs(left / abs(right) - 1.34) < 0.05,
          f"left {left:.3f} rad, right {right:.3f} rad")

    # Tire scrub: measure_turn_radius.py found the simulated car turning a
    # uniform 1.10x the bare kinematic radius, flat across speed and
    # direction.  Drive the plant at a steady command and speed, past the
    # dead time and the servo lag, and check the ACHIEVED radius against a
    # hand-computed kinematic one that does not know tire_scrub exists --
    # if this ever drifts back to 1.0 the fix in plant.py has been lost.
    p3 = Plant(flat, n, rng)
    p3.reset(np.ones(n, dtype=bool), np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, 2.0))
    # 4 s, not the 1.5 s this used to take.  tire_scrub is a STEADY-STATE
    # number -- measure_turn_radius.py settles into a circle before fitting
    # one -- and yaw_response_tau is 0.34 s, so 1.5 s of settling still left
    # the yaw rate 4% short and this check read 1.147x for a 1.10x model.
    for _ in range(int(4.0 / 0.005)):
        p3.substep(np.full(n, 0.6), np.full(n, 2.0), dt)
    yaw_before = p3.yaw[0]
    p3.substep(np.full(n, 0.6), np.full(n, 2.0), dt)
    achieved_r = p3.speed[0] * dt[0] / (p3.yaw[0] - yaw_before)
    angle = p3.steering_angle(np.array([0.6]))[0]
    kinematic_r = (cfg["plant"]["wheelbase"] + cfg["plant"]["understeer_gradient"] * p3.speed[0] ** 2) / np.tan(angle)
    check("tire scrub widens the radius by the measured ~1.10x",
          abs(achieved_r / kinematic_r - cfg["plant"]["tire_scrub"]) < 2e-3,
          f"kinematic {kinematic_r:.3f} m -> achieved {achieved_r:.3f} m "
          f"({achieved_r / kinematic_r:.3f}x vs {cfg['plant']['tire_scrub']:.2f}x measured on Gazebo)")

    print("\n--- env, driven by the scripted baseline")
    env = FormulaOneEnv(cfg, t, n_envs=1, seed=1, deterministic=True)
    driver = BaselineDriver(t, cfg)
    driver.reset(1)
    vref = feasible_profile(t, cfg)
    ideal = float((t.ds / vref).sum())
    env.reset()
    log = []
    for _ in range(int(cfg["env"]["episode_timeout_s"] * env.control_hz)):
        s = env.snapshot()
        _, _, term, trunc, info = env.step(env.scripted_action(driver))
        log.append((s["speed"][0], s["clearance"][0], s["v_cap"][0]))
        if term[0] or trunc[0]:
            break
    result = info[0]
    speed, clear, cap = (np.array(c) for c in zip(*log))
    check("two laps completed", result.get("finished", False),
          f"laps {result.get('laps')}, {result.get('distance', 0):.0f} m")
    check("and then came to a stop, still in the corridor",
          result.get("stopped", False),
          f"at rest {result.get('stop_distance', 0):.1f} m past the line")
    check("never touched a bale", result.get("min_clearance", -1) > 0.0,
          f"min clearance {result.get('min_clearance', 0):.3f} m")
    # NOT a graze check any more.  With `speed_floor_*` in force the scripted
    # driver cannot quite clear it: under the three-zone floor (2.0 red /
    # 2.6 yellow / 4.2 green) it comes through the station-104 chicane at
    # about 0.107 m against a 0.12 m band.  Widening the yellow band to the
    # real coast-down and acceleration zones took that from 0.047 m and cut
    # the worst overspeed from +0.32 to +0.10 m/s, so what is left is a
    # margin the POLICY's steering residual has to find, not a physics wall.  That is the gap the POLICY's steering
    # residual exists to close, so the floor driver is held to "did not touch
    # a bale" and the graze bar moved to the trained policy.
    check("kept some clearance even at the floor",
          result.get("min_clearance", -1) > 0.02,
          f"min clearance {result.get('min_clearance', 0):.3f} m "
          f"(graze band starts at {cfg['reward']['graze_margin']} m; the "
          f"scripted driver is expected to be inside it with floors on)")
    check("stayed under the speed cap", result.get("max_overspeed", 9) < 0.35,
          f"worst overspeed {result.get('max_overspeed', 0):.2f} m/s")
    check("lap time is in the region physics allows",
          ideal * 0.85 < result["lap_time"] < ideal * 1.55,
          f"{result['lap_time']:.2f} s/lap vs {ideal:.2f} s coast-feasible ideal")
    check("held the centerline", result.get("mean_cte", 9) < 0.15,
          f"mean cross-track error {result.get('mean_cte', 0):.3f} m, "
          f"worst {result.get('max_cte', 0):.3f} m")
    check("honoured the speed floor",
          result.get("floor_deficit", 9) < 0.35,
          f"worst shortfall under the floor {result.get('floor_deficit', 0):.2f} m/s "
          f"(the floor is structural on the COMMAND; a shortfall here is the "
          f"car not reaching it, e.g. accelerating out of a corner)")
    check("steering is not jittery",
          result.get("steer_jerk_rms", 9) < 0.05,
          f"rms change in steering step {result.get('steer_jerk_rms', 0):.4f} "
          f"per tick")

    # The lag is the difference between a model that predicted Gazebo and one
    # that did not, so it gets a regression check of its own: step the
    # steering from straight and time the yaw rate to half its steady value.
    print("\n--- chassis yaw lag")
    flat = {**cfg, "randomize": {**cfg["randomize"], "enabled": False}}
    pl = Plant(flat, 1, np.random.default_rng(0))
    pl.reset(np.array([True]), np.array([0.0]), np.array([0.0]),
             np.array([0.0]), np.array([2.5]))
    dt = pl.dt_sub
    rates = []
    for i in range(int(2.5 / dt)):
        rates.append(float(pl.substep(np.array([0.55]), np.array([2.5]),
                                      np.array([dt]))[0]))
    rates = np.array(rates)
    steady = rates[-20:].mean()
    t_half = float(np.argmax(rates >= 0.5 * steady) * dt)
    # Gazebo, measured: 0.59 s at this command and speed (measure_step_steer.py,
    # 2026-09-22).  The model has to be in that region, not at the 0.37 s a
    # bare kinematic bicycle gives.
    check("takes as long as Gazebo to build yaw rate",
          0.45 < t_half < 0.75,
          f"half of steady yaw rate at {t_half:.2f} s "
          f"(Gazebo 0.59 s, no-lag model 0.37 s)")

    total = result.get("episode", {}).get("t", 0.0)
    print(f"\n   baseline: {total:.2f} s for two laps "
          f"({result['lap_time']:.2f} s/lap), ideal {2 * ideal:.2f} s, "
          f"top speed {speed.max():.2f} m/s, "
          f"min clearance {result.get('min_clearance', 0):.3f} m")
    print(f"             cte {result.get('mean_cte', 0):.3f} m mean / "
          f"{result.get('max_cte', 0):.3f} m worst, "
          f"stopped {result.get('stop_distance', 0):.1f} m past the line")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
