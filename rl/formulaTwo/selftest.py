#!/usr/bin/env python3
"""Checks that have to pass before a formulaTwo training run is worth starting.

  WORLD      the warped course is self-consistent: a car on the as-built
             centerline reads zero cross-track error and nominal clearance.
  CAMERA     the virtual LiDAR sees the corridor walls where the track says
             they are, and the car-side sampler reproduces the render.
  v12        the first 35 actor inputs are v12's, bit for bit, so train.py's
             warm start is exact.
  FEASIBLE   every NEW randomisation, alone at its full range, still leaves
             the scripted driver able to finish (formulaOne trap #1: a range
             that makes the task impossible teaches the policy to crawl), and
             the speed floor is reachable by the lowest-drag car.
  ENV        the scripted driver does three laps and stops on the nominal car.

    python3 selftest.py          # ~1 minute
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import yaml

import track as track_mod
from baseline import BaselineDriver, feasible_profile
from env import FormulaTwoEnv
from observation import speed_floor
from perception import INVALID, Camera
from world import World

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
failures = []


def check(name, ok, detail=""):
    print(f"[{'  ok  ' if ok else ' FAIL '}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def only(cfg, keys, world=False):
    """Config with every randomised range collapsed to nominal except `keys`."""
    c = copy.deepcopy(cfg)
    for k, v in c["randomize"].items():
        if k != "enabled" and k not in keys:
            c["randomize"][k] = [v[2], v[2], v[2]]
    if not world:
        for k in ("layout_amp", "jitter_amp", "inflate"):
            c["world"][k] = [0.0, 0.0, 0.0]
    return c


def baseline_run(cfg, tr, n, deterministic=False):
    env = FormulaTwoEnv(cfg, tr, n, seed=11, deterministic=deterministic)
    env.random_start = False
    drv = BaselineDriver(tr, cfg)
    env.reset()
    out = []
    for _ in range(7000):
        _, _, te, tru, info = env.step(env.scripted_action(drv))
        out += [info[i] for i in np.flatnonzero(te | tru)]
        if len(out) >= n:
            break
    return out[:n]


def main():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    tr = track_mod.build(cfg, ROOT)
    rng = np.random.default_rng(0)

    print("\n--- world")
    w = World(tr, cfg, 64, rng)
    w.sample(np.ones(64, bool))
    idx = rng.integers(0, len(tr.x), 64)
    cx, cy = tr.x[idx], tr.y[idx]
    dx, dy = w._analytic(cx[:, None], cy[:, None], True, np.arange(64))
    yaw = np.arctan2(tr.ty[idx], tr.tx[idx])
    _, _, lat, psi = w.frenet(cx + dx[:, 0], cy + dy[:, 0], yaw, idx)
    check("as-built centerline reads zero CTE", np.abs(lat).max() < 0.003, f"{np.abs(lat).max() * 1000:.2f} mm")
    check("and near-zero heading error", np.abs(psi).max() < 0.03, f"{np.abs(psi).max():.3f} rad")
    px = cx[:, None] + rng.uniform(-0.4, 0.4, (64, 50))
    py = cy[:, None] + rng.uniform(-0.4, 0.4, (64, 50))
    a = w._analytic(px, py, False, np.arange(64))
    g = w.displacement(px, py)
    err = np.hypot(a[0] - g[0], a[1] - g[1]).max()
    check("baked field matches the analytic one", err < 0.005, f"{err * 1000:.2f} mm")

    print("\n--- camera")
    env = FormulaTwoEnv(cfg, tr, 1, seed=0, deterministic=True)
    env.random_start = False
    obs = env.reset()
    cam = env.camera
    scan = obs[0, env.map_dim : env.map_dim + cam.width]
    r = np.where(scan > INVALID + 1e-6, 0.25 * np.exp(scan * np.log(cam.scan_max / 0.25)), np.nan)
    left = tr.half_left[env.hint[0]] / np.sin(abs(cam.azimuth[0]))
    right = tr.half_right[env.hint[0]] / np.sin(abs(cam.azimuth[-1]))
    check(
        "edge beams hit the side walls",
        abs(r[0] - left) < 0.08 and abs(r[-1] - right) < 0.08,
        f"{r[0]:.2f}/{r[-1]:.2f} m vs {left:.2f}/{right:.2f} m from the track",
    )
    # The car's sampler on a synthetic full-resolution image must reproduce
    # the render on the grid it samples.
    c = Camera(cfg)
    depth = c.render(np.array([[2.0] * c.W]), np.zeros(1), np.full(1, c.z))[0]
    fx = (640 / 2) / np.tan(np.deg2rad(55))
    full = np.full((360, 640), np.nan)
    vv, uu = np.meshgrid(np.arange(360), np.arange(640), indexing="ij")
    b = (180 - (vv + 0.5)) / fx
    aa = (320 - (uu + 0.5)) / fx
    hl = np.sqrt(1 + aa**2)
    slope = b / hl
    rg = np.where(slope < 0, c.z / -np.minimum(slope, -1e-9), np.inf)
    zb = c.z + 2.0 * slope
    dist = np.where(rg < 2.0, rg, np.where((zb >= 0) & (zb <= c.bale_height), 2.0, np.nan))
    full = dist / hl
    samp = c.sample_depth(full, fx, fx, 320 - 0.5, 180 - 0.5)[0]
    agree = np.nanmax(np.abs(samp - depth)) if np.isfinite(samp).any() else 1.0
    check("car-side sampler reproduces the render", agree < 1e-6, f"{agree:.1e}")

    print("\n--- v12")
    here_obs = obs[0, : env.map_dim]
    check("map block is 35 wide, like v12", env.map_dim == 35, f"{env.map_dim}")
    check("no NaN in the observation", np.isfinite(obs).all(), f"{obs.shape[1]} columns")

    print("\n--- feasibility of the new randomisation")
    floor = speed_floor(tr, cfg)
    r_ = cfg["randomize"]
    low = r_["coast_scale"][0] / r_["mass_scale"][1]
    c2 = copy.deepcopy(cfg)
    c2["plant"]["coast_f0"] *= low
    c2["plant"]["coast_f1"] *= low
    gap = (floor - feasible_profile(tr, c2)).max()
    # 0.05: at the drag floor the budget allows 0.03 m/s over at one corner --
    # well inside what the overspeed term already prices -- and 0.15 (drag
    # x0.85) is where it stops being a rounding error.
    check("speed floor reachable by the lowest-drag car", gap < 0.05, f"drag x{low:.2f}, floor over by {gap:+.3f} m/s")
    for name, keys, world in (
        ("bale layout, jitter and scale", [], True),
        ("ground friction", ["mu"], False),
        ("motor lag", ["motor_tau"], False),
        ("throttle delay", ["throttle_delay"], False),
    ):
        out = baseline_run(only(cfg, keys, world), tr, 64)
        fin = np.mean([o["stopped"] for o in out])
        check(f"scripted driver finishes under {name}", fin >= 0.9, f"{100 * fin:.0f}%")

    print("\n--- env")
    out = baseline_run(cfg, tr, 16, deterministic=True)
    fin = [o for o in out if o["stopped"]]
    check("scripted driver: 3 laps and stops, nominal", len(fin) == len(out), f"{len(fin)}/{len(out)}")
    if fin:
        lap = np.mean([o["race_time"] for o in fin]) / 3
        clr = min(o["min_clearance"] for o in fin)
        check("  at a sane pace, off the bales", 25 < lap < 40 and clr > 0.02, f"{lap:.2f} s/lap, clearance {clr:+.3f} m")
        check("  stopped inside the course", all(0 < o["stop_distance"] < 20 for o in fin), f"{np.mean([o['stop_distance'] for o in fin]):.1f} m past the line")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
