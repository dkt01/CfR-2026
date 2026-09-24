#!/usr/bin/env python3
"""Checks of the analyser's numbers against cases with known answers.

    web/run-lab/.venv/bin/python web/run-lab/server/selftest.py

No bag, no ROS: each check builds the input it needs.  These are the parts a
wrong answer from would be believed -- a lag fit that reports the car's
chassis lag, a frame fit that places every clearance, the lap bookkeeping --
so each is held to a case where the truth is known.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
import course  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'  ok  ' if ok else ' FAIL '}] {name}   {detail}")
    if not ok:
        FAILED.append(name)


def steering_chain(t, sp, cmd, dead, servo_tau, yaw_tau, dt=0.005):
    """plant.py's steering path, integrated finely: dead time, servo lag,
    chassis lag.  Table and wheelbase terms folded into `cmd` directly."""
    out = np.zeros(len(t))
    sub = int(round(analyze.GRID_HZ**-1 / dt))
    k = int(round(dead * analyze.GRID_HZ))
    u = np.concatenate([np.full(k, cmd[0]), cmd[:-k]]) if k else cmd
    a = yr = 0.0
    for i in range(len(t)):
        for _ in range(sub):
            a += dt / (servo_tau + dt) * (u[i] - a)
            yr += dt / (yaw_tau + dt) * (sp[i] * a - yr)
        out[i] = yr
    return out


def test_lag_fit():
    rng = np.random.default_rng(3)
    t = np.arange(0, 60, 1 / analyze.GRID_HZ)
    sp = 2.5 + 0.8 * np.sin(t / 7)
    # A command that looks like driving: held segments joined by ramps.
    knots = rng.uniform(-0.6, 0.6, 40)
    cmd = np.interp(t, np.linspace(0, 60, 40), knots)
    for yaw_tau in (0.2, 0.34, 0.55):
        y = steering_chain(t, sp, cmd, dead=0.19, servo_tau=0.05, yaw_tau=yaw_tau)
        y *= np.where(cmd > 0, 0.9, 1.1)  # asymmetric authority, known
        fit = analyze.fit_lag(t, sp * cmd, y)
        total = fit["dead_time_s"] + fit["tau_s"]
        truth = 0.19 + 0.05 + yaw_tau
        check(
            f"lag fit recovers total delay (yaw tau {yaw_tau})",
            abs(total - truth) <= 0.08,
            f"fitted {total:.2f} s, true {truth:.2f} s",
        )
        check(
            f"lag fit recovers side gains (yaw tau {yaw_tau})",
            abs(fit["gain_left"] - 0.9) < 0.06 and abs(fit["gain_right"] - 1.1) < 0.06,
            f"L {fit['gain_left']:.3f} (0.9)  R {fit['gain_right']:.3f} (1.1)",
        )


def test_piecewise_transform():
    # Two anchors: identity until t=10, then rotated 30 deg and shifted.
    pieces = [(0.0, 0.0, 0.0, 0.0), (10.0, math.radians(30), 2.0, -1.0)]
    transform = (0.0, 0.0, 0.0, "test", 0.0, pieces)
    x = np.array([1.0, 1.0])
    y = np.array([0.0, 0.0])
    tx, ty = analyze.apply_xy(transform, x, y, np.array([5.0, 15.0]))
    check(
        "anchor before the re-anchor is used before it",
        abs(tx[0] - 1.0) < 1e-9 and abs(ty[0]) < 1e-9,
    )
    ok = (
        abs(tx[1] - (math.cos(math.radians(30)) + 2)) < 1e-9
        and abs(ty[1] - (math.sin(math.radians(30)) - 1)) < 1e-9
    )
    check(
        "anchor after the re-anchor is used after it", ok, f"({tx[1]:.3f}, {ty[1]:.3f})"
    )


def test_rigid_fit():
    rng = np.random.default_rng(1)
    src = rng.uniform(-10, 10, (200, 2))
    th = 1.1
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    dst = src @ rot.T + [3.0, -4.0] + rng.normal(0, 0.01, (200, 2))
    theta, trans, rms = analyze.rigid_fit(src, dst)
    check(
        "rigid fit recovers rotation and translation",
        abs(theta - th) < 1e-3 and np.allclose(trans, [3, -4], atol=0.01),
        f"theta {theta:.4f} rms {rms:.4f}",
    )


def test_laps():
    cfg = course.load_config()
    trk = course.get_track(cfg)
    L = trk.length
    t = np.arange(0, 80, 0.05)
    # Constant 3 m/s from the start station, stop after 2.2 laps.
    dist = np.minimum(3.0 * t, 2.2 * L)
    station = (trk.start_station + dist) % L
    cols = {
        "speed": np.gradient(dist, t),
        "clearance": np.full(len(t), 0.2),
        "cte": np.zeros(len(t)),
        "laps_target": np.full(len(t), 2.0),
    }
    info = analyze.laps_from_station(t, station, trk, np.ones(len(t), bool), cols)
    check(
        "counts completed laps at the start line",
        info["completed"] == 2,
        f"{info['completed']} laps",
    )
    check(
        "lap time is length / speed",
        abs(info["laps"][0]["time"] - L / 3.0) < 0.06,
        f"{info['laps'][0]['time']} s vs {L / 3:.2f}",
    )
    check(
        "finished flag and race time",
        info["finished"] and abs(info["race_time"] - 2 * L / 3.0) < 0.1,
        f"{info['race_time']} s",
    )


def test_hold_and_gaps():
    ts = np.array([0.0, 1.0, 2.0])
    vs = np.array([1.0, 2.0, 3.0])
    out = analyze.hold(ts, vs, np.array([-0.5, 0.5, 1.5, 9.0]))
    check(
        "zero-order hold with NaN before the first sample",
        np.isnan(out[0]) and list(out[1:]) == [1.0, 2.0, 3.0],
    )
    out = analyze.interp(
        np.array([0.0, 0.1, 5.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.05, 2.0]),
        max_gap=0.5,
    )
    check(
        "interpolation refuses to bridge a pose gap", out[0] == 0.5 and np.isnan(out[1])
    )


def test_rerun_export():
    """The recording the Replay page embeds: written, readable by the same
    Rerun version's own verifier, and carrying the entities the layout names."""
    import subprocess
    import tempfile

    import rerun_export

    cfg = course.load_config()
    geometry = course.geometry(cfg)
    t = np.arange(0, 5, 0.05)
    cols = {
        "t": t.tolist(),
        "x": (20 - t).tolist(),
        "y": [4.75] * len(t),
        "yaw": [math.pi] * len(t),
        "speed": [1.0] * len(t),
        "clearance": [0.2] * len(t),
    }
    series = {"columns": cols}
    summary = {
        "events": [
            {"t": 1.0, "kind": "graze", "severity": "warn", "text": "x", "station": 1.0}
        ],
        "window": {"t0": 0.0, "t1": 5.0},
        "verdicts": {"failed": []},
    }
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.rrd"
        rec = rerun_export.RunRecording(path, "selftest")
        rec.course(geometry)
        rec.car(series, geometry, summary["window"])
        rec.metrics(series)
        rec.events(summary, series)
        rec.cloud_frame(1.0, np.zeros((10, 3), np.float32), np.zeros((10, 3), np.uint8))
        rec.close()
        rerun = Path(sys.executable).with_name("rerun")
        verify = subprocess.run(
            [str(rerun), "rrd", "verify", str(path)], capture_output=True, text=True
        )
        check(
            "Rerun recording verifies",
            verify.returncode == 0,
            (verify.stdout + verify.stderr).strip().splitlines()[-1]
            if (verify.stdout + verify.stderr).strip()
            else "",
        )
        printed = subprocess.run(
            [str(rerun), "rrd", "print", str(path)], capture_output=True, text=True
        ).stdout
        missing = [
            e
            for e in (
                "/world/course/bales",
                "/world/car",
                "/world/path",
                "/metrics/speed/measured",
                "/world/cloud/frame",
                "/events",
            )
            if e not in printed
        ]
        check(
            "Rerun recording has the laid-out entities",
            not missing,
            f"missing {missing}" if missing else "",
        )


def main():
    test_rerun_export()
    test_lag_fit()
    test_piecewise_transform()
    test_rigid_fit()
    test_laps()
    test_hold_and_gaps()
    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
