#!/usr/bin/env python3
"""Drive rl/obstacleRacer's numpy sim of the Obstacle Course with Plan Z.

Development only: the fast way to see whether a change to the planner or a
knob still gets round every layout, before spending Gazebo time on it.  The
env supplies what the car's node would: the delivered scan and gates (12 Hz,
50-100 ms old), tachometer speed, yaw rate, and the pose.  One start-box lap
per car, or one obstacle at a time with --start.

It is harsher than Gazebo, and its numbers are for comparing one change with
another, not for saying how often the car gets round: its throttle dithers
below 1 m/s as the bench Arduino's does, where Gazebo's holds a speed, so
backing and filling between bales takes several times as many tries.

    python3 sim_numpy.py                               # 4 Gazebo seeds x 2
    python3 sim_numpy.py --start hoops --limit 60      # from just before the hoops
    python3 sim_numpy.py --start wide_open_region --beyond 32 --limit 150
    python3 sim_numpy.py --seeds heldout --starts 1    # the 32 held-out layouts
    python3 sim_numpy.py --seeds grade                 # 64 never-seen layouts
    python3 sim_numpy.py --steer-bias 0.035 --camera-yaw -3
    python3 sim_numpy.py speed_scale=1.2 sections.helical_ramp=0.8
    python3 sim_numpy.py --seeds 208 --plot /tmp/208.png --trace

The faults are put into the simulated car, not the driver: --steer-bias and
--steer-gain set the plant's steering offset and gain, --camera-yaw/-pitch/
-roll turn the camera the sensor model renders from.  --pose-drift and
--yaw-drift corrupt the pose the driver is given.  --randomize draws the
plant as training does (dead time, drag, lags, grip).

Run with rl/obstacleRacer's environment (numba).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RACER = HERE.parents[1] / "rl" / "obstacleRacer"
sys.path.insert(0, str(RACER))
sys.path.insert(0, str(HERE))

import course_model  # noqa: E402
import layouts  # noqa: E402
import observation as O  # noqa: E402
import plant as P  # noqa: E402
import sensor as S  # noqa: E402
from env import ObstacleEnv  # noqa: E402

from planner import CARWASH, HOOP, Planner, PoseDrift  # noqa: E402
from route import Route, load_config  # noqa: E402

SEED_SETS = {
    "gazebo": layouts.HELDOUT_SLOT_SEEDS,
    "heldout": layouts.HELDOUT_SEEDS,
    "grade": layouts.GRADE_SEEDS,
    "train": layouts.TRAIN_EVAL_SEEDS,
}


class CycleEnv(ObstacleEnv):
    """Deals the layouts in order, so every (layout, start) is driven once."""

    cycle = None
    _next = 0

    def _pick_layouts(self, k):
        pick = self.cycle[(self._next + np.arange(k)) % len(self.cycle)]
        self._next += k
        return pick


def misalign(sensor, yaw, pitch, roll):
    """Turn the camera the sensor model renders from (rad)."""
    read = sensor.read

    def turned(lay, state, idx=None, draw_for_all=False):
        st = state.copy()
        st[:, P.S_YAW] += yaw
        st[:, P.S_PITCH] += pitch
        st[:, P.S_ROLL] += roll
        return read(lay, st, idx, draw_for_all)

    sensor.read = turned


def gates_from(features, cfg):
    """The env's 8 gate features -> (kind, cx, cy, nx, ny, half_span)."""
    half = math.radians(float(cfg["sensor"]["fov_deg"])) / 2
    max_range = float(cfg["sensor"]["max_range"])
    out = []
    for slot, (kind, span) in enumerate(
        ((HOOP, S.HOOP_HALF_SPAN), (CARWASH, S.ARCH_HALF_SPAN))
    ):
        valid, bearing, rng, normal = features[4 * slot : 4 * slot + 4]
        if valid < 0.5:
            continue
        b, r, n = bearing * half, rng * max_range, normal * 0.5 * math.pi
        out.append(
            (kind, r * math.cos(b), r * math.sin(b), math.cos(n), math.sin(n), span)
        )
    return out


def hoops_seen(env, i, pose, scan, cfg):
    """Every hoop the camera can see, as the segmenter reports them.

    The env's own gate features carry the nearest hoop only (the policy has
    one slot for it); the segmenter the driver runs on reports each it
    finds.  The test is the sensor model's: both feet in range, in view, and
    the first thing the scan meets on their bearings.
    """
    sensor = cfg["sensor"]
    half = math.radians(float(sensor["fov_deg"])) / 2
    bins = len(scan)
    reach = float(sensor["gate_max_range"])
    cx = pose[0] + float(sensor["camera_forward"]) * math.cos(pose[2])
    cy = pose[1] + float(sensor["camera_forward"]) * math.sin(pose[2])
    out = []
    for kind, gx, gy, ax, ay, span in env.sensor.gates[int(env.plant.lay[i])]:
        if int(kind) != HOOP:
            continue
        seen = True
        for side in (-1.0, 1.0):
            fx, fy = gx + side * span * ax - cx, gy + side * span * ay - cy
            fr = math.hypot(fx, fy)
            fb = (math.atan2(fy, fx) - pose[2] + math.pi) % (2 * math.pi) - math.pi
            if (
                fr > reach
                or abs(fb) > half - math.radians(float(sensor["gate_frame_margin_deg"]))
                or fr < 0.3
            ):
                seen = False
                break
            if (
                scan[min(max(int((fb + half) / (2 * half / bins)), 0), bins - 1)]
                < fr - 0.20
            ):
                seen = False
                break
        if not seen:
            continue
        dx, dy = gx - cx, gy - cy
        c, s = math.cos(pose[2]), math.sin(pose[2])
        nx, ny = -ay, ax
        out.append(
            (
                HOOP,
                c * dx + s * dy,
                -s * dx + c * dy,
                c * nx + s * ny,
                -s * nx + c * ny,
                span,
            )
        )
    return out


def run(args, overrides):
    # One thread: numba's per-thread generators make the sensor noise, and
    # with several the same seed gives a different run each time.
    import numba

    numba.set_num_threads(1)
    cfg = yaml.safe_load(args.env_config.read_text())
    if not args.rl_rules:
        # The env ends a run where a policy in training should be stopped
        # and charged: 2.2 m off the hand line, six seconds against a bale.
        # None of that ends a run on the course (the bucket section is 4 m
        # wide, and a car that backs off a bale drives on), so the limits
        # are opened up to what gazebo_watch.py judges by: stuck for 25 s.
        cfg["env"].update(
            off_course_m=4.0, pinned_s=25.0, stall_s=25.0, progress_window_s=60.0
        )
    knobs = load_config(args.config, overrides)
    knobs["scan_bins"] = int(cfg["sensor"]["bins"])
    seeds = (
        SEED_SETS[args.seeds]
        if args.seeds in SEED_SETS
        else [int(s) for s in args.seeds.split(",")]
    )
    model = course_model.CourseModel(seeds)
    n = len(seeds) * args.starts
    env = CycleEnv(
        cfg,
        model,
        n,
        np.arange(len(seeds)),
        seed=args.seed,
        start_box_only=True,
        randomize=args.randomize,
    )
    env.cycle = np.tile(np.arange(len(seeds)), args.starts)
    misalign(
        env.sensor,
        math.radians(args.camera_yaw),
        math.radians(args.camera_pitch),
        math.radians(args.camera_roll),
    )
    section_end = None
    if args.start:
        # Dealt a little before one obstacle, on every layout: does the car
        # get through it?  The whole lap is 90 s of sim to find that out.
        zone = env.zone_names.index(args.start)
        lays = env.cycle[:n]
        env.forced_lay = lays
        env.forced_s = np.array(
            [env.section_entry(int(L), args.start) - args.back for L in lays]
        )
        section_end = np.array(
            [env.section_s[int(L)][zone][1] + args.beyond for L in lays]
        )
    obs = env.reset()
    if args.steer_bias is not None:
        env.plant.params[:, P.P_OFFSET] = args.steer_bias
    if args.steer_gain is not None:
        env.plant.params[:, P.P_GAIN] = args.steer_gain

    def pose_of(i):
        st = env.plant.state[i]
        return float(st[P.S_X]), float(st[P.S_Y]), float(st[P.S_YAW])

    drift = [PoseDrift(args.pose_drift, math.radians(args.yaw_drift)) for _ in range(n)]
    planners = [Planner(knobs, Route("obstacle", knobs)) for _ in range(n)]
    poses = [deque(maxlen=8) for _ in range(n)]
    for i, p in enumerate(planners):
        pose = drift[i](pose_of(i))
        p.reset(pose, 0.0)
        if args.start:
            p.route.place(pose)
            # Dealt past the Wide Section's slots: that stage is behind it.
            if env.zone_names.index(args.start) > env.zone_names.index("buckets"):
                p.route.stage = 99
        poses[i].append(pose)
        p.observe(env.scan_held[i], gates_from(env.gate_held[i], cfg), pose, 0.0)
    last_scan = env.scan_held.copy()

    v_cap = float(cfg["env"]["v_cap"])
    scale = float(cfg["prior"]["residual_scale"])
    speed_at = O.SCAN_BINS + O.GATE_DIM
    done = np.zeros(n, bool)
    results = [None] * n
    traces = [[] for _ in range(n)]
    seen = [[] for _ in range(n)]
    reversals = np.zeros(n, int)
    touch_log = [[] for _ in range(n)]
    was_reverse = np.zeros(n, bool)
    plan_time, plan_calls = 0.0, 0
    steps = int(float(cfg["env"]["episode_s"]) * float(cfg["env"]["control_hz"])) + 5
    dt = env.dt
    for step in range(steps):
        t = step * dt
        action = np.zeros((n, 2))
        action[:, 1] = O.speed_to_action(0.0, cfg)
        for i in np.flatnonzero(~done):
            pose = poses[i][-1]
            speed = float(obs[i, speed_at]) * v_cap
            yaw_rate = float(obs[i, speed_at + 1]) * O.YAW_RATE_SCALE
            t0 = time.perf_counter()
            steer, v = planners[i].step(pose, speed, yaw_rate, t)
            plan_time += time.perf_counter() - t0
            plan_calls += 1
            action[i, 0] = np.clip((steer - env.prior[i]) / scale, -1.0, 1.0)
            action[i, 1] = O.speed_to_action(v, cfg)
            now_reverse = planners[i].mode == "reverse"
            if i == args.car and any(abs(t - s) < 0.5 * dt for s in args.snap):
                snapshot(
                    args.trace_dir / f"snap_{t:06.2f}.png",
                    planners[i],
                    pose,
                    t,
                    env.scan_held[i],
                    steer,
                    v,
                )
            reversals[i] += now_reverse and not was_reverse[i]
            was_reverse[i] = now_reverse
            if args.trace or args.plot:
                st = env.plant.state[i]
                g = planners[i].goal
                traces[i].append(
                    (
                        t,
                        st[P.S_X],
                        st[P.S_Y],
                        st[P.S_V],
                        v,
                        steer,
                        planners[i].route.i,
                        g[0] if g is not None else np.nan,
                        g[1] if g is not None else np.nan,
                        float(now_reverse),
                        planners[i].info.get("free", np.nan),
                        planners[i].trim_est,
                        planners[i].info.get("offset", np.nan),
                        planners[i].info.get("turn", np.nan),
                        planners[i].info.get("bend", np.nan),
                        planners[i].info.get("want", np.nan),
                        float(bool(planners[i].info.get("vetoed", False))),
                        yaw_rate,
                        st[P.S_YAW],
                    )
                )
        before = env.touches.copy()
        obs, _, term, trunc, info = env.step(action)
        for i in np.flatnonzero((env.touches > before) & ~done):
            if len(touch_log[i]) < 6 and (
                not touch_log[i] or t - touch_log[i][-1][0] > 2.0
            ):
                st = env.plant.state[i]
                touch_log[i].append(
                    (
                        round(t, 2),
                        round(float(st[P.S_X]), 2),
                        round(float(st[P.S_Y]), 2),
                        round(float(st[P.S_V]), 2),
                    )
                )
        for i in np.flatnonzero(~done):
            if section_end is not None and not info[i]:
                if env.s[i] >= section_end[i] or t > args.limit:
                    info[i] = env._episode_info(
                        i, "finish" if env.s[i] >= section_end[i] else "timeout"
                    )
            if info[i]:
                results[i] = info[i]
                results[i]["reversals"] = int(reversals[i])
                results[i]["trim"] = float(planners[i].trim_est)
                results[i]["events"] = planners[i].events[:12]
                results[i]["touch_log"] = touch_log[i]
                done[i] = True
                continue
            pose = drift[i](pose_of(i))
            poses[i].append(pose)
            if not np.array_equal(env.scan_held[i], last_scan[i]):
                # Rendered cam_lat control steps before it was delivered.
                back = min(int(env.cam_lat[i]), len(poses[i]) - 1)
                t0 = time.perf_counter()
                seen_from = poses[i][-1 - back]
                gates = [g for g in gates_from(env.gate_held[i], cfg) if g[0] != HOOP]
                gates += hoops_seen(env, i, seen_from, env.scan_held[i], cfg)
                planners[i].observe(env.scan_held[i], gates, seen_from, t + dt)
                plan_time += time.perf_counter() - t0
                if args.plot:
                    seen[i].append(planners[i].mem.copy())
        last_scan = env.scan_held.copy()
        if done.all():
            break

    finishes = [r for r in results if r and r["outcome"] == "finish"]
    by_seed = {}
    for r in results:
        if r:
            by_seed.setdefault(r["seed"], []).append(r["outcome"] == "finish")
    ends = Counter(
        f"{r['outcome']}@{r['zone']}" for r in results if r and r["outcome"] != "finish"
    )
    for i, r in enumerate(results):
        if r is None:
            print(f"#{i}: did not end")
            continue
        if args.verbose or r["outcome"] != "finish":
            print(
                f"#{i} seed {r['seed']} {r['outcome']}@{r['zone']}  "
                f"{r['dist']:.1f}/{r['goal']:.1f} m in {r['time']:.1f} s  hoops {r['hoops']}  "
                f"touches {r['touches']}  min clear {r['min_clearance']:.3f}  "
                f"reversals {r['reversals']}  end ({r['pose'][0]:.2f}, {r['pose'][1]:.2f})"
            )
            print(f"    touches (t, x, y, v): {r['touch_log']}")
            if args.verbose:
                print(f"    reversals: {r['events']}")
    summary = dict(
        n=n,
        finish=len(finishes) / n,
        progress=float(
            np.mean([min(r["dist"] / r["goal"], 1.0) for r in results if r])
        ),
        lap_s=float(np.mean([r["time"] for r in finishes])) if finishes else None,
        lap_s_max=float(np.max([r["time"] for r in finishes])) if finishes else None,
        touches=float(np.mean([r["touches"] for r in results if r])),
        reversals=float(np.mean(reversals)),
        min_clearance=float(np.min([r["min_clearance"] for r in results if r])),
        trim=float(np.mean([r["trim"] for r in results if r])),
        lag=float(np.mean([p.tau for p in planners])),
        ends=dict(ends),
        failed_seeds=[s for s, v in sorted(by_seed.items()) if not all(v)],
        plan_ms=1000.0 * plan_time / max(plan_calls, 1),
    )
    print(
        f"{len(finishes)}/{n} laps ({100 * summary['finish']:.0f}%), "
        f"progress {100 * summary['progress']:.0f}%"
        + (
            f", mean {summary['lap_s']:.1f} s, slowest {summary['lap_s_max']:.1f} s"
            if finishes
            else ""
        )
        + f"; touches {summary['touches']:.1f}/run, reversals {summary['reversals']:.1f}/run, "
        f"trim {summary['trim']:+.3f}, lag {summary['lag']:.2f}; planner {summary['plan_ms']:.1f} ms/tick"
    )
    if ends:
        print(f"ends: {dict(ends)}")
    if args.out:
        args.out.write_text(
            json.dumps(
                dict(
                    args=vars(args) | dict(overrides=overrides),
                    summary=summary,
                    episodes=results,
                ),
                indent=1,
                default=str,
            )
        )
    if args.trace:
        for i, tr in enumerate(traces):
            np.save(args.trace_dir / f"trace_{i}.npy", np.asarray(tr))
    if args.plot:
        plot(args.plot, traces, seen, results, planners)
    return summary


def snapshot(path, p, pose, t, scan, steer, v):
    """What car 0's planner holds at one tick: grids, wavefront, goal, scan."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import planner as PL

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ext_c = [
        PL.COARSE_Y0 + PL.COARSE * PL.COARSE_NY,
        PL.COARSE_Y0,
        PL.COARSE_X0,
        PL.COARSE_X0 + PL.COARSE * PL.COARSE_NX,
    ]
    rel = PL.to_frame(np.array([[pose[0], pose[1]]]), p.grid_pose)[0]
    ax = axes[0]
    pts = PL.to_frame(p.points, p.grid_pose) if len(p.points) else np.zeros((0, 2))
    ax.plot(pts[:, 1], pts[:, 0], "k.", ms=3)
    d = getattr(p, "debug", None)
    if d:
        for k in range(1, len(d["kappa"]), 2):
            ax.plot(rel[1] + d["Y"][k], rel[0] + d["X"][k], "c-", lw=0.5)
        ax.plot(rel[1] + d["Y"][0], rel[0] + d["X"][0], "m-", lw=1.5)
    if p.path is not None:
        way = PL.to_frame(p.path, p.grid_pose)
        ax.plot(way[:, 1], way[:, 0], "g.-", ms=3, lw=0.8)
    if p.route is not None:
        lane = p.route.corridor()
        if lane is not None:
            lane = PL.to_frame(lane, p.grid_pose)
            ax.plot(lane[:, 1], lane[:, 0], "b:", lw=1)
    ax.plot(p.goal_local[1], p.goal_local[0], "g*", ms=14)
    ax.plot(rel[1], rel[0], "ro")
    b = p.bearings
    ax.plot(scan * np.sin(b), p.cam_x + scan * np.cos(b), "r.", ms=3)
    ax.set_xlim(3, -3)
    ax.set_ylim(-1.5, 5)
    ax.set_title(
        f"t {t:.2f} mode {p.mode} steer {steer:+.2f} v {v:.2f} kappa {p.kappa_cmd:+.2f} "
        f"i {p.route.i if p.route else -1} gate {p.gate is not None} "
        f"align {math.degrees(p.route.align_yaw):+.1f} deg {np.round(p.route.align_shift, 2)}"
    )
    ax = axes[1]
    if p.D is not None:
        D = np.where(np.isfinite(p.D), p.D, np.nan)
        ax.imshow(D[:, ::-1], origin="lower", extent=ext_c, cmap="viridis")
    ax.plot(p.goal_local[1], p.goal_local[0], "r*", ms=14)
    ax.plot(rel[1], rel[0], "ro")
    ax.set_title(
        str(
            {k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.info.items()}
        )
    )
    fig.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)


def plot(path, traces, seen, results, planners):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(traces)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 8 * rows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        if seen[i]:
            pts = np.concatenate(seen[i][::3])
            ax.plot(pts[:, 0], pts[:, 1], ".", color="0.6", ms=1)
        route = planners[i].route.xy
        ax.plot(route[:, 0], route[:, 1], "b:", lw=0.6)
        tr = np.asarray(traces[i])
        sc = ax.scatter(
            tr[:, 1], tr[:, 2], c=tr[:, 3], s=3, cmap="viridis", vmin=0, vmax=3
        )
        rev = tr[:, 9] > 0.5
        ax.plot(tr[rev, 1], tr[rev, 2], "m.", ms=4)
        ax.plot(tr[-1, 1], tr[-1, 2], "rx", ms=10)
        r = results[i]
        ax.set_title(
            f"seed {r['seed']}: {r['outcome']}@{r['zone']} {r['time']:.0f} s hoops {r['hoops']}"
            if r
            else "unfinished"
        )
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="m/s")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("overrides", nargs="*", help="knob=value")
    ap.add_argument("--seeds", default="gazebo")
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=10_002)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--env-config", type=Path, default=RACER / "config.yaml")
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument(
        "--rl-rules", action="store_true", help="keep the env's own run-ending limits"
    )
    ap.add_argument("--steer-bias", type=float, default=None, help="rad of road wheel")
    ap.add_argument("--steer-gain", type=float, default=None)
    ap.add_argument("--camera-yaw", type=float, default=0.0, help="deg, + left")
    ap.add_argument("--camera-pitch", type=float, default=0.0, help="deg")
    ap.add_argument("--camera-roll", type=float, default=0.0, help="deg")
    ap.add_argument("--pose-drift", type=float, default=0.0, help="m per m")
    ap.add_argument("--yaw-drift", type=float, default=0.0, help="deg per m")
    ap.add_argument("--plot", type=Path)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-dir", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--start", help="deal the cars before this obstacle, and stop after it"
    )
    ap.add_argument("--back", type=float, default=2.0, help="m before it they start")
    ap.add_argument("--beyond", type=float, default=1.5, help="m past it that counts")
    ap.add_argument("--limit", type=float, default=60.0, help="s allowed with --start")
    ap.add_argument("--car", type=int, default=0, help="which car --snap watches")
    ap.add_argument(
        "--snap",
        type=lambda s: [float(v) for v in s.split(",")],
        default=[],
        help="times (s) at which to save car 0's planner state to --trace-dir",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    overrides = dict(o.split("=", 1) for o in args.overrides)
    summary = run(args, overrides)
    return 0 if summary["finish"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
