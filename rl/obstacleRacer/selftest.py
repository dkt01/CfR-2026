#!/usr/bin/env python3
"""Checks that need neither ROS nor Gazebo, run before every training run.

  python3 selftest.py            # all of it
  python3 selftest.py --quick    # skip the long prior-only rollout

1. layouts match the randomizer's draw
2. reward: reachable thresholds and episode-level incentives
3. plant: static attitudes on the ramp and bank, a wheel in a recess, a
   bump transient, and the front going light off the deck crest
4. the env steps, and episodes end for the reasons they should
5. the prior alone, from the start box: how far it gets, and that the car
   pitched and rolled on the 3D sections (a regression that flattens the
   car back to planar fails here, before a run)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import course_model
import layouts
import plant as P
import reward

HERE = Path(__file__).resolve().parent


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
    return 0 if ok else 1


def plant_checks(cfg, model):
    failures = 0
    cfg = dict(cfg, randomize=dict(cfg["randomize"], enabled=False))
    spots = [
        ("ramp", 5.0, 0.0, 0.31, 0.0),
        ("bank", -4.03, -9.37, 0.09, -math.pi / 2),
        ("flat", 0.0, 0.1, 0.0, 0.0),
    ]
    pl = P.Plant(cfg, model, len(spots), np.random.default_rng(0))
    v = np.array([s[1:] for s in spots])
    z = np.zeros(len(spots))
    pl.reset(np.arange(len(spots)), 0, v[:, 0], v[:, 1], v[:, 2], v[:, 3], z)
    ramp, bank, flat = pl.state
    failures += check(
        "resting on the ramp pitches the body with it",
        abs(math.degrees(ramp[P.S_PITCH]) - math.degrees(math.atan(0.19))) < 1.2,
        f"{math.degrees(ramp[P.S_PITCH]):.2f} deg vs {math.degrees(math.atan(0.19)):.2f}",
    )
    failures += check(
        "resting on the bank rolls the body with it",
        abs(abs(math.degrees(bank[P.S_ROLL])) - 8.5) < 1.2,
        f"{math.degrees(bank[P.S_ROLL]):.2f} deg vs 8.5",
    )
    failures += check(
        "on the flat the springs sit at ride height",
        np.abs(flat[P.S_Q : P.S_Q + 4]).max() < 0.001 and abs(flat[P.S_Z]) < 0.001,
        f"q {np.abs(flat[P.S_Q : P.S_Q + 4]).max() * 1000:.2f} mm",
    )

    # Drive straight across the pothole board and off the deck crest, and
    # watch the body.
    runs = [
        ("potholes", -1.2, -8.3, 0.0, 0.0, 1.5, 60),
        ("deck crest", 6.2, 0.0, 0.52, 0.0, 3.0, 30),
    ]
    pl = P.Plant(cfg, model, len(runs), np.random.default_rng(0))
    v = np.array([r[1:5] for r in runs])
    speeds = np.array([r[5] for r in runs])
    pl.reset(np.arange(len(runs)), 0, v[:, 0], v[:, 1], v[:, 2], v[:, 3], speeds)
    trace = []
    for _ in range(max(r[6] for r in runs)):
        pl.step(np.zeros(len(runs)), speeds, 10)
        trace.append(pl.state.copy())
    trace = np.array(trace)
    pot = trace[:, 0]
    on_board = (pot[:, P.S_X] > 0.1) & (pot[:, P.S_X] < 2.4)
    pitch_sd = np.degrees(pot[on_board, P.S_PITCH]).std()
    roll_sd = np.degrees(pot[on_board, P.S_ROLL]).std()
    failures += check(
        "crossing the potholes moves the body in pitch and roll",
        pitch_sd > 0.1 and roll_sd > 0.1,
        f"pitch sd {pitch_sd:.2f} deg, roll sd {roll_sd:.2f} deg",
    )
    loads = trace[:, 0, P.S_LOAD : P.S_LOAD + 4][on_board]
    failures += check(
        "wheels load and unload over the recesses and bumps",
        (loads.max(0) - loads.min(0)).min() > 1.0,
        f"per-wheel load swing {np.round(loads.max(0) - loads.min(0), 1)} N",
    )
    crest = trace[:, 1]
    front = crest[:, P.S_LOAD] + crest[:, P.S_LOAD + 1]
    failures += check(
        "the front goes light over the deck crest at 3 m/s",
        front.min() < 0.6 * front[:4].mean(),
        f"front load min {front.min():.1f} N vs {front[:4].mean():.1f} N on the ramp",
    )
    return failures


def gate_parity(cfg, model):
    """sensor.py's gate features == observation.gate_features on the same gate.

    The car's node builds gate features from the segmenter's gates with
    observation.gate_features; training builds them in sensor.py's kernel.
    Put a car in front of hoop_0 and compare.
    """
    import observation as O
    import sensor as S

    cfg = dict(cfg, randomize=dict(cfg["randomize"], enabled=False))
    cfg["sensor"] = dict(
        cfg["sensor"], noise_a=0.0, noise_b=0.0, dropout=0.0, phantom=0.0
    )
    gates = S.gate_table(model)[0]
    hoop = next(g for g in gates if g[0] == S.HOOP)
    x, y, yaw = hoop[1] + 1.6, hoop[2] + 0.2, math.pi + 0.15
    pl = P.Plant(cfg, model, 1, np.random.default_rng(0))
    pl.reset(
        np.arange(1),
        0,
        np.array([x]),
        np.array([y]),
        np.zeros(1),
        np.array([yaw]),
        np.zeros(1),
    )
    sen = S.Sensor(cfg, model, 1, np.random.default_rng(0))
    _, gate = sen.read(pl.lay, pl.state)
    st = pl.state[0]
    cp = math.cos(st[P.S_PITCH])
    cx = st[P.S_X] + 0.315 * cp * math.cos(st[P.S_YAW])
    cy = st[P.S_Y] + 0.315 * cp * math.sin(st[P.S_YAW])
    c, s = math.cos(-st[P.S_YAW]), math.sin(-st[P.S_YAW])
    dx, dy = hoop[1] - cx, hoop[2] - cy
    ax, ay = hoop[3], hoop[4]
    local = (
        O.HOOP_KIND,
        c * dx - s * dy,
        s * dx + c * dy,
        c * ax - s * ay,
        s * ax + c * ay,
    )
    want = O.gate_features([local], cfg)
    err = float(np.abs(gate[0, :4] - want[:4]).max())
    return check(
        "sensor.py gate features match observation.gate_features",
        gate[0, 0] == 1.0 and err < 1e-6,
        f"max diff {err:.2e}",
    )


def sensor_on_slopes(cfg, model):
    """Up the ramp and down the helix, the lane ahead reads open.

    v1 trained with the ramp's own surface reading as a wall 0.3 m ahead in
    every bin -- the rays' ground reference started at the car, 0.06 m below
    the road under the camera -- so the car climbed the ramp blind.
    """
    import sensor as S

    cfg = dict(cfg, randomize=dict(cfg["randomize"], enabled=False))
    cfg["sensor"] = dict(
        cfg["sensor"], noise_a=0.0, noise_b=0.0, dropout=0.0, phantom=0.0
    )
    spots = [("ramp foot", 4.0, 0.0, 0.0), ("ramp", 4.6, 0.0, 0.0)]
    pl = P.Plant(cfg, model, len(spots), np.random.default_rng(0))
    v = np.array([s[1:] for s in spots])
    pl.reset(
        np.arange(len(spots)),
        0,
        v[:, 0],
        v[:, 1],
        np.ones(len(spots)),
        v[:, 2],
        np.zeros(len(spots)),
    )
    sen = S.Sensor(cfg, model, len(spots), np.random.default_rng(0))
    scan, _ = sen.read(pl.lay, pl.state)
    mid = scan.shape[1] // 2
    ahead = scan[:, mid - 1 : mid + 1].min(1)
    return check(
        "the lane ahead reads open on the ramp",
        bool((ahead > 3.0).all()),
        ", ".join(f"{s[0]} {a:.1f} m" for s, a in zip(spots, ahead)),
    )


def lap_bookkeeping(cfg, model):
    """A car carried round the loop from a dealt start finishes only there.

    The car is moved along its layout's line (threading each hoop) past the
    timing line and on round to where it started.  Progress has to wrap at
    the timing line without a jump, every hoop has to count, and the lap
    has to complete back at the start point -- not at the timing line.
    """
    import env as env_module

    env = env_module.ObstacleEnv(cfg, model, 3, [0], seed=0, randomize=False)
    env.reset()
    line = env.lines.lines[0]
    loop = float(env.loop[0])
    failures = 0
    for k, s0 in enumerate((0.0, 30.0, loop * 0.9)):
        x, y, z, yaw = line.pose_at(s0)
        i = np.array([k])
        env.plant.reset(
            i,
            np.array([0]),
            np.array([x]),
            np.array([y]),
            np.array([z]),
            np.array([yaw]),
            np.zeros(1),
        )
        env.idx[k] = line.index_at(s0)
        env.idx[i], env.s[i], _ = env.lines.project(
            np.array([0]), env.idx[i], np.array([x]), np.array([y]), np.array([z])
        )
        env.s_start[k] = env.s[k]
        env.dist[k] = 0.0
        env.goal[k] = max(loop, line.lap_length - env.s[k])
        env.hoop_d[k] = np.mod(env.hoop_s[0] - env.s[k], loop)
        env.hoop_state[k] = 0
        env.hoop_u[k] = env._hoop_u(i)[0]
        step, s, largest, done_at = 0.08, float(env.s[k]), 0.0, None
        for _ in range(int((loop + 2.0) / step)):
            s_next = s + step
            if s_next >= line.lap_length:
                s_next -= loop
            x, y, z, yaw = line.pose_at(s_next)
            st = env.plant.state.copy()
            st[k, P.S_X], st[k, P.S_Y], st[k, P.S_Z], st[k, P.S_YAW] = x, y, z, yaw
            env.plant.state[k] = st[k]
            ds, _, _, missed, _ = env._track(env.plant.state, env.plant.lay)
            largest = max(largest, abs(ds[k]))
            if missed[k]:
                break
            s = s_next
            if (
                done_at is None
                and env.dist[k] >= env.goal[k]
                and (env.hoop_state[k] == 1).all()
            ):
                done_at = float(env.dist[k])
        ok = done_at is not None and largest < 0.5 and abs(done_at - env.goal[k]) < 0.2
        failures += check(
            f"a lap from s={s0:.1f} m wraps, threads all hoops, ends back at the start",
            ok,
            f"finished at {done_at} m of {env.goal[k]:.1f}, largest step {largest:.2f} m, "
            f"hoops {env.hoop_state[k].tolist()}",
        )
    return failures


def _plant_at(cfg, model, poses, seed=0):
    """A plant with nominal parameters, cars at (x, y, yaw, speed) on layout 0."""
    cfg = dict(cfg, randomize=dict(cfg["randomize"], enabled=False))
    pl = P.Plant(cfg, model, len(poses), np.random.default_rng(seed))
    p = np.array(poses, dtype=np.float64)
    pl.reset(np.arange(len(p)), 0, p[:, 0], p[:, 1], np.zeros(len(p)), p[:, 2], p[:, 3])
    return pl


def reverse_checks(cfg, model):
    """Reverse as last week's characterization found it: coast, wait, drive back.

    docs/characterization-results.md: no braking (a reverse target while
    rolling forward coasts); the direction changes 0.4 s (tach timeout) +
    0.1 s after the car drops under the 0.3 m/s tach floor; drag and
    feedforward are the same both ways.
    """
    failures = 0
    # Open floor: the lane east of the start box, heading east, at 2 m/s.
    pl = _plant_at(cfg, model, [(0.3, 0.0, 0.0, 2.0), (0.3, 0.0, 0.0, 2.0)])
    dt = 1.0 / float(cfg["env"]["control_hz"])
    v = []
    flip_t = None
    below_t = None
    for i in range(int(4.0 / dt)):
        pl.step(np.zeros(2), np.array([-1.0, 0.0]), 10)
        v.append(pl.state[:, P.S_V].copy())
        if below_t is None and abs(pl.state[0, P.S_V]) < P.STOPPED_V:
            below_t = (i + 1) * dt
        if flip_t is None and pl.state[0, P.S_DIR] < 0:
            flip_t = (i + 1) * dt
    v = np.array(v)
    coasting = v[:, 1] > 0.35
    failures += check(
        "a reverse command while rolling forward only coasts (no braking)",
        np.abs(v[coasting, 0] - v[coasting, 1]).max() < 1e-9,
        f"largest difference from coasting {np.abs(v[coasting, 0] - v[coasting, 1]).max():.3f} m/s",
    )
    # Past the 0.19 s dead time, over half a second.
    decel = (v[5, 1] - v[15, 1]) / (10 * dt)
    mid = 0.5 * (v[5, 1] + v[15, 1])
    failures += check(
        "coast decel matches the fit, 0.606 + 0.130 v",
        abs(decel - (0.606 + 0.130 * mid)) < 0.05,
        f"{decel:.2f} m/s^2 at {mid:.2f} m/s, fit {0.606 + 0.130 * mid:.2f}",
    )
    wait = float(cfg["plant"]["reverse_wait"])
    ok = (
        flip_t is not None
        and below_t is not None
        and abs(flip_t - below_t - wait) < 2.5 * dt
    )
    failures += check(
        "the direction changes reverse_wait after the car drops under 0.3 m/s",
        ok,
        f"under the floor at {below_t} s, reversed at {flip_t} s (wait {wait} s)",
    )
    failures += check(
        "then it drives backwards at the commanded speed",
        abs(v[-1, 0] + 1.0) < 0.05 and v[-1, 1] == 0.0,
        f"{v[-1, 0]:+.2f} m/s (coasting car {v[-1, 1]:+.2f})",
    )
    return failures


def contact_checks(cfg, model):
    """Touching slides the car along what it touched, or stops it; never inside.

    On the straight lane south of the overpass (x ~7.1, where the car has
    0.2 m to spare turned across it): driven head-on into the lane's side,
    and at a glancing 12 degrees along it.  The lane pinches in 2.5 m on
    (y ~ -4.5), where the glancing car jams, so its slide is judged over the
    0.75 s after it first touches.
    """
    failures = 0
    import centerline as centerline_module

    line = centerline_module.Centerlines(model.layouts).lines[0]
    x, y, _, yaw = line.pose_at(18.0)
    head_on = (x, y, yaw + math.pi / 2, 1.0)
    glance = (x, y, yaw + math.radians(12), 2.0)
    pl = _plant_at(cfg, model, [head_on, glance])
    dt = 1.0 / float(cfg["env"]["control_hz"])
    first = [None, None]
    lost = np.zeros(2)
    inside = 0
    speed_after = []
    for i in range(int(3.0 / dt)):
        touched, v_lost, _ = pl.step(np.zeros(2), np.array([1.0, 2.0]), 10)
        lost += v_lost
        for k in range(2):
            if touched[k] and first[k] is None:
                first[k] = i
        inside += int(P.body_contact(pl.OBS, pl.lay, pl.state, 0.03).sum())
        if first[1] is not None:
            speed_after.append(pl.state[1, P.S_V])
    failures += check(
        "both cars reach the lane's side",
        None not in first,
        f"first touch steps {first}",
    )
    failures += check(
        "no car is ever left inside an obstacle",
        inside == 0,
        f"{inside} car-steps inside",
    )
    failures += check(
        "head-on, the car stops against it and loses its speed",
        abs(pl.state[0, P.S_V]) < 0.3 and lost[0] > 0.8,
        f"speed now {pl.state[0, P.S_V]:.2f} m/s, lost {lost[0]:.2f} m/s in all",
    )
    held = float(np.mean(speed_after[:15])) if speed_after else 0.0
    failures += check(
        "at a glancing angle it slides along, keeping most of its speed",
        held > 1.2,
        f"{held:.2f} m/s over the 0.75 s after touching",
    )
    return failures


def memory_checks(cfg):
    """The stacked frames and the tach-only memory features, by hand."""
    import observation as O

    failures = 0
    offsets = O.frame_offsets(cfg)
    stack = O.Stack(1, offsets, 1)
    stack.reset(np.array([0]), np.zeros((1, 1), np.float32))
    for k in range(1, 31):
        obs = stack.push(np.full((1, 1), k, np.float32))
    want = [30 - o for o in offsets]
    failures += check(
        "the stack holds the frames at frame_offsets steps back",
        obs[0].tolist() == want,
        f"offsets {offsets}: {obs[0].tolist()}, want {want}",
    )
    dt = 1.0 / float(cfg["env"]["control_hz"])
    mem = O.Memory(1, cfg)
    readings = [1.0] * 20 + [0.0] * 30 + [-0.6] * 10
    out = [mem.update(np.array([v]))[0] for v in readings]
    failures += check(
        "stopped time counts while the tach reads zero and caps",
        abs(out[29][0] - 10 * dt / O.STOP_CAP_S) < 1e-9
        and out[49][0] <= 1.0
        and out[50][0] == 0.0,
        f"after 0.5 s {out[29][0]:.3f}, capped {out[49][0]:.3f}, moving again {out[50][0]:.3f}",
    )
    failures += check(
        "direction holds through the tach's silence and flips on a reverse reading",
        out[49][2] == 1.0 and out[59][2] == -1.0,
        f"stopped {out[49][2]:+.0f}, reversing {out[59][2]:+.0f}",
    )
    failures += check(
        "mean speed follows the tach, signed",
        out[19][1] > 0 and out[59][1] < out[49][1],
        f"{out[19][1]:+.3f} driving, {out[59][1]:+.3f} backing",
    )
    return failures


def recovery_checks(cfg, model):
    """Stuck against a wall: time to back out, and stuck starts where it stopped.

    Two cars on layout 0, turned to face the lane's side at s = 18 m (the
    contact check's head-on spot).  One keeps pushing: the run has to end
    pinned, and not before pinned_s.  The other reverses once it has
    stopped: before the run is ended it has to be moving backwards, off the
    wall -- recovery is physically there to be learned.
    """
    import env as env_module
    import observation as O

    failures = 0
    cfg = dict(cfg, randomize=dict(cfg["randomize"], enabled=False))
    cfg["env"] = dict(cfg["env"], stuck_start_prob=0.0)
    env = env_module.ObstacleEnv(cfg, model, 2, np.array([0]), seed=0)
    env.forced_lay = np.zeros(2, np.int64)
    env.forced_s = np.full(2, 18.0)
    env.reset()
    st = env.plant.state
    x, y, z, yaw = (
        st[:, P.S_X].copy(),
        st[:, P.S_Y].copy(),
        st[:, P.S_Z].copy(),
        st[:, P.S_YAW].copy(),
    )
    env.plant.reset(
        np.arange(2), np.zeros(2, np.int64), x, y, z, yaw + math.pi / 2, np.full(2, 0.8)
    )
    dt = env.dt
    forward = O.speed_to_action(1.0, cfg)
    back = O.speed_to_action(-1.0, cfg)
    ends = [None, None]
    stopped_at = None
    backed = 0.0
    for i in range(int(12.0 / dt)):
        a = np.zeros((2, 2))
        a[:, 1] = forward
        if stopped_at is not None and ends[1] is None:
            a[1, 1] = back
        _, _, term, trunc, info = env.step(a)
        if (
            stopped_at is None
            and env.touch_age[1] == 0
            and abs(env.plant.state[1, P.S_V]) < 0.1
        ):
            stopped_at = (i + 1) * dt
        if stopped_at is not None and ends[1] is None:
            backed = min(backed, env.plant.state[1, P.S_V])
        for k in range(2):
            if (term[k] or trunc[k]) and ends[k] is None:
                ends[k] = ((i + 1) * dt, info[k]["outcome"])
                if k == 0:
                    env.forced_s = None  # let car 1's own run go on unforced
        if None not in ends:
            break
    pinned_s = float(cfg["env"].get("pinned_s", cfg["env"]["stall_s"]))
    failures += check(
        "pushing on a wall ends pinned, after pinned_s rather than stall_s",
        ends[0] is not None and ends[0][1] == "pinned" and ends[0][0] >= pinned_s,
        f"ended {ends[0]}, pinned_s {pinned_s} s, stall_s {cfg['env']['stall_s']} s",
    )
    failures += check(
        "reversing once stopped backs the car off before the run is ended",
        stopped_at is not None and backed < -float(cfg["env"]["stall_speed"]),
        f"stopped at {stopped_at} s, fastest backwards {backed:+.2f} m/s, run {ends[1]}",
    )
    # The pushing car's pinned pose is now in the stuck memory.
    k = int(min(env.stuck_n[0], env.stuck_pose.shape[1]))
    ok = k > 0
    detail = f"{env.stuck_n[0]} stuck poses recorded"
    if ok:
        env.forced_s = None
        env.forced_lay = np.zeros(2, np.int64)
        env.cfg["env"] = dict(env.cfg["env"], stuck_start_prob=1.0, start_box_prob=0.0)
        env._reset_idx(np.array([0]))
        pose = env.stuck_pose[0, :k]
        near = np.hypot(
            pose[:, 1] - env.plant.state[0, P.S_X],
            pose[:, 2] - env.plant.state[0, P.S_Y],
        ).min()
        ok = env.start_kind[0] == 2 and near < 0.05 and env.touch_age[0] == 0.0
        detail += f"; restarted {near:.3f} m from one, kind {env.start_kind[0]}"
    failures += check("a stuck start puts the car back where it was pinned", ok, detail)
    return failures


def coverage_checks(cfg, model):
    """Every obstacle on every layout is a section the env deals starts before."""
    import env as env_module

    env = env_module.ObstacleEnv(cfg, model, 4, np.arange(len(model.layouts)), seed=2)
    want = {
        "overpass_ramp",
        "helical_ramp",
        "tunnel",
        "gravel_pit",
        "banked_turn",
        "potholes",
        "buckets",
        "hoops",
        "car_wash",
    }
    missing = {}
    for k, spans in enumerate(env.section_s):
        names = {env.zone_names[z] for z in spans}
        if want - names:
            missing[int(model.seeds[k])] = sorted(want - names)
    failures = check(
        "every obstacle is a practice section on every layout",
        not missing,
        str(missing),
    )
    # Dealt section starts land before each of them, in proportion.
    env.cfg = dict(
        cfg,
        env=dict(
            cfg["env"], start_box_prob=0.0, fail_start_prob=0.0, section_start_prob=1.0
        ),
    )
    seen = Counter()
    for _ in range(60):
        env.reset()
        z = env.zone_of[env.plant.lay, env.idx]
        for i in range(env.n):
            nxt = [
                zz
                for zz, (s_in, _) in env.section_s[env.plant.lay[i]].items()
                if 0.0 <= s_in - env.s[i] <= 4.5
            ]
            seen.update(env.zone_names[zz] for zz in nxt)
        del z
    failures += check(
        "section starts are dealt before every obstacle",
        want <= set(seen),
        f"{dict(seen)}",
    )
    # Section starts lean toward what fails: one obstacle failing every time
    # it is met, the rest never, gets well over its uniform share -- and the
    # rest keep theirs above zero.
    tunnel = env.zone_names.index("tunnel")
    cands = list(env.section_s[0])
    env.practice_met[:] = 50.0
    env.practice_fail[:] = 0.0
    env.practice_fail[tunnel] = 50.0
    _, p = env.section_weights(0)
    share = dict(zip(cands, p))
    uniform = 1.0 / len(cands)
    failures += check(
        "section starts favor the obstacle that keeps failing",
        share[tunnel] > 3 * uniform
        and min(p) > 0.5 * float(cfg["env"].get("section_uniform_mix", 1.0)) * uniform,
        f"tunnel {share[tunnel]:.2f} vs uniform {uniform:.2f}, least {min(p):.3f}",
    )
    env.practice_met[:] = 0.0
    env.practice_fail[:] = 0.0
    # An obstacle is where the car drives through it, not every point inside
    # its 2D outline: each is one unbroken stretch of the line, and where the
    # deck crosses over the tunnel the label follows the line's height.
    split, wrong_level = {}, {}
    for k, line in enumerate(env.lines.lines):
        zones = env.zone_of[k, : len(line.points)]
        runs = zones[np.r_[True, zones[1:] != zones[:-1]]]
        again = sorted({env.zone_names[z] for z in runs if (runs == z).sum() > 1})
        if again:
            split[int(model.seeds[k])] = again
        for i in np.flatnonzero(zones == env.zone_names.index("tunnel")):
            if line.points[i, 2] > env_module.ELEVATED_Z:
                wrong_level[int(model.seeds[k])] = (
                    f"tunnel at z {line.points[i, 2]:.2f}"
                )
        for name in env_module.ELEVATED_REGIONS:
            for i in np.flatnonzero(zones == env.zone_names.index(name)):
                x, y, z = line.points[i]
                if (
                    z <= env_module.ELEVATED_Z
                    and len(env_module._regions().classify(x, y)) > 1
                ):
                    wrong_level[int(model.seeds[k])] = (
                        f"{name} on the floor at s {line.arc[i]:.1f}"
                    )
    failures += check(
        "each obstacle is one unbroken stretch of the line",
        not split,
        str(split),
    )
    failures += check(
        "over/under crossings are labeled by height (deck vs tunnel)",
        not wrong_level,
        str(wrong_level),
    )
    return failures


def hoop_incentives(cfg, model):
    """At the first hoop: through the middle, round the outside, stopped short.

    Cars are carried kinematically along the hoop's travel normal, starting
    4 m before its plane, at lateral offsets 0 (threads), 0.9 m (round it,
    inside the attempt gate) and 0 again but halted 1 m short.  Threading
    pays the hoop and the alignment shaping hands back what it gave;
    going round is a miss; stopping short is neither.
    """
    import env as env_module

    env = env_module.ObstacleEnv(cfg, model, 3, [0], seed=0, randomize=False)
    env.reset()
    c, a, nrm = env.hoop_c[0, 0], env.hoop_a[0, 0], env.hoop_n[0, 0]
    yaw = math.atan2(nrm[1], nrm[0])
    lateral = np.array([0.0, 0.9, 0.0])
    stop_at = np.array([np.inf, np.inf, -1.0])
    gamma = float(cfg["train"]["gamma"])
    shaped = np.zeros(3)
    phi_prev = np.zeros(3)
    threaded = np.zeros(3, np.int64)
    missed = np.zeros(3, bool)
    discount = 1.0

    def place(u):
        for k in range(3):
            xy = c + nrm * min(u, stop_at[k]) + a * lateral[k]
            env.plant.state[k, P.S_X], env.plant.state[k, P.S_Y] = xy
            env.plant.state[k, P.S_YAW] = yaw

    place(-4.0)
    env.idx[:] = env.lines.index_at(
        np.zeros(3, np.int64), np.full(3, env.hoop_s[0, 0] - 4.0)
    )
    env._track(env.plant.state, env.plant.lay)
    env.hoop_state[:] = 0
    env.hoop_u[:] = env._hoop_u(np.arange(3))
    env.dist[:] = 0.0
    env.hoop_d[:] = env.hoop_s[0][None, :] - (env.hoop_s[0, 0] - 4.0)
    for u in np.arange(-3.9, 2.0, 0.05):
        place(u)
        _, _, now, miss, phi = env._track(env.plant.state, env.plant.lay)
        threaded += now
        missed |= miss
        live = ~missed | miss
        # Discounted, as PPO values it: gamma*phi' - phi telescopes to zero.
        shaped += discount * np.where(
            live, np.where(miss, 0.0, gamma * phi) - phi_prev, 0.0
        )
        discount *= gamma
        phi_prev = np.where(miss, 0.0, phi)
    failures = check(
        "through the middle threads the hoop, no miss",
        threaded[0] == 1 and not missed[0],
        f"threaded {threaded[0]}, missed {missed[0]}",
    )
    failures += check(
        "the alignment shaping nets out (discounted) over a threaded approach",
        # Less whatever it now holds for the next hoop, whose window the
        # straight path has entered.
        abs(shaped[0] - discount * phi_prev[0]) < 0.05,
        f"net {shaped[0] - discount * phi_prev[0]:+.3f} beyond the "
        f"{discount * phi_prev[0]:.2f} held for the next hoop "
        f"(the hoop pays {cfg['reward']['hoop']})",
    )
    failures += check(
        "round the outside is a miss",
        missed[1] and threaded[1] == 0,
        f"missed {missed[1]}",
    )
    failures += check(
        "stopped short: neither threaded nor missed (stall does the charging)",
        not missed[2] and threaded[2] == 0,
        f"missed {missed[2]}",
    )
    return failures


def rollout(cfg, model, cars, seconds, start_box_only=True):
    import env as env_module

    env = env_module.ObstacleEnv(
        cfg,
        model,
        cars,
        np.arange(len(model.layouts)),
        seed=1,
        start_box_only=start_box_only,
    )
    env.reset()
    infos = []
    t = time.time()
    steps = int(seconds * float(cfg["env"]["control_hz"]))
    for _ in range(steps):
        _, _, term, trunc, info = env.step(env.scripted_action())
        infos += [i for i in info if i]
    rate = cars * steps / (time.time() - t)
    return env, infos, rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    failures = 0

    print("layouts")
    stale = [
        s
        for s in layouts.TRAIN_SEEDS + layouts.HELDOUT_SEEDS
        if layouts.load(s) != layouts.draw(s)
    ]
    failures += check(
        "exported layouts match the randomizer's draw", not stale, str(stale)
    )

    print("reward")
    try:
        reward.assert_reachable(cfg)
        reward.assert_episode_incentives(cfg)
        failures += check("thresholds reachable, episode incentives in order", True)
    except AssertionError as error:
        failures += check(
            "thresholds reachable, episode incentives in order", False, str(error)
        )

    model = course_model.CourseModel(layouts.TRAIN_SEEDS + layouts.HELDOUT_SEEDS)
    print("plant")
    failures += plant_checks(cfg, model)
    print("sensor")
    failures += gate_parity(cfg, model)
    failures += sensor_on_slopes(cfg, model)
    print("lap bookkeeping")
    failures += lap_bookkeeping(cfg, model)
    print("reverse and contact")
    failures += reverse_checks(cfg, model)
    failures += contact_checks(cfg, model)
    print("observation memory and recovery")
    failures += memory_checks(cfg)
    failures += recovery_checks(cfg, model)
    print("hoops")
    failures += hoop_incentives(cfg, model)
    print("coverage")
    failures += coverage_checks(cfg, model)

    print("env")
    env, infos, rate = rollout(cfg, model, 64, 4.0 if args.quick else 60.0)
    failures += check("env steps", rate > 0, f"{rate:.0f} car-steps/s")
    if infos:
        outcomes = Counter(i["outcome"] for i in infos)
        reach = np.array([i["dist"] for i in infos])
        print(
            f"  prior alone, start box: {len(infos)} episodes, outcomes {dict(outcomes)}"
        )
        print(
            f"  distance: median {np.median(reach):.1f} m, best {reach.max():.1f} m, "
            f"hoops {Counter(i['hoops'] for i in infos)}"
        )
        print(
            "  where they ended:",
            dict(Counter(i["zone"] for i in infos).most_common(8)),
        )
        if not args.quick:
            att = {}
            for i in infos:
                for zone, (p, r) in i["attitude"].items():
                    a = att.setdefault(zone, [0.0, 0.0])
                    a[0], a[1] = max(a[0], p), max(a[1], r)
            for zone in ("overpass_ramp", "helical_ramp", "banked_turn", "potholes"):
                if zone in att:
                    p, r = att[zone]
                    failures += check(
                        f"body moves on {zone}",
                        p > 0.5 or r > 0.5,
                        f"max |pitch| {p:.1f} deg, |roll| {r:.1f} deg",
                    )
                else:
                    print(f"  ---- {zone}: not reached by the prior; not checked")
    print("PASS" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
