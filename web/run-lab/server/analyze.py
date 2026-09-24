"""Turn one recorded run into what the Run Lab shows.

    python -m analyze <run_dir>          # or through the UI's "Process" button

Reads <run>/bag with `rosbags` (no ROS needed) and writes <run>/analysis/:

    summary.json    KPIs, verdicts, per-domain stats, events, lap + section tables
    series.json     every channel resampled onto one 20 Hz timeline
    course.json     centerline, cap zones and bales, for the track map
    logs.json       /rosout, every line
    recording.rrd   the run for the Rerun viewer: course, car, clouds, the
                    accumulated map, camera, every channel, logs -- all in the
                    track frame on the same timeline (see rerun_export.py)

The timeline: bag receive time, except in simulation (a /clock topic in the
bag) where it is mapped onto sim time -- Gazebo with a rendered camera runs
slower than real time, and a lap time measured in wall seconds is wrong.

The track frame: the driver latches the ZED map pose at the start signal and
pins it to the surveyed start pose.  When the bag has the driver's
~/telemetry, the map->track transform is FITTED from pose/telemetry pairs, so
it is exactly the transform the driver used.  Otherwise it is re-derived the
driver's way: latch at the first go / first motion.
"""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bagio  # noqa: E402
import course as course_mod  # noqa: E402
import rerun_export  # noqa: E402

VERSION = 4  # bump when outputs change shape; the UI re-processes older ones
GRID_HZ = 20.0
ZED = "/zed/zed_node"
GRAZE = 0.12  # m; the training reward's graze band
STALL_SPEED = 0.3
MODE_NAMES = {
    0: "E-STOP",
    1: "RC armed",
    2: "RC active",
    3: "AUTO armed",
    4: "AUTO active",
}
STATE_NAMES = {
    0: "waiting",
    1: "running",
    2: "stopping",
    3: "finished",
    4: "pose stale",
    5: "manual stop",
}
BATTERY_EMPTY_MV, BATTERY_FULL_MV = 10000.0, 12600.0  # maneuver_runner_node.py


class Progress:
    def __init__(self, callback):
        self.callback = callback or (lambda frac, text: None)

    def __call__(self, frac, text):
        self.callback(float(frac), text)


# ======================================================================= read


def read_streams(bag, progress):
    """One pass over every small topic; returns plain numpy arrays."""
    topics = bag.topics()
    small = [
        "/clock",
        "/arduino_bridge/status",
        "/drive_cmd",
        "/formula_one/telemetry",
        "/lap_counter/count",
        "/lap_counter/done",
        "/start_signal_detector/state",
        "/start_signal_detector/go",
        ZED + "/pose",
        ZED + "/odom",
        ZED + "/pose/status",
        ZED + "/imu/data",
        "/rosout",
        "/tf_static",
        "/tf",
    ]
    wanted = [t for t in small if t in topics]
    total = max(1, sum(topics[t]["count"] for t in wanted))
    s = {k: [] for k in wanted}
    tf = bagio.StaticTF()
    tf_dynamic_pairs = set()
    seen = 0
    last_imu = 0
    for topic, t_ns, msg in bag.messages(*wanted):
        seen += 1
        if seen % 5000 == 0:
            progress(
                0.05 + 0.35 * seen / total, f"reading messages ({seen:,}/{total:,})"
            )
        t = t_ns * 1e-9
        if topic == "/clock":
            s[topic].append((t, msg.clock.sec + msg.clock.nanosec * 1e-9))
        elif topic == "/arduino_bridge/status":
            s[topic].append(
                (
                    t,
                    msg.speed,
                    msg.target_speed,
                    msg.throttle_us,
                    msg.battery_level,
                    msg.mode,
                    msg.estop,
                    msg.link_ok,
                    msg.wheel_rpm,
                )
            )
        elif topic == "/drive_cmd":
            s[topic].append((t, msg.steering, msg.velocity, msg.auto_ready))
        elif topic == "/formula_one/telemetry":
            a = list(msg.action) + [math.nan, math.nan]
            s[topic].append(
                (
                    t,
                    msg.state,
                    msg.station,
                    msg.distance,
                    msg.lap,
                    msg.laps_target,
                    msg.lap_time,
                    msg.last_lap_time,
                    msg.race_time,
                    msg.x,
                    msg.y,
                    msg.yaw,
                    msg.cross_track,
                    msg.heading_error,
                    msg.clearance,
                    msg.speed,
                    msg.v_cap,
                    msg.v_floor,
                    msg.yaw_rate,
                    msg.speed_rate,
                    a[0],
                    a[1],
                    msg.steer_ff,
                    msg.steer_cmd,
                    msg.velocity_cmd,
                    msg.speed_scale,
                    min(msg.pose_age, 99.0),
                    msg.speed_from_tach,
                    msg.relocalized,
                )
            )
            s.setdefault("_driver", msg.driver)
        elif topic == "/lap_counter/count":
            s[topic].append(
                (t, msg.state, msg.laps, msg.target, msg.rejected, msg.loop_closures)
            )
        elif topic == "/lap_counter/done":
            s[topic].append((t, msg.data))
        elif topic == "/start_signal_detector/state":
            s[topic].append((t, msg.state, msg.go, msg.armed))
        elif topic == "/start_signal_detector/go":
            s[topic].append((t, msg.data))
        elif topic in (ZED + "/pose", ZED + "/odom"):
            pose = msg.pose if topic.endswith("pose") else msg.pose.pose
            p, q = pose.position, pose.orientation
            s[topic].append(
                (t, bagio.stamp_s(msg.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w)
            )
        elif topic == ZED + "/pose/status":
            # zed_msgs/PosTrackStatus; field names vary across wrapper
            # releases, so read whatever integer fields it has.
            row = {"t": t}
            for name in getattr(msg, "__dataclass_fields__", {}):
                value = getattr(msg, name)
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                    row[name] = int(value)
            s[topic].append(row)
        elif topic == ZED + "/imu/data":
            if t_ns - last_imu < 20_000_000:  # 50 Hz is plenty for stats
                continue
            last_imu = t_ns
            a, w = msg.linear_acceleration, msg.angular_velocity
            s[topic].append((t, a.x, a.y, a.z, w.x, w.y, w.z))
        elif topic == "/rosout":
            s[topic].append((t, int(msg.level), msg.name, msg.msg))
        elif topic == "/tf_static":
            tf.add(msg)
        elif topic == "/tf":
            for x in msg.transforms:
                tf_dynamic_pairs.add(
                    (x.header.frame_id.lstrip("/"), x.child_frame_id.lstrip("/"))
                )
    out = {}
    for k, rows in s.items():
        if k == "_driver":
            out[k] = rows
        elif k in (ZED + "/pose/status", "/rosout"):
            out[k] = rows
        elif rows:
            out[k] = np.array(rows, dtype=float)
    out["_tf"] = tf
    out["_tf_dynamic"] = tf_dynamic_pairs
    out["_topics"] = topics
    return out


# ================================================================== helpers


def smooth(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def derivative(t, v):
    if len(t) < 3:
        return np.zeros_like(v)
    return np.gradient(v, np.maximum.accumulate(t) + np.arange(len(t)) * 1e-9)


def hold(t_src, v_src, t_dst):
    """Zero-order hold resample; NaN before the first sample."""
    if len(t_src) == 0:
        return np.full(len(t_dst), np.nan)
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    out = np.asarray(v_src, dtype=float)[np.clip(idx, 0, None)]
    out = np.where(idx < 0, np.nan, out)
    return out


def interp(t_src, v_src, t_dst, max_gap=None):
    if len(t_src) < 2:
        return np.full(len(t_dst), np.nan)
    out = np.interp(t_dst, t_src, v_src, left=np.nan, right=np.nan)
    if max_gap:
        idx = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
        gap = t_src[idx] - t_src[idx - 1]
        out = np.where(gap > max_gap, np.nan, out)
    return out


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quat(x, y, z, w):
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def r(x, nd=3):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def stat(a, nd=3):
    a = finite(a)
    if len(a) == 0:
        return None
    return {
        "min": r(a.min(), nd),
        "max": r(a.max(), nd),
        "mean": r(a.mean(), nd),
        "p50": r(np.percentile(a, 50), nd),
        "p95": r(np.percentile(a, 95), nd),
    }


def episodes(mask, t, min_duration=0.0):
    """(start_i, end_i) of contiguous True runs lasting >= min_duration s."""
    out = []
    if len(mask) == 0:
        return out
    m = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(int)))
    for a, b in zip(edges[::2], edges[1::2]):
        b -= 1
        if t[b] - t[a] >= min_duration:
            out.append((a, b))
    return out


def rigid_fit(src, dst):
    """2D rotation + translation taking src (N,2) onto dst (N,2)."""
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    theta = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    trans = cd - rot @ cs
    resid = dst - (src @ rot.T + trans)
    return theta, trans, float(np.sqrt((resid**2).sum(1).mean()))


# ============================================================ track frame


def track_transform(streams, track, timebase):
    """(theta, tx, ty, method, rms) mapping ZED map-frame xy onto the track."""
    pose = streams.get(ZED + "/pose")
    if pose is None:
        return None
    if "/clock" in streams:
        # Gazebo's pose IS ground truth in the world frame, which is the
        # track frame.  Better than fitting to the driver: the driver's own
        # latch is what it believed, and its beliefs are kept as tel_*.
        return (
            0.0,
            0.0,
            0.0,
            "simulation: Gazebo ground truth (world frame = track frame)",
            0.0,
        )
    tel = streams.get("/formula_one/telemetry")
    if tel is not None:
        running = np.isin(tel[:, 1], (1, 2))
        if running.sum() > 20:
            tr = tel[running]
            # The driver re-anchors on every manual_start (and a reset): its
            # track-frame position then jumps while the ZED pose does not.
            # Split there and fit each anchor on its own.
            step = np.hypot(np.diff(tr[:, 9]), np.diff(tr[:, 10]))
            cuts = np.flatnonzero(step > 1.0) + 1
            pieces, total, n_all = [], 0.0, 0
            for seg in np.split(np.arange(len(tr)), cuts):
                if len(seg) < 10:
                    continue
                tt = tr[seg, 0]
                px = interp(pose[:, 0], pose[:, 2], tt)
                py = interp(pose[:, 0], pose[:, 3], tt)
                ok = np.isfinite(px) & np.isfinite(py)
                if ok.sum() < 10:
                    continue
                theta, trans, rms = rigid_fit(
                    np.column_stack([px[ok], py[ok]]), tr[seg][ok][:, 9:11]
                )
                pieces.append((float(tt[0]), theta, trans[0], trans[1]))
                total += rms**2 * ok.sum()
                n_all += ok.sum()
            if pieces:
                rms = math.sqrt(total / n_all)
                method = "fitted to the driver's telemetry"
                if len(pieces) > 1:
                    method += (
                        f" ({len(pieces)} anchors: the driver re-anchored mid-run)"
                    )
                theta, tx, ty = pieces[0][1:]
                return theta, tx, ty, method, rms, pieces
    # The driver's own latch: first go, else first AUTO_ACTIVE, else first motion.
    t_latch = None
    go = streams.get("/start_signal_detector/go")
    if go is not None and (go[:, 1] > 0).any():
        t_latch = go[go[:, 1] > 0][0, 0]
    status = streams.get("/arduino_bridge/status")
    if t_latch is None and status is not None:
        active = status[:, 5] == 4
        if active.any():
            t_latch = status[active][0, 0]
        else:
            moving = np.abs(status[:, 1]) > STALL_SPEED
            if moving.any():
                t_latch = status[moving][0, 0] - 0.5
    if t_latch is None:
        t_latch = pose[0, 0]
    i = int(np.clip(np.searchsorted(pose[:, 0], t_latch), 0, len(pose) - 1))
    yaw0 = float(yaw_from_quat(*pose[i, 5:9]))
    k = int(np.clip(np.searchsorted(track.s, track.start_station), 0, len(track.s) - 1))
    yaw_start = math.atan2(track.ty[k], track.tx[k])
    theta = yaw_start - yaw0
    c, s = math.cos(theta), math.sin(theta)
    tx = track.x[k] - (c * pose[i, 2] - s * pose[i, 3])
    ty = track.y[k] - (s * pose[i, 2] + c * pose[i, 3])
    return theta, tx, ty, "latched at the start (no driver telemetry in the bag)", None


def apply_xy(transform, x, y, t=None):
    """Map-frame xy -> track frame.  With several anchors (a transform with
    a 6th element, [(t_from, theta, tx, ty), ...]) and times `t`, each point
    takes the anchor in force at its time."""
    pieces = transform[5] if len(transform) > 5 else None
    if pieces and t is not None and len(pieces) > 1:
        x, y, t = np.asarray(x, float), np.asarray(y, float), np.asarray(t, float)
        starts = np.array([p[0] for p in pieces])
        which = np.clip(
            np.searchsorted(starts, t, side="right") - 1, 0, len(pieces) - 1
        )
        th = np.array([p[1] for p in pieces])[which]
        tx = np.array([p[2] for p in pieces])[which]
        ty = np.array([p[3] for p in pieces])[which]
        c, s = np.cos(th), np.sin(th)
        return c * x - s * y + tx, s * x + c * y + ty
    theta, tx, ty = transform[:3]
    c, s = math.cos(theta), math.sin(theta)
    return c * x - s * y + tx, s * x + c * y + ty


def anchor_theta(transform, t):
    """Rotation of the anchor in force at log times `t`."""
    pieces = transform[5] if len(transform) > 5 else None
    if not pieces or len(pieces) < 2:
        return np.full(np.shape(t), transform[0])
    starts = np.array([p[0] for p in pieces])
    which = np.clip(
        np.searchsorted(starts, np.asarray(t, float), side="right") - 1,
        0,
        len(pieces) - 1,
    )
    return np.array([p[1] for p in pieces])[which]


# ================================================================ timebase


class Timebase:
    """Log time -> analysis time (sim time when the bag has /clock)."""

    def __init__(self, streams, bag_start):
        clock = streams.get("/clock")
        self.sim = clock is not None and len(clock) > 10
        if self.sim:
            self.log_t, self.sim_t = clock[:, 0], clock[:, 1]
            self.origin = float(self.sim_t[0])
        else:
            self.origin = bag_start

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        if self.sim:
            return np.interp(t, self.log_t, self.sim_t) - self.origin
        return t - self.origin


# ================================================================= process


def process(run_dir: Path, callback=None, clouds=True, images=True):
    progress = Progress(callback)
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis"
    tmp_dir = run_dir / "analysis.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    started = time.monotonic()

    bag_dir = bagio.find_bag(run_dir)
    if bag_dir is None:
        raise FileNotFoundError(f"no rosbag found in {run_dir}")
    metadata = {}
    if (run_dir / "metadata.yaml").exists():
        try:
            metadata = yaml.safe_load((run_dir / "metadata.yaml").read_text()) or {}
        except yaml.YAMLError:
            metadata = {}

    config = course_mod.load_config(run_dir)
    progress(0.02, "loading the course")
    track = course_mod.get_track(config)
    geometry = course_mod.geometry(config)
    sections = course_mod.zones(config)
    (tmp_dir / "course.json").write_text(json.dumps({**geometry, "sections": sections}))
    recording = rerun_export.RunRecording(tmp_dir / "recording.rrd", run_dir.name)
    recording.course(geometry)

    with bagio.Bag(bag_dir) as bag:
        progress(0.04, "reading the bag")
        streams = read_streams(bag, progress)
        tb = Timebase(streams, bag.start_ns * 1e-9)
        bag_info = {
            "path": str(bag_dir.relative_to(run_dir)) if bag_dir != run_dir else ".",
            "duration_s": r((bag.end_ns - bag.start_ns) * 1e-9, 2),
            "topics": streams["_topics"],
        }

        progress(0.42, "building the timeline")
        result = build(streams, tb, track, config, geometry, sections, metadata)

        if clouds:
            progress(0.55, "accumulating point clouds")
            result["summary"]["perception"].update(
                write_clouds(bag, streams, tb, result["transform"], recording, progress)
            )
        if images:
            progress(0.85, "extracting camera frames")
            result["summary"]["perception"].update(write_images(bag, tb, recording))

    summary = result["summary"]
    summary["bag"] = bag_info
    summary["topic_health"] = topic_health(streams, bag_info, tb)
    summary["system"].update(tegrastats(run_dir))
    summary["meta"] = {
        "run": run_dir.name,
        "version": VERSION,
        "processed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_s": round(time.monotonic() - started, 1),
        "metadata": metadata,
        "simulation": tb.sim,
        "driver": streams.get("_driver") or metadata.get("driver") or None,
    }
    progress(0.93, "writing the Rerun recording")
    recording.car(result["series"], geometry, summary["window"])
    recording.metrics(result["series"])
    recording.events(summary, result["series"])
    recording.logs(result["logs"])
    recording.close()
    progress(0.97, "writing results")
    (tmp_dir / "series.json").write_text(json.dumps(result["series"], allow_nan=False))
    (tmp_dir / "logs.json").write_text(json.dumps(result["logs"], allow_nan=False))
    (tmp_dir / "summary.json").write_text(
        json.dumps(clean(summary), allow_nan=False, indent=1)
    )
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    progress(1.0, "done")
    return summary


def clean(obj):
    """NaN/inf -> None, numpy -> python, recursively; JSON cannot carry NaN."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


# =================================================================== build


def build(st, tb, track, config, geometry, sections, metadata):
    pose = st.get(ZED + "/pose")
    status = st.get("/arduino_bridge/status")
    cmd = st.get("/drive_cmd")
    tel = st.get("/formula_one/telemetry")
    odom = st.get(ZED + "/odom")
    imu = st.get(ZED + "/imu/data")

    # ------------------------------------------------------------ timeline
    starts, ends = [], []
    for arr in (pose, status, cmd, tel):
        if arr is not None and len(arr):
            starts.append(tb(arr[0, 0]))
            ends.append(tb(arr[-1, 0]))
    if not starts:
        raise ValueError("the bag has none of pose, status, drive_cmd or telemetry")
    t0, t1 = max(0.0, min(starts)), max(ends)
    t = np.arange(t0, t1, 1.0 / GRID_HZ)
    cols = {"t": t}

    # ------------------------------------------------------------ car link
    if status is not None:
        ts = tb(status[:, 0])
        cols["speed"] = hold(ts, status[:, 1], t)
        cols["target_speed"] = hold(ts, status[:, 2], t)
        cols["throttle_us"] = hold(ts, status[:, 3], t)
        cols["battery"] = hold(
            ts,
            BATTERY_EMPTY_MV / 1000
            + status[:, 4] / 255 * (BATTERY_FULL_MV - BATTERY_EMPTY_MV) / 1000,
            t,
        )
        cols["mode"] = hold(ts, status[:, 5], t)
        cols["estop"] = hold(ts, status[:, 6], t)
        cols["link_ok"] = hold(ts, status[:, 7], t)
    if cmd is not None:
        tc = tb(cmd[:, 0])
        cols["cmd_steer"] = hold(tc, cmd[:, 1], t)
        cols["cmd_speed"] = hold(tc, cmd[:, 2], t)
        cols["auto_ready"] = hold(tc, cmd[:, 3], t)

    # ------------------------------------------------------------ pose
    transform = track_transform(st, track, tb) if pose is not None else None
    pose_info = {}
    if pose is not None:
        tp = tb(pose[:, 0])
        yaw_map = np.unwrap(
            yaw_from_quat(pose[:, 5], pose[:, 6], pose[:, 7], pose[:, 8])
        )
        stamp = pose[:, 1]
        if tb.sim:
            stamp = tb(pose[:, 0]) + tb.origin  # sim stamps ARE the clock
        steps = np.hypot(np.diff(pose[:, 2]), np.diff(pose[:, 3]))
        dts = np.diff(tp)
        jump_idx = np.flatnonzero(steps > np.maximum(0.3, 12.0 * np.maximum(dts, 0.02)))
        cols["x_map"] = interp(tp, pose[:, 2], t, max_gap=0.5)
        cols["y_map"] = interp(tp, pose[:, 3], t, max_gap=0.5)
        cols["yaw_map"] = interp(tp, yaw_map, t, max_gap=0.5)
        if transform:
            xt, yt = apply_xy(transform, pose[:, 2], pose[:, 3], pose[:, 0])
            yawt = yaw_map + anchor_theta(transform, pose[:, 0])
        else:
            xt, yt, yawt = pose[:, 2], pose[:, 3], yaw_map
        # Speed and yaw rate off the pose, over its own stamps, lightly
        # smoothed: a 30 Hz pose differenced raw is mostly noise.
        dt_stamp = np.gradient(stamp)
        dt_stamp = np.where(dt_stamp > 1e-4, dt_stamp, np.nan)
        vx = np.gradient(smooth(xt, 5)) / dt_stamp
        vy = np.gradient(smooth(yt, 5)) / dt_stamp
        v_pose = np.hypot(vx, vy)
        v_pose[np.clip(jump_idx + 1, 0, len(v_pose) - 1)] = np.nan
        yaw_rate = np.gradient(smooth(yawt, 5)) / dt_stamp
        yaw_rate[np.abs(yaw_rate) > 8] = np.nan
        cols["speed_pose"] = interp(tp, np.nan_to_num(v_pose, nan=0.0), t, max_gap=0.5)
        cols["yaw_rate"] = interp(tp, np.nan_to_num(yaw_rate, nan=0.0), t, max_gap=0.5)
        cols["x"] = interp(tp, xt, t, max_gap=0.5)
        cols["y"] = interp(tp, yt, t, max_gap=0.5)
        cols["yaw"] = wrap(interp(tp, yawt, t, max_gap=0.5))
        pose_info = {"t": tp, "jumps": jump_idx, "steps": steps, "dts": dts}

        # Frenet quantities off the analyser's own projection, so a run with
        # no driver telemetry still gets them.  Incremental like the driver.
        ok = np.isfinite(cols["x"])
        station = np.full(len(t), np.nan)
        lateral = np.full(len(t), np.nan)
        clearance = np.full(len(t), np.nan)
        heading = np.full(len(t), np.nan)
        if ok.any():
            first = np.flatnonzero(ok)[0]
            idx, _ = track.locate(
                cols["x"][first : first + 1],
                cols["y"][first : first + 1],
                cols["yaw"][first : first + 1],
            )
            hint = idx
            for i in np.flatnonzero(ok):
                near, s_, lat = track.project(
                    cols["x"][i : i + 1], cols["y"][i : i + 1], hint
                )
                hint = near
                station[i], lateral[i] = s_[0], lat[0]
                heading[i] = track.heading_error(cols["yaw"][i : i + 1], near)[0]
            half_l, half_w = (
                config["vehicle"]["length"] / 2,
                config["vehicle"]["width"] / 2,
            )
            clearance[ok] = track.body_clearance(
                cols["x"][ok], cols["y"][ok], cols["yaw"][ok], half_l, half_w
            )
        cols["station"] = station
        cols["cte"] = lateral
        cols["heading_err"] = heading
        cols["clearance"] = clearance
        cols["v_cap"] = np.where(
            np.isfinite(station), track.at(np.nan_to_num(station), track.v_cap), np.nan
        )

    if odom is not None and pose is not None:
        to = tb(odom[:, 0])
        ox, oy = interp(to, odom[:, 2], t, 0.5), interp(to, odom[:, 3], t, 0.5)
        cols["odom_divergence"] = np.hypot(ox - cols["x_map"], oy - cols["y_map"])

    if imu is not None:
        ti = tb(imu[:, 0])
        cols["imu_ax"] = interp(ti, smooth(imu[:, 1], 5), t, 0.2)
        cols["imu_ay"] = interp(ti, smooth(imu[:, 2], 5), t, 0.2)
        cols["imu_wz"] = interp(ti, smooth(imu[:, 6], 5), t, 0.2)

    # ------------------------------------------------------------ driver
    if tel is not None:
        tt = tb(tel[:, 0])
        names = [
            "state",
            "tel_station",
            "tel_distance",
            "lap",
            "laps_target",
            "lap_time",
            "last_lap_time",
            "race_time",
            "tel_x",
            "tel_y",
            "tel_yaw",
            "tel_cte",
            "tel_heading",
            "tel_clearance",
            "tel_speed",
            "tel_v_cap",
            "v_floor",
            "tel_yaw_rate",
            "speed_rate",
            "act_steer",
            "act_throttle",
            "steer_ff",
            "steer_out",
            "speed_out",
            "speed_scale",
            "pose_age",
            "speed_from_tach",
            "relocalized",
        ]
        for j, name in enumerate(names, start=1):
            cols[name] = hold(tt, tel[:, j], t)
        running = np.isin(cols["state"], (1, 2))
        # On the car, where the driver was running, ITS view is the best
        # record of what it decided on; keep ours for everywhere else.  In
        # simulation ours is ground truth, which beats the driver's belief.
        if tb.sim:
            running = np.zeros(len(t), dtype=bool)
        for ours, theirs in (
            ("station", "tel_station"),
            ("cte", "tel_cte"),
            ("clearance", "tel_clearance"),
            ("v_cap", "tel_v_cap"),
        ):
            if ours in cols:
                cols[ours] = np.where(running, cols[theirs], cols[ours])
            else:
                cols[ours] = np.where(running, cols[theirs], np.nan)
        cols["residual"] = cols["steer_out"] - cols["steer_ff"]

    # ------------------------------------------------------------ derived
    speed = cols.get("speed")
    if speed is None or not np.isfinite(speed).any():
        speed = cols.get("speed_pose", np.zeros(len(t)))
    speed_abs = np.abs(np.nan_to_num(speed))
    cols["accel"] = smooth(derivative(t, smooth(np.nan_to_num(speed), 5)), 5)
    if "yaw_rate" in cols:
        cols["lat_acc"] = speed_abs * cols["yaw_rate"]

    # Run window: driver RUNNING/STOPPING, else AUTO_ACTIVE, else moving.
    if "state" in cols and np.isin(cols["state"], (1, 2)).any():
        active = np.isin(cols["state"], (1, 2))
        window_source = "driver running"
    elif "mode" in cols and (cols["mode"] == 4).any():
        active = cols["mode"] == 4
        window_source = "Arduino AUTO_ACTIVE"
    else:
        active = speed_abs > STALL_SPEED
        window_source = "car moving"
    act_idx = np.flatnonzero(active)
    if len(act_idx):
        run_t0, run_t1 = t[act_idx[0]], t[act_idx[-1]]
    else:
        run_t0, run_t1 = t[0], t[-1]
    in_run = (t >= run_t0) & (t <= run_t1)
    racing = in_run & (cols["state"] == 1) if "state" in cols else in_run
    moving = in_run & (speed_abs > STALL_SPEED)

    # Progress & laps from the analyser's own station, unwrapped.
    laps_info = laps_from_station(t, cols.get("station"), track, in_run, cols)
    cols["progress"] = laps_info.pop("progress")

    events = build_events(t, cols, st, tb, pose_info, in_run, racing, laps_info)
    stats = {
        "speed": speed_stats(t, cols, in_run, moving, racing, config),
        "steering": steering_stats(t, cols, in_run, moving, config),
        "localization": localization_stats(t, cols, st, tb, pose_info, in_run),
        "policy": policy_stats(t, cols, in_run, racing, laps_info, config),
        "system": system_stats(t, cols, st, tb, in_run),
        "perception": {},
    }
    section_table = section_stats(t, cols, sections, track, in_run, laps_info)
    verdicts = build_verdicts(
        stats, laps_info, events, section_table, cols, in_run, pose_info, st
    )
    kpis = build_kpis(stats, laps_info, run_t0, run_t1)

    series_cols = {}
    for name, values in cols.items():
        arr = np.asarray(values, dtype=float)
        if not np.isfinite(arr).any():
            continue
        nd = 3 if name not in ("t",) else 3
        series_cols[name] = [
            None if not math.isfinite(v) else round(float(v), nd) for v in arr
        ]
    series = {
        "hz": GRID_HZ,
        "run_t0": r(run_t0),
        "run_t1": r(run_t1),
        "columns": series_cols,
    }

    logs = [
        {"t": r(float(tb(row[0])), 3), "level": row[1], "node": row[2], "msg": row[3]}
        for row in st.get("/rosout", [])
    ]
    summary = {
        "kpis": kpis,
        "verdicts": verdicts,
        "events": events,
        "laps": laps_info,
        "sections": section_table,
        "window": {"t0": r(run_t0), "t1": r(run_t1), "source": window_source},
        "frame": {
            "method": transform[3] if transform else "no pose in the bag",
            "fit_rms_m": r(transform[4])
            if transform and transform[4] is not None
            else None,
            "anchors": len(transform[5]) if transform and len(transform) > 5 else 1,
            "theta_deg": r(math.degrees(transform[0]), 2) if transform else None,
        },
        **stats,
        "log_counts": log_counts(logs),
    }
    return {"summary": summary, "series": series, "logs": logs, "transform": transform}


# =================================================================== laps


def laps_from_station(t, station, track, in_run, cols):
    out = {
        "laps": [],
        "completed": 0,
        "target": None,
        "race_time": None,
        "finished": False,
        "progress": np.full(len(t), np.nan),
    }
    if "laps_target" in cols and np.isfinite(cols["laps_target"]).any():
        out["target"] = int(np.nanmax(cols["laps_target"]))
    if station is None or not np.isfinite(station[in_run]).any():
        return out
    idx = np.flatnonzero(in_run & np.isfinite(station))
    s = station[idx]
    ds = np.diff(s)
    ds = (ds + track.length / 2) % track.length - track.length / 2
    ds[np.abs(ds) > 2.0] = 0.0  # a re-localisation, not driving
    prog = np.concatenate([[0.0], np.cumsum(ds)])
    out["progress"][idx] = prog
    # Laps are counted at the start line, as the driver counts them: the car
    # parks ON the start station, so progress = k * length.
    t_run = t[idx]
    boundaries = []
    k = 1
    while True:
        hit = np.flatnonzero(prog >= k * track.length)
        if not len(hit):
            break
        j = hit[0]
        # Interpolate the crossing inside the tick.
        if j > 0 and prog[j] > prog[j - 1]:
            f = (k * track.length - prog[j - 1]) / (prog[j] - prog[j - 1])
            boundaries.append(t_run[j - 1] + f * (t_run[j] - t_run[j - 1]))
        else:
            boundaries.append(t_run[j])
        k += 1
    moving = (
        np.abs(
            np.nan_to_num(cols.get("speed", cols.get("speed_pose", np.zeros(len(t)))))
        )
        > STALL_SPEED
    )
    mv = np.flatnonzero(in_run & moving)
    t_go = t[mv[0]] if len(mv) else t_run[0]
    prev = t_go
    for n, tb_ in enumerate(boundaries, start=1):
        seg = (t >= prev) & (t < tb_)
        sp = np.abs(np.nan_to_num(cols.get("speed", cols.get("speed_pose")))[seg])
        clr = cols["clearance"][seg] if "clearance" in cols else np.array([])
        cte = cols["cte"][seg] if "cte" in cols else np.array([])
        out["laps"].append(
            {
                "lap": n,
                "t0": r(prev),
                "t1": r(tb_),
                "time": r(tb_ - prev, 2),
                "max_speed": r(sp.max() if len(sp) else None, 2),
                "mean_speed": r(track.length / (tb_ - prev), 2),
                "min_clearance": r(np.nanmin(clr) if np.isfinite(clr).any() else None),
                "mean_abs_cte": r(
                    np.nanmean(np.abs(cte)) if np.isfinite(cte).any() else None
                ),
            }
        )
        prev = tb_
    out["completed"] = len(boundaries)
    out["t_go"] = r(t_go)
    if out["target"] and len(boundaries) >= out["target"]:
        out["finished"] = True
        out["race_time"] = r(boundaries[out["target"] - 1] - t_go, 2)
    out["distance_m"] = r(prog[-1], 1)
    for i in range(1, len(out["laps"])):
        out["laps"][i]["delta"] = r(
            out["laps"][i]["time"] - out["laps"][i - 1]["time"], 2
        )
    return out


# ================================================================== events


def build_events(t, c, st, tb, pose_info, in_run, racing, laps):
    ev = []

    def add(time_s, kind, severity, text, station=None):
        ev.append(
            {
                "t": r(time_s, 2),
                "kind": kind,
                "severity": severity,
                "text": text,
                "station": r(station, 1),
            }
        )

    def station_at(i):
        return (
            float(c["station"][i])
            if "station" in c and np.isfinite(c["station"][i])
            else None
        )

    go = st.get("/start_signal_detector/go")
    if go is not None and (go[:, 1] > 0).any():
        add(float(tb(go[go[:, 1] > 0][0, 0])), "start", "info", "start signal: GREEN")
    if laps.get("t_go") is not None:
        add(laps["t_go"], "start", "info", "car starts moving")
    for lap in laps.get("laps", []):
        extra = (
            f" ({lap['delta']:+.2f} s vs previous)"
            if lap.get("delta") is not None
            else ""
        )
        add(
            lap["t1"],
            "lap",
            "good",
            f"lap {lap['lap']} complete in {lap['time']:.2f} s{extra}",
        )
    if "state" in c:
        st_ = c["state"]
        for code, kind, sev, text in (
            (2, "finish", "good", "laps done, coasting to a stop"),
            (3, "finish", "good", "stopped: run complete"),
            (4, "pose", "bad", "driver: pose stale, commanding neutral"),
            (5, "stop", "warn", "manual stop"),
        ):
            for a, b in episodes(st_ == code, t):
                add(
                    t[a],
                    kind,
                    sev,
                    text + (f" for {t[b] - t[a]:.1f} s" if code == 4 else ""),
                    station_at(a),
                )
        if "relocalized" in c:
            for a, _ in episodes(c["relocalized"] > 0.5, t):
                add(
                    t[a],
                    "pose",
                    "warn",
                    "driver re-localised after a pose jump",
                    station_at(a),
                )

    if "clearance" in c:
        clr = np.where(in_run, c["clearance"], np.nan)
        for a, b in episodes(np.nan_to_num(clr, nan=9) < 0.0, t):
            m = a + int(np.nanargmin(clr[a : b + 1]))
            add(
                t[m],
                "contact",
                "bad",
                f"CONTACT with a bale (map clearance {clr[m]:+.3f} m, {t[b] - t[a] + 0.05:.2f} s)",
                station_at(m),
            )
        for a, b in episodes(
            (np.nan_to_num(clr, nan=9) < GRAZE) & (np.nan_to_num(clr, nan=9) >= 0.0), t
        ):
            m = a + int(np.nanargmin(clr[a : b + 1]))
            add(
                t[m],
                "graze",
                "warn",
                f"graze band: {clr[m] * 100:.0f} mm from a bale",
                station_at(m),
            )

    speed = np.abs(np.nan_to_num(c.get("speed", c.get("speed_pose", np.zeros(len(t))))))
    for a, b in episodes(racing & (speed < STALL_SPEED), t, min_duration=1.5):
        add(
            t[a],
            "stall",
            "bad",
            f"stalled for {t[b] - t[a]:.1f} s while racing",
            station_at(a),
        )

    if "v_cap" in c:
        over = np.where(racing, speed - c["v_cap"], np.nan)
        for a, b in episodes(np.nan_to_num(over, nan=-9) > 0.15, t, min_duration=0.3):
            m = a + int(np.nanargmax(over[a : b + 1]))
            add(
                t[m],
                "overspeed",
                "warn",
                f"over the rule cap by {over[m]:.2f} m/s for {t[b] - t[a]:.1f} s",
                station_at(m),
            )

    if pose_info:
        tp, steps, dts = pose_info["t"], pose_info["steps"], pose_info["dts"]
        for j in pose_info["jumps"]:
            add(
                tp[j + 1],
                "pose",
                "bad" if steps[j] > 1.0 else "warn",
                f"ZED pose jumped {steps[j]:.2f} m in {dts[j] * 1000:.0f} ms",
                None,
            )
        for j in np.flatnonzero(dts > 0.25):
            if tp[j] >= t[in_run][0] if in_run.any() else True:
                add(tp[j], "pose", "warn", f"ZED pose gap of {dts[j]:.2f} s", None)

    if "link_ok" in c:
        for a, b in episodes(in_run & (c["link_ok"] < 0.5), t):
            add(
                t[a],
                "link",
                "bad",
                f"Arduino link lost for {t[b] - t[a] + 0.05:.2f} s",
                station_at(a),
            )
    if "estop" in c:
        for a, _ in episodes(c["estop"] > 0.5, t):
            add(t[a], "estop", "bad", "E-STOP", station_at(a))
    if "mode" in c:
        m = c["mode"]
        change = np.flatnonzero(np.diff(np.nan_to_num(m, nan=-1)) != 0) + 1
        for i in change:
            if np.isfinite(m[i]):
                add(
                    t[i],
                    "mode",
                    "info",
                    f"Arduino mode -> {MODE_NAMES.get(int(m[i]), int(m[i]))}",
                    station_at(i),
                )

    for row in st.get("/rosout", []):
        if row[1] >= 40:
            add(float(tb(row[0])), "log", "bad", f"[{row[2]}] {row[3][:160]}")

    ev.sort(key=lambda e: e["t"] if e["t"] is not None else 0)
    # Cap chatty kinds so one pathological run cannot bury the rest.
    capped, per_kind = [], {}
    for e in ev:
        per_kind[e["kind"]] = per_kind.get(e["kind"], 0) + 1
        if per_kind[e["kind"]] <= 150:
            capped.append(e)
    return capped


# =================================================================== stats


def speed_stats(t, c, in_run, moving, racing, config):
    tach = c.get("speed")
    sp = (
        np.abs(tach)
        if tach is not None and np.isfinite(tach).any()
        else np.abs(c.get("speed_pose", np.zeros(len(t))))
    )
    go = np.flatnonzero(racing & moving)
    settled = racing & moving & (t >= (t[go[0]] + 3.0 if len(go) else np.inf))
    out = {
        "source": "tachometer"
        if tach is not None and np.isfinite(tach).any()
        else "ZED pose",
        "max": r(np.nanmax(np.where(in_run, sp, np.nan)) if in_run.any() else None, 2),
        "moving": stat(sp[moving], 2),
        # After the launch: the standing start is not "the slowest it raced".
        "min_while_racing": r(
            np.nanmin(np.where(settled, sp, np.nan)) if settled.any() else None, 2
        ),
        "time_moving_s": r(moving.sum() / GRID_HZ, 1),
        # 99th percentiles: the tachometer reads nothing below ~0.3 m/s, so
        # the launch is a 0 -> 0.3 step that differentiates to a spike.
        "accel": {
            "max_accel": r(
                np.nanpercentile(c["accel"][moving], 99) if moving.sum() > 20 else None,
                2,
            ),
            "max_decel": r(
                -np.nanpercentile(c["accel"][moving], 1) if moving.sum() > 20 else None,
                2,
            ),
            "note": "99th percentile",
        },
    }
    if "speed_pose" in c and tach is not None and np.isfinite(tach).any():
        steady = moving & (np.abs(c["accel"]) < 0.5) & (sp > 1.0)
        ratio = finite(np.abs(c["speed_pose"][steady]) / np.maximum(sp[steady], 1e-3))
        if len(ratio) > 20:
            out["pose_vs_tach"] = {
                "median_ratio": r(np.median(ratio), 3),
                "note": "ZED ground speed / tachometer, steady driving over 1 m/s.  "
                "Far from 1.0 means the tire diameter, wheel slip or the pose scale is off.",
            }
    # Speed tracking, in two halves, because they fail for different reasons:
    #   command -> target   the bridge's slew limit (speed_slew_rate); lag
    #                       here is by design, not a fault
    #   target  -> measured the Arduino's closed speed loop and the drivetrain
    # The coast-to-stop phase is excluded: throttle is zero there on purpose.
    cmd = c.get("cmd_speed")
    target = c.get("target_speed")
    driving = racing if "state" in c else in_run
    if cmd is not None and tach is not None:
        m = driving & np.isfinite(cmd) & np.isfinite(tach)
        if m.sum() > 40:
            meas = np.abs(tach[m])
            track = {
                "cmd_to_measured_rms": r(np.sqrt(np.mean((cmd[m] - meas) ** 2)), 3),
                "cmd_to_measured_lag_s": r(best_lag(cmd[m], meas, 2.0), 2),
                "model_dead_time_s": config.get("plant", {}).get("command_dead_time"),
                "slew_limit": config.get("plant", {}).get("speed_slew_rate"),
            }
            if target is not None and np.isfinite(target[m]).any():
                tg = np.abs(target[m])
                track["target_to_measured_rms"] = r(
                    np.sqrt(np.nanmean((tg - meas) ** 2)), 3
                )
                track["target_to_measured_lag_s"] = r(
                    best_lag(np.nan_to_num(tg), meas, 2.0), 2
                )
                track["cmd_to_target_rms"] = r(
                    np.sqrt(np.nanmean((cmd[m] - tg) ** 2)), 3
                )
            out["tracking"] = track
    # Coast-down: throttle zero and still rolling -> the drag the plant assumes.
    if cmd is not None and tach is not None:
        coast = in_run & (np.nan_to_num(cmd, nan=1) < 0.05) & (sp > 0.6)
        if coast.sum() > 20:
            v = sp[coast]
            a = -c["accel"][coast]
            ok = np.isfinite(a) & (a > -1) & (a < 5)
            if ok.sum() > 20:
                slope, icpt = np.polyfit(v[ok], a[ok], 1)
                plant = config.get("plant", {})
                m_ = plant.get("mass", 3.599)
                out["coast"] = {
                    "fit": f"decel = {icpt:.3f} + {slope:.3f} v  m/s^2",
                    "decel_at_2ms": r(icpt + 2 * slope, 3),
                    "model_at_2ms": r(
                        (plant.get("coast_f0", 2.18) + 2 * plant.get("coast_f1", 0.47))
                        / m_,
                        3,
                    ),
                    "samples": int(ok.sum()),
                }
    if "v_cap" in c:
        over = np.where(racing, sp - c["v_cap"], np.nan)
        use = np.where(racing & moving, sp / np.maximum(c["v_cap"], 0.1), np.nan)
        out["cap"] = {
            "max_over": r(np.nanmax(over) if np.isfinite(over).any() else None, 2),
            "time_over_s": r(np.sum(np.nan_to_num(over, nan=-1) > 0.05) / GRID_HZ, 2),
            "mean_use_of_cap": r(
                np.nanmean(use) if np.isfinite(use).any() else None, 3
            ),
        }
        if "v_floor" in c:
            under = np.where(racing & moving, c["v_floor"] - sp, np.nan)
            out["cap"]["time_under_floor_s"] = r(
                np.sum(np.nan_to_num(under, nan=-1) > 0.1) / GRID_HZ, 2
            )
    return out


def best_lag(cmd, resp, max_lag):
    """Shift (s) of cmd that best explains resp, by least squares."""
    n = int(max_lag * GRID_HZ)
    best, arg = np.inf, 0
    for k in range(0, n + 1):
        a = cmd[: len(cmd) - k] if k else cmd
        b = resp[k:]
        e = np.mean((a - b) ** 2)
        if e < best:
            best, arg = e, k
    return arg / GRID_HZ


def steering_stats(t, c, in_run, moving, config):
    if "cmd_steer" not in c:
        return {}
    cs = c["cmd_steer"]
    m = in_run & np.isfinite(cs)
    if m.sum() < 10:
        return {}
    s = cs[m]
    d1 = np.diff(s)
    d2 = np.diff(s, 2)
    signs = np.sign(d1[np.abs(d1) > 0.02])
    reversals = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    out = {
        "max_left": r(s.max(), 3),
        "max_right": r(s.min(), 3),
        "mean_abs": r(np.mean(np.abs(s)), 3),
        "saturated_pct": r(100 * np.mean(np.abs(s) > 0.98), 2),
        "rate_rms_per_s": r(np.sqrt(np.mean(d1**2)) * GRID_HZ, 3),
        "jerk_ms": r(np.mean(d2**2), 5),
        "reversals_per_s": r(reversals / max(m.sum() / GRID_HZ, 1e-3), 2),
        "left_share_pct": r(100 * np.mean(s > 0.02), 1),
        "right_share_pct": r(100 * np.mean(s < -0.02), 1),
    }
    # How much the car actually turns per command, against the PLANT the
    # policy was trained on: R = tire_scrub * (L + K v^2) / tan(delta), with
    # delta from the measured command->angle table (plant.py).  Pure
    # kinematics would call a correctly-modelled car 25% short of authority.
    plant = config.get("plant", {})
    cmd_pts = np.array(plant.get("steering_command_points", [-1, -0.5, 0, 0.5, 1]))
    ang_pts = np.array(
        plant.get("steering_angle_points", [-0.382, -0.191, 0, 0.256, 0.512])
    )
    L = float(plant.get("wheelbase", 0.324))
    K = float(plant.get("understeer_gradient", 0.0))
    scrub = float(plant.get("tire_scrub", 1.0))
    if "yaw_rate" in c:
        sp = np.abs(np.nan_to_num(c.get("speed", c.get("speed_pose"))))
        ok = moving & (sp > 1.0) & np.isfinite(c["yaw_rate"]) & np.isfinite(cs)
        kappa_meas = c["yaw_rate"] / np.maximum(sp, 0.1)
        kappa_plant = np.tan(np.interp(cs, cmd_pts, ang_pts)) / (
            (L + K * sp**2) * scrub
        )
        # Steady corners, for the table: command held within 0.08 for 0.6 s
        # and a real turn (|cmd| > 0.2) -- near-straight samples make the
        # ratio of two small numbers and say nothing about authority.
        window = int(0.6 * GRID_HZ)
        rolling = np.array(
            [
                np.ptp(cs[max(0, i - window) : i + 1])
                if np.isfinite(cs[max(0, i - window) : i + 1]).all()
                else 9
                for i in range(len(cs))
            ]
        )
        steady = ok & (rolling < 0.08) & (np.abs(cs) > 0.2)
        bins = np.linspace(-1, 1, 11)
        table = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = steady & (cs >= lo) & (cs < hi)
            if sel.sum() >= 3:
                table.append(
                    {
                        "cmd": r((lo + hi) / 2, 2),
                        "n": int(sel.sum()),
                        "kappa_measured": r(np.median(kappa_meas[sel]), 3),
                        "kappa_plant": r(np.median(kappa_plant[sel]), 3),
                    }
                )
        out["authority"] = table
        out["steady_samples"] = int(steady.sum())
        # Dynamic fit over the whole run: yaw rate = lag(gain_side * plant
        # rate).  Uses every moving sample, not just the rare steady ones --
        # a policy seldom holds a command still for long.
        # What the plant predicts for this exact command stream: dead time,
        # servo lag and chassis lag as modelled.  Against the measured yaw
        # rate this is the sim-to-real gap, drawn.
        implied_all = np.where(np.isfinite(kappa_plant), sp * kappa_plant, 0.0)
        delay = int(round((plant.get("command_dead_time") or 0.0) * GRID_HZ))
        u = (
            np.concatenate([np.zeros(delay), implied_all[: len(implied_all) - delay]])
            if delay
            else implied_all
        )
        tau = (plant.get("yaw_response_tau") or 0.0) + (
            plant.get("steering_tau") or 0.0
        )
        c["yaw_rate_plant"] = np.where(
            moving, _first_order(u, (1 / GRID_HZ) / (tau + 1 / GRID_HZ), 0.0), np.nan
        )
        if ok.sum() > 100:
            implied = sp * kappa_plant
            fit = fit_lag(t[ok], implied[ok], c["yaw_rate"][ok])
            if fit:
                out["yaw_lag"] = {
                    **fit,
                    "model_tau_s": plant.get("yaw_response_tau"),
                    "model_dead_time_s": plant.get("command_dead_time"),
                    "model_servo_tau_s": plant.get("steering_tau"),
                }
                out["gain_left"] = fit["gain_left"]
                out["gain_right"] = fit["gain_right"]
    return out


def _first_order(u, alpha, y0):
    est = np.empty_like(u)
    acc = y0
    for i, ui in enumerate(u):
        acc += alpha * (ui - acc)
        est[i] = acc
    return est


def fit_lag(t, implied, measured):
    """Fit measured ~ lag_{delay,tau}(g_L * u+ + g_R * u-) by grid + least squares.

    The filter is linear, so for each (delay, tau) the two side gains come
    from one 2-column least-squares solve; only delay and tau are gridded.
    Returns None when there is too little turning to say anything.
    """
    dt = 1.0 / GRID_HZ
    breaks = np.flatnonzero(np.diff(t) > 1.5 * dt)
    segments = [s_ for s_ in np.split(np.arange(len(t)), breaks + 1) if len(s_) > 40]
    if not segments:
        return None
    y = np.concatenate([measured[s_] for s_ in segments])
    scale = math.sqrt(np.mean(y**2)) or 1.0
    if scale < 0.05:
        return None
    best, grid = None, {}
    for delay in np.arange(0.0, 0.41, 0.05):
        k = int(round(delay / dt))
        for tau in np.arange(0.0, 1.51, 0.03):
            alpha = dt / (tau + dt)
            cols = []
            for side in (1, -1):
                parts = []
                for s_ in segments:
                    u = implied[s_] * ((implied[s_] * side) > 0)
                    if k:
                        u = np.concatenate([np.full(k, u[0]), u[:-k]])
                    parts.append(_first_order(u, alpha, 0.0))
                cols.append(np.concatenate(parts))
            X = np.column_stack(cols)
            g, *_ = np.linalg.lstsq(X, y, rcond=None)
            e = math.sqrt(np.mean((X @ g - y) ** 2)) / scale
            grid[(round(delay, 2), round(tau, 2))] = (e, g)
            if best is None or e < best[0]:
                best = (e, delay, tau, g)
    e0, g0 = grid[(0.0, 0.0)]
    return {
        "tau_s": r(best[2], 2),
        "dead_time_s": r(best[1], 2),
        "gain_left": r(best[3][0], 3),
        "gain_right": r(best[3][1], 3),
        "relative_error": r(best[0], 3),
        "relative_error_no_lag": r(e0, 3),
        "at_search_bound": bool(best[2] >= 1.5 - 1e-6 or best[1] >= 0.4 - 1e-6),
        "samples": int(len(y)),
    }


def localization_stats(t, c, st, tb, pose_info, in_run):
    out = {}
    pose = st.get(ZED + "/pose")
    if pose is None:
        return {"available": False}
    tp = pose_info["t"]
    dts = pose_info["dts"]
    run = (tp[:-1] >= (t[in_run][0] if in_run.any() else tp[0])) & (
        tp[:-1] <= (t[in_run][-1] if in_run.any() else tp[-1])
    )
    out["available"] = True
    out["rate_hz"] = r(1.0 / np.median(dts) if len(dts) else None, 1)
    out["gaps_over_100ms"] = int(np.sum(dts[run] > 0.1))
    out["max_gap_s"] = r(dts[run].max() if run.any() else None, 3)
    out["jumps"] = int(len(pose_info["jumps"]))
    out["largest_jump_m"] = r(
        pose_info["steps"][pose_info["jumps"]].max()
        if len(pose_info["jumps"])
        else 0.0,
        2,
    )
    if "odom_divergence" in c:
        d = c["odom_divergence"][in_run]
        out["odom_vs_pose_end_m"] = r(finite(d)[-1] if len(finite(d)) else None, 3)
        out["odom_vs_pose_max_m"] = r(np.nanmax(d) if np.isfinite(d).any() else None, 3)
    lc = st.get("/lap_counter/count")
    if lc is not None:
        out["lap_counter_loop_closures"] = int(lc[:, 5].max())
        out["lap_counter_rejected_crossings"] = int(lc[:, 4].max())
    status = st.get(ZED + "/pose/status")
    if status:
        fields = sorted({k for row in status for k in row if k != "t"})
        dist = {}
        for f in fields:
            values = [row.get(f) for row in status if f in row]
            uniq, counts = np.unique(values, return_counts=True)
            dist[f] = {
                str(int(u)): r(100 * cnt / len(values), 1)
                for u, cnt in zip(uniq, counts)
            }
        out["tracking_status_pct"] = dist
    if "tel_x" in c:
        run_m = in_run & np.isin(c.get("state", np.zeros(len(t))), (1, 2))
        dx = np.hypot(c["tel_x"] - c["x"], c["tel_y"] - c["y"])
        if run_m.any() and np.isfinite(dx[run_m]).any():
            out["driver_vs_analyser_m"] = r(np.nanmax(dx[run_m]), 3)
    return out


def policy_stats(t, c, in_run, racing, laps, config):
    out = {}
    if "cte" in c:
        cte = c["cte"][racing] if racing.any() else c["cte"][in_run]
        out["cte"] = {
            "mean_abs": r(
                np.nanmean(np.abs(cte)) if np.isfinite(cte).any() else None, 3
            ),
            "rms": r(
                np.sqrt(np.nanmean(cte**2)) if np.isfinite(cte).any() else None, 3
            ),
            "max_abs": r(np.nanmax(np.abs(cte)) if np.isfinite(cte).any() else None, 3),
        }
    if "heading_err" in c:
        h = c["heading_err"][racing] if racing.any() else c["heading_err"][in_run]
        out["heading_rms_deg"] = r(
            np.degrees(np.sqrt(np.nanmean(h**2))) if np.isfinite(h).any() else None, 2
        )
    if "clearance" in c:
        clr = np.where(racing if racing.any() else in_run, c["clearance"], np.nan)
        if np.isfinite(clr).any():
            i = int(np.nanargmin(clr))
            out["clearance"] = {
                "min": r(clr[i], 3),
                "at_t": r(t[i], 2),
                "at_station": r(c["station"][i], 1) if "station" in c else None,
                "time_in_graze_s": r(
                    np.sum(np.nan_to_num(clr, nan=9) < GRAZE) / GRID_HZ, 2
                ),
                "time_in_contact_s": r(
                    np.sum(np.nan_to_num(clr, nan=9) < 0) / GRID_HZ, 2
                ),
                "p05": r(np.nanpercentile(clr, 5), 3),
            }
    if "act_steer" in c:
        a_s, a_t = c["act_steer"][racing], c["act_throttle"][racing]
        res = c["residual"][racing]
        out["actions"] = {
            "steer_saturated_pct": r(
                100 * np.nanmean(np.abs(a_s) > 0.99)
                if np.isfinite(a_s).any()
                else None,
                2,
            ),
            "throttle_saturated_pct": r(
                100 * np.nanmean(np.abs(a_t) > 0.99)
                if np.isfinite(a_t).any()
                else None,
                2,
            ),
            "throttle_mean": r(np.nanmean(a_t) if np.isfinite(a_t).any() else None, 3),
            "residual_rms": r(
                np.sqrt(np.nanmean(res**2)) if np.isfinite(res).any() else None, 3
            ),
            "prior_rms": r(
                np.sqrt(np.nanmean(c["steer_ff"][racing] ** 2))
                if racing.any()
                else None,
                3,
            ),
        }
        steady_res = finite(res)
        if len(steady_res):
            out["actions"]["residual_share"] = r(
                np.sqrt(np.mean(steady_res**2))
                / max(np.sqrt(np.nanmean(c["steer_out"][racing] ** 2)), 1e-3),
                3,
            )
    if "pose_age" in c:
        age = c["pose_age"][in_run]
        out["pose_age_p95_s"] = r(
            np.nanpercentile(age, 95) if np.isfinite(age).any() else None, 3
        )
    if "speed_from_tach" in c:
        tach = c["speed_from_tach"][racing]
        out["speed_from_tach_pct"] = r(
            100 * np.nanmean(tach) if np.isfinite(tach).any() else None, 1
        )
    out["laps_completed"] = laps.get("completed")
    out["laps_target"] = laps.get("target")
    out["race_time_s"] = laps.get("race_time")
    return out


def system_stats(t, c, st, tb, in_run):
    out = {}
    if "battery" in c:
        b = c["battery"][in_run] if in_run.any() else c["battery"]
        fb = finite(b)
        if len(fb):
            out["battery"] = {
                "start_v": r(fb[0], 2),
                "end_v": r(fb[-1], 2),
                "min_v": r(fb.min(), 2),
                "note": "approximate, from the 0-255 level code",
            }
    if "link_ok" in c:
        lk = c["link_ok"][in_run]
        out["link_ok_pct"] = r(
            100 * np.nanmean(lk) if np.isfinite(lk).any() else None, 2
        )
    if "mode" in c:
        m = c["mode"][in_run]
        uniq, cnt = np.unique(m[np.isfinite(m)], return_counts=True)
        out["mode_time_s"] = {
            MODE_NAMES.get(int(u), str(int(u))): r(n / GRID_HZ, 1)
            for u, n in zip(uniq, cnt)
        }
    if "estop" in c:
        out["estop_events"] = len(episodes(c["estop"] > 0.5, t))
    return out


def log_counts(logs):
    counts = {"debug": 0, "info": 0, "warn": 0, "error": 0, "fatal": 0}
    names = {10: "debug", 20: "info", 30: "warn", 40: "error", 50: "fatal"}
    for row in logs:
        counts[names.get(row["level"], "info")] += 1
    return counts


def section_stats(t, c, sections, track, in_run, laps):
    if "station" not in c:
        return []
    speed = np.abs(np.nan_to_num(c.get("speed", c.get("speed_pose", np.zeros(len(t))))))
    station = c["station"]
    lap_bounds = [(lap["t0"], lap["t1"]) for lap in laps.get("laps", [])]
    rows = []
    for sec in sections:
        s0, s1 = sec["s0"] % track.length, sec["s1"] % track.length
        inside = (
            ((station >= s0) & (station < s1))
            if s0 < s1
            else ((station >= s0) | (station < s1))
        )
        inside &= in_run & np.isfinite(station)
        if not inside.any():
            continue
        per_lap = []
        for n, (a, b) in enumerate(lap_bounds, start=1):
            m = inside & (t >= a) & (t < b)
            if m.sum() >= 2:
                per_lap.append({"lap": n, **_sec_row(t, c, speed, m)})
        rows.append({**sec, "all": _sec_row(t, c, speed, inside), "per_lap": per_lap})
    # Rank: worst clearance first is what "what went wrong" means here.
    for row in rows:
        clr = row["all"].get("min_clearance")
        row["risk"] = (
            "bad"
            if clr is not None and clr < 0
            else "warn"
            if clr is not None and clr < GRAZE
            else "good"
        )
    return rows


def _sec_row(t, c, speed, m):
    clr = c["clearance"][m] if "clearance" in c else np.array([np.nan])
    cte = c["cte"][m] if "cte" in c else np.array([np.nan])
    cap = c["v_cap"][m] if "v_cap" in c else np.array([np.nan])
    idx = np.flatnonzero(m)
    return {
        "time": r(len(idx) / GRID_HZ, 2),
        "entry_speed": r(speed[idx[0]], 2),
        "exit_speed": r(speed[idx[-1]], 2),
        "min_speed": r(speed[m].min(), 2),
        "max_speed": r(speed[m].max(), 2),
        "mean_cap": r(np.nanmean(cap) if np.isfinite(cap).any() else None, 2),
        "max_over_cap": r(
            np.nanmax(speed[m] - cap) if np.isfinite(cap).any() else None, 2
        ),
        "min_clearance": r(np.nanmin(clr) if np.isfinite(clr).any() else None, 3),
        "max_abs_cte": r(np.nanmax(np.abs(cte)) if np.isfinite(cte).any() else None, 3),
        "steer_rms": r(
            np.sqrt(np.nanmean(c["cmd_steer"][m] ** 2)) if "cmd_steer" in c else None, 3
        ),
    }


# ================================================================ verdicts


def build_verdicts(stats, laps, events, sections, c, in_run, pose_info, st):
    """What worked and what did not, in plain words, each with a jump target."""
    good, bad = [], []

    def v(lst, title, detail, t=None, severity=None):
        lst.append(
            {
                "title": title,
                "detail": detail,
                "t": r(t, 2) if t is not None else None,
                "severity": severity or ("good" if lst is good else "bad"),
            }
        )

    target = laps.get("target")
    done = laps.get("completed", 0)
    if target:
        if laps.get("finished"):
            v(
                good,
                f"Finished {target} laps in {laps['race_time']:.2f} s",
                f"{laps.get('distance_m')} m driven; lap times "
                + ", ".join(f"{lap['time']:.2f} s" for lap in laps["laps"]),
                laps["laps"][-1]["t1"] if laps["laps"] else None,
            )
        else:
            v(
                bad,
                f"Did not finish: {done} of {target} laps",
                "See the event list for where it stopped.",
                None,
            )
    elif done:
        v(
            good,
            f"{done} lap(s) completed",
            ", ".join(f"{lap['time']:.2f} s" for lap in laps["laps"]),
        )
    if len(laps.get("laps", [])) >= 2:
        d = laps["laps"][1]["time"] - laps["laps"][0]["time"]
        (
            v(
                good,
                f"Lap 2 was {-d:.2f} s faster than lap 1",
                "The flying lap benefits from the rolling start.",
            )
            if d < 0
            else v(
                bad,
                f"Lap 2 was {d:.2f} s slower than lap 1",
                "A slower flying lap usually means the car was fighting the line or lifting early.",
                None,
                "warn",
            )
        )

    pol = stats["policy"]
    clr = pol.get("clearance")
    if clr:
        if clr["min"] is not None and clr["min"] < 0:
            v(
                bad,
                f"Hit a bale: map clearance reached {clr['min']:+.3f} m",
                f"At station {clr['at_station']} m.  Map clearance comes from the pose, so check the pose was sound there before blaming the policy.",
                clr["at_t"],
            )
        elif clr["min"] is not None and clr["min"] < GRAZE:
            v(
                bad,
                f"Entered the graze band: {clr['min'] * 1000:.0f} mm minimum clearance",
                f"{clr['time_in_graze_s']} s inside {GRAZE * 1000:.0f} mm, worst at station {clr['at_station']} m.",
                clr["at_t"],
                "warn",
            )
        elif clr["min"] is not None:
            v(
                good,
                f"Stayed clear of the bales: {clr['min'] * 1000:.0f} mm minimum",
                f"Never inside the {GRAZE * 1000:.0f} mm graze band.",
                clr["at_t"],
            )
    cte = pol.get("cte")
    if cte and cte["mean_abs"] is not None:
        (
            v(
                good,
                f"Held the line: mean |CTE| {cte['mean_abs'] * 100:.1f} cm",
                f"RMS {cte['rms'] * 100:.1f} cm, worst {cte['max_abs'] * 100:.1f} cm.",
            )
            if cte["mean_abs"] < 0.06
            else v(
                bad,
                f"Loose line: mean |CTE| {cte['mean_abs'] * 100:.1f} cm",
                f"Worst {cte['max_abs'] * 100:.1f} cm.  The corridor leaves about +/-26 cm of play.",
                None,
                "warn",
            )
        )

    sp = stats["speed"]
    cap = sp.get("cap") or {}
    if cap.get("max_over") is not None:
        (
            v(
                good,
                "Respected the rule cap",
                f"Worst {cap['max_over']:+.2f} m/s against the cap.",
            )
            if cap["max_over"] <= 0.15
            else v(
                bad,
                f"Exceeded the rule cap by {cap['max_over']:.2f} m/s",
                f"{cap['time_over_s']} s over.  The car has no brakes: an overspeed entering a hairpin is only fixable by lifting earlier.",
            )
        )
    if cap.get("mean_use_of_cap") is not None:
        u = cap["mean_use_of_cap"]
        (
            v(
                good,
                f"Used {u * 100:.0f}% of the available speed",
                "Mean of speed / cap while racing.",
            )
            if u >= 0.75
            else v(
                bad,
                f"Only used {u * 100:.0f}% of the available speed",
                "Mean of speed / cap while racing; the straights allow 5.2 m/s.",
                None,
                "warn",
            )
        )
    tr = sp.get("tracking") or {}
    loop_rms = tr.get("target_to_measured_rms")
    if loop_rms is not None:
        if loop_rms > 0.4:
            v(
                bad,
                f"Speed loop is loose: {loop_rms:.2f} m/s RMS from target to measured",
                f"Measured speed trails the Arduino's own target by ~{tr['target_to_measured_lag_s']} s.  Battery sag, gains, or the drivetrain.",
                None,
                "warn",
            )
        else:
            v(
                good,
                f"Speed loop tracks its target: {loop_rms:.2f} m/s RMS",
                f"~{tr['target_to_measured_lag_s']} s behind the target; the command itself is slew-limited to {tr.get('slew_limit')} m/s^2 before that, by design.",
            )

    stl = stats["steering"]
    if stl.get("saturated_pct") is not None:
        if stl["saturated_pct"] > 5:
            v(
                bad,
                f"Steering saturated {stl['saturated_pct']:.1f}% of the time",
                "At full lock the controller has no authority left; look at where it happens in the section table.",
                None,
                "warn",
            )
        if stl.get("reversals_per_s", 0) > 3:
            v(
                bad,
                f"Steering saws: {stl['reversals_per_s']:.1f} reversals/s",
                "Oscillation between the prior and the residual, or a noisy yaw-rate estimate.",
                None,
                "warn",
            )
    lag = stl.get("yaw_lag")
    if lag and lag.get("model_tau_s") is not None:
        # The plant's chassis lag sits AFTER the servo lag and the command
        # dead time, so compare the total delay-plus-lag, not tau alone.
        # The fit folds the servo's own lag into its delay (validated against
        # plant.py driven by a recorded command stream: totals recovered to
        # within 0.05 s), so the plant side sums all three.
        measured = lag["tau_s"] + lag["dead_time_s"]
        modelled = (
            lag["model_tau_s"]
            + (lag.get("model_dead_time_s") or 0.0)
            + (lag.get("model_servo_tau_s") or 0.0)
        )
        lag["measured_total_s"] = r(measured, 2)
        lag["model_total_s"] = r(modelled, 2)
        text = f"Command-to-yaw response: {measured:.2f} s measured vs {modelled:.2f} s in the plant"
        note = f"Fitted {lag['dead_time_s']:.2f} s delay + {lag['tau_s']:.2f} s first-order lag; plant is {lag.get('model_dead_time_s') or 0:.2f} s dead time + {lag.get('model_servo_tau_s') or 0:.2f} s servo + {lag['model_tau_s']:.2f} s chassis.  Fit error {lag['relative_error']:.2f} (vs {lag['relative_error_no_lag']:.2f} with no lag)."
        if lag.get("at_search_bound"):
            v(
                bad,
                text,
                note
                + "  The fit hit the edge of its search range -- treat it as unreliable.",
                None,
                "warn",
            )
        elif abs(measured - modelled) <= 0.15:
            v(
                good,
                text,
                "The response lag the policy was trained on matches this car.  " + note,
            )
        else:
            v(
                bad,
                text,
                "The steering prior predicts through this lag; a mismatch shifts every turn-in.  Refit plant.yaw_response_tau (fit_yaw_lag.py).  "
                + note,
                None,
                "warn",
            )
    for side in ("left", "right"):
        g = stl.get(f"gain_{side}")
        if g is not None:
            (
                v(
                    good,
                    f"Steering authority {side}: {g:.2f}x the plant",
                    "Yaw rate per command matches the model the policy was trained on.",
                )
                if 0.85 <= g <= 1.15
                else v(
                    bad,
                    f"Steering authority {side}: {g:.2f}x the plant",
                    "The car turns "
                    + ("less" if g < 1 else "more")
                    + " per command than plant.py assumes (effective_angle_table, understeer, tire_scrub).  The policy's prior steers by that model.",
                    None,
                    "warn",
                )
            )

    loc = stats["localization"]
    if loc.get("available"):
        if loc["jumps"]:
            v(
                bad,
                f"ZED pose jumped {loc['jumps']} time(s), largest {loc['largest_jump_m']} m",
                "The driver has no camera input -- it drives off this pose.  A jump moves every decision to the wrong place.",
                next((e["t"] for e in events if e["kind"] == "pose"), None),
            )
        else:
            v(
                good,
                "ZED pose had no jumps",
                f"{loc['rate_hz']} Hz, longest gap {loc['max_gap_s']} s.",
            )
        if loc.get("max_gap_s") and loc["max_gap_s"] > 0.3:
            v(
                bad,
                f"ZED pose went silent for {loc['max_gap_s']:.2f} s",
                "The driver commands neutral after pose_timeout (0.5 s).",
                None,
                "warn",
            )
    else:
        v(bad, "No ZED pose in the bag", "Nothing can be placed on the track.")

    stalls = [e for e in events if e["kind"] == "stall"]
    if stalls:
        v(bad, f"Stalled {len(stalls)} time(s)", stalls[0]["text"], stalls[0]["t"])
    links = [e for e in events if e["kind"] == "link"]
    if links:
        v(
            bad,
            f"Arduino link dropped {len(links)} time(s)",
            links[0]["text"],
            links[0]["t"],
        )
    errs = [e for e in events if e["kind"] == "log"]
    if errs:
        v(
            bad,
            f"{len(errs)} error line(s) in /rosout",
            errs[0]["text"],
            errs[0]["t"],
            "warn",
        )

    worst = sorted(
        [s_ for s_ in sections if s_["all"].get("min_clearance") is not None],
        key=lambda s_: s_["all"]["min_clearance"],
    )[:1]
    for s_ in worst:
        if s_["all"]["min_clearance"] < 0.2:
            v(
                bad,
                f"Tightest section: {s_['name']} ({s_.get('range') or f'{s_['s0']:.0f}-{s_['s1']:.0f} m'})",
                f"min clearance {s_['all']['min_clearance'] * 1000:.0f} mm, entry {s_['all']['entry_speed']} m/s, worst |CTE| {s_['all']['max_abs_cte']} m.",
                None,
                "warn",
            )
    return {"worked": good, "failed": bad}


def build_kpis(stats, laps, t0, t1):
    pol, sp, stl, loc = (
        stats["policy"],
        stats["speed"],
        stats["steering"],
        stats["localization"],
    )
    return [
        {
            "key": "result",
            "label": "Result",
            "value": (
                "FINISHED"
                if laps.get("finished")
                else f"{laps.get('completed', 0)}/{laps.get('target') or '?'} laps"
            ),
            "tone": "good" if laps.get("finished") else "bad",
        },
        {
            "key": "race_time",
            "label": "Race time",
            "value": laps.get("race_time"),
            "unit": "s",
        },
        {
            "key": "max_speed",
            "label": "Top speed",
            "value": sp.get("max"),
            "unit": "m/s",
        },
        {
            "key": "min_clear",
            "label": "Min clearance",
            "value": (pol.get("clearance") or {}).get("min"),
            "unit": "m",
            "tone": _tone_clear((pol.get("clearance") or {}).get("min")),
        },
        {
            "key": "cte",
            "label": "Mean |CTE|",
            "value": (pol.get("cte") or {}).get("mean_abs"),
            "unit": "m",
        },
        {
            "key": "steer_sat",
            "label": "Steer saturated",
            "value": stl.get("saturated_pct"),
            "unit": "%",
        },
        {
            "key": "jumps",
            "label": "Pose jumps",
            "value": loc.get("jumps"),
            "tone": "bad" if loc.get("jumps") else "good",
        },
        {"key": "duration", "label": "Run window", "value": r(t1 - t0, 1), "unit": "s"},
    ]


def _tone_clear(v):
    if v is None:
        return None
    return "bad" if v < 0 else "warn" if v < GRAZE else "good"


# ============================================================ topic health

EXPECTED_HZ = {
    ZED + "/pose": 15,
    ZED + "/odom": 15,
    ZED + "/imu/data": 100,
    "/arduino_bridge/status": 20,
    "/drive_cmd": 20,
    "/formula_one/telemetry": 20,
}


def topic_health(st, bag_info, tb):
    rows = []
    dur = max(bag_info["duration_s"] or 1.0, 1e-3)
    for topic, info in bag_info["topics"].items():
        row = {
            "topic": topic,
            "type": info["type"],
            "count": info["count"],
            "rate_hz": r(info["count"] / dur, 1),
        }
        exp = EXPECTED_HZ.get(topic)
        if exp:
            row["expected_hz"] = exp
            row["ok"] = row["rate_hz"] >= 0.6 * exp
        rows.append(row)
    for topic in EXPECTED_HZ:
        if topic not in bag_info["topics"]:
            rows.append(
                {
                    "topic": topic,
                    "type": "",
                    "count": 0,
                    "rate_hz": 0,
                    "expected_hz": EXPECTED_HZ[topic],
                    "ok": False,
                    "missing": True,
                }
            )
    return rows


# ============================================================== tegrastats


def tegrastats(run_dir):
    path = run_dir / "logs" / "tegrastats.log"
    if not path.exists():
        return {}
    cpu, gpu, ram, temps = [], [], [], []
    for line in path.read_text(errors="ignore").splitlines():
        m = re.search(r"CPU \[([^\]]+)\]", line)
        if m:
            loads = [int(x) for x in re.findall(r"(\d+)%@", m.group(1))]
            if loads:
                cpu.append(sum(loads) / len(loads))
        m = re.search(r"GR3D_FREQ (\d+)%", line)
        if m:
            gpu.append(int(m.group(1)))
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if m:
            ram.append(int(m.group(1)) / int(m.group(2)) * 100)
        found = [float(x) for x in re.findall(r"@(-?\d+(?:\.\d+)?)C", line)]
        if found:
            temps.append(max(found))
    return {
        "tegrastats": {
            "samples": len(cpu),
            "cpu_pct": {"mean": r(np.mean(cpu), 1), "max": r(np.max(cpu), 1)}
            if cpu
            else None,
            "gpu_pct": {"mean": r(np.mean(gpu), 1), "max": r(np.max(gpu), 1)}
            if gpu
            else None,
            "ram_pct_max": r(np.max(ram), 1) if ram else None,
            "temp_max_c": r(np.max(temps), 1) if temps else None,
            "series": {
                "cpu": [r(x, 1) for x in cpu],
                "gpu": gpu,
                "temp": [r(x, 1) for x in temps],
            },
        }
    }


# ================================================================== clouds

CLOUD_TOPICS = [ZED + "/point_cloud/cloud_registered"]
SIM_CAMERA = (0.315, 0.0, 0.20)  # sensors_world.py: the rendered ZED's mount


def write_clouds(
    bag,
    st,
    tb,
    transform,
    recording,
    progress,
    max_frames=400,
    frame_voxel=0.06,
    map_voxel=0.05,
    frame_cap=40000,
):
    """Every cloud frame (decimated to <= max_frames) into the track frame,
    logged per frame and accumulated into one voxel map."""
    topic = next((tp for tp in CLOUD_TOPICS if bag.has(tp)), None)
    pose = st.get(ZED + "/pose")
    info = {"clouds": 0, "map_points": 0, "fused_points": 0}
    if pose is None:
        return info
    tf = st["_tf"]
    base_frame = None
    for parent, child in st.get("_tf_dynamic", set()):
        if parent in ("odom", "map"):
            base_frame = child
    total = bag.topics().get(topic, {}).get("count", 0) if topic else 0
    every = 0
    if total > max_frames and topic:
        dur = (bag.end_ns - bag.start_ns) or 1
        every = int(dur / max_frames)
    map_xyz, map_rgb = [], []
    tf_note = None
    # The ground is not at z = 0 on the car: the ZED runs two_d_mode with the
    # CAMERA pinned at z = 0.  The first frames set the offset (their 5th
    # percentile of z is the ground in front of the car), then every frame is
    # logged with it, so the map sits on the course's plane.
    pending, ground = [], None
    frames = 0

    def emit(t_run, pts, rgb):
        nonlocal frames
        pts = pts.copy()
        pts[:, 2] -= ground
        recording.cloud_frame(t_run, pts, rgb)
        map_xyz.append(pts)
        map_rgb.append(rgb)
        frames += 1
        if len(map_xyz) >= 30:
            a, b = bagio.voxel_downsample(
                np.concatenate(map_xyz), np.concatenate(map_rgb), map_voxel
            )
            map_xyz[:] = [a]
            map_rgb[:] = [b]

    if topic:
        for n, (_, t_ns, msg) in enumerate(bag.messages(topic, every_ns=every)):
            xyz, rgb = bagio.cloud_to_numpy(msg)
            if len(xyz) == 0:
                continue
            # cloud frame -> the frame the pose describes
            m_base = tf.lookup(base_frame, msg.header.frame_id) if base_frame else None
            if m_base is None:
                m_base = np.eye(4)
                m_base[:3, 3] = SIM_CAMERA
                tf_note = f"no TF from {msg.header.frame_id!r} to the pose frame; assumed the simulated ZED mount"
            tq = t_ns * 1e-9
            i = int(np.clip(np.searchsorted(pose[:, 0], tq), 1, len(pose) - 1))
            if abs(pose[i, 0] - tq) > 0.2 and abs(pose[i - 1, 0] - tq) > 0.2:
                continue
            row = (
                pose[i]
                if abs(pose[i, 0] - tq) < abs(pose[i - 1, 0] - tq)
                else pose[i - 1]
            )
            m_pose = np.eye(4)
            q = type("Q", (), {"x": row[5], "y": row[6], "z": row[7], "w": row[8]})
            m_pose[:3, :3] = bagio.quat_matrix(q)
            m_pose[:3, 3] = row[2:5]
            m = m_pose @ m_base
            pts = xyz @ m[:3, :3].T + m[:3, 3]
            if transform:
                x2, y2 = apply_xy(
                    transform, pts[:, 0], pts[:, 1], np.full(len(pts), tq)
                )
                pts = np.column_stack([x2, y2, pts[:, 2]]).astype(np.float32)
            f_xyz, f_rgb = bagio.voxel_downsample(pts, rgb, frame_voxel)
            if len(f_xyz) > frame_cap:
                pick = np.random.default_rng(0).choice(
                    len(f_xyz), frame_cap, replace=False
                )
                f_xyz, f_rgb = f_xyz[pick], f_rgb[pick]
            t_run = float(tb(tq))
            if ground is None:
                pending.append((t_run, f_xyz, f_rgb))
                if len(pending) >= 15:
                    ground = float(
                        np.median([np.percentile(p[1][:, 2], 5) for p in pending])
                    )
                    for item in pending:
                        emit(*item)
                    pending = []
            else:
                emit(t_run, f_xyz, f_rgb)
            if n % 20 == 0:
                progress(
                    0.55 + 0.28 * min(1, n / max(1, min(total, max_frames))),
                    f"point clouds ({n} frames)",
                )
    if pending:
        ground = float(np.median([np.percentile(p[1][:, 2], 5) for p in pending]))
        for item in pending:
            emit(*item)
    info["clouds"] = frames
    if map_xyz:
        a, b = bagio.voxel_downsample(
            np.concatenate(map_xyz), np.concatenate(map_rgb), map_voxel
        )
        recording.cloud_static("map", a, b)
        info["map_points"] = int(len(a))
        info["ground_z"] = r(ground, 3)
    info["cloud_topic"] = topic
    info["cloud_tf_note"] = tf_note

    # The ZED's own spatial map (recorded with --map): the last fused cloud.
    fused_topic = ZED + "/mapping/fused_cloud"
    if bag.has(fused_topic):
        last = None
        for _, _, msg in bag.messages(fused_topic):
            last = msg
        if last is not None:
            xyz, rgb = bagio.cloud_to_numpy(last, max_range=1e4)
            if transform and len(xyz):
                x2, y2 = apply_xy(transform, xyz[:, 0], xyz[:, 1])
                xyz = np.column_stack([x2, y2, xyz[:, 2] - (ground or 0.0)]).astype(
                    np.float32
                )
            recording.cloud_static("zed_map", xyz, rgb)
            info["fused_points"] = int(len(xyz))
    return info


# ================================================================== images

IMAGE_TOPICS = [
    ZED + "/rgb/color/rect/image/compressed",
    ZED + "/left/image_rect_color",
    ZED + "/rgb/color/rect/image",
]


def write_images(bag, tb, recording, hz=5.0, max_width=640):
    topic = next((tp for tp in IMAGE_TOPICS if bag.has(tp)), None)
    if topic is None:
        return {"images": 0}
    try:
        from PIL import Image as PILImage
    except ImportError:
        PILImage = None
    count = 0
    for _, t_ns, msg in bag.messages(topic, every_ns=int(1e9 / hz)):
        if topic.endswith("compressed"):
            data = bytes(msg.data)
            if "png" in msg.format.lower() and PILImage:
                img = PILImage.open(io.BytesIO(data)).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=80)
                data = buf.getvalue()
        else:
            if PILImage is None:
                break
            data = raw_to_jpeg(msg, PILImage, max_width)
            if data is None:
                continue
        recording.image(float(tb(t_ns * 1e-9)), data)
        count += 1
    return {"images": count, "image_topic": topic}


def raw_to_jpeg(msg, PILImage, max_width):
    enc = msg.encoding.lower()
    buf = np.frombuffer(
        bytes(msg.data) if not isinstance(msg.data, np.ndarray) else msg.data, np.uint8
    )
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(enc)
    if channels is None:
        return None
    try:
        arr = (
            buf[: msg.height * msg.step]
            .reshape(msg.height, msg.step)[:, : msg.width * channels]
            .reshape(msg.height, msg.width, channels)
        )
    except ValueError:
        return None
    if enc.startswith("bgr"):
        arr = arr[..., [2, 1, 0]]
    elif channels == 4:
        arr = arr[..., :3]
    img = (
        PILImage.fromarray(arr.squeeze())
        if channels > 1
        else PILImage.fromarray(arr[..., 0], "L")
    )
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    out = io.BytesIO()
    img.convert("RGB").save(out, "JPEG", quality=78)
    return out.getvalue()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--no-clouds", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    s = process(
        Path(a.run_dir),
        lambda f, txt: print(f"{f * 100:5.1f}%  {txt}"),
        clouds=not a.no_clouds,
        images=not a.no_images,
    )
    print(json.dumps({"kpis": s["kpis"], "verdicts": s["verdicts"]}, indent=1))
