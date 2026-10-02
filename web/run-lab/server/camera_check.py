"""Does this run's own data agree with the camera mount it drove on?

Run by analyze.py on every run, and by calibration.py after a
straight_line_trim or figure_eight_calib (`camera_check.py <run> --report`).
Everything comes from what record_run.py and the maneuver runner keep:

    floor     depth frames, parked ones first (waiting for GO): a ground-plane
              fit gives the lens height, pitch and roll, which must match the
              mount.  The IMU's gravity at the same moments says whether the
              floor itself was level, i.e. whether a disagreement is the
              mount's fault.
    band      the same fit as ground truth: how much flat floor lands in the
              drivers' obstacle height band when they level the depth by the
              IMU and believe the mount's height, as formula_two_node does.
              Also the real depth FOV against the one the driver config
              assumes.
    attitude  the pose's and odom's roll and pitch against the IMU's.  IMU and
              camera are one rigid body, so a steady difference is the ZED's
              tracking frame tilted against gravity, not the mount.
    motion    the pose track while driving: how far ahead of the rear axle
              the pose point moves sideways, and how far its heading sits off
              the direction of travel (camera_extrinsics.fit_lever_arm).  Needs
              turns both ways, which a figure-8 gives.

The mount is the one in the run's metadata.yaml (what the driver used),
else the repo's vehicle.yaml.  See jetson/scripts/launchCalibration.sh for
measuring it properly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "jetson" / "scripts"))
import camera_extrinsics as ce  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depthview  # noqa: E402

ZED = "/zed/zed_node"
# Past these the error matters to the drivers (vehicle.yaml camera_mount).
WARN = {"height_m": 0.015, "angle_deg": 0.5, "lever_m": 0.05, "yaw_deg": 1.0}
# The drivers keep the nearest in-band point per column, so a few floor points
# are already a phantom wall; 2% of the band is where it starts to show.
FLOOR_SHARE = 0.02
# Beyond IMU noise and the camera's own tracking jitter.
TILT_DEG = 2.0
FOV_DEG = 2.0
PARKED_SPEED = 0.03  # m/s
MAX_FRAMES = 15
BAND_FRAMES = 40
FLOOR_TOL = 0.03  # m: a point this close to the fitted plane is floor


def mount_of(metadata):
    """(Mount, where it came from)."""
    rec = (metadata or {}).get("camera_mount") or {}
    if "mount" in rec:
        m = rec["mount"]
        return (
            ce.Mount(
                *(float(m[k]) for k in ce.MOUNT_KEYS),
                lens=tuple(rec["lens_offset"]),
                provenance=rec.get("provenance", "?"),
            ),
            "run metadata",
        )
    return ce.load_mount(), "repo vehicle.yaml (not recorded with the run)"


def wheelbase():
    data = yaml.safe_load(ce.VEHICLE_YAML.read_text())
    return float(data["geometry"]["wheelbase"]["value"])


def pose_track(streams):
    pose = streams.get(ZED + "/pose")
    if pose is None or len(pose) < 50:
        return None
    t, x, y = pose[:, 0], pose[:, 2], pose[:, 3]
    qx, qy, qz, qw = pose[:, 5], pose[:, 6], pose[:, 7], pose[:, 8]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return t, x, y, yaw


def parked_at(track, t, half=0.5):
    if track is None:
        return False
    tt, x, y, _ = track
    i0, i1 = np.searchsorted(tt, [t - half, t + half])
    if i1 - i0 < 3:
        return False
    span = tt[i1 - 1] - tt[i0]
    return (
        span > 0
        and math.hypot(x[i1 - 1] - x[i0], y[i1 - 1] - y[i0]) / span < PARKED_SPEED
    )


def _floor_fits(bag, streams, mount, track, topic, parked_only):
    d = mount.depth_origin()
    imu = streams.get(ZED + "/imu/data")
    fits, attitudes, k, moving = [], [], None, False
    cfg, _ = depthview.camera_config(Path("."))
    cam = depthview._perception().Camera(cfg)
    cam.hfov_deg = float(cfg["camera"]["hfov_deg"])
    for _, t_ns, msg in bag.messages(topic):
        t = t_ns * 1e-9
        parked = parked_at(track, t)
        if parked_only and not parked:
            continue
        depth = depthview.decode(topic, msg)
        if depth is None:
            continue
        if k is None:
            k, _ = depthview.intrinsics(bag, cam, depth.shape[1], depth.shape[0])
        # Gated around the mount's height: wide enough for the ~5 cm the guessed
        # mount was off on the car.
        fit = ce.fit_floor(
            ce.back_project(depth, *k), d[2], mount.roll, mount.pitch, gate=0.12
        )
        if fit is None:
            continue
        fits.append((t, fit))
        moving |= not parked
        if imu is not None and len(imu):
            j = int(np.clip(np.searchsorted(imu[:, 0], t), 0, len(imu) - 1))
            if abs(imu[j, 0] - t) < 0.2:
                # Accelerometer gravity when parked; moving, it carries the
                # car's own acceleration, so the fused orientation instead.
                attitudes.append(
                    ce.attitude_from_up(imu[j, 1:4]) if parked else tuple(imu[j, 7:9])
                )
        if len(fits) >= MAX_FRAMES:
            break
    return fits, attitudes, moving


def check_floor(bag, streams, mount, track):
    topic = next((tp for tp in depthview.DEPTH_TOPICS if bag.has(tp)), None)
    if topic is None or track is None:
        return None
    # Parked frames first: no pitch from braking or the suspension.  A
    # calibration run may park only briefly, so fall back to moving ones.
    fits, attitudes, moving = _floor_fits(bag, streams, mount, track, topic, True)
    if len(fits) < 3:
        fits, attitudes, moving = _floor_fits(bag, streams, mount, track, topic, False)
    if not fits:
        return None
    d = mount.depth_origin()
    h = np.median([f.height for _, f in fits])
    roll = np.median([f.roll for _, f in fits])
    pitch = np.median([f.pitch for _, f in fits])
    out = {
        "frames": len(fits),
        "moving": moving,
        "t": fits[0][0],
        "height_m": round(float(h), 4),
        "pitch_deg": round(math.degrees(pitch), 3),
        "roll_deg": round(math.degrees(roll), 3),
        "height_err_m": round(float(h - d[2]), 4),
        "pitch_err_deg": round(math.degrees(pitch - mount.pitch), 3),
        "roll_err_deg": round(math.degrees(roll - mount.roll), 3),
        "rms_mm": round(float(np.median([f.rms for _, f in fits]) * 1000), 2),
    }
    if attitudes:
        r_i, p_i = np.median(np.array(attitudes), axis=0)
        out["imu_pitch_deg"] = round(math.degrees(p_i), 3)
        out["imu_roll_deg"] = round(math.degrees(r_i), 3)
        # IMU against the floor fit: this is the floor's slope, not the mount.
        out["floor_slope_deg"] = round(
            max(abs(math.degrees(p_i - pitch)), abs(math.degrees(r_i - roll))), 3
        )
    return out


def check_motion(track, mount, sim):
    if track is None:
        return None
    fit = ce.fit_lever_arm(*track)
    if fit is None:
        return None
    half = wheelbase() / 2
    # Gazebo's pose is the car center; the ZED's is camera_link.
    expect_x, expect_yaw = (0.0, 0.0) if sim else (mount.x, mount.yaw)
    fit["pose_point_x_m"] = round(fit["ahead_of_rear_axle_m"] - half, 4)
    fit["expected_x_m"] = round(expect_x, 4)
    fit["x_err_m"] = round(fit["pose_point_x_m"] - expect_x, 4)
    fit["yaw_err_deg"] = round(fit["yaw_deg"] - math.degrees(expect_yaw), 3)
    fit["both_directions"] = (
        min(fit["left_turn_samples"], fit["right_turn_samples"]) >= 100
    )
    return fit


def _roll_pitch(rows):
    """(roll, pitch) arrays from pose/odom stream rows (quaternion in 5:9)."""
    qx, qy, qz, qw = rows[:, 5], rows[:, 6], rows[:, 7], rows[:, 8]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    return roll, pitch


def _wrap_deg(a):
    return np.degrees(np.arctan2(np.sin(a), np.cos(a)))


def check_attitude(streams):
    imu = streams.get(ZED + "/imu/data")
    if imu is None or len(imu) < 50:
        return None
    out = {}
    for name in ("pose", "odom"):
        rows = streams.get(f"{ZED}/{name}")
        if rows is None or len(rows) < 50:
            continue
        roll, pitch = _roll_pitch(rows)
        j = np.clip(np.searchsorted(imu[:, 0], rows[:, 0]), 0, len(imu) - 1)
        near = np.abs(imu[j, 0] - rows[:, 0]) < 0.05
        if near.sum() < 50:
            continue
        dr = _wrap_deg(roll[near] - imu[j[near], 7])
        dp = _wrap_deg(pitch[near] - imu[j[near], 8])
        out[name] = {
            "samples": int(near.sum()),
            "roll_err_deg": round(float(np.median(dr)), 2),
            "pitch_err_deg": round(float(np.median(dp)), 2),
            "tilt_p95_deg": round(float(np.percentile(np.hypot(dr, dp), 95)), 2),
            "z_span_m": round(float(np.ptp(rows[:, 4])), 3),
        }
    return out or None


def check_band(bag, streams, mount, run_dir, floor):
    """Floor share of the drivers' band, believing the mount and at the fit."""
    topic = next((tp for tp in depthview.DEPTH_TOPICS if bag.has(tp)), None)
    imu = streams.get(ZED + "/imu/data")
    if topic is None or imu is None or len(imu) < 10:
        return None
    cfg, cfg_source = depthview.camera_config(Path(run_dir or "."))
    c = cfg["camera"]
    lo, hi = (float(v) for v in c["band"])
    cam = depthview._perception().Camera(cfg)
    cam.hfov_deg = float(c["hfov_deg"])
    believed = float(mount.depth_origin()[2])
    prior = floor["height_m"] if floor else believed
    count = bag.topics()[topic]["count"]
    every = max(1, count // BAND_FRAMES)
    k, size, rows = None, None, []
    for i, (_, t_ns, msg) in enumerate(bag.messages(topic)):
        if i % every:
            continue
        depth = depthview.decode(topic, msg)
        if depth is None:
            continue
        if k is None:
            size = depth.shape[1]
            k, _ = depthview.intrinsics(bag, cam, depth.shape[1], depth.shape[0])
        t = t_ns * 1e-9
        j = int(np.clip(np.searchsorted(imu[:, 0], t), 0, len(imu) - 1))
        if abs(imu[j, 0] - t) > 0.2:
            continue
        pts = ce.back_project(depth, *k, max_range=min(cam.scan_max, 10.0))
        fit = ce.fit_floor(pts, prior, gate=0.12)
        if fit is None:
            continue
        on_floor = (
            np.abs(pts @ ce.up_from_attitude(fit.roll, fit.pitch) + fit.height)
            < FLOOR_TOL
        )
        # What the driver computes: levelled by the IMU, lifted by its height.
        lifted = pts @ ce.up_from_attitude(imu[j, 7], imu[j, 8])
        rng = np.hypot(pts[:, 0], pts[:, 1])
        row = []
        for height in (believed, fit.height):
            h = lifted + height
            band = (h >= lo) & (h <= hi)
            leak = band & on_floor
            row += [
                leak.sum() / max(band.sum(), 1),
                rng[leak].min() if leak.any() else np.nan,
            ]
        rows.append(row)
    if not rows:
        return None
    a = np.array(rows)
    fx = k[0]
    hfov = math.degrees(2 * math.atan(size / 2 / fx))
    blind = int((np.abs(np.degrees(cam.azimuth)) > hfov / 2).sum())
    return {
        "frames": len(a),
        "config": cfg_source,
        "band_m": [lo, hi],
        "believed_height_m": round(believed, 4),
        "floor_share": round(float(np.median(a[:, 0])), 4),
        "floor_share_p90": round(float(np.percentile(a[:, 0], 90)), 4),
        "frames_over": round(float(np.mean(a[:, 0] > FLOOR_SHARE)), 3),
        "leak_range_m": None
        if np.all(np.isnan(a[:, 1]))
        else round(float(np.nanmedian(a[:, 1])), 2),
        "fitted_floor_share": round(float(np.median(a[:, 2])), 4),
        "hfov_deg": round(hfov, 1),
        "config_hfov_deg": float(c["hfov_deg"]),
        "blind_columns": blind,
        "columns": int(cam.W),
    }


def run(bag, streams, metadata, sim, tb, run_dir=None):
    """summary["perception"]["camera_check"], and verdict rows for it."""
    mount, source = mount_of(metadata)
    if mount is None:
        return {"camera_check": {"error": "no camera_mount anywhere"}}, []
    track = pose_track(streams)
    floor = check_floor(bag, streams, mount, track)
    band = check_band(bag, streams, mount, run_dir, floor)
    attitude = None if sim else check_attitude(streams)
    motion = check_motion(track, mount, sim)
    verdicts = []

    def v(ok, title, detail, t=None):
        verdicts.append(
            {
                "ok": ok,
                "title": title,
                "detail": detail,
                "t": round(float(tb(t)), 2) if t is not None else None,
            }
        )

    tag = f"mount from {source}, {mount.provenance}"
    if floor:
        high = abs(floor["height_err_m"]) > WARN["height_m"]
        tilted = (
            max(abs(floor["pitch_err_deg"]), abs(floor["roll_err_deg"]))
            > WARN["angle_deg"]
        )
        bad = high or tilted
        detail = (
            f"{'Floor fit while moving' if floor['moving'] else 'Parked floor fit'} "
            f"({floor['frames']} frames): lens {floor['height_m']:.3f} m up, "
            f"pitch {floor['pitch_deg']:+.2f}, roll {floor['roll_deg']:+.2f} deg; against the mount "
            f"{floor['height_err_m'] * 1000:+.0f} mm, {floor['pitch_err_deg']:+.2f} / "
            f"{floor['roll_err_deg']:+.2f} deg ({tag})."
        )
        if high:
            # Height is about the drivers' 0.07 m band floor, not the slope.
            detail += (
                f" The drivers put the ground {-floor['height_err_m'] * 1000:+.0f} mm off, "
                "which moves it toward or out of the obstacle height band."
            )
        if tilted and floor.get("floor_slope_deg", 0) > WARN["angle_deg"]:
            detail += (
                f" The IMU says the floor itself slopes {floor['floor_slope_deg']:.1f} deg here,"
                " so the pitch/roll part may be the floor, not the mount."
            )
        if bad:
            detail += " Re-run launchCalibration.sh."
        v(
            not bad,
            "Camera mount matches the floor"
            if not bad
            else "Camera mount disagrees with the floor",
            detail,
            floor["t"],
        )
    if motion and motion["both_directions"]:
        bad = abs(motion["x_err_m"]) > max(
            WARN["lever_m"], 3 * motion["sigma_m"]
        ) or abs(motion["yaw_err_deg"]) > max(
            WARN["yaw_deg"], 3 * motion["sigma_yaw_deg"]
        )
        v(
            not bad,
            "Pose lever arm matches the mount"
            if not bad
            else "Pose lever arm disagrees with the mount",
            f"Driving says the pose point is {motion['ahead_of_rear_axle_m']:.3f} m ahead of the rear "
            f"axle (x {motion['pose_point_x_m']:+.3f} m from the wheelbase midpoint, expected "
            f"{motion['expected_x_m']:+.3f}) and its heading is {motion['yaw_deg']:+.2f} deg off the "
            f"direction of travel; +/-{motion['sigma_m'] * 1000:.0f} mm, {motion['samples']} samples ({tag}).",
        )
    if band:
        bad = band["floor_share"] > FLOOR_SHARE or band["frames_over"] > 0.10
        lo, hi = band["band_m"]
        detail = (
            f"Believing the lens is {band['believed_height_m']:.3f} m up and levelling by the IMU, "
            f"{band['floor_share'] * 100:.1f}% of the points in the {lo:.2f}-{hi:.2f} m band are "
            f"floor (worst tenth {band['floor_share_p90'] * 100:.1f}%; "
            f"{band['frames_over'] * 100:.0f}% of {band['frames']} frames over "
            f"{FLOOR_SHARE * 100:.0f}%)"
        )
        if band["leak_range_m"] is not None:
            detail += f", the nearest at ~{band['leak_range_m']:.1f} m"
        detail += (
            f". At the fitted height it is {band['fitted_floor_share'] * 100:.1f}%. "
            f"Band from the {band['config']} config."
        )
        if bad:
            detail += (
                " The drivers keep the nearest in-band point per column, so these are"
                " phantom walls."
            )
        v(
            not bad,
            "Floor stays out of the obstacle band"
            if not bad
            else "Floor leaks into the obstacle band",
            detail,
        )
        if abs(band["hfov_deg"] - band["config_hfov_deg"]) > FOV_DEG:
            v(
                False,
                "Depth FOV narrower than the driver config"
                if band["hfov_deg"] < band["config_hfov_deg"]
                else "Depth FOV wider than the driver config",
                f"The camera's depth spans {band['hfov_deg']:.1f} deg; the config assumes "
                f"{band['config_hfov_deg']:.0f}, so {band['blind_columns']} of "
                f"{band['columns']} scan columns never see anything.",
            )
    pose = (attitude or {}).get("pose")
    if pose:
        odom = attitude.get("odom")
        bad = max(abs(pose["roll_err_deg"]), abs(pose["pitch_err_deg"])) > TILT_DEG
        detail = (
            f"/pose against the IMU: roll {pose['roll_err_deg']:+.1f}, pitch "
            f"{pose['pitch_err_deg']:+.1f} deg (95th pct {pose['tilt_p95_deg']:.1f}), "
            f"z spans {pose['z_span_m']:.2f} m"
        )
        if odom:
            detail += (
                f"; /odom roll {odom['roll_err_deg']:+.1f}, pitch {odom['pitch_err_deg']:+.1f} "
                f"deg, z spans {odom['z_span_m']:.2f} m"
            )
        detail += "."
        if bad:
            detail += (
                " The IMU is in the camera, so this is the ZED's tracking frame tilted"
                " against gravity, not the mount. The drivers level depth by the IMU;"
                " the tilt reaches the pose's x/y and anything drawn from TF."
            )
        v(
            not bad,
            "ZED tracking frame is level"
            if not bad
            else "ZED tracking frame is tilted against gravity",
            detail,
        )
    return {
        "camera_check": {
            "mount": mount.as_dict(),
            "mount_source": source,
            "floor": floor,
            "band": band,
            "attitude": attitude,
            "motion": motion,
        }
    }, verdicts


# ------------------------------------------------------------------- CLI

REPORT_HEADING = "## Camera and pose"


def report(run_dir: Path):
    """camera_check on one run outside analyze.process, for calibration runs.

    Returns (result dict, markdown section).
    """
    import bagio
    from analyze import Timebase, read_streams

    metadata = {}
    if (run_dir / "metadata.yaml").exists():
        metadata = yaml.safe_load((run_dir / "metadata.yaml").read_text()) or {}
    bag_dir = bagio.find_bag(run_dir)
    if bag_dir is None:
        raise FileNotFoundError(f"no rosbag found in {run_dir}")
    with bagio.Bag(bag_dir) as bag:
        streams = read_streams(bag, lambda *_: None)
        tb = Timebase(streams, bag.start_ns * 1e-9)
        checked, rows = run(bag, streams, metadata, tb.sim, tb, run_dir)
    lines = [REPORT_HEADING, ""]
    for row in rows:
        lines.append(
            f"- **{'OK' if row['ok'] else 'CHECK'}: {row['title']}.** {row['detail']}"
        )
    found = checked["camera_check"]
    missing = [
        what
        for what, key in (
            ("depth (floor fit, band)", "floor"),
            ("IMU + pose (tracking-frame tilt)", "attitude"),
            ("turns both ways (lever arm)", "motion"),
        )
        if not found.get(key)
    ]
    if missing:
        lines.append(f"- Not checked, the run has no {', no '.join(missing)}.")
    return {**found, "verdicts": rows}, "\n".join(lines) + "\n"


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, help="run directory")
    ap.add_argument(
        "--report",
        action="store_true",
        help="also write <run>/camera_check.json and replace this section of <run>/report.md",
    )
    args = ap.parse_args()
    result, section = report(args.run)
    print(section, end="")
    if not args.report:
        return
    (args.run / "camera_check.json").write_text(
        json.dumps(result, indent=1, default=float)
    )
    path = args.run / "report.md"
    text = path.read_text() if path.exists() else ""
    if REPORT_HEADING in text:
        # Re-running replaces the section rather than stacking copies.
        head, _, rest = text.partition(REPORT_HEADING)
        nxt = rest.find("\n## ")
        text = head + (rest[nxt + 1 :] if nxt >= 0 else "")
    path.write_text(text.rstrip("\n") + ("\n\n" if text.strip() else "") + section)


if __name__ == "__main__":
    main()
