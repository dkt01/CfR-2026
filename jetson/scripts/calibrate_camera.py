#!/usr/bin/env python3
"""Measure where the ZED sits on the car, from a parked capture.  No actuators.

    ./calibrate_camera.py                                  # floor + IMU
    ./calibrate_camera.py --target 2.0,0 --target 3.0,0.6  # + x, y, yaw
    ./calibrate_camera.py --tape 0.47,0,0.19 --from rear-axle
    ./calibrate_camera.py --offline ~/cfr_runs/<run>       # re-analyze

Normally run through launchCalibration.sh.  Park on flat, level floor with
0.5-4 m of open floor ahead.  What each input measures:

    floor plane   lens height, pitch and roll against the floor.  Always.
    IMU gravity   pitch and roll against gravity.  A cross-check: it should
                  match the floor unless the floor slopes.
    --target X,Y  a box set square to the car, its front face centered at
                  (X, Y) in the vehicle frame.  One target gives the lens x
                  and y, two or more give yaw as well.
    --tape X,Y,Z  camera_link (the mounting hole) measured by hand.  Used for
                  x and y when there are no targets, and compared with the
                  floor fit's z.

Writes ~/cfr_runs/<UTC>_camera_calib[_label]/ with capture.npz, result.json,
report.md and a vehicle_patch.yaml for apply_vehicle_patch.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_extrinsics as ce  # noqa: E402

ZED = "/zed/zed_node"
HALF_WHEELBASE = 0.162  # vehicle.yaml geometry.wheelbase / 2
# Beyond these the drivers' height band sees the floor or misses bale faces
# (see vehicle.yaml camera_mount.pitch); below them a change is noise.
TOL = {"height_m": 0.01, "angle_deg": 0.3, "xy_m": 0.02}
BAND_FLOOR = 0.07  # m, formulaTwo/Three camera.band[0]: lowest point called obstacle


def xy_pair(text, n):
    vals = [float(v) for v in text.split(",")]
    if len(vals) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers")
    return vals


# ------------------------------------------------------------------ capture


def capture(args):
    """Grab depth frames, IMU and pose for --frames frames, parked."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from geometry_msgs.msg import PoseStamped

    rclpy.init()
    node = Node("camera_calibration")
    got = {
        "depth": [],
        "imu": [],
        "gyro": [],
        "quat": [],
        "pose": [],
        "k": None,
        "size": None,
    }

    def on_depth(msg):
        if msg.encoding != "32FC1" or len(got["depth"]) >= args.frames:
            return
        rows = np.frombuffer(bytes(msg.data), dtype="<f4").reshape(
            msg.height, msg.step // 4
        )
        got["depth"].append(rows[:, : msg.width].copy())

    def on_info(msg):
        if msg.k[0] > 0:
            got["k"] = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            got["size"] = (msg.width, msg.height)

    def on_imu(msg):
        a, w, q = msg.linear_acceleration, msg.angular_velocity, msg.orientation
        got["imu"].append((a.x, a.y, a.z))
        got["gyro"].append((w.x, w.y, w.z))
        got["quat"].append((q.x, q.y, q.z, q.w))

    def on_pose(msg):
        p = msg.pose.position
        got["pose"].append((p.x, p.y, p.z))

    qos = qos_profile_sensor_data
    node.create_subscription(Image, ZED + "/depth/depth_registered", on_depth, qos)
    node.create_subscription(CameraInfo, ZED + "/depth/camera_info", on_info, qos)
    node.create_subscription(Imu, ZED + "/imu/data", on_imu, qos)
    node.create_subscription(PoseStamped, ZED + "/pose", on_pose, qos)

    lens, lens_source = None, None
    try:
        import tf2_ros

        buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(buffer, node)
    except ImportError:
        buffer = None

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and (
        len(got["depth"]) < args.frames or got["k"] is None
    ):
        rclpy.spin_once(node, timeout_sec=0.1)
        if buffer is not None and lens is None:
            try:
                tf = buffer.lookup_transform(
                    args.camera_name + "_camera_link",
                    args.camera_name + "_left_camera_frame",
                    rclpy.time.Time(),
                )
                tr = tf.transform.translation
                lens, lens_source = (tr.x, tr.y, tr.z), "tf"
            except Exception:  # noqa: BLE001 -- not published yet; keep spinning
                pass
    node.destroy_node()
    rclpy.shutdown()
    if not got["depth"] or got["k"] is None:
        raise SystemExit(
            f"error: {len(got['depth'])} depth frame(s), camera_info "
            f"{'yes' if got['k'] else 'no'} in {args.timeout:.0f} s -- is the ZED up "
            "(launch.sh --no-bridge)?"
        )
    depth = np.stack(got["depth"]).astype(np.float32)
    k = got["k"]
    # camera_info can describe the full-resolution sensor; scale to the image.
    if got["size"] and tuple(got["size"]) != (depth.shape[2], depth.shape[1]):
        sx, sy = depth.shape[2] / got["size"][0], depth.shape[1] / got["size"][1]
        k = (k[0] * sx, k[1] * sy, (k[2] + 0.5) * sx - 0.5, (k[3] + 0.5) * sy - 0.5)
    return {
        "depth": depth,
        "k": np.array(k),
        "imu": np.array(got["imu"]).reshape(-1, 3),
        "gyro": np.array(got["gyro"]).reshape(-1, 3),
        "quat": np.array(got["quat"]).reshape(-1, 4),
        "pose": np.array(got["pose"]).reshape(-1, 3),
        "lens": np.array(lens if lens else [np.nan] * 3),
        "lens_source": np.array(lens_source or "vehicle.yaml"),
    }


# ----------------------------------------------------------------- analysis


def analyze(cap, prior: ce.Mount, targets, tape):
    """Everything the capture says about the mount, against `prior`."""
    notes = []
    lens = cap["lens"]
    if np.all(np.isfinite(lens)):
        lens = tuple(float(v) for v in lens)
    else:
        lens = tuple(prior.lens)
        notes.append("Lens offset not read from TF; using vehicle.yaml's.")
    prior_d = prior.depth_origin()
    fx, fy, cx, cy = cap["k"]

    fits, points = [], []
    for frame in cap["depth"]:
        pts = ce.back_project(frame.astype(float), fx, fy, cx, cy, stride=4)
        fit = ce.fit_floor(pts, prior_d[2], prior.roll, prior.pitch)
        if fit is not None:
            fits.append(fit)
            points.append(pts)
    if not fits:
        raise SystemExit(
            "error: no floor plane found -- is there open, flat floor 0.5-4 m ahead, "
            "and is the prior within ~0.1 m of the true height?"
        )
    h = np.array([f.height for f in fits])
    rl = np.array([f.roll for f in fits])
    pt = np.array([f.pitch for f in fits])
    floor = {
        "frames": len(fits),
        "height_m": float(np.median(h)),
        "roll": float(np.median(rl)),
        "pitch": float(np.median(pt)),
        "spread_height_mm": float(np.std(h) * 1000),
        "spread_pitch_deg": float(np.degrees(np.std(pt))),
        "rms_mm": float(np.median([f.rms for f in fits]) * 1000),
        "inliers": int(np.median([f.inliers for f in fits])),
    }

    imu = {}
    if len(cap["imu"]):
        acc = cap["imu"].mean(axis=0)
        r_a, p_a = ce.attitude_from_up(acc)
        imu = {
            "accel_roll": r_a,
            "accel_pitch": p_a,
            "g": float(np.linalg.norm(acc)),
            "gyro_max": float(np.abs(cap["gyro"]).max()) if len(cap["gyro"]) else None,
        }
        if len(cap["quat"]):
            x, y, z, w = cap["quat"].mean(axis=0)
            imu["quat_roll"] = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            imu["quat_pitch"] = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        if imu["gyro_max"] is not None and imu["gyro_max"] > 0.05:
            notes.append(
                f"Gyro peaked at {imu['gyro_max']:.2f} rad/s -- the car moved or was touched."
            )
    if len(cap["pose"]) > 2:
        drift = float(np.linalg.norm(cap["pose"].max(0) - cap["pose"].min(0)))
        if drift > 0.02:
            notes.append(f"Pose moved {drift * 100:.1f} cm during the capture.")

    # Lens x, y (and yaw) in the vehicle frame: targets, else tape, else prior.
    yaw, xy_source, target_rows = prior.yaw, "prior", []
    dx, dy = float(prior_d[0]), float(prior_d[1])
    if targets:
        level = np.concatenate(
            [
                ce.level(p, floor["roll"], floor["pitch"], floor["height_m"])
                for p in points[:8]
            ]
        )
        measured, truth = [], []
        for tx, ty in targets:
            expect = (tx - prior_d[0], ty - prior_d[1])
            found = ce.locate_target(level, expect)
            target_rows.append({"truth": [tx, ty], "found": found})
            if found is None:
                notes.append(
                    f"Target at ({tx}, {ty}) not found within 0.45 m of where expected."
                )
                continue
            measured.append(found[:2])
            truth.append((tx, ty))
        if measured:
            dx, dy, yaw, rms = ce.solve_planar(measured, truth, prior.yaw)
            xy_source = f"{len(measured)} target(s), rms {rms * 1000:.0f} mm"
            if len(measured) < 2:
                notes.append("One target: yaw held at vehicle.yaml's value.")
    elif tape is not None:
        r_tape = ce.rotation(floor["roll"], floor["pitch"], prior.yaw)
        dx, dy = (np.asarray(tape[:2]) + (r_tape @ np.asarray(lens))[:2]).tolist()
        xy_source = "tape"

    rot = ce.rotation(floor["roll"], floor["pitch"], yaw)
    depth_origin = np.array([dx, dy, floor["height_m"]])
    mount_xyz = depth_origin - rot @ np.asarray(lens)
    new = ce.Mount(*mount_xyz, floor["roll"], floor["pitch"], yaw, lens=lens)

    if tape is not None and abs(tape[2] - mount_xyz[2]) > 0.01:
        notes.append(
            f"Taped mount height {tape[2]:.3f} m vs {mount_xyz[2]:.3f} m from the floor "
            "fit -- re-measure to the mounting hole, or the lens offset is off."
        )
    if imu:
        d_p = math.degrees(imu["accel_pitch"] - floor["pitch"])
        d_r = math.degrees(imu["accel_roll"] - floor["roll"])
        if max(abs(d_p), abs(d_r)) > 0.5:
            notes.append(
                f"IMU and floor disagree by {d_p:+.2f} deg pitch, {d_r:+.2f} deg roll: "
                "either the floor slopes (check with a level) or the IMU is off."
            )

    values = {
        "camera_mount.z": float(mount_xyz[2]),
        "camera_mount.roll": float(floor["roll"]),
        "camera_mount.pitch": float(floor["pitch"]),
    }
    if xy_source != "prior":
        values["camera_mount.x"] = float(mount_xyz[0])
        values["camera_mount.y"] = float(mount_xyz[1])
    if targets and len([r for r in target_rows if r["found"]]) >= 2:
        values["camera_mount.yaw"] = float(yaw)
    if str(cap["lens_source"]) == "tf":
        for key, v in zip(ce.LENS_KEYS, lens):
            values[f"camera_mount.{key}"] = float(v)

    return {
        "prior": prior.as_dict(),
        "measured": new.as_dict(),
        "floor": floor,
        "imu": imu,
        "xy_source": xy_source,
        "targets": target_rows,
        "lens_source": str(cap["lens_source"]),
        "values": values,
        "notes": notes,
    }


# ------------------------------------------------------------------- report


def report(result):
    p, m = result["prior"], result["measured"]
    deg = math.degrees
    lines = ["# Camera calibration", ""]
    rows = [
        (
            "depth origin x",
            p["depth_origin"][0],
            m["depth_origin"][0],
            "m",
            TOL["xy_m"],
        ),
        (
            "depth origin y",
            p["depth_origin"][1],
            m["depth_origin"][1],
            "m",
            TOL["xy_m"],
        ),
        (
            "depth origin z",
            p["depth_origin"][2],
            m["depth_origin"][2],
            "m",
            TOL["height_m"],
        ),
        (
            "roll",
            deg(p["mount"]["roll"]),
            deg(m["mount"]["roll"]),
            "deg",
            TOL["angle_deg"],
        ),
        (
            "pitch (+ nose down)",
            deg(p["mount"]["pitch"]),
            deg(m["mount"]["pitch"]),
            "deg",
            TOL["angle_deg"],
        ),
        (
            "yaw (+ left)",
            deg(p["mount"]["yaw"]),
            deg(m["mount"]["yaw"]),
            "deg",
            TOL["angle_deg"],
        ),
    ]
    lines += ["| | vehicle.yaml | measured | change | |", "|---|---|---|---|---|"]
    for name, a, b, unit, tol in rows:
        flag = "**update**" if abs(b - a) > tol else "ok"
        lines.append(
            f"| {name} | {a:.4f} {unit} | {b:.4f} {unit} | {b - a:+.4f} | {flag} |"
        )
    f = result["floor"]
    lines += [
        "",
        f"Floor: {f['frames']} frames, {f['inliers']} inliers each, plane rms {f['rms_mm']:.1f} mm, "
        f"frame-to-frame spread {f['spread_height_mm']:.1f} mm / {f['spread_pitch_deg']:.3f} deg.",
        f"x, y from: {result['xy_source']}.  Lens offset from: {result['lens_source']}.",
    ]
    i = result["imu"]
    if i:
        lines.append(
            f"IMU gravity: pitch {deg(i['accel_pitch']):+.2f} deg, roll {deg(i['accel_roll']):+.2f} deg "
            f"(|g| {i['g']:.2f})"
            + (
                f"; fused orientation pitch {deg(i['quat_pitch']):+.2f}, roll {deg(i['quat_roll']):+.2f}."
                if "quat_pitch" in i
                else "."
            )
        )
    # Where the drivers, believing vehicle.yaml, put flat floor: dh + r tan(dp)
    # above the ground, an obstacle once that passes the 0.07 m band floor.
    dh = p["depth_origin"][2] - m["depth_origin"][2]
    dp = m["mount"]["pitch"] - p["mount"]["pitch"]
    if abs(dh) > TOL["height_m"] or abs(deg(dp)) > TOL["angle_deg"]:
        reach = (BAND_FLOOR - dh) / math.tan(dp) if dp > 1e-4 else math.inf
        lines.append(
            f"Believing vehicle.yaml, the drivers put flat floor {dh * 1000:+.0f} mm "
            f"{'+ r tan(%.2f deg) ' % deg(dp) if abs(dp) > 1e-4 else ''}high"
            + (
                f": inside the {BAND_FLOOR} m obstacle band from {max(reach, 0):.1f} m out."
                if math.isfinite(reach) or dh >= BAND_FLOOR
                else "."
            )
        )
    if result["notes"]:
        lines += ["", "## Notes", ""] + [f"* {n}" for n in result["notes"]]
    lines += ["", "## Patch", ""] + [
        f"* `{k}` = {v:.6g}" for k, v in sorted(result["values"].items())
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument(
        "--target",
        action="append",
        default=[],
        type=lambda s: xy_pair(s, 2),
        metavar="X,Y",
        help="front-face center of a box, vehicle frame (repeatable)",
    )
    ap.add_argument(
        "--tape",
        type=lambda s: xy_pair(s, 3),
        metavar="X,Y,Z",
        help="camera_link (mounting hole) measured by hand",
    )
    ap.add_argument(
        "--from",
        dest="origin",
        choices=("midpoint", "rear-axle"),
        default="midpoint",
        help="what --target and --tape x are measured from (default: wheelbase midpoint)",
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--runs-dir", default="~/cfr_runs")
    ap.add_argument("--vehicle", default=str(ce.VEHICLE_YAML))
    ap.add_argument("--camera-name", default="zed")
    ap.add_argument(
        "--offline", metavar="RUN_DIR", help="re-analyze a previous capture"
    )
    args = ap.parse_args(argv)

    shift = -HALF_WHEELBASE if args.origin == "rear-axle" else 0.0
    targets = [(x + shift, y) for x, y in args.target]
    tape = None if args.tape is None else (args.tape[0] + shift, *args.tape[1:])
    prior = ce.load_mount(args.vehicle)
    if prior is None:
        raise SystemExit(f"error: no camera_mount section in {args.vehicle}")

    if args.offline:
        run_dir = Path(args.offline).expanduser()
        cap = dict(np.load(run_dir / "capture.npz"))
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}_camera_calib" + (f"_{args.label}" if args.label else "")
        run_dir = Path(args.runs_dir).expanduser() / name
        print(f"capturing {args.frames} depth frames -- keep the car still")
        cap = capture(args)
        run_dir.mkdir(parents=True)
        np.savez_compressed(run_dir / "capture.npz", **cap)

    result = analyze(cap, prior, targets, tape)
    text = report(result)
    (run_dir / "report.md").write_text(text)
    (run_dir / "result.json").write_text(json.dumps(result, indent=1, default=float))
    (run_dir / "vehicle_patch.yaml").write_text(
        yaml.safe_dump(
            {"kind": "camera_calibration", "values": result["values"]}, sort_keys=True
        )
    )
    meta = run_dir / "metadata.yaml"
    if not meta.exists():
        meta.write_text(
            yaml.safe_dump(
                {
                    "kind": "camera_calibration",
                    "label": args.label or "camera calibration",
                    "started_utc": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "options": {
                        "targets": targets,
                        "tape": tape,
                        "from": args.origin,
                        "frames": int(len(cap["depth"])),
                    },
                },
                sort_keys=False,
            )
        )
    print(text)
    print(f"run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
