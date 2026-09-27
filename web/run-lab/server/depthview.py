"""The ZED depth image, and what the formulaTwo driver made of it.

record_run.py keeps one depth frame every 1/--depth-hz s (default 2 Hz) as
16-bit millimeters on /run_recorder/depth[/compressed].  Each one is logged
to Rerun with the driver's sampling grid drawn over it -- the same pixels
formula_two_node reads, computed by the same code (rl/formulaTwo/perception.py,
imported, not copied) -- and the virtual-LiDAR scan that grid gives, top-down.

Beside that recomputed scan goes the one the driver actually acted on: the
newest frame of the scan stack in /formula_one/telemetry's `observation`,
decoded back to meters.  The two should agree to within a frame's motion; if
they do not, the node's leveling or intrinsics differ from what the Run Lab
assumed.

Entities (see rerun_export.RunRecording.depth_*):

    depth/image     DepthImage, mm
    depth/grid      Points2D on the image: gray sampled, green in the height
                    band, orange the nearest in-band pixel of each column
    scan/driver     the driver's beams, 10 Hz (red stubs: invalid columns)
    scan/depth      the scan recomputed from the recorded frame, 2 Hz
    scan/car, scan/rings   static, for scale

The scan views are top-down in a 2D view: screen x = -y (left is left),
screen y = -x (forward is up), meters, camera frame origin at the car's
reference point.
"""

from __future__ import annotations

import importlib.util
import io
import math
import sys
from pathlib import Path

import numpy as np
import rerun_export
import yaml

REPO = Path(__file__).resolve().parents[3]
F2 = REPO / "rl" / "formulaTwo"
ZED = "/zed/zed_node"

DEPTH_TOPICS = [
    "/run_recorder/depth/compressed",
    "/run_recorder/depth",
    ZED + "/depth/depth_registered",  # only if recorded with --extra-topic
]
INFO_TOPICS = [
    ZED + "/depth/camera_info",
    ZED + "/left/image_rect_color/camera_info",  # Gazebo's rgbd_camera
]
TELEMETRY = "/formula_one/telemetry"


def _perception():
    """rl/formulaTwo/perception.py under its own name: rl/formulaOne is on
    sys.path already (course.py), and it must not shadow or be shadowed."""
    name = "cfr_f2_perception"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, F2 / "perception.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def camera_config(run_dir: Path):
    """The config whose `camera` block the grid follows, and where it came
    from: the run's own (a formulaTwo run), else formulaTwo's default."""
    shipped = Path(run_dir) / "policy" / "config.yaml"
    if shipped.exists():
        try:
            cfg = yaml.safe_load(shipped.read_text()) or {}
            if "camera" in cfg:
                return cfg, "run"
        except yaml.YAMLError:
            pass
    return yaml.safe_load((F2 / "config.yaml").read_text()), "formulaTwo default"


def decode(topic, msg):
    """A recorded depth message -> (H, W) float meters, NaN where none."""
    if topic.endswith("compressed"):
        from PIL import Image as PILImage

        mm = np.array(PILImage.open(io.BytesIO(bytes(msg.data))), dtype=np.float64)
        depth = mm / 1000.0
        depth[mm <= 0] = np.nan
        return depth
    data = (
        bytes(msg.data) if not isinstance(msg.data, np.ndarray) else msg.data.tobytes()
    )
    big = bool(msg.is_bigendian)
    if msg.encoding == "32FC1":
        rows = np.frombuffer(data, dtype=">f4" if big else "<f4").reshape(
            msg.height, msg.step // 4
        )
        depth = rows[:, : msg.width].astype(np.float64)
    elif msg.encoding in ("16UC1", "mono16"):
        rows = np.frombuffer(data, dtype=">u2" if big else "<u2").reshape(
            msg.height, msg.step // 2
        )
        mm = rows[:, : msg.width].astype(np.float64)
        depth = mm / 1000.0
        depth[mm <= 0] = np.nan
    else:
        return None
    depth[~np.isfinite(depth)] = np.nan
    return depth


def intrinsics(bag, cam, width, height):
    """(fx, fy, cx, cy) for a width x height depth image: from the recorded
    camera_info, scaled as formula_two_node scales it, else the config's
    nominal pinhole."""
    for topic in INFO_TOPICS:
        if not bag.has(topic):
            continue
        for _, _, msg in bag.messages(topic):
            k = list(msg.k)
            if k[0] <= 0:
                continue
            fx, fy, cx, cy = k[0], k[4], k[2], k[5]
            if (msg.width, msg.height) != (width, height) and msg.width and msg.height:
                sx, sy = width / msg.width, height / msg.height
                fx, cx = fx * sx, (cx + 0.5) * sx - 0.5
                fy, cy = fy * sy, (cy + 0.5) * sy - 0.5
            return (fx, fy, cx, cy), topic
    f = (width / 2) / math.tan(math.radians(cam.hfov_deg) / 2)
    return (f, f, width / 2 - 0.5, height / 2 - 0.5), None


def attitude_at(streams, t):
    """(roll, pitch) at bag time t: the ZED IMU within 0.2 s, else the pose,
    else level -- formula_two_node's order."""
    imu = streams.get(ZED + "/imu/data")
    if imu is not None and imu.shape[1] >= 9 and len(imu):
        i = int(np.clip(np.searchsorted(imu[:, 0], t), 1, len(imu) - 1))
        j = i if abs(imu[i, 0] - t) < abs(imu[i - 1, 0] - t) else i - 1
        if abs(imu[j, 0] - t) < 0.2:
            return float(imu[j, 7]), float(imu[j, 8])
    pose = streams.get(ZED + "/pose")
    if pose is not None and len(pose):
        i = int(np.clip(np.searchsorted(pose[:, 0], t), 1, len(pose) - 1))
        x, y, z, w = pose[i, 5], pose[i, 6], pose[i, 7], pose[i, 8]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return roll, pitch
    return 0.0, 0.0


def grid_pixels(cam, fx, fy, cx, cy, width, height):
    """(rows, W) pixel centers of the driver's grid, as sample_depth picks them."""
    u = np.round(cx - cam.a * fx).astype(int)
    v = np.round(cy - cam.b * fy).astype(int)
    ok = ((v >= 0) & (v < height))[:, None] & ((u >= 0) & (u < width))[None, :]
    uu = np.broadcast_to(u[None, :], ok.shape)
    vv = np.broadcast_to(v[:, None], ok.shape)
    return uu, vv, ok


def in_band(cam, grid, height, pitch, roll):
    """(rows, W) mask of the grid points depth_to_scan keeps, and (W,) the
    row index each column's beam came from (-1: none).  depth_to_scan itself
    returns only the ranges; this repeats its geometry to say WHICH pixel."""
    a = cam.a[None, :]
    bb = cam.b[:, None]
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    y1 = a * cr - bb * sr
    z1 = a * sr + bb * cr
    x2 = cp + z1 * sp
    z2 = -sp + z1 * cp
    with np.errstate(invalid="ignore"):
        r = grid * np.sqrt(x2 * x2 + y1 * y1)
        z = height + grid * z2
        keep = np.isfinite(grid) & (z >= cam.band[0]) & (z <= cam.band[1])
    rk = np.where(keep, r, np.inf)
    beam = np.where(np.isfinite(rk).any(axis=0), rk.argmin(axis=0), -1)
    return keep, beam


def decode_scan(cam, encoded):
    """Encoded beams -> meters (inf: saturated at scan_max, nan: invalid)."""
    e = np.asarray(encoded, dtype=float)
    r = cam_r0() * np.exp(e * math.log(cam.scan_max / cam_r0()))
    r = np.where(e >= 1.0 - 1e-6, np.inf, r)
    return np.where(e < 0, np.nan, r)


def cam_r0():
    return float(getattr(_perception(), "_R0", 0.25))


def beams_xy(cam, ranges, stub=0.3):
    """Beam endpoints in the car frame; invalid columns get a short stub and
    saturated ones end at scan_max.  Returns (starts, ends, invalid mask)."""
    r = np.asarray(ranges, dtype=float)
    invalid = np.isnan(r)
    r = np.where(invalid, stub, np.minimum(r, cam.scan_max))
    x = cam.x + r * np.cos(cam.azimuth)
    y = r * np.sin(cam.azimuth)
    start = np.column_stack([np.full_like(x, cam.x), np.zeros_like(y)])
    return start, np.column_stack([x, y]), invalid


def depth_size(bag):
    """(width, height) of the depth frames as logged to Rerun, or None."""
    topic = next((tp for tp in DEPTH_TOPICS if bag.has(tp)), None)
    if topic is None:
        return None
    for _, _, msg in bag.messages(topic):
        depth = decode(topic, msg)
        if depth is not None:
            step = rerun_export.depth_step(depth.shape[1])
            return [
                len(range(0, depth.shape[1], step)),
                len(range(0, depth.shape[0], step)),
            ]
    return None


def write_depth(bag, run_dir, streams, tb, recording, progress, hz=2.0, scan_hz=10.0):
    topic = next((tp for tp in DEPTH_TOPICS if bag.has(tp)), None)
    tel_ok = bag.has(TELEMETRY)
    info = {"depth_frames": 0, "driver_scans": 0}
    if topic is None and not tel_ok:
        return info
    P = _perception()
    cfg, source = camera_config(run_dir)
    cam = P.Camera(cfg)
    cam.hfov_deg = float(cfg["camera"]["hfov_deg"])
    info["grid_config"] = source
    recording.depth_static(cam)

    # The driver's own scan, from its observation: [map | stack x W | age].
    if tel_ok:
        span = cam.stack * cam.width
        last = -1e9
        for _, t_ns, msg in bag.messages(TELEMETRY):
            obs = np.asarray(msg.observation, dtype=float)
            if not len(obs):
                continue  # waiting for the start: nothing observed yet
            if len(obs) < span + 2:
                break  # a formulaOne driver: no depth in its observation
            t = t_ns * 1e-9
            if t - last < 1.0 / scan_hz:
                continue
            last = t
            newest = obs[len(obs) - 1 - span : len(obs) - 1 - span + cam.width]
            start, end, invalid = beams_xy(cam, decode_scan(cam, newest))
            recording.depth_scan("driver", float(tb(t)), start, end, invalid)
            info["driver_scans"] += 1

    if topic is None:
        return info
    total = bag.topics()[topic]["count"]
    every = int(1e9 / hz) if not topic.startswith("/run_recorder") else 0
    k = None
    for n, (_, t_ns, msg) in enumerate(bag.messages(topic, every_ns=every)):
        depth = decode(topic, msg)
        if depth is None:
            continue
        hgt, wid = depth.shape
        if k is None:
            k, info_topic = intrinsics(bag, cam, wid, hgt)
            # As logged (rerun_export shrinks it), which the layout frames.
            step = rerun_export.depth_step(wid)
            info["depth_size"] = [len(range(0, wid, step)), len(range(0, hgt, step))]
            info["depth_recorded_size"] = [wid, hgt]
            info["depth_intrinsics"] = info_topic or "nominal (no camera_info recorded)"
        t = t_ns * 1e-9
        roll, pitch = attitude_at(streams, t)
        grid = cam.sample_depth(depth, *k)[0]
        keep, beam = in_band(cam, grid, cam.z, pitch, roll)
        scan = cam.depth_to_scan(grid[None], None, np.array([pitch]), np.array([roll]))[
            0
        ]
        uu, vv, ok = grid_pixels(cam, *k, wid, hgt)
        start, end, invalid = beams_xy(cam, np.where(np.isfinite(scan), scan, np.nan))
        recording.depth_frame(
            float(tb(t)),
            depth,
            uu,
            vv,
            ok,
            keep,
            beam,
        )
        recording.depth_scan("depth", float(tb(t)), start, end, invalid)
        info["depth_frames"] += 1
        if n % 20 == 0:
            progress(0.88 + 0.04 * min(1.0, n / max(1, total)), f"depth ({n} frames)")
    info["depth_topic"] = topic
    return info
