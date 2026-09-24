#!/usr/bin/env python3
"""Record one drive on the Orin into a self-contained run directory.

    ./record_run.py --label f1_v3.5 --policy ~/cfr/rl/formulaOne/runs/v3.5/policy.npz
    ./record_run.py --label wall_test --svo --map      # heavier ZED capture

Normally started for you by formula_one.launch.py (record:=auto records
whenever use_sim_time is false, i.e. on the car).  Ctrl-C / SIGTERM finishes
the run cleanly: the bag is closed, the ZED area memory saved, ROS logs
copied, metadata finalised.  A run killed with SIGKILL still leaves a readable
bag -- MCAP is written incrementally -- but no finished metadata.

Layout, under ~/cfr_runs so sync_runs.sh and the Run Lab both find it:

    <run>/metadata.yaml     kind: drive, label, policy hash, git, options, result
    <run>/RECORDING         present only while recording is in progress
    <run>/status.json       live progress, rewritten every 2 s while recording
    <run>/bag/              rosbag2, MCAP storage
    <run>/policy/           the exact policy.npz + config.yaml that drove
    <run>/params/           parameter dumps of the nodes that were up
    <run>/zed/              area memory (.area), SVO2 if --svo
    <run>/logs/             ROS log files written during the run, tegrastats
    <run>/recorder.log      what this script said

Why the point cloud is handled specially: the ZED's registered cloud is ~3.7
MB a frame, 55 MB/s at 15 Hz -- about a gigabyte for a two-lap run, and more
write bandwidth than is sensible alongside a live drive.  `depth.point_cloud_freq`
is a dynamic ZED parameter, so the cloud rate is turned down to --cloud-hz for
the run and restored afterwards.  Nothing in the formulaOne driver reads the
cloud, so this cannot change how the car drives.

Why a regex rather than --topics: rosbag2 discovers topics as they appear, and
a regex is the one filter whose meaning is the same in every Jazzy release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions

SCRIPT_DIR = Path(__file__).resolve().parent
ZED = "/zed/zed_node"

# What a drive is judged on.  Anchored and grouped so the regex reads as a
# list; each line is one family of topics.
RECORD_TOPICS = [
    r"/rosout",  # every log line of every node, the node's telemetry lines included
    r"/tf",
    r"/tf_static",
    r"/clock",  # present in simulation only; tells the analyser the bag is sim
    r"/diagnostics",
    r"/drive_cmd",
    r"/cmd_vel",
    r"/arduino_bridge/status",
    r"/formula_one/(telemetry|car|centerline|markers)",
    r"/lap_counter/(count|done)",
    r"/start_signal_detector/(state|go)",
    ZED + r"/pose",
    ZED + r"/pose/status",
    ZED + r"/pose_with_covariance",
    ZED + r"/odom",
    ZED + r"/odom/status",
    ZED + r"/path_map",
    ZED + r"/imu/data",
    ZED + r"/status/health",
    ZED + r"/point_cloud/cloud_registered",  # rate-limited, see the docstring
    ZED + r"/mapping/fused_cloud",  # only published with --map
    ZED + r"/rgb/color/rect/camera_info",
    ZED + r"/rgb/color/rect/image/compressed",
    ZED + r"/left/camera_info",
    # Gazebo's stand-ins for the ZED, so a simulated run is analysable too.
    ZED + r"/left/image_rect_color",
]
IMAGE_TOPICS = [
    ZED + r"/rgb/color/rect/image/compressed",
    ZED + r"/left/image_rect_color",
]
CLOUD_TOPIC = ZED + r"/point_cloud/cloud_registered"

# Parameter snapshots, taken a few seconds in so late starters are up.
PARAM_NODES = [
    ZED,
    "/arduino_bridge",
    "/formula_one",
    "/lap_counter",
    "/start_signal_detector",
]


def utc_now():
    return datetime.now(timezone.utc)


def git_sha():
    """Best effort: the Orin's tree is rsynced source, often not a checkout."""
    for root in (SCRIPT_DIR, Path.home() / "cfr", Path.cwd()):
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return ""


def sha256(path: Path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dir_size(path: Path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class Recorder(Node):
    def __init__(self, args, run_dir: Path, log):
        super().__init__("run_recorder")
        self.args = args
        self.run_dir = run_dir
        self.log = log
        self.stop_requested = False
        self.bag = None
        self.tegrastats = None
        self.started = utc_now()
        self.started_mono = time.monotonic()
        self.cloud_freq_before = None
        self.cloud_dropped = False
        self.live = {
            "driver_state": None,
            "lap": None,
            "laps_target": None,
            "station": None,
            "speed": None,
            "battery_level": None,
            "mode": None,
            "estop": None,
            "pose_msgs": 0,
            "telemetry_msgs": 0,
            "min_clearance": None,
            "race_time": None,
            "finished": False,
        }

        # Cheap subscriptions only: the live status file is what the Run Lab
        # shows while a run is in progress, and none of this may cost the
        # driver anything.  The cloud and images are NOT subscribed here.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            _msg("geometry_msgs.msg", "PoseStamped"),
            ZED + "/pose",
            self.on_pose,
            qos_profile_sensor_data,
        )
        status_type = _msg("cfr_interfaces.msg", "ArduinoStatus")
        if status_type:
            self.create_subscription(
                status_type, "/arduino_bridge/status", self.on_status, 10
            )
        telemetry_type = _msg("cfr_interfaces.msg", "DriverTelemetry")
        if telemetry_type:
            self.create_subscription(
                telemetry_type, "/formula_one/telemetry", self.on_telemetry, 50
            )
        self.create_subscription(
            _msg("std_msgs.msg", "Bool"), "/lap_counter/done", self.on_done, latched
        )
        self.create_timer(2.0, self.write_status)

    # ---------------------------------------------------------------- live

    def on_pose(self, _msg_in):
        self.live["pose_msgs"] += 1

    def on_status(self, msg):
        self.live["speed"] = round(float(msg.speed), 3)
        self.live["battery_level"] = int(msg.battery_level)
        self.live["mode"] = int(msg.mode)
        self.live["estop"] = bool(msg.estop)

    def on_telemetry(self, msg):
        self.live["telemetry_msgs"] += 1
        self.live["driver_state"] = int(msg.state)
        self.live["lap"] = int(msg.lap)
        self.live["laps_target"] = int(msg.laps_target)
        if msg.state in (1, 2):  # running, stopping
            self.live["station"] = round(float(msg.station), 2)
            self.live["race_time"] = round(float(msg.race_time), 2)
            clear = float(msg.clearance)
            prev = self.live["min_clearance"]
            self.live["min_clearance"] = round(
                clear if prev is None else min(prev, clear), 4
            )
        if msg.state == 3:
            self.live["finished"] = True

    def on_done(self, msg):
        if msg.data:
            self.live["finished"] = True

    def write_status(self):
        status = {
            "recording": True,
            "run": self.run_dir.name,
            "elapsed_s": round(time.monotonic() - self.started_mono, 1),
            "bag_bytes": dir_size(self.run_dir / "bag"),
            "bag_alive": self.bag is not None and self.bag.poll() is None,
            "free_bytes": shutil.disk_usage(self.run_dir).free,
            **self.live,
        }
        tmp = self.run_dir / "status.json.tmp"
        tmp.write_text(json.dumps(status))
        tmp.replace(self.run_dir / "status.json")
        if not status["bag_alive"] and self.bag is not None:
            self.log(
                f"WARNING: ros2 bag record exited with {self.bag.poll()}; see bag.log"
            )

    # ------------------------------------------------------------ services

    def call(self, srv_type, name, request, timeout=5.0):
        """Call a service if it exists.  None when it does not or times out."""
        client = self.create_client(srv_type, name)
        try:
            if not client.wait_for_service(timeout_sec=min(2.0, timeout)):
                return None
            future = client.call_async(request)
            deadline = time.monotonic() + timeout
            while not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            return future.result() if future.done() else None
        finally:
            self.destroy_client(client)

    def get_param(self, node, name):
        from rcl_interfaces.srv import GetParameters

        response = self.call(
            GetParameters, f"{node}/get_parameters", GetParameters.Request(names=[name])
        )
        if response is None or not response.values:
            return None
        value = response.values[0]
        return {3: value.double_value, 2: value.integer_value}.get(value.type)

    def set_param_double(self, node, name, number):
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters

        param = Parameter(
            name=name,
            value=ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE, double_value=float(number)
            ),
        )
        response = self.call(
            SetParameters,
            f"{node}/set_parameters",
            SetParameters.Request(parameters=[param]),
        )
        return bool(response and response.results and response.results[0].successful)

    # --------------------------------------------------------------- start

    def start(self):
        args = self.args
        if args.cloud_hz > 0:
            self.cloud_freq_before = self.get_param(ZED, "depth.point_cloud_freq")
            if self.cloud_freq_before is not None and self.set_param_double(
                ZED, "depth.point_cloud_freq", args.cloud_hz
            ):
                self.log(
                    f"ZED point cloud {self.cloud_freq_before} Hz -> {args.cloud_hz} Hz for the run"
                )
            elif args.cloud_any_rate:
                self.log(
                    "ZED cloud rate not controllable; recording it at its own rate (--cloud-any-rate)"
                )
            else:
                # Uncontrolled, the cloud is ~55 MB/s: a two-lap run would be
                # gigabytes and could fill the Orin mid-drive.  Leave it out.
                self.cloud_dropped = True
                self.log(
                    "WARNING: could not lower the ZED cloud rate -- the point cloud is NOT "
                    "recorded (pass --cloud-any-rate to record it anyway)"
                )

        if args.map:
            from std_srvs.srv import SetBool

            ok = self.call(
                SetBool, f"{ZED}/enable_mapping", SetBool.Request(data=True), 10.0
            )
            self.log(
                "ZED spatial mapping ON"
                if ok and ok.success
                else f"ZED spatial mapping NOT enabled ({getattr(ok, 'message', 'no service')})"
            )

        if args.svo:
            svo_type = _srv("zed_msgs.srv", "StartSvoRec")
            if svo_type is None:
                self.log("zed_msgs not importable; --svo ignored")
            else:
                req = svo_type.Request()
                req.bitrate = 0
                req.compression_mode = 1  # H264
                req.target_framerate = 0
                req.input_transcode = False
                req.svo_filename = str(self.run_dir / "zed" / "run.svo2")
                ok = self.call(svo_type, f"{ZED}/start_svo_rec", req, 10.0)
                self.log(
                    f"SVO recording to {req.svo_filename}"
                    if ok and ok.success
                    else f"SVO NOT started ({getattr(ok, 'message', 'no service')})"
                )

        regex = "^(" + "|".join(self.topic_list()) + ")$"
        command = [
            "ros2",
            "bag",
            "record",
            "-s",
            "mcap",
            "-o",
            str(self.run_dir / "bag"),
            "-e",
            regex,
            "--polling-interval",
            "200",
        ]
        if args.compress:
            command += ["--storage-preset-profile", "zstd_fast"]
        self.bag = subprocess.Popen(
            command,
            stdout=open(self.run_dir / "bag.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.log(f"ros2 bag record (pid {self.bag.pid}) -> {self.run_dir / 'bag'}")

        if shutil.which("tegrastats"):
            self.tegrastats = subprocess.Popen(
                ["tegrastats", "--interval", "1000"],
                stdout=open(self.run_dir / "logs" / "tegrastats.log", "w"),
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        threading.Thread(target=self.dump_params, daemon=True).start()

    def topic_list(self):
        topics = list(RECORD_TOPICS)
        if self.args.no_images:
            topics = [t for t in topics if t not in IMAGE_TOPICS]
        if self.args.cloud_hz <= 0 or self.cloud_dropped:
            topics = [t for t in topics if t != CLOUD_TOPIC]
        topics += [re.escape(t) for t in self.args.extra_topic]
        return topics

    def dump_params(self):
        time.sleep(4.0)
        out = self.run_dir / "params"
        for node in PARAM_NODES:
            try:
                result = subprocess.run(
                    ["ros2", "param", "dump", node],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0 and result.stdout.strip():
                (out / (node.strip("/").replace("/", "__") + ".yaml")).write_text(
                    result.stdout
                )

    # -------------------------------------------------------------- finish

    def finish(self):
        """Every step on its own: one failing must not cost the others --
        least of all closing the bag."""
        for step in (
            self.finish_zed,
            self.restore_cloud_rate,
            lambda: stop_group(self.bag, "bag", self.log, timeout=30),
            lambda: stop_group(self.tegrastats, "tegrastats", self.log, timeout=5),
            lambda: copy_ros_logs(
                self.run_dir / "logs" / "ros", self.started.timestamp(), self.log
            ),
        ):
            try:
                step()
            except Exception as error:  # noqa: BLE001
                self.log(f"finish step failed: {error!r}")

    def finish_zed(self):
        args = self.args
        if args.svo:
            from std_srvs.srv import Trigger

            ok = self.call(Trigger, f"{ZED}/stop_svo_rec", Trigger.Request(), 15.0)
            self.log("SVO closed" if ok and ok.success else "SVO stop: no answer")

        if not args.no_area:
            area_type = _srv("zed_msgs.srv", "SaveAreaMemory")
            if area_type is not None:
                req = area_type.Request()
                req.area_file_path = str(self.run_dir / "zed" / "area_memory.area")
                ok = self.call(area_type, f"{ZED}/save_area_memory", req, 20.0)
                self.log(
                    f"area memory -> {req.area_file_path}"
                    if ok and ok.success
                    else f"area memory not saved ({getattr(ok, 'message', 'no service')})"
                )

        if args.map:
            # Leave one more fused cloud in the bag before stopping; the
            # wrapper publishes at mapping.fused_pointcloud_freq (1 Hz default).
            time.sleep(1.5)

    def restore_cloud_rate(self):
        if self.cloud_freq_before is not None:
            self.set_param_double(ZED, "depth.point_cloud_freq", self.cloud_freq_before)
            self.log(f"ZED point cloud restored to {self.cloud_freq_before} Hz")


def _msg(module, name):
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except (ImportError, AttributeError):
        return None


_srv = _msg


def stop_group(process, what, log, timeout):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout)
        log(f"{what} stopped")
    except subprocess.TimeoutExpired:
        log(f"{what} did not stop in {timeout} s; killing")
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def copy_ros_logs(dest: Path, since: float, log, cap_bytes=200 << 20):
    """Every ROS log file written during the run.

    ROS_LOG_DIR, else ~/.ros/log.  The launch log carries the driver's own
    stdout (it runs as an ExecuteProcess with output=screen), so this is the
    only copy of anything printed before rclpy's logger was up.
    """
    root = Path(os.environ.get("ROS_LOG_DIR") or Path.home() / ".ros" / "log")
    if not root.is_dir():
        return
    copied = 0
    for path in sorted(root.rglob("*")):
        try:
            if not path.is_file() or path.stat().st_mtime < since - 5:
                continue
            size = path.stat().st_size
            if copied + size > cap_bytes:
                log(f"ROS log copy capped at {cap_bytes >> 20} MB")
                break
            target = dest / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += size
        except OSError:
            continue
    log(f"copied {copied / 1e6:.1f} MB of ROS logs")


def bag_summary(bag_dir: Path):
    meta = bag_dir / "metadata.yaml"
    if not meta.exists():
        return None
    try:
        info = yaml.safe_load(meta.read_text())["rosbag2_bagfile_information"]
        # Jazzy writes the singular key; tolerate the plural some tools use.
        topics = info.get("topics_with_message_count") or info.get(
            "topics_with_message_counts", []
        )
        return {
            "duration_s": round(info["duration"]["nanoseconds"] * 1e-9, 2),
            "messages": info["message_count"],
            "topics": {t["topic_metadata"]["name"]: t["message_count"] for t in topics},
        }
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--label", default="drive", help="short name, goes in the directory name"
    )
    parser.add_argument("--runs-dir", default=os.environ.get("CFR_RUNS", "~/cfr_runs"))
    parser.add_argument(
        "--policy", default="", help="policy.npz that is driving (copied in)"
    )
    parser.add_argument(
        "--config", default="", help="config.yaml that is driving (copied in)"
    )
    parser.add_argument("--driver", default="", help="policy | baseline | anything")
    parser.add_argument("--speed-scale", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--surface", default="")
    parser.add_argument(
        "--cloud-hz",
        type=float,
        default=1.0,
        help="ZED cloud rate for the run; 0 = no cloud",
    )
    parser.add_argument(
        "--cloud-any-rate",
        action="store_true",
        help="record the cloud even when its rate cannot be lowered (simulation)",
    )
    parser.add_argument(
        "--no-images", action="store_true", help="skip the camera stream"
    )
    parser.add_argument("--map", action="store_true", help="enable ZED spatial mapping")
    parser.add_argument(
        "--svo", action="store_true", help="record a ZED SVO2 alongside"
    )
    parser.add_argument(
        "--no-area", action="store_true", help="skip saving ZED area memory"
    )
    parser.add_argument(
        "--compress", action="store_true", help="zstd MCAP chunks (costs CPU)"
    )
    parser.add_argument("--min-free-gb", type=float, default=3.0)
    parser.add_argument("--extra-topic", action="append", default=[])
    # ros2 launch appends --ros-args; tolerate it.
    args, _unknown = parser.parse_known_args()

    runs = Path(os.path.expanduser(args.runs_dir))
    runs.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(runs).free / 1e9
    if free_gb < args.min_free_gb:
        print(
            f"record_run: only {free_gb:.1f} GB free under {runs}; refusing to record.\n"
            f"            pull runs with the Run Lab (or sync_runs.sh --purge) first.",
            file=sys.stderr,
        )
        return 2

    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.label).strip("_") or "drive"
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs / f"{stamp}_{label}"
    for sub in ("policy", "params", "zed", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "RECORDING").write_text(str(os.getpid()))

    log_file = open(run_dir / "recorder.log", "a", buffering=1)

    def log(text):
        line = f"[{utc_now().strftime('%H:%M:%S')}] {text}"
        print(f"record_run: {text}", flush=True)
        log_file.write(line + "\n")

    shipped = {}
    for kind in ("policy", "config"):
        source = getattr(args, kind)
        if source and Path(os.path.expanduser(source)).is_file():
            source_path = Path(os.path.expanduser(source))
            shutil.copy2(source_path, run_dir / "policy" / source_path.name)
            shipped[kind] = {"path": str(source_path), "sha256": sha256(source_path)}

    metadata = {
        "kind": "drive",
        "label": label,
        "started_utc": utc_now().isoformat(),
        "host": socket.gethostname(),
        "git_sha": git_sha(),
        "driver": args.driver,
        "speed_scale": args.speed_scale,
        "surface": args.surface,
        "notes": args.notes,
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "recorder": {
            "cloud_hz": args.cloud_hz,
            "cloud_any_rate": args.cloud_any_rate,
            "images": not args.no_images,
            "map": args.map,
            "svo": args.svo,
            "area_memory": not args.no_area,
            "compress": args.compress,
        },
        "free_gb_at_start": round(free_gb, 1),
        **shipped,
    }
    meta_path = run_dir / "metadata.yaml"
    meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    log(f"recording into {run_dir}  ({free_gb:.1f} GB free)")

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Recorder(args, run_dir, log)

    def request_stop(signum, _frame):
        if not node.stop_requested:
            log(f"signal {signum}: finishing the run")
        node.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        node.start()
        while not node.stop_requested and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.bag.poll() is not None:
                log("ros2 bag record exited on its own; finishing")
                break
        node.finish()
    finally:
        metadata.update(
            {
                "finished_utc": utc_now().isoformat(),
                "duration_s": round(time.monotonic() - node.started_mono, 1),
                "bag": bag_summary(run_dir / "bag"),
                "live_at_end": node.live,
                "run_bytes": dir_size(run_dir),
            }
        )
        meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
        status_path = run_dir / "status.json"
        status = {"recording": False, "run": run_dir.name, **node.live}
        status_path.write_text(json.dumps(status))
        (run_dir / "RECORDING").unlink(missing_ok=True)
        log(f"run complete: {run_dir}  ({metadata['run_bytes'] / 1e6:.0f} MB)")
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
