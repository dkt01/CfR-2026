"""Replays that run on this laptop's ROS: a Gazebo ghost, or the bag into RViz.

Both are ordinary processes started under the system ROS environment (the Run
Lab server itself runs in a plain venv).  Lessons from validate.sh apply:

  * own ROS_DOMAIN_ID and GZ_PARTITION, so a replay can never interleave with
    a simulator somebody else has up on this machine;
  * every child leads its own process group and is stopped by GROUP -- `ros2
    launch` forks Gazebo, the bridge and sim_vehicle, and orphans keep running;
  * ports are checked for actually being bound, not assumed.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DOMAIN = int(os.environ.get("RUNLAB_REPLAY_DOMAIN", "78"))
PARTITION = os.environ.get("RUNLAB_REPLAY_PARTITION", "runlab_replay")
TELEPORT_PORT = 9023
GHOST_PORT = 9031
WEBSOCKET_PORT = 9002  # fixed by worlds/websocket.gzlaunch
VIEWER_PORT = 5173
LOG_DIR = Path(os.environ.get("RUNLAB_LOG_DIR", "/tmp/runlab"))


def port_open(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def ros_env_prefix():
    setup = REPO / "install" / "setup.bash"
    parts = ["source /opt/ros/jazzy/setup.bash"]
    if setup.exists():
        parts.append(f"source {shlex.quote(str(setup))}")
    parts.append(
        f"export ROS_DOMAIN_ID={DOMAIN} GZ_PARTITION={PARTITION} CFR_TELEPORT_PORT={TELEPORT_PORT}"
    )
    return " && ".join(parts)


def ros_available():
    return Path("/opt/ros/jazzy/setup.bash").exists()


class Proc:
    def __init__(self, name, command, log_path):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.command = command
        self.log_path = log_path
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "VIRTUAL_ENV")
        }
        env["PATH"] = ":".join(
            p for p in env.get("PATH", "").split(":") if ".venv" not in p
        )
        self.popen = subprocess.Popen(
            ["bash", "-c", command],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    def alive(self):
        return self.popen.poll() is None

    def stop(self, grace=6.0):
        if not self.alive():
            return
        for sig, wait in (
            (signal.SIGINT, grace),
            (signal.SIGTERM, 3.0),
            (signal.SIGKILL, 1.0),
        ):
            try:
                os.killpg(self.popen.pid, sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if not self.alive():
                    return
                time.sleep(0.1)

    def tail(self, lines=30):
        try:
            return Path(self.log_path).read_text(errors="ignore").splitlines()[-lines:]
        except OSError:
            return []


class ReplayManager:
    def __init__(self):
        self.procs: dict[str, Proc] = {}
        self.mode = None
        self.run = None
        self.info = {}

    # ------------------------------------------------------------- common

    def status(self):
        procs = {
            name: {"alive": p.alive(), "log": p.tail(12)}
            for name, p in self.procs.items()
        }
        ghost = None
        if (
            self.mode == "gazebo"
            and "ghost" in self.procs
            and self.procs["ghost"].alive()
        ):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{GHOST_PORT}/", timeout=0.5
                ) as r:
                    ghost = json.loads(r.read())
            except OSError:
                ghost = None
        return {
            "ros_available": ros_available(),
            "active": any(p.alive() for p in self.procs.values()),
            "mode": self.mode,
            "run": self.run,
            "procs": procs,
            "ghost": ghost,
            "domain": DOMAIN,
            "partition": PARTITION,
            "websocket": port_open(WEBSOCKET_PORT),
            "viewer": port_open(VIEWER_PORT),
            **self.info,
        }

    def stop(self):
        for name in ("ghost", "rviz", "play", "viewer", "gui", "sim"):
            if name in self.procs:
                self.procs[name].stop()
        self.procs.clear()
        self.mode = None
        self.run = None
        self.info = {}
        return self.status()

    def ghost_control(self, path, body=None):
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{GHOST_PORT}{path}", data=data, method="POST"
        )
        with urllib.request.urlopen(req, timeout=1.0) as r:
            return json.loads(r.read())

    # ------------------------------------------------------------- gazebo

    def start_gazebo(
        self, run_dir: Path, gui=False, web=True, rate=1.0, start=0.0, loop=False
    ):
        if not ros_available():
            raise RuntimeError(
                "ROS 2 Jazzy is not installed on this machine (/opt/ros/jazzy)"
            )
        if not (REPO / "install" / "cfr_arduino_bridge").exists():
            raise RuntimeError(
                "the workspace is not built: run `colcon build` at the repo root"
            )
        series = json.loads((run_dir / "analysis" / "series.json").read_text())[
            "columns"
        ]
        if "x" not in series:
            raise RuntimeError("this run has no track-frame pose to replay")
        self.stop()
        poses = LOG_DIR / "ghost_poses.json"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        poses.write_text(
            json.dumps(
                {
                    "t": series["t"],
                    "x": series["x"],
                    "y": series["y"],
                    "yaw": series["yaw"],
                }
            )
        )

        web_ok = web and not port_open(WEBSOCKET_PORT)
        self.info = {}
        if web and not web_ok:
            self.info["websocket_note"] = (
                f"port {WEBSOCKET_PORT} is already taken (another simulator with the web viewer?); "
                "this replay runs without it -- use the Gazebo window instead"
            )
        prefix = ros_env_prefix()
        launch = (
            "ros2 launch cfr_arduino_bridge speed_course.launch.py "
            "path_follower:=false cmd_vel_to_drive:=false sensors:=false "
            f"gui:={'true' if gui else 'false'} websocket:={'true' if web_ok else 'false'}"
        )
        self.procs["sim"] = Proc(
            "sim", f"{prefix} && exec {launch}", LOG_DIR / "replay_sim.log"
        )
        bridge = "/world/cfr_speed_course/set_pose@ros_gz_interfaces/srv/SetEntityPose"
        ghost = (
            f"{prefix} && (ros2 run ros_gz_bridge parameter_bridge {bridge} & "
            f"exec python3 {shlex.quote(str(HERE / 'gz_ghost.py'))} --poses {shlex.quote(str(poses))} "
            f"--rate {float(rate)} --start {float(start)} --control-port {GHOST_PORT}"
            + (" --loop" if loop else "")
            + ")"
        )
        # Give Gazebo a head start; the ghost waits for the service anyway.
        time.sleep(1.0)
        self.procs["ghost"] = Proc("ghost", ghost, LOG_DIR / "replay_ghost.log")
        if web_ok and not port_open(VIEWER_PORT):
            viewer_dir = REPO / "web" / "gzweb-viewer"
            if (viewer_dir / "node_modules").exists():
                self.procs["viewer"] = Proc(
                    "viewer",
                    f"cd {shlex.quote(str(viewer_dir))} && exec npm run dev -- --port {VIEWER_PORT} --strictPort",
                    LOG_DIR / "replay_viewer.log",
                )
            else:
                self.info["viewer_note"] = (
                    "web/gzweb-viewer has no node_modules; run `npm ci` there once"
                )
        self.mode = "gazebo"
        self.run = run_dir.name
        self.info.update(
            {
                "viewer_url": f"http://localhost:{VIEWER_PORT}/" if web_ok else None,
                "gui_command": f"{prefix} && gz sim -g",
            }
        )
        return self.status()

    def open_gazebo_gui(self):
        """A Gazebo GUI client attached to the replay's server (same partition)."""
        if self.mode != "gazebo":
            raise RuntimeError("no Gazebo replay is running")
        if "gui" in self.procs and self.procs["gui"].alive():
            return self.status()
        self.procs["gui"] = Proc(
            "gui", f"{ros_env_prefix()} && exec gz sim -g", LOG_DIR / "replay_gui.log"
        )
        return self.status()

    # --------------------------------------------------------------- rviz

    def start_rviz(self, run_dir: Path, rate=1.0, start=0.0, loop=False):
        if not ros_available():
            raise RuntimeError(
                "ROS 2 Jazzy is not installed on this machine (/opt/ros/jazzy)"
            )
        bag = run_dir / "bag"
        if not bag.exists():
            raise RuntimeError("this run has no bag directory")
        self.stop()
        prefix = ros_env_prefix()
        rviz_cfg = REPO / "rl" / "formulaOne" / "rviz" / "formula_one.rviz"
        # A simulated run recorded Gazebo's /clock, and its header stamps are
        # sim time: replay that clock rather than inventing one from receive
        # time, or every stamp is out by the sim's real-time factor.
        sim = False
        try:
            sim = json.loads((run_dir / "analysis" / "summary.json").read_text())[
                "meta"
            ]["simulation"]
        except (OSError, KeyError, ValueError):
            pass
        clock = "" if sim else " --clock 50"
        play = f"ros2 bag play {shlex.quote(str(bag))}{clock} --rate {float(rate)} --start-offset {float(start)}"
        if loop:
            play += " --loop"
        self.procs["play"] = Proc(
            "play", f"{prefix} && exec {play}", LOG_DIR / "replay_play.log"
        )
        rviz = (
            "rviz2"
            + (f" -d {shlex.quote(str(rviz_cfg))}" if rviz_cfg.exists() else "")
            + " --ros-args -p use_sim_time:=true"
        )
        self.procs["rviz"] = Proc(
            "rviz", f"{prefix} && exec {rviz}", LOG_DIR / "replay_rviz.log"
        )
        self.mode = "rviz"
        self.run = run_dir.name
        self.info = {
            "note": "The bag's own /clock is used; the formula_one RViz layout is loaded."
        }
        return self.status()
