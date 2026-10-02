"""Launch a plant-checkout profile on the Orin, then run the same analyze ->
apply -> propagate -> consistency-check chain the manual workflow already
uses (docs/characterization-steering.md), from the browser.

The hardware E-Stop interlock is unchanged: maneuver_runner_node.py still
requires the witnessed assert/clear cycle before it arms anything.  This
module is a convenience wrapper around the exact `ros2 launch` command
someone would otherwise type at the bench -- it has no more authority over
the car than that command does, and it cannot skip the interlock.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orin  # noqa: E402

REPO = HERE.parents[2]
JETSON_SCRIPTS = REPO / "jetson" / "scripts"
VEHICLE_YAML = REPO / "jetson" / "cfr_arduino_bridge" / "config" / "vehicle.yaml"
ARDUINO_BRIDGE_YAML = (
    REPO / "jetson" / "cfr_arduino_bridge" / "config" / "arduino_bridge.yaml"
)

sys.path.insert(0, str(JETSON_SCRIPTS))

# Profiles this page offers, and what each one's patch should propagate to
# once applied.  "trim" -> vehicle.yaml + the real steering_trim parameter, and
# then propagate_plant.py too: the sim and RL tables are vehicle.yaml's raw
# table shifted by center_offset, so a new offset moves every copy even though
# the table itself was not re-measured.  "plant" -> vehicle.yaml + the same
# two steps, since a figure-8 re-measures both the table and the offset.
# Past this the Orin's clock is wrong, not just drifting.
CLOCK_SKEW_S = 60

PROFILES = {
    "straight_line_trim": {
        "description": "10 m at 1 m/s, dead straight - a fast steering.center_offset re-check.",
        "propagation": "trim",
    },
    "figure_eight_calib": {
        "description": "Combined turn-in-lag / steering-map / understeer checkout, one session.",
        "propagation": "plant",
    },
}


def launch(profile, job, host):
    """SSH-launch characterize.launch.py, streaming the runner's log into
    `job` and returning the run directory name it created.

    Blocks until the launch exits: the runner finished, aborted, or the ssh
    connection dropped. Requires the same physical presence at the bench with
    the E-Stop remote that typing this command locally would.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    now = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    try:
        skew = int(orin.ssh(host, "date -u +%s", timeout=10).stdout.strip()) - now
    except (ValueError, subprocess.TimeoutExpired):
        skew = None
    if skew is None or abs(skew) > CLOCK_SKEW_S:
        reads = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now + skew))
            if skew is not None
            else "unreadable"
        )
        job.log(
            f"WARNING: the Orin's clock reads {reads} UTC, not {stamp}. The run is named "
            f"from this laptop's clock, but its file times and metadata timestamps are "
            f"the Orin's. Set it: ssh {host} 'sudo date -u -s @{int(now)}'"
        )
    command = orin.ros_command(
        f"ros2 launch cfr_arduino_bridge characterize.launch.py profile:={shlex.quote(profile)}"
        f" stamp:={stamp}"
    )
    job.log(f"ssh {host} {command}")
    proc = subprocess.Popen(
        ["ssh", *orin.SSH_OPTS, host, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    run_dir = None
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch != "\n":
            buf += ch
            continue
        line = buf.strip()
        buf = ""
        if not line:
            continue
        job.log(line)
        match = re.search(r"recording to (\S+)", line) or re.search(
            r"run directory:\s*(\S+)", line
        )
        if match:
            run_dir = match.group(1).rstrip(":")
    if buf.strip():
        job.log(buf.strip())
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"characterize.launch.py exited {code} - see the log above")
    if run_dir is None:
        raise RuntimeError(
            "launch exited cleanly but never logged a run directory - check runner.log on the car"
        )
    return {"profile": profile, "run": Path(run_dir).name}


def pull_and_analyze(run_name, job, host, remote, runs_local):
    """Pull the run (orin.pull, unchanged) then run jetson/scripts/analyze_run.py
    locally -- it is stdlib-only by design, so it needs no ROS install here."""
    orin.pull(run_name, runs_local, job, host, remote)
    job.update(0.99, "analyzing")
    run_path = runs_local / run_name
    result = subprocess.run(
        [
            sys.executable,
            str(JETSON_SCRIPTS / "analyze_run.py"),
            str(run_path),
            "--speed-source",
            "wheel_rpm",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"analyze_run.py failed:\n{result.stdout}\n{result.stderr}")
    # Camera mount, depth band and tracking-frame checks into the same
    # report.  A failure here must not lose the steering result above.
    job.update(0.995, "checking the camera and pose")
    check = subprocess.run(
        [sys.executable, str(HERE / "camera_check.py"), str(run_path), "--report"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        job.log(f"camera_check.py failed:\n{check.stdout}\n{check.stderr}")
    patch_path = run_path / "vehicle_patch.yaml"
    report_path = run_path / "report.md"
    import yaml

    patch = yaml.safe_load(patch_path.read_text()) if patch_path.exists() else None
    camera_path = run_path / "camera_check.json"
    return {
        "run": run_name,
        "report": report_path.read_text() if report_path.exists() else "",
        "values": (patch or {}).get("values", {}),
        "camera_check": json.loads(camera_path.read_text())
        if camera_path.exists()
        else None,
    }


def _run_script(name, *args):
    result = subprocess.run(
        [sys.executable, str(JETSON_SCRIPTS / name), *args],
        capture_output=True,
        text=True,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "output": (result.stdout + result.stderr).strip(),
    }


def apply(run_name, profile, runs_local):
    """apply_vehicle_patch.py, then whichever propagation this profile needs,
    then check_steering_consistency.py -- the same chain the manual workflow
    runs, in the same order, surfacing every step's own output."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    run_path = runs_local / run_name
    steps = [
        ("apply_vehicle_patch", _run_script("apply_vehicle_patch.py", str(run_path)))
    ]

    # Both profiles can move center_offset, and propagate_plant.py shifts the
    # table by it, so the bridge trim and the table copies must change together.
    steps.append(("apply_steering_trim", _run_script("apply_steering_trim.py")))
    steps.append(("propagate_plant", _run_script("propagate_plant.py")))

    consistency = _run_script("check_steering_consistency.py")
    steps.append(("check_steering_consistency", consistency))

    return {"run": run_name, "steps": steps, "consistent": consistency["ok"]}
