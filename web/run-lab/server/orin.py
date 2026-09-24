"""Talking to the Orin: list runs, pull them, and (only when asked) delete them.

Over plain ssh/rsync, the way syncSoftware.sh and sync_runs.sh already do it,
so there is nothing to install or keep running on the car.  Plugged in over
USB-C the Orin is 192.168.55.1 (the L4T USB device-mode network); over Wi-Fi
set the host in the UI or ORIN_HOST.

Key-based ssh is assumed (BatchMode): a password prompt would hang a web
request.  `ssh-copy-id tejam@192.168.55.1` once, from this laptop.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

DEFAULT_HOST = os.environ.get("ORIN_HOST", "tejam@192.168.55.1")
DEFAULT_REMOTE = os.environ.get("ORIN_RUNS", "~/cfr_runs")
SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=4",
    "-o",
    "StrictHostKeyChecking=accept-new",
]

# Runs on the Orin with its stock python3 -- no yaml import, because the
# Orin's python may be the one without it; metadata is read line-wise.
LIST_SCRIPT = r"""
import json, os, sys, shutil
root = os.path.expanduser(sys.argv[1])
out = {"host": os.uname().nodename, "root": root, "runs": []}
if os.path.isdir(root):
    du = shutil.disk_usage(root)
    out["free_bytes"], out["total_bytes"] = du.free, du.total
    for name in sorted(os.listdir(root), reverse=True):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        size = 0
        for r, _, files in os.walk(path):
            for f in files:
                try:
                    size += os.path.getsize(os.path.join(r, f))
                except OSError:
                    pass
        meta = {}
        mp = os.path.join(path, "metadata.yaml")
        if os.path.exists(mp):
            for line in open(mp, errors="ignore"):
                if line[:1] not in (" ", "-", "#") and ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip("'\"")
        status = None
        sp = os.path.join(path, "status.json")
        if os.path.exists(sp):
            try:
                status = json.load(open(sp))
            except Exception:
                status = None
        out["runs"].append({
            "name": name, "bytes": size,
            "recording": os.path.exists(os.path.join(path, "RECORDING")),
            "has_bag": os.path.isdir(os.path.join(path, "bag")),
            "meta": {k: meta.get(k) for k in ("kind", "label", "profile", "started_utc", "finished_utc", "duration_s", "driver", "result")},
            "status": status,
        })
else:
    du = shutil.disk_usage(os.path.expanduser("~"))
    out["free_bytes"], out["total_bytes"] = du.free, du.total
print(json.dumps(out))
"""


def usb_link():
    """True when this laptop has an address on the Orin's USB-C network."""
    try:
        out = subprocess.run(
            ["ip", "-o", "-4", "addr"], capture_output=True, text=True, timeout=2
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return "192.168.55." in out


def ssh(host, command, stdin=None, timeout=20):
    return subprocess.run(
        ["ssh", *SSH_OPTS, host, command],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def status(host=DEFAULT_HOST, remote=DEFAULT_REMOTE):
    info = {"host": host, "remote": remote, "usb_link": usb_link()}
    try:
        result = ssh(
            host, f"python3 - {shlex.quote(remote)}", stdin=LIST_SCRIPT, timeout=25
        )
    except subprocess.TimeoutExpired:
        return {**info, "reachable": False, "error": "ssh timed out"}
    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()
        hint = ""
        if any("Permission denied" in e for e in err):
            hint = f"key-based ssh is required: run `ssh-copy-id {host}` once from this laptop"
        return {
            **info,
            "reachable": False,
            "error": (err[-1] if err else f"ssh exit {result.returncode}"),
            "hint": hint,
        }
    try:
        data = json.loads(result.stdout)
    except ValueError:
        return {**info, "reachable": False, "error": "unexpected reply from the Orin"}
    return {**info, "reachable": True, **data}


_PROGRESS = re.compile(r"([\d,]+)\s+(\d+)%\s+([\d.]+\S+/s)")


def pull(run, local_root: Path, job, host=DEFAULT_HOST, remote=DEFAULT_REMOTE):
    """rsync one run down, reporting progress into `job`."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run):
        raise ValueError("bad run name")
    local = local_root / run
    local.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-a",
        "--partial",
        "--info=progress2",
        "--no-inc-recursive",
        "-e",
        "ssh " + " ".join(SSH_OPTS),
        f"{host}:{remote}/{run}/",
        f"{local}/",
    ]
    job.log(" ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in "\r\n":
            m = _PROGRESS.search(buf)
            if m:
                job.update(
                    int(m.group(2)) / 100 * 0.95,
                    f"{int(m.group(1).replace(',', '')) / 1e6:.1f} MB at {m.group(3)}",
                )
            elif buf.strip():
                job.log(buf.strip())
            buf = ""
        else:
            buf += ch
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"rsync failed with exit code {code}")
    job.update(0.97, "verifying")
    remote_bytes = remote_size(host, remote, run)
    local_bytes = local_size(local)
    verified = remote_bytes is not None and remote_bytes == local_bytes
    (local / ".pulled.json").write_text(
        json.dumps(
            {
                "host": host,
                "remote": f"{remote}/{run}",
                "bytes": local_bytes,
                "verified": verified,
            }
        )
    )
    return {
        "run": run,
        "bytes": local_bytes,
        "remote_bytes": remote_bytes,
        "verified": verified,
    }


def local_size(run_dir: Path):
    """Bytes of what came off the car: excludes the laptop's own analysis."""
    skip = {"analysis", "analysis.tmp"}
    total = 0
    for p in run_dir.rglob("*"):
        rel = p.relative_to(run_dir).parts
        if p.is_file() and rel[0] not in skip and p.name != ".pulled.json":
            total += p.stat().st_size
    return total


def remote_size(host, remote, run):
    # Sum of file sizes, the same quantity counted locally (not du's blocks).
    result = ssh(
        host,
        f"find {remote}/{shlex.quote(run)} -type f -printf '%s\\n' | awk '{{s+=$1}} END {{print s+0}}'",
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def delete(run, local_root: Path, host=DEFAULT_HOST, remote=DEFAULT_REMOTE):
    """Remove a run from the Orin -- only after a verified local copy exists."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run):
        raise ValueError("bad run name")
    marker = local_root / run / ".pulled.json"
    if not marker.exists():
        raise RuntimeError("refusing: this run has not been pulled to the laptop")
    local_bytes = local_size(local_root / run)
    remote_bytes = remote_size(host, remote, run)
    if remote_bytes is None or remote_bytes != local_bytes:
        raise RuntimeError(
            f"refusing: local copy ({local_bytes} B) does not match the Orin ({remote_bytes} B); pull again"
        )
    result = ssh(
        host,
        f"test ! -e {remote}/{shlex.quote(run)}/RECORDING && rm -rf {remote}/{shlex.quote(run)}",
    )
    if result.returncode != 0:
        raise RuntimeError("the Orin refused (is it still recording?)")
    return {"deleted": run}
