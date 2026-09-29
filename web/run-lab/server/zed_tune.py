"""Live ZED exposure/ROI tuning: `ros2 param set` over ssh for immediate
feedback while at the bench, plus saving the finished values into
jetson/cfr_arduino_bridge/config/cfr_zed2i.yaml (the one copy -
zed/config/cfr_zed2i.yaml is a symlink to it).

Live tuning needs zed_live_tuning.launch.py running on the car already
(rosbridge for `ros2 param set`/live values, web_video_server for the image
and the ROI mask) -- see that launch file for why it is camera-only and
carries none of launch.sh's actuator-arming risk. Saving only touches this
laptop's checkout; syncing it onto the car is the same deploy-to-car flow any
other config change uses.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orin  # noqa: E402

REPO = HERE.parents[2]
ZED_CONFIG = REPO / "jetson" / "cfr_arduino_bridge" / "config" / "cfr_zed2i.yaml"

NODE = "/zed/zed_node"


def set_param(host, name, value):
    """`ros2 param set` on the live node, for immediate visual feedback.

    Not persisted anywhere by itself -- a relaunch of the ZED node reverts to
    whatever cfr_zed2i.yaml says, which is what `save()` below writes.
    """
    command = f"ros2 param set {NODE} {shlex.quote(name)} {shlex.quote(str(value))}"
    result = orin.ssh(host, command, timeout=10)
    ok = result.returncode == 0 and "Set parameter successful" in result.stdout
    return {"ok": ok, "output": (result.stdout + result.stderr).strip()}


def get_param(host, name):
    command = f"ros2 param get {NODE} {shlex.quote(name)}"
    result = orin.ssh(host, command, timeout=10)
    return {
        "ok": result.returncode == 0,
        "output": result.stdout.strip() or result.stderr.strip(),
    }


def stream_urls(host):
    """web_video_server/rosbridge URLs. Same host as the ssh connection, not
    a settings knob of their own -- one fewer place tuning drifts out of sync
    with orin.DEFAULT_HOST."""
    hostname = host.split("@")[-1]
    return {
        "rosbridge": f"ws://{hostname}:9090",
        "video": f"http://{hostname}:8080/stream?topic={NODE}/rgb/image_rect_color",
        "roi_mask": f"http://{hostname}:8080/stream?topic={NODE}/roi_mask/image",
    }


_EXPOSURE_LINE = re.compile(r"(?m)^(\s*)#?\s*(exposure|gain):[ \t]*(\S*)[ \t]*$")
_AUTO_LINE = re.compile(r"(?m)^(\s*auto_exposure_gain:)[ \t]*\S+")
_ROI_LINE = re.compile(r"(?m)^(\s*manual_polygon:)[ \t]*'[^']*'")


def _set_exposure(text, auto, exposure, gain):
    """Toggle auto_exposure_gain and, in manual mode, exposure/gain -
    matching this file's own convention of commenting the pair out under
    auto (see its header: both are dynamic, tune on the course)."""
    text = _AUTO_LINE.sub(rf"\g<1> {'true' if auto else 'false'}", text, count=1)

    def rewrite(match):
        indent, name, current = match.groups()
        value = exposure if name == "exposure" else gain
        if value is None:
            value = current or "0"
        prefix = "#" if auto else ""
        return f"{indent}{prefix}{name}: {value}"

    return _EXPOSURE_LINE.sub(rewrite, text)


def _set_roi(text, polygon):
    rendered = "[" + ",".join(f"[{x},{y}]" for x, y in polygon) + "]"
    return _ROI_LINE.sub(rf"\g<1> '{rendered}'", text, count=1)


def save(auto_exposure_gain, exposure, gain, roi_polygon):
    """Write the chosen values into cfr_zed2i.yaml, preserving every comment.

    A regex edit, not a YAML round-trip, for the same reason
    apply_vehicle_patch.py edits vehicle.yaml as text: this file's comments
    (why 60 fps needs HD720, why the ROI cut is where it is) are the point.
    """
    text = ZED_CONFIG.read_text(encoding="utf-8")
    text = _set_exposure(text, auto_exposure_gain, exposure, gain)
    if roi_polygon:
        text = _set_roi(text, roi_polygon)
    ZED_CONFIG.write_text(text, encoding="utf-8")
    return {"saved": str(ZED_CONFIG)}
