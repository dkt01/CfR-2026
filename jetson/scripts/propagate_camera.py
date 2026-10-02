#!/usr/bin/env python3
"""Push vehicle.yaml's camera_mount into the copies that cannot read it.

    ./scripts/propagate_camera.py           # write
    ./scripts/propagate_camera.py --check   # exit 1 if any copy disagrees

The formula drivers, record_run.py and the Run Lab read vehicle.yaml through
camera_extrinsics.py at startup.  These cannot:

  * Gazebo's ZED (launch/sensors_world.py): the rgbd sensor pose IS the depth
    origin, so it gets the left-lens position and the mount attitude.  A twin
    whose camera sits elsewhere than the car's hands the drivers a different
    scan in sim than on the car.
  * formulaSubZero's physical_camera_offset_*: the camera_link point the real
    /zed/zed_node/pose reports, which SubZero shifts back to the car center.

The RL training configs' camera.x / camera.z are NOT touched: they describe
the virtual camera a policy was trained with, and the drivers shift the real
scan onto it (see perception.depth_to_scan's `origin`).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_extrinsics as ce  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SENSORS_WORLD = REPO / "jetson" / "cfr_arduino_bridge" / "launch" / "sensors_world.py"
SUBZERO = REPO / "rl" / "formulaSubZero" / "config.yaml"


def fmt(v):
    text = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def edits(mount):
    d = mount.depth_origin()
    pose = " ".join(fmt(v) for v in (*d, mount.roll, mount.pitch, mount.yaw))
    return [
        (
            SENSORS_WORLD,
            re.compile(
                r'(<sensor name="zed2i" type="rgbd_camera">\s*<pose>)([^<]*)(</pose>)'
            ),
            pose,
        ),
        (
            SUBZERO,
            re.compile(r"(\n  physical_camera_offset_x_m: )(\S+)()"),
            fmt(mount.x),
        ),
        (
            SUBZERO,
            re.compile(r"(\n  physical_camera_offset_y_m: )(\S+)()"),
            fmt(mount.y),
        ),
    ]


def same(a, b):
    try:
        return all(
            math.isclose(float(x), float(y), abs_tol=5e-4)
            for x, y in zip(a.split(), b.split(), strict=True)
        )
    except ValueError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--vehicle", default=str(ce.VEHICLE_YAML))
    args = ap.parse_args(argv)
    mount = ce.load_mount(args.vehicle)
    if mount is None:
        print(f"error: no camera_mount in {args.vehicle}", file=sys.stderr)
        return 1

    stale, texts = [], {}
    for path, pattern, want in edits(mount):
        text = texts.get(path) or path.read_text()
        match = pattern.search(text)
        if match is None:
            print(
                f"error: {path.relative_to(REPO)}: pattern {pattern.pattern!r} not found",
                file=sys.stderr,
            )
            return 1
        if not same(match.group(2), want):
            stale.append(
                f"{path.relative_to(REPO)}: {match.group(2).strip()} -> {want}"
            )
            text = text[: match.start(2)] + want + text[match.end(2) :]
        texts[path] = text

    if not stale:
        print("camera mount: every copy matches vehicle.yaml")
        return 0
    for line in stale:
        print(("  stale    " if args.check else "  updated  ") + line)
    if args.check:
        print("run jetson/scripts/propagate_camera.py", file=sys.stderr)
        return 1
    for path, text in texts.items():
        path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
