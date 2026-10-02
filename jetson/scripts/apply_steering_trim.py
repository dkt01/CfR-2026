#!/usr/bin/env python3
"""Push vehicle.yaml's measured steering.center_offset onto the real car.

    ./scripts/apply_steering_trim.py

apply_vehicle_patch.py only records center_offset in vehicle.yaml - it does
not by itself make the car any straighter. The actual correction lives in
arduino_bridge.yaml's steering_trim parameter, applied once in
arduino_bridge_node.cpp to every DriveCommand's steering (RL drivers, the
characterization runner, teleop) on the real car's active driving path. This
script is the one step that copies the measured number from the record into
the thing that acts on it; run it right after apply_vehicle_patch.py.

Does NOT touch any rl/formula*/config.yaml: center_offset describes a real
linkage/servo defect being trimmed OUT of the hardware, not a plant shape the
simulator should reproduce - the simulator already assumes a car trimmed to
exactly this good.
"""

import argparse
import os
import re
import sys

import yaml

from apply_vehicle_patch import DEFAULT_VEHICLE  # noqa: E402

DEFAULT_ARDUINO_BRIDGE = os.path.join(
    os.path.dirname(DEFAULT_VEHICLE), "arduino_bridge.yaml"
)


class TrimError(RuntimeError):
    pass


def _set_steering_trim(text, value):
    pattern = re.compile(r"(?m)^(    steering_trim:)[ \t]*[-+0-9.eE]+(.*)$")
    match = pattern.search(text)
    if not match:
        raise TrimError(
            'no "steering_trim" parameter under arduino_bridge.ros__parameters '
            "- has arduino_bridge_node.cpp's steering_trim param been added?"
        )
    # Always a YAML float: the bridge declares a double, and rclcpp refuses
    # to set it from an int, so "0" (what %g makes of 0.0) kills the node.
    number = f"{float(value):.6g}"
    if not any(c in number for c in ".en"):
        number += ".0"
    replacement = f"{match.group(1)} {number}{match.group(2)}"
    return text[: match.start()] + replacement + text[match.end() :]


def apply_trim(vehicle_path, bridge_path, dry_run=False):
    vehicle = yaml.safe_load(open(vehicle_path, encoding="utf-8").read())
    center_offset = vehicle["steering"]["center_offset"]["value"]

    with open(bridge_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    updated = _set_steering_trim(text, center_offset)
    if not dry_run and updated != text:
        with open(bridge_path, "w", encoding="utf-8") as handle:
            handle.write(updated)
    return center_offset, updated != text


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--vehicle", default=DEFAULT_VEHICLE, help="path to vehicle.yaml"
    )
    parser.add_argument(
        "--arduino-bridge",
        default=DEFAULT_ARDUINO_BRIDGE,
        help="path to arduino_bridge.yaml",
    )
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args(argv)

    try:
        center_offset, changed = apply_trim(
            args.vehicle, args.arduino_bridge, args.dry_run
        )
    except (KeyError, TrimError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not changed:
        print(f"steering_trim already {center_offset:.6g}, nothing to do")
    elif args.dry_run:
        print(
            f"would set steering_trim to {center_offset:.6g} in {args.arduino_bridge}"
        )
    else:
        print(f"steering_trim set to {center_offset:.6g} in {args.arduino_bridge}")
        print(
            "Rebuild and sync to the car for this to take effect (arduino_bridge_node"
        )
        print("reads it as a launch parameter, not a runtime `ros2 param set`).")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
