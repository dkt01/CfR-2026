#!/usr/bin/env python3
"""Push vehicle.yaml's measured steering table into the RL driver configs.

    ./scripts/propagate_plant.py

apply_vehicle_patch.py only touches config/vehicle.yaml -- the record of what
the real car measured.  Nothing propagates that into the numpy simulators
rl/formulaOne, formulaTwo and formulaThree train against (and formulaSubZero
steers by), so a measured table
can sit correct in vehicle.yaml while every driver still trains on the old
one.  This script closes that one gap: vehicle.yaml's
steering.effective_angle_table is the only plant value with a clean,
unambiguous home on both sides today (see check_steering_consistency.py's own
note on why yaw_response_tau, understeer_gradient and tire_scrub are cross-
checked against each other instead of against vehicle.yaml -- they do not yet
have a matching per-value field there).

rl/formulaOne/config_lowdrift.yaml is deliberately excluded: it is a variant
with its own steering_slew, not a copy that should track the others.

The table in vehicle.yaml is in RAW servo-command space: the straight and
figure-8 runs were driven with steering_trim 0, so its zero crossing sits at
steering.center_offset rather than at 0.  arduino_bridge adds that trim to
every DriveCommand, so every consumer of DriveCommand.steering (the Gazebo
sim_vehicle, cmd_vel_to_drive's inverse, the RL plants) must see the table
shifted to post-trim space:  command_here = command_raw - center_offset.  That
is what this writes.  (The bridge clamps raw to +/-1, so at command +1 the car
reaches the table at raw 1 + center_offset: slightly less than full left lock.)

Run after apply_vehicle_patch.py and apply_steering_trim.py, then
check_steering_consistency.py to confirm every copy agrees.
"""

import argparse
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_vehicle_patch import DEFAULT_VEHICLE  # noqa: E402


def _format_scalar(value):
    """Render a table entry the way these files already write one.

    Not apply_vehicle_patch.py's `_format_scalar`: its plain `.6g` turns an
    exact `-1.0` into `-1`, which would rewrite every whole-number endpoint in
    the RL configs for no numeric reason and bury the real diff in noise.
    """
    text = f"{value:.6g}"
    if not any(c in text for c in ".eE"):
        text += ".0"
    return text


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRIDGE_CONFIG = os.path.join(
    REPO_ROOT, "jetson", "cfr_arduino_bridge", "config", "arduino_bridge.yaml"
)
RL_CONFIGS = [
    os.path.join(REPO_ROOT, "rl", "formulaOne", "config.yaml"),
    os.path.join(REPO_ROOT, "rl", "formulaTwo", "config.yaml"),
    os.path.join(REPO_ROOT, "rl", "formulaThree", "config.yaml"),
    os.path.join(REPO_ROOT, "rl", "formulaSubZero", "config.yaml"),
]


class PropagationError(RuntimeError):
    pass


def _replace_list(text, key, values, indent="  ", replace_all=False):
    """Replace a `key:` scalar list, in whichever style this file already uses.

    formulaOne/formulaTwo write it flow-style (`key: [a, b, c]`), formulaThree
    block-style (`key:` then one `- value` per line at the same indent). Kept
    as whichever the file already had, rather than normalizing, so a diff
    shows only the numbers that changed.
    """
    rendered = ", ".join(_format_scalar(value) for value in values)
    flow_pattern = re.compile(
        rf"(?m)^{re.escape(indent)}{re.escape(key)}:[ \t]*\[[^\]]*\](.*)$"
    )
    if replace_all and flow_pattern.search(text):
        return flow_pattern.sub(lambda m: f"{indent}{key}: [{rendered}]{m.group(1)}", text)
    match = flow_pattern.search(text)
    if match:
        replacement = f"{indent}{key}: [{rendered}]{match.group(1)}"
        return text[: match.start()] + replacement + text[match.end() :]

    block_pattern = re.compile(
        rf"(?m)^{re.escape(indent)}{re.escape(key)}:[ \t]*\n"
        rf"(?:{re.escape(indent)}- .*\n)+"
    )
    match = block_pattern.search(text)
    if not match:
        raise PropagationError(f'no "{key}" list (flow or block) found')
    block = f"{indent}{key}:\n" + "".join(
        f"{indent}- {_format_scalar(value)}\n" for value in values
    )
    return text[: match.start()] + block + text[match.end() :]


def propagate(vehicle_path, targets, dry_run=False):
    vehicle = yaml.safe_load(open(vehicle_path, encoding="utf-8").read())
    table = vehicle["steering"]["effective_angle_table"]["rows"]
    center_offset = vehicle["steering"]["center_offset"]["value"]
    commands = [round(row[0] - center_offset, 6) for row in table]
    angles = [row[1] for row in table]

    results = []
    for path in targets:
        if not os.path.isfile(path):
            results.append((path, None, "file not found"))
            continue
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        try:
            # arduino_bridge.yaml carries the table twice (cmd_vel_to_drive and
            # sim_vehicle), four-space indented under ros__parameters.
            bridge = os.path.abspath(path) == os.path.abspath(BRIDGE_CONFIG)
            indent = "    " if bridge else "  "
            updated = _replace_list(
                text, "steering_command_points", commands, indent, bridge
            )
            updated = _replace_list(
                updated, "steering_angle_points", angles, indent, bridge
            )
        except PropagationError as error:
            results.append((path, None, str(error)))
            continue
        changed = updated != text
        if not dry_run and changed:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(updated)
        results.append((path, changed, None))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--vehicle", default=DEFAULT_VEHICLE, help="path to vehicle.yaml"
    )
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args(argv)

    results = propagate(
        args.vehicle, [BRIDGE_CONFIG] + RL_CONFIGS, dry_run=args.dry_run
    )
    ok = True
    for path, changed, error in results:
        rel = os.path.relpath(path, REPO_ROOT)
        if error:
            print(f"  skipped  {rel} ({error})")
            ok = False
        elif changed:
            print(f"  {'would update' if args.dry_run else 'updated'}  {rel}")
        else:
            print(f"  unchanged  {rel} (already matches vehicle.yaml)")
    if ok:
        print("\nRun check_steering_consistency.py to confirm every copy agrees.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
