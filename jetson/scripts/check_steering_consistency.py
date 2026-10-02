#!/usr/bin/env python3
"""Compare every place the car's steering authority is written down.

    ./scripts/check_steering_consistency.py

There are four of them, and nothing made them agree:

    vehicle.yaml  steering.max_angle_left/right   <- generate_vehicle_model.py
                                                     builds the SDF from THIS
    vehicle.yaml  steering.effective_angle_table  <- what the car measured
    arduino_bridge.yaml  steering_*_points        <- what sim_vehicle_node
                                                     turns a command into
    worlds/*.sdf  <steering_limit>, joint limits  <- what Gazebo will allow

The first is tagged `guess` at 0.40 rad and the second reaches 0.512 rad left,
so the generated world clamps a command the runtime model believes it can
make.  Measured on the Gazebo car (scripts are in rl/formulaOne):

    cmd 0.80 -> 0.90 m radius
    cmd 1.00 -> 0.90 m radius   <- identical: the top 22% of left steering
                                   command does nothing at all

Everything else understeers a uniform ~1.10x against the kinematic model,
flat in both directions and across 1.5-3.0 m/s, which is a steering gain of
about 0.91 rather than an understeer_gradient effect (that would grow with
v^2).

Exits non-zero when the four disagree, so this can gate a build.  It reports
rather than fixes: which number is right is a measurement question, and
vehicle.yaml is explicit that `max_angle_*` is a guess and the table's end
points are extrapolated.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "cfr_arduino_bridge"
REPO_ROOT = Path(__file__).resolve().parents[2]
# formulaOne/config_lowdrift.yaml is deliberately excluded: it is a variant
# with its own steering_slew, not a copy that should track the others.
RL_CONFIGS = {
    "formulaOne": REPO_ROOT / "rl/formulaOne/config.yaml",
    "formulaTwo": REPO_ROOT / "rl/formulaTwo/config.yaml",
    "formulaThree": REPO_ROOT / "rl/formulaThree/config.yaml",
    "formulaSubZero": REPO_ROOT / "rl/formulaSubZero/config.yaml",
}
TOLERANCE = 1e-3
findings: list[str] = []


def note(ok: bool, headline: str, detail: str = "") -> None:
    print(f"[{'  ok  ' if ok else ' DIFF '}] {headline}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")
    if not ok:
        findings.append(headline)


def main() -> int:
    vehicle = yaml.safe_load((ROOT / "config/vehicle.yaml").read_text())
    bridge = yaml.safe_load((ROOT / "config/arduino_bridge.yaml").read_text())
    steering = vehicle["steering"]
    wheelbase = vehicle["geometry"]["wheelbase"]["value"]

    raw_table = steering["effective_angle_table"]["rows"]
    center_offset = steering["center_offset"]["value"]
    # vehicle.yaml's table is in raw servo-command space (measured with
    # steering_trim 0).  Everything downstream consumes DriveCommand.steering,
    # which arduino_bridge shifts by steering_trim, so their copies are the
    # raw table shifted by center_offset (see propagate_plant.py).
    table = [[round(c - center_offset, 6), a] for c, a in raw_table]
    table_left = max(angle for _, angle in raw_table)
    table_right = abs(min(angle for _, angle in raw_table))
    max_left = steering["max_angle_left"]["value"]
    max_right = steering["max_angle_right"]["value"]
    provenance = (
        steering["max_angle_left"]["provenance"],
        steering["effective_angle_table"]["provenance"],
    )

    print("\n--- vehicle.yaml, internally")
    note(
        abs(max_left - table_left) < TOLERANCE
        and abs(max_right - table_right) < TOLERANCE,
        "max_angle_* agrees with the effective_angle_table",
        f"max_angle_left/right   {max_left} / {max_right}   [{provenance[0]}]\n"
        f"table reaches          {table_left} / {table_right}   [{provenance[1]}]\n"
        "generate_vehicle_model.py builds the world's steering_limit from\n"
        "max_angle_*, so the SIMULATED car is held to the first pair while\n"
        "sim_vehicle_node commands yaw rates from the second.",
    )
    note(
        abs(max_left - max_right) < TOLERANCE or True,  # reported, not failed
        "left and right lock, as the table measures them",
        f"left {table_left} rad, right {table_right} rad "
        f"-- {100 * (table_left / table_right - 1):.0f}% asymmetric, so a controller\n"
        "that assumes a symmetric limit is wrong by that much on one side.",
    )

    print("\n--- vehicle.yaml against arduino_bridge.yaml")
    center_offset = steering["center_offset"]["value"]
    steering_trim = bridge["arduino_bridge"]["ros__parameters"].get("steering_trim")
    note(
        steering_trim is not None and abs(center_offset - steering_trim) < TOLERANCE,
        "arduino_bridge steering_trim matches vehicle.yaml's center_offset",
        f"vehicle.yaml steering.center_offset {center_offset}\n"
        f"arduino_bridge.yaml arduino_bridge.steering_trim {steering_trim}\n"
        "apply_steering_trim.py keeps these in step; run it after "
        "apply_vehicle_patch.py whenever center_offset changes."
        if steering_trim is not None
        else "arduino_bridge.yaml has no steering_trim parameter - has "
        "arduino_bridge_node.cpp's steering_trim param been added?",
    )
    note(
        isinstance(steering_trim, float),
        "arduino_bridge steering_trim is written as a float",
        f"arduino_bridge.yaml has steering_trim: {steering_trim!r}. The node declares a\n"
        "double and rclcpp will not set it from an int, so the bridge dies at launch.\n"
        "Write it with a decimal point (0.0, not 0).",
    )

    sim = bridge["sim_vehicle"]["ros__parameters"]
    pts = list(zip(sim["steering_command_points"], sim["steering_angle_points"]))
    note(
        all(
            abs(a - b) < TOLERANCE and abs(c - d) < TOLERANCE
            for (a, c), (b, d) in zip(table, pts)
        ),
        "sim_vehicle steering table matches vehicle.yaml",
        f"vehicle.yaml {table}\narduino_bridge {pts}",
    )
    c2d_params = bridge["cmd_vel_to_drive"]["ros__parameters"]
    c2d_table = list(
        zip(
            c2d_params.get("steering_command_points", []),
            c2d_params.get("steering_angle_points", []),
        )
    )
    if c2d_table:
        note(
            all(
                abs(a - b) < TOLERANCE and abs(c - d) < TOLERANCE
                for (a, c), (b, d) in zip(table, c2d_table)
            ),
            "cmd_vel_to_drive inverts the same table the car was measured on",
            "A desired angle becomes the command that actually produces it.\n"
            "Dividing by the symmetric max_steering_angle instead over-steers\n"
            f"LEFT by {100 * (table_left / (2 * c2d_params['max_steering_angle']) * 2 - 1):.0f}% "
            "at every angle, which is what it used to do.",
        )
    else:
        note(
            False,
            "cmd_vel_to_drive has no steering table, so it scales symmetrically",
            f"max_steering_angle {c2d_params['max_steering_angle']} rad against measured "
            f"{table_left} left / {table_right} right: over-steers left by "
            f"{100 * (table_left / c2d_params['max_steering_angle'] - 1):.0f}%.",
        )

    print("\n--- vehicle.yaml against the RL driver configs")
    # steering.effective_angle_table has a clean home to compare against in
    # each driver's plant: block. yaw_response_tau/understeer_gradient/
    # tire_scrub do not yet have a matching per-value home in vehicle.yaml
    # (inertia: has no yaw_response_tau field, lateral.understeer_gradient is
    # a single scalar while the skidpad analyzer produces per-side values) --
    # so those are cross-checked against EACH OTHER below instead, which
    # needs no new vehicle.yaml schema and still catches the three drivers
    # silently drifting apart, the gap nothing else here covers.
    rl_plants = {}
    for driver, path in RL_CONFIGS.items():
        if not path.is_file():
            continue
        plant = (yaml.safe_load(path.read_text()) or {}).get("plant") or {}
        rl_plants[driver] = plant
        rl_table = list(
            zip(
                plant.get("steering_command_points", []),
                plant.get("steering_angle_points", []),
            )
        )
        note(
            bool(rl_table)
            and all(
                abs(a - b) < TOLERANCE and abs(c - d) < TOLERANCE
                for (a, c), (b, d) in zip(table, rl_table)
            ),
            f"rl/{driver}/config.yaml steering table matches vehicle.yaml",
            f"vehicle.yaml {table}\nrl/{driver} {rl_table}",
        )

    for key in ("wheelbase", "yaw_response_tau", "understeer_gradient", "tire_scrub"):
        values = {
            driver: plant[key] for driver, plant in rl_plants.items() if key in plant
        }
        if len(values) < 2:
            continue
        spread = max(values.values()) - min(values.values())
        note(
            spread < TOLERANCE,
            f"rl/formula*/config.yaml agree on plant.{key}",
            "\n".join(f"{driver}: {value}" for driver, value in values.items())
            + "\nThese describe the same physical car; a driver-specific value here "
            "is either an intentional divergence that belongs in a comment, or a "
            "propagation that missed one file.",
        )

    print("\n--- the generated worlds")
    for world in sorted((ROOT / "worlds").glob("*.sdf")):
        text = world.read_text()
        limit = re.search(r"<steering_limit>([\d.]+)</steering_limit>", text)
        joint = re.search(
            r'front_left_steering_joint".*?<lower>(-?[\d.]+)</lower><upper>(-?[\d.]+)</upper>',
            text,
        )
        if not limit:
            continue
        plugin_limit = float(limit.group(1))
        note(
            plugin_limit >= table_left - TOLERANCE,
            f"{world.name}: steering_limit admits the measured left lock",
            f"steering_limit {plugin_limit} rad vs table's {table_left} rad left"
            + (
                ""
                if plugin_limit >= table_left - TOLERANCE
                else f"\nCommands past {100 * plugin_limit / table_left:.0f}% of full left are "
                "silently truncated: measured, 0.80 and 1.00 gave the\nsame radius in Gazebo."
            ),
        )
        if joint:
            upper = float(joint.group(2))
            note(
                upper >= table_left - TOLERANCE,
                f"{world.name}: steering joint limit admits the measured lock",
                f"joint upper {upper} rad vs table's {table_left} rad",
            )

    print("\n--- derived quantities")
    stated = vehicle["lateral"]["min_turn_radius"]["value"]
    from_guess = wheelbase / math.tan(max_left)
    from_table = wheelbase / math.tan(table_left)
    note(
        abs(stated - from_table) < 0.05,
        "min_turn_radius follows from the measured lock",
        f"stated {stated} m; from max_angle {from_guess:.3f} m; "
        f"from the table {from_table:.3f} m."
        + (
            ""
            if abs(stated - from_table) < 0.05
            else "\nIt tracks max_angle rather than the measured table."
        ),
    )

    print()
    if findings:
        print(f"{len(findings)} disagreement(s):")
        for f in findings:
            print(f"  - {f}")
        print(
            "\nvehicle.yaml calls max_angle_* a `guess` and the table's end points\n"
            "extrapolated, so neither is authoritative. Measuring the real car's\n"
            "lock settles it; until then the simulated car follows max_angle_*."
        )
        return 1
    print("every record of the steering authority agrees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
