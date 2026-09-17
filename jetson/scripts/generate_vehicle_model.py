#!/usr/bin/env python3
"""Write the <model name="slash"> block of every Gazebo world from vehicle.yaml.

    ./scripts/generate_vehicle_model.py            # rewrite every world in place
    ./scripts/generate_vehicle_model.py --check    # fail if any world is out of date
    ./scripts/generate_vehicle_model.py --world worlds/obstacle_course.sdf  # just one

Before this existed the vehicle model was hand-maintained, and it had drifted:
the wheel radius was 0.055 m while the drivetrain tire diameter was 0.1143 m
(0.05715), a silent 3.6% disagreement between the simulator's odometry and the
car's.  The wheelbase was a bare literal in five places and the steering limit in
five more.  Generating the block makes that class of divergence impossible - a
number lives in vehicle.yaml or it does not exist.  obstacle_course.sdf carried
its own hand-copied version of the same drift (mass 3.5 kg, wheel radius 0.055 m)
until it was folded into this same generator, keyed off WORLD_SPAWN_POSES.

--check runs in CI, so a vehicle.yaml edit that nobody regenerated fails the
build instead of quietly leaving the twin describing a different car.

This deliberately generates only the vehicle.  generate_speed_course.py owns the
bale layout in the same file, anchored on the same `<model name="slash">` line,
and the two are careful not to overlap.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "cfr_arduino_bridge"
VEHICLE_FILE = PACKAGE_DIR / "config" / "vehicle.yaml"

# Every world the vehicle gets spawned into, and where - a course layout fact,
# not a vehicle one, so it lives here rather than in vehicle.yaml.  (x, y, yaw).
WORLD_SPAWN_POSES = {
    "speed_course.sdf": (20.15, 4.76, 3.14),
    "obstacle_course.sdf": (-0.7, 0.0, 0.0),
}

GRAVITY = 9.81

# The block this tool owns, start marker to end marker inclusive.
# Settling margin above the ground at spawn, shared with teleport_api.py.
SPAWN_HEIGHT_M = 0.02

BEGIN = (
    "    <!-- BEGIN generated vehicle: edit config/vehicle.yaml, not this block. -->"
)
END = "    <!-- END generated vehicle -->"


def value_of(vehicle, dotted, default=None):
    section, _, key = dotted.partition(".")
    entry = (vehicle.get(section) or {}).get(key)
    if entry is None:
        if default is None:
            raise KeyError(f"vehicle.yaml is missing {dotted}")
        return default
    return entry["value"] if isinstance(entry, dict) else entry


def provenance_of(vehicle, dotted):
    section, _, key = dotted.partition(".")
    entry = (vehicle.get(section) or {}).get(key)
    return entry.get("provenance", "unknown") if isinstance(entry, dict) else "unknown"


def build_model(vehicle, spawn_pose):
    wheelbase = float(value_of(vehicle, "geometry.wheelbase"))
    track_front = float(value_of(vehicle, "geometry.track_front"))
    track_rear = float(value_of(vehicle, "geometry.track_rear"))
    body_l = float(value_of(vehicle, "geometry.chassis_length"))
    body_w = float(value_of(vehicle, "geometry.chassis_width"))
    body_h = float(value_of(vehicle, "geometry.chassis_height"))

    total_mass = float(value_of(vehicle, "mass.total"))
    wheel_mass = float(value_of(vehicle, "mass.wheel_mass"))
    knuckle_mass = float(value_of(vehicle, "mass.steering_knuckle_mass"))
    upright_mass = float(value_of(vehicle, "mass.upright_mass"))
    cg_x = float(value_of(vehicle, "mass.cg_x"))
    cg_y = float(value_of(vehicle, "mass.cg_y"))
    cg_z = float(value_of(vehicle, "mass.cg_z"))

    ixx = float(value_of(vehicle, "inertia.ixx"))
    iyy = float(value_of(vehicle, "inertia.iyy"))
    izz = float(value_of(vehicle, "inertia.izz"))

    radius = float(value_of(vehicle, "tire.diameter")) / 2.0
    width = float(value_of(vehicle, "tire.width"))
    mu = float(value_of(vehicle, "lateral.mu_lateral"))

    left = abs(float(value_of(vehicle, "steering.max_angle_left")))
    right = abs(float(value_of(vehicle, "steering.max_angle_right")))
    # The joint limit has to admit the larger lock or the wider side is clipped;
    # the plugin's steering_limit is what scales a command, so it takes the
    # smaller, which is the angle the car can actually reach on both sides.
    joint_limit = max(left, right)
    steering_limit = min(left, right)

    spring_rate = float(value_of(vehicle, "suspension.spring_rate"))
    damping = float(value_of(vehicle, "suspension.damping"))
    travel_bump = float(value_of(vehicle, "suspension.travel_bump"))
    travel_droop = float(value_of(vehicle, "suspension.travel_droop"))

    # The chassis link carries what is left once the wheels, uprights and
    # knuckles are accounted for, so the model's TOTAL mass is the measured
    # one rather than the measured one plus twelve unsprung parts.
    unsprung_mass = 4.0 * wheel_mass + 4.0 * upright_mass + 2.0 * knuckle_mass
    chassis_mass = total_mass - unsprung_mass
    if chassis_mass <= 0.0:
        raise ValueError(
            f"mass.total ({total_mass} kg) is not greater than the wheels, uprights "
            f"and knuckles ({unsprung_mass} kg); check mass.total in vehicle.yaml"
        )

    half_front = track_front / 2.0
    half_rear = track_rear / 2.0
    axle = wheelbase / 2.0
    # Wheel centres sit at one radius, which puts the model origin at ground
    # level and makes every z in vehicle.yaml a height above the ground.
    wheel_inertia = 0.5 * wheel_mass * radius * radius

    # geometry.ride_height was measured "shocks free" (Session A2), i.e. it IS
    # the static-sag position - so that position is the suspension joint's
    # zero, and spring_reference is offset from it by exactly the sag needed
    # to hold the corner's static weight (mg = k * spring_reference), the same
    # relationship a real spring satisfies at rest.  Front/rear split comes
    # from the same axle weight fractions A1 measured (see mass.cg_x).
    front_weight_fraction = 0.5 + cg_x / wheelbase
    rear_weight_fraction = 1.0 - front_weight_fraction
    front_corner_load_n = total_mass * front_weight_fraction / 2.0 * GRAVITY
    rear_corner_load_n = total_mass * rear_weight_fraction / 2.0 * GRAVITY
    front_spring_reference = front_corner_load_n / spring_rate
    rear_spring_reference = rear_corner_load_n / spring_rate

    def wheel(name, x, y):
        return (
            f'      <link name="{name}_wheel"><pose>{x:.4f} {y:.4f} {radius:.5f} -1.5708 0 0</pose>'
            f"<inertial><mass>{wheel_mass:.4f}</mass><inertia>"
            f"<ixx>{wheel_inertia / 2:.6f}</ixx><iyy>{wheel_inertia / 2:.6f}</iyy>"
            f"<izz>{wheel_inertia:.6f}</izz></inertia></inertial>"
            f'<collision name="collision"><geometry><cylinder><radius>{radius:.5f}</radius>'
            f"<length>{width:.4f}</length></cylinder></geometry><surface><friction>"
            f"<ode><mu>{mu:.3f}</mu><mu2>{mu:.3f}</mu2><fdir1>0 0 1</fdir1></ode>"
            f"<bullet><friction>{mu:.3f}</friction><friction2>{mu:.3f}</friction2>"
            f"<rolling_friction>0.001</rolling_friction></bullet></friction></surface></collision>"
            f'<visual name="visual"><geometry><cylinder><radius>{radius:.5f}</radius>'
            f"<length>{width:.4f}</length></cylinder></geometry>"
            f"<material><diffuse>0.04 0.04 0.04 1</diffuse></material></visual></link>"
        )

    def knuckle(name, x, y):
        return (
            f'      <link name="{name}_steering"><pose>{x:.4f} {y:.4f} {radius:.5f} 0 0 0</pose>'
            f"<inertial><mass>{knuckle_mass:.4f}</mass><inertia>"
            f"<ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial></link>"
        )

    def upright(name, x, y):
        return (
            f'      <link name="{name}_upright"><pose>{x:.4f} {y:.4f} {radius:.5f} 0 0 0</pose>'
            f"<inertial><mass>{upright_mass:.4f}</mass><inertia>"
            f"<ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial></link>"
        )

    def suspension_joint(name, spring_reference):
        # Parent is always chassis: the wheel end of the joint is the upright,
        # front or rear, so this is the one joint every corner has whether or
        # not it steers.  +z is compression (toward the chassis), matching the
        # bump/droop split above.
        return (
            f'      <joint name="{name}_suspension_joint" type="prismatic"><parent>chassis</parent>'
            f"<child>{name}_upright</child><axis><xyz>0 0 1</xyz>"
            f"<dynamics><damping>{damping:.3f}</damping>"
            f"<spring_stiffness>{spring_rate:.3f}</spring_stiffness>"
            f"<spring_reference>{spring_reference:.5f}</spring_reference></dynamics>"
            f"<limit><lower>{-travel_droop:.5f}</lower><upper>{travel_bump:.5f}</upper>"
            f"<effort>1000000</effort></limit></axis></joint>"
        )

    tags = ", ".join(
        f"{name} {provenance_of(vehicle, name)}"
        for name in (
            "mass.total",
            "mass.cg_z",
            "inertia.izz",
            "tire.diameter",
            "steering.max_angle_left",
            "lateral.mu_lateral",
            "suspension.spring_rate",
            "suspension.damping",
        )
    )

    spawn_x, spawn_y, spawn_yaw = spawn_pose
    lines = [
        BEGIN,
        f"    <!-- Provenance: {tags}. -->",
        "    <!-- Anything tagged `guess` is a placeholder; see docs/characterization.md. -->",
        "    <!-- Suspension is a prismatic joint per corner with SDF joint",
        "         dynamics (spring_stiffness/spring_reference/damping). Confirm the",
        "         loaded physics engine actually implements joint springs: some",
        "         gz-sim physics plugins only honour damping, not spring_stiffness,",
        "         in which case a corner sags to its limit instead of settling. -->",
        '    <model name="slash">',
        # Wheel centres sit at exactly one radius, so model-frame z = 0 IS ground
        # level and this is only a settling margin.  The hand-written model used
        # 0.12, dropping the car 12 cm onto its wheels on every spawn - and
        # teleport_api.py copied the number, so it did it again on every teleport.
        f"      <pose>{spawn_x:.4f} {spawn_y:.4f} {SPAWN_HEIGHT_M:.4f} 0 0 {spawn_yaw:.4f}</pose>",
        '      <link name="chassis">',
        # The inertial pose is the fix that matters most here: without it the
        # centre of mass sits at the link origin, which is GROUND level.
        f"        <inertial><pose>{cg_x:.4f} {cg_y:.4f} {cg_z:.4f} 0 0 0</pose>"
        f"<mass>{chassis_mass:.4f}</mass><inertia>"
        f"<ixx>{ixx:.5f}</ixx><iyy>{iyy:.5f}</iyy><izz>{izz:.5f}</izz></inertia></inertial>",
        f'        <collision name="collision"><pose>0 0 {body_h / 2 + radius * 0.6:.4f} 0 0 0</pose>'
        f"<geometry><box><size>{body_l:.4f} {body_w:.4f} {body_h:.4f}</size></box></geometry></collision>",
        f'        <visual name="body"><pose>0 0 {body_h / 2 + radius * 0.6:.4f} 0 0 0</pose>'
        f"<geometry><box><size>{body_l:.4f} {body_w:.4f} {body_h:.4f}</size></box></geometry>"
        f"<material><diffuse>0.85 0.08 0.04 1</diffuse></material></visual>",
        '        <visual name="forward_direction_marker"><pose>0.025 0 0.16 0 1.5708 0</pose>'
        "<geometry><cone><radius>0.12</radius><length>0.35</length></cone></geometry>"
        "<material><diffuse>0.05 1 0.08 1</diffuse><emissive>0.02 0.45 0.04 1</emissive></material></visual>",
        # Cosmetic only, but it is how anyone watching the GUI tells which way
        # the car points and where its camera looks.
        '        <visual name="zed2i_mount"><pose>0.25 0 0.155 0 0 0</pose>'
        "<geometry><box><size>0.06 0.12 0.05</size></box></geometry>"
        "<material><diffuse>0.20 0.23 0.26 1</diffuse></material></visual>",
        '        <visual name="zed2i_housing"><pose>0.295 0 0.20 0 0 0</pose>'
        "<geometry><box><size>0.03025 0.17525 0.04310</size></box></geometry>"
        "<material><diffuse>0.22 0.31 0.38 1</diffuse><emissive>0.01 0.03 0.05 1</emissive>"
        "<specular>0.45 0.45 0.45 1</specular></material></visual>",
        '        <visual name="zed2i_front_panel"><pose>0.311 0 0.20 0 0 0</pose>'
        "<geometry><box><size>0.002 0.168 0.035</size></box></geometry>"
        "<material><diffuse>0.03 0.17 0.25 1</diffuse><emissive>0.01 0.05 0.08 1</emissive></material></visual>",
        '        <visual name="zed2i_left_lens"><pose>0.315 0.06 0.20 0 1.5708 0</pose>'
        "<geometry><cylinder><radius>0.015</radius><length>0.006</length></cylinder></geometry>"
        "<material><diffuse>0.08 0.55 0.85 1</diffuse><emissive>0.02 0.16 0.28 1</emissive>"
        "<specular>0.7 0.7 0.7 1</specular></material></visual>",
        '        <visual name="zed2i_right_lens"><pose>0.315 -0.06 0.20 0 1.5708 0</pose>'
        "<geometry><cylinder><radius>0.015</radius><length>0.006</length></cylinder></geometry>"
        "<material><diffuse>0.08 0.55 0.85 1</diffuse><emissive>0.02 0.16 0.28 1</emissive>"
        "<specular>0.7 0.7 0.7 1</specular></material></visual>",
        # simulation.launch.py swaps this marker for the rendered ZED under
        # `sensors:=true`, and raises if it is missing -- so the generated
        # block has to carry it, not just the hand written part of the world.
        "          <!-- cfr:sensors-camera -->",
        "      </link>",
        # Suspension sits between chassis and upright at all four corners;
        # steering (front only) and wheel spin sit below the upright, so
        # vertical travel carries the whole corner assembly with it rather
        # than fighting either of the other two joints.
        upright("front_left", axle, half_front),
        upright("front_right", axle, -half_front),
        upright("rear_left", -axle, half_rear),
        upright("rear_right", -axle, -half_rear),
        knuckle("front_left", axle, half_front),
        knuckle("front_right", axle, -half_front),
        wheel("front_left", axle, half_front),
        wheel("front_right", axle, -half_front),
        wheel("rear_left", -axle, half_rear),
        wheel("rear_right", -axle, -half_rear),
        suspension_joint("front_left", front_spring_reference),
        suspension_joint("front_right", front_spring_reference),
        suspension_joint("rear_left", rear_spring_reference),
        suspension_joint("rear_right", rear_spring_reference),
        f'      <joint name="front_left_steering_joint" type="revolute"><parent>front_left_upright</parent>'
        f"<child>front_left_steering</child><axis><xyz>0 0 1</xyz><limit>"
        f"<lower>{-joint_limit:.5f}</lower><upper>{joint_limit:.5f}</upper>"
        f"<effort>1000000</effort></limit></axis></joint>",
        f'      <joint name="front_right_steering_joint" type="revolute"><parent>front_right_upright</parent>'
        f"<child>front_right_steering</child><axis><xyz>0 0 1</xyz><limit>"
        f"<lower>{-joint_limit:.5f}</lower><upper>{joint_limit:.5f}</upper>"
        f"<effort>1000000</effort></limit></axis></joint>",
    ]
    for name, parent in (
        ("front_left", "front_left_steering"),
        ("front_right", "front_right_steering"),
        ("rear_left", "rear_left_upright"),
        ("rear_right", "rear_right_upright"),
    ):
        lines.append(
            f'      <joint name="{name}_wheel_joint" type="revolute"><parent>{parent}</parent>'
            f"<child>{name}_wheel</child><axis><xyz>0 0 1</xyz><limit><lower>-1000000</lower>"
            f"<upper>1000000</upper><effort>1000000</effort></limit></axis></joint>"
        )

    lines += [
        '      <plugin filename="gz-sim-ackermann-steering-system" '
        'name="gz::sim::systems::AckermannSteering">',
        "        <topic>/sim/cmd_vel</topic><odom_topic>/model/slash/odometry</odom_topic>",
        "        <left_joint>front_left_wheel_joint</left_joint>"
        "<left_joint>rear_left_wheel_joint</left_joint>",
        "        <right_joint>front_right_wheel_joint</right_joint>"
        "<right_joint>rear_right_wheel_joint</right_joint>",
        "        <left_steering_joint>front_left_steering_joint</left_steering_joint>"
        "<right_steering_joint>front_right_steering_joint</right_steering_joint>",
        f"        <wheel_base>{wheelbase:.4f}</wheel_base>"
        f"<wheel_separation>{(track_front + track_rear) / 2:.4f}</wheel_separation>"
        f"<wheel_radius>{radius:.5f}</wheel_radius>"
        f"<steering_limit>{steering_limit:.5f}</steering_limit>",
        "      </plugin>",
        "      <!-- Ground truth pose, bridged to /zed/zed_node/pose.  It stands in for",
        "           the ZED's map frame topic, which the SDK corrects on loop closure;",
        "           the ackermann odometry above drifts and is never corrected, exactly",
        "           as the real camera's ~/odom is not.  lap_counter reads this one. -->",
        '      <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">',
        "        <publish_model_pose>true</publish_model_pose><publish_link_pose>false</publish_link_pose>",
        "        <publish_collision_pose>false</publish_collision_pose><publish_visual_pose>false</publish_visual_pose>",
        "        <publish_nested_model_pose>false</publish_nested_model_pose><use_pose_vector_msg>false</use_pose_vector_msg>",
        "        <update_frequency>30</update_frequency>",
        "      </plugin>",
        "    </model>",
        END,
    ]
    return "\n".join(lines)


def splice(world_text, model_text, world_name):
    if BEGIN in world_text:
        pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
        updated, count = pattern.subn(lambda _: model_text, world_text)
    else:
        # First run: adopt the hand-written block that is there today.
        pattern = re.compile(r'    <model name="slash">.*?    </model>', re.DOTALL)
        updated, count = pattern.subn(lambda _: model_text, world_text)
    if count != 1:
        raise RuntimeError(
            "could not find exactly one vehicle block in the world file "
            f"(found {count}); has {world_name} been edited by hand?"
        )
    return updated


def regenerate(vehicle, world_path, check):
    spawn_pose = WORLD_SPAWN_POSES.get(world_path.name)
    if spawn_pose is None:
        raise KeyError(
            f"{world_path.name} has no entry in WORLD_SPAWN_POSES; add its spawn "
            "(x, y, yaw) there"
        )
    world_text = world_path.read_text(encoding="utf-8")
    updated = splice(world_text, build_model(vehicle, spawn_pose), world_path.name)

    if check:
        if updated != world_text:
            print(
                f"{world_path} is out of date with vehicle.yaml.\n"
                f"Run scripts/generate_vehicle_model.py and commit the result.",
                file=sys.stderr,
            )
            return False
        print(f"{world_path.name} is up to date with vehicle.yaml")
        return True

    world_path.write_text(updated, encoding="utf-8")
    print(f"wrote the vehicle model into {world_path}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--vehicle", type=Path, default=VEHICLE_FILE)
    parser.add_argument(
        "--world",
        type=Path,
        default=None,
        help="regenerate only this world file (default: every world in "
        "WORLD_SPAWN_POSES)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the world is out of date (used by CI)",
    )
    args = parser.parse_args(argv)

    with open(args.vehicle, "r", encoding="utf-8") as handle:
        vehicle = yaml.safe_load(handle)

    worlds = (
        [args.world]
        if args.world is not None
        else [PACKAGE_DIR / "worlds" / name for name in WORLD_SPAWN_POSES]
    )

    if args.check:
        # A list, not a generator, so every world gets checked (and printed)
        # even once one has already failed.
        results = [regenerate(vehicle, world, check=True) for world in worlds]
        return 0 if all(results) else 1

    for world in worlds:
        regenerate(vehicle, world, check=False)

    guesses = [
        f"{section}.{key}"
        for section, entries in vehicle.items()
        if isinstance(entries, dict)
        for key, entry in entries.items()
        if isinstance(entry, dict) and entry.get("provenance") == "guess"
    ]
    if guesses:
        print(
            f"\n{len(guesses)} value(s) are still tagged `guess`, so the twin is still "
            f"partly fiction:"
        )
        for name in guesses:
            print(f"  {name}")
        print(
            "\nSee docs/characterization.md for the experiment that measures each one."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
