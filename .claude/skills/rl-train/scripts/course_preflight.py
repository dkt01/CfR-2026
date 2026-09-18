#!/usr/bin/env python3
"""Check whether the RL stack can actually train on a given course.

Everything in rl/bale_follower was built against the speed course, and several
of its assumptions are silently course-specific: the bale parser looks for one
exact link name, the pose bridge subscribes to one hard-coded world topic, the
env unpauses the world named in config.yaml, and it reads the car's pose from
index 0 of the dynamic-pose array. On the obstacle course each of those is
wrong in a different way, and the failures look nothing like their causes -- a
frozen car, a pose timeout, or a policy steering around geometry it cannot see.

So this checks the repo as it stands rather than hard-coding a verdict. When
the obstacle-course support lands, these checks start passing on their own and
the gate opens without anyone editing this file.

    python3 course_preflight.py --course obstacle
    python3 course_preflight.py --course speed --repo /path/to/CfR-2026

Exit 0 = every blocking check passed (warnings may still print).
Exit 1 = at least one blocking check failed; the message says what to change.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Models that are scenery, the car itself, or already handled. Anything else
# carrying collision geometry is an obstacle the analytic ray-cast cannot see.
KNOWN_MODELS = {
    "ground",
    "course_bales",
    "slash",
    "start_signal_frame",
    "start_signal_arms",
}


def world_path(repo, course):
    return repo / "jetson/cfr_arduino_bridge/worlds" / f"{course}_course.sdf"


def read(path):
    return path.read_text(errors="replace") if path.exists() else ""


def world_name(sdf_text):
    match = re.search(r'<world name="([^"]+)"', sdf_text)
    return match.group(1) if match else None


def model_declarations(sdf_text):
    """(name, is_static) for every model, in declaration order."""
    models = []
    for match in re.finditer(r'<model name="([^"]+)">(.{0,200})', sdf_text, re.S):
        name, tail = match.group(1), match.group(2)
        models.append((name, "<static>true</static>" in tail))
    return models


def check_bale_link(results, sdf_text, sdf_file):
    """bale_geometry.parse_bales looks up course_bales/bales by exact name."""
    match = re.search(
        r'<model name="course_bales">.{0,80}?<link name="([^"]+)">', sdf_text, re.S
    )
    if match is None:
        results.append(
            (
                False,
                "bale geometry",
                f"{sdf_file.name} has no course_bales model; "
                "bale_geometry.parse_bales has nothing to parse",
            )
        )
        return
    link = match.group(1)
    if link != "bales":
        results.append(
            (
                False,
                "bale geometry",
                f"{sdf_file.name} names the bale link "
                f"'{link}', but bale_geometry.parse_bales looks up "
                "model[@name='course_bales']/link[@name='bales'] and raises "
                "ValueError otherwise -- make the parser accept either name, or "
                "rename the link in the world",
            )
        )
        return
    count = len(re.findall(r"bale_\d+_collision", sdf_text))
    results.append(
        (True, "bale geometry", f"course_bales/bales with {count} bale boxes")
    )


def check_pose_bridge(results, repo, world):
    launch = repo / "jetson/cfr_arduino_bridge/launch/training.launch.py"
    text = read(launch)
    topics = re.findall(r"/world/([a-z_0-9]+)/dynamic_pose/info", text)
    if world in topics:
        results.append(
            (True, "pose bridge", f"training.launch.py bridges {world}'s dynamic_pose")
        )
        return
    results.append(
        (
            False,
            "pose bridge",
            f"training.launch.py bridges dynamic_pose for "
            f"{topics or ['nothing']}, not {world}. The bridge subscribes to a gz "
            "topic that does not exist in this world, so the ROS topic appears but "
            "never publishes and the env dies on its odom timeout. Derive the topic "
            "from the world argument instead of hard-coding it",
        )
    )


def check_config_world(results, repo, world):
    config = repo / "rl/bale_follower/config.yaml"
    match = re.search(r"^\s*world_name:\s*(\S+)", read(config), re.M)
    configured = match.group(1) if match else None
    if configured == world:
        results.append((True, "config world_name", f"config.yaml targets {world}"))
        return
    results.append(
        (
            False,
            "config world_name",
            f"config.yaml env.world_name is "
            f"{configured!r}, not {world!r}. env._unpause_world calls "
            "/world/<world_name>/control, so the wrong name leaves the server "
            "paused: /clock never advances, cmd_vel_to_drive treats every command "
            "as stale, and the car sits at neutral looking like a dead policy",
        )
    )


def check_pose_index(results, sdf_text, sdf_file):
    """env._on_pose reads transforms[0] and assumes that is the car."""
    dynamic = [name for name, static in model_declarations(sdf_text) if not static]
    # Nested sub-models (wheels, uprights) are declared inside <model name="slash">,
    # so collapse to the first top-level dynamic declaration.
    first = dynamic[0] if dynamic else None
    if first == "slash":
        results.append((True, "pose index", "slash is the first dynamic model"))
        return
    results.append(
        (
            False,
            "pose index",
            f"the first dynamic model in {sdf_file.name} is "
            f"{first!r}, not 'slash'. env._on_pose reads msg.transforms[0] and the "
            "bridge drops entity names, so the env would follow that model's pose "
            "instead of the car's. Either declare slash first or match by index "
            "resolved once at startup",
        )
    )


def check_world_argument(results, repo, course):
    """The wrappers never pass world:= , so they always launch the default."""
    if course == "speed":
        results.append(
            (True, "launch world arg", "speed course is training.launch.py's default")
        )
        return
    wrappers = ["rl/bale_follower/_sim_stack.sh", "rl/bale_follower/train_resilient.sh"]
    missing = [w for w in wrappers if "world:=" not in read(repo / w)]
    if not missing:
        results.append((True, "launch world arg", "wrappers pass world:= through"))
        return
    results.append(
        (
            False,
            "launch world arg",
            "these start training.launch.py without a "
            f"world:= argument, so they load the speed course whatever was asked "
            f"for: {', '.join(missing)}",
        )
    )


def check_teleport_world(results, repo, world):
    """teleport_api.py resolves its world from CFR_SIM_WORLD, defaulting to the
    speed course. simulation.launch.py passes it; training.launch.py does not."""
    launch = repo / "jetson/cfr_arduino_bridge/launch/training.launch.py"
    text = read(launch)
    if world == "cfr_speed_course":
        results.append((True, "teleport world", "teleport_api's default world matches"))
        return
    if "CFR_SIM_WORLD" in text:
        results.append(
            (True, "teleport world", "training.launch.py sets CFR_SIM_WORLD")
        )
        return
    results.append(
        (
            False,
            "teleport world",
            "training.launch.py starts teleport_api.py without "
            "CFR_SIM_WORLD, and teleport_api defaults to cfr_speed_course. Every "
            "episode reset would address a world that is not loaded, so resets "
            "silently do nothing. simulation.launch.py already passes it through "
            "additional_env={'CFR_SIM_WORLD': world_name} -- copy that",
        )
    )


def check_randomizer(results, repo):
    """Warning only: training never starts the obstacle randomizer."""
    text = read(repo / "jetson/cfr_arduino_bridge/launch/training.launch.py")
    if "obstacle_randomizer" not in text:
        results.append(
            (
                None,
                "domain randomization",
                "training.launch.py never starts "
                "obstacle_randomizer_node, and env.reset() never asks for a new "
                "layout, so every episode would run the nominal SDF bucket and hoop "
                "poses. That keeps static geometry parsing valid, at the cost of a "
                "policy that can memorize one layout -- worth deciding on purpose",
            )
        )


def check_unmodelled_obstacles(results, sdf_text, sdf_file):
    """Warning only: geometry the analytic scan and collision check cannot see."""
    obstacles = sorted(
        {
            name
            for name, _static in model_declarations(sdf_text)
            if name not in KNOWN_MODELS and re.search(r"^[a-z_]+(_\d+)?$", name)
        }
    )
    # Sub-model links (chassis, arms, body...) share the name pattern; keep the
    # ones that appear as top-level declarations with their own collision.
    obstacles = [
        name
        for name in obstacles
        if re.search(
            rf'<model name="{re.escape(name)}">.{{0,400}}?<collision', sdf_text, re.S
        )
    ]
    if obstacles:
        results.append(
            (
                None,
                "unmodelled geometry",
                f"{sdf_file.name} carries obstacle models the "
                "bale parser does not read, so they are invisible to both the "
                "observation and the collision check: "
                f"{', '.join(obstacles[:12])}"
                f"{' ...' if len(obstacles) > 12 else ''}. A policy trained here "
                "would learn to drive through them",
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", choices=["speed", "obstacle"], required=True)
    parser.add_argument("--repo", default=None, help="repo root (default: cwd upwards)")
    args = parser.parse_args()

    repo = (
        Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[4]
    )
    sdf_file = world_path(repo, args.course)
    sdf_text = read(sdf_file)
    if not sdf_text:
        print(f"no world file at {sdf_file}", file=sys.stderr)
        return 1
    world = world_name(sdf_text)

    results = []
    check_bale_link(results, sdf_text, sdf_file)
    check_pose_bridge(results, repo, world)
    check_config_world(results, repo, world)
    check_pose_index(results, sdf_text, sdf_file)
    check_teleport_world(results, repo, world)
    check_world_argument(results, repo, args.course)
    if args.course != "speed":
        check_unmodelled_obstacles(results, sdf_text, sdf_file)
        check_randomizer(results, repo)

    print(f"preflight: {args.course} course ({world})")
    failures = 0
    for passed, label, message in results:
        if passed is None:
            mark = "warn"
        elif passed:
            mark = "ok  "
        else:
            mark = "FAIL"
            failures += 1
        print(f"  [{mark}] {label}: {message}")

    if failures:
        print(
            f"\n{failures} blocking check(s) failed. Training on the "
            f"{args.course} course needs those code changes first; --force "
            "starts anyway and will fail at the first of them."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
