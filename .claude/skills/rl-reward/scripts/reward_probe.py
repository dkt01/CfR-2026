#!/usr/bin/env python3
"""Show what the reward actually pays, term by term, at representative states.

Reward coefficients are only meaningful against each other. A `k_touch` of 40
sounds like a strong deterrent until you notice a 5.5 m/s step earns +5.5 from
progress every 0.1 s while the touch penalty is charged once on a shallow
overlap -- which is how v7 ended up with a bale brush scoring positive. This
prints the per-term contribution at states the car really visits, so that
comparison happens before a run burns five hours, not after.

    python3 reward_probe.py                          # current config.yaml
    python3 reward_probe.py --set k_touch=200 --set k_center=6
    python3 reward_probe.py --compare proposed.yaml

Two things worth knowing about what this replaces:

`python reward.py` checks the RewardConfig *dataclass defaults*, not the
values in config.yaml -- and they have drifted apart (defaults carry
collision_penalty 50, the config trains with 200). Its ordering assertions are
still the right guardrail, so this re-runs the same invariants against the
values a run would really use.

The episode column answers the question coefficients hide: a small per-step
penalty charged 600 times an episode is not small. Compare it against
collision_penalty, which is charged once.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RL_DIR = REPO / "rl/bale_follower"


def load_reward_module():
    spec = importlib.util.spec_from_file_location("reward", RL_DIR / "reward.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is not there yet for a module loaded
    # straight from a path.
    sys.modules["reward"] = module
    spec.loader.exec_module(module)
    return module


def load_yaml_block(path, block):
    """Minimal flat-YAML reader, so this needs no PyYAML on the host."""
    values = {}
    inside = False
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            inside = line.split(":")[0].strip() == block
            continue
        if not inside:
            continue
        key, _, raw = line.strip().partition(":")
        raw = raw.split("#")[0].strip()
        if not raw:
            continue
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = raw
    return values


def scenarios(env, config):
    """States the car actually visits, in physical units, at control_hz.

    Progress distance is speed * dt, so these scale with the config's control
    rate: at 10 Hz a 5.5 m/s step covers 0.55 m and the progress term pays
    k_progress * 0.55 every step.
    """
    dt = 1.0 / float(env.get("control_hz", 10.0))
    top = float(env.get("max_speed", 2.5))
    cruise = top * 0.5
    clear = max(config.safe_clearance * 1.4, 0.45)
    brush = config.touch_clearance * 0.4

    return [
        (
            "clean, centered, top speed",
            dict(
                progress_distance=top * dt,
                min_clearance=clear,
                center_error=0.0,
                angular_z=0.1,
                prev_angular_z=0.08,
                steer_fraction=0.2,
                prev_steer_fraction=0.2,
                collided=False,
            ),
        ),
        (
            "clean, centered, cruise",
            dict(
                progress_distance=cruise * dt,
                min_clearance=clear,
                center_error=0.0,
                angular_z=0.1,
                prev_angular_z=0.08,
                steer_fraction=0.2,
                prev_steer_fraction=0.2,
                collided=False,
            ),
        ),
        (
            "hairpin, slow, near lock",
            dict(
                progress_distance=1.2 * dt,
                min_clearance=config.safe_clearance,
                center_error=0.15,
                angular_z=1.4,
                prev_angular_z=1.35,
                steer_fraction=0.95,
                prev_steer_fraction=0.95,
                collided=False,
            ),
        ),
        (
            "wall-hugging at top speed",
            dict(
                progress_distance=top * dt,
                min_clearance=config.safe_clearance * 0.7,
                center_error=0.30,
                angular_z=0.1,
                prev_angular_z=0.08,
                steer_fraction=0.2,
                prev_steer_fraction=0.2,
                collided=False,
            ),
        ),
        (
            "brushing a bale at top speed",
            dict(
                progress_distance=top * dt,
                min_clearance=brush,
                center_error=0.35,
                angular_z=0.1,
                prev_angular_z=0.08,
                steer_fraction=0.2,
                prev_steer_fraction=0.2,
                collided=False,
            ),
        ),
        (
            "sawing the steering at cruise",
            dict(
                progress_distance=cruise * dt,
                min_clearance=clear,
                center_error=0.0,
                angular_z=0.6,
                prev_angular_z=-0.6,
                steer_fraction=0.9,
                prev_steer_fraction=-0.9,
                collided=False,
            ),
        ),
        (
            "stalled in the open",
            dict(
                progress_distance=0.0,
                min_clearance=clear,
                center_error=0.0,
                angular_z=0.0,
                prev_angular_z=0.0,
                steer_fraction=0.0,
                prev_steer_fraction=0.0,
                collided=False,
            ),
        ),
        (
            "collision at top speed",
            dict(
                progress_distance=top * dt * 0.2,
                min_clearance=0.0,
                center_error=0.4,
                angular_z=0.4,
                prev_angular_z=-0.4,
                steer_fraction=0.5,
                prev_steer_fraction=-0.5,
                collided=True,
            ),
        ),
    ]


def evaluate(module, config, env):
    rows = []
    for name, state in scenarios(env, config):
        result = module.compute_reward(config, **state)
        rows.append((name, result))
    return rows


def print_table(rows, episode_steps, title):
    print(f"\n{title}")
    header = f"{'state':<30}{'total':>9}{'progress':>10}{'proximity':>11}{'touch':>9}{'center':>9}{'smooth':>9}{'/episode':>11}"
    print(header)
    print("-" * len(header))
    for name, result in rows:
        per_episode = result.total * episode_steps
        print(
            f"{name:<30}{result.total:>9.2f}{result.progress:>10.2f}"
            f"{result.proximity:>11.2f}{result.touch:>9.2f}{result.centering:>9.2f}"
            f"{result.smoothness:>9.2f}{per_episode:>11.0f}"
        )
    print(
        "\n/episode = this step's reward held for a whole episode "
        f"({episode_steps} steps). Compare against collision_penalty, charged once."
    )


def check_invariants(rows):
    """The orderings reward.py asserts, re-run against the loaded config.

    These are invariants rather than preferences: if a shortcut outscores the
    behaviour it shortcuts, PPO will find it long before a human notices.
    """
    by_name = {name: result.total for name, result in rows}
    checks = [
        (
            "centered beats wall-hugging",
            by_name["clean, centered, top speed"]
            > by_name["wall-hugging at top speed"],
        ),
        (
            "clearing beats brushing",
            by_name["wall-hugging at top speed"]
            > by_name["brushing a bale at top speed"],
        ),
        ("contact never pays", by_name["brushing a bale at top speed"] < 0),
        (
            "moving beats sitting still",
            by_name["clean, centered, cruise"] > by_name["stalled in the open"],
        ),
        (
            "anything beats crashing",
            by_name["stalled in the open"] > by_name["collision at top speed"],
        ),
        (
            "steady steering beats sawing",
            by_name["clean, centered, cruise"]
            > by_name["sawing the steering at cruise"],
        ),
        ("a survivable hairpin still pays", by_name["hairpin, slow, near lock"] > 0),
    ]
    print("\ninvariants")
    failed = 0
    for label, passed in checks:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {label}")
        failed += 0 if passed else 1
    return failed


def apply_overrides(values, overrides):
    for item in overrides:
        key, _, raw = item.partition("=")
        values[key.strip()] = float(raw)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(RL_DIR / "config.yaml"))
    parser.add_argument(
        "--compare",
        default=None,
        help="a second config.yaml (or the same one plus --set) to print beside it",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="reward coefficient override for the comparison column, e.g. --set k_touch=200",
    )
    args = parser.parse_args()

    module = load_reward_module()
    env = load_yaml_block(args.config, "env")
    base_values = load_yaml_block(args.config, "reward")
    base = module.RewardConfig(**base_values)

    episode_steps = int(
        float(env.get("episode_time_limit_s", 60.0))
        * float(env.get("control_hz", 10.0))
    )

    base_rows = evaluate(module, base, env)
    print(f"config: {args.config}")
    print("  " + "  ".join(f"{k}={v:g}" for k, v in base_values.items()))
    print_table(base_rows, episode_steps, "current")
    failed = check_invariants(base_rows)

    if args.compare or args.overrides:
        other_values = load_yaml_block(args.compare or args.config, "reward")
        other_values = apply_overrides(other_values, args.overrides)
        other = module.RewardConfig(**other_values)
        changed = {
            k: (base_values.get(k), v)
            for k, v in other_values.items()
            if base_values.get(k) != v
        }
        print("\n" + "=" * 72)
        print(
            "proposed: "
            + (
                ", ".join(f"{k} {a:g} -> {b:g}" for k, (a, b) in changed.items())
                or "no change"
            )
        )
        other_rows = evaluate(module, other, env)
        print_table(other_rows, episode_steps, "proposed")
        failed += check_invariants(other_rows)

        print("\ndelta (proposed - current)")
        for (name, before), (_, after) in zip(base_rows, other_rows):
            print(f"  {name:<30}{after.total - before.total:>+9.2f}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
