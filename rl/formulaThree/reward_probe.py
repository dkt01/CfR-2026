#!/usr/bin/env python3
"""Compare formulaTwo and formulaThree incentives before spending a training run."""

import importlib.util
from pathlib import Path

import numpy as np
import yaml

from reward import Reward

HERE = Path(__file__).resolve().parent


def state(speed=3.0, clearance=0.25, lateral=0.0, crash=False, dt=0.05):
    return dict(
        dt=dt,
        advance=speed * dt,
        speed=speed,
        v_cap=5.2,
        clearance=clearance,
        lateral=lateral,
        psi=0.0,
        steer_step=0.0,
        steer_jerk=0.0,
        lapped=0.0,
        finished=0.0,
        crashed=float(crash),
        stalled=0.0,
        stopping=0.0,
        stopped=0.0,
        lap_gain=0.0,
    )


def main():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    old_cfg = yaml.safe_load((HERE.parent / "formulaTwo/config.yaml").read_text())
    spec = importlib.util.spec_from_file_location(
        "formula_two_reward", HERE.parent / "formulaTwo/reward.py"
    )
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    old, new = old_module.Reward(old_cfg), Reward(cfg)
    rows = {
        "centered straight": state(speed=5.2),
        "survivable hairpin": state(speed=1.4, clearance=0.18, lateral=0.05),
        "wall hugging": state(speed=5.2, clearance=0.08, lateral=0.22),
        "graze at full speed": state(speed=5.2, clearance=0.0),
        "standing still": state(speed=0.0),
        "reverse": state(speed=-1.0),
    }
    print("State                     F2 / second    F3 / second    delta")
    rates = {}
    for name, values in rows.items():
        a = (
            float(old.step(**{**values, "crashed": float(values["clearance"] <= 0)})[0])
            / values["dt"]
        )
        current = {
            **values,
            "crashed": float(values["clearance"] <= cfg["collision"]["safety_margin"]),
        }
        b = float(new.step(**current)[0]) / values["dt"]
        rates[name] = b
        print(f"{name:25s} {a:12.2f} {b:14.2f} {b - a:+9.2f}")
    print(
        "Contact includes a one-time terminal charge; do not multiply that row by episode duration."
    )
    assert rates["centered straight"] > rates["wall hugging"]
    assert rates["survivable hairpin"] > rates["standing still"]
    assert rates["graze at full speed"] < 0
    assert rates["reverse"] < rates["standing still"]
    # Fixed route progress/bonuses cancel: every saved second is worth time.
    race_distance = 3 * 110.2

    def clean_return(seconds):
        return (
            cfg["reward"]["progress"] * race_distance - cfg["reward"]["time"] * seconds
        )

    assert clean_return(90) > clean_return(105) > clean_return(150)
    assert cfg["reward"]["lap_improve"] == 0  # No reward for sandbagging lap one.
    assert cfg["reward"]["crash"] <= cfg["reward"]["stall"]
    assert cfg["reward"]["time"] / cfg["reward"]["progress"] < 1.4
    a = state()
    b = state(dt=0.1)
    assert np.isclose(new.step(**a)[0] * 2, new.step(**b)[0])
    print("Clean 90 s vs 105 s race: +60 reward; no prior-lap sandbagging bonus.")
    print("Incentives passed. These probes do not establish lap performance.")


if __name__ == "__main__":
    main()
