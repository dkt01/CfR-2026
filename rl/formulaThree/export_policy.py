#!/usr/bin/env python3
"""Export only sensor-realizable actor inputs after checking numpy against torch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
import os
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO

    from train import make_policy_class

    model = PPO.load(
        args.checkpoint,
        device="cpu",
        custom_objects=dict(policy_class=make_policy_class()),
    )
    pol = model.policy
    actor_dim = int(pol.pi_features_extractor.actor_dim)
    layers = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    layers.append(pol.action_net)
    blob = {"n_layers": len(layers), "obs_dim": actor_dim, "activation": "tanh"}
    for i, layer in enumerate(layers):
        w = layer.weight.detach().cpu().numpy().T.astype(np.float64)
        if i == 0:
            dropped = np.abs(w[actor_dim:]).max() if w.shape[0] > actor_dim else 0.0
            w = w[:actor_dim]
        blob[f"w{i}"] = w
        blob[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float64)
    out = args.out or args.checkpoint.with_suffix(".npz")
    cfg = yaml.safe_load(args.checkpoint.with_name("config.yaml").read_text())
    if cfg.get("schema") != "formulaThree-v1":
        raise ValueError("Only formulaThree checkpoints can be exported")
    from policy import NumpyPolicy, config_digest

    blob["meta_schema"] = cfg["schema"]
    blob["meta_config_sha256"] = config_digest(cfg)
    print(
        f"checking {out}  ({actor_dim} -> "
        + " -> ".join(str(blob[f"w{i}"].shape[1]) for i in range(len(layers)))
        + f"; dropped privileged columns, largest weight {dropped:.1e})"
    )

    check = NumpyPolicy(
        [blob[f"w{i}"] for i in range(len(layers))],
        [blob[f"b{i}"] for i in range(len(layers))],
    )
    full = pol.observation_space.shape[0]
    probe = np.random.default_rng(0).normal(0, 1, (64, full)).astype(np.float32)
    with torch.no_grad():
        want, _, _ = pol(torch.as_tensor(probe), deterministic=True)
    want = np.clip(want.cpu().numpy(), -1.0, 1.0)
    error = float(np.abs(check.act(probe[:, :actor_dim]) - want).max())
    print(f"numpy (actor only) vs torch (full obs), worst of 64: {error:.2e}")
    if not np.isfinite(error) or error > 1e-5:
        raise SystemExit("EXPORT MISMATCH -- the actor reads privileged inputs")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out.parent, suffix=".npz", delete=False
    ) as tmp:
        temp = Path(tmp.name)
    try:
        np.savez(temp, **blob)
        os.replace(temp, out)
    finally:
        temp.unlink(missing_ok=True)
    out.with_name("config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote verified actor and config to {out.parent}")


if __name__ == "__main__":
    main()
