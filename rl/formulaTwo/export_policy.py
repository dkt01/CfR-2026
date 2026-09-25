#!/usr/bin/env python3
"""SB3 checkpoint -> the .npz the car runs: the ACTOR only.

    python3 export_policy.py runs/f2_v1/best_model.zip -o runs/f2_v1/policy.npz

The checkpoint's input layer is as wide as the whole observation, privileged
block included; the actor saw those columns as zeros (train.ActorSlice), so
they are dropped here and the exported network is exactly as wide as what the
car can build: 35 map features + the depth stack + its age.  The file then
loads with formulaOne's NumpyPolicy unchanged, and the export checks itself
against torch before it writes anything anyone might drive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


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
    np.savez(out, **blob)
    print(
        f"wrote {out}  ({actor_dim} -> "
        + " -> ".join(str(blob[f"w{i}"].shape[1]) for i in range(len(layers)))
        + f"; dropped privileged columns, largest weight {dropped:.1e})"
    )

    from policy import NumpyPolicy

    check = NumpyPolicy.load(out)
    full = pol.observation_space.shape[0]
    probe = np.random.default_rng(0).normal(0, 1, (64, full)).astype(np.float32)
    with torch.no_grad():
        want, _, _ = pol(torch.as_tensor(probe), deterministic=True)
    want = np.clip(want.cpu().numpy(), -1.0, 1.0)
    error = float(np.abs(check.act(probe[:, :actor_dim]) - want).max())
    print(f"numpy (actor only) vs torch (full obs), worst of 64: {error:.2e}")
    if error > 1e-5:
        raise SystemExit("EXPORT MISMATCH -- the actor reads privileged inputs")


if __name__ == "__main__":
    main()
