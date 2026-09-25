#!/usr/bin/env python3
"""stable-baselines3 checkpoint -> the .npz the car runs (copied from rl/formulaOne).

    python3 export_policy.py runs/v1/best_model.zip -o runs/v1/policy.npz

Deliberately a separate step rather than something `train.py` does silently.
The exported file is the deployment artefact: it is what `evaluate.py` scores,
what `validate.sh` drives in Gazebo, and what the ROS node loads on the Orin.
Exporting once and then testing that one file end to end is the only way the
thing that was measured and the thing that drives are the same thing.

The observation width is written into the file so that a policy trained
before a change to observation.py refuses to load rather than reading its
weights against a different observation and driving into a bale with perfect
confidence.
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

    model = PPO.load(args.checkpoint, device="cpu")
    layers = []
    for module in list(model.policy.mlp_extractor.policy_net) + [
        model.policy.action_net
    ]:
        if isinstance(module, torch.nn.Linear):
            layers.append(module)
    if not layers:
        raise SystemExit("no Linear layers found; is this a PPO MlpPolicy?")

    # From the FIRST LAYER's shape, not from a module constant.  The constant
    # tracks whatever config.yaml says today; the checkpoint was trained
    # against whatever it said then, and exporting the two against each other
    # is how a 35-wide policy gets shipped claiming to be 36-wide.
    # `layers` holds torch Linear MODULES, so the input width is
    # `in_features` -- indexing it like an array raised TypeError and took a
    # 39-minute training run's export with it.
    obs_dim = int(layers[0].in_features)
    squashed = bool(getattr(model.policy, "squashes_mean", False))
    blob = {
        "n_layers": len(layers),
        "obs_dim": obs_dim,
        "activation": "tanh",
        "output": "tanh" if squashed else "clip",
    }
    for i, layer in enumerate(layers):
        # Transposed once, here, so the runtime is a plain `obs @ w + b`.
        blob[f"w{i}"] = layer.weight.detach().cpu().numpy().T.astype(np.float64)
        blob[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float64)

    out = args.out or args.checkpoint.with_suffix(".npz")
    np.savez(out, **blob)
    shape = " -> ".join(
        [str(blob["w0"].shape[0])]
        + [str(blob[f"w{i}"].shape[1]) for i in range(len(layers))]
    )
    print(
        f"wrote {out}  ({shape}, {sum(b.size for b in blob.values() if hasattr(b, 'size')):,} parameters)"
    )

    # Prove the export before anyone drives it: the numpy path and torch must
    # agree on the same random observations.
    from policy import NumpyPolicy

    check = NumpyPolicy.load(out)
    probe = np.random.default_rng(0).normal(0, 1, (64, obs_dim)).astype(np.float32)
    with torch.no_grad():
        want, _, _ = model.policy(torch.as_tensor(probe), deterministic=True)
    want = np.clip(want.cpu().numpy(), -1.0, 1.0)
    error = float(np.abs(check.act(probe) - want).max())
    print(f"numpy vs torch, worst of 64 samples: {error:.2e}")
    if error > 1e-5:
        raise SystemExit("EXPORT MISMATCH -- do not deploy this file")


if __name__ == "__main__":
    main()
