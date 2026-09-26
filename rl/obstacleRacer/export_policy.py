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

    import ppo_policy

    model = ppo_policy.load(args.checkpoint, device="cpu")
    lstm = getattr(model.policy, "lstm_actor", None)
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
    obs_dim = int(layers[0].in_features) if lstm is None else int(lstm.input_size)
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
    if lstm is not None:
        if lstm.num_layers != 1:
            raise SystemExit("policy.py runs a one-layer LSTM")
        blob["lstm_w_ih"] = lstm.weight_ih_l0.detach().cpu().numpy().astype(np.float64)
        blob["lstm_w_hh"] = lstm.weight_hh_l0.detach().cpu().numpy().astype(np.float64)
        blob["lstm_b"] = (
            (lstm.bias_ih_l0 + lstm.bias_hh_l0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    out = args.out or args.checkpoint.with_suffix(".npz")
    np.savez(out, **blob)
    shape = " -> ".join(
        ([f"{obs_dim} -> LSTM {lstm.hidden_size}"] if lstm is not None else [])
        + [str(blob["w0"].shape[0])]
        + [str(blob[f"w{i}"].shape[1]) for i in range(len(layers))]
    )
    print(
        f"wrote {out}  ({shape}, {sum(b.size for b in blob.values() if hasattr(b, 'size')):,} parameters)"
    )

    # Prove the export before anyone drives it: the numpy path and torch must
    # agree on the same random observations.
    from policy import NumpyPolicy

    check = NumpyPolicy.load(out)
    rng = np.random.default_rng(0)
    if lstm is None:
        probe = rng.normal(0, 1, (64, obs_dim)).astype(np.float32)
        with torch.no_grad():
            want, _, _ = model.policy(torch.as_tensor(probe), deterministic=True)
        want = np.clip(want.cpu().numpy(), -1.0, 1.0)
        error = float(np.abs(check.act(probe) - want).max())
    else:
        # A 40-step episode for 16 cars: the hidden state must track too.
        check.reset(16)
        state, start, error = None, np.ones(16, bool), 0.0
        for _ in range(40):
            probe = rng.normal(0, 1, (16, obs_dim)).astype(np.float32)
            want, state = model.predict(
                probe, state=state, episode_start=start, deterministic=True
            )
            start = np.zeros(16, bool)
            got = check.act(probe)
            error = max(error, float(np.abs(got - np.clip(want, -1, 1)).max()))
    print(f"numpy vs torch, worst difference over the probe: {error:.2e}")
    if error > 1e-5:
        raise SystemExit("EXPORT MISMATCH -- do not deploy this file")


if __name__ == "__main__":
    main()
