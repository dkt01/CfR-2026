"""Run a trained policy with numpy alone.

The Orin does not need torch to drive.  `export_policy.py` flattens the
trained network into an .npz of weights and biases, and this file multiplies
them out -- a 30 -> 128 -> 128 -> 2 MLP, which is about 21k multiply-adds, or
roughly 60 microseconds a tick at 20 Hz.  What that buys is worth more than
the arithmetic:

  * no torch, no CUDA and no python version fight on the car,
  * the exported file is small enough to read and diff, and
  * the same four lines of arithmetic run in the evaluator and in the ROS
    node, so "the checkpoint that was validated" and "the checkpoint that
    drove" cannot be different objects.

Actions come out of the network unsquashed -- stable-baselines3 puts a
diagonal Gaussian on a Box action space and leaves the clipping to the env --
so the clip to [-1, 1] here is part of the policy, not a safety margin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyPolicy:
    def __init__(self, weights, biases, activation="tanh", meta=None):
        self.weights = weights
        self.biases = biases
        self.activation = (
            np.tanh if activation == "tanh" else lambda v: np.maximum(v, 0)
        )
        self.meta = meta or {}

    @classmethod
    def load(cls, path):
        blob = np.load(Path(path), allow_pickle=False)
        n = int(blob["n_layers"])
        weights = [blob[f"w{i}"] for i in range(n)]
        biases = [blob[f"b{i}"] for i in range(n)]
        meta = {k: blob[k] for k in blob.files if k.startswith("meta_")}
        act = str(blob["activation"]) if "activation" in blob.files else "tanh"
        policy = cls(weights, biases, act, meta)
        expected = int(blob["obs_dim"])
        if weights[0].shape[0] != expected:
            raise ValueError(
                f"{path} was trained on a {expected}-wide observation but its "
                f"first layer takes {weights[0].shape[0]}"
            )
        policy.obs_dim = expected
        return policy

    def act(self, obs):
        """(B, obs_dim) -> (B, 2) deterministic action, already clipped."""
        h = np.asarray(obs, dtype=np.float64)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = self.activation(h @ w + b)
        return np.clip(h @ self.weights[-1] + self.biases[-1], -1.0, 1.0)
