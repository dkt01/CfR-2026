"""Numpy actor inference on the Orin; no training dependencies required."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np


def config_digest(config):
    runtime = {k: v for k, v in config.items() if k != "train"}
    return hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()


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

    def check_config(self, config):
        if str(self.meta.get("meta_schema", "")) != "formulaThree-v1":
            raise ValueError(
                "Not a formulaThree actor; export with formulaThree/export_policy.py"
            )
        if str(self.meta.get("meta_config_sha256", "")) != config_digest(config):
            raise ValueError(
                "Policy/config mismatch: use the config exported beside this policy"
            )

    def act(self, obs):
        """(B, obs_dim) -> (B, 2) deterministic action, already clipped."""
        h = np.asarray(obs, dtype=np.float64)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = self.activation(h @ w + b)
        return np.clip(h @ self.weights[-1] + self.biases[-1], -1.0, 1.0)
