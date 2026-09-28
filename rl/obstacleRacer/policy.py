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

A recurrent policy (v7) has an LSTM in front of the MLP: `act` then carries
its hidden state from call to call, one call per control step, and `reset`
starts it fresh -- at the start of every run, as training does.

Actions come out of the network unsquashed -- stable-baselines3 puts a
diagonal Gaussian on a Box action space and leaves the clipping to the env --
so the clip to [-1, 1] here is part of the policy, not a safety margin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyPolicy:
    def __init__(
        self, weights, biases, activation="tanh", meta=None, output="clip", lstm=None
    ):
        self.weights = weights
        self.biases = biases
        # (w_ih (4H, D), w_hh (4H, H), b (4H,)), PyTorch's gate order i f g o.
        self.lstm = lstm
        self.h = self.c = None
        # "tanh": the policy's mean is squashed (ppo_policy.SquashedMeanPolicy).
        self.output = output
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
        output = str(blob["output"]) if "output" in blob.files else "clip"
        lstm = None
        if "lstm_w_ih" in blob.files:
            lstm = (blob["lstm_w_ih"], blob["lstm_w_hh"], blob["lstm_b"])
        policy = cls(weights, biases, act, meta, output, lstm)
        expected = int(blob["obs_dim"])
        width = lstm[0].shape[1] if lstm is not None else weights[0].shape[0]
        if width != expected:
            raise ValueError(
                f"{path} was trained on a {expected}-wide observation but its "
                f"first layer takes {width}"
            )
        policy.obs_dim = expected
        return policy

    def reset(self, batch=1):
        """Fresh LSTM state for `batch` cars: call at the start of every run."""
        if self.lstm is not None:
            hidden = self.lstm[1].shape[1]
            self.h = np.zeros((batch, hidden))
            self.c = np.zeros((batch, hidden))

    def act(self, obs):
        """(B, obs_dim) -> (B, 2) deterministic action, already clipped.

        With an LSTM this is one control step: it advances the hidden state.
        """
        h = np.asarray(obs, dtype=np.float64)
        if self.lstm is not None:
            if self.h is None or len(self.h) != len(h):
                self.reset(len(h))
            w_ih, w_hh, b = self.lstm
            gates = h @ w_ih.T + self.h @ w_hh.T + b
            i, f, g, o = np.split(gates, 4, axis=1)
            sig = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
            self.c = sig(f) * self.c + sig(i) * np.tanh(g)
            self.h = sig(o) * np.tanh(self.c)
            h = self.h
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = self.activation(h @ w + b)
        out = h @ self.weights[-1] + self.biases[-1]
        if self.output == "tanh":
            out = np.tanh(out)
        return np.clip(out, -1.0, 1.0)
