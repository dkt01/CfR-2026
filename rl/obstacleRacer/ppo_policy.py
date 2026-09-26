"""The PPO policy: a Gaussian whose mean stays inside the action box.

stable-baselines3's MlpPolicy puts an unbounded Gaussian on the action and
the env clips it.  v2 showed where that goes: the speed mean ran off to -6
at the bank, where every sample clips to the same full stop, so the policy
could never stumble onto the slow crawl that gets round; and because
exploration past the clip costs nothing, the entropy bonus grew the speed
std from 0.44 to 1.48 while the steering std collapsed to 0.012.

Here the mean is tanh(action_net), so it can approach but never sit beyond
+/-1, and the log std is held inside `log_std_range` by a sigmoid of a free
parameter.  Not a clamp: v3 clamped, and a clamp passes no gradient at its
bound, so once the std drifted to the floor (0.082) it could never come back
up.  The deterministic action is the mean itself; export_policy.py writes
`output: tanh` so the car's NumpyPolicy squashes it the same way.

From v7 the network is recurrent (train.recurrent): an LSTM carries a hidden
state from step to step, so the policy can remember what left the camera's
view -- the Wide Section's exit, hoop posts beside the car -- for as long as
that pays, not just the second the frame stack covers.
"""

from __future__ import annotations

import math

import numpy as np
import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class _SquashedMean:
    """The squashed mean and bounded std, for either policy class below."""

    squashes_mean = True

    def __init__(self, *args, log_std_range=(-2.5, -0.5), **kwargs):
        self.log_std_range = (float(log_std_range[0]), float(log_std_range[1]))
        lo, hi = self.log_std_range
        init = float(kwargs.get("log_std_init", 0.0))
        if not lo < init < hi:
            raise ValueError(f"log_std_init {init} is not inside {self.log_std_range}")
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError("SquashedMeanPolicy is for a Box action space")
        # stable-baselines3 built a free `log_std` parameter; keep the same
        # tensor (the optimizer already holds it) as the sigmoid's input and
        # serve `log_std` itself, bounded, from the property below -- which
        # is also what PPO logs as train/std.
        raw = self._parameters.pop("log_std")
        frac = (init - lo) / (hi - lo)
        raw.data.fill_(math.log(frac / (1.0 - frac)))
        self.register_parameter("log_std_raw", raw)

    @property
    def log_std(self) -> th.Tensor:
        lo, hi = self.log_std_range
        return lo + (hi - lo) * th.sigmoid(self.log_std_raw)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        mean = th.tanh(self.action_net(latent_pi))
        return self.action_dist.proba_distribution(mean, self.log_std)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["log_std_range"] = self.log_std_range
        return data


class SquashedMeanPolicy(_SquashedMean, ActorCriticPolicy):
    """v1-v6: a feed-forward MLP over the stacked frames."""


try:  # the car runs policy.py alone and needs neither
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
except ImportError:  # pragma: no cover
    RecurrentActorCriticPolicy = None

if RecurrentActorCriticPolicy is not None:

    class SquashedMeanLstmPolicy(_SquashedMean, RecurrentActorCriticPolicy):
        """v7: the same head behind an LSTM (sb3-contrib's RecurrentPPO)."""


def load(path, **kwargs):
    """A saved checkpoint, as PPO or RecurrentPPO -- whichever it was saved as."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        recurrent = b"Lstm" in z.read("data")
    if recurrent:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(path, **kwargs)
    from stable_baselines3 import PPO

    return PPO.load(path, **kwargs)


class Driver:
    """Deterministic actions for a batch of cars, carrying the LSTM state.

    Call it with the observation each step and tell it, via `ended`, which
    cars' episodes just ended (their env reset them), so their hidden state
    starts fresh.  An MLP policy ignores the state.
    """

    def __init__(self, model, n):
        self.model = model
        self.state = None
        self.start = np.ones(n, bool)

    def __call__(self, obs):
        action, self.state = self.model.predict(
            obs, state=self.state, episode_start=self.start, deterministic=True
        )
        self.start = np.zeros(len(obs), bool)
        return action

    def ended(self, mask):
        self.start = np.asarray(mask, bool).copy()
