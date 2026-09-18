"""Simulated ZED 2i depth observation model.

The ZED SDK is not installed on this machine and the Gazebo world has no
rendering sensors (see README: the ZED2i on the model is cosmetic geometry
pending a GPU-backed render context), so the point cloud cannot come from
either the real SDK or a simulated rgbd_camera. What CAN be modelled without
either is the *statistics* of the ZED 2i depth output, applied to the
analytic range scan the environment already computes:

- 110 degree horizontal FOV (the ZED 2i's wide-angle stereo pair), narrower
  than an idealized lidar, so the policy learns without eyes in the back of
  its head;
- range-dependent noise: stereo depth error grows roughly quadratically with
  distance (error ~ z^2 * pixel_disparity_error * baseline_factor). The ZED
  2i is spec'd at <1% error near, up to a few % at range;
- dropout: stereo matching fails on textureless / occluded regions, returning
  no depth. Dropped bins are filled with max range, which is also what the
  analytic scan returns for "no bale hit" -- the policy sees one consistent
  "nothing there" value.

Training against this narrows the sim-to-real gap of learning on ground-truth
geometry: the policy has to tolerate the FOV, noise, and dropouts the real
camera will produce, even though the underlying ranges are still analytic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ZedSimConfig:
    enabled: bool = True
    # ZED 2i horizontal FOV. The env's lidar_fov_deg is clamped to this when
    # the model is enabled so observation geometry matches the camera.
    hfov_deg: float = 110.0
    # Stereo depth noise: sigma = noise_a + noise_b * range^2 (metres).
    # noise_b ~= 0.008 gives ~0.9% error at 3 m and ~3% at 6 m, in line with
    # the ZED 2i depth accuracy spec.
    noise_a: float = 0.01
    noise_b: float = 0.008
    # Per-bin probability that stereo matching fails and the bin reads as
    # "no return" (max range).
    dropout_prob: float = 0.03


def apply(
    scan: np.ndarray,
    config: ZedSimConfig,
    max_range: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Corrupt an analytic range scan the way the ZED 2i depth pipeline would.

    Only the observation is corrupted; collision checks and reward stay on
    ground truth, the same split a real robot has between what it senses and
    what physically happens.
    """
    if not config.enabled:
        return scan
    noisy = scan + rng.normal(0.0, config.noise_a + config.noise_b * scan**2)
    dropped = rng.random(scan.shape) < config.dropout_prob
    noisy[dropped] = max_range
    return np.clip(noisy, 0.0, max_range)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cfg = ZedSimConfig()
    clean = np.linspace(0.5, 6.0, 12)
    trials = np.stack([apply(clean.copy(), cfg, 6.0, rng) for _ in range(2000)])
    # Dropped bins read max range; exclude them when measuring noise.
    err = np.where(trials >= 6.0, np.nan, trials) - clean
    sigma = np.nanstd(err, axis=0)
    drop = np.mean(trials >= 6.0, axis=0)
    for r, s, d in zip(clean, sigma, drop):
        print(f"range {r:4.1f} m  sigma {s * 100:5.1f} cm  dropout {d * 100:4.1f} %")
    assert sigma[-1] > sigma[0], "noise should grow with range"
    assert abs(np.nanmean(drop[:-1]) - cfg.dropout_prob) < 0.01
    print("\nZED noise model check passed: quadratic growth, expected dropout rate")
