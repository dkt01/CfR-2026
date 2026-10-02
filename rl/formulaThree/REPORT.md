# Implementation checks — 2026-09-27

No trained racing checkpoint is shipped, and no claim of beating formulaOne
or formulaTwo is supported yet.

- A 256-step PPO smoke run exercised the batched environment, asymmetric
  actor/critic, held-out evaluation and checkpoint selection. It used four
  environments and a five-second episode limit, so it cannot establish laps.
- Exported actor: 293 → 128 → 128 → 2. Maximum absolute action error against
  torch across 64 probes, with independent privileged inputs: **5.34e-9**.
  The exported policy loaded and evaluated with its saved config.
- The reward probe compares formulaTwo and formulaThree at the same states,
  using each environment's own contact termination rule. At 5.2 m/s, 8 cm
  clearance and 22 cm cross-track error, reward changes from **+12.08/s to
  −7.24/s**. A survivable 1.4 m/s corner still pays **+3.73/s**. Saving 15 s
  on an otherwise identical clean race earns 60 more reward.

Nominal, deterministic **numpy-model** baseline, one three-lap episode per
setting; clearance is the conservative body estimate including sample and
motion allowances. These are not Gazebo results or robustness statistics.

| Reference profile scale | Clean finish and stop | Race time | Minimum clearance | Mean absolute CTE |
|---|---:|---:|---:|---:|
| 0.90 | No, margin violated at 25 m | — | 0.037 m | 0.039 m |
| 0.80 (default baseline) | Yes | 104.85 s | 0.095 m | 0.031 m |
| 0.75 | Yes | 111.75 s | 0.110 m | 0.026 m |
| 0.70 | Yes | 119.65 s | 0.113 m | 0.021 m |

All three finishing settings had zero measured overspeed. This establishes a
feasible nominal reference for learning; it does not show that the policy can
beat it under randomized dynamics or rendered Gazebo depth.

Native validation follow-up: all nine numerical regression tests pass in the
existing formulaOne Python environment. Native ROS 2 Jazzy and the repository
workspace are available; Docker is not required. The setup/train wrappers
now run these tests natively, and `validate.sh` launches native Gazebo.
