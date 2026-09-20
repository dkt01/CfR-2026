# Characterization Results — Sessions B and C

Analysis of the 23 runs recorded 2026-09-18 and pulled off the Orin on
2026-09-17. Source runs are in `runs/`; each carries its own `report.md`.

**Read the sensor caveat first.** It invalidated the first pass of every
longitudinal number in this campaign, and it is the reason several values below
differ from what `analyze_run.py` originally printed.

---

## 0. The sensor caveat

`zed_static` is clean: 17.7 mm position noise, 0.019 deg yaw noise, zero
dropouts over 305 s. That result is real, and it is also misleading — it
measures the ZED **parked**.

In motion on low-texture asphalt the ZED's *velocity* output is unusable:

| | ZED `odom_vx` | tachometer `speed` |
| --- | --- | --- |
| gross outliers, coastdown | 408 / 8916 samples (4.6%) | 5 / 9353 (0.05%) |
| worst single-sample pose jump | 25.2 m in 20 ms | n/a |
| `odom_valid` during those jumps | **1** | n/a |

The failure is one-sided — it fabricates speed, it never withholds it — so the
old `_speed_column()` rule ("use odometry unless it dropped out") never fired,
and every longitudinal fit ran on the corrupted channel. Effects:

- coastdown residuals of **5042 N** on a 3.6 kg car, and a fitted *negative*
  rolling resistance
- a headline "forward and reverse drag are **94.3% apart**, genuinely
  asymmetric" — which on clean data is **3.3% apart**, i.e. symmetric
- pulse-map scatter of ±11.7 m/s on a car limited to 4.5 m/s

**What each channel is good for:**

- **ZED yaw** — trustworthy throughout. Cumulative yaw and integrated `odom_wz`
  agree within 5% on every segment, with zero jumps. All lateral results below
  rest on it.
- **ZED position** — usable if gross jumps are rejected; noise-dominated below
  ~0.5 m/s.
- **ZED velocity** — do not use in motion.
- **Tachometer** — reliable above ~1 m/s. Below ~0.6 m/s it reads exactly zero
  in ~50% of samples (one magnet, ~4 pulses/s against a 50 Hz log), so its
  *mean* underestimates badly; use the non-zero median, and not at all at crawl.

---

## 1. Longitudinal — coast drag

Pooled across all six coast legs, binned by speed, MAD-filtered, from the
tachometer:

```
decel = 0.606 + 0.130 * v      m/s^2          (v in m/s)
f0 = 2.18 N   f1 = 0.47 N/(m/s)   f2 ~ 0
```

| speed | measured decel | `vehicle.yaml` before this campaign |
| --- | --- | --- |
| 1 m/s | 0.74 m/s² | 0.1 fwd / 0.3 rev |
| 2 m/s | 0.87 m/s² | " |
| 3 m/s | 1.00 m/s² | " |

**The simulator coasts 7–10x too freely.** This is the largest single
sim-to-car error the campaign found.

Forward and reverse agree to 3.3% (0.891 vs 0.863 m/s² mean over 0.5–3.5 m/s),
so the direction-dependent fudge in the config is not needed — drop it and use
one curve.

`f2` (aero) is indistinguishable from zero and should be *fixed* at zero, not
fitted: at 3 m/s a 3.6 kg car sees ~0.08 N of aero drag against ~4 N total.
Leaving it free lets it absorb curvature and go negative, which is what the
per-leg fits did.

## 2. Longitudinal — feedforward

From 109 steady-cruise windows (≥1.2 s, speed sd < 0.12 m/s) pooled across
every run, since `pulse_staircase` itself could not deliver this (see §6):

```
forward:  kS = +28.8 us   kV = 11.13 us per m/s    (n=88, rms 7.9 us)
reverse:  kS = -27.9 us   kV = 11.48 us per m/s    (n=21, rms 7.9 us)
```

Symmetric to within 3%. Independently corroborated by the staircase deadband:
no motion at +30 us, motion at +40 us.

Pack sensitivity is **-14.2 us per volt** (adding it lifts r² from 0.598 to
0.646). Indicative only — the session spanned just 11.65–12.38 V.

## 3. Longitudinal — braking

**There is no usable braking authority.** Every non-zero brake limit produces
the *same* commanded pulse, and all of them stop the car **slower than simply
coasting**:

| brake limit | commanded pulse | mean decel |
| --- | --- | --- |
| 0 µs (coast) | 1500–1517 (neutral) | **1.22 m/s²** |
| 40 µs | ~1436 | 0.89 m/s² |
| 80 µs | ~1437 | 0.83 m/s² |
| 120 µs | ~1436 | 0.74 m/s² |

Two separate problems:

1. The `brake_limit` parameter does not scale the output — 40, 80 and 120 all
   clamp to ~1436 µs. Expected 1460 / 1420 / 1380.
2. That reverse-side pulse is read by the ESC as reverse *drive*, not braking,
   so commanding brake actively fights the deceleration. This is the documented
   "ESC can drive away backwards" behaviour, now quantified.

Model the car as coast-only until the firmware is fixed. `mu_longitudinal` is
**not** measurable from this run — the 0.12 g the report prints is the coast
drag, not a friction limit.

## 4. Lateral

**Direction correction (2026-09-20):** the recorded positive commands labelled
"left" produced negative ZED yaw, and negative commands labelled "right"
produced positive yaw. Thus the left/right labels in the original analysis
below describe commanded labels, not physical turn directions. Physical left
was the stronger side (0.256 rad at half command), and physical right was
0.191 rad. The bridge now inverts steering, and the simulator's effective
angle table swaps the side magnitudes accordingly. Full-lock values remain
extrapolations and require a new car run.

From three independent `skidpad` runs (tach speed + ZED yaw), which agree
closely:

| | at command ±0.50 |
| --- | --- |
| left | 0.191 rad (10.9°), R 1.71–2.11 m |
| right | 0.256 rad (14.7°), R 1.18–2.11 m |

The `effective_angle_table` previously assumed 0.20 rad at ±0.5. **Left is
essentially right; right is 28% low.** The ~34% left/right asymmetry is real
and consistent across runs — consistent with the known 1504 µs auto-mode centre
(A6 `center_offset`).

**Understeer gradient K ≈ +0.007 rad/(m/s²)** (left side, r² 0.71; standard
form `delta = L/R + K·a_y`). Turn radius grows ~14% from 0.7 to 2.9 m/s at
fixed steering. The simulator's `mu = 50` makes K = 0, so at 3.2 m/s on a 2 m
radius the car needs ~2° more steering than the sim predicts.

**Lateral grip ≥ 0.55 g** (peak 5.39 m/s²) — a lower bound; the sweep stayed
below the slide limit by design.

**Yaw step response: dead time ≈ 0.19 s, then ~90% within ~0.25 s.**
The `1.76 s` in the step_steer report is a fit artifact — the response is
essentially complete before the profile's second sample, so the first-order fit
had nothing to fit. Do not put it in `vehicle.yaml`; it would be ~9x too slow.

## 5. Closed-loop gain sweep

Comparing only the complete 12-step runs:

| kP | kI | MAE | settled MAE | settled SD | t90 | run |
| -- | -- | --- | --- | --- | --- | --- |
| 7.62 | 4.76 | 229 | 153 | 275 | 1.69 s | default (on-blocks tune) |
| 16.0 | 10.0 | 190 | 103 | 155 | 1.45 s | kp16ki10 |
| **24.0** | **10.0** | **169** | **89** | **114** | 1.61 s | kp24ki10 |

**kP = 24, kI = 10 wins** — settled error down 42% and settled SD down 59%
against the on-blocks default. An 8-step kp24 run agrees (174 / 100 / 123).

The trend has **not** turned over between 16 and 24, so the optimum is probably
higher. Try kP = 32 before committing.

This confirms the on-blocks tune was substantially under-gained on the ground.

---

## 6. What these runs cannot give

- **The open-loop pulse→speed map.** `pulse_staircase` holds are too short: no
  forward step reached steady state, and the holds *shrink* as the steps grow
  (5.3 s → 4.3 s) because the safety envelope truncates them. That flattens the
  top of the curve into fake saturation — ks065 and ks085 both read ~2.9 m/s
  while still accelerating. §2 works around it; a re-run wants holds long
  enough to settle, or an explicit settle criterion instead of a fixed time.
- **Steering authority / full lock.** `steer_authority` ran at ~0.6 m/s
  commanded, which is the one regime where *both* speed sensors fail — the ZED
  is noise-dominated and the tach reads zero half the time. The resulting map is
  non-monotonic (full right lock reads *less* angle than half right). Re-run it
  at 1.5–2.0 m/s with a correspondingly larger bubble. `max_angle_*` stays
  `guess`.

## 7. Simulator fidelity after these changes

The values below were folded into `config/vehicle.yaml` and
`config/arduino_bridge.yaml`, and `sim_vehicle_node` was rewritten to use them.
Each row was then **verified by running the same characterization profile in
Gazebo** (`use_sim:=true`) and analysing the result with the same script as the
car, rather than by inspection.

| Quantity | Car | Sim before | Sim now |
| --- | --- | --- | --- |
| coast decel, mean over a 3.2 m/s leg | 0.705 m/s² | 0.1 fwd / 0.3 rev | **0.717 m/s²** |
| coastdown `f0` | 2.18 N | n/a | **2.05–2.10 N** |
| fwd/rev drag asymmetry | 3.3% | built-in fudge | **0.0%** |
| 3.2 → 0.8 m/s command | ~2.6 s | 1 tick | coast-limited |
| effective steering, left / right at ±0.5 | 0.191 / 0.267 rad | 0.20 / 0.20 | **0.177 / 0.254** |
| left-right asymmetry | 40% | none | **44%** |
| turn radius growth, 1.5 → 3.0 m/s | 13.0% | 0% (kinematic) | **12.6%** |
| tachometer distinct speed values | 889 | **0** (hard-coded) | **503** |
| tach zeros through a coast leg | 27% | n/a | **28%** |
| tach goes blind at | ~0.3 m/s | never | **~0.3 m/s** |
| reported `throttle_us` range | 1409–1595 | 1500 always | **1425–1583** |

### The tachometer

`sim_vehicle_node` used to report `rpm = 0` and `throttle_us = 1500`
unconditionally, so a simulated run logged an empty speed channel while the car
logged ~1000 distinct values. Anything reading the speed or throttle trace in
simulation was reading a constant.

It now runs a transcription of the firmware's `TachSensor`, driven by the
simulated ground speed, and derives `wheel_rpm` / `speed` from it exactly as
`arduino_bridge_node` does — so the simulated speed channel inherits the
tachometer's blind spots instead of leaking the true speed. Two behaviours fall
out of the algorithm rather than being coded in, and both match the car:

- it needs **two** timestamps before it reports anything, so the first
  revolution after any stop reads zero
- it forgets its history after the 0.4 s stall timeout, putting a hard floor at
  150 spur RPM (~0.3 m/s)

Through the same `coast_fwd_3p2` leg, car and sim drop to exactly 0.00 at the
same point (4.4 s) from the same last reading (0.42 vs 0.33 m/s).

**Not modelled:** the ~5% merged-pulse rate (`sensors.tach_merged_pulse_rate`)
and the sign glitches the real tach shows through a direction reversal, where
the magnitude is sensor-derived but the sign comes from the controller's
direction estimate and briefly lags. The zero-fraction match is already 27% vs
28% without them, so adding stochastic pulse-dropping would be unvalidated
noise rather than fidelity.

### Caveats on the above

- The steering table's **±1.0 ends are extrapolated**, so the simulator's full
  lock is as unmeasured as it ever was. Only the ±0.5 rows are measured.
- `max_acceleration` is a flat 3.0 m/s². The real thrust tapers at high speed,
  but `pulse_staircase` was truncated (§6) so the taper is unresolved; the sim
  will out-accelerate the car above ~2.5 m/s.
- The simulated ZED is Gazebo's odometry and is **far cleaner than the real
  camera in motion** (§0). Nothing here reproduces the 25 m pose jumps, so the
  simulator remains optimistic about odometry in exactly the way that cost this
  campaign its first pass of results.

## 8. Recommended `vehicle.yaml` changes

Ready to apply as `measured`:

- `longitudinal.f0_rolling` = 2.18 N, `f1_viscous` = 0.47, `f2_aero` = 0 (fixed)
- drop the forward/reverse coast asymmetry — one curve
- `longitudinal.ks` = 28.8 µs, `kv` = 11.1 µs per m/s
- `lateral.understeer_gradient` = +0.007 rad/(m/s²)
- `lateral.mu_lateral` ≥ 0.55 (lower bound — tag `measured`, note the bound)
- `effective_angle_table` rows at ±0.5: `[-0.5, -0.256]`, `[0.5, 0.191]`
- `steering.tau` ≈ 0.19 s dead time + ~0.1 s rise

Leave as `guess`: `steering.max_angle_*`, `longitudinal.mu_longitudinal`,
`inertia.*` (the step_steer τ is unusable; A3 stays unvalidated).

Do **not** run `apply_vehicle_patch.py` blindly on these runs — several
generated `vehicle_patch.yaml` files still carry the artifacts called out
above (the step_steer τ, the brake-derived µ, the per-leg drag coefficients).
