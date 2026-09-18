# Field Card — Characterization

Print this. Full procedure is in [characterization.md](characterization.md).
Nothing here needs a network.

---

## Every run

```
ros2 launch cfr_arduino_bridge characterize.launch.py profile:=<NAME>
```

1. Wait for **"ARM: assert E-Stop now"**
2. **Press E-Stop**  → "E-Stop asserted"
3. **Clear E-Stop**, Auto Arm set → 3 s countdown → it drives
4. Straight profiles **reverse themselves back to you**
5. Launch exits on its own

### ABORT: press E-Stop. Always. It is the only safety device here.

### Wi-Fi will drop. It does not matter.
The run is autonomous and records to the Jetson. **Watch the E-Stop TUI**, not
the laptop — the XBee link does not depend on Wi-Fi.

---

## Record before and after every run

Read pack volts off the E-Stop TUI.

| Run | Profile                               | Volts before | Volts after | Result                      | Notes            |
| --- | ------------------------------------- | ------------ | ----------- | --------------------------- | ---------------- |
| 1   | zed_static                            | 11.55        | 11.51       |                             |                  |
| 2   | steer_authority                       | 11.50        | 11.46       |                             |                  |
| 3   | skidpad                               | 11.46        | 11.46       | failed (camera timeout)     |                  |
| 4   | skidpad                               | 11.46        |             | failed (camera timeout)     |                  |
| 5   | skidpad                               | 11.45        |             | failed (camera timeout)     |                  |
| 6   | skidpad                               | 11.41        |             |                             |                  |
| 7   | skidpad                               | 12.39        |             |                             |                  |
| 8   | skidpad                               | 12.33        | 12.25       |                             |                  |
| 9   | step_steer                            | 12.26        | 12.22       |                             |                  |
| 10  | pulse_staircase speed_slew_rate:=50.0 | 12.21        |             |                             |                  |
| 11  | pulse_staircase speed_slew_rate:=50.0 | 12.18        | 12.11       | turned too much to continue |                  |
| 12  | coastdown speed_slew_rate:=50.0       | 12.11        | 11.92       |                             |                  |
| 13  | brake_sweep                           | 11.96        | 11.88       |                             |                  |
| 14  | tune_profile                          | 11.89        | 11.87       |                             | ran off pavement |
| 15  | tune_profile                          | 11.88        | 11.81       |                             |                  |
| 16  | tune_profile kp24ki10                 | 11.84        | 11.80       | hit wall                    |                  |
| 17  | tune_profile kp24ki10                 | 11.79        | 11.76       |                             |                  |
| 18  | tune_profile kp16ki10                 | 11.79        | 11.71       |                             |                  |

---

## Session B — Parking lot (~1.5 h)

Car stays in one place. Needs roughly a 20 m clear square, clear on **both
sides**, not just ahead (`max_distance` = radius of the bubble, not typical
distance travelled).

- [x] `profile:=zed_static` — 5 min, **leave E-Stop asserted**, car will not move
- [x] `profile:=steer_authority` — 15 m bubble, both sides — the most valuable run of the campaign
- [x] `profile:=skidpad` — 20 m bubble; ≥6 m clear left AND right specifically
- [x] `profile:=step_steer` — 32 m bubble, both sides (alternating slalom)

---

## Session C — Bike path (~2 h)

Needs **60 m clear straight ahead** (nothing needed to the sides — all
straight-line). Every profile comes back to you on its own.

- [x] `profile:=pulse_staircase speed_slew_rate:=50.0`   ← ~40 m ahead, longest run
- [x] `profile:=coastdown speed_slew_rate:=50.0`   ← ~40 m ahead
- [x] `profile:=brake_sweep`   ← ~40 m ahead, thumb on the E-Stop, ESC can drive away backwards
- [x] `profile:=tune_profile`   ← ~38 m ahead, **no** `speed_slew_rate` override

### Gain sweep (repeat `tune_profile`, one number at a time)

```
ros2 launch cfr_arduino_bridge characterize.launch.py \
    profile:=tune_profile gains:="speed_kp=24.0 speed_ki=10.0" label:=kp24ki10
```

| kP  | kI  | Run label | Notes             |
| --- | --- | --------- | ----------------- |
| 16  | 10  | kp16ki10  | current default   |
| 24  | 10  | kp24ki10  | best of the sweep |
|     |     |           |                   |
|     |     |           |                   |

---

## Session A — Garage, no driving

Status as of 2026-09-14. Values are in `config/vehicle.yaml`.

- [x] A1 total mass **3.599** kg, front **1.698** rear **1.966**, left ______ right ______
      — L/R not weighed, assumed even (`cg_y` = 0, estimated). F+R = 3.664, 1.8% over total.
- [ ] A2 CG height — **block the shocks first**. Space: clear headroom behind
      the car ≥ H + wheelbase (~0.47 m) to tilt the rear up. H = **0.120** m,
      ΔW front = **37 g** (1698 → 1661)
      — `cg_z` ≈ 0.065 m, estimated. **Redo**: level and tilted back to back, same supports, bigger H.
- [ ] A3 inertia parts list — not started
- [x] A4 wheelbase **0.324** front track **0.290** rear track **0.290** ride height **0.050** (shocks free)
      — tire width and wheel/knuckle mass left as placeholders; Traxxas publishes neither.
- [x] A5 ten wheel revolutions = **3.556** m (140 in) → diameter = **0.1132** m
- [ ] A5 spur 28.5 turns → wheels ______ turns (expect 10) — deferred, 2.85 accepted from manual
- [ ] A6 overhead steering photos, −1 to +1 by 0.1, **both sweep directions**.
      Space: 1 m x 1 m level floor + overhead camera mount, car does not drive.
- [ ] A7 lock-to-lock slow-mo: ______ frames at 240 fps
- [ ] A8 suspension — confirm all 4 collars at **max preload**, sag ______ mm,
      bump ______ mm, droop ______ mm, added-mass spring-rate check, bounce
      decay on one corner at 240 fps. Space: clear access all 4 corners, ~0.3 m each side.

---

## If something says...

| Message                             | Do                                              |
| ----------------------------------- | ----------------------------------------------- |
| waiting for E-Stop **ASSERTED**     | press the button — it must see it pressed first |
| E-Stop clear but **auto not armed** | set Auto Arm on the controller                  |
| exceeded **max_distance**           | profile outran the space — shorten it           |
| gains **refused**                   | value out of firmware range; message names it   |

---

## Afterwards

```
./scripts/sync_runs.sh
./scripts/analyze_run.py runs/<run> --mass <kg from A1>
./scripts/apply_vehicle_patch.py runs/<run>
```
