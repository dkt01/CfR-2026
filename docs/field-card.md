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

| Run | Profile | Volts before | Volts after | Result | Notes |
| --- | ------- | ------------ | ----------- | ------ | ----- |
|  1  |         |              |             |        |       |
|  2  |         |              |             |        |       |
|  3  |         |              |             |        |       |
|  4  |         |              |             |        |       |
|  5  |         |              |             |        |       |
|  6  |         |              |             |        |       |
|  7  |         |              |             |        |       |
|  8  |         |              |             |        |       |

---

## Session B — Parking lot (~1.5 h)

Car stays in one place. Needs roughly a 20 m clear square.

- [ ] `profile:=zed_static` — 5 min, **leave E-Stop asserted**, car will not move
- [ ] `profile:=steer_authority` — the most valuable run of the campaign
- [ ] `profile:=skidpad`
- [ ] `profile:=step_steer`

---

## Session C — Bike path (~2 h)

Needs **60 m clear straight**. Every profile comes back to you on its own.

- [ ] `profile:=pulse_staircase speed_slew_rate:=50.0`   ← longest run, ~40 m out
- [ ] `profile:=coastdown speed_slew_rate:=50.0`
- [ ] `profile:=brake_sweep`   ← thumb on the E-Stop, ESC can drive away backwards
- [ ] `profile:=tune_profile`   ← **no** `speed_slew_rate` override

### Gain sweep (repeat `tune_profile`, one number at a time)

```
ros2 launch cfr_arduino_bridge characterize.launch.py \
    profile:=tune_profile gains:="speed_kp=24.0 speed_ki=10.0" label:=kp24ki10
```

| kP | kI | Run label | Notes |
| -- | -- | --------- | ----- |
| 16 | 10 |           | current default |
|    |    |           |       |
|    |    |           |       |
|    |    |           |       |

---

## Session A — Garage, no driving

- [ ] A1 total mass ______ kg, front ______ rear ______, left ______ right ______
- [ ] A2 CG height — **block the shocks first**. H = ______ m, ΔW front = ______
- [ ] A4 wheelbase ______ front track ______ rear track ______ ride height ______
- [ ] A5 ten wheel revolutions = ______ m  → diameter = dist / 31.416
- [ ] A5 spur 28.5 turns → wheels ______ turns (expect 10)
- [ ] A6 overhead steering photos, −1 to +1 by 0.1, **both sweep directions**
- [ ] A7 lock-to-lock slow-mo: ______ frames at 240 fps

---

## If something says...

| Message | Do |
| --- | --- |
| waiting for E-Stop **ASSERTED** | press the button — it must see it pressed first |
| E-Stop clear but **auto not armed** | set Auto Arm on the controller |
| exceeded **max_distance** | profile outran the space — shorten it |
| gains **refused** | value out of firmware range; message names it |

---

## Afterwards

```
./scripts/sync_runs.sh
./scripts/analyze_run.py runs/<run> --mass <kg from A1>
./scripts/apply_vehicle_patch.py runs/<run>
```
