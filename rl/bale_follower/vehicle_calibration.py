#!/usr/bin/env python3
"""Measure the simulated vehicle's real steering response, and patch the model.

Written after a week lost to a mis-modelled vehicle: every controller in this
project (PPO policies, CasADi smoother, MPC tracker) assumes the kinematic
bicycle `R = L/tan(delta)`, while the simulated Slash was turning at roughly
twice that radius and scrubbing off most of its commanded speed at large
steering angles. No amount of controller tuning fixes that.

Two modes:

    python vehicle_calibration.py measure            # sweep steering, report R(delta)
    python vehicle_calibration.py patch --out world.sdf   # write a corrected model

`measure` exists because the ad-hoc probes used during that investigation were
not trustworthy -- they timed motion across teleport transients and reported
speeds above the vehicle's 8 m/s ceiling. This one:

  * waits for the car to be genuinely STATIONARY after a teleport (a run of
    consecutive samples below a speed threshold) before it starts anything;
  * lets the turn reach steady state before the measurement window opens;
  * watches z and tilt, so a launched or flipped car is reported as such
    instead of being silently averaged into a radius;
  * rejects samples implying impossible speed, and reports how many it threw
    away, so a quiet failure cannot masquerade as data.

## Calibrating against the real car

Two measurements pin the model down. With the physical Slash:

  1. Set full steering lock; measure the front wheel angle against the chassis
     centerline (phone inclinometer on the wheel face works).
  2. Drive a full-lock circle at a slow steady speed; measure the diameter the
     tires trace.

Pass them in:

    python vehicle_calibration.py patch --full-lock-deg 25 --turning-circle-m 2.0

The first sets `<steering_limit>` and the steering joint limits. The second
gives the understeer factor -- measured radius divided by `L/tan(delta)` --
which is the number to aim the simulated vehicle at. Do NOT aim at the bare
kinematic radius: a real RC truck understeers, typically 1.2-1.4x, so a sim
that hits the ideal value exactly is wrong in the other direction.

Defaults come from published Traxxas specs and that understeer band, so the
script is runnable before the real measurements exist -- but the numbers it
writes are estimates until they are replaced by measured ones.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SDF = REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/speed_course.sdf"

# Published Traxxas Slash 4X4 figures.
WHEELBASE = 0.324          # m, confirmed against the SDF
TRACK_WIDTH = 0.296        # m (SDF carries 0.290)
VEHICLE_MASS = 2.28        # kg, 80.4 oz
BODY_LENGTH = 0.568
BODY_WIDTH = 0.296
# Traxxas 2075 servo: 0.17 s / 60 deg at 6 V, through a linkage that delivers
# roughly half to three-quarters of that at the road wheel.
SERVO_RATE = math.radians(60) / 0.17
LINKAGE_RATIO = 0.6
WHEEL_STEER_RATE = SERVO_RATE * LINKAGE_RATIO

DEFAULT_FULL_LOCK_DEG = 22.9   # the SDF's existing 0.40 rad
DEFAULT_UNDERSTEER = 1.3       # midpoint of the 1.2-1.4 band


def expected_radius(delta_rad: float, understeer: float = DEFAULT_UNDERSTEER) -> float:
    return WHEELBASE / math.tan(delta_rad) * understeer


# ---------------------------------------------------------------- measurement

def measure(args) -> None:
    import numpy as np
    import rclpy
    import requests
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from tf2_msgs.msg import TFMessage

    from env import _yaw_from_quaternion

    class Probe(Node):
        def __init__(self) -> None:
            super().__init__("vehicle_calibration")
            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.samples: list[tuple[float, float, float, float, float, float]] = []
            self.create_subscription(
                TFMessage, f"/world/{args.world_name}/dynamic_pose/info", self._on_pose, 20
            )

        def _on_pose(self, msg: TFMessage) -> None:
            if not msg.transforms:
                return
            t = msg.transforms[0].transform
            q = t.rotation
            # Roll/pitch catch a flipped or airborne car; z catches a launch.
            roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
            pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
            self.samples.append((
                time.monotonic(), t.translation.x, t.translation.y, t.translation.z,
                _yaw_from_quaternion(q.x, q.y, q.z, q.w), max(abs(roll), abs(pitch)),
            ))

        def spin(self, seconds: float) -> None:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.005)

        def latest(self):
            return self.samples[-1] if self.samples else None

        def current_speed(self, signed: bool = False) -> float:
            """Speed over a ~0.2 s baseline, long enough not to amplify noise.

            `signed` projects onto the heading, which braking needs: comparing
            an unsigned speed against a constant reverse command means the
            car settles at exactly the reverse speed and the loop never ends.
            """
            now = time.monotonic()
            tail = [s for s in self.samples if now - s[0] <= 0.25]
            if len(tail) < 4:
                return float("inf")
            span = tail[-1][0] - tail[0][0]
            if span < 0.05:
                return float("inf")
            dx = tail[-1][1] - tail[0][1]
            dy = tail[-1][2] - tail[0][2]
            if signed:
                yaw = tail[-1][4]
                return (dx * math.cos(yaw) + dy * math.sin(yaw)) / span
            return math.hypot(dx, dy) / span

        def brake(self, timeout: float = 6.0) -> None:
            """Actively brake to a stop.

            Neutral throttle is modelled as a coast at 0.1 m/s^2, so releasing
            the throttle leaves the car rolling for 15 s from test speed --
            far longer than any sane settling timeout. Drive it down instead.
            """
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                signed = self.current_speed(signed=True)
                if not math.isfinite(signed):
                    self.pub.publish(Twist())
                    self.spin(0.05)
                    continue
                if abs(signed) < 0.12:
                    break
                # Oppose the direction of travel, scaled to what is left, so
                # the command decays to zero instead of driving the other way.
                twist = Twist()
                twist.linear.x = -math.copysign(min(0.5, abs(signed) * 0.6), signed)
                self.pub.publish(twist)
                self.spin(0.05)
            for _ in range(15):
                self.pub.publish(Twist())
                self.spin(0.02)

        def wait_until_stationary(self, timeout: float = 6.0) -> bool:
            """Hold neutral until several consecutive samples show no motion.

            Without this the measurement window can open on the tail of a
            teleport, which is how earlier probes produced 13 m/s readings on
            a vehicle capped at 8 m/s.
            """
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                self.pub.publish(Twist())
                self.spin(0.05)
                # Time-window, not sample-count: the pose stream runs near
                # 60 Hz, so a fixed count spans far less time than intended.
                # Judge by NET displacement, not per-sample jitter: the resting
                # vehicle buzzes about 4 mm per sample against the ground
                # contact while going nowhere, which is above any sane
                # per-sample threshold but nets out to zero.
                speed = self.current_speed()
                if math.isfinite(speed):
                    self._last_residual = speed
                    if speed < 0.08:
                        return True
            return False

    rclpy.init()
    probe = Probe()
    print(f"{'delta':>7} {'R_ideal':>8} {'R_meas':>8} {'ratio':>6} {'v':>6} {'rejects':>8}  status")
    rows = []
    try:
        for delta_deg in args.angles:
            delta = math.radians(delta_deg)
            probe.brake()
            requests.post(args.teleport_url,
                          json={"x": args.x, "y": args.y, "heading": 0}, timeout=5)
            probe.spin(0.4)
            if not probe.wait_until_stationary():
                residual = getattr(probe, "_last_residual", float("nan"))
                speed = probe.current_speed()
                pos = probe.latest()
                print(f"{delta_deg:7.1f} {'':>8} {'':>8} {'':>6} {'':>6} {'':>8}  "
                      f"NOT SETTLED (net {residual:.2f} m/s, "
                      f"speed {speed:.2f} m/s, at {pos[1]:.1f},{pos[2]:.1f})")
                continue

            twist = Twist()
            twist.linear.x = args.speed
            twist.angular.z = (args.speed / WHEELBASE) * math.tan(delta)
            # Settle: let the steering reach the commanded angle and the speed
            # reach steady state before the window opens.
            end = time.monotonic() + args.settle
            while time.monotonic() < end:
                probe.pub.publish(twist)
                probe.spin(0.02)

            probe.samples.clear()
            end = time.monotonic() + args.window
            while time.monotonic() < end:
                probe.pub.publish(twist)
                probe.spin(0.02)
            probe.pub.publish(Twist())

            window = probe.samples[:]
            if len(window) < 8:
                print(f"{delta_deg:7.1f} {'':>8} {'':>8} {'':>6} {'':>6} {'':>8}  NO DATA")
                continue

            # Per-interval speeds; drop anything physically impossible rather
            # than letting it into the average.
            # Difference over a ~0.1 s stride rather than adjacent samples:
            # at 60 Hz, millimetre pose jitter across a 16 ms gap reads as
            # metres per second of phantom speed.
            stride = max(1, int(0.1 / max(1e-3, (window[-1][0] - window[0][0]) / len(window))))
            speeds, yaw_rates, rejects = [], [], 0
            for a, b in zip(window, window[stride:]):
                dt = b[0] - a[0]
                if dt <= 1e-3:
                    continue
                v = math.hypot(b[1] - a[1], b[2] - a[2]) / dt
                dyaw = math.atan2(math.sin(b[4] - a[4]), math.cos(b[4] - a[4])) / dt
                if v > args.max_plausible_speed:
                    rejects += 1
                    continue
                speeds.append(v)
                yaw_rates.append(dyaw)

            max_z = max(s[3] for s in window)
            max_tilt = max(s[5] for s in window)
            status = "ok"
            if max_z > 0.25:
                status = f"AIRBORNE z={max_z:.2f}"
            elif max_tilt > 0.5:
                status = f"TILTED {math.degrees(max_tilt):.0f}deg"
            elif rejects > len(window) * 0.25:
                status = "UNRELIABLE"

            if not speeds:
                print(f"{delta_deg:7.1f} {'':>8} {'':>8} {'':>6} {'':>6} {rejects:8d}  ALL REJECTED")
                continue
            v_mean = statistics.fmean(speeds)
            yaw_mean = abs(statistics.fmean(yaw_rates))
            r_ideal = WHEELBASE / math.tan(delta)
            r_meas = v_mean / yaw_mean if yaw_mean > 1e-3 else float("inf")
            ratio = r_meas / r_ideal if math.isfinite(r_meas) else float("inf")
            rows.append((delta_deg, r_ideal, r_meas, ratio, v_mean, status))
            print(f"{delta_deg:7.1f} {r_ideal:8.2f} {r_meas:8.2f} {ratio:6.2f} "
                  f"{v_mean:6.2f} {rejects:8d}  {status}")
    finally:
        probe.pub.publish(Twist())
        probe.destroy_node()
        rclpy.shutdown()

    clean = [r for r in rows if r[5] == "ok" and math.isfinite(r[3])]
    if clean:
        mean_ratio = statistics.fmean(r[3] for r in clean)
        full = min(clean, key=lambda r: abs(r[0] - max(args.angles)))
        print(f"\nundersteer ratio (measured / kinematic): {mean_ratio:.2f} "
              f"over {len(clean)} clean points")
        print(f"full-lock radius: {full[2]:.2f} m   target for a real Slash: "
              f"{expected_radius(math.radians(full[0])):.2f} m "
              f"(=1.2-1.4x kinematic {full[1]:.2f} m)")
        print(f"hairpin apex needs 1.30 m -> "
              f"{'DRIVABLE' if full[2] < 1.25 else 'NOT DRIVABLE in one sweep'}")
    else:
        print("\nno clean measurements -- fix the flagged failures before trusting any number")


# --------------------------------------------------------------------- patch

def patch(args) -> None:
    source = Path(args.sdf_path).read_text()
    patched = source
    notes = []

    full_lock = math.radians(args.full_lock_deg)

    # 1. Wheel friction. The stock model carries mu=50 (about 50x rubber) with
    #    fdir1 pointing straight up -- a friction direction perpendicular to the
    #    contact patch is meaningless for a wheel, and leaves the solver
    #    resolving near-infinite grip against tire scrub. mu is longitudinal
    #    (rolling) grip, mu2 lateral; omitting fdir1 lets the solver pick its
    #    own contact-tangent basis.
    old_wheel = "<ode><mu>50</mu><mu2>1</mu2><fdir1>0 0 1</fdir1></ode>"
    new_wheel = f"<ode><mu>{args.mu}</mu><mu2>{args.mu2}</mu2></ode>"
    if old_wheel in patched:
        patched = patched.replace(old_wheel, new_wheel)
        notes.append(f"wheel friction -> mu={args.mu} mu2={args.mu2}, bogus fdir1 removed")

    old_ground = "<ode><mu>50</mu></ode>"
    if old_ground in patched:
        patched = patched.replace(old_ground, f"<ode><mu>{args.mu}</mu></ode>")
        notes.append(f"ground friction -> mu={args.mu}")

    # 2. Mass and inertia. Real Slash 4X4 is 2.28 kg; the model totals 4.08 kg.
    #    Mass does not change the friction-limited radius (lateral accel is
    #    mu*g either way) but it does change contact resolution and how ESC
    #    torque becomes acceleration.
    chassis_mass = round(VEHICLE_MASS - 4 * 0.12 - 2 * 0.05, 2)
    box_l, box_w, box_h = 0.55, 0.30, 0.12
    ixx = round(chassis_mass * (box_w**2 + box_h**2) / 12, 4)
    iyy = round(chassis_mass * (box_l**2 + box_h**2) / 12, 4)
    izz = round(chassis_mass * (box_l**2 + box_w**2) / 12, 4)
    old_inertial = ("<inertial><mass>3.5</mass><inertia><ixx>0.08</ixx>"
                    "<iyy>0.12</iyy><izz>0.16</izz></inertia></inertial>")
    new_inertial = (f"<inertial><mass>{chassis_mass}</mass><inertia><ixx>{ixx}</ixx>"
                    f"<iyy>{iyy}</iyy><izz>{izz}</izz></inertia></inertial>")
    if old_inertial in patched:
        patched = patched.replace(old_inertial, new_inertial)
        notes.append(f"chassis mass 3.5 -> {chassis_mass} kg (total {VEHICLE_MASS} kg), "
                     f"inertia recomputed for the box")

    # 3. Drive joints. AckermannSteering applies one velocity per side, but the
    #    front wheels are also the steered wheels and need a different rotation
    #    rate than the rears -- listing both makes them fight each other. Drive
    #    the rears only; the plugin still steers the fronts.
    if args.rear_wheel_drive:
        for side in ("left", "right"):
            both = (f"<{side}_joint>front_{side}_wheel_joint</{side}_joint>"
                    f"<{side}_joint>rear_{side}_wheel_joint</{side}_joint>")
            if both in patched:
                patched = patched.replace(
                    both, f"<{side}_joint>rear_{side}_wheel_joint</{side}_joint>")
                notes.append(f"{side}: drive rear joint only (was front+rear, conflicting)")

    # 4. Steering limit, from the measured full-lock angle.
    if abs(args.full_lock_deg - DEFAULT_FULL_LOCK_DEG) > 0.05:
        limit = round(full_lock, 3)
        patched = patched.replace("<steering_limit>0.40</steering_limit>",
                                  f"<steering_limit>{limit}</steering_limit>")
        patched = patched.replace("<lower>-0.40</lower><upper>0.40</upper>",
                                  f"<lower>-{limit}</lower><upper>{limit}</upper>")
        notes.append(f"steering limit 0.40 -> {limit} rad ({args.full_lock_deg} deg measured)")

    # 5. Track width, published 296 mm against the model's 290 mm.
    if args.fix_track_width:
        if "<wheel_separation>0.290</wheel_separation>" in patched:
            patched = patched.replace("<wheel_separation>0.290</wheel_separation>",
                                      f"<wheel_separation>{TRACK_WIDTH}</wheel_separation>")
            notes.append(f"wheel separation 0.290 -> {TRACK_WIDTH} m (published track)")

    out = Path(args.out)
    out.write_text(patched)
    print(f"wrote {out}")
    for note in notes:
        print(f"  - {note}")
    if not notes:
        print("  (nothing matched -- is this SDF already patched?)")

    print("\nValidate before adopting. This writes a candidate world, never the")
    print("committed one, because an earlier friction-only patch fixed the")
    print("turning radius and destabilized the vehicle at the same time:")
    print(f"  ros2 launch cfr_arduino_bridge training.launch.py world:={out}")
    print(f"  python vehicle_calibration.py measure")
    print(f"Adopt only if every row reads 'ok' AND the understeer ratio lands "
          f"in 1.2-1.4 (full-lock R around {expected_radius(full_lock):.2f} m).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="sweep steering angles against a running sim")
    m.add_argument("--angles", type=float, nargs="+",
                   default=[10, 15, 20, 22.9], help="steering angles in degrees")
    m.add_argument("--speed", type=float, default=1.5, help="test speed (m/s)")
    m.add_argument("--settle", type=float, default=2.0, help="settling time before measuring (s)")
    m.add_argument("--window", type=float, default=2.5, help="measurement window (s)")
    m.add_argument("--x", type=float, default=20.0, help="open-ground test x")
    m.add_argument("--y", type=float, default=-11.0, help="open-ground test y (outside the course)")
    m.add_argument("--world-name", default="cfr_speed_course")
    m.add_argument("--teleport-url", default="http://localhost:9003/api/sim/teleport")
    m.add_argument("--max-plausible-speed", type=float, default=8.0,
                   help="reject samples implying more than this (m/s)")
    m.set_defaults(func=measure)

    p = sub.add_parser("patch", help="write a corrected copy of the world")
    p.add_argument("--sdf-path", default=str(DEFAULT_SDF))
    p.add_argument("--out", default="/tmp/speed_course_calibrated.sdf")
    p.add_argument("--full-lock-deg", type=float, default=DEFAULT_FULL_LOCK_DEG,
                   help="MEASURED front wheel angle at full lock (degrees)")
    p.add_argument("--turning-circle-m", type=float, default=None,
                   help="MEASURED full-lock circle diameter (m); reported as an understeer factor")
    p.add_argument("--mu", type=float, default=1.2, help="longitudinal tire friction")
    p.add_argument("--mu2", type=float, default=1.0, help="lateral tire friction")
    # Default is 4WD: the Slash 4X4 Ultimate is four-wheel drive, and once the
    # friction is sane it measures as well as rear-drive (full lock 0.97 m vs
    # 0.95 m, understeer 1.14 vs 1.09 -- the 4WD figure is actually closer to
    # the realistic band). The front/rear joint conflict was a real modelling
    # error but not the dominant one; mu=50 was.
    p.add_argument("--rear-wheel-drive", action="store_true", default=False,
                   help="drive only the rear joints (a 2WD Slash, or to isolate "
                        "the plugin's one-velocity-per-side drive conflict)")
    p.add_argument("--fix-track-width", action="store_true", default=True)
    p.set_defaults(func=patch)

    args = parser.parse_args()
    if getattr(args, "turning_circle_m", None):
        measured_r = args.turning_circle_m / 2
        ideal = WHEELBASE / math.tan(math.radians(args.full_lock_deg))
        print(f"measured full-lock radius {measured_r:.2f} m vs kinematic {ideal:.2f} m "
              f"-> understeer factor {measured_r / ideal:.2f}\n")
    args.func(args)


if __name__ == "__main__":
    main()
