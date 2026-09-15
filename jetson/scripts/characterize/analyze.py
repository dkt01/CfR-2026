"""Turn a run directory into fitted numbers and a report.

One analyzer per profile.  Each returns a dict with:

    'summary'  human-readable lines for the report
    'vehicle'  the vehicle.yaml entries it justifies, as dotted paths
    'plots'    SVG files written next to the run

The `vehicle` block is the point: the output of a run is not a plot, it is a
set of numbers with a provenance trail back to the run that produced them.
"""

import math
import os

from . import fits, svgplot
from .linalg import fit_line

__all__ = ["analyze", "ANALYZERS"]

# Kept in step with drivetrain.hpp; overridden from the run's own parameters
# wherever those were recorded, so an old run analyses against what it actually
# ran with rather than against today's configuration.
DEFAULT_WHEELBASE = 0.324


def _wheelbase(run):
    return float(run.metadata.get("wheelbase") or DEFAULT_WHEELBASE)


def _mass(run, mass):
    if mass is not None:
        return mass
    raise ValueError(
        "this fit needs the vehicle mass: pass --mass, or measure it (A1) and "
        "put it in config/vehicle.yaml"
    )


def _speed_column(segment):
    """Ground speed for a segment, preferring odometry over wheel RPM.

    Odometry is the better reference: the RPM-derived speed assumes no wheel
    slip and a tire diameter that is itself a guess until A5 is done.  Falling
    back to it keeps an analysis working when odometry dropped out.
    """
    times, speeds = segment.pair("t_ros", "odom_vx")
    if len(speeds) >= 8:
        return times, speeds, "odometry"
    times, speeds = segment.pair("t_ros", "speed")
    return times, speeds, "wheel rpm"


# --------------------------------------------------------------------- B0


def analyze_zed_static(run, options):
    rows = [
        row
        for row in run.rows
        if row.get("odom_x") is not None and row.get("odom_valid")
    ]
    if len(rows) < 100:
        return {
            "summary": [f"only {len(rows)} odometry samples - nothing to fit"],
            "vehicle": {},
            "plots": [],
        }

    noise = fits.odometry_noise(
        [row["t_ros"] for row in rows],
        [row["odom_x"] for row in rows],
        [row["odom_y"] for row in rows],
        [row["odom_yaw"] for row in rows],
    )

    start = rows[0]["t_ros"]
    plot = svgplot.scatter(
        os.path.join(run.path, "plot_odom_static.svg"),
        [
            {
                "label": "yaw (rad)",
                "mode": "line",
                "x": [row["t_ros"] - start for row in rows],
                "y": [row["odom_yaw"] for row in rows],
            }
        ],
        title="Stationary ZED yaw",
        x_label="time (s)",
        y_label="yaw (rad)",
    )

    deadband = 0.01  # path_follower's heading_deadband
    verdict = "within" if noise["yaw_noise"] < deadband else "ABOVE"
    return {
        "summary": [
            f"duration {noise['duration']:.0f} s at {noise['rate_hz']:.1f} Hz, "
            f"{noise['samples']} samples",
            f"yaw noise {noise['yaw_noise']:.5f} rad "
            f"({math.degrees(noise['yaw_noise']):.3f} deg) - {verdict} the "
            f"0.01 rad heading_deadband",
            f"yaw drift {noise['yaw_drift_rate']:.6f} rad/s "
            f"({math.degrees(noise['yaw_drift_rate']) * 60:.2f} deg/min)",
            f"position noise {noise['position_noise']:.4f} m, "
            f"drift {noise['position_drift_rate']:.4f} m/s",
            f"total yaw excursion {noise['total_yaw_excursion']:.4f} rad over the window",
            f"dropouts {noise['dropouts']} ({noise['dropout_fraction'] * 100:.2f}% of intervals)",
        ],
        "vehicle": {
            "sensors.odom_rate": noise["rate_hz"],
            "sensors.odom_yaw_noise": noise["yaw_noise"],
            "sensors.odom_position_noise": noise["position_noise"],
            "sensors.odom_yaw_drift": noise["yaw_drift_rate"],
            "sensors.odom_dropout_rate": noise["dropout_fraction"],
        },
        "plots": [plot],
    }


# --------------------------------------------------------------------- B1


def analyze_steer_authority(run, options):
    wheelbase = _wheelbase(run)
    summary, points = [], []
    for segment in run.segments():
        if segment.label in (None, "settle", "between", "straight_reference"):
            continue
        command = fits.steady_state(*segment.pair("t_ros", "cmd_steering"))["mean"]
        speed = fits.steady_state(*segment.pair("t_ros", "odom_vx"))
        yaw_rate = fits.steady_state(*segment.pair("t_ros", "odom_wz"))
        if abs(speed["mean"]) < 0.05:
            summary.append(f"{segment.label}: car never moved, skipped")
            continue
        delta = fits.fit_effective_steering(speed["mean"], yaw_rate["mean"], wheelbase)
        radius = (
            abs(speed["mean"] / yaw_rate["mean"])
            if abs(yaw_rate["mean"]) > 1e-6
            else float("inf")
        )
        points.append((command, delta))
        summary.append(
            f"{segment.label:14s} cmd {command:+.2f}  v {speed['mean']:.2f} m/s  "
            f"yaw {yaw_rate['mean']:+.3f} rad/s  R {radius:5.2f} m  "
            f"delta {delta:+.4f} rad ({math.degrees(delta):+.2f} deg)"
        )

    if len(points) < 2:
        return {"summary": summary or ["no usable arcs"], "vehicle": {}, "plots": []}

    points.sort()
    left = [delta for command, delta in points if command > 0.9]
    right = [delta for command, delta in points if command < -0.9]
    slope, intercept, r_squared = fit_line(
        [c for c, _ in points], [d for _, d in points]
    )

    vehicle = {
        "steering.effective_angle_table": [
            [round(c, 3), round(d, 5)] for c, d in points
        ],
    }
    if left:
        vehicle["steering.max_angle_left"] = abs(sum(left) / len(left))
    if right:
        vehicle["steering.max_angle_right"] = abs(sum(right) / len(right))
    if left or right:
        limit = max([abs(value) for value in left + right])
        vehicle["lateral.min_turn_radius"] = wheelbase / math.tan(limit)
    # Where the fitted line crosses zero angle is the command that actually
    # points the wheels straight, which is not necessarily zero.
    if abs(slope) > 1e-9:
        vehicle["steering.center_offset"] = -intercept / slope

    summary.append("")
    summary.append(
        f"linear fit: delta = {slope:.4f} * command {intercept:+.5f}  (r2 {r_squared:.4f})"
    )
    summary.append(
        f"the configured max_steering_angle is 0.40 rad; measured full lock is "
        f"{vehicle.get('steering.max_angle_left', float('nan')):.4f} left / "
        f"{vehicle.get('steering.max_angle_right', float('nan')):.4f} right"
    )
    if "lateral.min_turn_radius" in vehicle:
        summary.append(
            f"minimum turn radius {vehicle['lateral.min_turn_radius']:.3f} m"
        )

    plot = svgplot.scatter(
        os.path.join(run.path, "plot_steering_map.svg"),
        [
            {
                "label": "measured",
                "mode": "points",
                "x": [c for c, _ in points],
                "y": [d for _, d in points],
            },
            {
                "label": "assumed (0.40 rad linear)",
                "mode": "line",
                "x": [-1.0, 1.0],
                "y": [-0.40, 0.40],
            },
        ],
        title="Effective steering angle vs command",
        x_label="normalized steering command",
        y_label="delta_eff (rad)",
    )

    return {"summary": summary, "vehicle": vehicle, "plots": [plot]}


# --------------------------------------------------------------------- B2


def analyze_skidpad(run, options):
    wheelbase = _wheelbase(run)
    summary, groups = [], {"left": [], "right": []}
    for segment in run.segments():
        if not segment.label or not segment.label.startswith(("left_", "right_")):
            continue
        side = segment.label.split("_")[0]
        speed = fits.steady_state(*segment.pair("t_ros", "odom_vx"))["mean"]
        yaw_rate = fits.steady_state(*segment.pair("t_ros", "odom_wz"))["mean"]
        if abs(speed) < 0.2 or abs(yaw_rate) < 1e-3:
            continue
        radius = abs(speed / yaw_rate)
        lateral = speed * speed / radius
        delta = abs(fits.fit_effective_steering(speed, yaw_rate, wheelbase))
        groups[side].append((lateral, delta))
        summary.append(
            f"{segment.label:14s} v {speed:.2f}  R {radius:5.2f} m  "
            f"a_y {lateral:5.2f} m/s2  delta {delta:.4f} rad"
        )

    vehicle, series = {}, []
    for side, points in groups.items():
        if len(points) < 3:
            continue
        points.sort()
        result = fits.fit_understeer([a for a, _ in points], [d for _, d in points])
        summary.append("")
        summary.append(
            f"{side}: understeer gradient {result['understeer_gradient']:+.5f} "
            f"rad/(m/s2), kinematic angle {result['kinematic_angle']:.4f} rad "
            f"(r2 {result['r_squared']:.3f})"
        )
        vehicle[f"lateral.understeer_gradient_{side}"] = result["understeer_gradient"]
        series.append(
            {
                "label": side,
                "mode": "points",
                "x": [a for a, _ in points],
                "y": [d for _, d in points],
            }
        )
        # The simulator, with mu=50, always achieves the kinematic radius. This
        # is the size of that error at the speed the follower actually cruises.
        excess = result["understeer_gradient"] * (3.2**2 / max(points[-1][0], 1e-6))
        if result["understeer_gradient"] > 0:
            summary.append(
                f"  -> at 3.2 m/s the car needs roughly {excess:.4f} rad more "
                f"steering than the kinematic model predicts"
            )

    both = [value for side in groups.values() for value in side]
    if both:
        peak = max(a for a, _ in both)
        vehicle["lateral.mu_lateral"] = peak / 9.81
        summary.append("")
        summary.append(
            f"peak lateral acceleration reached {peak:.2f} m/s2 "
            f"= {peak / 9.81:.2f} g, a LOWER BOUND on lateral mu "
            f"(the sweep stayed below the slide limit by design)"
        )

    plot = (
        svgplot.scatter(
            os.path.join(run.path, "plot_skidpad.svg"),
            series,
            title="Steering angle vs lateral acceleration",
            x_label="a_y (m/s^2)",
            y_label="delta_eff (rad)",
        )
        if series
        else None
    )

    return {"summary": summary, "vehicle": vehicle, "plots": [plot] if plot else []}


# --------------------------------------------------------------------- B3


def analyze_step_steer(run, options):
    summary, vehicle = [], {}
    taus = []
    for segment in run.matching("step_"):
        times, yaw_rates = segment.pair("t_ros", "odom_wz")
        if len(times) < 8:
            continue
        settled = fits.steady_state(times, yaw_rates, tail_fraction=0.4)
        try:
            fit = fits.fit_first_order(
                times, yaw_rates, initial=yaw_rates[0], final=settled["mean"]
            )
        except ValueError as error:
            summary.append(f"{segment.label:20s} no usable step: {error}")
            continue
        peak = max(yaw_rates, key=abs)
        overshoot = (
            (abs(peak) - abs(settled["mean"])) / abs(settled["mean"])
            if settled["mean"]
            else 0.0
        )
        taus.append(fit["tau"])
        summary.append(
            f"{segment.label:20s} settled {settled['mean']:+.3f} rad/s  "
            f"tau {fit['tau']:.3f} s  overshoot {overshoot * 100:+.1f}%"
        )

    if taus:
        mean_tau = sum(taus) / len(taus)
        summary.append("")
        summary.append(
            f"mean yaw-rate time constant {mean_tau:.3f} s over {len(taus)} steps"
        )
        summary.append(
            "Compare against the simulator running the same profile. A large "
            "mismatch points at the inertia ESTIMATE in vehicle.yaml (A3), "
            "which was calculated rather than measured."
        )
        vehicle["inertia.yaw_response_tau_measured"] = mean_tau
    return {"summary": summary, "vehicle": vehicle, "plots": []}


# --------------------------------------------------------------------- C1


def analyze_pulse_staircase(run, options):
    summary, forward, reverse = [], [], []
    for segment in run.segments():
        label = segment.label or ""
        if not (label.startswith("fwd_ks") or label.startswith("rev_ks")):
            continue
        times, speeds, source = _speed_column(segment)
        if len(speeds) < 8:
            continue
        settled = fits.steady_state(times, speeds)
        pulse = fits.steady_state(*segment.pair("t_ros", "throttle_us"))["mean"]
        rpm = fits.steady_state(*segment.pair("t_ros", "speed"))
        offset = pulse - 1500.0
        entry = (abs(offset), abs(settled["mean"]))
        (forward if label.startswith("fwd") else reverse).append(entry)
        summary.append(
            f"{label:12s} pulse {pulse:7.1f} us ({offset:+6.1f})  "
            f"speed {settled['mean']:+.3f} m/s (+/-{settled['std']:.3f}, {source})  "
            f"rpm-derived {rpm['mean']:+.3f} m/s"
        )

    vehicle = {}
    for name, points in (("forward", forward), ("reverse", reverse)):
        moving = [(offset, speed) for offset, speed in points if speed > 0.15]
        if len(moving) < 3:
            continue
        slope, intercept, r_squared = fit_line(
            [o for o, _ in moving], [s for _, s in moving]
        )
        # Where the fitted line reaches zero speed is the pulse offset the car
        # needs before it moves at all: kS, in the controller's own units.
        ks = -intercept / slope if abs(slope) > 1e-9 else float("nan")
        summary.append("")
        summary.append(f"{name}: {slope:.5f} (m/s) per us, r2 {r_squared:.4f}")
        summary.append(
            f"  -> static feedforward kS ~ {ks:.1f} us "
            f"(the pulse offset at which the car starts to move)"
        )
        summary.append(f"  -> velocity feedforward kV ~ {1.0 / slope:.2f} us per m/s")
        if name == "forward":
            vehicle["longitudinal.ks_from_fit"] = ks
            vehicle["longitudinal.kv_from_fit"] = 1.0 / slope

    # The two speed estimates should agree once the tire radius is right; where
    # they do not, one of A5 (rolling radius) or the ZED scale is wrong.
    both = [
        (row["speed"], row["odom_vx"])
        for row in run.rows
        if row.get("speed") and row.get("odom_vx") and abs(row["odom_vx"]) > 0.5
    ]
    if len(both) > 50:
        ratio = sum(abs(r) / abs(o) for r, o in both) / len(both)
        summary.append("")
        summary.append(
            f"RPM-derived speed / odometry speed = {ratio:.4f} over {len(both)} samples"
        )
        summary.append(
            "  1.000 means the tire diameter and the ZED scale agree. A consistent "
            "offset means one of them is wrong - A5 pins the tire, so suspect the "
            "ZED scale first."
        )
        vehicle["sensors.rpm_to_odom_speed_ratio"] = ratio

    series = [
        {
            "label": name,
            "mode": "points",
            "x": [o for o, _ in points],
            "y": [s for _, s in points],
        }
        for name, points in (("forward", forward), ("reverse", reverse))
        if points
    ]
    plot = (
        svgplot.scatter(
            os.path.join(run.path, "plot_pulse_map.svg"),
            series,
            title="Steady speed vs throttle pulse offset",
            x_label="|pulse - 1500| (us)",
            y_label="|speed| (m/s)",
        )
        if series
        else None
    )

    return {"summary": summary, "vehicle": vehicle, "plots": [plot] if plot else []}


# --------------------------------------------------------------------- C2


def analyze_coastdown(run, options):
    mass = _mass(run, options.get("mass"))
    summary, results, series = [], [], []
    for segment in run.matching("coast_"):
        times, speeds, source = _speed_column(segment)
        if len(speeds) < 20:
            summary.append(f"{segment.label}: too few samples")
            continue
        start = times[0]
        try:
            fit = fits.fit_resistance(times, speeds, mass)
        except ValueError as error:
            summary.append(f"{segment.label:20s} {error}")
            continue
        results.append((segment.label, fit))
        decel = (fit["f0"] + fit["f1"] * 2.0 + fit["f2"] * 4.0) / mass
        summary.append(
            f"{segment.label:20s} f0 {fit['f0']:6.3f} N  f1 {fit['f1']:+.4f}  "
            f"f2 {fit['f2']:+.5f}  ({fit['points']} pts, rms {fit['rms_newtons']:.3f} N, "
            f"{source})  decel at 2 m/s = {decel:.3f} m/s2"
        )
        series.append(
            {
                "label": segment.label,
                "mode": "line",
                "x": [time - start for time in times],
                "y": [abs(s) for s in speeds],
            }
        )

    vehicle = {}
    if results:
        # Average across speeds and directions: each individual roll-down is
        # short and the three terms trade off against each other in any one fit.
        for key in ("f0", "f1", "f2"):
            vehicle[
                f"longitudinal.{key}_"
                + {"f0": "rolling", "f1": "viscous", "f2": "aero"}[key]
            ] = sum(fit[key] for _, fit in results) / len(results)

        forward = [fit for label, fit in results if "fwd" in label]
        reverse = [fit for label, fit in results if "rev" in label]
        if forward and reverse:
            f_mean = sum(fit["f0"] for fit in forward) / len(forward)
            r_mean = sum(fit["f0"] for fit in reverse) / len(reverse)
            asymmetry = abs(f_mean - r_mean) / max(abs(f_mean), abs(r_mean), 1e-9)
            summary.append("")
            summary.append(
                f"forward f0 {f_mean:.3f} N vs reverse f0 {r_mean:.3f} N "
                f"({asymmetry * 100:.1f}% apart)"
            )
            if asymmetry < 0.15:
                summary.append(
                    "  -> SYMMETRIC within 15%. The separate forward/reverse coast "
                    "decelerations in arduino_bridge.yaml model the simulator, not "
                    "the car, and should be replaced by these coefficients."
                )
            else:
                summary.append(
                    "  -> genuinely asymmetric; the simulator needs to reproduce it "
                    "from physics rather than from a command-side constant."
                )

    plot = (
        svgplot.scatter(
            os.path.join(run.path, "plot_coastdown.svg"),
            series,
            title="Coastdown",
            x_label="time into coast (s)",
            y_label="|speed| (m/s)",
        )
        if series
        else None
    )

    return {"summary": summary, "vehicle": vehicle, "plots": [plot] if plot else []}


# --------------------------------------------------------------------- C3


def analyze_brake_sweep(run, options):
    summary, points = [], []
    for segment in run.matching("stop_brake"):
        times, speeds, _ = _speed_column(segment)
        if len(speeds) < 10:
            continue
        limit = float(segment.label.replace("stop_brake", ""))
        start_speed = abs(speeds[0])
        stopped = next(
            (time for time, speed in zip(times, speeds) if abs(speed) < 0.2), None
        )
        if stopped is None:
            summary.append(
                f"brake {limit:5.0f} us: never reached a stop inside the hold"
            )
            continue
        duration = stopped - times[0]
        distance = abs(segment.rows[-1]["dist_along"] - segment.rows[0]["dist_along"])
        decel = start_speed / duration if duration > 0 else 0.0
        points.append((limit, decel))
        summary.append(
            f"brake {limit:5.0f} us: {start_speed:.2f} m/s to stop in "
            f"{duration:5.2f} s over {distance:5.2f} m  (mean decel {decel:.3f} m/s2)"
        )

    vehicle = {}
    if len(points) >= 2:
        slope, intercept, _ = fit_line(
            [limit for limit, _ in points], [d for _, d in points]
        )
        summary.append("")
        summary.append(
            f"braking authority {slope:.5f} (m/s2) per us of brake limit, "
            f"coasting baseline {intercept:.3f} m/s2"
        )
        vehicle["longitudinal.brake_authority"] = slope
        peak = max(d for _, d in points)
        vehicle["longitudinal.mu_longitudinal_lower_bound"] = peak / 9.81
        summary.append(f"peak mean deceleration {peak:.2f} m/s2 = {peak / 9.81:.2f} g")
    return {"summary": summary, "vehicle": vehicle, "plots": []}


# --------------------------------------------------------------------- C4


def analyze_tune_profile(run, options):
    """Score the closed loop on the same four metrics as the on-blocks table.

    Reported in spur RPM so the numbers sit directly beside the "Bench Tuning"
    table in the top level README rather than needing conversion to compare.
    """
    ratio = float(run.parameter("spur_to_wheel_ratio", 2.85) or 2.85)
    diameter = float(run.parameter("tire_diameter", 0.1143) or 0.1143)
    per_mps = 60.0 * ratio / (math.pi * diameter)

    errors, settled_errors, rise_times = [], [], []
    summary = []
    for segment in run.segments():
        if segment.label in (None, "settle"):
            continue
        rows = [
            row
            for row in segment.rows
            if row.get("speed") is not None and row.get("target_speed") is not None
        ]
        if len(rows) < 10:
            continue
        segment_errors = [
            abs(row["speed"] - row["target_speed"]) * per_mps for row in rows
        ]
        errors.extend(segment_errors)
        tail = segment_errors[len(segment_errors) // 2 :]
        settled_errors.extend(tail)

        target = rows[-1]["target_speed"]
        start = rows[0]["speed"]
        if abs(target - start) > 0.4:
            threshold = start + 0.9 * (target - start)
            times = [row["t_ros"] for row in rows]
            crossed = next(
                (
                    time
                    for time, row in zip(times, rows)
                    if (row["speed"] - threshold) * (1 if target > start else -1) >= 0
                ),
                None,
            )
            if crossed is not None:
                rise_times.append(crossed - times[0])
        summary.append(
            f"{segment.label:16s} target {target:+.2f} m/s  "
            f"mean |err| {sum(segment_errors) / len(segment_errors):7.1f} rpm  "
            f"settled {sum(tail) / len(tail):7.1f} rpm"
        )

    vehicle = {}
    if errors:
        mean = sum(errors) / len(errors)
        settled_mean = sum(settled_errors) / len(settled_errors)
        variance = sum((error - settled_mean) ** 2 for error in settled_errors) / len(
            settled_errors
        )
        rise = sum(rise_times) / len(rise_times) if rise_times else float("nan")
        gains = run.metadata.get("profile_gains") or {}
        summary.append("")
        summary.append(
            'Comparable with the "Bench Tuning" table in the top level README:'
        )
        summary.append("")
        summary.append(
            "| kP | kI | Mean Abs. Error | Settled Mean Abs. Error | Settled Std. Dev. | Time to 90% |"
        )
        summary.append(
            "| -- | -- | --------------- | ----------------------- | ----------------- | ----------- |"
        )
        summary.append(
            f"| {run.parameter('speed_kp', gains.get('speed_kp', '?'))} "
            f"| {run.parameter('speed_ki', gains.get('speed_ki', '?'))} "
            f"| {mean:.0f} | {settled_mean:.0f} | {variance**0.5:.0f} | {rise:.2f} s |"
        )
        vehicle["longitudinal.tune_mean_abs_error_rpm"] = mean
        vehicle["longitudinal.tune_settled_std_rpm"] = variance**0.5
    return {"summary": summary, "vehicle": vehicle, "plots": []}


ANALYZERS = {
    "zed_static": analyze_zed_static,
    "steer_authority": analyze_steer_authority,
    "skidpad": analyze_skidpad,
    "step_steer": analyze_step_steer,
    "pulse_staircase": analyze_pulse_staircase,
    "coastdown": analyze_coastdown,
    "brake_sweep": analyze_brake_sweep,
    "tune_profile": analyze_tune_profile,
}


def analyze(run, options=None):
    """Dispatch on the run's profile name."""
    options = options or {}
    analyzer = ANALYZERS.get(run.profile)
    if analyzer is None:
        raise KeyError(
            f"no analyzer for profile {run.profile!r}; "
            f"known profiles: {', '.join(sorted(ANALYZERS))}"
        )
    return analyzer(run, options)
