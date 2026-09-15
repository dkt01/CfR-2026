"""The physics fits, as pure functions over plain lists.

Kept free of file formats and of the run directory layout so each one can be
tested against synthetic data with a known answer - which is how you find out
that a fit is wrong, rather than discovering it when the simulator disagrees
with the car and nobody can tell which half is at fault.
"""

import math

from .linalg import fit_line, least_squares

__all__ = [
    "steady_state",
    "fit_resistance",
    "fit_first_order",
    "fit_effective_steering",
    "fit_understeer",
    "odometry_noise",
    "wrap_to_pi",
]


def wrap_to_pi(angle):
    """Wrap to (-pi, pi], matching WrapToPi in path_geometry.cpp."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def steady_state(times, values, tail_fraction=0.4):
    """Mean and standard deviation over the last `tail_fraction` of a hold.

    Every steady-state number in this campaign comes from the tail of a hold,
    never the whole of it: the head is the transient the car is still settling
    through, and averaging it in biases the result toward the previous step.
    """
    if not times or len(times) != len(values):
        raise ValueError("steady_state needs matching, non-empty times and values")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be within (0, 1]")
    span = times[-1] - times[0]
    cutoff = times[-1] - span * tail_fraction
    tail = [value for time, value in zip(times, values) if time >= cutoff]
    if not tail:
        tail = [values[-1]]
    mean = sum(tail) / len(tail)
    variance = sum((value - mean) ** 2 for value in tail) / len(tail)
    return {"mean": mean, "std": variance**0.5, "samples": len(tail)}


def fit_resistance(times, speeds, mass, min_speed=0.25):
    """Fit m dv/dt = -(f0 + f1 v + f2 v^2) to a coastdown.

    Returns rolling (N), viscous (N per m/s) and aero (N per (m/s)^2) terms.

    Samples below `min_speed` are dropped: the tachometer cannot resolve below
    roughly 0.3 m/s (one magnet, 400 ms stall timeout), so the tail of a
    coastdown is quantisation noise pretending to be data, and it lands right
    where it would do the most damage to f0.

    Speeds may be signed; the fit is done on magnitude so a reverse coastdown
    goes through the same function and comes out comparable.
    """
    if len(times) != len(speeds):
        raise ValueError("fit_resistance needs matching times and speeds")
    if mass <= 0.0:
        raise ValueError("mass must be positive")

    samples = [
        (time, abs(speed))
        for time, speed in zip(times, speeds)
        if abs(speed) >= min_speed
    ]
    if len(samples) < 8:
        raise ValueError(
            f"coastdown has only {len(samples)} usable samples above {min_speed} m/s"
        )

    # Central differences for dv/dt, which is far less noisy than a forward
    # difference on a signal this close to quantised.
    design, targets = [], []
    for index in range(1, len(samples) - 1):
        dt = samples[index + 1][0] - samples[index - 1][0]
        if dt <= 0.0:
            continue
        acceleration = (samples[index + 1][1] - samples[index - 1][1]) / dt
        speed = samples[index][1]
        design.append([1.0, speed, speed * speed])
        targets.append(-mass * acceleration)
    if len(design) < 8:
        raise ValueError("coastdown has too few usable difference points")

    (f0, f1, f2), rms = least_squares(design, targets)
    return {"f0": f0, "f1": f1, "f2": f2, "rms_newtons": rms, "points": len(design)}


def fit_first_order(times, values, initial=None, final=None):
    """Fit a first-order rise/fall, returning time constant and dead time.

    Linearised: ln((final - value) / (final - initial)) = -(t - dead) / tau.
    Only the 10%-90% band is used, which keeps the fit away from the dead time
    at one end and the measurement noise floor at the other.
    """
    if len(times) != len(values) or len(times) < 4:
        raise ValueError("fit_first_order needs at least four matching samples")
    initial = values[0] if initial is None else initial
    final = values[-1] if final is None else final
    span = final - initial
    if abs(span) < 1e-9:
        raise ValueError("step has no amplitude to fit")

    xs, ys = [], []
    for time, value in zip(times, values):
        fraction = (value - initial) / span
        if 0.1 <= fraction <= 0.9:
            xs.append(time - times[0])
            ys.append(math.log(1.0 - fraction))
    if len(xs) < 3:
        raise ValueError("step never passed cleanly through the 10-90% band")

    slope, intercept, r_squared = fit_line(xs, ys)
    if slope >= 0.0:
        raise ValueError("step does not decay toward its final value")
    tau = -1.0 / slope
    return {
        "tau": tau,
        "dead_time": max(0.0, intercept * tau),
        "initial": initial,
        "final": final,
        "r_squared": r_squared,
        "points": len(xs),
    }


def fit_effective_steering(speed, yaw_rate, wheelbase):
    """Effective bicycle steering angle from a steady arc.

    delta = atan(wheelbase * yaw_rate / speed).  This is what the simulator
    should interpolate, in preference to any per-wheel geometry: it already
    contains the linkage, the Ackermann error and any slip present at the speed
    it was measured at.
    """
    if wheelbase <= 0.0:
        raise ValueError("wheelbase must be positive")
    if abs(speed) < 1e-6:
        raise ValueError("effective steering is undefined at zero speed")
    return math.atan(wheelbase * yaw_rate / speed)


def fit_understeer(lateral_accelerations, steering_angles):
    """Understeer gradient from a skidpad speed sweep.

    delta = delta_kinematic + K * a_y, so the slope is K in rad per m/s^2.
    Positive K is understeer: the car needs more steering angle to hold the same
    radius as speed rises, which is to say it runs wider than the kinematic
    prediction the simulator currently always achieves.
    """
    slope, intercept, r_squared = fit_line(
        list(lateral_accelerations), list(steering_angles)
    )
    return {
        "understeer_gradient": slope,
        "kinematic_angle": intercept,
        "r_squared": r_squared,
        "points": len(lateral_accelerations),
    }


def odometry_noise(times, xs, ys, yaws):
    """Noise and drift of a stationary odometry recording.

    Position noise is reported as the standard deviation about the mean, and
    drift as the least-squares rate of change over the window - the two failure
    modes are different and a single number hides one of them.  Yaw is unwrapped
    first so a recording that crosses +/-pi does not report enormous noise.
    """
    if not (len(times) == len(xs) == len(ys) == len(yaws)) or len(times) < 10:
        raise ValueError("odometry_noise needs at least ten matching samples")

    unwrapped, previous = [], yaws[0]
    accumulated = yaws[0]
    for yaw in yaws:
        accumulated += wrap_to_pi(yaw - previous)
        previous = yaw
        unwrapped.append(accumulated)

    def trend_and_noise(values):
        """Drift rate, and the noise left once the drift is taken out.

        Measuring noise as spread about the MEAN conflates the two: a sensor
        drifting 0.004 rad/s over a five minute recording reports a quarter
        radian of "noise" that is really a ramp.  The simulator needs the two
        separately - one becomes a per-sample perturbation, the other a slow
        bias - so the noise here is the residual about the fitted trend.
        """
        slope, intercept, _ = fit_line(times, values)
        residuals = [
            value - (slope * time + intercept) for time, value in zip(times, values)
        ]
        mean = sum(residuals) / len(residuals)
        variance = sum((residual - mean) ** 2 for residual in residuals) / len(
            residuals
        )
        return slope, variance**0.5

    yaw_slope, yaw_noise = trend_and_noise(unwrapped)
    x_slope, x_noise = trend_and_noise(xs)
    y_slope, y_noise = trend_and_noise(ys)

    intervals = [
        later - earlier for earlier, later in zip(times, times[1:]) if later > earlier
    ]
    nominal = sorted(intervals)[len(intervals) // 2] if intervals else 0.0
    dropouts = sum(
        1 for interval in intervals if nominal > 0.0 and interval > 2.0 * nominal
    )

    return {
        "duration": times[-1] - times[0],
        "samples": len(times),
        "rate_hz": 1.0 / nominal if nominal > 0.0 else 0.0,
        "yaw_noise": yaw_noise,
        "position_noise": max(x_noise, y_noise),
        "yaw_drift_rate": yaw_slope,
        "position_drift_rate": math.hypot(x_slope, y_slope),
        "total_yaw_excursion": max(unwrapped) - min(unwrapped),
        "dropouts": dropouts,
        "dropout_fraction": dropouts / len(intervals) if intervals else 0.0,
    }
