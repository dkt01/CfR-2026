"""Tests for the offline characterization analysis.

Plain unittest so these run under pytest in CI and as a bare script on a laptop
with nothing installed - which is the same laptop that will be analysing runs in
a car park, so "the tests need a test runner installed" is not an option.

The fits are checked by synthesising data with a KNOWN answer and asserting it
comes back.  A fit that silently returns something plausible but wrong is the
expensive failure here: it would be discovered much later, as an unexplained
disagreement between the car and its twin, with no way to tell which half is
lying.
"""

import math
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts",
    ),
)

from characterize import fits, runs, traces  # noqa: E402
from characterize.linalg import SingularMatrixError, fit_line, least_squares  # noqa: E402


class TestLinalg(unittest.TestCase):
    def test_fit_line_recovers_slope_and_intercept(self):
        slope, intercept, r_squared = fit_line([0, 1, 2, 3, 4], [1, 4, 7, 10, 13])
        self.assertAlmostEqual(slope, 3.0, places=9)
        self.assertAlmostEqual(intercept, 1.0, places=9)
        self.assertAlmostEqual(r_squared, 1.0, places=9)

    def test_least_squares_recovers_quadratic(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        design = [[1.0, x, x * x] for x in xs]
        targets = [2.0 + 3.0 * x + 0.5 * x * x for x in xs]
        (a, b, c), rms = least_squares(design, targets)
        self.assertAlmostEqual(a, 2.0, places=6)
        self.assertAlmostEqual(b, 3.0, places=6)
        self.assertAlmostEqual(c, 0.5, places=6)
        self.assertLess(rms, 1e-9)

    def test_degenerate_system_raises_rather_than_returning_nonsense(self):
        with self.assertRaises(SingularMatrixError):
            least_squares([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0])

    def test_too_few_samples_rejected(self):
        with self.assertRaises(ValueError):
            least_squares([[1.0, 2.0, 3.0]], [1.0])


class TestTraceParsing(unittest.TestCase):
    """Field order must track the snprintf calls in arduino_rcm.ino.

    If the firmware reorders a field and nobody updates traces.py, every
    analysis silently reads the wrong column. These sample lines are the tripwire.
    """

    DEBUG = "D,12345,100,2,1,127,1500,4,1504,1560,1420,11800,0,,64,3,12,1"
    TRACE = "T,12345,1500,1487,560,30,120,0,710,1571,0,0"

    def test_debug_line_fields(self):
        parsed = traces.parse_line(self.DEBUG)
        self.assertEqual(parsed.kind, "D")
        self.assertEqual(parsed["arduino_ms"], 12345)
        self.assertEqual(parsed["cmd_steering"], 127)
        self.assertEqual(parsed["steering_us"], 1504)
        self.assertEqual(parsed["throttle_us"], 1560)
        self.assertEqual(parsed["rpm"], 1420)
        self.assertEqual(parsed["battery_mv"], 11800)
        self.assertEqual(parsed["dither_duty"], 64)
        self.assertEqual(parsed["merged_pulses"], 12)
        self.assertEqual(parsed["rejected_edges"], 1)
        self.assertIsNone(parsed["host_time"])

    def test_trace_line_scales_tenths_of_microseconds(self):
        parsed = traces.parse_line(self.TRACE)
        self.assertEqual(parsed.kind, "T")
        self.assertEqual(parsed["tracked_rpm"], 1500)
        self.assertEqual(parsed["measured_rpm"], 1487)
        self.assertAlmostEqual(parsed["feedforward_us"], 56.0)
        self.assertAlmostEqual(parsed["integral_us"], 12.0)
        self.assertAlmostEqual(parsed["output_us"], 71.0)
        self.assertEqual(parsed["throttle_us"], 1571)

    def test_host_timestamp_prefix_is_separated(self):
        parsed = traces.parse_line(f"1726300000.123456 {self.TRACE}")
        self.assertAlmostEqual(parsed["host_time"], 1726300000.123456, places=6)
        self.assertEqual(parsed["tracked_rpm"], 1500)

    def test_non_trace_and_malformed_lines_are_skipped_not_raised(self):
        # A status frame, a truncated line, and noise all appear in real traces.
        self.assertIsNone(traces.parse_line("0,1,0,3,255,0,0,1500,1,"))
        self.assertIsNone(traces.parse_line("T,1,2,3"))
        self.assertIsNone(traces.parse_line(""))
        self.assertIsNone(traces.parse_line("D,not,a,number"))

    def test_arduino_clock_alignment(self):
        lines = [
            traces.parse_line(
                f"{1000.0 + index * 0.02:.6f} T,{index * 20},0,0,0,0,0,0,0,1500,0,0"
            )
            for index in range(200)
        ]
        alignment = traces.align_arduino_clock(lines)
        self.assertAlmostEqual(alignment["scale"], 0.001, places=6)
        self.assertGreater(alignment["r_squared"], 0.999)


class TestFits(unittest.TestCase):
    def test_coastdown_recovers_known_resistance(self):
        mass, f0, f1, f2 = 4.5, 2.4, 0.15, 0.06
        speed, time, dt = 5.0, 0.0, 0.001
        times, speeds = [], []
        while speed > 0.15 and time < 90.0:
            if round(time * 1000) % 20 == 0:
                times.append(time)
                speeds.append(speed)
            speed -= ((f0 + f1 * speed + f2 * speed * speed) / mass) * dt
            time += dt
        fit = fits.fit_resistance(times, speeds, mass)
        self.assertAlmostEqual(fit["f0"], f0, places=2)
        self.assertAlmostEqual(fit["f1"], f1, places=2)
        self.assertAlmostEqual(fit["f2"], f2, places=3)

    def test_coastdown_handles_reverse_as_magnitude(self):
        mass = 4.5
        times = [index * 0.02 for index in range(300)]
        speeds = [-(5.0 * math.exp(-t / 4.0)) for t in times]
        fit = fits.fit_resistance(times, speeds, mass)
        self.assertGreater(fit["f0"], 0.0)

    def test_coastdown_rejects_a_run_with_nothing_usable(self):
        with self.assertRaises(ValueError):
            fits.fit_resistance([0.0, 0.1, 0.2], [0.05, 0.04, 0.0], 4.5)

    def test_first_order_recovers_tau_and_dead_time(self):
        tau, dead = 0.55, 0.12
        times = [index * 0.02 for index in range(200)]
        values = [
            0.0 if t < dead else 3.2 * (1.0 - math.exp(-(t - dead) / tau))
            for t in times
        ]
        fit = fits.fit_first_order(times, values, initial=0.0, final=3.2)
        self.assertAlmostEqual(fit["tau"], tau, places=3)
        self.assertAlmostEqual(fit["dead_time"], dead, places=3)

    def test_steady_state_uses_the_tail_not_the_transient(self):
        times = [index * 0.1 for index in range(100)]
        values = [0.0] * 50 + [3.0] * 50
        # Averaging the whole hold would give 1.5; the tail is what settled.
        self.assertAlmostEqual(
            fits.steady_state(times, values, 0.4)["mean"], 3.0, places=6
        )

    def test_effective_steering_matches_the_bicycle_model(self):
        wheelbase, speed, radius = 0.324, 2.0, 4.0
        delta = fits.fit_effective_steering(speed, speed / radius, wheelbase)
        self.assertAlmostEqual(delta, math.atan(wheelbase / radius), places=9)

    def test_understeer_gradient_recovered(self):
        accelerations = [1.0, 2.0, 3.5, 5.0, 7.0]
        result = fits.fit_understeer(
            accelerations, [0.20 + 0.012 * a for a in accelerations]
        )
        self.assertAlmostEqual(result["understeer_gradient"], 0.012, places=6)
        self.assertAlmostEqual(result["kinematic_angle"], 0.20, places=6)

    def test_odometry_noise_separates_noise_from_drift(self):
        """A drifting sensor must not report its drift as noise.

        Measuring spread about the mean would report ~0.07 rad of "noise" here,
        nine times the real figure, and the simulator would be given a sensor
        model that is wrong in both directions at once.
        """
        random.seed(7)
        times = [index / 30.0 for index in range(1800)]
        result = fits.odometry_noise(
            times,
            [random.gauss(0.0, 0.01) for _ in times],
            [random.gauss(0.0, 0.01) for _ in times],
            [0.004 * t + random.gauss(0.0, 0.008) for t in times],
        )
        self.assertAlmostEqual(result["yaw_noise"], 0.008, delta=0.001)
        self.assertAlmostEqual(result["yaw_drift_rate"], 0.004, delta=0.0005)
        self.assertAlmostEqual(result["position_noise"], 0.010, delta=0.001)
        self.assertAlmostEqual(result["rate_hz"], 30.0, delta=0.5)

    def test_odometry_noise_unwraps_yaw_across_pi(self):
        times = [index / 30.0 for index in range(600)]
        yaws = [
            fits.wrap_to_pi(math.pi - 0.001 + index * 0.0001) for index in range(600)
        ]
        result = fits.odometry_noise(times, [0.0] * 600, [0.0] * 600, yaws)
        # Without unwrapping the +pi/-pi crossing reads as a 6 rad excursion.
        self.assertLess(result["total_yaw_excursion"], 0.1)

    def test_wrap_to_pi_matches_the_cpp_convention(self):
        self.assertAlmostEqual(fits.wrap_to_pi(math.pi), math.pi, places=9)
        self.assertAlmostEqual(fits.wrap_to_pi(-math.pi), math.pi, places=9)
        self.assertAlmostEqual(fits.wrap_to_pi(3.0 * math.pi), math.pi, places=9)
        self.assertAlmostEqual(fits.wrap_to_pi(0.5), 0.5, places=9)


class TestRunLoading(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)

    def _write(self, rows):
        from characterize.runs import _NUMERIC  # noqa: F401  (import guard)

        header = (
            "t_ros,t_elapsed,phase,step_index,step_label,step_phase,cmd_steering,"
            "cmd_velocity,auto_ready,mode,link_ok,estop,gains_applied,battery_level,"
            "battery_volts,rpm,wheel_rpm,speed,target_speed,throttle_us,odom_valid,"
            "odom_x,odom_y,odom_yaw,odom_vx,odom_wz,dist_along,dist_total"
        )
        with open(
            os.path.join(self.directory, "telemetry.csv"), "w", encoding="utf-8"
        ) as handle:
            handle.write(header + "\n")
            for row in rows:
                handle.write(",".join(str(value) for value in row) + "\n")

    def test_segments_exclude_the_gains_wait_window(self):
        """A gains_wait row is the PREVIOUS step's motion, under the next label.

        Counting it against the new step would attribute one throttle pulse's
        steady state to the pulse that replaced it - which is exactly the error
        the staircase profile is built to avoid.
        """
        rows = []
        for index in range(10):
            rows.append(
                [
                    index * 0.02,
                    index * 0.02,
                    "running",
                    0,
                    "stepA",
                    "hold",
                    0.0,
                    1.0,
                    1,
                    4,
                    1,
                    0,
                    1,
                    200,
                    12.0,
                    500,
                    175,
                    1.0,
                    1.0,
                    1560,
                    1,
                    0,
                    0,
                    0,
                    1.0,
                    0,
                    0,
                    0,
                ]
            )
        for index in range(5):
            rows.append(
                [
                    0.2 + index * 0.02,
                    0.2,
                    "running",
                    1,
                    "stepB",
                    "gains_wait",
                    0.0,
                    1.0,
                    1,
                    4,
                    1,
                    0,
                    0,
                    200,
                    12.0,
                    500,
                    175,
                    1.0,
                    1.0,
                    1560,
                    1,
                    0,
                    0,
                    0,
                    1.0,
                    0,
                    0,
                    0,
                ]
            )
        for index in range(10):
            rows.append(
                [
                    0.3 + index * 0.02,
                    0.3,
                    "running",
                    1,
                    "stepB",
                    "hold",
                    0.0,
                    2.0,
                    1,
                    4,
                    1,
                    0,
                    1,
                    199,
                    12.0,
                    900,
                    315,
                    2.0,
                    2.0,
                    1590,
                    1,
                    0,
                    0,
                    0,
                    2.0,
                    0,
                    0,
                    0,
                ]
            )
        self._write(rows)

        run = runs.load(self.directory)
        segments = run.segments()
        self.assertEqual([segment.label for segment in segments], ["stepA", "stepB"])
        self.assertEqual(len(segments[0]), 10)
        self.assertEqual(len(segments[1]), 10)  # not 15
        self.assertAlmostEqual(
            fits.steady_state(*segments[1].pair("t_ros", "odom_vx"))["mean"], 2.0
        )

    def test_missing_columns_become_none_rather_than_crashing(self):
        rows = [
            [
                0.0,
                "",
                "wait_link",
                "",
                "",
                "",
                0.0,
                0.0,
                0,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                0,
                "",
                "",
                "",
                "",
                "",
                0.0,
                0.0,
            ]
        ]
        self._write(rows)
        run = runs.load(self.directory)
        self.assertIsNone(run.rows[0]["mode"])
        self.assertIsNone(run.rows[0]["odom_x"])
        self.assertEqual(run.rows[0]["phase"], "wait_link")

    def test_a_directory_without_telemetry_is_reported_clearly(self):
        with self.assertRaises(FileNotFoundError):
            runs.load(tempfile.mkdtemp())


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_yaml(path):
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TestVehiclePatch(unittest.TestCase):
    def setUp(self):
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "scripts",
            ),
        )
        self.source = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "vehicle.yaml",
        )
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.target = os.path.join(self.directory, "vehicle.yaml")
        shutil.copy(self.source, self.target)

    def test_patch_updates_values_and_keeps_every_comment(self):
        """The comments ARE the file.

        vehicle.yaml is mostly an explanation of where each number came from and
        why it matters. A round trip through PyYAML would drop all of it and
        leave a list of bare numbers nobody can audit.
        """
        from apply_vehicle_patch import apply_patch

        before = _read(self.source).count("#")
        applied, _ = apply_patch(
            self.target,
            {"mass.total": 4.72, "steering.max_angle_left": 0.3412},
            "/home/user/cfr_runs/test",
            today="2026-09-15",
        )

        self.assertEqual(sorted(applied), ["mass.total", "steering.max_angle_left"])
        self.assertEqual(_read(self.target).count("#"), before)

        loaded = _load_yaml(self.target)
        self.assertAlmostEqual(loaded["mass"]["total"]["value"], 4.72)
        self.assertEqual(loaded["mass"]["total"]["provenance"], "measured")
        self.assertEqual(loaded["mass"]["total"]["run"], "/home/user/cfr_runs/test")
        self.assertEqual(loaded["mass"]["total"]["experiment"], "A1")  # kept
        # An untouched entry keeps its guess tag, so the file still says what is
        # measured and what is not.
        self.assertEqual(loaded["tire"]["diameter"]["provenance"], "guess")

    def test_table_patch_replaces_every_row(self):
        """Regression: the audit lines were landing inside the rows list.

        That truncated the steering table at the insertion point, silently
        dropping full lock from a measurement whose whole purpose is full lock.
        """
        from apply_vehicle_patch import apply_patch

        table = [
            [-1.0, -0.341],
            [-0.5, -0.171],
            [0.0, 0.002],
            [0.5, 0.174],
            [1.0, 0.345],
        ]
        apply_patch(
            self.target,
            {"steering.effective_angle_table": table},
            "/home/user/cfr_runs/test",
            today="2026-09-15",
        )
        loaded = _load_yaml(self.target)
        entry = loaded["steering"]["effective_angle_table"]
        self.assertEqual(len(entry["rows"]), 5)
        self.assertAlmostEqual(entry["rows"][-1][1], 0.345)
        self.assertEqual(entry["provenance"], "measured")

    def test_reapplying_does_not_stack_audit_lines(self):
        from apply_vehicle_patch import apply_patch

        apply_patch(self.target, {"mass.total": 4.72}, "/run/one", today="2026-09-15")
        apply_patch(self.target, {"mass.total": 4.80}, "/run/two", today="2026-09-16")
        text = _read(self.target)
        self.assertEqual(text.count("run: /run/two"), 1)
        self.assertEqual(text.count("run: /run/one"), 0)
        loaded = _load_yaml(self.target)
        self.assertAlmostEqual(loaded["mass"]["total"]["value"], 4.80)

    def test_unknown_key_is_skipped_not_fatal(self):
        """A fit may produce a diagnostic with no home in vehicle.yaml."""
        from apply_vehicle_patch import apply_patch

        applied, skipped = apply_patch(
            self.target,
            {"sensors.rpm_to_odom_speed_ratio": 0.98},
            "/run/x",
            today="2026-09-15",
        )
        self.assertEqual(applied, [])
        self.assertEqual(len(skipped), 1)


class TestProfiles(unittest.TestCase):
    """Every shipped profile must load and stay inside its own limits."""

    def setUp(self):
        self.directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "profiles",
        )

    def test_all_profiles_parse_and_are_self_consistent(self):
        found = sorted(
            name for name in os.listdir(self.directory) if name.endswith(".yaml")
        )
        self.assertTrue(found, "no profiles found")
        for name in found:
            with self.subTest(profile=name):
                path = os.path.join(self.directory, name)
                data = _load_yaml(path)
                limits = data["limits"]
                self.assertGreater(limits["max_distance"], 0.0)
                self.assertTrue(data["steps"], f"{name} has no steps")
                nominal = sum(step["hold"] for step in data["steps"])
                self.assertLessEqual(
                    nominal,
                    limits["max_duration"],
                    f"{name} nominally runs {nominal:.0f} s but aborts at "
                    f"{limits['max_duration']:.0f} s",
                )
                for index, step in enumerate(data["steps"]):
                    self.assertLessEqual(
                        abs(step.get("velocity", 0.0)),
                        limits["max_speed"] + 1e-9,
                        f"{name} step {index} exceeds its own max_speed",
                    )
                    self.assertLessEqual(abs(step.get("steering", 0.0)), 1.0)

    def test_non_arming_profiles_never_command_motion(self):
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".yaml"):
                continue
            data = _load_yaml(os.path.join(self.directory, name))
            if data.get("arming") != "none":
                continue
            with self.subTest(profile=name):
                for step in data["steps"]:
                    self.assertEqual(step.get("velocity", 0.0), 0.0)

    def test_no_profile_outruns_its_own_distance_limit(self):
        """Simulate each profile kinematically and check it stays in bounds.

        A profile that commands more ground than its max_distance allows will
        abort partway through, and the cost of finding that out is a trip to a
        car park. Integrating a bicycle model over the commanded steps catches it
        at build time instead.

        Steering is taken at the CONFIGURED 0.40 rad full lock. That number is a
        guess (it is what B1 exists to measure), so this is an estimate - but it
        is the same estimate the profiles were sized against, and it fails loudly
        if a step list grows past the space it was designed for.
        """
        wheelbase, max_steer = 0.324, 0.40
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".yaml"):
                continue
            data = _load_yaml(os.path.join(self.directory, name))
            with self.subTest(profile=name):
                x = y = yaw = 0.0
                peak = 0.0
                for step in data["steps"]:
                    speed = float(step.get("velocity", 0.0))
                    delta = float(step.get("steering", 0.0)) * max_steer
                    yaw_rate = speed * math.tan(delta) / wheelbase
                    dt, elapsed = 0.05, 0.0
                    while elapsed < float(step["hold"]):
                        x += speed * math.cos(yaw) * dt
                        y += speed * math.sin(yaw) * dt
                        yaw += yaw_rate * dt
                        elapsed += dt
                        peak = max(peak, math.hypot(x, y))
                self.assertLessEqual(
                    peak,
                    data["limits"]["max_distance"],
                    f"{name} reaches {peak:.1f} m from the start but aborts at "
                    f"{data['limits']['max_distance']:.1f} m",
                )

    def test_coastdown_keeps_braking_disabled(self):
        """A brake limit would make the coastdown measure the brakes instead."""
        data = _load_yaml(os.path.join(self.directory, "coastdown.yaml"))
        self.assertEqual(data["gains"]["speed_brake_limit"], 0.0)
        for step in data["steps"]:
            self.assertEqual(step.get("gains", {}).get("speed_brake_limit", 0.0), 0.0)

    def test_pulse_staircase_is_open_loop(self):
        """kV/kP/kI/kD must all be zero or the sweep is not a pulse sweep.

        With any of them nonzero the controller reacts to speed and the output
        is no longer just kS, which is the entire trick that lets this run
        without a firmware change.
        """
        data = _load_yaml(os.path.join(self.directory, "pulse_staircase.yaml"))
        for gain in ("speed_kv", "speed_kp", "speed_ki", "speed_kd"):
            self.assertEqual(data["gains"][gain], 0.0, f"{gain} must be zero")
        for step in data["steps"]:
            if step.get("velocity", 0.0) != 0.0:
                continue
        self.assertGreaterEqual(
            data["gains"]["speed_output_limit"],
            max(step.get("gains", {}).get("speed_ks", 0.0) for step in data["steps"]),
            "output limit clips the top of the sweep",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
