#!/usr/bin/env python3
"""Does the corridor estimate hold up everywhere on the course, or only on straights?

`corridor.py` passes its synthetic tests exactly -- offset and heading come
back to within a millimetre and a tenth of a degree on a ray-cast straight
lane. That proves the arithmetic and nothing about the perception. The
question that decides whether it is worth training on is what the ZED
actually returns at each point of this particular course, which has at least
two places built to defeat a wall follower:

  * the **car wash**, whose ribbons `generate_obstacle_course.py` creates with
    `collide=False` because they are streamer weight -- the car drives
    straight through them -- but which the depth camera renders like any
    other surface, so the scan reports a solid wall across the opening;

  * the **wide open region** before the buckets, which has no lane walls at
    all, and whose bucket entrance is closed by a free-standing gap-bale wall
    rather than by anything a corridor fit would call a side.

So this teleports the car to sampled poses right round the lap, reads the
real point cloud at each one, and scores the estimate against ground truth
taken from the course centerline. Poses are deliberately perturbed off the
centerline -- an estimator that always answered "dead centre" would score
perfectly against unperturbed samples and be useless.

Ground truth conventions, which the `--self-test` flag checks rather than
assumes: a positive `lateral_offset_m` means the car sits to the LEFT of the
corridor centre, and a positive `heading_error_rad` means the corridor runs
off to the car's left, i.e. the car is aimed to the right of it.

    python3 validate_corridor.py --samples 12 --report corridor_report.json

Needs the simulator up with sensors rendering (the same stack
train_resilient_obstacle.sh starts). Exits non-zero if any region fails its
error bar, so it can gate a training run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0, str(REPO_ROOT / ".claude" / "skills" / "obstacle-course-regions" / "scripts")
)

from corridor import estimate_corridor, free_gap  # noqa: E402
from obstacle_course_path import CourseProgress  # noqa: E402

import regions  # noqa: E402

DEFAULT_SDF = REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/obstacle_course.sdf"

# Perturbations applied at each arc length. Zero offset is included so a
# systematic bias shows up unmixed with the gain error, and the rest span
# what a driving car actually sees.
OFFSETS_M = (0.0, +0.15, -0.15)
HEADINGS_DEG = (0.0, +12.0, -12.0)

# What counts as usable. These are not precision targets -- the feature only
# has to be better than the raw bins the policy already fails to read -- but
# an estimate wrong by more than half a lane width would steer the car into
# the wall it is meant to avoid.
OFFSET_TOLERANCE_M = 0.20
HEADING_TOLERANCE_DEG = 20.0

# The free gap is a coarser instrument than the corridor fit and is judged
# accordingly: it only has to point the car at the right opening, not measure
# it. A 30 degree tolerance is about one eighth of the field of view, which
# is the difference between "go right" and "go straight" -- enough to be
# useful at a corner, loose enough not to fail on the fact that the widest
# opening and the centerline are not the same line.
GAP_TOLERANCE_DEG = 30.0


def _wrap(angle: float) -> float:
    """Into (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _fold(angle: float) -> float:
    """Into [-pi/2, pi/2], the range a line direction occupies."""
    if angle > math.pi / 2.0:
        return angle - math.pi
    if angle < -math.pi / 2.0:
        return angle + math.pi
    return angle


def region_of(x: float, y: float) -> str:
    matches = regions.classify(x, y)
    if not matches:
        return "connecting_lane"
    return matches[0].name


def sample_poses(course: CourseProgress, samples: int):
    """(s, x, y, z, yaw, true_offset, true_heading_error) for every probe pose.

    The perturbation is applied in the centerline's own frame: `offset` slides
    the car along the centerline's left normal, `heading` turns it in place.
    The corridor the car then sees should be displaced and rotated by exactly
    those amounts, which is what makes this a test rather than a demo.
    """
    poses = []
    for index in range(samples):
        s = course.lap_length * index / samples
        x0, y0, z0, yaw0 = course.pose_at(s)
        left_x, left_y = -math.sin(yaw0), math.cos(yaw0)
        # Where the course goes next, as a bearing in the car's own frame.
        # This is the target for the free gap: the gap is not trying to find
        # the centerline, it is trying to find the way out, and on a course
        # the two coincide. Taken 1.5 m ahead rather than at the tangent,
        # because at a 90 degree corner the tangent points across the turn
        # (pose_at's docstring measured up to 69 degrees of error) while the
        # opening the car has to drive into is round it.
        ahead = course.points[course.index_at(min(s + 1.5, course.lap_length))]
        # The corridor's own direction, as the chord of the centerline over
        # roughly the span the side-sector fit sees -- NOT the instantaneous
        # tangent. On a curve those differ, and scoring against the tangent
        # charges the estimator for the curvature of the course: on the
        # banked turn it read a constant 29 degree bias while tracking every
        # perturbation correctly to within a degree or two, which is the
        # signature of a wrong reference rather than a wrong measurement.
        # Independent of the car's lateral offset, which the bearing to a
        # point ahead is not.
        chord_angle = math.atan2(ahead[1] - y0, ahead[0] - x0)

        for offset in OFFSETS_M:
            for heading_deg in HEADINGS_DEG:
                heading = math.radians(heading_deg)
                car_x = x0 + left_x * offset
                car_y = y0 + left_y * offset
                car_yaw = yaw0 + heading
                course_bearing = _wrap(
                    math.atan2(ahead[1] - car_y, ahead[0] - car_x) - car_yaw
                )
                poses.append(
                    {
                        "s": s,
                        "x": car_x,
                        "y": car_y,
                        "z": z0,
                        "yaw": car_yaw,
                        "course_bearing": course_bearing,
                        # The car is displaced to the left by `offset`, so the
                        # corridor centre is to its right and the reported
                        # offset should be +offset. The car is turned left by
                        # `heading`, so the corridor runs off to its right and
                        # the reported heading error should be -heading.
                        "true_offset": offset,
                        # Folded into [-90, +90]: a corridor has a direction
                        # but no forward end, so a wall at +95 degrees and
                        # one at -85 are the same wall, and the estimator
                        # reports the forward-pointing representative.
                        "true_heading": _fold(_wrap(chord_angle - car_yaw)),
                    }
                )
    return poses


def wait_for_fresh_cloud(env, settle_s: float, timeout_s: float = 4.0) -> bool:
    """Block until the camera has published a cloud rendered AFTER the teleport.

    A fixed sleep is not enough and fails silently in the worst possible way.
    The first run of this harness slept 0.6 s and scored several poses against
    the previous pose's cloud: the giveaway was pairs of rows with opposite
    yaw perturbations reporting byte-identical estimates and an identical
    scan minimum, which cannot happen if the car really moved. Those rows
    then counted as estimator failures when the estimator had never seen the
    geometry they were scored against.

    So this watches the cloud's own header stamp and waits for it to advance,
    then waits `settle_s` more for the physics to stop wobbling after the
    teleport drops the car.
    """
    with env._cloud_lock:
        previous = env._latest_cloud
    previous_stamp = None
    if previous is not None:
        previous_stamp = (previous.header.stamp.sec, previous.header.stamp.nanosec)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with env._cloud_lock:
            current = env._latest_cloud
        if current is not None:
            stamp = (current.header.stamp.sec, current.header.stamp.nanosec)
            if previous_stamp is None or stamp != previous_stamp:
                time.sleep(settle_s)
                return True
        time.sleep(0.02)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="arc lengths round the lap; each gets 9 perturbed poses",
    )
    parser.add_argument("--sdf", default=str(DEFAULT_SDF))
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.6,
        help="wait after teleporting before trusting the cloud",
    )
    parser.add_argument("--report", default=None, help="write per-pose JSON here")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the sign conventions on synthetic geometry and exit",
    )
    parser.add_argument(
        "--noisy",
        action="store_true",
        help="apply the ZED noise model before estimating, which is what the "
        "policy is actually given -- the clean scan flatters the estimator",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    import zed_sim
    from obstacle_env import ObstacleCourseEnv

    env = ObstacleCourseEnv(sdf_path=args.sdf, start_anywhere_prob=0.0)
    course = CourseProgress()
    poses = sample_poses(course, args.samples)
    print(f"probing {len(poses)} poses at {args.samples} arc lengths", flush=True)

    rows = []
    for index, pose in enumerate(poses):
        try:
            env._teleport(pose["x"], pose["y"], math.degrees(pose["yaw"]), pose["z"])
        except RuntimeError as error:
            print(f"  teleport failed at s={pose['s']:.1f}: {error}", flush=True)
            continue
        if not wait_for_fresh_cloud(env, args.settle_s):
            print(f"  cloud never refreshed at s={pose['s']:.1f}", flush=True)
            continue
        scan = env._raw_cloud_scan()
        if scan is None:
            print(f"  no cloud at s={pose['s']:.1f}", flush=True)
            continue

        if args.noisy:
            scan = zed_sim.apply(
                scan.copy(), env.zed_config, env.lidar_max_range, env._rng
            )
        gap_bearing, gap_depth = free_gap(scan, env.lidar_fov_deg, env.lidar_max_range)
        estimate = estimate_corridor(
            scan,
            env.lidar_fov_deg,
            env.lidar_max_range,
            gap_bearing_rad=gap_bearing,
        )
        row = {
            "gap_bearing_deg": math.degrees(gap_bearing),
            "gap_depth": gap_depth,
            "true_course_bearing_deg": math.degrees(pose["course_bearing"]),
            "s": pose["s"],
            "x": pose["x"],
            "y": pose["y"],
            "region": region_of(pose["x"], pose["y"]),
            "true_offset": pose["true_offset"],
            "true_heading_deg": math.degrees(pose["true_heading"]),
            "offset": estimate.lateral_offset_m,
            "heading_deg": math.degrees(estimate.heading_error_rad),
            "half_width": estimate.half_width_m,
            "confidence": estimate.confidence,
            "scan_min": float(scan.min()),
        }
        row["offset_error"] = row["offset"] - row["true_offset"]
        # Folded: both sides are line directions, so +89 against -89 is a
        # two-degree error, not 178.
        row["heading_error_deg"] = math.degrees(
            _fold(_wrap(math.radians(row["heading_deg"] - row["true_heading_deg"])))
        )
        row["gap_error_deg"] = row["gap_bearing_deg"] - row["true_course_bearing_deg"]
        rows.append(row)
        if index % 9 == 0:
            print(
                f"  s={pose['s']:5.1f}  {row['region']:<18}"
                f"  conf={row['confidence']:.2f}",
                flush=True,
            )

    env.close()
    if not rows:
        print("no usable samples -- is the simulator up with sensors rendering?")
        return 1

    if args.report:
        Path(args.report).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.report}")

    return summarise(rows)


def summarise(rows) -> int:
    """Per-region error, and a pass/fail that only judges confident answers.

    A region where the estimator declines to answer is not a failure -- the
    open floor genuinely has no corridor, and saying so is the correct
    output. What would be a failure is answering confidently and wrongly, so
    the error bars are applied to the confident rows and the unconfident ones
    are reported separately as coverage.
    """
    by_region = {}
    for row in rows:
        by_region.setdefault(row["region"], []).append(row)

    print(
        f"\n{'region':<20} {'n':>4} {'conf':>6} {'cover':>6} "
        f"{'|off|':>7} {'|hdg|':>7} {'|gap|':>7}  verdict"
    )
    print("-" * 82)

    failures = []
    for region in sorted(by_region):
        group = by_region[region]
        confident = [r for r in group if r["confidence"] >= 0.5]
        coverage = len(confident) / len(group)
        mean_confidence = statistics.mean(r["confidence"] for r in group)
        # The free gap needs no walls, so it is scored over every pose, not
        # only the ones the corridor fit was confident about. That is the
        # whole reason it is here: it has to carry the corners and the open
        # floor, which are exactly the poses missing from `confident`.
        gap_error = statistics.mean(abs(_wrap_deg(r["gap_error_deg"])) for r in group)
        gap_bad = gap_error > GAP_TOLERANCE_DEG

        if not confident:
            verdict = "no corridor" if not gap_bad else "FAIL (gap)"
            if gap_bad:
                failures.append((region, None, None, gap_error))
            print(
                f"{region:<20} {len(group):>4} {mean_confidence:>6.2f} "
                f"{coverage:>6.0%} {'--':>7} {'--':>7} {gap_error:>7.1f}  {verdict}"
            )
            continue

        offset_error = statistics.mean(abs(r["offset_error"]) for r in confident)
        heading_error = statistics.mean(abs(r["heading_error_deg"]) for r in confident)
        bad = (
            offset_error > OFFSET_TOLERANCE_M
            or heading_error > HEADING_TOLERANCE_DEG
            or gap_bad
        )
        verdict = "FAIL" if bad else "ok"
        if bad:
            failures.append((region, offset_error, heading_error, gap_error))
        print(
            f"{region:<20} {len(group):>4} {mean_confidence:>6.2f} "
            f"{coverage:>6.0%} {offset_error:>7.3f} {heading_error:>7.1f} "
            f"{gap_error:>7.1f}  {verdict}"
        )

    print()
    if failures:
        print("FAILED regions:")
        for region, offset_error, heading_error, gap_error in failures:
            if offset_error is None:
                print(
                    f"  {region}: no corridor (fine), but the free gap points "
                    f"{gap_error:.1f} deg off the way the course goes "
                    f"(tol {GAP_TOLERANCE_DEG})"
                )
                continue
            print(
                f"  {region}: offset off by {offset_error:.3f} m "
                f"(tol {OFFSET_TOLERANCE_M}), heading off by "
                f"{heading_error:.1f} deg (tol {HEADING_TOLERANCE_DEG}), "
                f"gap off by {gap_error:.1f} deg (tol {GAP_TOLERANCE_DEG})"
            )
        return 1
    print("every region either estimates within tolerance or declines to answer,")
    print("and the free gap points the way the course goes everywhere")
    return 0


def _wrap_deg(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def run_self_test() -> int:
    """Sign conventions, on geometry with no simulator in it.

    Worth its own path because a sign error here would not fail loudly -- it
    would quietly report the mirror image of the truth, and the training run
    that followed would steer confidently into walls.
    """
    fov, bins, max_range = 110.0, 36, 6.0
    edges = np.linspace(-fov / 2, fov / 2, bins + 1)
    bearings = np.radians((edges[:-1] + edges[1:]) / 2)

    def cast(half_width, offset, heading):
        scan = np.full(bins, max_range, dtype=np.float32)
        direction = np.array([math.cos(heading), math.sin(heading)])
        normal = np.array([-math.sin(heading), math.cos(heading)])
        del direction
        for index, bearing in enumerate(bearings):
            best = max_range
            for side in (+1, -1):
                centre = normal * (side * half_width - offset)
                ray = np.array([math.cos(bearing), math.sin(bearing)])
                denominator = ray @ normal
                if abs(denominator) < 1e-9:
                    continue
                distance = (centre @ normal) / denominator
                if 0.15 < distance < best:
                    best = distance
            scan[index] = best
        return scan

    checks = [
        ("car left of centre", cast(0.4065, +0.20, 0.0), +0.20, 0.0),
        ("car right of centre", cast(0.4065, -0.20, 0.0), -0.20, 0.0),
        ("corridor runs left", cast(0.4065, 0.0, math.radians(15)), 0.0, +15.0),
        ("corridor runs right", cast(0.4065, 0.0, math.radians(-15)), 0.0, -15.0),
    ]
    failed = 0
    for label, scan, want_offset, want_heading in checks:
        estimate = estimate_corridor(scan, fov, max_range)
        got_offset = estimate.lateral_offset_m
        got_heading = math.degrees(estimate.heading_error_rad)
        ok = (
            abs(got_offset - want_offset) < 0.02
            and abs(got_heading - want_heading) < 2.0
        )
        failed += not ok
        print(
            f"  {'ok ' if ok else 'BAD'} {label:<22} "
            f"offset {got_offset:+.3f} (want {want_offset:+.3f})  "
            f"heading {got_heading:+.1f} (want {want_heading:+.1f})"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
