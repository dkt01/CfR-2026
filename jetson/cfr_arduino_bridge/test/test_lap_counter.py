"""Tests for lap counting and the gates that keep it honest.

The traces here are synthetic, built to the dimensions the speed course
actually has, so they run with no simulator and no ROS.  What they cannot
check is that the ZED's map frame holds still enough over three laps for the
latched reference to stay meaningful; only driving the course does that.

The speed course is a 135 ft x 47 ft oval -- about 41 m by 14 m -- so the
figures below are the real ones: a lap is roughly 98 m, and the return leg of
the oval sits about 14 m to the side of the start straight, travelling the
opposite way.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "src" / "lap_counter.py"
_spec = importlib.util.spec_from_file_location("lap_counter", MODULE)
lap_counter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lap_counter
_spec.loader.exec_module(lap_counter)

Pose2D = lap_counter.Pose2D
Geometry = lap_counter.Geometry
LapTracker = lap_counter.LapTracker

LANE_LENGTH = 41.0  # m, the long side of the oval
LANE_WIDTH = 14.0  # m, how far the return leg sits from the start straight
BEHIND = -2.0  # m, where the start straight is rejoined, behind the line
START = Pose2D(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------- setup


def tracker(target=3, **geometry):
    """A tracker armed at the origin, facing down the lane."""
    counter = LapTracker(target, Geometry(**geometry))
    counter.arm(START)
    return counter


def feed(counter, poses, counting=True):
    """Push a trace through, returning every crossing it produced."""
    crossings = []
    for pose in poses:
        observation = counter.update(pose, counting=counting)
        if observation.crossing is not None:
            crossings.append(observation.crossing)
    return crossings


def line(start, end, step=0.25):
    """Poses along a straight run from `start` to `end`, heading with travel."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    span = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    count = max(int(span / step), 1)
    return [
        Pose2D(start[0] + dx * i / count, start[1] + dy * i / count, yaw)
        for i in range(count + 1)
    ]


def depart():
    """Out of the start box, across the line, down to the far end."""
    return line((0.0, 0.0), (LANE_LENGTH, 0.0))


def rejoin(end=(LANE_LENGTH, 0.0)):
    """Round the oval from the far end and back up the start straight to `end`.

    Squared off rather than rounded -- the corners are not what is under test,
    and a rectangle makes the distances easy to read.  It rejoins the straight
    behind the line, so the run up to `end` passes forward through the line.
    """
    return (
        line((LANE_LENGTH, 0.0), (LANE_LENGTH, -LANE_WIDTH))
        + line((LANE_LENGTH, -LANE_WIDTH), (BEHIND, -LANE_WIDTH))
        + line((BEHIND, -LANE_WIDTH), (BEHIND, 0.0))
        + line((BEHIND, 0.0), end)
    )


def circuit():
    """One full lap: round the oval, across the line, back to the far end."""
    return rejoin()


def departed(**geometry):
    """A tracker that has already left the line and is out on course."""
    counter = tracker(**geometry)
    feed(counter, depart())
    return counter


def verdicts(crossings):
    return [crossing.verdict for crossing in crossings]


def counted(crossings):
    return [crossing for crossing in crossings if crossing.counted]


# ----------------------------------------------------------------- the basics


def test_the_outbound_crossing_arms_rather_than_scores():
    """The car is parked behind the line, so it crosses it on the way out.

    That pass is the run starting, not a lap completed -- issue #27 asks for
    three laps of the speed course, which is four crossings in total.
    """
    counter = tracker()
    crossings = feed(counter, line((0.0, 0.0), (10.0, 0.0)))

    assert verdicts(crossings) == [lap_counter.DEPARTURE]
    assert counter.laps == 0
    assert counter.state == lap_counter.STATE_RUNNING
    # Not a rejection either: nothing went wrong, so nothing to report.
    assert counter.rejected == 0


def test_three_laps_of_the_speed_course_finish_the_run():
    counter = tracker(target=3)
    crossings = feed(counter, depart() + circuit() * 3)

    assert verdicts(crossings) == [
        lap_counter.DEPARTURE,
        lap_counter.COUNTED,
        lap_counter.COUNTED,
        lap_counter.COUNTED,
    ]
    assert counter.laps == 3
    assert counter.done
    assert counter.state == lap_counter.STATE_DONE


def test_two_laps_of_the_obstacle_course_finish_the_run():
    counter = tracker(target=2)
    feed(counter, depart() + circuit() * 2)

    assert counter.laps == 2
    assert counter.done


def test_the_run_is_not_done_a_lap_early():
    """The failure that loses a heat: stopping on lap two of three."""
    counter = tracker(target=3)
    feed(counter, depart() + circuit() * 2)

    assert counter.laps == 2
    assert not counter.done


def test_a_lap_measures_about_the_ovals_perimeter():
    """Guards the trace itself: if `circuit()` is wrong, the distance gate
    tests below are not testing what they claim to."""
    counter = tracker()
    crossings = feed(counter, depart() + circuit())

    # 2 * 43 + 2 * 14 = 114 m round the squared-off rectangle.
    assert counter.laps == 1
    assert counted(crossings)[0].travelled == pytest.approx(114.0, abs=2.0)


# ------------------------------------------------------------- the back straight


def test_the_return_leg_of_the_oval_never_scores():
    """The reason a bare plane crossing will not do.

    The oval's far side crosses the plane of the line too, 14 m out and
    travelling the opposite way.  Heading is what rejects it.
    """
    counter = tracker()
    feed(counter, depart())
    crossings = feed(
        counter,
        line((LANE_LENGTH, -LANE_WIDTH), (BEHIND, -LANE_WIDTH)),
    )

    # Travelling -x, so it never passes through the plane forwards at all.
    assert crossings == []
    assert counter.laps == 0


def test_a_reversed_pass_over_the_line_itself_never_scores():
    """The case a capture radius gets wrong and heading gets right.

    Here the car passes directly over the line -- zero cross-track, so any
    distance-from-the-start gate would wave it through -- but pointing the
    wrong way.  It is not a lap however close to the line it is.
    """
    counter = tracker()
    feed(counter, depart())
    before = counter.laps

    # Roll backwards over the line, nose still pointing down the lane, then
    # forwards again: the forward pass is the one that has to be judged.
    reversed_pass = [Pose2D(x, 0.0, math.pi) for x in (5.0, 3.0, 1.0, -1.0)]
    forward_again = [Pose2D(x, 0.0, math.pi) for x in (-1.0, 1.0, 3.0)]
    crossings = feed(counter, reversed_pass + forward_again)

    assert verdicts(crossings) == [lap_counter.REJECT_HEADING]
    assert counter.laps == before


def test_a_lap_that_returns_wide_still_scores():
    """The physical course will not match the idealized one.

    A car that comes back 12 m off the centerline has still completed a lap,
    and with the lateral gate off -- the default -- it is counted.  This is
    the case the capture radius would have silently stopped scoring.
    """
    counter = tracker()
    feed(counter, depart())
    crossings = feed(
        counter,
        line((LANE_LENGTH, -LANE_WIDTH), (BEHIND, -12.0))
        + line((BEHIND, -12.0), (5.0, -12.0)),
    )

    assert len(counted(crossings)) == 1
    assert counter.laps == 1


def test_the_lateral_gate_rejects_that_same_lap_when_asked():
    """The backstop is there for a course that needs it, off by default."""
    counter = tracker(lateral_gate=5.0)
    feed(counter, depart())
    crossings = feed(
        counter,
        line((LANE_LENGTH, -LANE_WIDTH), (BEHIND, -12.0))
        + line((BEHIND, -12.0), (5.0, -12.0)),
    )

    assert verdicts(crossings) == [lap_counter.REJECT_LATERAL]
    assert counter.laps == 0


# --------------------------------------------------------------------- jitter


def test_a_car_stopped_astride_the_line_scores_once_not_many():
    """Creeping back and forth over the line must not run up the count."""
    counter = tracker()
    feed(counter, depart() + rejoin(end=(1.0, 0.0)))
    assert counter.laps == 1

    jitter = []
    for _ in range(5):
        jitter += line((1.0, 0.0), (0.4, 0.0)) + line((0.4, 0.0), (1.0, 0.0))
    crossings = feed(counter, jitter)

    assert verdicts(crossings) == [lap_counter.REJECT_DISTANCE] * 5
    assert counter.laps == 1


# --------------------------------------------------------------- loop closure


def test_a_loop_closure_near_the_line_still_scores_its_lap():
    """The correction arrives exactly where it is least convenient.

    A metre-scale jump that lands the car across the line, after a full lap of
    travel, is still that lap: the counter does not re-seed on a closure.
    """
    counter = tracker()
    feed(counter, depart())
    feed(counter, rejoin(end=(-0.5, 0.0)))  # round to just short of the line
    observation = counter.update(Pose2D(0.9, 0.0, 0.0))

    assert observation.loop_closure == pytest.approx(1.4, abs=1e-6)
    assert observation.crossing is not None
    assert observation.crossing.counted
    assert counter.laps == 1
    assert counter.loop_closures == 1


def test_a_loop_closure_on_the_far_side_scores_nothing():
    counter = tracker()
    feed(counter, line((0.0, 0.0), (LANE_LENGTH, 0.0)))
    observation = counter.update(Pose2D(LANE_LENGTH + 3.0, -2.0, 0.0))

    assert observation.loop_closure > 1.0
    assert observation.crossing is None
    assert counter.laps == 0


def test_a_loop_closure_is_not_distance_the_car_drove():
    """Otherwise a jump could push a short lap past the distance gate."""
    counter = tracker()
    feed(counter, line((0.0, 0.0), (5.0, 0.0)))
    before = counter.lap_distance
    counter.update(Pose2D(12.0, 0.0, 0.0))

    assert counter.lap_distance == pytest.approx(before, abs=1e-6)
    assert counter.loop_closures == 1


# --------------------------------------------------------------- e-stop carry


def test_no_lap_is_scored_while_the_car_is_e_stopped():
    """The rules allow an e-stop to lift the car past an obstacle."""
    counter = tracker()
    feed(counter, depart())
    crossings = feed(counter, circuit(), counting=False)

    assert counted(crossings) == []
    assert counter.laps == 0


def test_being_carried_over_the_line_under_e_stop_scores_nothing():
    """Lifted from behind the line to well past it, then re-armed.

    The displacement must not read as a lap, and it must not leave the
    counter thinking the car drove there either.
    """
    counter = tracker()
    feed(counter, depart())

    counter.update(Pose2D(-3.0, 0.0, 0.0), counting=False)
    counter.update(Pose2D(20.0, 0.0, 0.0), counting=False)
    counter.resume()
    observation = counter.update(Pose2D(20.0, 0.0, 0.0))

    assert observation.crossing is None
    assert counter.laps == 0


def test_a_genuine_lap_after_an_e_stop_still_scores():
    """Suspending the count must not break it for the rest of the run."""
    counter = tracker()
    feed(counter, depart())

    feed(counter, line((LANE_LENGTH, 0.0), (LANE_LENGTH, -5.0)), counting=False)
    counter.resume()
    crossings = feed(
        counter,
        line((LANE_LENGTH, -5.0), (BEHIND, -LANE_WIDTH))
        + line((BEHIND, -LANE_WIDTH), (BEHIND, 0.0))
        + line((BEHIND, 0.0), (3.0, 0.0)),
    )

    assert len(counted(crossings)) == 1
    assert counter.laps == 1


def test_a_car_carried_back_behind_the_line_does_not_score_on_returning():
    """The failure this guards is a lap the car never drove.

    Lifted from the back straight and set down behind the line, the car has
    already banked most of a lap's distance.  Driving the few metres forward
    over the line would otherwise satisfy the distance gate with travel from
    before the e-stop, and score.
    """
    counter = tracker()
    feed(counter, depart())
    # Lifted off the back straight, set down behind the line, re-armed.
    feed(
        counter,
        [Pose2D(LANE_LENGTH, -LANE_WIDTH, 0.0), Pose2D(-1.0, 0.0, 0.0)],
        counting=False,
    )

    crossings = feed(counter, line((-1.0, 0.0), (5.0, 0.0)))

    assert verdicts(crossings) == [lap_counter.REJECT_DISTANCE]
    assert counter.laps == 0


def test_stopping_in_place_keeps_the_lap_the_car_had_driven():
    """An e-stop that does not move the car must not cost it its lap.

    Otherwise every pause would force another `min_lap_distance` before the
    line would score, and the car would run on past the finish.
    """
    counter = tracker()
    feed(counter, depart())
    banked = counter.lap_distance

    # Stopped, not moved: a few samples in the same spot.
    feed(counter, [Pose2D(LANE_LENGTH, 0.0, 0.0)] * 5, counting=False)
    counter.update(Pose2D(LANE_LENGTH, 0.0, 0.0))

    assert counter.lap_distance == pytest.approx(banked, abs=1e-6)

    crossings = feed(counter, rejoin())
    assert len(counted(crossings)) == 1


def test_the_carry_is_reported_so_it_can_be_seen_in_a_log():
    counter = tracker()
    feed(counter, depart())
    counter.update(Pose2D(LANE_LENGTH, -LANE_WIDTH, 0.0), counting=False)
    observation = counter.update(Pose2D(-1.0, 0.0, 0.0))

    assert observation.carried == pytest.approx(math.hypot(42.0, 14.0), abs=0.01)
    assert counter.lap_distance == 0.0


# ------------------------------------------------------------------ reporting


def test_every_rejected_crossing_names_the_gate_that_rejected_it():
    """The physical course is tuned against these, so they have to be right."""
    # A tracker apiece, because each gate needs the run in a different state
    # and a shared one would smuggle the previous case's travel into the next.
    wrong_way = departed(lateral_gate=5.0)
    assert verdicts(
        feed(wrong_way, [Pose2D(-1.0, 0.0, math.pi), Pose2D(2.0, 0.0, math.pi)])
    ) == [lap_counter.REJECT_HEADING]

    too_wide = departed(lateral_gate=5.0)
    assert verdicts(
        feed(too_wide, [Pose2D(-1.0, 9.0, 0.0), Pose2D(2.0, 9.0, 0.0)])
    ) == [lap_counter.REJECT_LATERAL]

    # Out a few metres and straight back: a real crossing, far too soon.
    too_short = tracker()
    feed(too_short, line((0.0, 0.0), (3.0, 0.0)) + line((3.0, 0.0), (-1.0, 0.0)))
    assert verdicts(feed(too_short, line((-1.0, 0.0), (3.0, 0.0)))) == [
        lap_counter.REJECT_DISTANCE
    ]

    assert [c.rejected for c in (wrong_way, too_wide, too_short)] == [1, 1, 1]


def test_a_crossing_reports_where_and_how_it_happened():
    counter = tracker()
    crossings = feed(counter, depart() + circuit())
    crossing = counted(crossings)[0]

    assert crossing.along_track >= counter.geometry.line_offset
    assert crossing.cross_track == pytest.approx(0.0, abs=0.1)
    assert crossing.heading_error == pytest.approx(0.0, abs=0.1)
    assert crossing.travelled > counter.geometry.min_lap_distance


def test_an_unarmed_tracker_reports_idle_and_counts_nothing():
    counter = LapTracker(3)
    observation = counter.update(Pose2D(5.0, 5.0, 1.0))

    assert observation.state == lap_counter.STATE_IDLE
    assert observation.laps == 0
    assert not counter.armed


def test_reset_clears_the_latch_and_the_count():
    counter = tracker(target=1)
    feed(counter, depart() + circuit())
    assert counter.done

    counter.reset()

    assert counter.laps == 0
    assert not counter.done
    assert counter.state == lap_counter.STATE_IDLE
    assert not counter.armed


def test_re_arming_puts_the_line_where_the_car_now_stands():
    """A second run starts from wherever the car was put back, not the first
    run's origin -- the map frame did not move, but the car did."""
    counter = tracker()
    feed(counter, depart() + circuit())
    counter.reset()
    counter.arm(Pose2D(100.0, 50.0, math.pi / 2.0))

    crossings = feed(counter, line((100.0, 50.0), (100.0, 60.0)))

    assert verdicts(crossings) == [lap_counter.DEPARTURE]


# ------------------------------------------------------------------- geometry


def test_pose_relative_to_matches_the_cpp_implementation():
    """The same case as PoseRelativeToTest.RebasesPositionAndHeading in
    test_path_geometry.cpp, because these two must not drift apart."""
    reference = Pose2D(10.0, -3.0, math.pi / 2.0)
    pose = Pose2D(9.0, -1.0, 3.0 * math.pi / 4.0)

    relative = lap_counter.pose_relative_to(pose, reference)

    assert relative.x == pytest.approx(2.0, abs=1e-6)
    assert relative.y == pytest.approx(1.0, abs=1e-6)
    assert relative.yaw == pytest.approx(math.pi / 4.0, abs=1e-6)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2.0, -math.pi / 2.0, 2.0, -2.0])
def test_yaw_from_quaternion_matches_the_cpp_implementation(yaw):
    """The cases from YawFromQuaternionTest, plus a couple more."""
    w = math.cos(yaw / 2.0)
    z = math.sin(yaw / 2.0)

    assert lap_counter.yaw_from_quaternion(w, 0.0, 0.0, z) == pytest.approx(
        yaw, abs=1e-6
    )


def test_wrap_to_pi_wraps_past_pi():
    assert lap_counter.wrap_to_pi(3.0 * math.pi / 2.0) == pytest.approx(
        -math.pi / 2.0, abs=1e-6
    )
    assert lap_counter.wrap_to_pi(-3.0 * math.pi / 2.0) == pytest.approx(
        math.pi / 2.0, abs=1e-6
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("heading_tolerance", 0.0),
        ("heading_tolerance", 4.0),
        ("min_lap_distance", -1.0),
        ("max_step", 0.0),
        ("lateral_gate", -1.0),
    ],
)
def test_nonsense_geometry_is_refused(field, value):
    """The node turns a rejected parameter set into a warning rather than
    taking it, so these have to raise."""
    with pytest.raises(ValueError):
        Geometry(**{field: value})


def test_a_target_below_one_lap_is_refused():
    with pytest.raises(ValueError):
        LapTracker(0)
