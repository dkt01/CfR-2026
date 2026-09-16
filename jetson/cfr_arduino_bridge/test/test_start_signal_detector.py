"""Tests for the start signal detector's colour decision and its latch.

The frames here are synthetic, built from the diffuse colours the two worlds
actually use, so they run with no camera, no simulator and no ROS.  What they
cannot check is that Gazebo renders those colours where this expects them;
`scripts/check_start_signal.py` does that against a running simulation.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).parents[1] / "src" / "start_signal_detector.py"
_spec = importlib.util.spec_from_file_location("start_signal_detector", MODULE)
detector = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = detector
_spec.loader.exec_module(detector)

WIDTH, HEIGHT = 640, 360

# The worlds' diffuse colours, in 8-bit.  Every one of these lands in frame
# from the start line, so the detector has to pick the arms out from among
# them rather than merely off a black background.
SKY = (128, 128, 128)  # no sky model; the background renders flat grey
GROUND = (41, 64, 33)  # 0.16 0.25 0.13, the ground plane
BALE = (184, 122, 31)  # 0.72 0.48 0.12, 202 of them on the speed course
BOARD = (135, 206, 235)  # 0.53 0.81 0.92, the signal's own board
ARM_RED = (217, 23, 18)  # 0.85 0.09 0.07
ARM_GREEN = (26, 179, 51)  # 0.10 0.70 0.20
RIBBON_RED = (230, 64, 51)  # 0.90 0.25 0.20, twenty car wash ribbons

# Where the arm lands from the start line.  Both courses stand the signal 8 ft
# down a 32 in lane, which puts the arm about 11 degrees off the lane axis and
# 10 degrees up, 2.9 m away: 42 px left of centre for a camera with a
# 110 degree field -- left, because a positive bearing is to port (REP-103)
# and image x grows to starboard -- and 41 px above it.  Measured through the
# simulated camera the red arm lands at (278, 139) on the obstacle course and
# (273, 139) on the speed course, and the green arm some 15 px higher.
ARM_X, ARM_Y = 278, 139
ARM_SIZE = (21, 16)


def scene() -> np.ndarray:
    """A frame with a horizon, bales along it and the signal's board."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = SKY
    frame[HEIGHT // 2 :] = GROUND
    # A run of bales along the horizon, as they appear down the lane.
    frame[HEIGHT // 2 - 14 : HEIGHT // 2 + 6] = BALE
    return patch(frame, BOARD, (ARM_X, ARM_Y + 12), (40, 60))


def patch(frame, colour, centre, size=ARM_SIZE) -> np.ndarray:
    frame = np.array(frame, dtype=np.uint8)
    x0, y0 = centre[0] - size[0] // 2, centre[1] - size[1] // 2
    frame[y0 : y0 + size[1], x0 : x0 + size[0]] = colour
    return frame


def arm(colour, centre=(ARM_X, ARM_Y), size=ARM_SIZE) -> np.ndarray:
    return patch(scene(), colour, centre, size)


def classify(frame, **kwargs) -> object:
    return detector.Classifier(**kwargs).classify(frame)


# --------------------------------------------------------------------- colour


def test_world_colours_are_where_the_bands_say_they_are():
    """The hue the bands are drawn around, for each colour in the worlds."""
    colours = np.array([[SKY, GROUND, BALE, BOARD, ARM_RED, ARM_GREEN]], dtype=np.uint8)
    hue, saturation, _value = detector.hsv(colours)
    assert hue[0].tolist() == pytest.approx([0, 105, 36, 197, 1.5, 130], abs=1.0)
    # The ground is green enough in hue to matter, which is why the band
    # starts at 110 and the region of interest stops at the horizon.
    assert saturation[0].tolist() == pytest.approx(
        [0.0, 0.48, 0.83, 0.43, 0.92, 0.86], abs=0.01
    )


def test_the_course_alone_is_not_a_signal():
    """Bales, ground, board and background must read as no signal at all."""
    observation = classify(scene())
    assert observation.state == detector.UNKNOWN
    assert (observation.red.pixels, observation.green.pixels) == (0, 0)
    assert observation.position is None


def test_red_arm_reads_red_where_it_stands():
    observation = classify(arm(ARM_RED))
    assert observation.state == detector.RED
    assert observation.red.pixels == ARM_SIZE[0] * ARM_SIZE[1]
    assert observation.position == pytest.approx((ARM_X, ARM_Y), abs=1.0)
    assert observation.green.pixels == 0


def test_green_arm_reads_green_where_it_stands():
    observation = classify(arm(ARM_GREEN))
    assert observation.state == detector.GREEN
    assert observation.green.pixels == ARM_SIZE[0] * ARM_SIZE[1]
    assert observation.position == pytest.approx((ARM_X, ARM_Y), abs=1.0)


def test_a_part_way_round_arm_still_reads():
    """Mid-sweep the arm is foreshortened; 70 px is what the sim shows."""
    observation = classify(arm(ARM_RED, size=(10, 7)))
    assert observation.state == detector.RED
    assert observation.red.pixels == 70


def test_an_arm_too_small_to_be_one_is_not_read():
    """Below min_pixels is noise, not a signal 4 m away."""
    assert classify(arm(ARM_RED, size=(2, 2))).state == detector.UNKNOWN


def test_scattered_matches_do_not_add_up_to_an_arm():
    """The threshold is on the densest window, not the whole region.

    Twenty times min_pixels of signal red, spread over the region of interest
    rather than gathered into a patch, is what a raw pixel count would trip
    over.
    """
    frame = scene()
    random = np.random.default_rng(20260916)
    columns = random.integers(0, WIDTH, 240)
    rows = random.integers(0, HEIGHT // 2, 240)
    frame[rows, columns] = ARM_RED

    observation = classify(frame)
    assert observation.red.total >= 200
    assert observation.state == detector.UNKNOWN


def test_the_car_wash_ribbons_read_as_red_when_they_are_in_frame():
    """Not a false positive to guard against here, but a real risk to state.

    The ribbons are the same red as the arms and hang in a block, so hue and
    cluster tests cannot tell them apart.  They sit at the far end of the
    obstacle course, behind the car at the start line and out of the region of
    interest by the time it is looking at them; what keeps them from starting
    a run is that a start needs green *where red was*, which the transition
    distance enforces.
    """
    observation = classify(patch(scene(), RIBBON_RED, (120, 100), (40, 80)))
    assert observation.state == detector.RED


def test_region_of_interest_ignores_what_is_outside_it():
    frame = arm(ARM_GREEN, centre=(120, 100))
    assert classify(frame).state == detector.GREEN
    narrow = detector.Region(0.5, 0.0, 1.0, 0.55)
    assert classify(frame, region=narrow).state == detector.UNKNOWN


def test_the_default_region_stops_at_the_horizon():
    """The ground plane fills the bottom half and is nearly green enough.

    Its hue is 105 against a band that starts at 110, which is not much margin
    to leave to whatever the renderer's lighting does to it.  So the frame here
    is a ground plane shifted well inside the band -- and still not a signal,
    because the region of interest stops at the horizon the ground lies below.
    """
    bluer = (33, 64, 45)
    shifted = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    shifted[:] = SKY
    shifted[HEIGHT // 2 :] = bluer
    assert detector.hsv(np.array([[bluer]], np.uint8))[0][0, 0] == pytest.approx(
        143.0, abs=1.0
    )
    assert classify(shifted, region=detector.Region()).state == detector.GREEN
    assert classify(shifted).state == detector.UNKNOWN


def test_thresholds_can_be_loosened_or_tightened():
    faded = patch(scene(), (150, 110, 105), (ARM_X, ARM_Y))  # desaturated red
    assert classify(faded).state == detector.UNKNOWN
    loose = detector.Thresholds(min_saturation=0.25)
    assert classify(faded, thresholds=loose).state == detector.RED


# ------------------------------------------------------------------- decoding


def image_bytes(frame, order, channels, step=None):
    height, width = frame.shape[:2]
    pixels = np.zeros((height, width, channels), dtype=np.uint8)
    for source, destination in enumerate(order):
        pixels[:, :, destination] = frame[:, :, source]
    if step is None:
        return pixels.tobytes()
    padded = np.zeros((height, step), dtype=np.uint8)
    padded[:, : width * channels] = pixels.reshape(height, width * channels)
    return padded.tobytes()


@pytest.mark.parametrize(
    ("encoding", "order", "channels"),
    [
        ("rgb8", (0, 1, 2), 3),
        ("bgr8", (2, 1, 0), 3),
        ("rgba8", (0, 1, 2), 4),
        ("bgra8", (2, 1, 0), 4),
        ("8UC3", (2, 1, 0), 3),
    ],
)
def test_every_encoding_reads_the_same_arm(encoding, order, channels):
    """The simulator publishes rgb8 and the ZED wrapper bgra8."""
    frame = arm(ARM_RED)
    data = image_bytes(frame, order, channels)
    decoded = detector.decode(encoding, WIDTH, HEIGHT, WIDTH * channels, data)
    assert np.array_equal(decoded, frame)
    assert classify(decoded).state == detector.RED


def test_row_padding_is_honoured():
    frame = arm(ARM_GREEN)
    step = WIDTH * 3 + 17
    data = image_bytes(frame, (0, 1, 2), 3, step=step)
    decoded = detector.decode("rgb8", WIDTH, HEIGHT, step, data)
    assert np.array_equal(decoded, frame)


def test_a_zero_step_is_taken_as_a_packed_row():
    frame = arm(ARM_GREEN)
    data = image_bytes(frame, (0, 1, 2), 3)
    assert np.array_equal(detector.decode("rgb8", WIDTH, HEIGHT, 0, data), frame)


def test_unreadable_images_are_refused_rather_than_guessed_at():
    with pytest.raises(detector.ImageFormatError, match="mono8"):
        detector.decode("mono8", 4, 4, 4, bytes(16))
    with pytest.raises(detector.ImageFormatError, match="short of"):
        detector.decode("rgb8", 4, 4, 12, bytes(12 * 3))
    with pytest.raises(detector.ImageFormatError, match="step"):
        detector.decode("rgb8", 4, 4, 6, bytes(4 * 4 * 3))
    with pytest.raises(detector.ImageFormatError, match="0x0"):
        detector.decode("rgb8", 0, 0, 0, b"")


# ---------------------------------------------------------------------- latch


def observed(state, position=(ARM_X, ARM_Y)):
    """An Observation of `state` at `position`, without rendering a frame."""
    cluster = detector.Cluster(pixels=180, total=180, x=position[0], y=position[1])
    empty = detector.EMPTY_CLUSTER
    return detector.Observation(
        state=state,
        red=cluster if state == detector.RED else empty,
        green=cluster if state == detector.GREEN else empty,
    )


def run(latch, states, position=(ARM_X, ARM_Y)):
    return [latch.update(observed(state, position)) for state in states]


def test_a_red_to_green_transition_starts_the_run():
    latch = detector.StartLatch()
    went = run(
        latch,
        [detector.RED] * 3 + [detector.UNKNOWN, detector.GREEN, detector.GREEN],
    )
    assert went == [False, False, False, False, False, True]
    assert latch.go


def test_the_sweeps_blind_frames_do_not_undo_the_red():
    """Both arms pass edge-on at 45 degrees; the count has to hold through it.

    The simulated sweep shows two blind frames at 15 Hz and more at the 5 Hz
    llvmpipe manages, so this is the case the latch exists to get right.
    """
    latch = detector.StartLatch()
    run(latch, [detector.RED] * 2)
    assert run(latch, [detector.UNKNOWN] * 6) == [False] * 6
    assert latch.armed
    assert run(latch, [detector.GREEN] * 2) == [False, True]


def test_green_alone_does_not_start_the_run():
    """Nothing showed red first, so this is a green thing, not a signal."""
    latch = detector.StartLatch()
    assert run(latch, [detector.GREEN] * 10) == [False] * 10
    assert latch.rejection(observed(detector.GREEN)).startswith("no confirmed red")


def test_green_alone_starts_the_run_when_red_is_not_required():
    latch = detector.StartLatch(require_red_first=False)
    assert run(latch, [detector.GREEN] * 3) == [False, True, True]


def test_green_somewhere_else_is_not_the_signal_turning():
    """Both arms turn about one pivot, so the transition happens in one place."""
    latch = detector.StartLatch()
    run(latch, [detector.RED] * 2)
    assert run(latch, [detector.GREEN] * 4, position=(120, 300)) == [False] * 4
    assert "px from where red was" in latch.rejection(
        observed(detector.GREEN, (120, 300))
    )
    # ...and the real transition, a few pixels off the red arm's centroid
    # because the green arm is a different shape, still starts the run.
    assert run(latch, [detector.GREEN] * 2, position=(ARM_X + 9, ARM_Y - 6)) == [
        False,
        True,
    ]


def test_the_transition_distance_can_be_switched_off():
    latch = detector.StartLatch(max_transition_distance=0.0)
    run(latch, [detector.RED] * 2)
    assert run(latch, [detector.GREEN] * 2, position=(10, 10)) == [False, True]


def test_one_green_frame_is_not_enough_by_default():
    latch = detector.StartLatch()
    run(latch, [detector.RED] * 2)
    assert latch.update(observed(detector.GREEN)) is False
    assert latch.update(observed(detector.GREEN)) is True


def test_the_confirmation_count_is_configurable():
    latch = detector.StartLatch(confirm_frames=4)
    states = [detector.RED] * 4 + [detector.GREEN] * 4
    assert run(latch, states) == [False] * 7 + [True]
    # A frame of red mid-confirmation means the arm was not where the last
    # three frames suggested, so the green count starts again.
    latch.reset()
    states = [detector.RED] * 4 + [detector.GREEN] * 3 + [detector.RED, detector.GREEN]
    assert not any(run(latch, states))
    assert latch.green_frames == 1


def test_a_returning_red_does_not_unlatch_a_started_run():
    """The latch is the start trigger, not a live view of the signal.

    A driver watching for a red flag mid-run wants ~/state, which reports
    every frame; ~/go answers "has the run been started" and a momentary
    mis-hued frame must not answer that with no.
    """
    latch = detector.StartLatch()
    run(latch, [detector.RED] * 2 + [detector.GREEN] * 2)
    assert all(run(latch, [detector.RED] * 5))
    assert latch.go


def test_reset_waits_for_another_start():
    latch = detector.StartLatch()
    run(latch, [detector.RED] * 2 + [detector.GREEN] * 2)
    latch.reset()
    assert not latch.go and not latch.armed
    assert run(latch, [detector.GREEN] * 4) == [False] * 4
    assert run(latch, [detector.RED] * 2 + [detector.GREEN] * 2)[-1]


def test_the_simulated_sweep_starts_the_run_when_the_arm_has_turned():
    """The frame-by-frame sequence a real sweep produced, replayed in full.

    Pixel counts are the ones measured through the simulated camera and
    quoted in the README: the arms trade projected area as they turn, so the
    verdict follows the crossover rather than the first green pixel.  What
    matters is where in the sequence `go` comes up.
    """
    sweep = [
        (0.00, 246, 0),
        (0.46, 234, 119),
        (0.59, 211, 179),
        (0.66, 188, 205),
        (0.79, 121, 233),
        (0.99, 24, 246),
    ]
    latch = detector.StartLatch()
    thresholds = detector.Thresholds()
    went = []
    for _time, red, green in sweep:
        if max(red, green) < thresholds.min_pixels:
            state = detector.UNKNOWN
        else:
            state = detector.RED if red >= green else detector.GREEN
        went.append(latch.update(observed(state)))
    assert went == [False, False, False, False, True, True]


# ------------------------------------------------------------------- annotate


def test_annotation_boxes_the_arm_without_touching_the_frame():
    frame = arm(ARM_RED)
    observation = classify(frame)
    marked = detector.annotate(frame, observation, detector.DEFAULT_REGION)

    assert marked.shape == frame.shape
    assert np.array_equal(frame, arm(ARM_RED))  # unchanged
    box = marked[ARM_Y - 12 : ARM_Y + 12, ARM_X - 12 : ARM_X + 12]
    assert (box == (255, 0, 0)).all(axis=-1).sum() >= 40
    assert (marked[HEIGHT // 2 - 1] == (128, 128, 128)).all(axis=-1).any()


def test_annotation_of_a_frame_with_no_signal_still_shows_the_region():
    marked = detector.annotate(scene(), classify(scene()), detector.Region(0.25, 0.25))
    assert (marked[HEIGHT // 4, WIDTH // 4 :] == (128, 128, 128)).all(axis=-1).any()


# ------------------------------------------------------------- region geometry


def test_regions_cover_whole_pixels_and_reject_nonsense():
    assert detector.Region().pixels(640, 360) == (0, 0, 640, 360)
    assert detector.DEFAULT_REGION.pixels(640, 360) == (0, 0, 640, 180)
    quarter = detector.Region(0.5, 0.5, 0.75, 0.75)
    assert quarter.pixels(640, 360) == (320, 180, 480, 270)
    # A region far smaller than one pixel still has one to look at.
    assert detector.Region(0.0, 0.0, 1e-6, 1e-6).pixels(640, 360) == (0, 0, 1, 1)
    for bad in ((0.5, 0.0, 0.5, 1.0), (0.0, 0.8, 1.0, 0.2), (-0.1, 0.0, 1.0, 1.0)):
        with pytest.raises(ValueError, match="fractions"):
            detector.Region(*bad)


def test_hue_bands_wrap_through_zero():
    hue = np.array([0.0, 14.0, 16.0, 130.0, 344.0, 346.0, 359.0])
    red = detector.HueBand(345.0, 15.0).mask(hue).tolist()
    assert red == [True, True, False, False, False, True, True]
    green = detector.HueBand(110.0, 170.0).mask(hue).tolist()
    assert green == [False, False, False, True, False, False, False]


def test_the_densest_window_reports_the_centroid_in_frame_coordinates():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:14, 20:24] = True
    cluster = detector.densest(mask, 24, offset=(100, 200))
    assert (cluster.pixels, cluster.total) == (16, 16)
    assert (cluster.x, cluster.y) == pytest.approx((121.5, 211.5))
    assert detector.densest(np.zeros((40, 40), bool), 24, (0, 0)).pixels == 0


def test_the_densest_window_is_not_confused_by_a_window_larger_than_the_mask():
    mask = np.ones((6, 8), dtype=bool)
    cluster = detector.densest(mask, 24, offset=(0, 0))
    assert cluster.pixels == 48
    assert (cluster.x, cluster.y) == pytest.approx((3.5, 2.5))


def test_observations_describe_themselves_for_a_log_line():
    assert f"red at ({ARM_X}, {ARM_Y}), red 180/180 px" in str(observed(detector.RED))
    assert str(observed(detector.UNKNOWN)).startswith("unknown, red 0/0 px")


def test_the_transition_distance_is_measured_in_pixels():
    latch = detector.StartLatch(max_transition_distance=30.0)
    run(latch, [detector.RED] * 2)
    offset = (ARM_X + 25, ARM_Y + 25)
    assert math.dist((ARM_X, ARM_Y), offset) > 30.0
    assert not any(run(latch, [detector.GREEN] * 3, position=offset))
