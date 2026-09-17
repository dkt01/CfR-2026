"""Tests for the start signal detector's color decision and its latch.

The frames here are synthetic, built from the diffuse colors the two worlds
actually use, so they run with no camera, no simulator and no ROS.  What they
cannot check is that Gazebo renders those colors where this expects them;
`scripts/check_start_signal.py` does that against a running simulation.

The course is outdoors, so a synthetic frame in the simulator's flat lighting
is not the hard case.  `under` re-lights a whole frame the way weather does --
scaled for shade or sun, and washed with the flat light that sky glare and a
clipping sensor both add -- and the section marked "outdoors" replays starts
through it, with people and hedges in frame.
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

# The worlds' diffuse colors, in 8-bit.  Every one of these lands in frame
# from the start line, so the detector has to pick the arms out from among
# them rather than merely off a black background.
SKY = (128, 128, 128)  # no sky model; the background renders flat gray
GROUND = (41, 64, 33)  # 0.16 0.25 0.13, the ground plane
BALE = (184, 122, 31)  # 0.72 0.48 0.12, 202 of them on the speed course
BOARD = (0, 162, 234)  # 0.00 0.64 0.92, the signal's own board (Oasis Blue)
ARM_RED = (226, 53, 37)  # 0.89 0.21 0.15, Poppy Red
ARM_GREEN = (88, 154, 80)  # 0.35 0.60 0.31, Leafy Green
RIBBON_RED = (230, 64, 51)  # 0.90 0.25 0.20, twenty car wash ribbons

# Things an outdoor course puts in frame that the simulator does not.  The
# shirt is the same red as the signal on purpose -- that is the point of it --
# and the foliage is the green of a hedge in sun, which is not.
SHIRT_RED = (180, 30, 40)
SHIRT_GREEN = (40, 170, 90)
FOLIAGE = (70, 110, 60)

# Where the arm lands from the start line.  Both courses stand the signal 8 ft
# down a 32 in lane, which puts the arm about 11 degrees off the lane axis and
# 10 degrees up, 2.9 m away: 42 px left of center for a camera with a
# 110 degree field -- left, because a positive bearing is to port (REP-103)
# and image x grows to starboard -- and 41 px above it.  Measured through the
# simulated camera the red arm lands at (278, 139) on the obstacle course and
# (273, 139) on the speed course, and the green arm some 15 px higher.
ARM_X, ARM_Y = 278, 139
ARM_SIZE = (21, 16)

# Where a person stands in these frames: off to one side of the signal, well
# outside the box the detector ends up watching, and the size somebody 5 m
# away subtends.
SHIRT_AT = (460, 80)
SHIRT_SIZE = (46, 64)


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


def shirt(frame, colour=SHIRT_RED) -> np.ndarray:
    """Somebody standing beside the course in a shirt the signal's color."""
    return patch(frame, colour, SHIRT_AT, SHIRT_SIZE)


def under(frame, gain: float, glare: int = 0) -> np.ndarray:
    """A frame under different light: scaled, then washed with flat light.

    Two effects, because they are the two that cost the detector anything.
    `gain` is how much light there is -- deep shade through to direct sun --
    and leaves hue and saturation alone, which is why they are what the bands
    are drawn on.  `glare` is light *added* to everything: sky glare over a
    backlit signal, and the same arithmetic as a sensor clipping in the sun.
    That is what costs saturation, and it is what the floors are set for.
    """
    return np.clip(frame.astype(np.float32) * gain + glare, 0, 255).astype(np.uint8)


# Backlit: the sun has come round behind the signal, the arm's face is in
# shade and the sky behind it is glaring over the lens.  A quarter of the
# light and a wash that leaves the arms at (255, 228, 224) and (237, 253,
# 235) -- hue 8 and 113 still, chroma 0.12 and 0.07 still, but saturation
# 0.12 and 0.07.  Both are under the floor a frame has to clear before the
# signal has been found; Leafy Green's own saturation is low enough that it
# only just clears the relaxed one that applies afterwards, which is why
# `focus_relaxation` sits lower than a more saturated green would need.
BACKLIT = dict(gain=0.25, glare=215)


def classify(frame, focus=None, **kwargs):
    return detector.Classifier(**kwargs).classify(frame, focus)


def watching(position=(ARM_X, ARM_Y), radius=60.0):
    return detector.Focus(position[0], position[1], radius)


def replay(frames, classifier=None, latch=None):
    """Run frames through a whole Detector, as the node runs them.

    Returns the detector and the per-frame observations, so a test can assert
    on both the trigger and what each frame was read as.
    """
    unit = detector.Detector(classifier, latch)
    return unit, [unit.process(frame) for frame in frames]


def states(observations):
    return [observation.state for observation in observations]


# --------------------------------------------------------------------- color


def test_world_colours_are_where_the_bands_say_they_are():
    """The hue the bands are drawn around, for each color in the worlds."""
    colours = np.array([[SKY, GROUND, BALE, BOARD, ARM_RED, ARM_GREEN]], dtype=np.uint8)
    hue, saturation, _value = detector.hsv(colours)
    assert hue[0].tolist() == pytest.approx([0, 105, 36, 198, 5, 114], abs=1.0)
    # Gray has no hue at all, which is why saturation and not hue is what
    # excludes it.  Leafy Green's own saturation (0.48) is much lower than
    # Poppy Red's (0.84) or the board's (1.0, since its blue channel alone
    # carries all the light) -- that gap is what drives the thinner margins
    # elsewhere in this module.
    assert saturation[0].tolist() == pytest.approx(
        [0.0, 0.48, 0.83, 1.0, 0.84, 0.48], abs=0.02
    )

    bands = detector.Thresholds()
    for colour, red, green in (
        (BALE, False, False),
        (GROUND, False, False),
        (BOARD, False, False),
        (ARM_RED, True, False),
        (ARM_GREEN, False, True),
        (SHIRT_RED, True, False),
        (SHIRT_GREEN, False, True),
        (FOLIAGE, False, False),
    ):
        one = np.array([[colour]], dtype=np.uint8)
        assert bands.red.mask(detector.hsv(one)[0])[0, 0] == red, colour
        assert bands.green.mask(detector.hsv(one)[0])[0, 0] == green, colour


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
    # It fits inside one window and there is nothing else of its color
    # around it, which is what says it is an arm and not part of something.
    assert observation.red.spread == pytest.approx(1.0)


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
    observation = classify(arm(ARM_RED, size=(2, 2)))
    assert observation.state == detector.UNKNOWN
    # ...and the frame still says what it saw and why it does not count.
    assert observation.red.pixels == 4
    assert "under the 12 px" in observation.red_reject


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


def test_region_of_interest_ignores_what_is_outside_it():
    frame = arm(ARM_GREEN, centre=(120, 100))
    assert classify(frame).state == detector.GREEN
    narrow = detector.Region(0.5, 0.0, 1.0, 0.5)
    assert classify(frame, region=narrow).state == detector.UNKNOWN


def test_the_ground_is_not_a_signal_even_when_it_is_green_enough():
    """Turf, dirt and the simulated ground plane all reach for the green band.

    Two things keep the ground out and either would do: the region of
    interest stops at the horizon it lies below, and a hillside's worth of
    green is nothing like the size of an arm.  The frame here is a ground
    plane shifted well inside the band, to check both.
    """
    bluer = (33, 64, 45)
    shifted = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    shifted[:] = SKY
    shifted[HEIGHT // 2 :] = bluer
    assert detector.hsv(np.array([[bluer]], np.uint8))[0][0, 0] == pytest.approx(
        143.0, abs=1.0
    )

    whole = classify(shifted, region=detector.Region())
    assert whole.state == detector.UNKNOWN
    assert whole.green.total > 50000  # the hue matches, in quantity
    assert "much larger" in whole.green_reject

    # And within the default region there is none of it to reject.
    assert classify(shifted).green.total == 0


def test_thresholds_can_be_loosened_or_tightened():
    faded = patch(scene(), (150, 132, 130), (ARM_X, ARM_Y))  # hue 6, sat 0.13
    assert classify(faded).state == detector.UNKNOWN
    loose = detector.Thresholds(min_saturation=0.10)
    assert classify(faded, thresholds=loose).state == detector.RED


# ------------------------------------------------------------------- outdoors


def test_a_backlit_arm_is_still_read():
    """The sun behind the signal: the arm's face is in shade, under glare.

    This is the case the color floors exist for.  Saturation is what glare
    costs, so the bar for it is set low and an absolute chroma floor keeps
    the noise out instead -- the arm here has lost four fifths of its light
    and over half its saturation, and still reads red.
    """
    backlit = under(arm(ARM_RED), gain=0.25, glare=140)
    hue, saturation, value = detector.hsv(backlit[ARM_Y : ARM_Y + 1, ARM_X : ARM_X + 1])
    assert hue[0, 0] == pytest.approx(5.1, abs=1.0)
    # The glare has taken most of the saturation and none of the chroma,
    # which is why chroma is the floor that matters.
    assert saturation[0, 0] == pytest.approx(0.24, abs=0.02)
    assert saturation[0, 0] * value[0, 0] == pytest.approx(0.18, abs=0.02)
    assert value[0, 0] == pytest.approx(0.77, abs=0.03)

    assert classify(backlit).state == detector.RED
    # Under the floors this used to carry, the same frame read as nothing.
    was = detector.Thresholds(min_saturation=0.45, min_value=0.15)
    assert classify(backlit, thresholds=was).state == detector.UNKNOWN


def test_an_arm_in_direct_sun_is_still_read():
    """Clipping costs saturation the same way glare does: red, but paler.

    Poppy Red's hue drifts towards orange under clipping faster than a
    placeholder red does -- full clipping (gain 1.4 here) pushes it past even
    the widened 16 degree band, so this stops at gain 1.2, which is what the
    red band's margin against skin actually covers.
    """
    glaring = under(arm(ARM_RED), gain=1.2, glare=130)
    assert glaring[ARM_Y, ARM_X].tolist() == [255, 193, 174]
    assert classify(glaring).state == detector.RED
    was = detector.Thresholds(min_saturation=0.45, min_value=0.15)
    assert classify(glaring, thresholds=was).state == detector.UNKNOWN


def test_an_arm_in_deep_shade_is_still_read():
    """A dark frame is not a washed one: hue and saturation both survive it.

    Leafy Green's own chroma is low enough that a quarter of the light would
    already put it under a stricter floor than this one; `min_chroma` sits at
    0.03 rather than a placeholder green's 0.04 because of exactly this case.
    """
    dusk = under(arm(ARM_GREEN), gain=0.12)
    assert classify(dusk).state == detector.GREEN
    was = detector.Thresholds(min_saturation=0.45, min_value=0.15)
    assert classify(dusk, thresholds=was).state == detector.UNKNOWN


def test_dark_noise_is_not_an_arm():
    """What the chroma floor is for: saturation alone believes the dark.

    A nearly black pixel with a unit or two of sensor noise in the red
    channel has a convincing hue and a saturation of 0.8, because saturation
    is a ratio.  Its chroma says what it really is.
    """
    noise = patch(under(scene(), gain=0.1), (6, 1, 1), (ARM_X, ARM_Y))
    _hue, saturation, _value = detector.hsv(noise[ARM_Y : ARM_Y + 1, ARM_X : ARM_X + 1])
    assert saturation[0, 0] > 0.75
    assert classify(noise).state == detector.UNKNOWN


def test_a_shirt_the_signals_colour_is_not_an_arm():
    """Somebody standing by the course in a red shirt, and in a green one.

    Hue cannot tell a shirt from an arm -- it is the same red -- so size
    does.  A person 5 m away is several windows wide; the arm at 3 m fits
    inside one.
    """
    for colour in (SHIRT_RED, SHIRT_GREEN):
        observation = classify(shirt(scene(), colour))
        assert observation.state == detector.UNKNOWN, colour
        rejected = observation.red_reject or observation.green_reject
        assert "much larger" in rejected, colour


def test_someone_close_to_the_camera_is_not_an_arm():
    """Nor is a shirt that fills a third of the frame, which is the easy case."""
    close = patch(scene(), SHIRT_RED, (200, 90), (160, 170))
    assert classify(close).state == detector.UNKNOWN


def test_a_hedge_is_not_a_green_arm():
    """Foliage above the horizon, both as it really is and as the band's own green.

    A hedge in sun sits at hue 108, below the band, which is why the band
    starts where it does.  One that happened to match the signal's green
    exactly would still be thrown out on its size.
    """
    assert classify(patch(scene(), FOLIAGE, (500, 60), (220, 90))).state == (
        detector.UNKNOWN
    )
    exact = patch(scene(), ARM_GREEN, (500, 60), (220, 90))
    observation = classify(exact)
    assert observation.state == detector.UNKNOWN
    assert "much larger" in observation.green_reject


def test_a_shirt_does_not_hide_the_arm_behind_it():
    """The signal has to be found in spite of a bigger patch of its own color.

    Taking the densest cluster in frame and stopping there would report the
    shirt: it is 576 px against the arm's 336.  Several candidates are
    searched, so the arm is found as well and it is the one that is
    arm-sized.
    """
    frame = shirt(arm(ARM_RED))
    limits = detector.Thresholds()
    biggest = detector.densest(
        limits.mask(limits.red, detector.hsv(frame[:180])),
        limits.cluster_window,
        (0, 0),
    )
    assert biggest.pixels > classify(frame).red.pixels
    assert abs(biggest.x - SHIRT_AT[0]) < SHIRT_SIZE[0]
    assert abs(biggest.y - SHIRT_AT[1]) < SHIRT_SIZE[1]

    observation = classify(frame)
    assert observation.state == detector.RED
    assert observation.position == pytest.approx((ARM_X, ARM_Y), abs=1.0)
    assert len(observation.reds) == 1


def test_a_start_still_triggers_with_people_in_frame():
    """The whole thing, in a crowd: red arm, people about, arm turns green.

    A red shirt on one side and a green one on the other, both bigger than
    the arm and neither of them it.
    """

    def crowded(colour):
        return shirt(shirt(arm(colour), SHIRT_RED), SHIRT_GREEN)

    unit, observations = replay([crowded(ARM_RED)] * 6 + [crowded(ARM_GREEN)] * 2)
    assert states(observations) == [detector.RED] * 6 + [detector.GREEN] * 2
    assert unit.go
    assert unit.latch.started_at == pytest.approx((ARM_X, ARM_Y), abs=8.0)


def test_people_alone_never_start_a_run():
    """Nobody walking about in front of the course can release the car.

    Red shirt, then a green one in the same place, which is the shape of a
    false start: neither is ever an arm-sized candidate, so no site is ever
    armed and `~/go` stays down.
    """
    red = shirt(scene(), SHIRT_RED)
    green = shirt(scene(), SHIRT_GREEN)
    unit, observations = replay([red] * 20 + [green] * 20)
    assert not unit.go
    assert not unit.latch.armed
    assert set(states(observations)) == {detector.UNKNOWN}


def test_the_sun_coming_round_behind_the_signal_does_not_lose_it():
    """A start read on the relaxed floors that apply once the signal is found.

    The washed arm here is under the floor a frame has to clear to be taken
    for the signal in the first place -- deliberately, because that floor is
    also what keeps the rest of the course out -- and over the relaxed one
    that applies inside the box around a signal already found.  So the same
    frame reads as nothing on its own and as green to a detector that knows
    where to look, which is what carries a start through the light changing
    mid-wait.
    """
    washed_green = under(arm(ARM_GREEN), **BACKLIT)
    assert classify(washed_green).state == detector.UNKNOWN
    assert classify(washed_green, focus=watching()).state == detector.GREEN

    washed_red = under(arm(ARM_RED), **BACKLIT)
    unit, observations = replay(
        [arm(ARM_RED)] * 6 + [washed_red] * 3 + [washed_green] * 2
    )
    assert states(observations) == [detector.RED] * 9 + [detector.GREEN] * 2
    assert unit.go


def test_the_relaxed_floors_do_not_let_the_box_read_anything_at_all():
    """Relaxing the color floors inside the box does not relax the rest.

    Straw is the thing nearest the red band with something to spare, and the
    bales are right under the signal: 20 rows of them cross the box the
    detector watches.  Nothing in the box reads as an arm on hue alone.
    """
    observation = classify(scene(), focus=watching())
    assert observation.state == detector.UNKNOWN
    assert (observation.red.total, observation.green.total) == (0, 0)


# ------------------------------------------------------------------- decoding


ENCODED = [
    ("rgb8", (0, 1, 2), 3),
    ("bgr8", (2, 1, 0), 3),
    ("rgba8", (0, 1, 2), 4),
    ("bgra8", (2, 1, 0), 4),
    ("8UC3", (2, 1, 0), 3),
    ("8UC4", (2, 1, 0), 4),
]


@pytest.mark.parametrize("encoding,order,channels", ENCODED)
def test_every_encoding_reads_the_same_arm(encoding, order, channels):
    frame = arm(ARM_RED)
    raw = np.zeros((HEIGHT, WIDTH, channels), dtype=np.uint8)
    for source, target in enumerate(order):
        raw[:, :, target] = frame[:, :, source]
    decoded = detector.decode(encoding, WIDTH, HEIGHT, 0, raw.tobytes())
    assert np.array_equal(decoded, frame)


def test_row_padding_is_honoured():
    frame = arm(ARM_GREEN)
    step = WIDTH * 3 + 17
    padded = np.zeros((HEIGHT, step), dtype=np.uint8)
    padded[:, : WIDTH * 3] = frame.reshape(HEIGHT, WIDTH * 3)
    decoded = detector.decode("rgb8", WIDTH, HEIGHT, step, padded.tobytes())
    assert np.array_equal(decoded, frame)


def test_a_zero_step_is_taken_as_a_packed_row():
    frame = arm(ARM_RED)
    decoded = detector.decode("rgb8", WIDTH, HEIGHT, 0, frame.tobytes())
    assert np.array_equal(decoded, frame)


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
    cluster = detector.Cluster(
        pixels=180, total=180, x=position[0], y=position[1], surround=180
    )
    empty = detector.EMPTY_CLUSTER
    red = cluster if state == detector.RED else empty
    green = cluster if state == detector.GREEN else empty
    return detector.Observation(
        state=state,
        red=red,
        green=green,
        reds=(red,) if state == detector.RED else (),
        greens=(green,) if state == detector.GREEN else (),
    )


def run(latch, sequence, position=(ARM_X, ARM_Y)):
    return [latch.update(observed(state, position)) for state in sequence]


def armed(latch, position=(ARM_X, ARM_Y)):
    """Hold red at `position` for as long as it takes to be taken for the signal."""
    run(latch, [detector.RED] * latch.arm_frames, position)
    assert latch.armed
    return latch


def test_a_red_to_green_transition_starts_the_run():
    latch = detector.StartLatch()
    went = run(
        latch,
        [detector.RED] * 5 + [detector.UNKNOWN, detector.GREEN, detector.GREEN],
    )
    assert went == [False] * 7 + [True]
    assert latch.go


def test_red_has_to_hold_in_one_place_before_it_is_the_signal():
    """Four frames of red is not the signal; four frames in one place is.

    Waiting is free -- the car stands at the line until the flag drops -- and
    what it buys is that nothing passing through frame can be mistaken for a
    signal that has stood there all along.
    """
    latch = detector.StartLatch(arm_frames=4)
    wandering = [(100, 40), (200, 60), (300, 80), (400, 100)]
    for position in wandering:
        run(latch, [detector.RED], position)
    assert not latch.armed
    assert latch.red_frames == 1

    run(latch, [detector.RED] * 4, (ARM_X, ARM_Y))
    assert latch.armed


def test_the_sweeps_blind_frames_do_not_undo_the_red():
    """Both arms pass edge-on at 45 degrees; the count has to hold through it.

    The simulated sweep shows two blind frames at 15 Hz and more at the 5 Hz
    llvmpipe manages, so this is the case the latch exists to get right.
    """
    latch = armed(detector.StartLatch())
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
    latch = armed(detector.StartLatch())
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


def test_only_the_place_that_turned_green_starts_the_run():
    """Two red things in frame, both arm-sized, and one of them turns.

    A start is a place going from red to green, so what the other place is
    doing has no bearing on it -- it neither starts the run by being red nor
    holds the run back by staying red.
    """
    decoy = (ARM_X - 180, ARM_Y - 40)
    latch = detector.StartLatch()
    for _ in range(latch.arm_frames):
        latch.update(both(detector.RED, detector.RED, decoy))
    assert latch.armed
    assert not latch.update(both(detector.GREEN, detector.RED, decoy))
    assert latch.update(both(detector.GREEN, detector.RED, decoy))
    assert latch.started_at == pytest.approx((ARM_X, ARM_Y), abs=6.0)


def both(signal_state, decoy_state, decoy):
    """One observation with the signal in one state and a decoy in another."""
    signal = detector.Cluster(pixels=180, total=360, x=ARM_X, y=ARM_Y, surround=180)
    other = detector.Cluster(
        pixels=200, total=360, x=decoy[0], y=decoy[1], surround=200
    )
    clusters = {detector.RED: [], detector.GREEN: []}
    clusters[signal_state].append(signal)
    clusters[decoy_state].append(other)
    return detector.Observation(
        state=signal_state,
        red=clusters[detector.RED][0]
        if clusters[detector.RED]
        else (detector.EMPTY_CLUSTER),
        green=clusters[detector.GREEN][0]
        if clusters[detector.GREEN]
        else (detector.EMPTY_CLUSTER),
        reds=tuple(clusters[detector.RED]),
        greens=tuple(clusters[detector.GREEN]),
    )


def test_a_place_that_goes_quiet_is_forgotten():
    """A red that leaves cannot pair up with a green that arrives later.

    The signal is seen every frame while the car waits, so the only thing
    dropping a place costs is the memory of one that has gone -- and what it
    buys is that a red object removed from a spot and a green one put there a
    second later is not a transition.
    """
    latch = armed(detector.StartLatch(forget_frames=10))
    run(latch, [detector.UNKNOWN] * 10)
    assert not latch.armed and latch.sites == []
    assert not any(run(latch, [detector.GREEN] * 4))

    # ...and having forgotten it, it can find it again.
    latch = armed(detector.StartLatch(forget_frames=10))
    assert run(latch, [detector.UNKNOWN] * 9 + [detector.GREEN, detector.GREEN])[-1]


def test_the_places_carried_at_once_are_capped():
    """A busy scene cannot grow the list without bound; the best reds stay."""
    latch = detector.StartLatch(max_sites=3, arm_frames=2)
    run(latch, [detector.RED] * 5, (100, 100))
    for index in range(10):
        run(latch, [detector.RED], (300 + 30 * index, 40))
    assert len(latch.sites) == 3
    assert latch.signal.position == pytest.approx((100, 100), abs=1.0)


def test_the_transition_distance_can_be_switched_off():
    latch = armed(detector.StartLatch(max_transition_distance=0.0))
    assert run(latch, [detector.GREEN] * 2, position=(10, 10)) == [False, True]
    assert latch.focus() is None


def test_one_green_frame_is_not_enough_by_default():
    latch = armed(detector.StartLatch())
    assert latch.update(observed(detector.GREEN)) is False
    assert latch.update(observed(detector.GREEN)) is True


def test_the_confirmation_count_is_configurable():
    latch = armed(detector.StartLatch(confirm_frames=4))
    assert run(latch, [detector.GREEN] * 4) == [False] * 3 + [True]
    # A frame of red mid-confirmation means the arm was not where the last
    # three frames suggested, so the green count starts again.
    latch.reset()
    armed(latch)
    sequence = [detector.GREEN] * 3 + [detector.RED, detector.GREEN]
    assert not any(run(latch, sequence))
    assert latch.green_frames == 1


def test_a_returning_red_does_not_unlatch_a_started_run():
    """The latch is the start trigger, not a live view of the signal.

    A driver watching for a red flag mid-run wants ~/state, which reports
    every frame; ~/go answers "has the run been started" and a momentary
    mis-hued frame must not answer that with no.
    """
    latch = armed(detector.StartLatch())
    run(latch, [detector.GREEN] * 2)
    assert all(run(latch, [detector.RED] * 5))
    assert latch.go


def test_reset_waits_for_another_start():
    latch = armed(detector.StartLatch())
    run(latch, [detector.GREEN] * 2)
    latch.reset()
    assert not latch.go and not latch.armed and latch.focus() is None
    assert run(latch, [detector.GREEN] * 4) == [False] * 4
    assert run(latch, [detector.RED] * 5 + [detector.GREEN] * 2)[-1]


def test_the_focus_follows_the_signal_and_is_dropped_when_it_goes():
    latch = detector.StartLatch(arm_frames=3, forget_frames=5)
    assert latch.focus() is None  # nothing found yet: search everything
    run(latch, [detector.RED] * 3, (300, 120))
    focus = latch.focus()
    assert (focus.x, focus.y) == pytest.approx((300, 120), abs=1.0)
    assert focus.radius == latch.max_transition_distance
    assert focus.holds((330, 100)) and not focus.holds((300, 220))

    run(latch, [detector.UNKNOWN] * 5)
    assert latch.focus() is None


# One sweep through the simulated camera at 15 Hz, from
# scripts/check_start_signal.py: seconds after the sweep was commanded, and
# the pixels of each color in the densest window.  Both arms are in frame
# for most of it, which is the shape the latch has to read: they are 90
# degrees apart on one pivot, so they trade projected area and the total
# stays near 250 px.
SWEEP = [
    (0.00, 249, 0),
    (0.07, 249, 0),
    (0.13, 249, 0),
    (0.20, 247, 0),
    (0.26, 244, 6),
    (0.33, 243, 25),
    (0.40, 240, 53),
    (0.46, 237, 91),
    (0.53, 230, 123),
    (0.59, 226, 155),
    (0.66, 208, 185),
    (0.73, 182, 208),  # green takes over
    (0.79, 137, 230),
    (0.86, 100, 237),
    (0.92, 63, 242),
    (0.99, 30, 249),
    (1.06, 5, 246),
]


def turning(red_pixels, green_pixels):
    """One frame of a sweep, with both arms in frame at their own centroids."""
    limits = detector.Thresholds()
    total = red_pixels + green_pixels
    red = detector.Cluster(
        pixels=red_pixels, total=total, x=ARM_X, y=ARM_Y, surround=red_pixels
    )
    green = detector.Cluster(
        pixels=green_pixels,
        total=total,
        x=ARM_X - 1,
        y=ARM_Y - 12,
        surround=green_pixels,
    )
    reds = (red,) if red_pixels >= limits.min_pixels else ()
    greens = (green,) if green_pixels >= limits.min_pixels else ()
    if not reds and not greens:
        state = detector.UNKNOWN
    else:
        state = detector.RED if red_pixels >= green_pixels else detector.GREEN
    return detector.Observation(
        state=state,
        red=red if reds else detector.EMPTY_CLUSTER,
        green=green if greens else detector.EMPTY_CLUSTER,
        reds=reds,
        greens=greens,
    )


def test_the_simulated_sweep_starts_the_run_when_the_arm_has_turned():
    """The frame-by-frame sequence a real sweep produced, replayed in full.

    What matters is where in the sequence `go` comes up: on the second frame
    after the green arm took the place over, and not a frame later.  Both
    arms are in frame together for two thirds of the turn, so a latch that
    counted them separately would sit at one green frame until the red arm
    had gone edge-on altogether -- another 0.3 s of the run, and longer at
    the 5 Hz a software-rendered world manages.
    """
    latch = armed(detector.StartLatch())  # the wait at the line
    went = [latch.update(turning(red, green)) for _time, red, green in SWEEP]

    crossover = next(
        index for index, (_time, red, green) in enumerate(SWEEP) if green > red
    )
    assert went.index(True) == crossover + 1
    assert went == [False] * (crossover + 1) + [True] * (len(SWEEP) - crossover - 1)
    # 0.13 s after the arms crossed over, at the camera's 15 Hz.
    assert SWEEP[went.index(True)][0] - SWEEP[crossover][0] == pytest.approx(
        0.06, abs=0.01
    )


def test_both_arms_in_frame_at_once_still_confirms_the_green_one():
    """The two arms overlap through the turn, and only the fuller one counts.

    Reduced from the sweep above to the two frames that matter, because this
    is the difference between starting when the signal turns and starting
    when it has finished turning.
    """
    latch = armed(detector.StartLatch())
    assert not latch.update(turning(182, 208))
    assert latch.update(turning(137, 230))
    assert latch.sites[0].green_frames == 2


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


def test_annotation_shows_the_box_the_detector_is_watching():
    frame = arm(ARM_GREEN)
    focus = watching((200, 100), 40.0)
    marked = detector.annotate(frame, classify(frame), detector.DEFAULT_REGION, focus)
    assert (marked[60, 160:241] == (255, 255, 255)).all(axis=-1).any()
    assert (marked[60:141, 160] == (255, 255, 255)).all(axis=-1).any()


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


def test_the_focus_box_is_clipped_to_the_frame():
    assert watching((100, 100), 20.0).box(640, 360) == (80, 80, 121, 121)
    assert watching((10, 5), 20.0).box(640, 360) == (0, 0, 31, 26)
    assert watching((630, 350), 20.0).box(640, 360) == (610, 330, 640, 360)


def test_hue_bands_wrap_through_zero():
    hue = np.array([0.0, 11.0, 17.0, 130.0, 337.0, 339.0, 359.0])
    bands = detector.Thresholds()
    assert bands.red.mask(hue).tolist() == [True, True, False, False, False, True, True]
    assert bands.green.mask(hue).tolist() == [False] * 3 + [True] + [False] * 3


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


def test_candidates_are_distinct_places_biggest_first():
    """Three patches, and the same patch is not reported twice."""
    mask = np.zeros((200, 300), dtype=bool)
    mask[20:32, 40:52] = True  # 144 px
    mask[100:116, 150:166] = True  # 256 px
    mask[150:154, 250:254] = True  # 16 px
    found, blocked = detector.clusters(mask, 24, (0, 0), limit=6)
    assert [cluster.pixels for cluster in found] == [256, 144, 16]
    assert [round(cluster.x) for cluster in found] == [158, 46, 252]
    assert blocked is None
    assert detector.clusters(mask, 24, (0, 0), limit=2)[0][-1].pixels == 144


def test_a_blob_uses_up_no_candidates_and_is_reported_on_its_own():
    """A person in frame must not crowd the arm out of the candidate list.

    Six windows fit inside a shirt, so counting them as candidates would fill
    the list with the same person six times over and never reach the arm.  A
    box that spreads too far is skipped instead, and the search steps a whole
    surround past it.
    """
    mask = np.zeros((200, 300), dtype=bool)
    mask[20:84, 30:76] = True  # a shirt, 46 x 64
    mask[100:116, 200:221] = True  # an arm, 21 x 16
    found, blocked = detector.clusters(mask, 24, (0, 0), 6, max_spread=3.0)
    assert [cluster.pixels for cluster in found] == [336]
    assert found[0].position == pytest.approx((210, 107.5), abs=1.0)
    assert blocked.pixels == 576 and blocked.spread == pytest.approx(3.83, abs=0.05)


def test_a_clusters_spread_measures_what_surrounds_it():
    """An arm inside its window spreads by 1; a hillside of color by 4.

    Four and not the nine the surrounding box holds, because the densest
    window lands on a corner of a big blob rather than in the middle of it --
    which is the number `max_spread` has to sit under.  An arm that overfills
    its window, as it does at the ZED's 1280 x 720, scores its own area over
    the window's and has to stay below it.
    """
    compact = np.zeros((200, 300), dtype=bool)
    compact[100:116, 150:166] = True
    assert detector.densest(compact, 24, (0, 0)).spread == pytest.approx(1.0)

    everywhere = np.ones((200, 300), dtype=bool)
    assert detector.densest(everywhere, 24, (0, 0)).spread == pytest.approx(4.0)
    assert detector.EMPTY_CLUSTER.spread == 0.0

    zed = np.zeros((200, 300), dtype=bool)
    zed[100:132, 150:192] = True  # 42 x 32 px, the arm at 1280 x 720
    spread = detector.densest(zed, 24, (0, 0)).spread
    assert spread == pytest.approx(42 * 32 / 24**2, abs=0.05)
    assert spread < detector.Thresholds().max_spread


def test_observations_describe_themselves_for_a_log_line():
    assert f"red at ({ARM_X}, {ARM_Y}), red 180/180 px" in str(observed(detector.RED))
    assert str(observed(detector.UNKNOWN)).startswith("unknown, red 0/0 px")
    ignored = classify(shirt(scene()))
    assert "red ignored (the color spreads" in str(ignored)


def test_the_transition_distance_is_measured_in_pixels():
    latch = armed(detector.StartLatch(max_transition_distance=30.0))
    offset = (ARM_X + 25, ARM_Y + 25)
    assert math.dist((ARM_X, ARM_Y), offset) > 30.0
    assert not any(run(latch, [detector.GREEN] * 3, position=offset))
