"""Find the course's start signal in a camera frame and say when it turns green.

Kept free of ROS, like `path_geometry` on the C++ side, so the decision this
makes can be tested against synthetic frames without a camera, a simulator or
a running graph.  `start_signal_node.py` is the thin wrapper that subscribes,
calls `Classifier.classify` on each frame and feeds `StartLatch`.

The signal carries a red and a green arm 90 degrees apart on one pivot, so at
most one of them faces the car -- and *neither* does while the pair passes
edge-on halfway through the turn.  Through the simulated camera one sweep looks
like this, which is the sequence the latch has to read correctly:

       t(s)   red px   green px
       0.00     246         0
       0.46     234       119
       0.59     211       179
       0.66     188       205     <- the counts cross over: this frame is green
       0.99      24       246

The two arms are 90 degrees apart on one pivot, so through the turn they trade
projected area and the total stays near 250 px: the verdict is whichever count
is ahead.  Under a tighter saturation floor the crossover becomes a hole
instead -- both arms wash out around 45 degrees and a frame or two reads as
neither -- so the latch has to cope with both, and an UNKNOWN frame must not be
read as "no longer red".  Its counters hold through one; only a frame of the
*other* colour clears them.

Four things guard against something else in frame being taken for the signal,
because at the 3 m both courses stand the signal at the arm is only about
21 x 16 px of a 640 x 360 frame and a false start is expensive:

* A region of interest that starts at the top half of the frame, because the
  arm stands above the camera's horizon from anywhere on the course and the
  ground plane below it is dark green.  See `DEFAULT_REGION`.
* Hue bands, not channel comparisons.  The straw bales that line both courses
  are (0.72, 0.48, 0.12) -- hue 36 degrees, saturated, and 202 of them fill
  much of the frame, so the red band has to stop well short of orange.
* A cluster test rather than a pixel count.  The threshold is applied to the
  densest `cluster_window` window, so scattered matches across the region of
  interest do not add up to a detection the way a raw count would.
* Green has to appear where red was, within `max_transition_distance`.  Both
  arms turn about the same pivot, so the transition happens in one place; the
  obstacle course's car wash hangs twenty red and twenty blue ribbons of much
  the same red as the signal, and this is what stops a green elsewhere in
  frame pairing up with one of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Mirrors the constants in cfr_interfaces/msg/StartSignal.msg.  Duplicated
# rather than imported so this module stays free of ROS; the node asserts they
# still agree.
UNKNOWN = 0
RED = 1
GREEN = 2

STATE_NAMES = {UNKNOWN: "unknown", RED: "red", GREEN: "green"}

# Encodings this can read, as (bytes per pixel, indices of R, G and B).  The
# simulated camera publishes rgb8 and the ZED wrapper bgra8, so both orders
# and both widths have to work.  Anything else -- bayer, mono, 16 bit -- is
# rejected loudly rather than guessed at.
ENCODINGS = {
    "rgb8": (3, (0, 1, 2)),
    "bgr8": (3, (2, 1, 0)),
    "rgba8": (4, (0, 1, 2)),
    "bgra8": (4, (2, 1, 0)),
    # cv_bridge's name for three untagged 8-bit channels, which by OpenCV
    # convention are BGR.
    "8UC3": (3, (2, 1, 0)),
    "8UC4": (4, (2, 1, 0)),
}


class ImageFormatError(ValueError):
    """An image this cannot read: unknown encoding, or short of data."""


def decode(encoding: str, width: int, height: int, step: int, data) -> np.ndarray:
    """A `height` x `width` x 3 uint8 view of one ROS image, in RGB order.

    A view, not a copy, wherever the layout allows one, so this costs nothing
    at 15 Hz; treat the result as read only.  `step` is honoured rather than
    assumed, because a row is allowed to carry padding past the last pixel.
    """
    try:
        channels, order = ENCODINGS[encoding]
    except KeyError:
        raise ImageFormatError(
            f"cannot read encoding '{encoding}'; expected one of "
            f"{', '.join(sorted(ENCODINGS))}"
        ) from None
    if width <= 0 or height <= 0:
        raise ImageFormatError(f"image is {width}x{height}")
    row = width * channels
    step = step or row
    if step < row:
        raise ImageFormatError(
            f"step {step} is short of {row} bytes for {width} {encoding} pixels"
        )
    frame = np.frombuffer(data, dtype=np.uint8)
    if frame.size < step * height:
        raise ImageFormatError(
            f"{frame.size} bytes is short of the {step * height} a "
            f"{width}x{height} {encoding} image needs"
        )
    frame = frame[: step * height].reshape(height, step)
    return frame[:, :row].reshape(height, width, channels)[:, :, order]


def hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue in degrees, saturation and value in 0..1, from a uint8 RGB frame.

    Hue is what identifies the signal: the simulator's red arm renders at hue
    1.5 and its green at 130, and Gazebo's lighting moves value about far more
    than it moves either of those.  Done here in numpy rather than through
    OpenCV because it is six lines and cv_bridge is then not a dependency of
    the car's perception at all.
    """
    scaled = rgb.astype(np.float32) / 255.0
    red, green, blue = scaled[..., 0], scaled[..., 1], scaled[..., 2]
    value = scaled.max(axis=-1)
    chroma = value - scaled.min(axis=-1)
    # Grey pixels have no hue and would divide by zero; they are excluded by
    # the saturation threshold anyway, so any finite hue will do for them.
    divisor = np.where(chroma > 0, chroma, 1.0)
    hue = np.select(
        [chroma <= 0, value == red, value == green],
        [
            np.zeros_like(value),
            60.0 * (((green - blue) / divisor) % 6.0),
            60.0 * ((blue - red) / divisor + 2.0),
        ],
        default=60.0 * ((red - green) / divisor + 4.0),
    )
    saturation = np.where(value > 0, chroma / np.where(value > 0, value, 1.0), 0.0)
    return hue, saturation, value


@dataclass(frozen=True)
class HueBand:
    """A closed band of hue, in degrees, which may wrap through 0."""

    low: float
    high: float

    def mask(self, hue: np.ndarray) -> np.ndarray:
        if self.low <= self.high:
            return (hue >= self.low) & (hue <= self.high)
        # Red sits astride 0, so its band is the union of the two ends.
        return (hue >= self.low) | (hue <= self.high)


@dataclass(frozen=True)
class Region:
    """Where in the frame to look, as fractions of width and height.

    Fractions rather than pixels so the same numbers hold for the simulator's
    640 x 360 and the ZED's 1280 x 720.  `Region()` is the whole frame;
    `DEFAULT_REGION` below is what the node actually starts with and why.
    """

    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 1.0
    y_max: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.x_min < self.x_max <= 1.0:
            raise ValueError(f"x fractions {self.x_min}..{self.x_max} are not 0..1")
        if not 0.0 <= self.y_min < self.y_max <= 1.0:
            raise ValueError(f"y fractions {self.y_min}..{self.y_max} are not 0..1")

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) in pixels, x1/y1 exclusive and at least one wide.

        Rounded to the nearest pixel edge rather than floored or ceilinged,
        because the fractions people write do not land on pixel boundaries in
        binary floating point: 0.55 of 360 rows is 198.00000000000003, which a
        ceiling would take as 199 rows.
        """
        x0 = min(round(self.x_min * width), width - 1)
        y0 = min(round(self.y_min * height), height - 1)
        x1 = min(max(round(self.x_max * width), x0 + 1), width)
        y1 = min(max(round(self.y_max * height), y0 + 1), height)
        return x0, y0, x1, y1


# The arm sits 0.813 m up and the camera 0.20 m up, so the arm is above the
# camera's horizon from anywhere on the course -- and for a camera looking
# level, that horizon is the middle row of the image.  Below it is ground, and
# the simulated ground plane is (0.16, 0.25, 0.13): dark, but hue 105 and
# saturation 0.48, which is close enough to the green band to be worth not
# looking at.  Searching the top half alone excludes all of it.
#
# There is no margin below the horizon because a margin would buy nothing: it
# would only admit ground.  The margin that matters is the other way round,
# and it is 34 rows -- the arm lands at row 146 of 360 from the start line, so
# the car can sit a good 8 degrees nose-up before the signal reaches the
# bottom of the region.
DEFAULT_REGION = Region(0.0, 0.0, 1.0, 0.5)


@dataclass(frozen=True)
class Thresholds:
    """What counts as signal colour, and how much of it counts as an arm.

    The bands stop short of the four other saturated hues either course puts
    in frame: the straw bales at 36 degrees, the ground plane at 105, the
    signal's own sky blue board at 197, and the car wash's blue ribbons at
    212.  The arms themselves render at 1.5 and 130.

    `min_pixels` is set against the 21 x 16 px the arm subtends at the 3 m
    both courses stand the signal at, with room to spare for an arm part way
    round and for the further 4 m the signal used to stand at.
    """

    red: HueBand = field(default_factory=lambda: HueBand(345.0, 15.0))
    green: HueBand = field(default_factory=lambda: HueBand(110.0, 170.0))
    min_saturation: float = 0.45
    min_value: float = 0.15
    min_pixels: int = 12
    cluster_window: int = 24


@dataclass(frozen=True)
class Cluster:
    """The densest window of one colour, and how much of it there was in all."""

    pixels: int
    total: int
    x: float
    y: float


@dataclass(frozen=True)
class Observation:
    """What one frame shows, before any history is taken into account."""

    state: int
    red: Cluster
    green: Cluster

    @property
    def position(self) -> tuple[float, float] | None:
        """Centre of the arm this frame found, or None if it found neither."""
        if self.state == RED:
            return self.red.x, self.red.y
        if self.state == GREEN:
            return self.green.x, self.green.y
        return None

    def __str__(self) -> str:
        position = self.position
        where = (
            "" if position is None else f" at ({position[0]:.0f}, {position[1]:.0f})"
        )
        return (
            f"{STATE_NAMES[self.state]}{where}"
            f", red {self.red.pixels}/{self.red.total} px"
            f", green {self.green.pixels}/{self.green.total} px"
        )


EMPTY_CLUSTER = Cluster(0, 0, -1.0, -1.0)


def densest(mask: np.ndarray, window: int, offset: tuple[int, int]) -> Cluster:
    """The `window` x `window` box of `mask` holding the most set pixels.

    An arm is a compact patch, so this is the number to threshold on: a stray
    match here and there across a wide region of interest never fills one box,
    however many of them there are.  Done with a summed-area table, so the
    cost does not depend on the window size, and the position reported is the
    centroid of the matches inside the winning box rather than the box centre.

    `offset` is the region of interest's top left corner, added back so the
    position is in full-frame pixels.
    """
    total = int(mask.sum())
    if total == 0:
        return EMPTY_CLUSTER

    height, width = mask.shape
    span_y, span_x = min(window, height), min(window, width)
    integral = np.zeros((height + 1, width + 1), dtype=np.int32)
    np.cumsum(np.cumsum(mask, axis=0, dtype=np.int32), axis=1, out=integral[1:, 1:])
    sums = (
        integral[span_y:, span_x:]
        - integral[: height + 1 - span_y, span_x:]
        - integral[span_y:, : width + 1 - span_x]
        + integral[: height + 1 - span_y, : width + 1 - span_x]
    )
    y0, x0 = np.unravel_index(int(np.argmax(sums)), sums.shape)
    box = mask[y0 : y0 + span_y, x0 : x0 + span_x]
    rows, columns = np.nonzero(box)
    return Cluster(
        pixels=int(sums[y0, x0]),
        total=total,
        x=offset[0] + x0 + float(columns.mean()),
        y=offset[1] + y0 + float(rows.mean()),
    )


class Classifier:
    """Turns one frame into an Observation, with no memory between frames."""

    def __init__(
        self,
        thresholds: Thresholds | None = None,
        region: Region | None = None,
    ) -> None:
        self.thresholds = thresholds or Thresholds()
        self.region = region or DEFAULT_REGION

    def classify(self, rgb: np.ndarray) -> Observation:
        """Which arm, if either, is facing the camera in this frame."""
        height, width = rgb.shape[:2]
        x0, y0, x1, y1 = self.region.pixels(width, height)
        hue, saturation, value = hsv(rgb[y0:y1, x0:x1])

        limits = self.thresholds
        lit = (saturation >= limits.min_saturation) & (value >= limits.min_value)
        clusters = [
            densest(band.mask(hue) & lit, limits.cluster_window, (x0, y0))
            for band in (limits.red, limits.green)
        ]
        red, green = clusters

        # Both arms are on one pivot 90 degrees apart, so only one can face the
        # camera; if both bands somehow clear the threshold, the fuller cluster
        # is the arm and the other is something else in frame.
        best = max(red.pixels, green.pixels)
        if best < limits.min_pixels:
            state = UNKNOWN
        elif red.pixels >= green.pixels:
            state = RED
        else:
            state = GREEN
        # Both clusters are reported whatever the verdict, including on an
        # UNKNOWN frame: "8 px against a threshold of 12" is the number
        # somebody tuning this at the course needs to see.
        return Observation(state=state, red=red, green=green)


class StartLatch:
    """Reads a sequence of Observations and decides when the run starts.

    Green alone is not the trigger.  A start is a red arm that *becomes* a
    green one, in the same place, and holding out for that is what makes the
    latch safe to wire straight to a driver: a green thing somewhere in frame
    at startup, or a single mis-hued frame, cannot release the car.

    `require_red_first` relaxes the first half of that for a bench test where
    nothing ever showed red.  Nothing relaxes the confirmation count, which is
    cheap: at the camera's 15 Hz two frames cost 133 ms.
    """

    def __init__(
        self,
        confirm_frames: int = 2,
        require_red_first: bool = True,
        max_transition_distance: float = 60.0,
    ) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.require_red_first = require_red_first
        self.max_transition_distance = max_transition_distance
        self.reset()

    def reset(self) -> None:
        """Back to waiting for a start, for another run without a restart."""
        self.go = False
        self.armed = False
        self.red_frames = 0
        self.green_frames = 0
        self.red_position: tuple[float, float] | None = None
        self.green_position: tuple[float, float] | None = None

    def update(self, observation: Observation) -> bool:
        """Fold one frame in and return the latch, which never goes back down.

        An UNKNOWN frame deliberately changes nothing.  The arms pass edge-on
        partway through every turn, so treating that as evidence against red
        would throw away the very transition being waited for.
        """
        if observation.state == RED:
            self.red_frames += 1
            self.green_frames = 0
            if self.red_frames >= self.confirm_frames:
                self.armed = True
                self.red_position = observation.position
        elif observation.state == GREEN:
            # A count of frames only means anything if they are frames of the
            # same green thing, so a cluster that jumps starts the count again
            # rather than inheriting the last one's.
            self.green_frames = 1 if self.jumped(observation) else self.green_frames + 1
            self.green_position = observation.position
            self.red_frames = 0
            if self.green_frames >= self.confirm_frames and self.started(observation):
                self.go = True
        return self.go

    def jumped(self, observation: Observation) -> bool:
        """Whether this green cluster is somewhere other than the last one."""
        if self.green_position is None or self.max_transition_distance <= 0:
            return False
        distance = math.dist(self.green_position, observation.position)
        return distance > self.max_transition_distance

    def started(self, observation: Observation) -> bool:
        """Whether a confirmed green is the start, or green from elsewhere."""
        if not self.armed:
            return not self.require_red_first
        if self.max_transition_distance <= 0 or self.red_position is None:
            return True
        return (
            math.dist(self.red_position, observation.position)
            <= self.max_transition_distance
        )

    def rejection(self, observation: Observation) -> str | None:
        """Why a confirmed green was not the start, for a log line."""
        if self.go or observation.state != GREEN:
            return None
        if self.green_frames < self.confirm_frames:
            return None
        if not self.armed:
            return "no confirmed red has been seen yet, so this is not a transition"
        if self.red_position is None or self.max_transition_distance <= 0:
            return None
        distance = math.dist(self.red_position, observation.position)
        return (
            f"green is {distance:.0f} px from where red was, past the "
            f"{self.max_transition_distance:.0f} px a turn on one pivot allows"
        )


def annotate(rgb: np.ndarray, observation: Observation, region: Region) -> np.ndarray:
    """A copy of the frame with the region of interest and the cluster boxed.

    For `~/debug_image`, which is what tuning the thresholds on the real course
    will be done through: whether the box lands on the signal answers most of
    the questions a pixel count on its own raises.
    """
    marked = np.array(rgb, dtype=np.uint8)
    height, width = marked.shape[:2]
    x0, y0, x1, y1 = region.pixels(width, height)
    grey = np.array((128, 128, 128), dtype=np.uint8)
    marked[y0, x0:x1] = marked[y1 - 1, x0:x1] = grey
    marked[y0:y1, x0] = marked[y0:y1, x1 - 1] = grey

    position = observation.position
    if position is None:
        return marked
    colour = np.array(
        (255, 0, 0) if observation.state == RED else (0, 255, 0), dtype=np.uint8
    )
    half = 12
    bx0 = max(0, min(width - 1, int(position[0]) - half))
    by0 = max(0, min(height - 1, int(position[1]) - half))
    bx1 = max(bx0 + 1, min(width, int(position[0]) + half))
    by1 = max(by0 + 1, min(height, int(position[1]) + half))
    marked[by0, bx0:bx1] = marked[by1 - 1, bx0:bx1] = colour
    marked[by0:by1, bx0] = marked[by0:by1, bx1 - 1] = colour
    return marked
