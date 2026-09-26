"""Find the course's start signal in a camera frame and say when it turns green.

Kept free of ROS, like `path_geometry` on the C++ side, so the decision this
makes can be tested against synthetic frames without a camera, a simulator or
a running graph.  `start_signal_detector_node.py` is the thin wrapper that
subscribes and feeds `Detector.process` one frame at a time.

The signal carries a red and a green arm 90 degrees apart on one pivot, so the
one lying horizontal is the one the car reads and a quarter turn swaps them.
Through the simulated camera at 15 Hz one sweep looks like this, which is the
sequence the latch has to read correctly:

       t(s)   red px   green px
       0.00     249         0
       0.33     243        25
       0.59     226       155
       0.66     208       185
       0.73     182       208     <- the counts cross over: green from here
       0.79     137       230     <- two green frames: the run starts
       1.06       5       246

Both arms are in frame together for two thirds of the turn -- they trade
projected area and the total stays near 250 px -- so the verdict at a place is
whichever of the two is ahead there, and the start is confirmed 0.13 s after
they cross rather than when the red arm finally goes edge-on.  Under a tighter
color floor the crossover becomes a hole instead: both arms wash out around
45 degrees and a frame or two reads as neither.  Both shapes have to work, and
a frame that reads as neither must not be read as "no longer red".

Outdoors
--------

The course is outside, the run may be at any hour, and the sun may be behind
the signal.  Nothing here trusts absolute brightness, and nothing trusts that
the signal is the only red or green thing in frame, because neither holds:

* **Color, not brightness.**  Pixels are judged on hue and on chroma -- how
  far from gray they are -- because those are what weather leaves alone.
  Exposure scales all three channels, which leaves hue and saturation
  untouched; glare behind the signal and clipping in the sun add to all three,
  which leaves hue and *chroma* untouched.  So the floors are chroma first and
  saturation only to reject gray, and the bands are drawn to clear skin,
  straw, turf and foliage rather than only the simulator's palette.  See
  `Thresholds`.
* **Arm-sized clusters, not pixel counts.**  The threshold is applied to the
  densest `cluster_window` window, and a candidate is thrown out if the same
  color keeps going well outside that window -- see `Cluster.spread`.  At the
  3 m both courses stand the signal at the arm is about 21 x 16 px of a
  640 x 360 frame; a shirt, a tent or a hedge is many times that, and fails on
  its own size however perfect its hue.
* **Several candidates per frame, not just the densest.**  A red shirt is
  bigger than the arm, so taking the single densest cluster would report the
  shirt and hide the signal.  `clusters` returns the best few of each color
  and every arm-sized one of them is tracked.
* **Places, not colors.**  `StartLatch` keeps a `Site` for each spot in the
  image where arm-sized signal color keeps turning up, and counts red and
  green frames per site.  A start is one site going from confirmed red to
  confirmed green: both arms turn about one pivot, so a real transition
  happens in one place.  Somebody in a red shirt standing near the course has
  their own site and can do nothing from it but stand there; the obstacle
  course's car wash hangs yellow ribbons, outside either signal color band.
* **A prior once the signal is found.**  The best site is fed back as a
  `Focus`, and inside that box the color floors relax by
  `focus_relaxation` -- a known signal is read on weaker evidence than an
  unknown blob has to produce, which is what gets a backlit arm read after the
  sun has come round behind it.  The whole region is still searched at full
  strength as well, so the prior can only add candidates, never hide them.

The asymmetry between `arm_frames` and `confirm_frames` is deliberate.  The
car stands at the line for as long as it takes, so waiting `arm_frames` for
the red arm costs nothing; the green has to be caught inside the second the
arm takes to turn, so `confirm_frames` is small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

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
    at 15 Hz; treat the result as read only.  `step` is honored rather than
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
    5 and its green at 114, and lighting -- Gazebo's or the sun's -- moves
    value about far more than it moves either of those.  Saturation is chroma
    over value, so it too is unchanged by a scene that is simply darker; what
    does cost saturation is light *added* to the arm's own color, which is
    exactly what sky glare on a backlit signal and clipping in direct sun both
    do.  The absolute chroma of a pixel, for the noise floor, is the product
    of the two returned here.

    Done in numpy rather than through OpenCV because it is six lines and
    cv_bridge is then not a dependency of the car's perception at all.
    """
    scaled = rgb.astype(np.float32) / 255.0
    red, green, blue = scaled[..., 0], scaled[..., 1], scaled[..., 2]
    value = scaled.max(axis=-1)
    chroma = value - scaled.min(axis=-1)
    # Gray pixels have no hue and would divide by zero; they are excluded by
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
# level, that horizon is the middle row of the image.  Below it is ground:
# turf, dirt and the simulated ground plane are all within a hue band's reach
# of the green arm, and there is a great deal of them.  Searching the top half
# alone excludes all of it.
#
# There is no margin below the horizon because a margin would buy nothing: it
# would only admit ground.  The margin that matters is the other way round,
# and it is 34 rows -- the arm lands at row 146 of 360 from the start line, so
# the car can sit a good 8 degrees nose-up before the signal reaches the
# bottom of the region.
#
# Narrowing this further is the cheapest way to make a hard course easier: on
# a course where the signal's place in frame is known, a region drawn round it
# makes the detector as selective before it has found the signal as it is
# afterwards.
DEFAULT_REGION = Region(0.0, 0.0, 1.0, 0.5)


@dataclass(frozen=True)
class Thresholds:
    """What counts as signal color, and what counts as an arm.

    The hue bands are drawn to clear what an outdoor course puts in frame, not
    merely what the simulator does:

    * Red stops at 16 degrees, short of skin at 20 to 35, of straw and
      dry grass at 30 to 50, and of the orange of cones and barrels.  It
      reaches back to 338 instead, because shade under an open sky is lit blue
      and takes red *towards* magenta, never towards orange.  Poppy Red is
      warmer than a placeholder red would be, and drifts further towards
      orange under direct-sun clipping than 12 degrees leaves room for; 16
      covers that drift with 4 degrees still held clear of skin.
    * Green starts at 110, at the edge of turf and foliage, which sit between
      80 and 110 even in full sun, and above the simulated ground plane's
      105, well above the car wash's yellow ribbons.  It stops short of the
      signal's own oasis blue board at 198. The arms themselves render at 5
      and 114 -- Leafy Green sits closer to real foliage than a more saturated
      green would, which is why the margin here is thinner than red's.

    `min_chroma` is the floor that does the work, and it is in absolute terms
    -- how far from gray the pixel is, in 0..1 -- because that is the quantity
    weather leaves alone.  Sky glare behind the signal and a sensor clipping
    in the sun both *add* light to every channel, which moves a pixel towards
    white without changing how far apart its channels are, so it costs
    saturation and not chroma.  Exposure is the other way round: it scales all
    three together, so it costs chroma and not saturation.  Between them,
    0.03 is about eight levels of an 8-bit channel, which is close to a tenth
    of what Leafy Green -- the less saturated of the two arms -- renders at in
    the simulator, and still several times the noise in a daylight frame.

    `min_saturation` is then only there to reject gray, which is what a low
    chroma over a bright value is: warm-lit concrete, a pinkish cloud, a
    hazed-over sky.  It is 0.15, low enough for an arm that has lost five
    sixths of its purity to glare, because the evidence that a candidate is
    the signal is its size and its place, not the purity of its color.
    `min_value` only drops what is black enough to have no color at all.

    `min_pixels` is set against the 21 x 16 px the arm subtends at 3 m, with
    room to spare for an arm part way round and for a signal further off.
    `max_spread` is what rejects everything bigger, and `cluster_window` has
    to be about the size of the arm in frame for it to mean anything: see
    `Cluster.spread`.
    """

    red: HueBand = field(default_factory=lambda: HueBand(338.0, 16.0))
    green: HueBand = field(default_factory=lambda: HueBand(110.0, 175.0))
    min_saturation: float = 0.15
    min_chroma: float = 0.03
    min_value: float = 0.05
    min_pixels: int = 12
    cluster_window: int = 24
    max_spread: float = 3.0
    candidates: int = 6
    focus_relaxation: float = 0.4

    def mask(self, band: HueBand, planes) -> np.ndarray:
        """Pixels of `band`'s color that are lit and colored enough to count."""
        hue, saturation, value = planes
        return (
            band.mask(hue)
            & (saturation >= self.min_saturation)
            & (saturation * value >= self.min_chroma)
            & (value >= self.min_value)
        )

    def relaxed(self) -> Thresholds:
        """The same thresholds, with the color floors lowered by the prior.

        Used inside a `Focus`, where the signal has already been found: the
        place is known, so weaker color is enough, and the size gates that do
        the real work against shirts and hedges are untouched.
        """
        return replace(
            self,
            min_saturation=self.min_saturation * self.focus_relaxation,
            min_chroma=self.min_chroma * self.focus_relaxation,
        )


@dataclass(frozen=True)
class Focus:
    """Where the signal was last seen, as a box to look harder inside."""

    x: float
    y: float
    radius: float

    def holds(self, position: tuple[float, float] | None) -> bool:
        return position is not None and math.dist((self.x, self.y), position) <= (
            self.radius
        )

    def box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) in pixels, clipped to the frame."""
        return (
            max(0, int(self.x - self.radius)),
            max(0, int(self.y - self.radius)),
            min(width, int(self.x + self.radius) + 1),
            min(height, int(self.y + self.radius) + 1),
        )


@dataclass(frozen=True)
class Cluster:
    """The densest window of one color, and how much of it surrounds it."""

    pixels: int = 0
    total: int = 0
    x: float = -1.0
    y: float = -1.0
    surround: int = 0

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def spread(self) -> float:
        """Matching pixels in three windows' width, over those in one.

        An arm that fits inside its window scores 1.0, however bright the day
        is.  Anything whose color keeps going outside the window scores
        more, and a candidate over `Thresholds.max_spread` is thrown out on
        its size alone: a shirt on somebody 5 m away scores 3.8, and a hedge
        or a hillside 4.0 -- the window lands on a corner of a big blob, not
        in the middle of it, so 4.0 rather than the 9.0 the box holds is what
        filling the whole neighborhood looks like.

        `cluster_window` therefore has to be at least the size of the arm in
        frame, because an arm that overfills its window scores its own area
        over the window's: 2.3 for the 42 x 32 px it subtends on the ZED's
        1280 x 720 against the default 24 px window, which still passes, and
        more than that if either the window is made smaller or the signal is
        nearer than the 3 m the course stands it at.

        This is the one test that does not care about hue at all, which is
        why it is what holds up against something that really is the same
        color as the signal.
        """
        return self.surround / self.pixels if self.pixels else 0.0


EMPTY_CLUSTER = Cluster()

# How much wider than the cluster window the surround is measured over.  Three
# windows across keeps the sample local to the candidate -- at 640 x 360 that
# is 72 px, about three arms' width -- while still leaving a shirt nowhere to
# hide.
SURROUND_SCALE = 3


def _box_sum(integral: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    """Set pixels in [y0, y1) x [x0, x1), from a summed-area table, clipped."""
    height, width = integral.shape
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(height - 1, y1), min(width - 1, x1)
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    )


def clusters(
    mask: np.ndarray,
    window: int,
    offset: tuple[int, int],
    limit: int = 1,
    max_spread: float | None = None,
) -> tuple[list[Cluster], Cluster | None]:
    """The `limit` densest arm-sized `window` x `window` boxes of `mask`.

    An arm is a compact patch, so the pixels in one box is the number to
    threshold on: a stray match here and there across a wide region never
    fills a box, however many of them there are.  Done with a summed-area
    table, so neither the window size nor the surround costs anything, and
    each position reported is the centroid of the matches inside its box
    rather than the box center.

    More than one box, because the biggest patch of red in frame need not be
    the signal -- somebody in a red shirt is larger than the arm and nearer
    the camera.  Boxes are taken greedily, densest first, and each one blanks
    out those it overlaps so that the candidates are distinct places rather
    than the same patch shifted by a pixel.

    `max_spread` is what keeps a big blob from using up the candidates: a box
    over one is not returned *and* does not count against `limit`, and the
    whole surrounding box is blanked out so that the search steps past the
    blob instead of walking across it a window at a time.  That can blank out
    an arm within a window or so of something bigger in the same color --
    but an arm that close to it has the thing in its own surround and would
    be thrown out for spread anyway, so nothing is lost that would have been
    believed.

    Returns those candidates and, separately, the fullest box that was
    rejected for spread, which is what there is to report when nothing else
    was found.

    `offset` is the searched area's top left corner, added back so positions
    are in full-frame pixels.
    """
    total = int(mask.sum())
    if total == 0:
        return [], None

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
    pad_y = span_y * (SURROUND_SCALE - 1) // 2
    pad_x = span_x * (SURROUND_SCALE - 1) // 2

    found: list[Cluster] = []
    blocked: Cluster | None = None
    # Bounded rather than while-true: every pass blanks out at least one
    # window, so this terminates on its own, but a frame full of color
    # should not be able to spend a hundred passes proving it.
    for _ in range(max(1, limit) + 64):
        if len(found) >= max(1, limit):
            break
        y0, x0 = np.unravel_index(int(np.argmax(sums)), sums.shape)
        pixels = int(sums[y0, x0])
        if pixels <= 0:
            break
        box = mask[y0 : y0 + span_y, x0 : x0 + span_x]
        rows, columns = np.nonzero(box)
        cluster = Cluster(
            pixels=pixels,
            total=total,
            x=offset[0] + x0 + float(columns.mean()),
            y=offset[1] + y0 + float(rows.mean()),
            surround=_box_sum(
                integral,
                y0 - pad_y,
                x0 - pad_x,
                y0 + span_y + pad_y,
                x0 + span_x + pad_x,
            ),
        )
        if max_spread is not None and cluster.spread > max_spread:
            if blocked is None or cluster.pixels > blocked.pixels:
                blocked = cluster
            blank(sums, y0, x0, span_y + pad_y, span_x + pad_x)
            continue
        found.append(cluster)
        blank(sums, y0, x0, span_y, span_x)
    return found, blocked


def blank(sums: np.ndarray, y0: int, x0: int, span_y: int, span_x: int) -> None:
    """Zero the window origins within `span` of (y0, x0), so none is picked next."""
    sums[
        max(0, y0 - span_y + 1) : y0 + span_y,
        max(0, x0 - span_x + 1) : x0 + span_x,
    ] = 0


def densest(mask: np.ndarray, window: int, offset: tuple[int, int]) -> Cluster:
    """The single densest window of `mask`, or `EMPTY_CLUSTER` if it is empty."""
    found, _blocked = clusters(mask, window, offset)
    return found[0] if found else EMPTY_CLUSTER


@dataclass(frozen=True)
class Observation:
    """What one frame shows, before any history is taken into account.

    `red` and `green` are the headline clusters -- what the frame is reported
    as showing -- while `reds` and `greens` are every arm-sized candidate of
    each color, which is what the latch tracks.  The headline is reported
    even when it was thrown out, with `red_reject`/`green_reject` saying why:
    "8 px against a threshold of 12" is the number somebody tuning this at the
    course needs to see.
    """

    state: int
    red: Cluster = EMPTY_CLUSTER
    green: Cluster = EMPTY_CLUSTER
    reds: tuple[Cluster, ...] = ()
    greens: tuple[Cluster, ...] = ()
    red_reject: str | None = None
    green_reject: str | None = None

    @property
    def position(self) -> tuple[float, float] | None:
        """Center of the arm this frame found, or None if it found neither."""
        if self.state == RED:
            return self.red.position
        if self.state == GREEN:
            return self.green.position
        return None

    def __str__(self) -> str:
        position = self.position
        where = (
            "" if position is None else f" at ({position[0]:.0f}, {position[1]:.0f})"
        )
        ignored = [
            f"{STATE_NAMES[colour]} ignored ({reason})"
            for colour, reason in ((RED, self.red_reject), (GREEN, self.green_reject))
            if reason
        ]
        return (
            f"{STATE_NAMES[self.state]}{where}"
            f", red {self.red.pixels}/{self.red.total} px"
            f", green {self.green.pixels}/{self.green.total} px"
            + ("; " + ", ".join(ignored) if ignored else "")
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

    def classify(self, rgb: np.ndarray, focus: Focus | None = None) -> Observation:
        """Which arm, if either, is facing the camera in this frame.

        `focus` is where the signal was last seen, from `StartLatch.focus`.
        It adds a second, gentler pass over that box -- it never replaces the
        full-strength search, so a prior can only ever find the signal in
        worse light, not hide something the strict pass would have caught.
        """
        height, width = rgb.shape[:2]
        area = self.region.pixels(width, height)
        passes = [(area, self.thresholds)]
        if focus is not None:
            window = overlap(area, focus.box(width, height))
            if window is not None:
                passes.insert(0, (window, self.thresholds.relaxed()))

        reds: list[Cluster] = []
        greens: list[Cluster] = []
        blocked: dict[int, Cluster | None] = {RED: None, GREEN: None}
        for (x0, y0, x1, y1), limits in passes:
            planes = hsv(rgb[y0:y1, x0:x1])
            for colour, band, found in (
                (RED, limits.red, reds),
                (GREEN, limits.green, greens),
            ):
                more, oversized = clusters(
                    limits.mask(band, planes),
                    limits.cluster_window,
                    (x0, y0),
                    limits.candidates,
                    limits.max_spread,
                )
                add(found, more, limits.cluster_window)
                if oversized is not None and (
                    blocked[colour] is None or oversized.pixels > blocked[colour].pixels
                ):
                    blocked[colour] = oversized

        red_accepted, red, red_reject = self.accept(reds, blocked[RED], focus)
        green_accepted, green, green_reject = self.accept(greens, blocked[GREEN], focus)

        # Both arms are on one pivot 90 degrees apart, so only one can face
        # the camera.  If both colors have an arm-sized candidate, the one in
        # the focus box wins, and failing that the fuller one; the other is
        # something else in frame.
        if not red_accepted and not green_accepted:
            state = UNKNOWN
        elif not green_accepted or (
            red_accepted and rank(red, focus) >= rank(green, focus)
        ):
            state = RED
        else:
            state = GREEN
        return Observation(
            state=state,
            red=red,
            green=green,
            reds=red_accepted,
            greens=green_accepted,
            red_reject=red_reject,
            green_reject=green_reject,
        )

    def accept(
        self, found: list[Cluster], blocked: Cluster | None, focus: Focus | None
    ) -> tuple[tuple[Cluster, ...], Cluster, str | None]:
        """(candidates that are arm-sized, the headline, why it was ignored)."""
        passed = tuple(
            sorted(
                (cluster for cluster in found if self.gate(cluster) is None),
                key=lambda cluster: rank(cluster, focus),
                reverse=True,
            )
        )
        if passed:
            return passed, passed[0], None
        # Nothing was arm-sized, so the fullest of what there was gets
        # reported along with the reason it does not count -- including the
        # blob that was too big for the search to return, which is the thing
        # somebody looking at a frame that read as nothing needs to see.
        candidates = found + ([blocked] if blocked is not None else [])
        if not candidates:
            return (), EMPTY_CLUSTER, None
        best = max(candidates, key=lambda cluster: cluster.pixels)
        return (), best, self.gate(best)

    def gate(self, cluster: Cluster) -> str | None:
        """Why this cluster is not an arm, or None if it could be one."""
        limits = self.thresholds
        if cluster.pixels < limits.min_pixels:
            return (
                f"{cluster.pixels} px in the densest window, under the "
                f"{limits.min_pixels} px an arm fills"
            )
        if cluster.spread > limits.max_spread:
            return (
                f"the color spreads {cluster.spread:.1f}x past the window, "
                f"over the {limits.max_spread:.1f}x an arm does, so this is "
                "part of something much larger"
            )
        return None


def rank(cluster: Cluster, focus: Focus | None) -> tuple[bool, int]:
    """Sort key that prefers the signal's known place, then the fuller patch."""
    return (focus is not None and focus.holds(cluster.position), cluster.pixels)


def overlap(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Two (x0, y0, x1, y1) boxes intersected, or None if they do not."""
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def add(found: list[Cluster], more: list[Cluster], window: int) -> None:
    """Extend `found` with the candidates in `more` it does not already hold.

    The focused pass covers ground the full-strength pass has already
    searched, so the same arm comes back from both; a candidate within a
    window of one already found is that same patch seen twice.
    """
    for cluster in more:
        if all(math.dist(cluster.position, other.position) > window for other in found):
            found.append(cluster)


@dataclass(eq=False)
class Site:
    """A place in the image where arm-sized signal color keeps appearing.

    One of these is the signal; the others are whatever else on the course
    happens to be red or green and about the size of an arm at 3 m.  Counting
    per site is what makes a start "this place went from red to green" rather
    than "something was red and then something was green".
    """

    x: float
    y: float
    red_frames: int = 0
    green_frames: int = 0
    unseen: int = 0

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def pull(self, position: tuple[float, float], easing: float = 0.25) -> None:
        """Follow the cluster a little, so the site tracks the arm it is on.

        Eased rather than jumped: the green arm's centroid is a dozen pixels
        from the red one's -- they are different shapes on a common pivot --
        and the site wants to sit between them, not chase whichever was last.
        """
        self.x += (position[0] - self.x) * easing
        self.y += (position[1] - self.y) * easing


class StartLatch:
    """Reads a sequence of Observations and decides when the run starts.

    Green alone is not the trigger.  A start is one `Site` -- one place in the
    image -- holding confirmed red and then holding confirmed green, which is
    what makes this safe to wire straight to a driver: a green thing somewhere
    in frame at startup, a single mis-hued frame, or a marshal in a green
    shirt cannot release the car.

    `require_red_first` relaxes the first half of that for a bench test where
    nothing ever showed red.  Nothing relaxes `confirm_frames`, which is
    cheap: at the camera's 15 Hz two frames cost 133 ms.
    """

    def __init__(
        self,
        confirm_frames: int = 2,
        arm_frames: int = 5,
        require_red_first: bool = True,
        max_transition_distance: float = 60.0,
        forget_frames: int = 30,
        max_sites: int = 8,
    ) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.arm_frames = max(1, int(arm_frames))
        self.require_red_first = require_red_first
        self.max_transition_distance = max_transition_distance
        self.forget_frames = max(1, int(forget_frames))
        self.max_sites = max(1, int(max_sites))
        self.reset()

    def reset(self) -> None:
        """Back to waiting for a start, for another run without a restart."""
        self.go = False
        self.sites: list[Site] = []
        self.started_at: tuple[float, float] | None = None

    # ------------------------------------------------------------ the record

    @property
    def armed(self) -> bool:
        """Whether anywhere in frame has held red long enough to be the signal."""
        return any(site.red_frames >= self.arm_frames for site in self.sites)

    @property
    def signal(self) -> Site | None:
        """The site most likely to be the signal: the best confirmed red.

        Ties break towards the one seen most recently, so a site that has been
        gone for a second does not hold the focus against one that is there
        now.
        """
        candidates = [site for site in self.sites if site.red_frames]
        if not candidates:
            return None
        return max(candidates, key=lambda site: (site.red_frames, -site.unseen))

    @property
    def red_frames(self) -> int:
        """Confirmed red frames at the site that has the most of them."""
        return max((site.red_frames for site in self.sites), default=0)

    @property
    def green_frames(self) -> int:
        """Green frames at the site that has the most of them."""
        return max((site.green_frames for site in self.sites), default=0)

    def focus(self) -> Focus | None:
        """Where to look hardest next frame, or None while nothing is found.

        Dropped once the signal has been missing for `forget_frames`, so a
        detector that locked onto the wrong thing -- or a car that has driven
        away from the signal it started on -- goes back to searching the whole
        region rather than staring at an empty box.
        """
        site = self.signal
        if site is None or self.max_transition_distance <= 0:
            return None
        if site.unseen >= self.forget_frames:
            return None
        return Focus(site.x, site.y, self.max_transition_distance)

    # ------------------------------------------------------------ the decision

    def nearest(self, position: tuple[float, float]) -> Site | None:
        """The site `position` belongs to, if any.

        Both arms turn about one pivot, so `max_transition_distance` is both
        how far the green arm may be from the red one and how far apart two
        places have to be to be different things.  Zero switches the whole
        idea off -- every cluster is then the same place -- which is what a
        bench test with one colored card in front of the camera wants.
        """
        if self.max_transition_distance <= 0:
            return self.sites[0] if self.sites else None
        near = [
            site
            for site in self.sites
            if math.dist(site.position, position) <= self.max_transition_distance
        ]
        return min(
            near, key=lambda site: math.dist(site.position, position), default=None
        )

    def update(self, observation: Observation) -> bool:
        """Fold one frame in and return the latch, which never goes back down.

        A frame showing neither arm deliberately changes nothing but the age
        of each site.  The arms pass edge-on partway through every turn, so
        treating that as evidence against red would throw away the very
        transition being waited for.
        """
        for site in self.sites:
            site.unseen += 1
        for site, cluster, colour in self.assign(observation):
            site.unseen = 0
            site.pull(cluster.position)
            if colour == RED:
                site.red_frames += 1
                # The arm has turned back, or never turned: whatever green was
                # counted here is not the start being waited for.
                site.green_frames = 0
            else:
                site.green_frames += 1
        self.forget()

        if not self.go:
            for site in self.sites:
                if self.starts(site):
                    self.go = True
                    self.started_at = site.position
                    break
        return self.go

    def assign(self, observation: Observation) -> list[tuple[Site, Cluster, int]]:
        """This frame's candidates as one verdict for each place in frame.

        One place shows one color at a time, and the fuller candidate there
        is the arm facing the car: the two arms are 90 degrees apart on one
        pivot, so for most of the turn both are in frame at once and both are
        arm-sized -- the sweep the module docstring quotes has 185 px of green
        against 208 px of red two thirds of the way through it.  Counting the two
        separately would hold a site at one green frame until the red arm had
        gone edge-on altogether, which at the course means waiting out the
        rest of the turn before the run starts.

        It is also what folds the same arm found twice -- once by the
        full-strength pass and once by the focused one -- back into one frame.
        """
        best: dict[int, tuple[Site, Cluster, int]] = {}
        for colour, found in ((RED, observation.reds), (GREEN, observation.greens)):
            for cluster in found:
                site = self.nearest(cluster.position) or self.open(cluster.position)
                current = best.get(id(site))
                if current is None or cluster.pixels > current[1].pixels:
                    best[id(site)] = (site, cluster, colour)
        return list(best.values())

    def open(self, position: tuple[float, float]) -> Site:
        """A new place to watch, at `position`."""
        site = Site(*position)
        self.sites.append(site)
        return site

    def starts(self, site: Site) -> bool:
        """Whether this site has gone from a confirmed red to a confirmed green."""
        if site.green_frames < self.confirm_frames:
            return False
        return site.red_frames >= self.arm_frames or not self.require_red_first

    def forget(self) -> None:
        """Drop sites that have gone, and cap how many are carried at once.

        A site is dropped once nothing has been seen there for
        `forget_frames`; what is left is capped so that a busy scene -- people
        walking about in front of the course -- cannot grow the list without
        bound.  The best confirmed reds are the ones kept.
        """
        self.sites = [site for site in self.sites if site.unseen < self.forget_frames]
        if len(self.sites) > self.max_sites:
            self.sites = sorted(
                self.sites,
                key=lambda site: (site.red_frames, site.green_frames, -site.unseen),
                reverse=True,
            )[: self.max_sites]

    def rejection(self, observation: Observation) -> str | None:
        """Why a confirmed green was not the start, for a log line."""
        if self.go or observation.state != GREEN:
            return None
        site = self.nearest(observation.position or (0.0, 0.0))
        if site is None or site.green_frames < self.confirm_frames:
            return None
        if site.red_frames >= self.arm_frames:
            return None
        armed = [other for other in self.sites if other.red_frames >= self.arm_frames]
        if armed and self.max_transition_distance > 0:
            distance = min(math.dist(other.position, site.position) for other in armed)
            return (
                f"green is {distance:.0f} px from where red was, past the "
                f"{self.max_transition_distance:.0f} px a turn on one pivot allows"
            )
        if site.red_frames:
            return (
                f"red has only been seen here for {site.red_frames} frames, short "
                f"of the {self.arm_frames} it takes to confirm the signal"
            )
        return "no confirmed red has been seen yet, so this is not a transition"


class Detector:
    """A classifier and a latch, wired together the way the node wires them.

    Kept here rather than in the node so the feedback between the two -- the
    latch's `focus` steering the next frame's search -- is tested against
    frames rather than only against a running camera.
    """

    def __init__(
        self, classifier: Classifier | None = None, latch: StartLatch | None = None
    ) -> None:
        self.classifier = classifier or Classifier()
        self.latch = latch or StartLatch()

    @property
    def go(self) -> bool:
        return self.latch.go

    def process(self, rgb: np.ndarray) -> Observation:
        """Classify one frame, fold it into the latch, and report the frame."""
        observation = self.classifier.classify(rgb, self.latch.focus())
        self.latch.update(observation)
        return observation

    def reset(self) -> None:
        self.latch.reset()


def annotate(
    rgb: np.ndarray,
    observation: Observation,
    region: Region,
    focus: Focus | None = None,
) -> np.ndarray:
    """A copy of the frame with the region, the focus and the cluster boxed.

    For `~/debug_image`, which is what tuning the thresholds on the real
    course will be done through: whether the boxes land on the signal answers
    most of the questions a pixel count on its own raises.
    """
    marked = np.array(rgb, dtype=np.uint8)
    height, width = marked.shape[:2]
    outline(marked, region.pixels(width, height), (128, 128, 128))
    if focus is not None:
        # White, so the box the detector is favoring is the one that stands
        # out: if it is not on the signal, nothing else in the frame matters.
        outline(marked, focus.box(width, height), (255, 255, 255))

    position = observation.position
    if position is None:
        return marked
    colour = (255, 0, 0) if observation.state == RED else (0, 255, 0)
    half = 12
    outline(
        marked,
        (
            int(position[0]) - half,
            int(position[1]) - half,
            int(position[0]) + half,
            int(position[1]) + half,
        ),
        colour,
    )
    return marked


def outline(marked: np.ndarray, box: tuple[int, int, int, int], colour) -> None:
    """Draw a one pixel box on `marked`, clipped to it."""
    height, width = marked.shape[:2]
    paint = np.array(colour, dtype=np.uint8)
    x0 = max(0, min(width - 1, box[0]))
    y0 = max(0, min(height - 1, box[1]))
    x1 = max(x0 + 1, min(width, box[2]))
    y1 = max(y0 + 1, min(height, box[3]))
    marked[y0, x0:x1] = marked[y1 - 1, x0:x1] = paint
    marked[y0:y1, x0] = marked[y0:y1, x1 - 1] = paint
