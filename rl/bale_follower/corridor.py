"""The drivable corridor, summarised from the same scan the policy already sees.

Thirty-six range bins carry the corridor's geometry, but only implicitly: to
steer from them a policy has to discover, from reward alone, that bins on
opposite sides of the car mean "wall left" and "wall right" and that the
difference between them is what to correct. Runs v1-v6 did not discover it
-- v6 finished 86% of its episodes pinned against something while commanding
only 1.0 m/s of a possible 3.5, which is not a car driving badly, it is a car
that cannot see where the lane is.

So this reduces the scan to the two numbers a driver actually steers on --
how far off the corridor's centre the car is, and how far its heading is off
the corridor's direction -- plus the corridor's width and, the part that
matters most on this course, a confidence that says when none of it means
anything.

Everything here is computed from the binned scan, which `cloud_scan` builds
from the ZED's own point cloud. Nothing reads sim pose, the world SDF, or the
course centerline, so the same function runs on the robot.

Three properties of this course drove the design:

  * **Boundaries are beside you, obstacles are ahead.** Fits use only the
    side sectors (beyond `side_min_deg` off the nose) and ignore the central
    ones. That is what makes the car wash survivable: its ribbons are visual
    -- `generate_obstacle_course.py` builds them with `collide=False` because
    they are streamer weight -- but the depth camera renders them like any
    other surface, so the scan reports a solid wall across a gap the car is
    supposed to drive straight through. That false wall lands in the central
    bins, where it cannot move the corridor estimate. The real edges there
    are the arch uprights at y = +-0.576 m, and those are exactly what the
    side sectors see. The raw bins still go to the policy unchanged, so the
    frontal return is not hidden from it -- it is only kept out of the
    steering summary.

  * **Some of the course has no corridor at all.** The wide open region
    before the buckets is open floor; the bucket entrance is blocked by a
    free-standing gap-bale wall rather than by lane walls. A wall follower
    asked for an answer there will invent one, and the invented answer is
    confidently wrong, which is worse than no answer. So `confidence` is a
    first-class output: it falls to zero when a side has too few returns,
    when the two sides disagree about direction, or when the implied
    corridor is too wide to be a lane. A policy given a confidence channel
    can learn to fall back on the raw bins where the summary is blind.

  * **A single stray return must not move the answer.** Each side is fitted
    with one round of residual trimming, and a side whose survivors are too
    scattered to be a line is discarded rather than averaged in.
"""

from __future__ import annotations

import math

import numpy as np

# Beyond this many degrees off the nose a return is "beside" the car and may
# define a boundary; inside it the return is "ahead" and may not. The floor
# is set by what a lane wall looks like: the lane is 32 in (0.813 m), so a
# wall sits about 0.4 m to the side, and at 20 degrees that is a return 1.2 m
# up the road -- near enough to be about the corridor the car is in rather
# than one it may never reach.
SIDE_MIN_DEG = 20.0

# A corridor wider than this is not a lane. The course's lane is 0.813 m and
# its widest genuine passage is the car wash's 1.151 m between uprights, so
# anything past 2.5 m means the fit has latched onto two things that are not
# a pair of walls -- the usual case being open floor with a stray return on
# each side.
MAX_CORRIDOR_WIDTH_M = 2.5

# ...and narrower than this is not one either. The car is 0.30 m wide, so a
# gap under 0.45 m is not somewhere it can go, and a "corridor" that narrow
# means both fits have landed on the same object from opposite sides. The
# bucket section produced exactly this: two sides 0.04 m apart, reported at
# confidence 0.69.
MIN_CORRIDOR_WIDTH_M = 0.45

# A single wall further away than this says nothing about the corridor the
# car is in. Measured in the bucket section, the one-sided branch latched
# onto a return 4.4 m off and reported it as a boundary.
MAX_SINGLE_WALL_M = 1.6

# NOTE: there is deliberately no maximum-heading rejection here, and adding
# one would be a mistake worth spelling out. An earlier version dropped any
# corridor running more than 45 degrees off the car's heading, on the theory
# that such a fit had latched onto scattered obstacles rather than a lane.
# The bucket section did produce garbage at 62 to 81 degrees -- but so does
# every genuine 90 degree turn on this course, of which there are at least
# three: the banked turn, the potholes into the open area, and the entry to
# the bucket section. Rejecting by angle throws the estimate away precisely
# at the corners, which is where the car most needs telling which way to go.
#
# The garbage is rejected by width instead, which separates the two cases
# cleanly: every one of those bucket-section fits reported a corridor either
# 0.04 m wide or 3.4 m away, and none of them survives MIN_CORRIDOR_WIDTH_M
# or MAX_SINGLE_WALL_M. A real corner is a normal-width corridor at a large
# angle; scattered obstacles are an absurd width at any angle.

# Returns at or past this fraction of max_range are "no return", not a
# boundary at that distance. cloud_scan fills empty bins with max_range.
NO_RETURN_FRACTION = 0.98

# Each side needs this many surviving returns to be called a wall.
MIN_SIDE_POINTS = 3

# After trimming, a side whose residuals scatter more than this is not a
# straight boundary and is dropped.
MAX_SIDE_RESIDUAL_M = 0.18

# The two sides must agree about which way the corridor runs. Walls that
# disagree by more than this are not two sides of one corridor.
MAX_SIDE_DISAGREEMENT_DEG = 35.0

# A corridor the car cannot drive into is not the car's corridor. The side
# fits say which way the boundaries run; free_gap says where the opening is.
# When those two disagree by more than this, the fit has latched onto
# something that crosses the path rather than flanking it -- measured at the
# car wash, whose hanging ribbons render as a solid curtain and fit as a
# "corridor" running 90 degrees across the lane the car has to drive through.
#
# This is NOT the angle rejection that was deliberately left out (see
# estimate_corridor): it does not care how far the corridor turns, only that
# the walls and the opening tell the same story. A real 90-degree corner --
# the banked turn, the entry to the open area, the entry to the buckets --
# passes, because there the gap points round the corner too.
#
# Measured over 373 confident fits across 540 poses on the real course:
# at 60 degrees this drops 8 of the 34 fits whose heading was more than
# 20 degrees wrong, and 1 of the 339 that were right. The margin is wide
# (good fits' 95th percentile 41.6 degrees, bad fits' 90th 89.9), but the
# number itself was picked against that one dataset, so treat it as tuned,
# not derived.
MAX_CORRIDOR_GAP_DISAGREEMENT_DEG = 60.0


class CorridorEstimate:
    """Offset, heading and width of the drivable corridor, with a confidence.

    `confidence` is 0.0 when no corridor was found, and the other fields are
    then zero rather than stale or guessed -- a caller that ignores the
    confidence gets "dead centre, dead straight", which is the safest thing
    to be wrong about and, more importantly, carries no false gradient.
    """

    __slots__ = (
        "lateral_offset_m",
        "heading_error_rad",
        "half_width_m",
        "confidence",
        "left_found",
        "right_found",
    )

    def __init__(
        self,
        lateral_offset_m: float = 0.0,
        heading_error_rad: float = 0.0,
        half_width_m: float = 0.0,
        confidence: float = 0.0,
        left_found: bool = False,
        right_found: bool = False,
    ):
        self.lateral_offset_m = lateral_offset_m
        self.heading_error_rad = heading_error_rad
        self.half_width_m = half_width_m
        self.confidence = confidence
        self.left_found = left_found
        self.right_found = right_found

    def __repr__(self) -> str:
        return (
            f"CorridorEstimate(offset={self.lateral_offset_m:+.3f} m, "
            f"heading={math.degrees(self.heading_error_rad):+.1f} deg, "
            f"half_width={self.half_width_m:.3f} m, "
            f"confidence={self.confidence:.2f}, "
            f"left={self.left_found}, right={self.right_found})"
        )


def _angle_between(first: float, second: float) -> float:
    """Unsigned angle between two line directions, accounting for mod-pi.

    A line has no head or tail, so directions of +89 and -89 degrees describe
    walls two degrees apart, not 178. Comparing them naively would call a
    perfectly ordinary pair of walls at a corner a disagreement and throw the
    corridor away -- the same failure as rejecting by angle, arriving by a
    different route.
    """
    difference = (first - second) % math.pi
    return min(difference, math.pi - difference)


def _mean_angle(first: float, second: float) -> float:
    """Average of two line directions, on the mod-pi circle.

    Doubling the angles maps the mod-pi circle onto a full circle, where a
    plain vector mean is correct, then halving comes back. Averaging +89 and
    -89 directly would give 0, which points across both walls instead of
    along them.
    """
    x = math.cos(2.0 * first) + math.cos(2.0 * second)
    y = math.sin(2.0 * first) + math.sin(2.0 * second)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return first
    mean = math.atan2(y, x) / 2.0
    # Back into [-90, +90], the range a forward-pointing direction occupies.
    if mean > math.pi / 2.0:
        mean -= math.pi
    elif mean < -math.pi / 2.0:
        mean += math.pi
    return mean


def _orthogonal_line(xs: np.ndarray, ys: np.ndarray):
    """(angle, signed lateral offset, RMS residual) of the best line, any angle.

    Total least squares, via the principal axis of the centred points, rather
    than a fit of y against x. That choice is the whole point: this course
    turns through 90 degrees in at least three places -- the banked turn, the
    potholes into the open area, and the entry to the bucket section -- and at
    a corner like that the wall sweeping past the car runs across its heading,
    not along it. Expressed as y = a*x + b such a wall is vertical, so the fit
    either blows up or is thrown out by the degeneracy guard, and the corridor
    estimate vanishes exactly where the car most needs to be told which way to
    steer. An orthogonal fit has no preferred orientation and handles it.

    `angle` is the line's direction in the car's frame, resolved into
    [-90, +90] degrees by taking whichever of the two opposite directions
    points forward. At a true right angle the two are equally forward and the
    sign is genuinely undecidable from one wall; the caller reflects that in
    the confidence rather than committing to a guess.
    """
    if xs.size < MIN_SIDE_POINTS:
        return None

    points = np.stack([xs, ys], axis=1)
    centroid = points.mean(axis=0)
    centred = points - centroid
    # Smallest singular vector is the normal; largest is the direction.
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    direction = vectors[0]
    normal = np.array([-direction[1], direction[0]])
    residuals = np.abs(centred @ normal)

    # One round of trimming, at twice the median absolute residual: enough to
    # drop a bucket standing off a wall, not so aggressive that a gently
    # curving wall is whittled down to the three points that happen to be
    # collinear.
    keep = residuals <= max(2.0 * float(np.median(residuals)), 0.05)
    if keep.sum() < MIN_SIDE_POINTS:
        return None

    points = points[keep]
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    direction = vectors[0]

    # The SVD's sign is arbitrary, and both the reported angle and the sign of
    # the offset hang off it, so it is pinned here rather than left to
    # whichever way the decomposition happened to come out. The normal is
    # made to point to the car's left; the direction then follows from it and
    # automatically points forward. Doing it in this order (normal first)
    # keeps "positive offset means the wall is on my left" true at every
    # orientation, including the perpendicular walls of a 90 degree corner,
    # where fixing the direction's sign instead would flip the offset.
    normal = np.array([-direction[1], direction[0]])
    if normal[1] < 0.0:
        normal = -normal
    direction = np.array([normal[1], -normal[0]])

    spread = float(np.sqrt(np.mean((centred @ normal) ** 2)))
    if spread > MAX_SIDE_RESIDUAL_M:
        return None

    angle = math.atan2(direction[1], direction[0])
    offset = float(centroid @ normal)
    return angle, offset, spread


def _disagrees_with_gap(angle: float, gap_bearing_rad: float | None) -> bool:
    """True when the fitted corridor runs across the only way out.

    `angle` is a line direction, so this folds mod pi: a corridor at -89
    degrees and one at +89 differ by 2, not 178.
    """
    if gap_bearing_rad is None:
        return False
    gap = float(gap_bearing_rad)
    return _angle_between(angle, gap) > math.radians(MAX_CORRIDOR_GAP_DISAGREEMENT_DEG)


def estimate_corridor(
    scan: np.ndarray,
    fov_deg: float,
    max_range: float,
    side_min_deg: float = SIDE_MIN_DEG,
    max_width_m: float = MAX_CORRIDOR_WIDTH_M,
    gap_bearing_rad: float | None = None,
) -> CorridorEstimate:
    """Reduce a bearing-binned scan to the corridor the car is driving in.

    `scan` is `cloud_scan`'s output: one range per bearing bin, lowest bin at
    the rightmost bearing, empty bins filled with `max_range`.

    Pass `free_gap`'s bearing as `gap_bearing_rad` to enable the consistency
    check described at MAX_CORRIDOR_GAP_DISAGREEMENT_DEG. Callers that leave
    it out get the side fits ungated, which is what the unit tests want but
    not what the car should be driving on.
    """
    bins = int(scan.shape[0])
    if bins < 4:
        return CorridorEstimate()

    half = fov_deg / 2.0
    edges = np.linspace(-half, half, bins + 1)
    bearings = (edges[:-1] + edges[1:]) / 2.0

    real = scan < max_range * NO_RETURN_FRACTION
    radians = np.radians(bearings)
    xs = scan * np.cos(radians)
    ys = scan * np.sin(radians)

    right = real & (bearings <= -side_min_deg)
    left = real & (bearings >= side_min_deg)

    left_fit = _orthogonal_line(xs[left], ys[left])
    right_fit = _orthogonal_line(xs[right], ys[right])

    if left_fit is None and right_fit is None:
        return CorridorEstimate()

    if left_fit is not None and right_fit is not None:
        left_angle, left_offset, left_spread = left_fit
        right_angle, right_offset, right_spread = right_fit

        disagreement = _angle_between(left_angle, right_angle)
        if disagreement > math.radians(MAX_SIDE_DISAGREEMENT_DEG):
            return CorridorEstimate(left_found=True, right_found=True)

        width = left_offset - right_offset
        if not (MIN_CORRIDOR_WIDTH_M <= width <= max_width_m):
            return CorridorEstimate(left_found=True, right_found=True)

        centre_y = (left_offset + right_offset) / 2.0
        angle = _mean_angle(left_angle, right_angle)
        if _disagrees_with_gap(angle, gap_bearing_rad):
            return CorridorEstimate(left_found=True, right_found=True)
        # Both walls seen and agreeing is the only case this is confident
        # about, and even then a scattered fit or a disagreement in direction
        # takes it back down.
        quality = 1.0 - min(
            1.0, (left_spread + right_spread) / (2.0 * MAX_SIDE_RESIDUAL_M)
        )
        agreement = 1.0 - disagreement / math.radians(MAX_SIDE_DISAGREEMENT_DEG)
        confidence = float(np.clip(0.5 + 0.5 * min(quality, agreement), 0.0, 1.0))
        return CorridorEstimate(
            lateral_offset_m=-centre_y,
            heading_error_rad=angle,
            half_width_m=width / 2.0,
            confidence=confidence,
            left_found=True,
            right_found=True,
        )

    # One wall only. The offset is unknowable -- a single wall says nothing
    # about where the far side is -- so it is reported as zero and the
    # confidence is capped low. The heading still carries real information:
    # one wall is enough to say which way the corridor runs, and that is the
    # signal that keeps a car parallel to a wall it is about to scrape.
    fit = left_fit if left_fit is not None else right_fit
    angle, offset, spread = fit
    if abs(offset) > MAX_SINGLE_WALL_M or _disagrees_with_gap(angle, gap_bearing_rad):
        return CorridorEstimate(
            left_found=left_fit is not None, right_found=right_fit is not None
        )
    quality = 1.0 - min(1.0, spread / MAX_SIDE_RESIDUAL_M)
    return CorridorEstimate(
        lateral_offset_m=0.0,
        heading_error_rad=angle,
        half_width_m=abs(offset),
        confidence=float(np.clip(0.35 * quality, 0.0, 0.35)),
        left_found=left_fit is not None,
        right_found=right_fit is not None,
    )


def free_gap(
    scan: np.ndarray,
    fov_deg: float,
    max_range: float,
    min_depth_m: float = 0.9,
) -> tuple[float, float]:
    """(bearing, depth) of the widest opening the car could drive into.

    The corridor fit above is precise where there is a corridor and silent
    where there is not, and measurement showed it goes silent in two places
    that matter: past about 30 degrees of corridor angle the side sectors run
    out of returns, because rays along a steeply angled wall escape instead of
    hitting it. That covers every 90 degree turn on this course -- the banked
    turn, the potholes into the open area, the bucket entrance -- and those
    are the moments the car most needs to be told where to go.

    This answers a cruder question that is always answerable: of the
    directions in front of me, which has the most room? It needs no walls, no
    pair of sides and no line fit, so it works at a corner, in the open floor
    before the buckets, and among scattered buckets. It is also exactly what
    gets a car through the car wash, whose ribbons the camera renders as a
    wall: the widest opening there is straight ahead between the uprights.

    Widest run rather than deepest ray, on purpose. A single bin reading
    6 m through a gap between two buckets is not somewhere a 0.30 m car fits;
    a broad run of moderately deep bins is. The run is scored on width times
    mean depth so that a wide shallow opening and a narrow deep one do not
    tie.

    That score alone picks the wrong opening at every narrow passage on this
    course, which measurement made plain before this weighting existed: the
    gap pointed 43 degrees off the way the course goes at the tunnel, 43 at
    the hoops and 33 in the bucket section. All three have the same shape --
    the car must thread something narrow while a wider, emptier space sits
    off to one side, and "widest" duly chose the space beside the tunnel over
    the tunnel.

    So the score is weighted by how far off the nose the opening lies. The
    weight only halves across the full field of view, which is deliberately
    mild: enough to prefer the passage the car is lined up with when the
    alternatives are comparable, not so strong that it refuses to look round
    a corner. A genuine 90 degree turn still wins, because there the forward
    direction is a wall a few tenths of a metre away and scores near zero
    however it is weighted.
    """
    bins = int(scan.shape[0])
    if bins < 4:
        return 0.0, 0.0

    half = fov_deg / 2.0
    edges = np.linspace(-half, half, bins + 1)
    bearings = (edges[:-1] + edges[1:]) / 2.0

    open_bins = scan >= min_depth_m
    if not open_bins.any():
        return 0.0, 0.0

    best_score = -1.0
    best_bearing = 0.0
    best_depth = 0.0
    start = None
    for index in range(bins + 1):
        inside = index < bins and open_bins[index]
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            run = slice(start, index)
            width_deg = float(bearings[index - 1] - bearings[start]) + fov_deg / bins
            depth = float(np.mean(scan[run]))
            centre_deg = float(np.mean(bearings[run]))
            forward = 1.0 - 0.5 * abs(centre_deg) / max(half, 1e-6)
            score = width_deg * depth * forward
            if score > best_score:
                best_score = score
                best_bearing = centre_deg
                best_depth = depth
            start = None
    return math.radians(best_bearing), min(best_depth, max_range)


def corridor_features(
    estimate: CorridorEstimate,
    gap_bearing_rad: float = 0.0,
    gap_depth_m: float = 0.0,
    offset_scale_m: float = 0.6,
    heading_scale_rad: float = math.pi / 2.0,
    width_scale_m: float = 1.5,
    gap_bearing_scale_rad: float = math.radians(55.0),
    gap_depth_scale_m: float = 3.0,
) -> np.ndarray:
    """The estimate as four observation channels in [0, 1].

    Offset and heading are signed and centred on 0.5, matching how speed and
    yaw rate are already encoded in the observation, so that "no information"
    (0.5) is distinct from "hard left" (0.0) rather than colliding with it.

    The heading scale is 90 degrees, not the 45 an earlier version used. A
    corridor really can run at a right angle to the car here -- this course
    turns through 90 degrees at the banked turn, at the potholes into the
    open area, and into the bucket section -- and a scale that saturated at
    45 would make every one of those corners indistinguishable from a
    moderate one, which is the same information loss as rejecting them, just
    quieter.

    The first four channels are multiplied by confidence, so an unconfident
    corridor arrives as a neutral 0.5 rather than as a number the policy has
    to learn to distrust. The last two are not: the free gap needs no walls
    and is always meaningful, and it is what carries the corners where the
    corridor fit falls silent.
    """
    offset = np.clip(estimate.lateral_offset_m / offset_scale_m, -1.0, 1.0)
    heading = np.clip(estimate.heading_error_rad / heading_scale_rad, -1.0, 1.0)
    width = np.clip(estimate.half_width_m / width_scale_m, 0.0, 1.0)
    confidence = np.clip(estimate.confidence, 0.0, 1.0)
    gap_bearing = np.clip(gap_bearing_rad / gap_bearing_scale_rad, -1.0, 1.0)
    gap_depth = np.clip(gap_depth_m / gap_depth_scale_m, 0.0, 1.0)
    return np.array(
        [
            0.5 + 0.5 * offset * confidence,
            0.5 + 0.5 * heading * confidence,
            width * confidence,
            confidence,
            0.5 + 0.5 * gap_bearing,
            gap_depth,
        ],
        dtype=np.float32,
    )


NUM_CORRIDOR_FEATURES = 6
