# What start_signal_node makes of the current camera frame, and whether the
# red-to-green transition that starts a run has been seen yet.
#
# The signal on the course carries a red and a green arm 90 degrees apart on a
# common pivot, so exactly one of them faces the car at a time -- and neither
# does while the pair passes edge-on part way through the turn, which is what
# UNKNOWN is for.  UNKNOWN is the normal state for a frame or two mid-turn, not
# an error.

uint8 UNKNOWN=0
uint8 RED=1
uint8 GREEN=2

# Stamped with the camera frame this was derived from, not the time it was
# processed, so a detection can be lined up against the image that caused it.
std_msgs/Header header

# One of the constants above: what this frame alone shows.
uint8 state

# Latched.  True once a confirmed green has followed a confirmed red, and it
# stays true until ~/reset -- this is the run trigger, and a driver that comes
# up late still receives it because ~/go is transient local.
bool go

# Pixels of signal color in the densest cluster_window-sized window of the
# searched area, which is what the min_pixels threshold is applied to.  Both
# are reported whatever the state, including on an UNKNOWN frame: "8 px
# against a threshold of 12" is what tuning this at the course needs.
uint16 red_pixels
uint16 green_pixels

# Everything matching in the searched area, cluster or not -- the region of
# interest, or the box around the signal once it has been found.  Much larger
# than the counts above means something else in frame shares the signal's
# color, which is expected outdoors and is why a candidate has to be
# arm-sized and in the right place rather than merely the right hue.
uint16 red_total
uint16 green_total

# Center of the winning cluster, in full-frame pixels whatever the region of
# interest was, so it can be compared against a camera_info projection.  Both
# are -1 when the state is UNKNOWN.
float32 x
float32 y

# True once somewhere in frame has held red long enough to be the signal --
# the detector knows where it is and is waiting for it to turn.  A run that
# has not armed will not start, so this is the field to look at when the car
# is sitting at the line: false here means the signal has not been found, not
# that it has not turned yet.
bool armed

# Where the detector believes the signal is, in full-frame pixels: the place
# it is watching for the transition, which it reads on relaxed color
# thresholds because the position is already known.  Both are -1 until the
# signal has been found.
float32 lock_x
float32 lock_y
