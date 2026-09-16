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

# Pixels of signal colour in the densest cluster_window-sized window of the
# region of interest, which is what the min_pixels threshold is applied to.
# Both are zero on an UNKNOWN frame.
uint16 red_pixels
uint16 green_pixels

# Everything matching in the whole region of interest, cluster or not.  Much
# larger than the counts above means something else in frame shares the
# signal's colour; narrow the region of interest.
uint16 red_total
uint16 green_total

# Centre of the winning cluster, in full-frame pixels whatever the region of
# interest was, so it can be compared against a camera_info projection.  Both
# are -1 when the state is UNKNOWN.
float32 x
float32 y
