# Progress of an autonomous run, in laps of the start/finish line.
#
# The car waits behind the line, so it crosses it once on the way out.  That
# outbound crossing arms the counter and is not a lap: `laps` counts returns.

uint8 STATE_IDLE=0        # no reference pose yet, waiting to arm
uint8 STATE_ARMED=1       # reference latched, still behind the line
uint8 STATE_RUNNING=2     # out on course, laps counting
uint8 STATE_DONE=3        # target reached; latched until ~/reset

std_msgs/Header header    # the pose stamp this was derived from, not processing time
uint8   state
uint16  laps              # completed laps; the outbound crossing is not one
uint16  target            # target_laps, echoed so a driver need not be configured too
bool    done              # laps >= target
bool    counting          # auto active, not e-stopped, pose and status fresh
float32 along_track       # m ahead of the start pose, + the way the run started
float32 cross_track       # m left of the start pose's heading
float32 heading_error     # rad from the latched start heading, wrapped to (-pi, pi]
float32 lap_distance      # m driven since the last counted crossing
uint16  rejected          # plane crossings seen but not counted, since arming
uint16  loop_closures     # pose discontinuities seen since arming
builtin_interfaces/Time last_crossing   # stamp of the most recent counted crossing
