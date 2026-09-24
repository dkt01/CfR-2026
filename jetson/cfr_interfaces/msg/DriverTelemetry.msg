# One control tick of rl/formulaOne's driver, published as ~/telemetry.
#
# Everything the driver decided and everything it decided it FROM, on the tick
# it decided it.  The bag already carries the raw inputs (pose, tachometer)
# and the output (/drive_cmd), but not the track-frame view in between --
# station, cross-track error, map clearance, the rule cap, the steering prior,
# the policy's own action before scaling.  Those are what a run is judged on,
# and reconstructing them offline means re-running the observation builder
# against a pose stream that may have been re-anchored mid-run.
#
# Published on EVERY tick, including the ones where the driver is holding
# neutral, so a gap in this topic is a dead node rather than a stale pose.

uint8 STATE_WAITING=0     # no start signal yet
uint8 STATE_RUNNING=1     # driving
uint8 STATE_STOPPING=2    # laps done, throttle off, steering live
uint8 STATE_FINISHED=3    # at rest after the run
uint8 STATE_STALE=4       # started, but the pose is older than pose_timeout
uint8 STATE_MANUAL_STOP=5 # stopped through ~/manual_start false

std_msgs/Header header    # control-tick time; frame_id is the track frame

uint8   state
string  driver            # "policy" or "baseline"

# Progress
float32 station           # m along the centerline, driving direction
float32 distance          # m driven since the start, all laps
uint16  lap               # completed laps
uint16  laps_target
float32 lap_time          # s into the current lap
float32 last_lap_time     # s, 0 until the first lap completes
float32 race_time         # s since the car first moved under this run

# Where the driver thinks the car is, in the TRACK frame (after the anchor)
float32 x
float32 y
float32 yaw
float32 cross_track       # m, + left of the centerline
float32 heading_error     # rad, car yaw minus centerline tangent
float32 clearance         # m, body to nearest bale off the MAP -- not a sensor

# Speed
float32 speed             # m/s the policy was given (tach, or pose fallback)
float32 v_cap             # m/s rule cap here
float32 v_floor           # m/s floor here
float32 yaw_rate          # rad/s, the filtered estimate the prior uses
float32 speed_rate        # m/s^2, filtered achieved dv/dt

# Decision
float32[] action          # raw [steer, throttle] from the policy, pre-scaling
float32 steer_ff          # the centerline prior's feedforward, normalized
float32 steer_cmd         # what went on the wire
float32 velocity_cmd      # m/s that went on the wire, after the cap and speed_scale
float32 speed_scale

# Health
float32 pose_age          # s since the last pose arrived
bool    speed_from_tach   # false when the tachometer was stale and pose-differenced
bool    relocalized       # a pose jump re-localised the station this tick
float32[] observation     # the full policy input; width depends on the config
