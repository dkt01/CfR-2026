# Plan Z: the backup driver for both courses

The Speed Course and Obstacle Course drivers are learned policies
([formulaOne](../../rl/formulaOne/DEPLOY.md),
[formulaTwo](../../rl/formulaTwo/README.md),
[obstacleRacer](../../rl/obstacleRacer/README.md)). Plan Z is what to run
when one of them will not finish on the day. It has no policy and nothing
trained: it steers from the walls the ZED sees, uses a drawing of each course
only as a hint of which way the lane goes next, and has a short list of knobs
for the things most likely to be wrong with the car or the course.

| | |
|---|---|
| Courses | both, one driver: `--course speed` (3 laps) or `--course obstacle` (2 laps) |
| Inputs | ZED point cloud, ZED pose, ZED IMU (cloud leveling), tachometer |
| Speed Course in Gazebo | 3 laps and a stop, no contact, 47-50 s a lap |
| Obstacle Course in Gazebo | 2 laps with every hoop on all four test layouts; 70 s a lap on the open ones, up to 160 s where the Wide Section is tight |
| Tolerates, without touching a knob | steering bias of 0.035 rad either way, steering gain 0.82 to 1.10, the camera yawed 3 degrees either way (Speed Course, Gazebo) |
| Compute | 3-4 ms a control tick in a lane, 10-15 ms in the Wide Section, on a laptop core; numpy only |

It is slower than the policies and is meant to be: defaults are the fastest
settings that passed every Gazebo run, backed off.

## Running it

In Gazebo, in the sim container with the workspace built (see the
[sim-launch skill](../../.agents/skills/sim-launch/SKILL.md)):

```bash
ros2 launch drivers/planZ/plan_z_sim.launch.py course:=speed green_after:=20
ros2 launch drivers/planZ/plan_z_sim.launch.py course:=obstacle green_after:=60
```

On the car, after `jetson/scripts/syncSoftware.sh --build` from the
development machine. Keep the E-Stop remote in hand; the wrapper arms the
actuators once it is told GO:

```bash
# Terminal 1 on the Orin: Arduino bridge and ZED, without cmd_vel_to_drive
~/software/scripts/launch.sh --no-cmd-vel

# Terminal 2
~/software/scripts/launchPlanZ.sh --course obstacle          # 0.3 of full speed
~/software/scripts/launchPlanZ.sh --course speed -s 0.6
~/software/scripts/launchPlanZ.sh --course speed steer_trim=0.02 camera_yaw_deg=-2
~/software/scripts/launchPlanZ.sh --course obstacle -n       # print the command only
```

The wrapper checks the bridge, the pose and the cloud are live and that
nothing else publishes `/drive_cmd`, then waits for the start signal (or
`ros2 service call /obstacle_racer/manual_start std_srvs/srv/SetBool "{data: true}"`,
`/formula_one/...` on the Speed Course). `knob=value` arguments set any knob
below for that run; `--help` lists the rest.

**Park the car on the start mark, square to the lane.** The route is laid out
from where the car stands when it is released. It is corrected as the car
drives (below), so a few degrees and a hand's width do not matter, but a car
parked across the lane starts with the hint pointing at a wall.

## How it works

One ROS-free core, [planner.py](planner.py), run by
[plan_z_node.py](plan_z_node.py) at 20 Hz. The node takes the name
`obstacle_racer` or `formula_one` by course, so the recorder, the Run Lab and
the driver-video skill work with it unchanged.

1. **See.** The ZED cloud goes through the compiled segmenter, as it does for
   obstacleRacer, and comes out as a 36-bin scan of the nearest obstacle on
   each bearing, plus the hoops. The cloud is leveled on the ZED's IMU (the
   pose is flat on the car: `two_d_mode`).
2. **Remember.** Scan returns nearer than 3 m are kept for a few seconds in
   the pose frame, so the wall beside the car, which the 110 degree camera
   lost a meter ago, is still there. A point has to be seen twice to be
   believed and is dropped when the camera later sees through it.
3. **Know roughly where the lane goes.** [routes/](routes) holds one line per
   course, made by [make_routes.py](make_routes.py) from the course drawings.
   It is tied to the pose at the start and then kept on the lane by the walls
   themselves: every frame with a wall on both sides moves it a share of the
   way to the middle of them, and on straights its heading is fitted to the
   path driven. That is what absorbs a course built off the drawing, a car
   parked askew and a drifting pose.
4. **Drive the lane.** In a walled lane the path *is* the route, slid to the
   middle of the walls actually seen. The tracker asks for the path's own
   curvature, read a little ahead, plus a pull back onto the path, and every
   command is rolled out through the car's lag against the remembered walls
   before it is sent (`_veto`): what would touch is replaced by the nearest
   command that does not.
5. **Thread the hoops.** Every hoop the segmenter reports is tracked, and the
   lane's line is eased across to pass through the middle of each.
6. **Find a way where there is no lane.** In the Wide Section and among the
   buckets the route names only the ways out (the four wall slots, then the
   exit). A cost-to-goal field is spread over a grid of what has been seen,
   and the car takes whichever of its own steering arcs ends furthest down
   that field; when none gets nearer it looks for the backing move after which
   one does. Unseen ground beside a wall counts as wall until looked at.
7. **Go as fast as the car can answer for.** No brakes, so the speed along the
   route is planned backward from each bend under coast drag. On top of that
   the driver measures how late the yaw follows the steering (`auto_lag`) and
   will not travel further in one lag than `lag_reach`. This matters: Gazebo's
   car takes about a second, a real servo a fraction of that, and one fixed
   setting weaves on one or cuts in on the other.
8. **Get unstuck.** Not moving while driving, or no command that clears what
   is ahead: back off with the wheel turned to bring the nose round, as far
   as the remembered points behind allow, and try again.

It also learns two things about the car as it drives, both bounded and both
with an off switch: a steering trim (`auto_trim`, on straights) and the
camera's yaw on its mount (`auto_camera_yaw`, from the lane seeming to point
off to one side while the car runs true).

## Knobs

All in [config.yaml](config.yaml), each with a comment. Set one for a run with
`launchPlanZ.sh ... name=value`, or edit the file and sync.

| What you see | Knob | Which way |
|---|---|---|
| Too slow / too fast everywhere | `-s` (`speed_scale`), `v_max` | the wrapper starts at 0.3 |
| Too fast in one place | `sections.<name>` (e.g. `sections.helical_ramp=0.8`) | m/s cap for that stretch |
| Runs wide into bends, or onto the outside wall of hairpins | `lat_accel`, `lag_room` | lower |
| Weaves down the straights | `track_omega` lower; `lag_reach` lower | |
| Turns in late / early | `lead_per_tau` | raise / lower |
| Pulls to one side | `steer_trim` (rad, + if it pulls left) | `auto_trim=false` to pin it |
| Sits off the middle of every lane, to the same side | `camera_yaw_deg` (+ if the camera points left) | `auto_camera_yaw=false` to pin it |
| Brushes bales | `body_margin` | raise (0.05) |
| Stops for gaps it should fit through | `body_margin`, `noise_sigmas` | lower, raise |
| Brakes for, or steers round, things that are not there | `memory_s` lower; `insert_range` lower | |
| Floor read as wall on the ramps | `level_source=imu`, `camera_pitch_deg` | |
| Whole course shifted or turned from the drawing | `route_offset_x`, `route_offset_y`, `route_offset_yaw_deg` | start-frame meters / degrees |
| Route hint doing more harm than good | `route_hint=false` | drives on the scan alone (lanes only) |
| Misses a hoop | `sections.hoops` lower; `gate_blend_m` | |
| Hesitates in the car wash | `soft_half_width` | raise toward 0.5 |
| Backs off too soon / too late when wedged | `stuck_s`, `reverse_distance` | |
| Backs off again and again without driving on | `forward_hold_s` | raise |
| Wanders in the Wide Section | `open_memory_s`, `shadow_m` | |

## What is tested

- [selftest.py](selftest.py): the core against a small 2D car with the real
  lag chain, on ovals, a 0.66 m lane, a 0.51 m pinch, a helix whose inside
  wall cannot be seen, and the Speed Course's own bales. Each on a car as
  modeled, one twice as quick to steer, one three times as slow, one like
  Gazebo's, with the camera yawed 4 degrees either way, and parked 5 degrees
  askew. Runs in a couple of minutes on the host: `python selftest.py`.
- [sim_numpy.py](sim_numpy.py): the Obstacle Course in rl/obstacleRacer's
  numpy sim, whole laps or one obstacle at a time (`--start hoops`), with
  steering, camera and pose faults.
- [validate.sh](validate.sh): Gazebo, the real node on the rendered ZED,
  judged from outside the driver by [gazebo_watch.py](gazebo_watch.py).
  `--matrix speed` and `--matrix obstacle` run the fault cases below. The
  faults are put into the simulated car and camera, not into the driver's
  knobs.

### Gazebo results (2026-10-02)

Each run alone on the machine: two simulations at once starve the driver of
camera frames and the result is a crash that is the test's, not the car's.

Speed Course, 3 laps and a stop (`validate.sh --course speed`, and the first
rows of `--matrix speed`):

| Case | Result | Lap times, s | Closest to a bale |
|---|---|---|---|
| nominal | pass | 50.0 / 47.4 / 46.9 | 6.5 cm |
| steering bias -0.035 rad | pass | 63.6 / 64.0 / 64.1 | 4.5 cm |
| steering bias +0.035 rad | pass | 62.9 / 64.6 / 64.6 | 6.4 cm |
| steering gain 0.82 | pass | 63.4 / 64.4 / 64.4 | 11.0 cm |
| steering gain 1.10 | pass | 49.9 / 46.4 / 46.3 | 6.3 cm |
| camera yaw -3 deg | pass | 63.6 / 64.0 / 64.0 | 6.1 cm |
| camera yaw +3 deg | pass | 62.8 / 64.1 / 63.9 | 2.2 cm |

With a fault in the car the lag estimate reads higher (the car's answer to
the wheel no longer fits the model) and the driver slows down for it: 64 s
laps instead of 47. That is the design working, and it is why `steer_trim`
and `camera_yaw_deg` are worth setting once they are known.

Pushing the knobs (`v_max=3.6 lag_reach=2.7 lag_room=0.8 lat_accel=2.0`)
gives 40 s laps with 2 to 3 cm of clearance. The defaults are that, backed
off.

Obstacle Course, 2 laps (`validate.sh --course obstacle --seed N`), judged on
laps and hoops:

| Layout seed | Result | Lap times, s | Hoops | Backing moves |
|---|---|---|---|---|
| 208 | pass | 71.8 / 67.1 | 3 + 3 | none |
| 218 | pass | 71.3 / 67.9 | 3 + 3 | none |
| 201 | pass | 160.2 / 72.2 | 3 + 3 | many, Wide Section |
| 202 | pass, twice | 107.2 / 129.4 and 143.5 / 125.9 | 3 + 3 | many, Wide Section |

Not run, for want of time before the competition: camera pitch and roll,
the combined faults, pose drift and route offset on the Speed Course with
the final driver; any fault case on the Obstacle Course; the larger fault
values that would show where it breaks. `validate.sh --matrix speed` and
`--matrix obstacle` run them. The numpy sim covers pose drift of 2 % with
0.1 deg/m of yaw drift through the Obstacle Course's lanes (4 of 4).

## On the day

1. `syncSoftware.sh --build`, `launch.sh --no-cmd-vel`, then
   `launchPlanZ.sh --course ... -n` to read the command it would run.
2. First run at the default 0.3. Watch the log line: `lag` should settle
   between 0.1 and 0.5 on the car, `trim` and `camera` near zero, `route`
   (the route's correction, m and degrees) small and steady.
3. If it sits to one side of every lane, read `camera` and `trim` off the log
   and set `camera_yaw_deg` / `steer_trim` to them so it starts right.
4. Raise `-s` in steps (0.5, 0.7, 1.0). The first thing to go is usually the
   exit of a hairpin: `lag_room` down, or `sections.track` down.

## Limits

- The Wide Section and the buckets are a search, not a line. Where the bales
  leave gaps near half a meter (layouts 201 and 202) it gets through by
  backing and filling, which can cost a minute or more; earlier versions got
  stuck there, and a layout tighter than those may still defeat it.
- It does not know the car wash is there except by the route; if the route
  hint is switched off, its strands are a wall.
- On the Obstacle Course the lap counter needs its cross-track gate
  (`lateral_gate:=3.0`), or it counts a lap in the pothole lane and stops the
  car a lap early. `obstacle_course.launch.py`, `obstacle_racer_car.launch.py`
  and Plan Z's launch files set it.
- It reads slower in Gazebo than it will on the car if the car's steering is
  as quick as a servo's: Gazebo's car takes most of a second to answer the
  wheel, and the driver caps its speed by that.
