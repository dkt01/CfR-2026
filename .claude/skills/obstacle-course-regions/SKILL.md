---
name: obstacle-course-regions
description: Map CfR-2026 Obstacle Course obstacle names (overpass ramp, helical ramp, tunnel, narrow region, gravel pit, banked turn, potholes, wide/open region, buckets, hoops, car wash) to their world-meter coordinate bounds and driving order. Use this whenever the user asks which obstacle a telemetry point, RL episode position, crash location, or (x, y) coordinate falls in, wants to label or bucket simulation/log data by course section, asks about the order obstacles appear in on the obstacle course, or references any obstacle by name while discussing course layout, generate_obstacle_course.py, or the randomizer. Prefer this over re-deriving bounds from generate_obstacle_course.py's rect constants by hand -- the bundled script does that lookup and stays in sync with the generator automatically.
---

# Obstacle Course region lookup

Answers two questions about the CfR-2026 Obstacle Course: "what obstacle is
this point in/near" and "where is obstacle X." Backed by
`scripts/regions.py`, which imports `generate_obstacle_course.py`'s own rect
and circle constants (`GRAVEL`, `TUNNEL`, `BANK`, `HELIX_CENTRE`, ...) rather
than hardcoding a second copy of the course geometry -- if the DXF moves a
wall, regenerating the course keeps this script correct too, the same way
`obstacle_course_layout.yaml` stays correct for the randomizer.

Run it from the repo root:

```bash
python3 .claude/skills/obstacle-course-regions/scripts/regions.py list
python3 .claude/skills/obstacle-course-regions/scripts/regions.py show gravel_pit
python3 .claude/skills/obstacle-course-regions/scripts/regions.py near 2.1 -9.3
```

`list` prints every region with its bounds, source constant, and driving
order. `show <name>` accepts the canonical name or a common alias ("gravel
box", "bank", "helix"). `near <x> <y>` takes a world-meter point -- the same
frame ZED odometry, `/zed/zed_node/pose`, and the SDF's own poses use -- and
reports which region(s) contain it.

For labeling a batch of points (an RL episode's trajectory, a lap's
telemetry log) rather than one at a time, import the module directly instead
of shelling out per row:

```python
import sys
sys.path.insert(0, ".claude/skills/obstacle-course-regions/scripts")
from regions import classify

for x, y in trajectory:
    hits = classify(x, y)
    region_name = hits[0].name if hits else "lane"  # between named regions
```

## The eleven regions, in driving order

1. **overpass_ramp** -- the ramp up onto the bridge deck that passes over the tunnel
2. **helical_ramp** -- the 270 degree spiral down from the deck to the tunnel mouth
3. **tunnel** -- runs under the bridge deck
4. **narrow_region** -- the ordinary lane between the tunnel and the gravel pit
5. **gravel_pit** -- low-friction lid and scattered pebbles
6. **banked_turn** -- 8.5 degree bank
7. **potholes** -- raised bumps on a plywood board
8. **wide_open_region** -- the open floor around the car wash, before the bucket section
9. **buckets** -- randomized count and placement
10. **hoops** -- three, each sliding along its own line
11. **car_wash** -- five ribboned arches

**Two of these -- `narrow_region` and `wide_open_region` -- have no
dedicated rect constant in `generate_obstacle_course.py`.** They're real,
named stops in `jetson/README.md`'s tour of the course, but the drawing
doesn't box them the way it boxes the gravel pit or the bank: they're
"the lane in between" and "the open area before the buckets," not a poured
feature with edges. `regions.py` returns `bounds=None` for both rather than
inventing a box -- if you need a number for one of them anyway, say so
explicitly rather than presenting a guessed box as if it came from the
drawing (see `references/regions.md` for what's actually known about each).

The other nine are exact: they come straight from the generator's own
constants, in the same world-meter frame the simulation, telemetry, and
`obstacle_course_layout.yaml` all use. `helical_ramp` and `hoops` aren't
rectangles at all -- the helix is checked as a true annulus swept through
270 degrees (excluding its hollow centre and the wedge the tunnel sits in)
and hoops are checked as distance to each hoop's actual travel line, not a
shared bounding box. `list`/`show` still print a bounding box for those two
as a quick-glance number, but `near`/`classify` use the real shape -- don't
reason from the printed box for those two, ask the script instead.

## When a point matches zero or several regions

`classify()`/`near` can return nothing (the point is in the ordinary lane,
or in one of the two boxless regions) or, rarely, more than one (regions
this close never overlap in practice, but ramps sit close to the feature
they lead into). Report what actually came back rather than picking one --
if the caller wanted a single best guess, say which region is closest
instead of silently choosing.

See `references/regions.md` for the full table with sources and notes if
you want it in front of you without running the script.
