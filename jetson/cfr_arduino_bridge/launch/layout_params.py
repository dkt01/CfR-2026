"""The obstacle layout file's hoops, as parameters for hoop_monitor_node.

Not a launch file.  obstacle_course_layout.yaml is keyed by the randomizer's
node name, so handed to hoop_monitor whole (`parameters=[layout_file]`) the
monitor loads none of it: it logs "no hoops declared", learns the hoops from
the randomizer's layout topic instead, and gives every one of them yaw 0.
hoop_0 and hoop_1 stand at yaw pi/2, so their crossings were judged against
a plane a quarter turn off, and a car through the middle of one was a miss.
The launch files pass this instead, the same way they import sensors_world.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def hoop_parameters(layout_file):
    """{"hoops": {...}} from the layout file, or {} where it declares none
    (the Speed Course) or there is no file."""
    path = Path(str(layout_file))
    if not path.is_file():
        return {}
    config = yaml.safe_load(path.read_text()) or {}
    hoops = (
        config.get("obstacle_randomizer", {}).get("ros__parameters", {}).get("hoops")
    )
    return {"hoops": hoops} if hoops else {}
