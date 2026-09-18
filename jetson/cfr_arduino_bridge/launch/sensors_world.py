"""Rendered-sensor world resolution, shared by the simulation and training stacks.

Both stacks need the same simulated ZED: if the camera were defined twice they
would drift, and a policy trained against one would be reading a different
sensor at deployment. Imported by path (the launch directory is not a Python
package), so both launch files insert this directory into sys.path first.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from launch.substitutions import LaunchConfiguration

SYSTEM_MARKER = "<!-- cfr:sensors-system -->"
CAMERA_MARKER = "<!-- cfr:sensors-camera -->"

SENSORS_SYSTEM = """<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>"""

# One rgbd_camera rather than a colour and a depth sensor: they would share a
# calibration anyway, and Gazebo publishes image, depth_image, points and
# camera_info off this one.  simulation.launch.py's bridge renames them to the
# ZED's topics below.
SENSORS_CAMERA = """<sensor name="zed2i" type="rgbd_camera">
            <pose>0.315 0 0.20 0 0 0</pose>
            <always_on>1</always_on>
            <update_rate>15</update_rate>
            <topic>/zed/gz/rgbd</topic>
            <camera>
              <horizontal_fov>1.91986</horizontal_fov>
              <image><width>640</width><height>360</height><format>R8G8B8</format></image>
              <clip><near>0.2</near><far>20</far></clip>
            </camera>
          </sensor>"""


def resolve_world(context, *_args, **_kwargs):
    """Hand Gazebo the world, with the rendered sensors switched on or not.

    Without `sensors:=true` the world is used exactly as it sits in the
    package, markers and all -- an XML comment costs nothing.  With it, the
    markers are replaced and the result written beside the other simulation
    scratch files, because Gazebo takes a path and not a string.
    """
    world = Path(context.perform_substitution(LaunchConfiguration("world")))
    if context.perform_substitution(LaunchConfiguration("sensors")).lower() not in (
        "true",
        "1",
    ):
        return [world]

    text = world.read_text()
    for marker, replacement in (
        (SYSTEM_MARKER, SENSORS_SYSTEM),
        (CAMERA_MARKER, SENSORS_CAMERA),
    ):
        if marker not in text:
            raise RuntimeError(
                f"{world.name} has no {marker}, so sensors:=true cannot add the "
                "rendered ZED to it"
            )
        text = text.replace(marker, replacement, 1)

    scratch = Path(tempfile.gettempdir()) / "cfr_sim"
    scratch.mkdir(parents=True, exist_ok=True)
    rendered = scratch / world.name
    rendered.write_text(text)
    return [rendered]
