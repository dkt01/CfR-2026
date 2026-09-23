"""Small, ROS-free left-wall controller for the simulated bale course."""

import math
import re
import json
import xml.etree.ElementTree as ET


WHEELBASE = 0.335
MAX_STEER = 0.45


def load_bales(sdf):
    # The generated world contains descriptive comments with "--", which
    # Gazebo accepts but strict XML parsers reject. Comments are irrelevant.
    source = re.sub(r"<!--.*?-->", "", open(sdf, encoding="utf-8").read(), flags=re.S)
    root = ET.fromstring(source)
    link = root.find(".//model[@name='course_bales']/link[@name='bales']")
    if link is None:
        raise ValueError("course_bales missing from SDF")
    bales = []
    for collision in link.findall("collision"):
        if not collision.get("name", "").startswith("bale_"):
            continue
        x, y, _, _, _, yaw = map(float, collision.findtext("pose").split())
        sx, sy, _ = map(float, collision.findtext("geometry/box/size").split())
        bales.append((x, y, yaw, sx / 2, sy / 2))
    return bales


def range_to_bales(bales, x, y, yaw, bearing, max_range=5.0):
    """Nearest intersection of a horizontal ray and an oriented bale box."""
    angle = yaw + bearing
    dx, dy = math.cos(angle), math.sin(angle)
    closest = max_range
    for bx, by, byaw, hx, hy in bales:
        if math.hypot(bx - x, by - y) > closest + math.hypot(hx, hy):
            continue
        c, s = math.cos(byaw), math.sin(byaw)
        ox = c * (x - bx) + s * (y - by)
        oy = -s * (x - bx) + c * (y - by)
        vx = c * dx + s * dy
        vy = -s * dx + c * dy
        near, far = 0.0, closest
        for origin, direction, half in ((ox, vx, hx), (oy, vy, hy)):
            if abs(direction) < 1e-9:
                if abs(origin) > half:
                    break
            else:
                a, b = (-half - origin) / direction, (half - origin) / direction
                near, far = max(near, min(a, b)), min(far, max(a, b))
                if near > far:
                    break
        else:
            if 0.01 < near < closest:
                closest = near
    return closest


def command(bales, x, y, yaw):
    """Return speed and steering angle, keeping the left bale face ~0.55 m away.

    A forward-left ray estimates the wall tangent. The forward ray supplies
    an early right turn when the left wall bends across the lane. At an opening,
    the last visible left wall still gives a usable tangent.
    """
    left = range_to_bales(bales, x, y, yaw, math.pi / 2)
    ahead_left = range_to_bales(bales, x, y, yaw, math.pi / 4)
    front = min(range_to_bales(bales, x, y, yaw, math.radians(a)) for a in (-15, 0, 15))
    # The 45-degree ray hits a wall approximately 0.7 m farther forward.
    tangent = math.atan2(
        ahead_left / math.sqrt(2) - left, max(0.2, ahead_left / math.sqrt(2))
    )
    error = max(-1.0, min(1.0, left - 0.55))
    steer = 0.85 * tangent + 0.65 * error
    if front < 1.5:
        steer -= 0.8 * (1.5 - front)
    steer = max(-MAX_STEER, min(MAX_STEER, steer))
    speed = 0.65 if front < 1.5 or abs(steer) > 0.30 else 1.0
    return speed, steer


class CourseFollower:
    """Follow the wall-derived centerline with a small left-range correction.

    The centerline resolves gaps between individual bales and prevents a
    reactive range controller from circling a single bale near the start.
    """

    def __init__(self, path_file):
        path = json.load(open(path_file, encoding="utf-8"))
        # The stored polyline runs opposite the speed-course starting heading.
        self.path = list(zip(path["x"], path["y"]))[::-1]
        self.index = None

    def command(self, bales, x, y, yaw):
        points = self.path
        n = len(points)
        if self.index is None:
            self.index = min(range(n), key=lambda i: math.dist(points[i], (x, y)))
        else:
            self.index = min(
                ((self.index + j) % n for j in range(16)),
                key=lambda i: math.dist(points[i], (x, y)),
            )
        tx, ty = points[(self.index + 12) % n]
        distance = math.hypot(tx - x, ty - y)
        heading = math.atan2(ty - y, tx - x)
        error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        steer = math.atan2(2 * WHEELBASE * math.sin(error), max(distance, 0.1))
        # Keep an eye on the left boundary; a small bias corrects lateral
        # drift without letting a bale gap reverse the path direction.
        left = range_to_bales(bales, x, y, yaw, math.pi / 2)
        if left < 1.2:
            steer += 0.05 * (left - 0.55)
        steer = max(-MAX_STEER, min(MAX_STEER, steer))
        return 0.8, steer
