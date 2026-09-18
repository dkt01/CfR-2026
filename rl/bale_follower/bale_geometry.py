"""Bale-course geometry parsed from speed_course.sdf.

Pure NumPy/stdlib-XML -- no ROS, gym, or torch imports -- so it can be
unit-tested and visualized standalone. Serves three roles: the analytic
"virtual lidar" ray-caster, the collision check (car OBB vs. bale OBBs), and
the course polyline used for the progress reward.

The SDF has no rendering pipeline enabled for the ZED2i (see
speed_course.sdf's comment near the zed2i_housing visual), so there is no
live point cloud to consume. Bale poses/dimensions are exact ground truth in
the SDF, so ray-casting against them analytically gives the same conceptual
signal (range to nearest bale per angular bin) without needing GPU-backed
rendering. See rl/bale_follower/README.md for what would be needed to swap
this for a real point-cloud-derived lidar later.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

CHASSIS_LENGTH = 0.55
CHASSIS_WIDTH = 0.30
CHASSIS_LOCAL_OFFSET = (
    0.0,
    0.0,
)  # collision box is centered under the chassis origin in x/y


@dataclass
class Bale:
    index: int
    x: float
    y: float
    yaw: float
    half_x: float
    half_y: float

    def corners_2d(self) -> np.ndarray:
        return obb_corners(self.x, self.y, self.yaw, self.half_x, self.half_y)


def parse_bales(sdf_path: str) -> list[Bale]:
    """Parse every bale_<N>_collision box in the course_bales model."""
    with open(sdf_path, encoding="utf-8") as handle:
        text = handle.read()
    # The world files' prose comments use "--" as a dash, which is illegal
    # inside an XML comment body and trips ElementTree's strict parser
    # (Gazebo's own SDF parser is more lenient about it). The comments carry
    # no geometry, so stripping them before parsing is safe.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    root = ET.fromstring(text)
    bales_link = root.find(".//model[@name='course_bales']/link[@name='bales']")
    if bales_link is None:
        raise ValueError(f"course_bales/bales link not found in {sdf_path}")

    bales: list[Bale] = []
    for collision in bales_link.findall("collision"):
        name = collision.get("name", "")
        if not name.startswith("bale_") or not name.endswith("_collision"):
            continue
        index = int(name[len("bale_") : -len("_collision")])
        pose_text = collision.findtext("pose")
        size_text = collision.findtext("geometry/box/size")
        if pose_text is None or size_text is None:
            continue
        px, py, _pz, _roll, _pitch, yaw = (float(v) for v in pose_text.split())
        sx, sy, _sz = (float(v) for v in size_text.split())
        bales.append(
            Bale(index=index, x=px, y=py, yaw=yaw, half_x=sx / 2.0, half_y=sy / 2.0)
        )

    bales.sort(key=lambda bale: bale.index)
    return bales


def parse_vehicle_spawn(
    sdf_path: str, model_name: str = "slash"
) -> tuple[float, float, float]:
    """Return the vehicle's default (x, y, yaw) spawn pose from the SDF."""
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model = root.find(f".//model[@name='{model_name}']")
    if model is None:
        raise ValueError(f"model '{model_name}' not found in {sdf_path}")
    pose_text = model.findtext("pose")
    if pose_text is None:
        raise ValueError(f"model '{model_name}' has no <pose> in {sdf_path}")
    x, y, _z, _roll, _pitch, yaw = (float(v) for v in pose_text.split())
    return x, y, yaw


def obb_corners(
    x: float, y: float, yaw: float, half_x: float, half_y: float
) -> np.ndarray:
    """Four corners (4x2) of an oriented box centered at (x, y) with heading yaw."""
    local = np.array(
        [
            [half_x, half_y],
            [half_x, -half_y],
            [-half_x, -half_y],
            [-half_x, half_y],
        ]
    )
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return local @ rotation.T + np.array([x, y])


def car_obb_corners(x: float, y: float, yaw: float) -> np.ndarray:
    return obb_corners(x, y, yaw, CHASSIS_LENGTH / 2.0, CHASSIS_WIDTH / 2.0)


def obb_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """Separating Axis Theorem test for two convex quadrilaterals (4x2 arrays)."""
    for corners in (corners_a, corners_b):
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                continue
            axis /= norm
            proj_a = corners_a @ axis
            proj_b = corners_b @ axis
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False
    return True


# Every bale in this course shares the same footprint, so a bounding-circle
# radius (diagonal half-length) lets ray-cast/collision checks skip bales
# that are obviously too far away before doing exact box math.
_BALE_BOUNDING_RADIUS = math.hypot(0.9144 / 2.0, 0.4572 / 2.0)


def _ray_box_intersection(
    origin: np.ndarray, direction: np.ndarray, bale: Bale
) -> float | None:
    """Distance along `direction` (unit vector) to the nearest hit on bale's OBB, or None."""
    cos_yaw, sin_yaw = math.cos(-bale.yaw), math.sin(-bale.yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    local_origin = rotation @ (origin - np.array([bale.x, bale.y]))
    local_dir = rotation @ direction

    t_min, t_max = 0.0, math.inf
    for axis in range(2):
        half_extent = bale.half_x if axis == 0 else bale.half_y
        o, d = local_origin[axis], local_dir[axis]
        if abs(d) < 1e-9:
            if o < -half_extent or o > half_extent:
                return None
            continue
        t1 = (-half_extent - o) / d
        t2 = (half_extent - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    return t_min if t_min > 1e-9 else None


def nearest_bale_distance(
    bales: list[Bale], x: float, y: float, angle: float, max_range: float
) -> float:
    """Range to the nearest bale along a ray from (x, y) at world-frame `angle`."""
    origin = np.array([x, y])
    direction = np.array([math.cos(angle), math.sin(angle)])
    best = max_range
    for bale in bales:
        center_dist = math.hypot(bale.x - x, bale.y - y)
        if center_dist - _BALE_BOUNDING_RADIUS > best:
            continue
        hit = _ray_box_intersection(origin, direction, bale)
        if hit is not None and hit < best:
            best = hit
    return best


def lidar_scan(
    bales: list[Bale],
    x: float,
    y: float,
    yaw: float,
    num_bins: int,
    fov_deg: float,
    max_range: float,
) -> np.ndarray:
    """Range readings for `num_bins` angles spanning `fov_deg` centered on `yaw`."""
    half_fov = math.radians(fov_deg) / 2.0
    offsets = np.linspace(-half_fov, half_fov, num_bins)
    return np.array(
        [
            nearest_bale_distance(bales, x, y, yaw + offset, max_range)
            for offset in offsets
        ]
    )


def check_collision(bales: list[Bale], x: float, y: float, yaw: float) -> bool:
    """True if the car's chassis footprint overlaps any bale's footprint."""
    car_corners = car_obb_corners(x, y, yaw)
    for bale in bales:
        center_dist = math.hypot(bale.x - x, bale.y - y)
        car_radius = math.hypot(CHASSIS_LENGTH / 2.0, CHASSIS_WIDTH / 2.0)
        if center_dist - _BALE_BOUNDING_RADIUS - car_radius > 0:
            continue
        if obb_overlap(car_corners, bale.corners_2d()):
            return True
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect parsed bale-course geometry")
    parser.add_argument(
        "--sdf",
        default="../../jetson/cfr_arduino_bridge/worlds/speed_course.sdf",
        help="Path to speed_course.sdf",
    )
    parser.add_argument("--plot", action="store_true", help="Show a matplotlib plot")
    args = parser.parse_args()

    parsed_bales = parse_bales(args.sdf)
    spawn = parse_vehicle_spawn(args.sdf)
    print(f"Parsed {len(parsed_bales)} bales")
    for bale in parsed_bales[:3]:
        print(f"  bale_{bale.index}: x={bale.x:.4f} y={bale.y:.4f} yaw={bale.yaw:.5f}")
    print(f"Vehicle spawn: x={spawn[0]:.3f} y={spawn[1]:.3f} yaw={spawn[2]:.3f}")

    scan = lidar_scan(
        parsed_bales,
        spawn[0],
        spawn[1],
        spawn[2],
        num_bins=36,
        fov_deg=180,
        max_range=6.0,
    )
    print(
        f"Lidar scan at spawn (36 bins, 180 deg fov, 6m max): min={scan.min():.2f} max={scan.max():.2f}"
    )

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        for bale in parsed_bales:
            corners = np.vstack([bale.corners_2d(), bale.corners_2d()[0]])
            ax.plot(corners[:, 0], corners[:, 1], "saddlebrown", linewidth=1)
        ax.plot(spawn[0], spawn[1], "g*", markersize=18, label="vehicle spawn")
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"Parsed course geometry ({len(parsed_bales)} bales)")
        plt.show()
