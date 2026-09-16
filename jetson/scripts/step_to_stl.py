#!/usr/bin/env python3
"""Tessellate the obstacle CAD assembly into the STL visuals the world uses.

The per-part STEP exports carry assembly coordinates but no relationships, so
this reads the *assembled* ``All Obstacles.step`` instead: every obstacle's
pieces already sit in the right place relative to each other, and only the
placement of whole obstacles on the course has to come from the DXF.

Each obstacle is emitted as one STL in a canonical local frame -- footprint
centred on the origin, ground at ``z = 0`` -- so the generator can place it
with a plain ``<pose>`` and nothing has to know about the CAD origin.

Meshes are visual only -- collision stays on primitives in the generated SDF
-- so tessellation is decimated hard.  At CAD tolerance the assembly is 6.7 M
triangles, most of it thread detail on parts that read as a cylinder from a
camera 2 m away.  Parts that *are* a cylinder or a flat ribbon (the flanges
under the hoops and car-wash arches, the 40 hanging strips) are left out here
altogether and drawn as SDF primitives instead.  So is the whole ramp / flat
bridge / helical ramp structure: the car drives on it, the CAD helix radius
and the DXF one disagree by 5%, and the DXF is what the bale walls around it
were drawn from -- so the generator builds that from the DXF arc directly.

Decimation floors out on boundary edges, which is why the tolerance above is
coarse: an open shell tessellated finely has more boundary edges than the
triangle budget, and no amount of collapsing gets under them.

Runs in the container built from ``step_to_stl.Dockerfile``; see
``convert_obstacle_meshes.sh``, which builds it and wraps the invocation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import cascadio
import fast_simplification
import numpy as np
import trimesh

# Top of the 25 m ground plane the assembly is laid out on.  Everything is
# shifted by this so that the course floor is z = 0.
GROUND_Z = -5.2336

# Tessellation tolerance, in metres, before decimation.  Finer than the final
# meshes need, because decimating a good mesh beats tessellating a coarse one.
TOL_LINEAR = 0.02
TOL_ANGULAR = 0.4

# An obstacle is the set of assembly nodes whose name starts with one of
# `parts` and whose centroid falls inside `region` (x0, x1, y0, y1), in
# assembly coordinates.  Names alone are not enough to tell the parts apart:
# the assembly holds three identical hoops, fourteen identical pothole bumps
# and two `1_5_ramp`s in different sections, so the region picks the one
# instance each mesh is cut from.
#
# `budget` is the triangle count to decimate to.  Without `anchor` the local
# origin is the footprint centre at ground level; `"pivot"` instead centres
# the signal arms on the axis they swing about.
OBSTACLES = [
    {
        "name": "hoop",
        "parts": ("hoop_base_1", "hoop_1"),
        "region": (-6.9, -6.1, 6.5, 7.1),
        "budget": 2500,
    },
    {
        "name": "bucket",
        "parts": ("Bucket",),
        "region": (-1.6, -1.1, 14.9, 15.4),
        "budget": 1200,
    },
    {
        "name": "start_signal_frame",
        "parts": (
            "base",
            "top_brace",
            "face_board",
            "pivot_brace",
            "side_brace",
            "rear_face",
        ),
        "region": (-12.0, -11.1, 9.0, 9.2),
        "budget": 1200,
    },
    {
        "name": "start_signal_arm_red",
        "parts": ("red_signal",),
        "region": (-12.0, -11.1, 9.0, 9.2),
        "budget": 500,
        "anchor": "pivot",
    },
    {
        "name": "start_signal_arm_green",
        "parts": ("green_signal",),
        "region": (-12.0, -11.1, 9.0, 9.2),
        "budget": 500,
        "anchor": "pivot",
    },
    {
        "name": "car_wash_base",
        "parts": ("carwash_base",),
        "region": (-7.6, -5.0, 11.0, 12.4),
        "budget": 100,
    },
    {
        # One arch.  The five on the course are identical and evenly spaced,
        # so the world instances this mesh rather than carrying five copies of
        # it: a fifth of the committed bytes and a fifth of the triangles in
        # the scene.  Decimation floors out around 2 000 triangles an arch --
        # they are open tube shells and the boundary edges cannot collapse --
        # so five baked into one mesh would not fit pre-commit's 500 KB limit
        # either.
        "name": "car_wash_arch",
        "parts": ("carwash_structure",),
        "region": (-7.25, -7.19, 11.0, 12.4),
        "budget": 2000,
    },
    {
        "name": "tunnel",
        "parts": ("tunnel",),
        "region": (-12.2, -10.6, 15.6, 16.5),
        "budget": 200,
    },
    {
        "name": "gravel_box",
        "parts": ("box_base", "box_end", "box_side", "box_gravel_fill"),
        "region": (-7.6, -5.0, 15.5, 16.8),
        "budget": 400,
    },
    {
        "name": "pothole_board",
        "parts": ("pothole_base", "pothole_board"),
        "region": (-7.5, -5.0, 13.6, 14.9),
        "budget": 4000,
    },
    {
        "name": "pothole_bump",
        "parts": ("pothole_bump_1",),
        "region": (-7.3, -7.0, 13.6, 13.9),
        "budget": 200,
    },
    {
        "name": "ramp_up",
        "parts": ("2_32_ramp",),
        "region": (-8.5, -7.5, 15.7, 16.6),
        "budget": 100,
    },
    {
        "name": "ramp_narrowing",
        "parts": ("2_narrowing_ramp",),
        "region": (-5.1, -4.1, 15.6, 16.7),
        "budget": 100,
    },
    {
        "name": "ramp_pothole_entry",
        "parts": ("1_5_ramp_1",),
        "region": (-8.4, -7.4, 13.7, 14.8),
        "budget": 100,
    },
    {
        "name": "bank",
        "parts": ("base panel", "base lifter", "rear panel", "side panel"),
        "region": (-1.5, -0.1, 13.5, 16.9),
        "budget": 600,
    },
]


def load_assembly(step_file: Path) -> trimesh.Scene:
    glb = Path(tempfile.gettempdir()) / "cfr_obstacles.glb"
    cascadio.step_to_glb(
        str(step_file), str(glb), tol_linear=TOL_LINEAR, tol_angular=TOL_ANGULAR
    )
    return trimesh.load(str(glb))


def placed_meshes(scene: trimesh.Scene) -> list[tuple[str, trimesh.Trimesh]]:
    """Every geometry node, baked into assembly coordinates."""
    out = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        out.append((node, mesh))
    return out


def select(meshes, spec) -> list[trimesh.Trimesh]:
    x0, x1, y0, y1 = spec["region"]
    chosen = []
    for name, mesh in meshes:
        if not name.startswith(spec["parts"]):
            continue
        centre = mesh.bounds.mean(axis=0)
        if x0 <= centre[0] <= x1 and y0 <= centre[1] <= y1:
            chosen.append(mesh)
    return chosen


def decimate(parts: list[trimesh.Trimesh], budget: int) -> trimesh.Trimesh:
    """Reduce an obstacle's parts to `budget` triangles between them.

    Each part gets a share of the budget proportional to what it costs, rather
    than the obstacle being concatenated and decimated as one.  Decimating the
    whole spends the budget where the triangles are and deletes everything
    else: the car wash is 300 k triangles of threaded arch around a 12
    triangle base plate, and collapsing that as a single mesh loses the plate
    entirely.  The floor of 12 leaves parts that are already a plain box
    untouched.
    """
    total = sum(len(part.faces) for part in parts)
    reduced = []
    for part in parts:
        part.merge_vertices()
        share = max(12, round(budget * len(part.faces) / total))
        if len(part.faces) > share:
            vertices, faces = fast_simplification.simplify(
                part.vertices.astype(np.float32),
                part.faces.astype(np.uint32),
                target_reduction=1.0 - share / len(part.faces),
            )
            part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        reduced.append(part)
    return trimesh.util.concatenate(reduced)


def plate_angle(mesh: trimesh.Trimesh) -> float:
    """Rotation of a flat signal arm about the assembly Y axis.

    The arms are drawn mid-swing, 45 degrees either side of the position they
    rest in, so their tessellated pose is not the pose the model wants.  The
    arm is a long thin plate, so the angle that flattens its z extent is the
    angle it was drawn at.  A search beats fitting a principal axis here --
    the plate's rounded ends and mounting holes put enough vertices off the
    long axis to pull a least-squares fit well away from it.
    """
    x, z = mesh.vertices[:, 0], mesh.vertices[:, 2]
    best, step, centre = 0.0, math.radians(1.0), 0.0
    for _ in range(4):
        angles = [centre + step * k for k in range(-90, 91)]
        extents = [np.ptp(-x * math.sin(a) + z * math.cos(a)) for a in angles]
        centre = angles[int(np.argmin(extents))]
        best, step = centre, step / 10.0
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_file", type=Path, help="assembled All Obstacles.step")
    parser.add_argument("out_dir", type=Path, help="directory to write STL files to")
    args = parser.parse_args()

    scene = load_assembly(args.step_file)
    meshes = placed_meshes(scene)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    origins: dict[str, np.ndarray] = {}
    for spec in OBSTACLES:
        parts = select(meshes, spec)
        if not parts:
            print(f"{spec['name']}: no parts matched", file=sys.stderr)
            return 1
        whole = trimesh.util.concatenate(parts)

        if spec.get("anchor") == "pivot":
            # Both arms turn about their own centroid, which is where the
            # pivot brace crosses them.  Undo the drawn swing so the exported
            # arm lies along +x and the model can state its own angle.
            origin = whole.vertices.mean(axis=0)
            # ptp is unchanged by translation, so the angle measured on the
            # part in place is still the angle to undo once it is centred.
            swing = trimesh.transformations.rotation_matrix(
                plate_angle(whole), [0, 1, 0], [0, 0, 0]
            )
            for part in parts:
                part.apply_translation(-origin)
                part.apply_transform(swing)
        else:
            low, high = whole.bounds
            origin = np.array(
                [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, GROUND_Z]
            )
            if "origin_of" in spec:
                origin = origins[spec["origin_of"]]
            for part in parts:
                part.apply_translation(-origin)
        origins[spec["name"]] = origin

        mesh = decimate(parts, spec["budget"])
        target = args.out_dir / f"{spec['name']}.stl"
        mesh.export(target)
        low, high = mesh.bounds
        print(
            f"{spec['name']:24s} {len(mesh.faces):6d} tris  "
            f"size {high[0] - low[0]:6.3f} x {high[1] - low[1]:6.3f} x {high[2] - low[2]:6.3f} m  "
            f"-> {target.name} ({os.path.getsize(target) / 1024:.0f} kB)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
