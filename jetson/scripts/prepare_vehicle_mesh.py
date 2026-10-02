#!/usr/bin/env python3
"""Turn the Onshape OBJ of the car into the meshes generate_vehicle_model.py uses.

    ./scripts/prepare_vehicle_mesh.py [--source ../path/to/robot.obj]

The CAD export is in metres but at an arbitrary rotation, and it is ONE mesh with
the wheels baked in, so on its own the wheels could neither spin nor steer.  This
rotates it into the chassis frame (x forward, y left, z up), cuts the four tyres
out, and writes

    vehicle_body.obj                     origin = midpoint of the four axle centres
    vehicle_wheel_{front,rear}_{left,right}.obj
                                         origin = that wheel's axle centre, axle
                                         along local z, as the wheel links want it
    vehicle.mtl                          shared materials

Each wheel is written in its own link frame (x forward, y down, z left, because
the wheel links are posed with a -90 degree roll), so the generator can use the
files unmodified and nothing is mirrored.

The wheel positions come from the mesh, not from vehicle.yaml, and differ from
it by a couple of centimetres (the CAD is 0.335 m wheelbase / 0.25 m track
against the measured 0.324 / 0.290).  The generator puts each wheel mesh on the
simulator's wheel link, so physics and tyre contact keep the measured geometry;
only the body is placed from the mesh.

The orientation below (FORWARD / UP, in the CAD's own axes) was read off the
export: the four tyres sit on a diagonal in the CAD's x-y plane with their axles
along its z, and the ZED 2i is the grey part at the front.  Re-export the model
at a different rotation and these two vectors are the only thing to redo.
"""

import argparse
from pathlib import Path

import numpy as np

MESHES = Path(__file__).resolve().parents[1] / "cfr_arduino_bridge" / "meshes"

# CAD-frame unit vectors.  Lateral is derived: left = up x forward.
FORWARD = np.array([-0.63, -0.78, 0.0])
UP = np.array([0.78, -0.63, 0.0])

# The tyre is the only part drawn in this (pure black) material.
TIRE_MATERIAL_PREFIX = "0.000000_"
# Anything with its centre inside this cylinder travels with the wheel, so the
# rim and hub spin with the tyre instead of staying on the body.
WHEEL_RADIUS_M = 0.062
WHEEL_HALF_WIDTH_M = 0.032


def read_obj(path):
    verts, normals, faces, face_mat = [], [], [], []
    mtl_order, current = [], None
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vn "):
            normals.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("usemtl"):
            current = line.split()[1]
            if current not in mtl_order:
                mtl_order.append(current)
        elif line.startswith("f "):
            ids = [t.split("/") for t in line.split()[1:]]
            faces.append([(int(t[0]) - 1, int(t[2]) - 1 if len(t) > 2 else -1) for t in ids])
            face_mat.append(current)
    return np.array(verts), np.array(normals), faces, face_mat, mtl_order


def write_obj(path, verts, normals, faces, face_mat, rotation, offset):
    """Write the chosen faces, rotating positions/normals and shifting positions."""
    used_v = sorted({v for f in faces for v, _ in f})
    used_n = sorted({n for f in faces for _, n in f if n >= 0})
    vmap = {v: i + 1 for i, v in enumerate(used_v)}
    nmap = {n: i + 1 for i, n in enumerate(used_n)}
    out = ["mtllib vehicle.mtl"]
    out += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts[used_v] @ rotation.T - offset]
    out += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in normals[used_n] @ rotation.T]
    last = None
    for f, m in zip(faces, face_mat):
        if m != last:
            out.append(f"usemtl {m}")
            last = m
        out.append("f " + " ".join(f"{vmap[v]}//{nmap[n]}" for v, n in f))
    path.write_text("\n".join(out) + "\n")
    return path.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=MESHES / "robot.obj")
    args = parser.parse_args()

    verts, normals, faces, face_mat, mtl_order = read_obj(args.source)

    fwd = FORWARD / np.linalg.norm(FORWARD)
    up = UP / np.linalg.norm(UP)
    # Rows are the chassis axes expressed in CAD axes: x forward, y left, z up.
    to_chassis = np.stack([fwd, np.cross(up, fwd), up])
    pos = verts @ to_chassis.T

    centroids = np.array([pos[[v for v, _ in f]].mean(axis=0) for f in faces])
    is_tire = np.array([m.startswith(TIRE_MATERIAL_PREFIX) for m in face_mat])

    # Four tyres = the four quadrants of the tyre faces about their own midpoint.
    tire_pts = centroids[is_tire]
    mid = (tire_pts.min(axis=0) + tire_pts.max(axis=0)) / 2.0
    axle_centres = {}
    for fb in ("front", "rear"):
        for side in ("left", "right"):
            quadrant = (
                ((tire_pts[:, 0] > mid[0]) == (fb == "front"))
                & ((tire_pts[:, 1] > mid[1]) == (side == "left"))
            )
            pts = tire_pts[quadrant]
            axle_centres[(fb, side)] = (pts.min(axis=0) + pts.max(axis=0)) / 2.0

    # Axle midpoint = the chassis frame origin the generator expects.
    body_origin = np.mean(list(axle_centres.values()), axis=0)
    wheelbase = axle_centres[("front", "left")][0] - axle_centres[("rear", "left")][0]
    track = axle_centres[("front", "left")][1] - axle_centres[("front", "right")][1]
    print(f"mesh wheelbase {wheelbase:.4f} m, track {track:.4f} m")

    claimed = np.zeros(len(faces), dtype=bool)
    # Rotation that takes chassis-frame vectors into a wheel link's frame.
    wheel_axes = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    for (fb, side), centre in axle_centres.items():
        rel = centroids - centre
        inside = (np.hypot(rel[:, 0], rel[:, 2]) < WHEEL_RADIUS_M) & (
            np.abs(rel[:, 1]) < WHEEL_HALF_WIDTH_M
        )
        mine = inside & ~claimed
        claimed |= mine
        wheel_faces = [faces[i] for i in np.flatnonzero(mine)]
        wheel_mat = [face_mat[i] for i in np.flatnonzero(mine)]
        name = f"vehicle_wheel_{fb}_{side}.obj"
        # Positions are shifted in chassis frame first, then rotated into the link
        # frame; write_obj shifts after rotating, so fold the shift into `offset`.
        rotation = wheel_axes @ to_chassis
        offset = wheel_axes @ (centre)
        size = write_obj(MESHES / name, verts, normals, wheel_faces, wheel_mat, rotation, offset)
        print(f"{name}: {len(wheel_faces)} faces, {size / 1e6:.1f} MB")

    body_idx = np.flatnonzero(~claimed)
    size = write_obj(
        MESHES / "vehicle_body.obj",
        verts,
        normals,
        [faces[i] for i in body_idx],
        [face_mat[i] for i in body_idx],
        to_chassis,
        body_origin,
    )
    print(f"vehicle_body.obj: {len(body_idx)} faces, {size / 1e6:.1f} MB")

    src_mtl = args.source.with_suffix(".mtl")
    (MESHES / "vehicle.mtl").write_text(src_mtl.read_text())
    print(f"vehicle.mtl copied from {src_mtl.name}; {len(mtl_order)} materials")


if __name__ == "__main__":
    main()
