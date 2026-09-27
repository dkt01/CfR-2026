#!/usr/bin/env python3
"""Generate the straw-bale look for the Gazebo worlds: textures and loose strands.

    python3 jetson/scripts/generate_straw_assets.py     # then generate_speed_course.py

Writes, into jetson/cfr_arduino_bridge:

    materials/straw_bale_{0,1,2}.png   albedo, three variants so a wall of
                                       202 bales does not repeat one tile
    materials/straw_bale_normal.png    the stalks' relief, for the lighting
    meshes/straw_bale.obj              the bale itself, 36 x 18 x 14 in with its
                                       top and vertical edges rounded (a real
                                       bale is not a sharp box), smooth normals
                                       and a texture coordinate per face
    meshes/bale_strands_{0,1,2,3}.stl  loose straw around one bale: stalks
                                       poking out of the faces, bunched along
                                       the top edges, and litter on the ground
                                       along the long sides

What it is copying, from photos of the real course: pale gold stalks lying
mostly along the bale with dark gaps between them, sun-bleached tops, and a
fringe of loose straw everywhere -- sticking out of every face, drooping off
the top edges and scattered on the asphalt at the foot of each wall.

All of it is VISUAL.  The collision stays the 36 x 18 x 14 in box, so the car
hits exactly what it hit before.  But the rendered camera does see the
strands, so Gazebo's depth image does too: a few pixels per column now land
up to ~15 cm in front of the bale face, as loose straw does for the real ZED.
generate_speed_course.py --no-strands leaves them out.  Both courses use
them: bale_visuals below is the XML both generators write.

Seeded, so re-running reproduces the same files byte for byte.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PACKAGE = Path(__file__).resolve().parents[1] / "cfr_arduino_bridge"
INCH = 0.0254
LENGTH, WIDTH, HEIGHT = 36 * INCH, 18 * INCH, 14 * INCH

# Albedo size: the long face is 0.914 x 0.356 m, so 1024 x 400 keeps a stalk
# about 3-4 px wide for a 3-4 mm straw and undistorted on the faces the car
# looks at.  The top and ends reuse it, slightly stretched.
TEX_W, TEX_H = 1024, 400
VARIANTS = 3
STRAND_VARIANTS = 4

# sRGB, from the photos: sunlit gold, pale bleached, deeper tan, and the
# brown-gray of the gaps between stalks.  Measured off a course photo, the
# bale faces are hue ~0.10, saturation 0.29-0.38 with a contrast (std/mean)
# near 0.3 -- far grayer than the old flat 0.72 0.48 0.12 (saturation 0.83).
# The finished texture is pulled to that; brightness is left to the lighting.
GOLD = np.array([206, 182, 128])
PALE = np.array([226, 212, 170])
TAN = np.array([172, 148, 100])
GAP = np.array([98, 86, 62])
TARGET_SATURATION = 0.36
TARGET_CONTRAST = 0.30
# The bale's <diffuse>: a light straw tint rather than white, so a renderer
# that cannot find the PNG (the web viewer, before its asset list is rebuilt)
# still draws a straw bale.  The COLOR stays in the texture: the two
# renderers disagree on the product -- three.js multiplies the texture by the
# diffuse, Gazebo's ogre2 only partly (moving all the color into the diffuse
# rendered 0.19 saturation in Gazebo against the photo's 0.36).  Light enough
# that the browser's product stays close to the texture.
STRAW_DIFFUSE = (1.0, 0.94, 0.82)


# ------------------------------------------------------------- in a world
# The XML both course generators write for a straw bale's visuals: the
# rounded body with one of the textures, and one of the strand meshes.
# Visual only -- each generator writes the bale's box collision itself.

MATERIALS_URI = "model://cfr_arduino_bridge/materials"
MESH_URI = "model://cfr_arduino_bridge/meshes"
STRAND_COLOR = "0.74 0.63 0.40 1"


def bale_look(index: int) -> tuple[int, int, float]:
    """(texture, strand mesh, tint) for a bale, fixed by a number, so a
    wall does not repeat one bale and regenerating does not reshuffle it."""
    h = (index * 2654435761) & 0xFFFFFFFF
    return (
        h % VARIANTS,
        (h >> 8) % STRAND_VARIANTS,
        0.88 + 0.12 * ((h >> 16) % 100) / 99,
    )


def bale_visuals(name: str, pose: str, look: int, strands: bool = True) -> str:
    """<visual name="{name}_visual"> and, with strands, "{name}_strands".

    Seen as the rounded-edge bale and collided with as the box: the same
    size, so nothing the car can hit moves.  One line each, six-space
    indented, joined by newlines, no trailing newline.
    """
    texture, strand_mesh, tint = bale_look(look)
    straw = " ".join(f"{c * tint:.3f}" for c in STRAW_DIFFUSE)
    material = (
        f"<material><diffuse>{straw} 1</diffuse>"
        "<specular>0.04 0.04 0.03 1</specular><pbr><metal>"
        f"<albedo_map>{MATERIALS_URI}/straw_bale_{texture}.png</albedo_map>"
        f"<normal_map>{MATERIALS_URI}/straw_bale_normal.png</normal_map>"
        "<roughness>0.95</roughness><metalness>0</metalness></metal></pbr></material>"
    )
    out = (
        f'      <visual name="{name}_visual"><pose>{pose}</pose><geometry>'
        f"<mesh><uri>{MESH_URI}/straw_bale.obj</uri></mesh></geometry>{material}</visual>"
    )
    if strands:
        out += (
            f'\n      <visual name="{name}_strands"><pose>{pose}</pose><geometry><mesh>'
            f"<uri>{MESH_URI}/bale_strands_{strand_mesh}.stl</uri></mesh></geometry>"
            f"<material><diffuse>{STRAND_COLOR}</diffuse><specular>0.05 0.05 0.04 1</specular>"
            "</material><cast_shadows>false</cast_shadows></visual>"
        )
    return out


def smooth_noise(rng, h, w, scale):
    """Low-frequency noise in [0, 1], by upsampling a coarse random grid."""
    coarse = rng.random((max(2, h // scale), max(2, w // scale)))
    img = Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=float) / 255.0


def texture(rng):
    """(albedo RGB, height field) of one bale face."""
    base_mix = smooth_noise(rng, TEX_H, TEX_W, 48)
    base = GAP[None, None, :] * (0.8 + 0.4 * base_mix[..., None])
    albedo = Image.fromarray(base.clip(0, 255).astype(np.uint8), "RGB")
    height = Image.new("L", (TEX_W, TEX_H), 0)
    draw_c = ImageDraw.Draw(albedo)
    draw_h = ImageDraw.Draw(height)
    tone = smooth_noise(rng, TEX_H, TEX_W, 120)
    # Back to front: each layer of stalks sits on (and shades) the one below,
    # which is what gives a bale its depth -- dark gaps, bright top stalks.
    for layer, count in enumerate((2600, 2600, 2200)):
        lift = 0.55 + 0.2 * layer
        for _ in range(count):
            x, y = rng.uniform(-60, TEX_W + 60), rng.uniform(-20, TEX_H + 20)
            # Mostly along the bale, some crossing: the baler lays them long.
            ang = (
                rng.normal(0.0, 0.22) if rng.random() < 0.85 else rng.uniform(-1.2, 1.2)
            )
            length = rng.gamma(2.2, 34.0) + 15
            width = int(rng.choice([2, 3, 3, 4, 4, 5]))
            dx, dy = math.cos(ang) * length / 2, math.sin(ang) * length / 2
            # A little bow, so they are not ruler-straight.
            bend = rng.normal(0, 3.0)
            pts = [
                (x - dx, y - dy),
                (x + bend * math.sin(ang), y - bend * math.cos(ang)),
                (x + dx, y + dy),
            ]
            t = tone[int(np.clip(y, 0, TEX_H - 1)), int(np.clip(x, 0, TEX_W - 1))]
            pick = rng.random()
            color = GOLD if pick < 0.5 else PALE if pick < 0.75 else TAN
            color = color * (lift + 0.25 * t) * rng.uniform(0.85, 1.12)
            shade = tuple(int(c) for c in (GAP * 0.7))
            draw_c.line(
                [(p[0] + 1, p[1] + 2) for p in pts], fill=shade, width=width + 1
            )
            draw_c.line(
                pts, fill=tuple(int(c) for c in color.clip(0, 255)), width=width
            )
            # A highlight down one edge of the fatter stalks: they are tubes.
            if width >= 4:
                hi = tuple(int(c) for c in (color * 1.18).clip(0, 255))
                draw_c.line([(p[0], p[1] - 1) for p in pts], fill=hi, width=1)
            draw_h.line(
                pts, fill=int(90 + 55 * layer + rng.uniform(0, 30)), width=width
            )
    albedo = albedo.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(albedo, dtype=float)
    # Fine grain and a little sun-bleaching toward the top of the face.
    arr *= 1.0 + rng.normal(0, 0.035, arr.shape[:2])[..., None]
    fade = np.linspace(1.08, 0.94, TEX_H)[:, None, None]
    arr = arr * fade + (PALE - arr) * 0.06 * np.linspace(1, 0, TEX_H)[:, None, None]
    arr = match_photo(arr)
    return arr.clip(0, 255).astype(np.uint8), np.asarray(
        height.filter(ImageFilter.GaussianBlur(1.0)), dtype=float
    )


def match_photo(arr):
    """Scale the texture's contrast and mean saturation to the photo's."""
    mean = arr.reshape(-1, 3).mean(0)
    lum = arr.mean(axis=2, keepdims=True)
    contrast = lum.std() / lum.mean()
    arr = mean + (arr - mean) * (TARGET_CONTRAST / contrast)
    lum = arr.mean(axis=2, keepdims=True)
    m = arr.reshape(-1, 3).mean(0)
    sat = (m.max() - m.min()) / m.max()
    return lum + (arr - lum) * (TARGET_SATURATION / sat)


def normal_map(height, strength=2.5):
    """Tangent-space normal map (OpenGL convention, +y up) from a height field."""
    h = height / 255.0
    gx = np.gradient(h, axis=1) * strength * 10
    gy = np.gradient(h, axis=0) * strength * 10
    n = np.dstack([-gx, gy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


# ---------------------------------------------------------------- strands


def prism(a, b, radius, up_hint):
    """Triangles of a three-sided prism from a to b: a straw seen from any
    side, in 6 triangles, no caps."""
    axis = b - a
    axis = axis / (np.linalg.norm(axis) or 1.0)
    side = np.cross(axis, up_hint)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)
    ring = [
        side * math.cos(k) * radius + other * math.sin(k) * radius
        for k in (0, 2.094, 4.189)
    ]
    tris = []
    for k in range(3):
        p0, p1 = ring[k], ring[(k + 1) % 3]
        # Wound so the face normals point OUT of the straw: renderers cull
        # back faces, and inside-out stalks draw lit from the wrong side.
        tris.append((a + p0, b + p1, b + p0))
        tris.append((a + p0, a + p1, b + p1))
    return tris


def spike(a, b, radius, up_hint):
    """Triangles of a straw tapering from a to a point at b: three, the
    fewest that look solid from every side.  The open base is inside the
    bale (or on the ground), where nothing sees it."""
    axis = b - a
    axis = axis / (np.linalg.norm(axis) or 1.0)
    side = np.cross(axis, up_hint)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)
    ring = [
        a + side * math.cos(k) * radius + other * math.sin(k) * radius
        for k in (0, 2.094, 4.189)
    ]
    return [(ring[k], ring[(k + 1) % 3], b) for k in range(3)]


def strand(rng, root, out, droop, length, radius):
    """A spike, or for a long stalk a tube that bends under its weight into
    a spike: 3 or 9 triangles."""
    up = np.array([0.0, 0.0, 1.0])
    if length < 0.07:
        return spike(root, root + out * length, radius, up)
    mid = root + out * length * 0.55
    bent = out * 0.45 + np.array([0, 0, -droop])
    bent /= np.linalg.norm(bent)
    return prism(root, mid, radius, up) + spike(
        mid, mid + bent * length * 0.45, radius * 0.8, up
    )


def unit(v):
    return v / np.linalg.norm(v)


def strands(rng):
    """Triangles of the loose straw around one bale, in the bale's frame
    (x along it, z up, origin at its center).

    Most of it is FUZZ: short stalks leaving the face at a glancing angle,
    dense enough that the face stops reading as a flat box.  Then tufts along
    the top edges, which are what make a real bale's outline ragged, a few
    long drooping strays, and litter on the ground at the foot of the wall.
    """
    lx, ly, lz = LENGTH / 2, WIDTH / 2, HEIGHT / 2
    up = np.array([0.0, 0.0, 1.0])
    tris = []

    def radius():
        return rng.uniform(0.0011, 0.0019)

    def long_length():
        return float(min(0.16, rng.gamma(2.0, 0.03) + 0.05))

    for sign in (-1, 1):
        normal = np.array([0.0, sign, 0.0])
        # Fuzz on the long faces, where the car looks.
        for _ in range(180):
            x, z = rng.uniform(-lx, lx), rng.uniform(-lz + 0.01, lz - 0.005)
            along = unit(np.array([rng.choice([-1.0, 1.0]), 0.0, rng.normal(0, 0.5)]))
            out = unit(normal * rng.uniform(0.15, 0.6) + along)
            root = np.array([x, sign * (ly - 0.003), z])
            tris += spike(
                root, root + out * rng.uniform(0.02, 0.06), radius() * 1.3, up
            )
        # A few long strays that droop.
        for _ in range(18):
            x, z = rng.uniform(-lx, lx), rng.uniform(-lz + 0.03, lz)
            along = np.array([rng.choice([-1.0, 1.0]), 0.0, rng.normal(0, 0.4)])
            out = unit(
                normal * rng.uniform(0.3, 0.9) + unit(along) * rng.uniform(0.4, 1.0)
            )
            root = np.array([x, sign * (ly - 0.004), z])
            tris += strand(
                rng, root, out, rng.uniform(0.1, 0.8), long_length(), radius()
            )
        # Tufts along the top edge: clumps of stalks fanning up and over.
        for _ in range(11):
            cx = rng.uniform(-lx + 0.03, lx - 0.03)
            for _ in range(int(rng.integers(4, 8))):
                root = np.array(
                    [
                        cx + rng.normal(0, 0.03),
                        sign * (ly - rng.uniform(0.0, 0.025)),
                        lz - rng.uniform(0.0, 0.02),
                    ]
                )
                out = unit(
                    np.array(
                        [
                            rng.normal(0, 0.7),
                            sign * rng.uniform(0.3, 1.0),
                            rng.uniform(0.1, 1.0),
                        ]
                    )
                )
                length = rng.uniform(0.03, 0.11)
                tris += strand(rng, root, out, rng.uniform(0.3, 1.2), length, radius())
        # Litter on the ground at the foot of the wall.
        for _ in range(24):
            x = rng.uniform(-lx - 0.1, lx + 0.1)
            y = sign * (ly + abs(rng.normal(0.0, 0.1)))
            z = -lz + rng.uniform(0.002, 0.008)
            ang = rng.normal(0.0, 0.5) + (math.pi if rng.random() < 0.5 else 0.0)
            d = np.array([math.cos(ang), math.sin(ang), rng.uniform(-0.02, 0.05)])
            root = np.array([x, y, z])
            tris += spike(
                root, root + unit(d) * rng.uniform(0.04, 0.14), radius() * 1.3, up
            )
    # Fuzz and a few risers on the top.
    for _ in range(45):
        root = np.array([rng.uniform(-lx, lx), rng.uniform(-ly, ly), lz - 0.003])
        out = unit(
            np.array([rng.normal(0, 1), rng.normal(0, 0.6), rng.uniform(0.1, 0.7)])
        )
        tris += spike(root, root + out * rng.uniform(0.02, 0.07), radius() * 1.3, up)
    # The cut ends.
    for sign in (-1, 1):
        for _ in range(22):
            root = np.array(
                [sign * (lx - 0.003), rng.uniform(-ly, ly), rng.uniform(-lz + 0.01, lz)]
            )
            out = unit(
                np.array(
                    [
                        sign * rng.uniform(0.3, 1.0),
                        rng.normal(0, 0.6),
                        rng.normal(0, 0.5),
                    ]
                )
            )
            tris += spike(
                root, root + out * rng.uniform(0.02, 0.07), radius() * 1.3, up
            )
    return tris


# ---------------------------------------------------------------- the bale

# Rounding of the top and vertical edges.  The bottom stays square: it sits
# on the ground.  5 cm rounds the top edge back by only 6 mm at 0.33 m, the
# top of formulaTwo's depth band, so the policy's scan barely sees it.
BALE_RADIUS = 0.05
BALE_BAND_STEPS = 4  # segments per rounded edge band, on each face


def _axis_coords(half, rounded_lo, rounded_hi, r, n):
    """Grid coordinates along one face axis, dense inside the rounding bands
    so the curve is smooth: equal angles, which on a face run 45 deg -> 0."""
    band = [r - r * math.tan(math.radians(45.0 * (1 - k / n))) for k in range(n + 1)]
    lo = [-half + d for d in band] if rounded_lo else [-half]
    hi = [half - d for d in reversed(band)] if rounded_hi else [half]
    return lo + hi


def rounded_bale(r=BALE_RADIUS, n=BALE_BAND_STEPS):
    """(vertices, uvs, normals, faces) of the bale, faces as index triples.

    Each point of the box's surface is pulled onto a box of the same size
    with rounded edges: clamp it into the inner box (shrunk by r everywhere
    except at the bottom), then push it out r along the difference.  Faces
    are separate grids, each with its own [0, 1] texture coordinates, so the
    straw texture sits on every face as it did on the box.
    """
    hx, hy, hz = LENGTH / 2, WIDTH / 2, HEIGHT / 2
    lo = np.array([-hx + r, -hy + r, -hz])
    hi = np.array([hx - r, hy - r, hz - r])
    verts, uvs, norms, faces = [], [], [], []

    def face(axis, sign, u_axis, v_axis, u_coords, v_coords):
        base = len(verts)
        normal = np.zeros(3)
        normal[axis] = sign
        half = np.array([hx, hy, hz])
        for v in v_coords:
            for u in u_coords:
                p = np.zeros(3)
                p[axis] = sign * half[axis]
                p[u_axis], p[v_axis] = u, v
                q = np.clip(p, lo, hi)
                d = p - q
                length = np.linalg.norm(d)
                nrm = d / length if length > 1e-9 else normal
                verts.append(q + nrm * r if length > 1e-9 else p)
                norms.append(nrm)
                uvs.append(
                    (
                        (u + half[u_axis]) / (2 * half[u_axis]),
                        (v + half[v_axis]) / (2 * half[v_axis]),
                    )
                )
        nu = len(u_coords)
        for j in range(len(v_coords) - 1):
            for i in range(nu - 1):
                a, b = base + j * nu + i, base + j * nu + i + 1
                c, d = base + (j + 1) * nu + i + 1, base + (j + 1) * nu + i
                for tri in ((a, b, c), (a, c, d)):
                    p0, p1, p2 = (verts[k] for k in tri)
                    # Wound outward, whatever the face's axis order.
                    if np.dot(np.cross(p1 - p0, p2 - p0), normal) < 0:
                        tri = (tri[0], tri[2], tri[1])
                    faces.append(tri)

    xs = _axis_coords(hx, True, True, r, n)
    ys = _axis_coords(hy, True, True, r, n)
    zs = _axis_coords(hz, False, True, r, n)
    face(2, 1, 0, 1, xs, ys)  # top: u along the bale, v across it
    for sign in (-1, 1):
        face(1, sign, 0, 2, xs, zs)  # long sides: u along, v up
        face(0, sign, 1, 2, ys, zs)  # ends: u across, v up
    return np.array(verts), np.array(uvs), np.array(norms), faces


def write_obj(path, verts, uvs, norms, faces):
    with open(path, "w", newline="\n") as f:
        f.write("# cfr straw bale: 36 x 18 x 14 in, rounded top and vertical edges\n")
        f.write("# generated by jetson/scripts/generate_straw_assets.py\n")
        f.write("o straw_bale\n")
        for v in verts:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in uvs:
            f.write(f"vt {t[0]:.5f} {t[1]:.5f}\n")
        for nrm in norms:
            f.write(f"vn {nrm[0]:.4f} {nrm[1]:.4f} {nrm[2]:.4f}\n")
        for a, b, c in faces:
            f.write(
                f"f {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1} {c + 1}/{c + 1}/{c + 1}\n"
            )


def write_stl(path, tris):
    """Binary STL, with face normals."""
    with open(path, "wb") as f:
        f.write(b"cfr straw bale strands".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    heights = []
    for k in range(VARIANTS):
        albedo, height = texture(rng)
        path = PACKAGE / "materials" / f"straw_bale_{k}.png"
        Image.fromarray(albedo, "RGB").save(path, optimize=True)
        heights.append(height)
        print(f"wrote {path.relative_to(PACKAGE.parent.parent)}")
    path = PACKAGE / "materials" / "straw_bale_normal.png"
    Image.fromarray(normal_map(heights[0]), "RGB").save(path, optimize=True)
    print(f"wrote {path.relative_to(PACKAGE.parent.parent)}")

    path = PACKAGE / "meshes" / "straw_bale.obj"
    verts, uvs, norms, faces = rounded_bale()
    write_obj(path, verts, uvs, norms, faces)
    print(f"wrote {path.relative_to(PACKAGE.parent.parent)}  ({len(faces)} triangles)")

    for k in range(STRAND_VARIANTS):
        tris = strands(rng)
        path = PACKAGE / "meshes" / f"bale_strands_{k}.stl"
        write_stl(path, tris)
        print(
            f"wrote {path.relative_to(PACKAGE.parent.parent)}  ({len(tris)} triangles)"
        )


if __name__ == "__main__":
    main()
