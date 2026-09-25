"""Render a top-down map of the Obstacle Course with the named region bounds
overlaid, for review. Usage: render_map.py [out.png]"""

import sys
import math
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regions as R
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle, Polygon

c = R.course
root = ET.parse(
    R.REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/obstacle_course.sdf"
).getroot()


def pose(e):
    p = e.find("pose")
    if p is None or not p.text:
        return (0, 0, 0, 0)
    v = [float(t) for t in p.text.split()]
    return (v[0], v[1], v[2], v[5])


def compose(a, b):
    x, y, z, t = a
    bx, by, bz, bt = b
    return (
        x + bx * math.cos(t) - by * math.sin(t),
        y + bx * math.sin(t) + by * math.cos(t),
        z + bz,
        t + bt,
    )


items = []  # (kind, pose, dims, color, z)
for m in root.iter("model"):
    mp = pose(m)
    for link in m.findall("link"):
        lp = compose(mp, pose(link))
        for v in link.findall("visual"):
            g = v.find("geometry")
            if g is None:
                continue
            vp = compose(lp, pose(v))
            d = v.find("material/diffuse")
            col = (
                tuple(float(t) for t in d.text.split()[:3])
                if d is not None
                else (0.6, 0.6, 0.6)
            )
            b = g.find("box/size")
            cy = g.find("cylinder")
            if b is not None:
                sx, sy, sz = map(float, b.text.split())
                items.append(("box", vp, (sx, sy), col, vp[2] + sz / 2))
            elif cy is not None:
                items.append(("cyl", vp, (float(cy.find("radius").text),), col, vp[2]))
items.sort(key=lambda i: i[4])


def draw_course(ax):
    ax.add_patch(
        Rectangle((0.005 - 15, -4.54 - 12), 30, 24, fc="#d8d8d8", ec="none", zorder=0)
    )
    for k, (x, y, z, t), dims, col, zz in items:
        if k == "box":
            sx, sy = dims
            if sx > 20 or sy > 20:
                continue
            cs, sn = math.cos(t), math.sin(t)
            pts = [
                (x + px * cs - py * sn, y + px * sn + py * cs)
                for px, py in [
                    (-sx / 2, -sy / 2),
                    (sx / 2, -sy / 2),
                    (sx / 2, sy / 2),
                    (-sx / 2, sy / 2),
                ]
            ]
            ax.add_patch(Polygon(pts, fc=col, ec="k", lw=0.3, zorder=1 + zz))
        else:
            ax.add_patch(Circle((x, y), dims[0], fc=col, ec="k", lw=0.3, zorder=1 + zz))
    cx, cy = c.to_world(*c.HELIX_CENTRE)
    s = math.degrees(math.radians(c.HELIX_START_DEG) + math.pi)
    ax.add_patch(
        Wedge(
            (cx, cy),
            c.HELIX_OUTER_R * c.FOOT,
            s,
            s + c.HELIX_SWEEP_DEG,
            width=(c.HELIX_OUTER_R - c.HELIX_INNER_R) * c.FOOT,
            fc="#9aa",
            ec="k",
            lw=0.5,
            zorder=1,
        )
    )
    ax.set_aspect("equal")
    ax.set_xlim(-10.5, 10.8)
    ax.set_ylim(-11.8, 3.5)
    ax.grid(alpha=0.25)


fig, axes = plt.subplots(2, 1, figsize=(14, 21))
draw_course(axes[0])
axes[0].set_title(
    "Course as built (sdf visuals, top-down; helix wedge from generator constants)"
)
draw_course(axes[1])
ax = axes[1]
cols = plt.cm.tab20.colors
lab = {
    "tunnel": (7.1, 1.45),
    "overpass_ramp": (5.4, -0.95),
    "helical_ramp": (9.2, 3.05),
    "hoops": (-7.5, 1.0),
    "narrow_region": (7.15, -4.0),
    "wide_open_region": (4.6, -2.6),
}
for r in R._build_regions():
    col = cols[(r.order - 1) * 2 % 20]
    if r.name == "helical_ramp":
        cx, cy = c.to_world(*c.HELIX_CENTRE)
        s = math.degrees(math.radians(c.HELIX_START_DEG) + math.pi)
        ax.add_patch(
            Wedge(
                (cx, cy),
                c.HELIX_OUTER_R * c.FOOT,
                s,
                s + c.HELIX_SWEEP_DEG,
                width=(c.HELIX_OUTER_R - c.HELIX_INNER_R) * c.FOOT,
                fc=col,
                ec=col,
                lw=2.5,
                alpha=0.4,
                zorder=100,
            )
        )
    elif r.name in R.POLYGONS:
        ax.add_patch(
            Polygon(R.POLYGONS[r.name], fc=col, ec=col, lw=2.5, alpha=0.4, zorder=100)
        )
    elif r.bounds:
        x0, y0, x1, y1 = r.bounds
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fc=col,
                ec=col,
                lw=2.5,
                alpha=0.4,
                zorder=100,
            )
        )
    else:
        continue
    if r.bounds:
        x0, y0, x1, y1 = r.bounds
        tx, ty = lab.get(r.name, ((x0 + x1) / 2, y1 + 0.25))
    ax.annotate(
        f"{r.order}. {r.name}",
        (tx, ty),
        ha="center",
        fontsize=10,
        weight="bold",
        bbox=dict(fc="w", ec="none", alpha=0.85, pad=1),
        zorder=200,
    )
ax.text(
    0.01,
    0.01,
    "hoops, narrow_region and wide_open_region are bale-derived outlines (open ends follow a hand-painted guide)",
    transform=ax.transAxes,
    fontsize=9,
    style="italic",
    zorder=200,
)
ax.set_title("Same view with named section bounds overlaid")
for a in axes:
    a.set_xlabel("world x (m)")
    a.set_ylabel("world y (m)")
fig.tight_layout()
fig.savefig(
    sys.argv[1] if len(sys.argv) > 1 else "obstacle_course_regions_review.png", dpi=90
)
