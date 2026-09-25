#!/usr/bin/env python3
"""Derive trusted outlines for hoops / narrow_region / wide_open_region.

The hand-painted outlines in references/region_guides.json are only a guide
to WHICH free space each region means. The trusted edges come from the hay
bale geometry in the obstacle course SDF: each guide is grown by GROW meters,
the bale footprints (and the exact-rect regions) are cut out, and the free-space component that contains
the guide's core is kept. Bale faces therefore define the edges wherever a
wall exists; the guide only closes the open ends (where a region joins its
neighbor). Writes references/region_polygons.json.

Usage: derive_polygons.py
"""

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regions as R  # noqa: E402  (for REPO_ROOT)

SKILL = Path(__file__).resolve().parents[1]
SDF = R.REPO_ROOT / "jetson/cfr_arduino_bridge/worlds/obstacle_course.sdf"
RES = 0.02  # m per cell
X0, Y0, X1, Y1 = -11.0, -12.5, 11.5, 4.0
GROW = 0.5  # m the guide is grown to reach bale faces
CORE = 0.35  # m the guide is shrunk to seed the flood fill
EPS = 0.04  # m polygon simplification


def px(x, y):
    return int(round((x - X0) / RES)), int(round((Y1 - y) / RES))


def unpx(c, r):
    return X0 + c * RES, Y1 - r * RES


def bale_polys():
    root = ET.parse(SDF).getroot()
    out = []
    for m in root.iter("model"):
        name = m.get("name", "")
        if name != "course_bales" and not name.startswith("gap_bale"):
            continue
        mp = [float(t) for t in (m.findtext("pose") or "0 0 0 0 0 0").split()]
        for col in m.iter("collision"):
            size = col.find("geometry/box/size")
            if size is None:
                continue
            sx, sy, _ = map(float, size.text.split())
            p = [float(t) for t in (col.findtext("pose") or "0 0 0 0 0 0").split()]
            t = p[5] + mp[5]
            x = mp[0] + p[0] * math.cos(mp[5]) - p[1] * math.sin(mp[5])
            y = mp[1] + p[0] * math.sin(mp[5]) + p[1] * math.cos(mp[5])
            cs, sn = math.cos(t), math.sin(t)
            out.append(
                [
                    (x + a * cs - b * sn, y + a * sn + b * cs)
                    for a, b in [
                        (-sx / 2, -sy / 2),
                        (sx / 2, -sy / 2),
                        (sx / 2, sy / 2),
                        (-sx / 2, sy / 2),
                    ]
                ]
            )
    return out


def fill(polys, shape):
    img = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(img, [np.array([px(*q) for q in p], np.int32)], 1)
    return img


def kernel(m):
    k = int(round(m / RES)) * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def main():
    shape = (int(round((Y1 - Y0) / RES)), int(round((X1 - X0) / RES)))
    bales = fill(bale_polys(), shape)
    # Regions with exact rectangles are carved out so the derived outlines
    # never overlap them.
    for r in R.REGIONS:
        if (
            r.name in R.POLYGONS
            or r.name in ("helical_ramp", "overpass_ramp")
            or not r.bounds
        ):
            continue
        x0, y0, x1, y1 = r.bounds
        bales[
            int((Y1 - y1) / RES) : int((Y1 - y0) / RES) + 1,
            int((x0 - X0) / RES) : int((x1 - X0) / RES) + 1,
        ] = 1
    guides = json.loads((SKILL / "references/region_guides.json").read_text())
    out = {
        "_comment": "Outlines derived by scripts/derive_polygons.py from the hay bale "
        "footprints in the obstacle course SDF, using region_guides.json (hand-painted) "
        "only to pick and close each region. World meters (x, y)."
    }
    for name, pts in guides.items():
        if name.startswith("_"):
            continue
        guide = fill([pts], shape)
        cand = cv2.dilate(guide, kernel(GROW)) & (1 - bales)
        core = cv2.erode(guide, kernel(CORE)) & cand
        n, lab = cv2.connectedComponents(cand)
        keep = np.unique(lab[core > 0])
        keep = keep[keep > 0]
        region = np.isin(lab, keep).astype(np.uint8)
        # Take only what the guide roughly covers plus the growth band, then
        # smooth pixel noise.
        region = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel(0.05))
        cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, EPS / RES, True)[:, 0, :]
        out[name] = [[round(v, 3) for v in unpx(cc, rr)] for cc, rr in ap]
        print(
            name,
            len(out[name]),
            "pts, area %.2f m2" % (cv2.contourArea(c) * RES * RES),
            "guide area %.2f m2" % (guide.sum() * RES * RES),
        )
    with open(SKILL / "references/region_polygons.json", "w", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
