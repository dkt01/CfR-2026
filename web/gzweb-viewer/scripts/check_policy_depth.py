#!/usr/bin/env python3
"""Check src/policy-depth.js against rl/formulaTwo/perception.py.

    python3 web/gzweb-viewer/scripts/check_policy_depth.py

Renders random depth frames (a floor, bale walls at random ranges, dropouts),
with off-integer intrinsics and a random roll and pitch, runs both pipelines
on them and compares the sampled grid, the scan and its encoding.  Needs
numpy, pyyaml and node.  Exit status 1 on any mismatch.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
VIEWER = HERE.parent
REPO = VIEWER.parents[1]
POLICY = REPO / "rl" / "formulaTwo" / "bestModel" / "f2_v3_59M" / "config.yaml"
sys.path.insert(0, str(REPO / "rl" / "formulaTwo"))
import perception  # noqa: E402

# `?raw` is a Vite import; node gets a hook that does the same thing.
HOOKS = """
import { readFileSync } from "node:fs";
export async function load(url, context, next) {
  if (url.endsWith("?raw")) {
    const text = readFileSync(new URL(url.slice(0, -4)), "utf8");
    return { format: "module", shortCircuit: true, source: `export default ${JSON.stringify(text)};` };
  }
  return next(url, context);
}
"""
REGISTER = 'import { register } from "node:module"; register("./hooks.mjs", import.meta.url);\n'
RUN = """
import { readFileSync } from "node:fs";
import * as P from "%s";
const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const cam = P.makeCamera();
const out = cases.map((c) => {
  const depth = Float32Array.from(c.depth, (v) => (v === null ? NaN : v));
  const grid = P.sampleDepth(cam, depth, c.w, c.h, c.fx, c.fy, c.cx, c.cy);
  const { scan } = P.depthToScan(cam, grid, cam.z, c.pitch, c.roll);
  const nan = (a) => Array.from(a, (v) => (Number.isFinite(v) ? v : null));
  return { grid: nan(grid), scan: nan(scan), encoded: Array.from(P.encode(cam, scan)), W: cam.W, rows: cam.rows };
});
process.stdout.write(JSON.stringify(out));
"""


def frame(rng, w=640, h=360):
    fx = fy = (w / 2) / math.tan(math.radians(55)) * rng.uniform(0.97, 1.03)
    cx, cy = w / 2 - 0.5 + rng.uniform(-3, 3), h / 2 - 0.5 + rng.uniform(-3, 3)
    a = (cx - np.arange(w))[None, :] / fx
    b = (cy - np.arange(h))[:, None] / fy
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(b < 0, 0.2 / -b, np.inf) * np.ones((h, w))
        for _ in range(4):
            r = rng.uniform(0.4, 12.0)
            lo, hi = sorted(rng.uniform(-1.4, 1.4, 2))
            face = (
                (a * r >= lo)
                & (a * r <= hi)
                & (0.2 + b * r >= 0)
                & (0.2 + b * r <= 0.356)
            )
            z = np.where(face, np.minimum(z, r), z)
    z[rng.random(z.shape) < 0.05] = np.nan
    z[~np.isfinite(z)] = np.nan
    return z.astype(np.float32), fx, fy, cx, cy


def main():
    cam = perception.Camera(yaml.safe_load(POLICY.read_text()))
    rng = np.random.default_rng(7)
    cases, expected = [], []
    for _ in range(6):
        depth, fx, fy, cx, cy = frame(rng)
        roll, pitch = rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05)
        grid = cam.sample_depth(depth.astype(float), fx, fy, cx, cy)
        scan = cam.depth_to_scan(grid, None, np.array([pitch]), np.array([roll]))
        expected.append((grid[0], scan[0], cam.encode(scan)[0]))
        cases.append(
            {
                "depth": [
                    None if not np.isfinite(v) else float(v) for v in depth.ravel()
                ],
                "w": depth.shape[1],
                "h": depth.shape[0],
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "roll": roll,
                "pitch": pitch,
            }
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "hooks.mjs").write_text(HOOKS)
        (tmp / "register.mjs").write_text(REGISTER)
        (tmp / "run.mjs").write_text(
            RUN % (VIEWER / "src" / "policy-depth.js").as_uri()
        )
        (tmp / "cases.json").write_text(json.dumps(cases))
        result = subprocess.run(
            ["node", "--import", "./register.mjs", "run.mjs", "cases.json"],
            cwd=tmp,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode:
        print(result.stderr)
        return 1
    ok = True
    for n, (got, (grid, scan, enc)) in enumerate(
        zip(json.loads(result.stdout), expected)
    ):
        if (got["rows"], got["W"]) != grid.shape:
            print(f"case {n}: grid {got['rows']}x{got['W']} vs {grid.shape}")
            ok = False
            continue
        jg = np.array([np.nan if v is None else v for v in got["grid"]]).reshape(
            grid.shape
        )
        js = np.array([np.inf if v is None else v for v in got["scan"]])
        je = np.array(got["encoded"])
        same_nan = np.array_equal(np.isnan(jg), np.isnan(grid))
        dg = np.nanmax(np.abs(jg - grid)) if np.isfinite(grid).any() else 0.0
        same_inf = np.array_equal(np.isinf(js), np.isinf(scan))
        fin = np.isfinite(scan)
        ds = np.max(np.abs(js[fin] - scan[fin])) if fin.any() else 0.0
        de = np.max(np.abs(je - enc))
        good = same_nan and same_inf and dg < 1e-5 and ds < 1e-6 and de < 1e-9
        ok &= good
        print(
            f"case {n}: {'ok  ' if good else 'FAIL'} grid {grid.shape[0]}x{grid.shape[1]} "
            f"|dgrid| {dg:.1e}  beams {int(fin.sum())}/64 |dscan| {ds:.1e}  |denc| {de:.1e}"
        )
    print("policy-depth.js matches perception.py" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
