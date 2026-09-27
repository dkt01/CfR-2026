// formulaTwo's depth -> virtual LiDAR, in the browser: the same pipeline as
// rl/formulaTwo/perception.py, line for line, so the viewer can show what the
// f2_v3 policy is given from Gazebo's depth image.
//
//   1. sampleDepth   every 10th column / 5th row of the 640x360 image, at the
//                    pixel nearest each canonical ray      (Camera.sample_depth)
//   2. depthToScan   back-project with the mount, level by roll/pitch, keep
//                    points 0.07-0.33 m above the ground, nearest per column
//                                                          (Camera.depth_to_scan)
//   3. encode        log range in [0, 1], invalid -0.25    (Camera.encode)
//
// In training, step 1 is `render` + `corrupt` on the same grid; steps 2-3 are
// shared.  So the grid drawn here is exactly the shape of what training
// rendered, and the scan is exactly what the policy's input is built from.
//
// The numbers come from the policy's own config.yaml, not copies of them.
// scripts/check_policy_depth.py compares this file against perception.py.

import f2Config from "../../../rl/formulaTwo/bestModel/f2_v3_59M/config.yaml?raw";

export const POLICY_NAME = "f2_v3_59M";
export const INVALID = -0.25;
const R0 = 0.25;

// Just enough YAML for the flat `camera:` block: `key: number` and
// `key: [a, b]`, comments stripped.
export function cameraBlock(text) {
  const out = {};
  let inside = false;
  for (const raw of text.split("\n")) {
    const line = raw.replace(/#.*$/, "").trimEnd();
    if (!line.trim()) {
      continue;
    }
    if (/^\S/.test(line)) {
      inside = line.startsWith("camera:");
      continue;
    }
    if (!inside) {
      continue;
    }
    const match = line.match(/^\s+(\w+):\s*(.+)$/);
    if (!match) {
      continue;
    }
    const value = match[2].trim();
    out[match[1]] = value.startsWith("[")
      ? value.slice(1, -1).split(",").map(Number)
      : Number(value);
  }
  return out;
}

// numpy.round: halves go to the even neighbor.
function rint(x) {
  const r = Math.round(x);
  return Math.abs(x - Math.trunc(x)) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function makeCamera(c = cameraBlock(f2Config)) {
  const wi = c.image_width;
  const hi = c.image_height;
  const cs = c.col_stride;
  const rs = c.row_stride;
  const W = Math.floor(wi / cs);
  const H = Math.floor(hi / rs);
  const f = wi / 2 / Math.tan((c.hfov_deg * Math.PI) / 360);
  const a = new Float64Array(W);
  for (let i = 0; i < W; i += 1) {
    a[i] = (wi / 2 - (cs * i + Math.floor(cs / 2) + 0.5)) / f;
  }
  const b = [];
  for (let j = 0; j < H; j += 1) {
    const value = (hi / 2 - (rs * j + Math.floor(rs / 2) + 0.5)) / f;
    if (Math.abs(value) <= c.row_slope_limit) {
      b.push(value);
    }
  }
  return {
    W,
    rows: b.length,
    a,
    b: Float64Array.from(b),
    azimuth: a.map(Math.atan),
    x: c.x,
    z: c.z,
    minRange: c.min_range,
    maxRange: c.max_range,
    band: c.band,
    scanMax: c.scan_max,
    imageWidth: wi,
    imageHeight: hi,
    hfovDeg: c.hfov_deg,
  };
}

// The pixel each canonical ray reads, for the overlay: (rows x W) u, v and
// whether it is inside the image.
export function gridPixels(cam, fx, fy, cx, cy, w0, h0) {
  const u = Array.from(cam.a, (a) => rint(cx - a * fx));
  const v = Array.from(cam.b, (b) => rint(cy - b * fy));
  return { u, v, okU: u.map((x) => x >= 0 && x < w0), okV: v.map((y) => y >= 0 && y < h0) };
}

// (rows x W) depth on the grid, NaN where there is none.
export function sampleDepth(cam, depth, w0, h0, fx, fy, cx, cy) {
  const { u, v, okU, okV } = gridPixels(cam, fx, fy, cx, cy, w0, h0);
  const out = new Float64Array(cam.rows * cam.W);
  for (let r = 0; r < cam.rows; r += 1) {
    for (let col = 0; col < cam.W; col += 1) {
      let z = NaN;
      if (okU[col] && okV[r]) {
        z = depth[v[r] * w0 + u[col]];
        if (!Number.isFinite(z) || z < cam.minRange || z > cam.maxRange) {
          z = NaN;
        }
      }
      out[r * cam.W + col] = z;
    }
  }
  return out;
}

// Nearest in-band horizontal range per column (Infinity: none), which grid
// cells were in the band, and which row each column's beam came from (-1).
// pitch + is nose down, roll + is left side up (ROS), as perception.py.
export function depthToScan(cam, grid, height, pitch, roll) {
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const scan = new Float64Array(cam.W).fill(Infinity);
  const beamRow = new Int32Array(cam.W).fill(-1);
  const keep = new Uint8Array(cam.rows * cam.W);
  for (let r = 0; r < cam.rows; r += 1) {
    const bb = cam.b[r];
    for (let col = 0; col < cam.W; col += 1) {
      const depth = grid[r * cam.W + col];
      if (!Number.isFinite(depth)) {
        continue;
      }
      const a = cam.a[col];
      const y1 = a * cr - bb * sr;
      const z1 = a * sr + bb * cr;
      const x2 = cp + z1 * sp;
      const z2 = -sp + z1 * cp;
      const range = depth * Math.sqrt(x2 * x2 + y1 * y1);
      const z = height + depth * z2;
      if (z >= cam.band[0] && z <= cam.band[1]) {
        keep[r * cam.W + col] = 1;
        if (range < scan[col]) {
          scan[col] = range;
          beamRow[col] = r;
        }
      }
    }
  }
  return { scan, keep, beamRow };
}

export function encode(cam, scan) {
  const out = new Float64Array(scan.length);
  const span = Math.log(cam.scanMax / R0);
  for (let i = 0; i < scan.length; i += 1) {
    out[i] = Number.isFinite(scan[i])
      ? Math.min(1, Math.max(0, Math.log(Math.max(scan[i], R0) / R0) / span))
      : INVALID;
  }
  return out;
}

// Roll and pitch of a quaternion, formula_two_node.roll_pitch_of.
export function rollPitch(q) {
  const roll = Math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y));
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (q.w * q.y - q.z * q.x))));
  return { roll, pitch };
}
