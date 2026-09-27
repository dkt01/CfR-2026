// The simulated ZED depth image as the formulaTwo policy (f2_v3_59M) sees it.
//
// Left: Gazebo's depth image, or (Policy grid) only the 64-column grid the
// policy samples from it -- the grid training renders.  Over either: every
// sampled pixel (gray), the ones inside the 0.07-0.33 m height band (green),
// and each column's nearest in-band pixel, which is its beam (orange ring).
// Right: the 64-beam virtual LiDAR from above, forward up, colored by the
// encoded range the policy is fed; red stubs are invalid columns.
//
// The pipeline is policy-depth.js, which matches perception.py exactly (see
// scripts/check_policy_depth.py).  Leveled on the car's roll and pitch from
// Gazebo's pose stream -- ground truth, as the node's IMU is in simulation.
// Gazebo's image is clean: training's depth noise, dropout and blobs are not
// added here, so this is the policy's input at zero randomization.

import {
  INVALID,
  POLICY_NAME,
  depthToScan,
  encode,
  gridPixels,
  makeCamera,
  rollPitch,
  sampleDepth,
} from "./policy-depth.js";

const R_FLOAT32 = 13; // gz.msgs.PixelFormatType
const L_INT16 = 2;
const NO_DEPTH = [17, 25, 29];
const SCAN_W = 480;
const SCAN_H = 300;

// Turbo (Mikhailov 2019), polynomial fit; t = 0 blue ... 1 red.
function turbo(t) {
  const x = Math.min(1, Math.max(0, t));
  const r = 0.1357 + x * (4.5974 - x * (42.3277 - x * (130.5887 - x * (150.5666 - x * 58.1375))));
  const g = 0.0914 + x * (2.1856 + x * (4.8052 - x * (14.0195 - x * (4.2109 + x * 2.7747))));
  const b = 0.1067 + x * (12.5925 - x * (60.1097 - x * (109.0745 - x * (88.5066 - x * 26.8183))));
  return [r, g, b].map((v) => Math.round(255 * Math.min(1, Math.max(0, v))));
}
// Indexed by the policy's own encoding (log range), near red, far blue.
const LUT = Array.from({ length: 256 }, (_, i) => turbo(1 - i / 255));

export function createDepthView(root) {
  const cam = makeCamera();
  const imageCanvas = root.querySelector("#depth-image");
  const scanCanvas = root.querySelector("#depth-scan");
  const stats = root.querySelector("#depth-stats");
  const hint = root.querySelector("#depth-hint");
  const modeButton = root.querySelector("#depth-mode");
  const imageCtx = imageCanvas.getContext("2d");
  const scanCtx = scanCanvas.getContext("2d");
  scanCanvas.width = SCAN_W;
  scanCanvas.height = SCAN_H;
  let info = null;
  let gridOnly = false;
  let last = null;
  let frames = 0;
  let rateStart = performance.now();
  let rateFrames = 0;
  let fps = 0;

  modeButton.addEventListener("click", () => {
    gridOnly = !gridOnly;
    modeButton.textContent = gridOnly ? "Full image" : "Policy grid";
    if (last) {
      drawImage(last);
    }
  });

  // encode() for one value, inline: this runs for every pixel of every frame.
  const logSpan = Math.log(cam.scanMax / 0.25);
  function color(depth) {
    if (!Number.isFinite(depth) || depth <= 0) {
      return NO_DEPTH;
    }
    const e = Math.min(1, Math.max(0, Math.log(Math.max(depth, 0.25) / 0.25) / logSpan));
    return LUT[Math.round(e * 255)];
  }

  function setInfo(message) {
    const k = message?.intrinsics?.k;
    if (k && k[0] > 0) {
      info = { fx: k[0], fy: k[4], cx: k[2], cy: k[5], w: message.width, h: message.height };
    }
  }

  // Gazebo image -> row-major meters (NaN: none), width, height.
  function decode(message) {
    const { width: w, height: h, step } = message;
    const bytes = message.data;
    const depth = new Float32Array(w * h);
    if (message.pixel_format_type === R_FLOAT32) {
      // A fresh copy: the payload's offset inside the frame is rarely
      // 4-aligned, and slice() on a Buffer is a view, not a copy.
      const rows = new Float32Array(new Uint8Array(bytes.subarray(0, step * h)).buffer);
      const stride = step / 4;
      for (let y = 0; y < h; y += 1) {
        depth.set(rows.subarray(y * stride, y * stride + w), y * w);
      }
    } else if (message.pixel_format_type === L_INT16) {
      const rows = new Uint16Array(new Uint8Array(bytes.subarray(0, step * h)).buffer);
      const stride = step / 2;
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
          const mm = rows[y * stride + x];
          depth[y * w + x] = mm > 0 ? mm / 1000 : NaN;
        }
      }
    } else {
      return null;
    }
    return { depth, w, h };
  }

  function intrinsicsFor(w, h) {
    if (info) {
      let { fx, fy, cx, cy } = info;
      // Scaled to the image as formula_two_node does.
      if ((info.w !== w || info.h !== h) && info.w && info.h) {
        const sx = w / info.w;
        const sy = h / info.h;
        fx *= sx;
        fy *= sy;
        cx = (cx + 0.5) * sx - 0.5;
        cy = (cy + 0.5) * sy - 0.5;
      }
      return { fx, fy, cx, cy, nominal: false };
    }
    const f = w / 2 / Math.tan((cam.hfovDeg * Math.PI) / 360);
    return { fx: f, fy: f, cx: w / 2 - 0.5, cy: h / 2 - 0.5, nominal: true };
  }

  function drawImage(state) {
    const { depth, w, h, k, grid, keep, beamRow } = state;
    if (imageCanvas.width !== w || imageCanvas.height !== h) {
      imageCanvas.width = w;
      imageCanvas.height = h;
    }
    const { u, v, okU, okV } = gridPixels(cam, k.fx, k.fy, k.cx, k.cy, w, h);
    if (gridOnly) {
      // What training renders: one depth per grid cell and nothing else.
      imageCtx.fillStyle = `rgb(${NO_DEPTH})`;
      imageCtx.fillRect(0, 0, w, h);
      const cw = w / cam.W;
      const ch = (v.length > 1 ? Math.abs(v[1] - v[0]) : 5) || 5;
      for (let r = 0; r < cam.rows; r += 1) {
        for (let c = 0; c < cam.W; c += 1) {
          if (!okU[c] || !okV[r]) {
            continue;
          }
          imageCtx.fillStyle = `rgb(${color(grid[r * cam.W + c])})`;
          imageCtx.fillRect(u[c] - cw / 2, v[r] - ch / 2, cw, ch);
        }
      }
    } else {
      const image = imageCtx.createImageData(w, h);
      for (let i = 0; i < w * h; i += 1) {
        const [r, g, b] = color(depth[i]);
        image.data[4 * i] = r;
        image.data[4 * i + 1] = g;
        image.data[4 * i + 2] = b;
        image.data[4 * i + 3] = 255;
      }
      imageCtx.putImageData(image, 0, 0);
    }
    for (let r = 0; r < cam.rows; r += 1) {
      for (let c = 0; c < cam.W; c += 1) {
        if (!okU[c] || !okV[r]) {
          continue;
        }
        const inBand = keep[r * cam.W + c];
        imageCtx.fillStyle = inBand ? "#1baf7a" : "rgba(235, 235, 230, 0.55)";
        const s = inBand ? 3 : 2;
        imageCtx.fillRect(u[c] + 0.5 - s / 2, v[r] + 0.5 - s / 2, s, s);
      }
    }
    imageCtx.strokeStyle = "#eb6834";
    imageCtx.lineWidth = 2;
    for (let c = 0; c < cam.W; c += 1) {
      const r = beamRow[c];
      if (r >= 0 && okU[c] && okV[r]) {
        imageCtx.beginPath();
        imageCtx.arc(u[c] + 0.5, v[r] + 0.5, 4, 0, 2 * Math.PI);
        imageCtx.stroke();
      }
    }
  }

  function drawScan(scan, encoded) {
    const ctx = scanCtx;
    const scale = (SCAN_H - 34) / (cam.scanMax + 0.5);
    const ox = SCAN_W / 2;
    const oy = SCAN_H - 26;
    // Car frame, x forward (up the canvas), y left (left on the canvas).
    const px = (x, y) => [ox - y * scale, oy - (x - cam.x) * scale];
    ctx.fillStyle = "#11191d";
    ctx.fillRect(0, 0, SCAN_W, SCAN_H);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
    ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
    ctx.font = "11px ui-monospace, monospace";
    ctx.lineWidth = 1;
    const half = (cam.hfovDeg * Math.PI) / 360;
    for (const r of [1, 2, 5, cam.scanMax]) {
      ctx.beginPath();
      for (let i = 0; i <= 40; i += 1) {
        const th = -half + (2 * half * i) / 40;
        const [x, y] = px(cam.x + r * Math.cos(th), r * Math.sin(th));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      const [lx, ly] = px(cam.x + r, 0);
      ctx.fillText(`${r} m`, lx + 3, ly - 2);
    }
    for (let c = 0; c < cam.W; c += 1) {
      const invalid = encoded[c] <= INVALID + 1e-9;
      const r = invalid ? 0.3 : Math.min(scan[c], cam.scanMax);
      const az = cam.azimuth[c];
      const [x0, y0] = px(cam.x, 0);
      const [x1, y1] = px(cam.x + r * Math.cos(az), r * Math.sin(az));
      ctx.strokeStyle = invalid ? "#d03b3b" : `rgb(${LUT[Math.round(encoded[c] * 255)]})`;
      ctx.lineWidth = invalid ? 2 : 1.5;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    // The chassis (0.55 x 0.30), camera at its nose.
    const [bx, by] = px(cam.x, 0.15);
    ctx.fillStyle = "#2ee86a";
    ctx.fillRect(bx, by, 0.3 * scale, 0.55 * scale);
  }

  function draw(message, pose) {
    const frame = decode(message);
    if (!frame) {
      stats.textContent = `Unsupported depth format ${message.pixel_format_type}`;
      return;
    }
    const { depth, w, h } = frame;
    const k = intrinsicsFor(w, h);
    const { roll, pitch } = pose?.orientation ? rollPitch(pose.orientation) : { roll: 0, pitch: 0 };
    const grid = sampleDepth(cam, depth, w, h, k.fx, k.fy, k.cx, k.cy);
    const { scan, keep, beamRow } = depthToScan(cam, grid, cam.z, pitch, roll);
    const encoded = encode(cam, scan);
    last = { depth, w, h, k, grid, keep, beamRow };
    drawImage(last);
    drawScan(scan, encoded);

    frames += 1;
    rateFrames += 1;
    const now = performance.now();
    if (now - rateStart > 2000) {
      fps = (1000 * rateFrames) / (now - rateStart);
      rateStart = now;
      rateFrames = 0;
    }
    let valid = 0;
    let nearest = Infinity;
    for (let c = 0; c < cam.W; c += 1) {
      if (encoded[c] > INVALID + 1e-9) {
        valid += 1;
        nearest = Math.min(nearest, scan[c]);
      }
    }
    const deg = (x) => ((x * 180) / Math.PI).toFixed(1);
    stats.textContent =
      `${POLICY_NAME} grid ${cam.W}x${cam.rows} · ${valid}/${cam.W} beams valid` +
      (Number.isFinite(nearest) ? ` · nearest ${nearest.toFixed(2)} m` : "") +
      ` · roll ${deg(roll)}° pitch ${deg(pitch)}°` +
      (k.nominal ? " · nominal intrinsics (no camera_info yet)" : "") +
      (fps ? ` · ${fps.toFixed(1)} fps shown` : "");
    hint.hidden = true;
  }

  return {
    draw,
    setInfo,
    get frames() {
      return frames;
    },
    setHint(text) {
      hint.textContent = text;
      hint.hidden = !text;
    },
  };
}
