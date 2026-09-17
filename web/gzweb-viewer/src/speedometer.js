// Analog speed gauge overlaid on the 3D scene.
//
// The dial is an SVG built here rather than in index.html: the tick ring is 20
// elements that differ only by angle, and both it and the sweep have to be
// recomputed whenever DIAL_MAX changes, so generating it keeps the two in step.
//
// Speed is reported from pose differencing (see main.js) at whatever rate the
// simulation publishes, which is far above the display rate and jittery in dt.
// The needle therefore chases the reported value in a requestAnimationFrame
// loop instead of being written directly, so its motion is tied to frames
// rather than to message arrival.

const DIAL_MAX = 6; // m/s full scale; the planned line tops out at 5.5
const REDLINE = 5.5; // above the fastest plan we run -- see rl/bale_follower/course_path.py
const START_ANGLE = 135; // SVG degrees, 0 = +x. 135 -> 405 leaves the gap at the bottom.
const SWEEP = 270;
const RADIUS = 78;
const CENTER = 100;
const ARC_LENGTH = RADIUS * SWEEP * (Math.PI / 180);
const SVG_NS = "http://www.w3.org/2000/svg";

// Needle tracking. TAU_RISE is shorter than TAU_FALL so the gauge answers the
// throttle promptly but does not snap back to zero on a single dropped frame.
const TAU_RISE = 0.07;
const TAU_FALL = 0.16;
const STALE_MS = 400; // no pose this long -> the car is not moving, or we lost the stream

function polar(radius, degrees) {
  const radians = (degrees * Math.PI) / 180;
  return {
    x: CENTER + radius * Math.cos(radians),
    y: CENTER + radius * Math.sin(radians),
  };
}

function angleFor(speed) {
  const fraction = Math.min(Math.max(speed / DIAL_MAX, 0), 1);
  return START_ANGLE + fraction * SWEEP;
}

function arcPath(radius, startDegrees, endDegrees) {
  const start = polar(radius, startDegrees);
  const end = polar(radius, endDegrees);
  const largeArc = Math.abs(endDegrees - startDegrees) > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

function element(name, attributes) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function buildTicks(group) {
  // Majors every 1 m/s with a label, minors every 0.5.
  for (let step = 0; step <= DIAL_MAX * 2; step += 1) {
    const speed = step / 2;
    const major = step % 2 === 0;
    const degrees = angleFor(speed);
    const inner = polar(major ? 66 : 71, degrees);
    const outer = polar(78, degrees);
    group.appendChild(element("line", {
      x1: inner.x,
      y1: inner.y,
      x2: outer.x,
      y2: outer.y,
      class: major ? "tick tick-major" : "tick",
    }));
    if (!major) {
      continue;
    }
    const label = polar(54, degrees);
    const text = element("text", { x: label.x, y: label.y, class: "tick-label" });
    text.textContent = String(speed);
    group.appendChild(text);
  }
}

export function createSpeedometer(container) {
  container.innerHTML = "";

  const svg = element("svg", { viewBox: "0 0 200 200", class: "speedo-dial" });
  svg.appendChild(element("circle", { cx: CENTER, cy: CENTER, r: 94, class: "dial-face" }));
  svg.appendChild(element("path", {
    d: arcPath(RADIUS, START_ANGLE, START_ANGLE + SWEEP),
    class: "dial-track",
  }));
  svg.appendChild(element("path", {
    d: arcPath(RADIUS, angleFor(REDLINE), START_ANGLE + SWEEP),
    class: "dial-redline",
  }));

  const ticks = element("g", {});
  buildTicks(ticks);
  svg.appendChild(ticks);

  // One static path dashed down to length, rather than a path rebuilt per
  // frame: the arc geometry never changes, only how much of it is painted.
  const value = element("path", {
    d: arcPath(RADIUS, START_ANGLE, START_ANGLE + SWEEP),
    class: "dial-value",
    "stroke-dasharray": `0 ${ARC_LENGTH}`,
  });
  svg.appendChild(value);

  const peakMark = element("line", { class: "dial-peak", x1: 0, y1: 0, x2: 0, y2: 0 });
  svg.appendChild(peakMark);

  const needle = element("g", { class: "dial-needle" });
  needle.appendChild(element("path", { d: "M 92 96.6 L 170 99.4 L 170 100.6 L 92 103.4 Z" }));
  svg.appendChild(needle);
  svg.appendChild(element("circle", { cx: CENTER, cy: CENTER, r: 6.5, class: "dial-hub" }));

  container.appendChild(svg);

  const readout = document.createElement("div");
  readout.className = "speedo-readout";
  readout.innerHTML = `
    <p class="speedo-value">0.00</p>
    <p class="speedo-unit">m/s</p>
    <p class="speedo-peak">peak <span>0.00</span></p>
  `;
  container.appendChild(readout);

  const valueText = readout.querySelector(".speedo-value");
  const peakText = readout.querySelector(".speedo-peak span");

  let reported = 0;
  let displayed = 0;
  let peak = 0;
  let lastReportAt = 0;
  let lastFrameAt = performance.now();

  function paint() {
    const fraction = Math.min(displayed / DIAL_MAX, 1);
    value.setAttribute("stroke-dasharray", `${fraction * ARC_LENGTH} ${ARC_LENGTH}`);
    value.setAttribute(
      "class",
      displayed >= REDLINE ? "dial-value over" : displayed >= DIAL_MAX * 0.6 ? "dial-value warm" : "dial-value",
    );
    needle.setAttribute("transform", `rotate(${angleFor(displayed)} ${CENTER} ${CENTER})`);
    valueText.textContent = displayed.toFixed(2);
  }

  function frame(now) {
    const dt = Math.min(Math.max((now - lastFrameAt) / 1000, 0), 0.1);
    lastFrameAt = now;

    // A stream that has gone quiet means no motion to show. Fall to zero
    // rather than freezing the needle at the last value, which would read as
    // a car still travelling.
    const target = now - lastReportAt > STALE_MS ? 0 : reported;
    const tau = target > displayed ? TAU_RISE : TAU_FALL;
    displayed += (target - displayed) * (1 - Math.exp(-dt / tau));
    if (Math.abs(target - displayed) < 0.005) {
      displayed = target;
    }
    paint();
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
  paint();

  return {
    report(speed) {
      if (!Number.isFinite(speed)) {
        return;
      }
      reported = Math.max(speed, 0);
      lastReportAt = performance.now();
      if (reported > peak) {
        peak = reported;
        peakText.textContent = peak.toFixed(2);
        const degrees = angleFor(peak);
        const inner = polar(64, degrees);
        const outer = polar(82, degrees);
        peakMark.setAttribute("x1", inner.x);
        peakMark.setAttribute("y1", inner.y);
        peakMark.setAttribute("x2", outer.x);
        peakMark.setAttribute("y2", outer.y);
        peakMark.setAttribute("class", "dial-peak shown");
      }
    },
    reset() {
      reported = 0;
      peak = 0;
      peakText.textContent = "0.00";
      peakMark.setAttribute("class", "dial-peak");
    },
  };
}
