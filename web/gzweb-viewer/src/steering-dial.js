// Analog steering gauge overlaid on the 3D scene, next to the speedometer.
//
// It reads the commanded gz.msgs.Twist on /sim/cmd_vel (see main.js), which
// is the input Gazebo's AckermannSteering plugin actually acts on, and turns
// it back into a steering angle with the same bicycle model
// cmd_vel_to_drive_node.cpp uses to go the other way (steering angle ->
// yaw rate).  Curvature is derived from the displayed (smoothed) angle
// rather than smoothed separately, so the needle and the printed number
// never disagree.
//
// The gauge is centered on zero: negative commands (right turns) sweep the
// left half, positive (left turns) sweep the right half, same convention as
// a signed number line. The rotating steering-wheel icon at the centre is
// the more intuitive "which way is it turning" read; the arc is the "how
// hard" read.

import { CENTER, RADIUS, START_ANGLE, SWEEP, polar, arcPath, element } from "./dial-utils.js";

// Mirrors cmd_vel_to_drive_node.cpp and the steering joints in
// jetson/scripts/generate_vehicle_model.py / generate_obstacle_course.py, so
// the dial's range and printed curvature match what the linkage can do.
export const WHEELBASE = 0.324; // m
export const MAX_STEERING_ANGLE = 0.40; // rad, steering_limit on the joints
export const MIN_SPEED_FOR_STEERING = 0.3; // m/s floor before yaw_rate/speed blows up

// A real steering wheel turns much more than the road wheels; this ratio is
// far smaller than an actual steering gear, just enough that a small
// commanded angle is still a visible turn of the icon.
const WHEEL_ICON_RATIO = 4;
const WHEEL_RADIUS = 24;

const TAU = 0.08; // needle smoothing time constant, seconds
const STALE_MS = 400; // no command this long -> treat as centered, not last-known

const HALF_ARC_LENGTH = RADIUS * (SWEEP / 2) * (Math.PI / 180);
const WARM_FRACTION = 0.6;
const REDLINE_FRACTION = 0.92;

function angleFor(steer) {
  const fraction = Math.min(Math.max((steer / MAX_STEERING_ANGLE + 1) / 2, 0), 1);
  return START_ANGLE + fraction * SWEEP;
}

function buildTicks(group, center) {
  // Majors every 10 degrees with a label, minors every 5, out to the last
  // multiple of 5 inside the joint limit (~22.9 deg -> +/-20).
  const maxDegrees = Math.floor((MAX_STEERING_ANGLE * 180) / Math.PI / 5) * 5;
  for (let degrees = -maxDegrees; degrees <= maxDegrees; degrees += 5) {
    const major = degrees % 10 === 0;
    const angle = angleFor((degrees * Math.PI) / 180);
    const inner = polar(major ? 66 : 71, angle);
    const outer = polar(78, angle);
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
    const label = polar(54, angle);
    const text = element("text", { x: label.x, y: label.y, class: "tick-label" });
    text.textContent = String(degrees);
    group.appendChild(text);
  }
  const zero = polar(64, center);
  const zeroOuter = polar(82, center);
  group.appendChild(element("line", {
    x1: zero.x,
    y1: zero.y,
    x2: zeroOuter.x,
    y2: zeroOuter.y,
    class: "dial-center-mark",
  }));
}

export function createSteeringDial(container) {
  container.innerHTML = "";

  const center = START_ANGLE + SWEEP / 2; // 270: straight up, zero steer

  const svg = element("svg", { viewBox: "0 0 200 200", class: "speedo-dial" });
  svg.appendChild(element("circle", { cx: CENTER, cy: CENTER, r: 94, class: "dial-face" }));
  svg.appendChild(element("path", {
    d: arcPath(RADIUS, START_ANGLE, START_ANGLE + SWEEP),
    class: "dial-track",
  }));
  // Redline zones at both ends -- past the joint limit either direction.
  svg.appendChild(element("path", {
    d: arcPath(RADIUS, angleFor(-MAX_STEERING_ANGLE), angleFor(-MAX_STEERING_ANGLE * REDLINE_FRACTION)),
    class: "dial-redline",
  }));
  svg.appendChild(element("path", {
    d: arcPath(RADIUS, angleFor(MAX_STEERING_ANGLE * REDLINE_FRACTION), angleFor(MAX_STEERING_ANGLE)),
    class: "dial-redline",
  }));

  const ticks = element("g", {});
  buildTicks(ticks, center);
  svg.appendChild(ticks);

  // Two static half-arcs rather than one, so the fill can grow outward from
  // centre in either direction.  Both are built start-to-end in increasing
  // angle order (matching arcPath's known-good winding); the right half's
  // path already starts at centre, so it grows like the speedometer's single
  // arc does.  The left half's path starts at its -max end, so it is grown
  // from its *end* instead, via a negative dashoffset -- see paint().
  const rightValue = element("path", {
    d: arcPath(RADIUS, center, START_ANGLE + SWEEP),
    class: "dial-value",
    "stroke-dasharray": `0 ${HALF_ARC_LENGTH * 2}`,
  });
  const leftValue = element("path", {
    d: arcPath(RADIUS, START_ANGLE, center),
    class: "dial-value",
    "stroke-dasharray": `0 ${HALF_ARC_LENGTH * 2}`,
  });
  svg.appendChild(leftValue);
  svg.appendChild(rightValue);

  // The steering wheel itself: rim, three spokes (the top one picked out so
  // rotation direction reads at a glance), hub.
  const wheel = element("g", { class: "steer-wheel" });
  wheel.appendChild(element("circle", { cx: CENTER, cy: CENTER, r: WHEEL_RADIUS, class: "steer-wheel-rim" }));
  [center, center + 120, center + 240].forEach((spokeAngle, index) => {
    const inner = polar(7, spokeAngle);
    const outer = polar(WHEEL_RADIUS, spokeAngle);
    wheel.appendChild(element("line", {
      x1: inner.x,
      y1: inner.y,
      x2: outer.x,
      y2: outer.y,
      class: index === 0 ? "steer-wheel-spoke steer-wheel-spoke-top" : "steer-wheel-spoke",
    }));
  });
  wheel.appendChild(element("circle", { cx: CENTER, cy: CENTER, r: 7, class: "steer-wheel-hub" }));
  svg.appendChild(wheel);

  container.appendChild(svg);

  const readout = document.createElement("div");
  readout.className = "speedo-readout";
  readout.innerHTML = `
    <p class="speedo-value">0.0</p>
    <p class="speedo-unit">deg</p>
    <p class="speedo-peak">&kappa; <span>0.00</span> 1/m</p>
  `;
  container.appendChild(readout);

  const valueText = readout.querySelector(".speedo-value");
  const curvatureText = readout.querySelector(".speedo-peak span");

  let reported = 0;
  let displayed = 0;
  let lastReportAt = 0;
  let lastFrameAt = performance.now();

  function paint() {
    const fraction = Math.max(-1, Math.min(1, displayed / MAX_STEERING_ANGLE));
    const magnitude = Math.abs(fraction);
    const cls = magnitude >= REDLINE_FRACTION ? "dial-value over" : magnitude >= WARM_FRACTION ? "dial-value warm" : "dial-value";

    if (fraction >= 0) {
      rightValue.setAttribute("stroke-dasharray", `${magnitude * HALF_ARC_LENGTH} ${HALF_ARC_LENGTH * 2}`);
      rightValue.setAttribute("class", cls);
      leftValue.setAttribute("stroke-dasharray", `0 ${HALF_ARC_LENGTH * 2}`);
    } else {
      leftValue.setAttribute("stroke-dasharray", `${magnitude * HALF_ARC_LENGTH} ${HALF_ARC_LENGTH * 2}`);
      leftValue.setAttribute("stroke-dashoffset", `${magnitude * HALF_ARC_LENGTH - HALF_ARC_LENGTH}`);
      leftValue.setAttribute("class", cls);
      rightValue.setAttribute("stroke-dasharray", `0 ${HALF_ARC_LENGTH * 2}`);
    }

    // Positive (left turn) is a counter-clockwise turn of a real wheel;
    // SVG's rotate() is clockwise-positive, so the sign flips here.
    const wheelDegrees = -(displayed * WHEEL_ICON_RATIO * 180) / Math.PI;
    wheel.setAttribute("transform", `rotate(${wheelDegrees} ${CENTER} ${CENTER})`);

    const curvature = Math.tan(displayed) / WHEELBASE;
    valueText.textContent = ((displayed * 180) / Math.PI).toFixed(1);
    curvatureText.textContent = curvature.toFixed(2);
  }

  function frame(now) {
    const dt = Math.min(Math.max((now - lastFrameAt) / 1000, 0), 0.1);
    lastFrameAt = now;

    // A stream that has gone quiet reads as centered, not as "last known
    // turn held forever" -- the same reasoning as the speedometer's stall.
    const target = now - lastReportAt > STALE_MS ? 0 : reported;
    displayed += (target - displayed) * (1 - Math.exp(-dt / TAU));
    if (Math.abs(target - displayed) < 0.0005) {
      displayed = target;
    }
    paint();
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
  paint();

  return {
    report(steeringAngleRadians) {
      if (!Number.isFinite(steeringAngleRadians)) {
        return;
      }
      reported = steeringAngleRadians;
      lastReportAt = performance.now();
    },
    reset() {
      reported = 0;
    },
  };
}
