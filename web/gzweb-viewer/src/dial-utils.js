// Shared SVG geometry for the HUD dials (speedometer, steering dial): a
// 200x200 viewBox, the face radius, and the sweep the needles ride on.  Kept
// in one place because both dials draw on the exact same geometry so they
// read consistently side by side.

export const SVG_NS = "http://www.w3.org/2000/svg";
export const CENTER = 100;
export const RADIUS = 78;
export const START_ANGLE = 135; // SVG degrees, 0 = +x. 135 -> 405 leaves the gap at the bottom.
export const SWEEP = 270;
export const ARC_LENGTH = RADIUS * SWEEP * (Math.PI / 180);

export function polar(radius, degrees) {
  const radians = (degrees * Math.PI) / 180;
  return {
    x: CENTER + radius * Math.cos(radians),
    y: CENTER + radius * Math.sin(radians),
  };
}

export function arcPath(radius, startDegrees, endDegrees) {
  const start = polar(radius, startDegrees);
  const end = polar(radius, endDegrees);
  const largeArc = Math.abs(endDegrees - startDegrees) > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

export function element(name, attributes) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}
