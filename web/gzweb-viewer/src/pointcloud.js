// Live point cloud from the sim's ZED RGB-D sensor (the "zed2i" rgbd_camera
// in jetson/cfr_arduino_bridge/launch/sensors_world.py), decoded straight off
// the gz-transport wire and rendered as a THREE.Points object.  main.js
// parents it to the "slash" model and is the only thing that makes it
// visible -- see its point-cloud-view toggle.
//
// gz-sim's rgbd_camera puts points on the wire already in the body
// convention (+x forward, +y left, +z up), not the optical convention (+z
// forward, +x right, +y down) a raw depth image uses -- see the header of
// rl/bale_follower/cloud_scan.py, which measured this directly off a live
// cloud (x spanning 0.25-12.7 m, y/z near zero) after an earlier version
// assumed optical and got an empty scan back for it. So no axis rotation is
// needed here, only the translation: the sensor's own pose,
// "0.315 0 0.20 0 0 0", is relative to the chassis link (which sits at the
// model root with no pose of its own) and already carries zero rotation,
// consistent with the cloud arriving pre-rotated into that same frame.
//
// "gz.msgs.PointCloudPacked" is decoded generically off whatever field
// layout the message itself declares (see decodeFields), the same way a ROS
// PointCloud2 would be, rather than assuming fixed byte offsets: the field
// list is small and this only runs a few times a second.

import * as THREE from "three";

const CAMERA_OFFSET = { x: 0.315, y: 0, z: 0.20 };
const POINT_SIZE = 0.03;
const DEFAULT_COLOR = new THREE.Color(0x2596d8);

function readScalar(view, offset, datatype, littleEndian) {
  switch (datatype) {
    case 1: return view.getInt8(offset);
    case 2: return view.getUint8(offset);
    case 3: return view.getInt16(offset, littleEndian);
    case 4: return view.getUint16(offset, littleEndian);
    case 5: return view.getInt32(offset, littleEndian);
    case 6: return view.getUint32(offset, littleEndian);
    case 7: return view.getFloat32(offset, littleEndian);
    case 8: return view.getFloat64(offset, littleEndian);
    default: return 0;
  }
}

export function createPointCloud() {
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.PointsMaterial({
    size: POINT_SIZE,
    vertexColors: true,
    sizeAttenuation: true,
  });
  const object3D = new THREE.Points(geometry, material);
  object3D.name = "zed-point-cloud";
  // The cloud's own bounds change every frame and are cheap to get wrong by
  // culling a car-sized object; not worth recomputing a frustum test for.
  object3D.frustumCulled = false;
  object3D.visible = false;

  let capacity = 0;
  let positions = new Float32Array(0);
  let colors = new Float32Array(0);
  let warnedSchema = false;

  function ensureCapacity(pointCount) {
    if (pointCount <= capacity) {
      return;
    }
    capacity = pointCount;
    positions = new Float32Array(capacity * 3);
    colors = new Float32Array(capacity * 3);
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  }

  function update(message) {
    const fields = message.field ?? [];
    const xField = fields.find((f) => f.name === "x");
    const yField = fields.find((f) => f.name === "y");
    const zField = fields.find((f) => f.name === "z");
    if (!xField || !yField || !zField) {
      if (!warnedSchema) {
        console.warn("Point cloud message has no x/y/z fields; cannot render it.", fields);
        warnedSchema = true;
      }
      return;
    }
    const rgbField = fields.find((f) => f.name === "rgb" || f.name === "rgba");
    const rField = fields.find((f) => f.name === "r");
    const gField = fields.find((f) => f.name === "g");
    const bField = fields.find((f) => f.name === "b");

    const bytes = message.data instanceof Uint8Array ? message.data : new Uint8Array(message.data);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const littleEndian = !message.is_bigendian;
    const pointStep = message.point_step;
    if (!(pointStep > 0)) {
      return;
    }
    const totalPoints = Math.floor(bytes.byteLength / pointStep);
    ensureCapacity(totalPoints);

    let count = 0;
    for (let i = 0; i < totalPoints; i += 1) {
      const base = i * pointStep;
      const bodyX = readScalar(view, base + xField.offset, xField.datatype, littleEndian);
      const bodyY = readScalar(view, base + yField.offset, yField.datatype, littleEndian);
      const bodyZ = readScalar(view, base + zField.offset, zField.datatype, littleEndian);
      if (!Number.isFinite(bodyX) || !Number.isFinite(bodyY) || !Number.isFinite(bodyZ)) {
        continue;
      }

      const out = count * 3;
      positions[out] = bodyX + CAMERA_OFFSET.x;
      positions[out + 1] = bodyY + CAMERA_OFFSET.y;
      positions[out + 2] = bodyZ + CAMERA_OFFSET.z;

      if (rgbField) {
        // Bit pattern is a packed 0xRRGGBB regardless of whether the field
        // declares itself FLOAT32 or UINT32 -- read the raw bytes, not the
        // field's nominal numeric type.
        const bits = view.getUint32(base + rgbField.offset, littleEndian);
        colors[out] = ((bits >> 16) & 0xff) / 255;
        colors[out + 1] = ((bits >> 8) & 0xff) / 255;
        colors[out + 2] = (bits & 0xff) / 255;
      } else if (rField && gField && bField) {
        colors[out] = readScalar(view, base + rField.offset, rField.datatype, littleEndian) / 255;
        colors[out + 1] = readScalar(view, base + gField.offset, gField.datatype, littleEndian) / 255;
        colors[out + 2] = readScalar(view, base + bField.offset, bField.datatype, littleEndian) / 255;
      } else {
        colors[out] = DEFAULT_COLOR.r;
        colors[out + 1] = DEFAULT_COLOR.g;
        colors[out + 2] = DEFAULT_COLOR.b;
      }

      count += 1;
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
    geometry.setDrawRange(0, count);
    geometry.computeBoundingSphere();
  }

  return { object3D, update };
}
