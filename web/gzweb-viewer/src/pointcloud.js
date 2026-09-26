// Decode Gazebo's ZED cloud once for both scene views. Points already use
// body axes (+x forward, +y left, +z up), so only the camera offset applies.
// Read the declared field layout, with a fast path for ZED's common layout.
// Gazebo and ROS PointCloud2 datatype enums differ by one: Gazebo FLOAT32 is
// 6, while ROS 6 is UINT32. Mixing them turns infinity into a finite distant
// point and breaks the viewer's bounds (see readScalar).

import * as THREE from "three";

const CAMERA_OFFSET = { x: 0.315, y: 0, z: 0.20 };
const POINT_SIZE = 0.03;
const DEFAULT_COLOR = new THREE.Color(0x63c2f0);

// Gamma lifts shadowed returns against the dark viewport without clipping
// highlights. 1.0 would show raw camera colors.
const COLOR_LIFT = 1.45;

// Precompute the byte conversion for ~150k points per frame.
const BRIGHTNESS = new Float32Array(256);
for (let i = 0; i < BRIGHTNESS.length; i += 1) {
  BRIGHTNESS[i] = Math.min(1, (i / 255) ** (1 / COLOR_LIFT));
}

// For the separate-r/g/b layouts, whose channels arrive as numbers rather
// than as bytes of a packed word.
function liftChannel(value) {
  const byte = Math.round(value);
  if (byte <= 0) {
    return 0;
  }
  return byte >= 255 ? 1 : BRIGHTNESS[byte];
}

const FLOAT32 = 6;

// How the colors for a frame are produced, decided once per message instead
// of re-asked per point.
const COLOR_NONE = 0;
const COLOR_PACKED = 1;
const COLOR_CHANNELS = 2;

// gz.msgs.PointCloudPacked.Field.DataType, which counts from ZERO:
//
//   INT8=0 UINT8=1 INT16=2 UINT16=3 INT32=4 UINT32=5 FLOAT32=6 FLOAT64=7
//
// sensor_msgs/PointField counts from one (INT8=1 ... FLOAT32=7, FLOAT64=8).
// This decodes the raw gz message off the websocket, so it is the gz enum
// that applies.  Unknown values return NaN rather than 0, so a schema this
// does not understand drops its points at the finite check instead of piling
// them all on the origin.
function readScalar(view, offset, datatype, littleEndian) {
  switch (datatype) {
    case 0: return view.getInt8(offset);
    case 1: return view.getUint8(offset);
    case 2: return view.getInt16(offset, littleEndian);
    case 3: return view.getUint16(offset, littleEndian);
    case 4: return view.getInt32(offset, littleEndian);
    case 5: return view.getUint32(offset, littleEndian);
    case 6: return view.getFloat32(offset, littleEndian);
    case 7: return view.getFloat64(offset, littleEndian);
    default: return NaN;
  }
}

// The decoded frame, owned by one decoder and read by every view that draws
// it.  Kept separate from the THREE.Points wrapper because the same 5.5 MB
// message feeds two scenes (the main view's cloud and the chase viewport's),
// and decoding 230k points twice per frame was the single most expensive
// thing this page did.  Views hold BufferAttributes straight over these
// arrays -- no copy -- so `generation` tells them when a reallocation has
// left their attribute pointing at a freed array.
export function createCloudBuffers() {
  const state = {
    positions: new Float32Array(0),
    colors: new Float32Array(0),
    count: 0,
    hasColor: false,
    generation: 0,
  };
  let capacity = 0;
  let warnedSchema = false;

  function ensureCapacity(pointCount) {
    if (pointCount <= capacity) {
      return;
    }
    // Grown in doubling steps rather than to exactly what this frame needs:
    // a reallocation forces every view to rebuild its BufferAttribute and
    // the GPU to re-upload the whole buffer, and a cloud that creeps up a
    // few points at a time would otherwise pay that on nearly every frame.
    capacity = Math.max(pointCount, capacity * 2);
    state.positions = new Float32Array(capacity * 3);
    state.colors = new Float32Array(capacity * 3);
    state.generation += 1;
  }

  // Which decode path this message's field layout allows.  The fast one
  // reads the payload as a Float32Array with a stride, which needs every
  // offset, the stride and the payload's own start to be 4-byte aligned and
  // every coordinate to be little-endian FLOAT32.  That is exactly what the
  // ZED publishes; anything else falls back to per-scalar DataView reads.
  function fastLayoutOk(fields, pointStep, byteOffset, littleEndian) {
    return littleEndian
      && Number.isInteger(pointStep / 4)
      && byteOffset % 4 === 0
      && fields.every((f) => f && f.datatype === FLOAT32 && f.offset % 4 === 0);
  }

  /**
   * Decode one gz.msgs.PointCloudPacked into the shared buffers.
   * Returns false if the message could not be used, in which case the
   * previous frame's contents are left alone.
   */
  function ingest(message) {
    const fields = message.field ?? [];
    const xField = fields.find((f) => f.name === "x");
    const yField = fields.find((f) => f.name === "y");
    const zField = fields.find((f) => f.name === "z");
    if (!xField || !yField || !zField) {
      if (!warnedSchema) {
        console.warn("Point cloud message has no x/y/z fields; cannot render it.", fields);
        warnedSchema = true;
      }
      return false;
    }
    const rgbField = fields.find((f) => f.name === "rgb" || f.name === "rgba");
    const rField = fields.find((f) => f.name === "r");
    const gField = fields.find((f) => f.name === "g");
    const bField = fields.find((f) => f.name === "b");

    const pointStep = message.point_step;
    if (!(pointStep > 0)) {
      return false;
    }
    const bytes = message.data instanceof Uint8Array ? message.data : new Uint8Array(message.data);
    const littleEndian = !message.is_bigendian;
    const totalPoints = Math.floor(bytes.byteLength / pointStep);
    ensureCapacity(totalPoints);

    let colorMode = COLOR_NONE;
    if (rgbField) {
      colorMode = COLOR_PACKED;
    } else if (rField && gField && bField) {
      colorMode = COLOR_CHANNELS;
    }

    const positions = state.positions;
    const colors = state.colors;
    const xOffset = xField.offset;
    const yOffset = yField.offset;
    const zOffset = zField.offset;
    let count = 0;

    // Separate r/g/b channels are byte-sized in every layout seen so far, so
    // they get no fast path; a packed rgb is read as raw bits and only has to
    // be word-aligned, whatever type it claims to be.
    const fast = colorMode !== COLOR_CHANNELS
      && fastLayoutOk([xField, yField, zField], pointStep, bytes.byteOffset, littleEndian)
      && (colorMode !== COLOR_PACKED || rgbField.offset % 4 === 0);

    if (fast) {
      // One Float32Array and one Uint32Array over the same payload: the
      // coordinates want the float values, the packed color wants the raw
      // bits.  Both index in 4-byte words, so the strides match.
      const stride = pointStep / 4;
      const words = totalPoints * stride;
      const floats = new Float32Array(bytes.buffer, bytes.byteOffset, words);
      const uints = colorMode === COLOR_PACKED
        ? new Uint32Array(bytes.buffer, bytes.byteOffset, words)
        : null;
      const xWord = xOffset / 4;
      const yWord = yOffset / 4;
      const zWord = zOffset / 4;
      const rgbWord = colorMode === COLOR_PACKED ? rgbField.offset / 4 : 0;

      for (let i = 0; i < totalPoints; i += 1) {
        const base = i * stride;
        const bodyX = floats[base + xWord];
        const bodyY = floats[base + yWord];
        const bodyZ = floats[base + zWord];
        // The sky comes back +Inf and unreturned rays NaN; both fail this.
        if (!Number.isFinite(bodyX) || !Number.isFinite(bodyY) || !Number.isFinite(bodyZ)) {
          continue;
        }
        const out = count * 3;
        positions[out] = bodyX + CAMERA_OFFSET.x;
        positions[out + 1] = bodyY + CAMERA_OFFSET.y;
        positions[out + 2] = bodyZ + CAMERA_OFFSET.z;
        if (uints) {
          const bits = uints[base + rgbWord];
          colors[out] = BRIGHTNESS[(bits >> 16) & 0xff];
          colors[out + 1] = BRIGHTNESS[(bits >> 8) & 0xff];
          colors[out + 2] = BRIGHTNESS[bits & 0xff];
        }
        count += 1;
      }
    } else {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let i = 0; i < totalPoints; i += 1) {
        const base = i * pointStep;
        const bodyX = readScalar(view, base + xOffset, xField.datatype, littleEndian);
        const bodyY = readScalar(view, base + yOffset, yField.datatype, littleEndian);
        const bodyZ = readScalar(view, base + zOffset, zField.datatype, littleEndian);
        if (!Number.isFinite(bodyX) || !Number.isFinite(bodyY) || !Number.isFinite(bodyZ)) {
          continue;
        }
        const out = count * 3;
        positions[out] = bodyX + CAMERA_OFFSET.x;
        positions[out + 1] = bodyY + CAMERA_OFFSET.y;
        positions[out + 2] = bodyZ + CAMERA_OFFSET.z;
        if (colorMode === COLOR_PACKED) {
          // Bit pattern is a packed 0xRRGGBB regardless of whether the field
          // declares itself FLOAT32 or UINT32 -- read the raw bytes, not the
          // field's nominal numeric type.
          const bits = view.getUint32(base + rgbField.offset, littleEndian);
          colors[out] = BRIGHTNESS[(bits >> 16) & 0xff];
          colors[out + 1] = BRIGHTNESS[(bits >> 8) & 0xff];
          colors[out + 2] = BRIGHTNESS[bits & 0xff];
        } else if (colorMode === COLOR_CHANNELS) {
          colors[out] = liftChannel(readScalar(view, base + rField.offset, rField.datatype, littleEndian));
          colors[out + 1] = liftChannel(readScalar(view, base + gField.offset, gField.datatype, littleEndian));
          colors[out + 2] = liftChannel(readScalar(view, base + bField.offset, bField.datatype, littleEndian));
        }
        count += 1;
      }
    }

    state.count = count;
    // A cloud with no color fields is drawn in one flat material color
    // instead of having DEFAULT_COLOR written into 690k array slots and
    // pushed to the GPU every frame, none of which ever change.
    state.hasColor = colorMode !== COLOR_NONE;
    return true;
  }

  // The state object itself is the handle: views read positions/colors/count
  // off it directly, and it stays identity-stable across reallocations.
  state.ingest = ingest;
  return state;
}

// One THREE.Points view onto a decoder's buffers.  Several of these can share
// one decoder; each keeps its own BufferAttributes because a three.js
// attribute caches its upload state per renderer and the two views render
// through two different WebGL contexts.
export function createPointCloud() {
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.PointsMaterial({
    size: POINT_SIZE,
    color: DEFAULT_COLOR,
    vertexColors: false,
    sizeAttenuation: true,
  });
  const object3D = new THREE.Points(geometry, material);
  object3D.name = "zed-point-cloud";
  // The cloud's own bounds change every frame and are cheap to get wrong by
  // culling a car-sized object; not worth recomputing a frustum test for.
  // Nothing raycasts it either, so its bounding sphere is never read and is
  // deliberately never computed -- that call walked the whole attribute
  // (capacity, not just the points in use) once per frame per view.
  object3D.frustumCulled = false;
  object3D.visible = false;

  let generation = -1;

  /** Point this view at the decoder's current frame. */
  function draw(buffers) {
    const { positions, colors, count, hasColor } = buffers;
    if (buffers.generation !== generation) {
      generation = buffers.generation;
      // DynamicDrawUsage: these are rewritten every frame, and the default
      // STATIC_DRAW hint makes drivers pick storage that is wrong for that.
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage),
      );
      geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage),
      );
    }

    // Only the points that survived the finite check are uploaded.  The
    // buffers are sized for the full 230400-point frame but typically two
    // thirds of that is sky, so a full upload would push ~800 KB of stale
    // tail per attribute per frame for nothing.  three.js resets updateRange
    // after each upload, so it is set again here every frame.
    const used = count * 3;
    const position = geometry.attributes.position;
    position.updateRange.offset = 0;
    position.updateRange.count = used;
    position.needsUpdate = true;

    if (hasColor) {
      const color = geometry.attributes.color;
      color.updateRange.offset = 0;
      color.updateRange.count = used;
      color.needsUpdate = true;
    }
    if (material.vertexColors !== hasColor) {
      material.vertexColors = hasColor;
      material.needsUpdate = true;
    }

    geometry.setDrawRange(0, count);
  }

  return { object3D, draw };
}
