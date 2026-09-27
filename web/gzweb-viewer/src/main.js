import { AssetViewer } from "gzweb";
import { Enum, parse } from "protobufjs";
import * as THREE from "three";
import { createSpeedometer } from "./speedometer.js";
import { createSteeringDial, WHEELBASE, MIN_SPEED_FOR_STEERING } from "./steering-dial.js";
import { createCloudView } from "./cloud-view.js";
import { createCloudBuffers, createPointCloud } from "./pointcloud.js";
import { createDepthView } from "./depth-view.js";
import { installInstancedBales, setStrandsVisible } from "./bales.js";
import "./style.css";

const status = document.querySelector("#viewer-status");
const statusIndicator = document.querySelector(".status span");
const websocketProtocol = window.location.protocol === "https:" ? "wss" : "ws";

// Which course the page shows, from ?course=.  The simulation only runs one
// at a time, so this has to match whichever launch file is up -- the world
// name is in every topic name, and a mismatch shows a static course that
// never connects.
const courses = {
  speed: {
    title: "Speed Course",
    world: "cfr_speed_course",
    // Looking down the 135 ft oval from outside the first turn.
    camera: [20.5, -20, 25],
    target: [20.5, 0, 0],
  },
  obstacle: {
    title: "Obstacle Course",
    world: "cfr_obstacle_course",
    // The course runs from x -10 to 10 and y -11 to 4; this frames all of it.
    camera: [1, -19, 15],
    target: [0, -4, 0],
  },
};
const courseName = new URLSearchParams(window.location.search).get("course");
const course = courses[courseName] ?? courses.speed;

// Vite serves these as root-relative paths, and the parser drops any URL that
// does not begin with "http" -- silently, as a console warning -- so they have
// to be made absolute before it sees them.
const absolute = (url) => new URL(url, document.baseURI).href;

// The parser matches mesh and albedo-map URIs against this list by filename,
// so include assets from both worlds even though only one is active at a time.
const assetUrls = [
  ...Object.values(import.meta.glob("../../../jetson/cfr_arduino_bridge/meshes/*.{stl,obj}", {
    eager: true,
    query: "?url",
    import: "default",
  })),
  ...Object.values(import.meta.glob("../../../jetson/cfr_arduino_bridge/materials/*.png", {
    eager: true,
    query: "?url",
    import: "default",
  })),
].map(absolute);
const worldUrls = import.meta.glob("../../../jetson/cfr_arduino_bridge/worlds/*.sdf", {
  eager: true,
  query: "?url",
  import: "default",
});
const worldFile = `${course.world.replace("cfr_", "")}.sdf`;
const worldUrl = absolute(
  Object.entries(worldUrls).find(([path]) => path.endsWith(`/${worldFile}`))[1],
);
const poseTopic = `/world/${course.world}/dynamic_pose/info`;
// The whole-world pose stream, which is the only one static models appear in.
// See sampleLayout, which is not a subscription for reasons explained there.
const layoutTopic = `/world/${course.world}/pose/info`;
// Not world-namespaced -- both AckermannSteering's <topic> in the world file
// and the ros_gz_bridge argument in simulation.launch.py spell it exactly
// this way regardless of course.  This is what the plugin is actually acting
// on, i.e. the closest thing to a "commanded" twist reachable from gz
// transport (see steering-dial.js for why it is not literally the
// autonomy stack's raw command).
const cmdVelTopic = "/sim/cmd_vel";
// Likewise unnamespaced; only published with `sensors:=true` at launch, so
// the point-cloud-view toggle simply shows nothing without it.
const pointCloudTopic = "/zed/gz/rgbd/points";
// The rgbd_camera's depth image and intrinsics -- what the bridge hands the
// formulaTwo node as /zed/zed_node/depth/depth_registered.
const depthImageTopic = "/zed/gz/rgbd/depth_image";
const depthInfoTopic = "/zed/gz/rgbd/camera_info";
// 0.9 MB a frame at 15 Hz is more than the page needs.  The websocket server
// throttles the topic for its browser clients; the ROS bridge is a separate
// Gazebo subscriber and still gets every frame.
const DEPTH_VIEW_HZ = 5;
const PIXEL_FORMAT_TYPE = Object.fromEntries(
  [
    "UNKNOWN_PIXEL_FORMAT", "L_INT8", "L_INT16", "RGB_INT8", "RGBA_INT8", "BGRA_INT8",
    "RGB_INT16", "RGB_INT32", "BGR_INT8", "BGR_INT16", "BGR_INT32", "R_FLOAT16",
    "RGB_FLOAT16", "R_FLOAT32", "RGB_FLOAT32", "BAYER_RGGB8", "BAYER_BGGR8",
    "BAYER_GBRG8", "BAYER_GRBG8",
  ].map((name, value) => [name, value]),
);
// The same cloud after cloud_segmentation_node, each point colored by the
// class it was given (legend in index.html).  Bridged ROS -> gz by
// simulation.launch.py, again only with `sensors:=true`.
const segmentedTopic = "/zed/segmented/points";
const LAYOUT_SAMPLE_MS = 3000;
const LAYOUT_SAMPLE_TIMEOUT_MS = 15000;
const worldControlService = `/world/${course.world}/control`;
const teleportApiUrl = `http://${window.location.hostname}:9003/api/sim/teleport`;
const signalApiUrl = `http://${window.location.hostname}:9003/api/sim/start-signal`;
const signalButton = document.querySelector("#start-signal");
let signalGo = false;
const followButton = document.querySelector("#follow-slash");
const resetRobotButton = document.querySelector("#reset-slash");
const capturePoseButton = document.querySelector("#capture-teleport-pose");
const previewButton = document.querySelector("#preview-teleport");
const teleportButton = document.querySelector("#teleport-slash");
const positionInputs = [
  document.querySelector("#teleport-x"),
  document.querySelector("#teleport-y"),
  document.querySelector("#teleport-heading-degrees"),
];
const speedometer = createSpeedometer(document.querySelector("#speedometer"));
const steeringDial = createSteeringDial(document.querySelector("#steering-dial"));
const pointCloud = createPointCloud();
// One decode per message, shared by both views: the frame is ~5.5 MB and
// 230400 points, and each view only wraps BufferAttributes over these arrays.
const cloudBuffers = createCloudBuffers();
// The always-on secondary viewport.  It has its own cloud instance and its own
// renderer; see cloud-view.js.
const cloudView = createCloudView(document.querySelector("#cloud-view"));
const pointCloudButton = document.querySelector("#point-cloud-view");
const segmentationButton = document.querySelector("#segmentation-view");
const segmentationLegend = document.querySelector("#segmentation-legend");
const depthButton = document.querySelector("#depth-view");
const depthWrap = document.querySelector("#depth-view-wrap");
const depthView = createDepthView(depthWrap);
let depthViewEnabled = false;
let depthSilenceTimer = 0;
// Which of the two clouds the views draw.  The other one is ignored rather
// than unsubscribed; the segmented one is only subscribed on first use.
let segmentationMode = false;
let segmentedSubscribed = false;
// How long to wait for a first cloud before telling the user the topic is
// silent.  The sensor runs at 15 Hz, so this is generous even on llvmpipe.
const POINT_CLOUD_SILENCE_MS = 4000;
let pointCloudSilenceTimer = 0;
let pointCloudFrames = 0;
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
// The pose stream carries position only, so speed is differenced from it.
// SPEED_TAU smooths the per-message dt jitter; anything above
// MAX_PLAUSIBLE_SPEED is a teleport or reset rather than travel.
const SPEED_TAU = 0.08;
const MAX_PLAUSIBLE_SPEED = 40;
let previousSpeedSample;
let filteredSpeed = 0;
let followSlash = false;
let latestSlashPose;
let followTargetOffset;
let followCameraOffset;
let updatingFollowView = false;
let followYaw;
let followYawStamp;
let simulationSocket;
let worldControlType;
let booleanType;
let cmdVelType;
let pointCloudType;
let imageType;
let cameraInfoType;
let teleportPreview;
// Parsed once from the live connection's dictionary, and used by the layout
// samples as well, which is why it is not local to connectPoseStream.
let poseType;
let pointCloudAttached = false;
let pointCloudSubscribed = false;
let pointCloudViewEnabled = false;
document.title = `CfR ${course.title} Viewer`;
document.querySelector("header h1").textContent = course.title;
// A link to each course.  Reloading is the switch: every topic name carries
// the world, so the page has to start over on the other course -- and it only
// comes alive if that is the course the simulation is running.
for (const [key, { title }] of Object.entries(courses)) {
  const link = document.createElement("a");
  const url = new URL(window.location.href);
  if (key === "speed") {
    url.searchParams.delete("course");
  } else {
    url.searchParams.set("course", key);
  }
  link.href = url.pathname + url.search;
  link.textContent = title;
  if (courses[key] === course) {
    link.setAttribute("aria-current", "page");
  }
  document.querySelector("#course-links").append(link);
}
document
  .querySelector("#gz-scene")
  .setAttribute("aria-label", `Gazebo ${course.title.toLowerCase()}`);

const viewer = new AssetViewer({
  elementId: "gz-scene",
  addModelLighting: true,
  enablePBR: true,
});

function showCourseOverview() {
  const scene = viewer["scene"];
  if (!scene) {
    return;
  }

  scene.camera.position.set(...course.camera);
  scene.camera.up.set(0, 0, 1);
  scene.controls.target.set(...course.target);
  scene.camera.lookAt(scene.controls.target);
  scene.controls.update();
}

function getSlashYaw(pose) {
  const { orientation } = pose;
  return Math.atan2(
    2 * (orientation.w * orientation.z + orientation.x * orientation.y),
    1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z),
  );
}

// Smooth heading briefly while keeping the camera level through banks and ramps.
const FOLLOW_YAW_TAU = 0.08;

function followRotation() {
  return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), followYaw);
}

function captureFollowView() {
  const scene = viewer["scene"];
  if (!followSlash || updatingFollowView || !latestSlashPose || !scene || followYaw === undefined) {
    return;
  }

  const inverse = followRotation().invert();
  const { x, y, z } = latestSlashPose.position;
  const position = new THREE.Vector3(x, y, z);
  const target = scene.controls.target;
  const targetOffset = target.clone().sub(position).applyQuaternion(inverse);
  const cameraOffset = scene.camera.position.clone().sub(target).applyQuaternion(inverse);
  // Preserve zoom and height, but keep the chase view behind the car.
  followTargetOffset.set(Math.max(0, targetOffset.x), 0, targetOffset.z);
  followCameraOffset.set(
    -Math.max(0.5, Math.hypot(cameraOffset.x, cameraOffset.y)),
    0,
    Math.max(0.3, cameraOffset.z),
  );
}

function startSlashFollowView(pose) {
  const scene = viewer["scene"];
  if (!scene || !pose) {
    return;
  }

  followYaw = getSlashYaw(pose);
  followYawStamp = performance.now();
  followTargetOffset = new THREE.Vector3(0.5, 0, 0.15);
  followCameraOffset = new THREE.Vector3(-3, 0, 3.85);
  updateSlashFollowView(pose);
}

function updateSlashFollowView(pose) {
  const scene = viewer["scene"];
  if (!scene || !pose || !followTargetOffset || !followCameraOffset) {
    return;
  }

  const now = performance.now();
  const dt = Math.min(Math.max((now - followYawStamp) / 1000, 0), 0.5);
  followYawStamp = now;
  const difference = getSlashYaw(pose) - followYaw;
  followYaw += Math.atan2(Math.sin(difference), Math.cos(difference)) *
    (1 - Math.exp(-dt / FOLLOW_YAW_TAU));
  const rotation = followRotation();
  const position = new THREE.Vector3(pose.position.x, pose.position.y, pose.position.z);
  const target = scene.controls.target;
  updatingFollowView = true;
  target.copy(position).add(followTargetOffset.clone().applyQuaternion(rotation));
  scene.camera.position.copy(target).add(followCameraOffset.clone().applyQuaternion(rotation));
  scene.camera.up.set(0, 0, 1);
  // gzweb controls update position without aiming the camera at their target.
  scene.camera.lookAt(target);
  scene.controls.update();
  updatingFollowView = false;
}

function toNumber(value) {
  // protobufjs hands back Long objects for the int64 fields in gz.msgs.Time.
  if (typeof value === "number") {
    return value;
  }
  if (value && typeof value.toNumber === "function") {
    return value.toNumber();
  }
  const converted = Number(value);
  return Number.isFinite(converted) ? converted : 0;
}

function poseTimeSeconds(message, pose) {
  // Simulation time, not wall clock: the sim does not always run at real time,
  // and differencing against the wall would misreport speed whenever it does not.
  const stamp = message?.header?.stamp ?? pose?.header?.stamp;
  if (stamp) {
    const seconds = toNumber(stamp.sec) + toNumber(stamp.nsec) * 1e-9;
    if (seconds > 0) {
      return seconds;
    }
  }
  return performance.now() / 1000;
}

function updateSpeed(message, pose) {
  const time = poseTimeSeconds(message, pose);
  const { x, y, z } = pose.position;
  const previous = previousSpeedSample;
  previousSpeedSample = { time, x, y, z };
  if (!previous) {
    return;
  }

  // Rejects dt <= 0 as well, which is what a sim-time reset looks like.
  const dt = time - previous.time;
  if (!(dt > 1e-4) || dt > 1) {
    return;
  }

  const raw = Math.hypot(x - previous.x, y - previous.y, z - previous.z) / dt;
  if (raw > MAX_PLAUSIBLE_SPEED) {
    filteredSpeed = 0;
    speedometer.report(0);
    return;
  }

  filteredSpeed += (raw - filteredSpeed) * (1 - Math.exp(-dt / SPEED_TAU));
  speedometer.report(filteredSpeed);
}

// Inverts the bicycle model cmd_vel_to_drive_node.cpp uses to turn a steering
// angle into a yaw rate, so the dial can go the other way on the twist
// Gazebo actually received.  The same MIN_SPEED_FOR_STEERING floor keeps a
// nearly-stopped car's yaw noise from reading as a hard-over command.
function updateCommandedSteering(twist) {
  const speed = twist.linear?.x;
  const yawRate = twist.angular?.z;
  if (!Number.isFinite(speed) || !Number.isFinite(yawRate)) {
    return;
  }
  const effectiveSpeed = Math.max(Math.abs(speed), MIN_SPEED_FOR_STEERING);
  steeringDial.report(Math.atan2(WHEELBASE * yawRate, effectiveSpeed));
}

// The point cloud is parented to "slash" rather than positioned in world
// coordinates every frame: it rides along with the vehicle for free once
// attached, the same way the vehicle mesh itself does.
function ensurePointCloudAttached(scene, slash) {
  if (pointCloudAttached || !slash) {
    return;
  }
  slash.add(pointCloud.object3D);
  pointCloudAttached = true;
}

// Objects this hid, so it can put back exactly what it took away rather than
// forcing everything visible again on the way out -- gzweb's own scene
// carries objects that are invisible on purpose (its GridHelper, named
// "grid", ships with visible=false and is centered on the world origin
// rather than the course), and blindly re-showing everything popped that
// grid in looking like the course had shifted.
let hiddenForPointCloudView = [];
let pointCloudViewClearColor = null;
// The page's own ink color (see style.css :root), just used as the render
// background instead of the course's default grey-green -- dark enough that
// the cloud's points read clearly, not full black.
const POINT_CLOUD_VIEW_BACKGROUND = 0x17242a;

// Every node from `node` up to (not including) `root`, `node` itself
// included.
function ancestorChain(node, root) {
  const chain = new Set();
  for (let current = node; current && current !== root; current = current.parent) {
    chain.add(current);
  }
  return chain;
}

// Point-cloud-only mode hides everything in the scene except the vehicle
// (which carries the point cloud as a child, see ensurePointCloudAttached)
// and whatever lights or views it.  AssetViewer.renderFromFiles loads the
// whole SDF world as a single object and adds that once, so "slash" sits
// nested inside it alongside the ground, bales, buckets and everything
// else -- it is not a sibling of them at the top of the scene.  Hiding
// top-level scene children would therefore hide the entire course,
// vehicle included, in one shot (and the point cloud with it, being a
// child of the now-invisible vehicle -- three.js does not render a visible
// object whose ancestor is not).  This instead walks up from the vehicle to
// the scene root, leaves every node on that walk (and the vehicle's own
// subtree) untouched, and hides only the branches that lead somewhere else.
function setPointCloudView(enabled) {
  pointCloudViewEnabled = enabled;
  pointCloud.object3D.visible = enabled;
  // Frames only go into this view while it is visible, so on the way in it
  // still holds whatever was current when it was last switched off -- catch
  // it up rather than showing a stale cloud until the next message lands.
  if (enabled && cloudBuffers.count > 0) {
    pointCloud.draw(cloudBuffers);
  }
  const scene = viewer["scene"];
  const slash = scene?.getByName("slash");
  const root = scene?.scene;

  if (root && slash) {
    if (enabled) {
      hiddenForPointCloudView = [];
      const keep = ancestorChain(slash, root);
      const hideExceptVehicle = (node) => {
        if (node === slash || node.isLight || node.isCamera) {
          return;
        }
        if (keep.has(node)) {
          node.children.forEach(hideExceptVehicle);
          return;
        }
        if (node.visible) {
          node.visible = false;
          hiddenForPointCloudView.push(node);
        }
      };
      root.children.forEach(hideExceptVehicle);
    } else {
      hiddenForPointCloudView.forEach((node) => { node.visible = true; });
      hiddenForPointCloudView = [];
    }
  }

  if (scene?.renderer) {
    if (enabled) {
      pointCloudViewClearColor = scene.renderer.getClearColor(new THREE.Color());
      scene.renderer.setClearColor(POINT_CLOUD_VIEW_BACKGROUND);
    } else if (pointCloudViewClearColor) {
      scene.renderer.setClearColor(pointCloudViewClearColor);
      pointCloudViewClearColor = null;
    }
  }

  // The subscription and the silent-topic warning both live at connect time
  // now, because the secondary viewport wants the cloud whether or not this
  // mode is on.  Nothing to arm here.
}

// The start signal's arms are the one other thing in either world that moves,
// and they move on a joint rather than by being teleported, so Gazebo reports
// them as a link pose within the model.  Looked up once and kept: the scene is
// several hundred objects and this runs on every pose message.
let signalArms;

function findSignalArms(scene) {
  if (signalArms === undefined) {
    const model = scene?.getByName("start_signal_arms");
    signalArms = model?.children.find((child) => child.name === "arms") ?? null;
  }
  return signalArms;
}

// The randomiser moves these, and nothing else in either world does.
const layoutPattern = /^(bucket|hoop)_\d+$/;
const layoutObjects = new Map();

function applyLayout(message) {
  const scene = viewer["scene"];
  for (const pose of message.pose) {
    if (!layoutPattern.test(pose.name)) {
      continue;
    }
    if (!layoutObjects.has(pose.name)) {
      layoutObjects.set(pose.name, scene?.getByName(pose.name) ?? null);
    }
    const object = layoutObjects.get(pose.name);
    if (object) {
      scene.setPose(object, pose.position, pose.orientation);
    }
  }
}

// Sampled over a connection of its own, once every few seconds, which looks
// wasteful and is the only thing that works.  Buckets and hoops are static
// models, so Gazebo leaves them out of the dynamic pose stream entirely; they
// are only in the whole-world one.  And the websocket server latches what it
// sends on that topic when a connection subscribes -- a long-lived
// subscription reports the layout that was there when the page opened, for
// ever, however often it resubscribes.  A connection opened fresh is always
// current.  A layout only changes when somebody calls the randomiser, so a
// few seconds late is not late.
function hasLayout() {
  let found = false;
  viewer["scene"]?.scene.traverse((object) => {
    found = found || layoutPattern.test(object.name || "");
  });
  return found;
}

let layoutSampleOpen = false;

function sampleLayout() {
  // One at a time.  Opening the connection, being sent the whole protobuf
  // dictionary and then a pose for every entity in the world takes a few
  // seconds on a loaded machine, which is longer than the interval.
  if (layoutSampleOpen || !poseType) {
    return;
  }
  layoutSampleOpen = true;
  const socket = new WebSocket(`${websocketProtocol}://${window.location.hostname}:9002`);
  const done = () => {
    layoutSampleOpen = false;
    clearTimeout(giveUp);
    socket.close();
  };
  const giveUp = setTimeout(done, LAYOUT_SAMPLE_TIMEOUT_MS);
  let subscribed = false;
  socket.addEventListener("open", () => socket.send("protos,,,"));
  socket.addEventListener("error", done);
  socket.addEventListener("close", () => { layoutSampleOpen = false; });
  socket.addEventListener("message", async ({ data }) => {
    // The first message is the dictionary, which the live connection has
    // already parsed for us; this one only has to know it has arrived.
    if (!subscribed) {
      subscribed = true;
      socket.send(`sub,${layoutTopic},,`);
      return;
    }
    const message = splitFrame(new Uint8Array(await data.arrayBuffer()));
    if (!message || message.topic !== layoutTopic) {
      return;
    }
    applyLayout(poseType.decode(message.payload));
    done();
  });
}

function updatePoses(message) {
  const scene = viewer["scene"];
  const slashPose = message.pose.find((pose) => pose.name === "slash");
  const slash = scene?.getByName("slash");
  if (slashPose) {
    updateSpeed(message, slashPose);
  }
  if (slashPose && slash) {
    latestSlashPose = slashPose;
    scene.setPose(slash, slashPose.position, slashPose.orientation);
    ensurePointCloudAttached(scene, slash);
    if (followSlash) {
      if (followTargetOffset) {
        updateSlashFollowView(slashPose);
      } else {
        startSlashFollowView(slashPose);
      }
    }
  }

  // Link poses come through relative to their model, which is what three.js
  // wants for a child object, so this needs no conversion.
  const armsPose = message.pose.find((pose) => pose.name === "arms");
  const arms = findSignalArms(scene);
  if (armsPose && arms) {
    scene.setPose(arms, armsPose.position, armsPose.orientation);
  }
}

function getTeleportPose() {
  const [xInput, yInput, headingInput] = positionInputs;
  const x = Number(xInput.value);
  const y = Number(yInput.value);
  const heading = Number(headingInput.value);
  if (![x, y, heading].every(Number.isFinite)) {
    return null;
  }

  const halfHeading = (heading * Math.PI) / 360;
  return {
    name: "slash",
    id: latestSlashPose?.id,
    position: { x, y, z: latestSlashPose?.position.z ?? 0.15 },
    orientation: { x: 0, y: 0, z: Math.sin(halfHeading), w: Math.cos(halfHeading) },
  };
}

function captureRobotPosition() {
  if (!latestSlashPose) {
    status.textContent = "Robot pose unavailable";
    return;
  }

  const [xInput, yInput, headingInput] = positionInputs;
  xInput.value = latestSlashPose.position.x.toFixed(2);
  yInput.value = latestSlashPose.position.y.toFixed(2);
  headingInput.value = ((getSlashYaw(latestSlashPose) * 180) / Math.PI).toFixed(1);
  teleportButton.disabled = true;
  status.textContent = "Robot position captured";
}

function previewTeleportPosition() {
  const pose = getTeleportPose();
  const scene = viewer["scene"];
  const slash = scene?.getByName("slash");
  if (!pose || !scene || !slash) {
    status.textContent = "Robot pose unavailable";
    return;
  }

  if (!teleportPreview) {
    teleportPreview = slash.clone(true);
    teleportPreview.name = "teleport-preview";
    teleportPreview.traverse((object) => {
      if (!object.material) {
        return;
      }
      object.material = object.material.clone();
      object.material.transparent = true;
      object.material.opacity = 0.38;
      object.material.depthWrite = false;
    });
    scene.scene.add(teleportPreview);
  }
  scene.setPose(teleportPreview, pose.position, pose.orientation);
  teleportButton.disabled = false;
  status.textContent = "Teleport preview ready";
}

function splitFrame(frame) {
  const commas = [];
  for (let index = 0; index < frame.length && commas.length < 3; index += 1) {
    if (frame[index] === 44) {
      commas.push(index);
    }
  }
  if (commas.length !== 3) {
    return null;
  }

  return {
    operation: textDecoder.decode(frame.slice(0, commas[0])),
    topic: textDecoder.decode(frame.slice(commas[0] + 1, commas[1])),
    type: textDecoder.decode(frame.slice(commas[1] + 1, commas[2])),
    payload: frame.slice(commas[2] + 1),
  };
}

function sendRequest(service, type, payload) {
  const header = textEncoder.encode(`req,${service},${type},`);
  const frame = new Uint8Array(header.length + payload.length);
  frame.set(header);
  frame.set(payload, header.length);
  simulationSocket.send(frame);
}

function resetRobot() {
  if (simulationSocket?.readyState !== WebSocket.OPEN || !worldControlType) {
    return;
  }

  resetRobotButton.disabled = true;
  status.textContent = "Resetting robot";
  // The car is about to jump back to spawn; drop the peak and the differencing
  // history so neither the teleport nor the old run's peak shows on the dial.
  previousSpeedSample = undefined;
  filteredSpeed = 0;
  speedometer.reset();
  steeringDial.reset();
  const request = worldControlType.encode({ reset: { all: true } }).finish();
  sendRequest(worldControlService, "gz.msgs.WorldControl", request);
}

function teleportRobot() {
  const pose = getTeleportPose();
  if (!pose) {
    return;
  }

  teleportButton.disabled = true;
  status.textContent = "Teleporting robot";
  fetch(teleportApiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      x: pose.position.x,
      y: pose.position.y,
      heading: Number(positionInputs[2].value),
    }),
  })
    .then(async (response) => {
      const result = await response.json();
      if (!response.ok || !result.success) {
        throw new Error(result.message || "Teleport failed");
      }
      status.textContent = "Robot teleported";
      teleportButton.disabled = false;
    })
    .catch((error) => {
      status.textContent = error.message;
      teleportButton.disabled = false;
    });
}

async function toggleStartSignal() {
  signalButton.disabled = true;
  const go = !signalGo;
  status.textContent = go ? "Turning signal green" : "Turning signal red";
  try {
    const response = await fetch(signalApiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ go }),
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
      throw new Error(result.message || "Signal command failed");
    }
    signalGo = go;
    signalButton.textContent = go ? "Set signal: Stop" : "Set signal: Go";
    status.textContent = go ? "Signal turning green" : "Signal turning red";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    signalButton.disabled = simulationSocket?.readyState !== WebSocket.OPEN;
  }
}

function connectPoseStream() {
  const socket = new WebSocket(`${websocketProtocol}://${window.location.hostname}:9002`);
  simulationSocket = socket;

  socket.addEventListener("open", () => socket.send("protos,,,"));
  socket.addEventListener("close", () => {
    status.textContent = "Simulation disconnected";
    statusIndicator.classList.remove("ready");
    resetRobotButton.disabled = true;
    signalButton.disabled = true;
    capturePoseButton.disabled = true;
    previewButton.disabled = true;
    teleportButton.disabled = true;
  });
  socket.addEventListener("error", () => {
    status.textContent = "Unable to connect to simulation";
    statusIndicator.classList.remove("ready");
    resetRobotButton.disabled = true;
    signalButton.disabled = true;
    capturePoseButton.disabled = true;
    previewButton.disabled = true;
    teleportButton.disabled = true;
  });
  socket.addEventListener("message", async ({ data }) => {
    if (!poseType) {
      const definitions = await data.text();
      const root = parse(definitions, { keepCase: true }).root;
      poseType = root.lookupType("gz.msgs.Pose_V");
      worldControlType = root.lookupType("gz.msgs.WorldControl");
      booleanType = root.lookupType("gz.msgs.Boolean");
      cmdVelType = root.lookupType("gz.msgs.Twist");
      pointCloudType = root.lookupType("gz.msgs.PointCloudPacked");
      // The server's bundle has gz.msgs.Image but not the PixelFormatType
      // enum it refers to (its own .proto), so without this every depth
      // frame throws on decode.  Values from gz-msgs10 image.proto.
      if (!root.lookup("gz.msgs.PixelFormatType")) {
        root.lookup("gz.msgs").add(new Enum("PixelFormatType", PIXEL_FORMAT_TYPE));
      }
      imageType = root.lookupType("gz.msgs.Image");
      cameraInfoType = root.lookupType("gz.msgs.CameraInfo");
      socket.send(`sub,${poseTopic},,`);
      socket.send(`sub,${cmdVelTopic},,`);
      // Subscribed here rather than when point-cloud-view is toggled, because
      // the secondary viewport shows the cloud all the time now.  It is a
      // 640x360 stream at ~5 MB a frame, which is the price of that viewport
      // being useful without being armed first.
      socket.send(`sub,${pointCloudTopic},,`);
      pointCloudSubscribed = true;
      pointCloudSilenceTimer = setTimeout(() => {
        if (pointCloudFrames === 0) {
          const why =
            "No point cloud on " + pointCloudTopic +
            " -- relaunch the simulation with sensors:=true";
          cloudView.setHint(why);
          status.textContent = why;
        }
      }, POINT_CLOUD_SILENCE_MS);
      // Only worth doing where something varies.  The speed course has no
      // buckets or hoops, and sampling a layout it does not have would open a
      // connection every few seconds to learn nothing.
      if (hasLayout()) {
        sampleLayout();
        setInterval(sampleLayout, LAYOUT_SAMPLE_MS);
      }
      status.textContent = "Live simulation connected";
      statusIndicator.classList.add("ready");
      resetRobotButton.disabled = false;
      signalButton.disabled = false;
      capturePoseButton.disabled = false;
      previewButton.disabled = false;
      // Gated on the same connection as the rest, not just resourceLoaded$:
      // setPointCloudView needs to find "slash" by name in the loaded scene,
      // which is only guaranteed to have finished by the time this fires.
      pointCloudButton.disabled = false;
      segmentationButton.disabled = false;
      depthButton.disabled = false;
      if (depthViewEnabled) {
        subscribeDepth(true);
      }
      positionInputs.forEach((input) => { input.disabled = false; });
      return;
    }

    const frame = new Uint8Array(await data.arrayBuffer());
    const message = splitFrame(frame);
    if (!message) {
      return;
    }
    if (message.operation === "req" && message.topic === worldControlService) {
      const response = booleanType.decode(message.payload);
      status.textContent = response.data ? "Robot reset" : "Robot reset failed";
      resetRobotButton.disabled = false;
      return;
    }
    if (message.topic === poseTopic) {
      updatePoses(poseType.decode(message.payload));
      return;
    }
    if (message.topic === cmdVelTopic) {
      updateCommandedSteering(cmdVelType.decode(message.payload));
      return;
    }
    if (message.topic === depthInfoTopic) {
      depthView.setInfo(cameraInfoType.decode(message.payload));
      return;
    }
    if (message.topic === depthImageTopic) {
      if (depthViewEnabled) {
        clearTimeout(depthSilenceTimer);
        // Said on the panel: an exception here would otherwise vanish into
        // the websocket handler and leave the panel waiting forever.
        try {
          depthView.draw(imageType.decode(message.payload), latestSlashPose);
        } catch (error) {
          depthView.setHint(`Depth frame arrived but could not be drawn: ${error.message}`);
        }
      }
      return;
    }
    const cloudTopic = segmentationMode ? segmentedTopic : pointCloudTopic;
    if (message.topic === pointCloudTopic || message.topic === segmentedTopic) {
      if (message.topic !== cloudTopic) {
        return;
      }
      // Unpacked ONCE and handed to both views: at ~5 MB and 230400 points a
      // frame, doing it twice is the most expensive thing this page could do
      // per message.  Both the protobuf decode and the point unpack happen
      // here; the views only wrap attributes over the result.
      const cloud = pointCloudType.decode(message.payload);
      if (!cloudBuffers.ingest(cloud)) {
        return;
      }
      if (pointCloudFrames === 0) {
        clearTimeout(pointCloudSilenceTimer);
      }
      pointCloudFrames += 1;
      cloudView.draw(cloudBuffers);
      // The main view's copy is only refreshed while it is actually on
      // screen; setPointCloudView draws the latest frame on the way in.
      if (pointCloudViewEnabled) {
        pointCloud.draw(cloudBuffers);
      }
    }
  });
}

viewer.resourceLoaded$.subscribe((loaded) => {
  if (loaded) {
    viewer["scene"].controls.addEventListener("change", captureFollowView);
    showCourseOverview();
    status.textContent = "Connecting to simulation";
    connectPoseStream();
  }
});

// gzweb 3.0.2 cannot load a binary STL over HTTP, and fails silently when it
// tries: its STLLoader.parse does `new DataView(data.buffer, data.byteOffset)`
// on what FileLoader hands it, which is a plain ArrayBuffer with no `.buffer`,
// so every mesh throws `First argument to DataView constructor must be an
// ArrayBuffer` into an onError that does nothing.  Its parse also returns a
// Mesh where the caller expects a BufferGeometry, so even a parse that
// succeeded would be wrapped into a second, empty Mesh.  Both are patched
// here rather than in the course generators: Gazebo itself reads these STLs
// correctly, so the meshes are not what is wrong.
function patchStlLoader(scene) {
  const loader = scene?.stlLoader;
  if (!loader || loader.parseFixed) {
    return;
  }
  const parse = loader.parse.bind(loader);
  loader.parse = (data) => {
    const parsed = parse(data instanceof ArrayBuffer ? new Uint8Array(data) : data);
    return parsed?.isMesh ? parsed.geometry : parsed;
  };
  loader.parseFixed = true;
}

patchStlLoader(viewer["scene"]);
// The straw bales are drawn instanced (bales.js), not by gzweb, which made a
// mesh, a material and a texture upload per bale -- see that file.
// ?bales=gzweb draws them the old way, for comparison.
if (new URLSearchParams(window.location.search).get("bales") !== "gzweb") {
  installInstancedBales(viewer, worldUrl, assetUrls);
}
if (import.meta.env.DEV) {
  window.cfrViewer = viewer; // for poking at the scene from the console
}

// "Loose straw": the strands around each bale, in this view only -- the
// simulator's camera sees them unless it was launched with strands:=false.
// Remembered per browser; storage can be missing or refuse, which just means
// the default (shown).
const strandsCheckbox = document.querySelector("#show-strands");
const STRANDS_KEY = "cfr-gzweb-show-strands";
try {
  strandsCheckbox.checked = window.localStorage.getItem(STRANDS_KEY) !== "false";
} catch {
  strandsCheckbox.checked = true;
}
const applyStrands = () => {
  setStrandsVisible(viewer, strandsCheckbox.checked);
  try {
    window.localStorage.setItem(STRANDS_KEY, String(strandsCheckbox.checked));
  } catch {
    // Not remembered; still applied.
  }
};
strandsCheckbox.addEventListener("change", applyStrands);
applyStrands();
// gzweb adds its own strand visuals (the movable bales) once the world has
// loaded, after the line above ran.
viewer.resourceLoaded$.subscribe((loaded) => {
  if (loaded) {
    setStrandsVisible(viewer, strandsCheckbox.checked);
  }
});
viewer.renderFromFiles([worldUrl, ...assetUrls]);
window.addEventListener("resize", () => viewer.resize());
followButton.addEventListener("click", () => {
  followSlash = !followSlash;
  followButton.textContent = followSlash ? "Stop following" : "Follow robot";
  if (followSlash) {
    startSlashFollowView(latestSlashPose);
  } else {
    const scene = viewer["scene"];
    if (scene) {
      scene.camera.up.set(0, 0, 1);
      scene.controls.update();
    }
  }
});
document.querySelector("#reset-view").addEventListener("click", () => {
  followSlash = false;
  followTargetOffset = undefined;
  followCameraOffset = undefined;
  followYaw = undefined;
  followButton.textContent = "Follow robot";
  showCourseOverview();
});
resetRobotButton.addEventListener("click", resetRobot);
signalButton.addEventListener("click", toggleStartSignal);
segmentationButton.addEventListener("click", () => {
  segmentationMode = !segmentationMode;
  if (segmentationMode && !segmentedSubscribed && simulationSocket) {
    simulationSocket.send(`sub,${segmentedTopic},,`);
    segmentedSubscribed = true;
  }
  segmentationButton.textContent = segmentationMode ? "Color by camera" : "Color by class";
  segmentationLegend.hidden = !segmentationMode;
});
// Subscribed only while shown: it is the heaviest stream on the page.
function subscribeDepth(on) {
  if (simulationSocket?.readyState !== WebSocket.OPEN) {
    return;
  }
  if (on) {
    simulationSocket.send(`sub,${depthInfoTopic},,`);
    simulationSocket.send(`throttle,${depthInfoTopic},na,1`);
    simulationSocket.send(`sub,${depthImageTopic},,`);
    simulationSocket.send(`throttle,${depthImageTopic},na,${DEPTH_VIEW_HZ}`);
    clearTimeout(depthSilenceTimer);
    depthSilenceTimer = setTimeout(() => {
      depthView.setHint(
        `No depth image on ${depthImageTopic} -- relaunch the simulation with sensors:=true`,
      );
    }, POINT_CLOUD_SILENCE_MS);
  } else {
    simulationSocket.send(`unsub,${depthImageTopic},,`);
    simulationSocket.send(`unsub,${depthInfoTopic},,`);
    clearTimeout(depthSilenceTimer);
  }
}
depthButton.addEventListener("click", () => {
  depthViewEnabled = !depthViewEnabled;
  depthWrap.hidden = !depthViewEnabled;
  depthButton.textContent = depthViewEnabled ? "Hide depth" : "Policy depth";
  subscribeDepth(depthViewEnabled);
  if (depthViewEnabled) {
    depthWrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
pointCloudButton.addEventListener("click", () => {
  setPointCloudView(!pointCloudViewEnabled);
  pointCloudButton.textContent = pointCloudViewEnabled ? "Show full scene" : "Point cloud only";
});
capturePoseButton.addEventListener("click", captureRobotPosition);
previewButton.addEventListener("click", previewTeleportPosition);
teleportButton.addEventListener("click", teleportRobot);
positionInputs.forEach((input) => input.addEventListener("input", () => {
  teleportButton.disabled = true;
}));
