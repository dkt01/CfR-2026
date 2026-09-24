import { AssetViewer } from "gzweb";
import { parse } from "protobufjs";
import * as THREE from "three";
import { createSpeedometer } from "./speedometer.js";
import { createSteeringDial, WHEELBASE, MIN_SPEED_FOR_STEERING } from "./steering-dial.js";
import { createCloudView } from "./cloud-view.js";
import { createCloudBuffers, createPointCloud } from "./pointcloud.js";
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

// The parser matches mesh URIs against this list by filename, so every mesh
// any world references has to be in it, not just the ones this course uses.
const assetUrls = Object.values(
  import.meta.glob("../../../jetson/cfr_arduino_bridge/meshes/*.stl", {
    eager: true,
    query: "?url",
    import: "default",
  }),
).map(absolute);
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
// The heading the CAMERA is using, which lags the car's - see advanceFollowYaw.
let followYaw;
let followYawStamp;
let simulationSocket;
let worldControlType;
let booleanType;
let cmdVelType;
let pointCloudType;
let teleportPreview;
// Parsed once from the live connection's dictionary, and used by the layout
// samples as well, which is why it is not local to connectPoseStream.
let poseType;
let pointCloudAttached = false;
let pointCloudSubscribed = false;
let pointCloudViewEnabled = false;
document.title = `CfR ${course.title} Viewer`;
document.querySelector("header h1").textContent = course.title;
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

function rotateOffset(offset, yaw) {
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  return {
    x: offset.x * cosine - offset.y * sine,
    y: offset.x * sine + offset.y * cosine,
    z: offset.z,
  };
}

// HOW FAST THE CAMERA IS ALLOWED TO SWING AROUND THE CAR.
//
// The follow camera used to be bolted rigidly to the car's heading, which is
// fine on a straight and unwatchable in a hairpin.  Measured over a scripted
// lap of the speed course: the car reaches 1.67 rad/s of yaw, which spins the
// whole scene at 96 deg/s and drags the camera - on a 4.9 m lever - past the
// car at 4.1 m/s while the car itself is doing 2.25 m/s.  The car never
// actually leaves the frame, because the target is locked to it; everything
// around it does, and that is what losing it in a hairpin looks like.
//
// So the TARGET still tracks the car's position exactly, and only the camera's
// orbit ANGLE is smoothed.  The cost is that the camera lags the car's heading,
// so a hairpin is watched from slightly outside the turn rather than from
// directly behind:
//
//   tau    cap        camera m/s in hairpins   scene spin   worst heading lag
//   none   -                        4.06         96 deg/s           0 deg
//   0.25   1.2 rad/s                3.00         69 deg/s          28 deg
//   0.30   1.2 rad/s                2.88         69 deg/s          31 deg  <--
//   0.35   1.0 rad/s                2.35         57 deg/s          54 deg
//
// 31 degrees still reads as a chase view, and outside the turn is where you
// want to watch an apex from anyway.  Past about 50 the car is being watched
// side-on and it stops looking like following at all.
const FOLLOW_YAW_TAU = 0.3;        // s, first-order lag on the camera heading
const FOLLOW_YAW_MAX_RATE = 1.2;   // rad/s, hard cap on top of it

function wrapToPi(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function advanceFollowYaw(pose, now) {
  const yaw = getSlashYaw(pose);
  if (followYaw === undefined) {
    followYaw = yaw;
    followYawStamp = now;
    return followYaw;
  }
  // Pose frames arrive irregularly - the websocket batches them and a hidden
  // tab throttles rendering - so the step comes off the clock rather than
  // being assumed constant, or the smoothing changes with the frame rate.
  // Capped at half a second so a stalled stream catches up over a few frames
  // instead of snapping.
  const dt = Math.min(Math.max((now - followYawStamp) / 1000, 0), 0.5);
  followYawStamp = now;
  if (dt === 0) {
    return followYaw;
  }
  // Shortest way round, so the camera does not unwind the long way when the
  // car's heading crosses +/-pi. That crossing happens in every hairpin.
  const step = wrapToPi(yaw - followYaw) * (1 - Math.exp(-dt / FOLLOW_YAW_TAU));
  const limit = FOLLOW_YAW_MAX_RATE * dt;
  followYaw = wrapToPi(followYaw + Math.max(-limit, Math.min(limit, step)));
  return followYaw;
}

function captureFollowView() {
  const scene = viewer["scene"];
  if (!followSlash || updatingFollowView || !latestSlashPose || !scene) {
    return;
  }

  // The camera's own heading, not the car's: these offsets are applied in
  // that frame, so capturing them in any other one rotates the view by the
  // difference the moment the next pose lands.  They are the same number
  // except while the smoothing is catching up - which is exactly when a
  // hairpin is being driven, and so exactly when someone grabs the mouse.
  const yaw = followYaw ?? getSlashYaw(latestSlashPose);
  const target = scene.controls.target;
  followTargetOffset = rotateOffset({
    x: target.x - latestSlashPose.position.x,
    y: target.y - latestSlashPose.position.y,
    z: target.z - latestSlashPose.position.z,
  }, -yaw);
  followCameraOffset = rotateOffset({
    x: scene.camera.position.x - target.x,
    y: scene.camera.position.y - target.y,
    z: scene.camera.position.z - target.z,
  }, -yaw);
}

function startSlashFollowView(pose) {
  const scene = viewer["scene"];
  if (!scene || !pose) {
    return;
  }

  const yaw = getSlashYaw(pose);
  followYaw = yaw;
  followYawStamp = performance.now();
  const forwardX = Math.cos(yaw);
  const forwardY = Math.sin(yaw);
  followTargetOffset = { x: 0.5, y: 0, z: 0.15 };
  followCameraOffset = { x: -3, y: 0, z: 3.85 };
  updatingFollowView = true;
  scene.camera.position.set(
    pose.position.x - 2.5 * forwardX,
    pose.position.y - 2.5 * forwardY,
    pose.position.z + 4,
  );
  scene.controls.target.set(
    pose.position.x + 0.5 * forwardX,
    pose.position.y + 0.5 * forwardY,
    pose.position.z + 0.15,
  );
  scene.camera.lookAt(scene.controls.target);
  scene.controls.update();
  updatingFollowView = false;
}

function updateSlashFollowView(pose) {
  const scene = viewer["scene"];
  if (!scene || !pose || !followTargetOffset || !followCameraOffset) {
    return;
  }

  const yaw = advanceFollowYaw(pose, performance.now());
  const targetOffset = rotateOffset(followTargetOffset, yaw);
  const cameraOffset = rotateOffset(followCameraOffset, yaw);
  const target = scene.controls.target;
  updatingFollowView = true;
  target.set(
    pose.position.x + targetOffset.x,
    pose.position.y + targetOffset.y,
    pose.position.z + targetOffset.z,
  );
  scene.camera.position.set(
    target.x + cameraOffset.x,
    target.y + cameraOffset.y,
    target.z + cameraOffset.z,
  );
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
      updateSlashFollowView(slashPose);
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
viewer.renderFromFiles([worldUrl, ...assetUrls]);
window.addEventListener("resize", () => viewer.resize());
followButton.addEventListener("click", () => {
  followSlash = !followSlash;
  followButton.textContent = followSlash ? "Stop following" : "Follow robot";
  if (followSlash) {
    startSlashFollowView(latestSlashPose);
  } else {
    // Dropped so the next follow starts from the car's real heading rather
    // than from wherever the camera had lagged to when it was switched off.
    followYaw = undefined;
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
