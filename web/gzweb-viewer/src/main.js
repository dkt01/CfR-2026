import { AssetViewer } from "gzweb";
import { parse } from "protobufjs";
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
const LAYOUT_SAMPLE_MS = 3000;
const LAYOUT_SAMPLE_TIMEOUT_MS = 15000;
const worldControlService = `/world/${course.world}/control`;
const teleportApiUrl = `http://${window.location.hostname}:9003/api/sim/teleport`;
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
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
let followSlash = false;
let latestSlashPose;
let followTargetOffset;
let followCameraOffset;
let updatingFollowView = false;
let simulationSocket;
let worldControlType;
let booleanType;
let teleportPreview;
// Parsed once from the live connection's dictionary, and used by the layout
// samples as well, which is why it is not local to connectPoseStream.
let poseType;
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

function captureFollowView() {
  const scene = viewer["scene"];
  if (!followSlash || updatingFollowView || !latestSlashPose || !scene) {
    return;
  }

  const yaw = getSlashYaw(latestSlashPose);
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

  const targetOffset = rotateOffset(followTargetOffset, getSlashYaw(pose));
  const cameraOffset = rotateOffset(followCameraOffset, getSlashYaw(pose));
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
  if (slashPose && slash) {
    latestSlashPose = slashPose;
    scene.setPose(slash, slashPose.position, slashPose.orientation);
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

function connectPoseStream() {
  const socket = new WebSocket(`${websocketProtocol}://${window.location.hostname}:9002`);
  simulationSocket = socket;

  socket.addEventListener("open", () => socket.send("protos,,,"));
  socket.addEventListener("close", () => {
    status.textContent = "Simulation disconnected";
    statusIndicator.classList.remove("ready");
    resetRobotButton.disabled = true;
    capturePoseButton.disabled = true;
    previewButton.disabled = true;
    teleportButton.disabled = true;
  });
  socket.addEventListener("error", () => {
    status.textContent = "Unable to connect to simulation";
    statusIndicator.classList.remove("ready");
    resetRobotButton.disabled = true;
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
      socket.send(`sub,${poseTopic},,`);
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
      capturePoseButton.disabled = false;
      previewButton.disabled = false;
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
  }
});
document.querySelector("#reset-view").addEventListener("click", () => {
  followSlash = false;
  followTargetOffset = undefined;
  followCameraOffset = undefined;
  followButton.textContent = "Follow robot";
  showCourseOverview();
});
resetRobotButton.addEventListener("click", resetRobot);
capturePoseButton.addEventListener("click", captureRobotPosition);
previewButton.addEventListener("click", previewTeleportPosition);
teleportButton.addEventListener("click", teleportRobot);
positionInputs.forEach((input) => input.addEventListener("input", () => {
  teleportButton.disabled = true;
}));
