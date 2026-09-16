import { AssetViewer } from "gzweb";
import { parse } from "protobufjs";
import { createSpeedometer } from "./speedometer.js";
import "./style.css";

const status = document.querySelector("#viewer-status");
const statusIndicator = document.querySelector(".status span");
const websocketProtocol = window.location.protocol === "https:" ? "wss" : "ws";
const worldUrl = new URL("../../../jetson/cfr_arduino_bridge/worlds/speed_course.sdf", import.meta.url).href;
const poseTopic = "/world/cfr_speed_course/dynamic_pose/info";
const worldControlService = "/world/cfr_speed_course/control";
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
const speedometer = createSpeedometer(document.querySelector("#speedometer"));
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
let simulationSocket;
let worldControlType;
let booleanType;
let teleportPreview;
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

  scene.camera.position.set(20.5, -20, 25);
  scene.controls.target.set(20.5, 0, 0);
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

function updateSlashPose(message) {
  const slashPose = message.pose.find((pose) => pose.name === "slash");
  const scene = viewer["scene"];
  const slash = scene?.getByName("slash");
  if (slashPose) {
    updateSpeed(message, slashPose);
  }
  if (slashPose && slash) {
    latestSlashPose = slashPose;
    scene.setPose(slash, slashPose.position, slashPose.orientation);
    if (followSlash) {
      updateSlashFollowView(slashPose);
    }
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
  let poseType;
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
      updateSlashPose(poseType.decode(message.payload));
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

viewer.renderFromFiles([worldUrl]);
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
