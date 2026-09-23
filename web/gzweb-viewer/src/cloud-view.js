// A small always-on chase view of the ZED cloud and the car that carries it.
//
// The main scene can show the cloud too, but only in "point cloud only" mode,
// which hides the course to do it -- so you could see the cloud OR where the
// car is, never both, and only while deliberately toggled.  This is the other
// half: a second viewport that is always live, always framed on the vehicle,
// and independent of whatever the main view is doing.
//
// It renders in the CAR'S OWN FRAME rather than in world coordinates, which is
// what makes it cheap: gz-sim's rgbd_camera already publishes points in the
// body convention (+x forward, +y left, +z up -- see pointcloud.js), so the
// cloud goes in at the origin, the car goes in at the origin, and neither has
// to be moved as the car drives.  The camera never moves either.  Nothing here
// subscribes to a pose.
//
// Its own renderer and its own requestAnimationFrame loop, deliberately:
// gzweb's AssetViewer owns the main canvas and its render loop, and borrowing
// them would couple this to the internals of a dependency.
//
// That loop draws on demand rather than every vsync.  Nothing in this scene
// moves on its own -- the camera is fixed, the car is a static likeness, and
// the only thing that ever changes is the cloud, which arrives at about 4 Hz.
// Redrawing 150k points 60 times a second to show 56 identical frames in a
// row was most of this page's GPU time.

import * as THREE from "three";

import { createPointCloud } from "./pointcloud.js";

// Traxxas Slash 4X4, from config/vehicle.yaml geometry.  Only a likeness --
// enough to read which way the car is pointing and how far the returns are.
const CHASSIS = { length: 0.55, width: 0.30, height: 0.12 };
const WHEELBASE = 0.324;
const TRACK = 0.29;
const WHEEL_RADIUS = 0.0566;
const WHEEL_WIDTH = 0.035;

const BACKGROUND = 0x11191d;

// The heading arrow on the car's roof.  Sized to sit inside the chassis
// footprint (0.55 x 0.30) once stretched.
const HEADING_MARKER_RADIUS = 0.105;
const HEADING_MARKER_COLOR = 0x2ee86a;

export function createCloudView(container) {
  if (!container) {
    return { draw() {}, frames: 0, setHint() {}, dispose() {} };
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BACKGROUND);

  // Z-up, to match the data and the rest of the project (REP-103).  three.js
  // defaults to Y-up, and every orbit/lookAt below depends on this being set
  // before the camera is aimed.
  const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 60);
  camera.up.set(0, 0, 1);
  camera.position.set(-1.5, 0, 1.0);
  camera.lookAt(3.0, 0, 0.1);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 0.6);
  key.position.set(-2, 1, 3);
  scene.add(key);

  // Ground reference. GridHelper lies in XZ, so it is rotated into XY to sit
  // on the floor of a Z-up world.
  const grid = new THREE.GridHelper(24, 48, 0x2f4f5a, 0x1e3038);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const car = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(CHASSIS.length, CHASSIS.width, CHASSIS.height),
    new THREE.MeshLambertMaterial({ color: 0xe0592a }),
  );
  body.position.set(0, 0, WHEEL_RADIUS + CHASSIS.height / 2);
  car.add(body);

  const wheelGeometry = new THREE.CylinderGeometry(
    WHEEL_RADIUS, WHEEL_RADIUS, WHEEL_WIDTH, 16,
  );
  const wheelMaterial = new THREE.MeshLambertMaterial({ color: 0x1b1b1b });
  for (const x of [WHEELBASE / 2, -WHEELBASE / 2]) {
    for (const y of [TRACK / 2, -TRACK / 2]) {
      const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
      // Cylinder's axis is +Y in three.js, which is already the wheel axle
      // direction in a Z-up body frame, so no rotation is needed.
      wheel.position.set(x, y, WHEEL_RADIUS);
      car.add(wheel);
    }
  }
  // Heading marker: a flat triangle laid on the roof, pointing the way the
  // car is facing, so "forward" is unambiguous at a glance.  CircleGeometry
  // with three segments puts its first vertex at angle 0, which is +x -- the
  // body frame's forward -- so it needs no rotation, only stretching along x
  // to read as an arrow rather than as a plain triangle.  Unlit
  // (MeshBasicMaterial) to keep it the same vivid green from every angle,
  // and double-sided so it is still there when the camera drops below it.
  const headingMarker = new THREE.Mesh(
    new THREE.CircleGeometry(HEADING_MARKER_RADIUS, 3),
    new THREE.MeshBasicMaterial({ color: HEADING_MARKER_COLOR, side: THREE.DoubleSide }),
  );
  headingMarker.scale.set(1.45, 0.95, 1);
  // Just clear of the roof: coplanar would z-fight with the chassis top.
  headingMarker.position.set(0.04, 0, WHEEL_RADIUS + CHASSIS.height + 0.004);
  car.add(headingMarker);
  scene.add(car);

  // Its own cloud instance rather than the main view's: a three.js object has
  // exactly one parent, so the same Points cannot appear in both scenes.
  const cloud = createPointCloud();
  cloud.object3D.visible = true;
  scene.add(cloud.object3D);

  let frames = 0;
  const hint = document.createElement("p");
  hint.className = "cloud-view-hint";
  hint.textContent = "waiting for the ZED point cloud";
  container.appendChild(hint);

  let needsRender = true;

  function resize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) {
      return;
    }
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    needsRender = true;
  }
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(container);

  let running = true;
  function tick() {
    if (!running) {
      return;
    }
    requestAnimationFrame(tick);
    if (!needsRender) {
      return;
    }
    needsRender = false;
    renderer.render(scene, camera);
  }
  tick();

  return {
    /** Draw the decoder's current frame; see createCloudBuffers. */
    draw(buffers) {
      cloud.draw(buffers);
      needsRender = true;
      if (frames === 0) {
        hint.remove();
      }
      frames += 1;
    },
    /** Frames seen, so the caller can report a topic that never publishes. */
    get frames() {
      return frames;
    },
    /** Replace the placeholder, e.g. to say why nothing is arriving. */
    setHint(text) {
      if (frames === 0) {
        hint.textContent = text;
      }
    },
    dispose() {
      running = false;
      observer.disconnect();
      renderer.dispose();
    },
  };
}
