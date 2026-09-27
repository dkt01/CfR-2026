// The straw bales -- the Speed Course's 202 and the Obstacle Course's 111
// course bales -- drawn as a handful of InstancedMeshes instead of by gzweb.
//
// gzweb builds every <visual> on its own: a new MeshStandardMaterial per
// visual, and its albedo and normal maps loaded again for each one (three's
// loader cache is off).  Each bale is two visuals -- the rounded body and its
// loose strands -- so the course was 404 meshes, 404 materials and ~400
// uploads of a 1024x400 texture, drawn one call at a time every frame.  That
// is what made the view stutter; the ~500k triangles were never the problem.
//
// Here the same visuals are lifted out of the world before gzweb sees it and
// drawn as one InstancedMesh per distinct mesh+texture: 3 body variants and 4
// strand variants, so 7 draw calls, 5 textures loaded once, and one material
// per group.  Everything comes from the world file itself -- poses, mesh
// URIs, textures, per-bale tint -- so regenerating the course
// (generate_speed_course.py) changes this view too, with nothing to keep in
// sync by hand.  Bales are static, so the instance matrices are set once.
//
// Only MESH visuals named bale_<n>_visual / bale_<n>_strands are taken; a
// world with box bales keeps them (taking those by name alone once removed
// them and drew nothing in their place).  The Obstacle Course's gap and Wide
// Section bales are their own models, named bale_visual, and move at run
// time, so they stay with gzweb, which follows their poses.

import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

const BALE_VISUAL = /^bale_\d+_(visual|strands)$/;

// The "Loose straw" checkbox.  Kept here so the instanced strands, which
// arrive after the page has set it, start out the way it says.
let strandsVisible = true;
const isStrands = (object) =>
  /_strands$/.test(object.name) || /^bales:bale_strands/.test(object.name);

// Show or hide every strand mesh in the scene: the instanced ones, and the
// ones gzweb draws itself on the Obstacle Course's movable bales.
export function setStrandsVisible(viewer, visible) {
  strandsVisible = visible;
  viewer.scene?.scene.traverse((object) => {
    if (isStrands(object)) {
      object.visible = visible;
    }
  });
}

// The URL a model:// URI's file is served at, from the viewer's asset list
// (gzweb matches by file name the same way).
function assetFor(uri, assetUrls) {
  const name = uri.split("/").pop();
  return assetUrls.find((url) => url.split("?")[0].split("/").pop() === name);
}

function text(node, selector) {
  return node.querySelector(selector)?.textContent.trim() ?? "";
}

// Pull the bale visuals out of the parsed world, returning what they were.
function extractBales(xml) {
  const bales = [];
  for (const visual of [...xml.getElementsByTagName("visual")]) {
    const name = visual.getAttribute("name") || "";
    if (!BALE_VISUAL.test(name) || !visual.querySelector("geometry mesh uri")) {
      continue;
    }
    // Only a visual's own <pose>, not one nested deeper.
    const poseNode = [...visual.children].find((child) => child.tagName === "pose");
    const pose = (poseNode?.textContent ?? "0 0 0 0 0 0").trim().split(/\s+/).map(Number);
    const diffuse = text(visual, "material > diffuse").split(/\s+/).map(Number);
    bales.push({
      mesh: text(visual, "geometry mesh uri"),
      albedo: text(visual, "albedo_map"),
      normal: text(visual, "normal_map"),
      roughness: Number(text(visual, "roughness") || 0.95),
      color: diffuse.length >= 3 ? diffuse.slice(0, 3) : [1, 1, 1],
      pose,
    });
    visual.remove();
  }
  return bales;
}

function loadGeometry(url) {
  if (url.split("?")[0].toLowerCase().endsWith(".stl")) {
    return new STLLoader().loadAsync(url);
  }
  return new OBJLoader().loadAsync(url).then((group) => {
    let geometry;
    group.traverse((child) => {
      if (!geometry && child.isMesh) {
        geometry = child.geometry;
      }
    });
    return geometry;
  });
}

// SDF pose (x y z roll pitch yaw, fixed-axis XYZ) as a matrix.
function poseMatrix([x, y, z, roll = 0, pitch = 0, yaw = 0]) {
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(roll, pitch, yaw, "ZYX"));
  return new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), q, new THREE.Vector3(1, 1, 1));
}

async function buildGroup(bales, assetUrls) {
  const group = new THREE.Group();
  group.name = "instanced_bales";
  const textures = new Map();
  const texture = (uri) => {
    if (!uri) {
      return null;
    }
    if (!textures.has(uri)) {
      const url = assetFor(uri, assetUrls);
      if (!url) {
        console.warn(`instanced bales: no asset for ${uri}`);
        textures.set(uri, null);
      } else {
        const map = new THREE.TextureLoader().load(url);
        map.wrapS = map.wrapT = THREE.RepeatWrapping;
        map.anisotropy = 4;
        textures.set(uri, map);
      }
    }
    return textures.get(uri);
  };

  // One instanced mesh per mesh + texture pair.
  const byKey = new Map();
  for (const bale of bales) {
    const key = `${bale.mesh}|${bale.albedo}`;
    if (!byKey.has(key)) {
      byKey.set(key, []);
    }
    byKey.get(key).push(bale);
  }
  await Promise.all(
    [...byKey.values()].map(async (members) => {
      const first = members[0];
      const url = assetFor(first.mesh, assetUrls);
      if (!url) {
        console.warn(`instanced bales: no asset for ${first.mesh}`);
        return;
      }
      const geometry = await loadGeometry(url);
      if (!geometry.attributes.normal) {
        geometry.computeVertexNormals();
      }
      const material = new THREE.MeshStandardMaterial({
        color: 0xffffff, // per-instance color carries the world's <diffuse>
        map: texture(first.albedo),
        normalMap: texture(first.normal),
        roughness: first.roughness,
        metalness: 0,
      });
      const mesh = new THREE.InstancedMesh(geometry, material, members.length);
      mesh.name = `bales:${first.mesh.split("/").pop()}`;
      members.forEach((bale, i) => {
        mesh.setMatrixAt(i, poseMatrix(bale.pose));
        mesh.setColorAt(i, new THREE.Color(...bale.color));
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
      // three culls an InstancedMesh by its GEOMETRY's bounds, which is one
      // bale at the origin -- the whole course would vanish whenever the
      // origin left the frame.  202 static bales are cheap to never cull.
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      // The bodies cast shadows as gzweb drew them; the strands do not (the
      // world turns theirs off too).  Instanced, a shadow pass is one more
      // call per group rather than one per bale.
      const strands = first.mesh.includes("strands");
      mesh.castShadow = !strands;
      mesh.receiveShadow = true;
      if (strands) {
        mesh.visible = strandsVisible;
      }
      group.add(mesh);
    }),
  );
  return group;
}

// Route the world load through here: strip the bale visuals, let gzweb build
// the rest, and add the instanced bales to its scene.  Call before
// viewer.renderFromFiles.
export function installInstancedBales(viewer, worldUrl, assetUrls) {
  const parser = viewer.sdfParser;
  if (!parser || parser.instancedBales) {
    return;
  }
  parser.instancedBales = true;
  const fileFromUrl = parser.fileFromUrl.bind(parser);
  parser.fileFromUrl = (url, callback) =>
    fileFromUrl(url, (xml) => {
      if (xml && url === worldUrl) {
        const bales = extractBales(xml);
        if (bales.length) {
          buildGroup(bales, assetUrls)
            .then((group) => viewer.scene?.scene.add(group))
            .catch((error) => console.error("instanced bales failed", error));
        }
      }
      callback(xml);
    });
}
