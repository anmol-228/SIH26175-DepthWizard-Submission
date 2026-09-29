import * as THREE from "three";
import { GLTFLoader } from "./vendor/loaders/GLTFLoader.js";
import { OrbitControls } from "./vendor/controls/OrbitControls.js";
import {
  surfaceMetaUrlFromGlbUrl,
  elevationUrlFromSurfaceMetaUrl,
  validateSurfaceMeta,
  resolveHorizontalScale,
  metricTifUrlFromSidecar,
} from "./sidecar.js";

const hudMode = document.getElementById("hud-mode");
const hudLoad = document.getElementById("hud-load");
const hudTris = document.getElementById("hud-tris");
const hudFps = document.getElementById("hud-fps");
const hudTier = document.getElementById("hud-tier");
const hudVexag = document.getElementById("hud-vexag");
const branchBanner = document.getElementById("branch-banner");
const sidecarStatus = document.getElementById("sidecar-status");
const placeholderBadge = document.getElementById("placeholder-badge");
const pickValue = document.getElementById("pick-value");
const pickWarning = document.getElementById("pick-warning");
const measurePanel = document.getElementById("measure-panel");
const mDist = document.getElementById("m-dist");
const mDh = document.getElementById("m-dh");
const mSlope = document.getElementById("m-slope");
const profileCanvas = document.getElementById("profile-canvas");
const referenceFileInput = document.getElementById("reference-file-input");
const referenceStatus = document.getElementById("reference-status");
const referenceResult = document.getElementById("reference-result");
const refEstimated = document.getElementById("ref-estimated");
const refReference = document.getElementById("ref-reference");
const refDelta = document.getElementById("ref-delta");

const uploadFileInput = document.getElementById("upload-file-input");
const uploadStatus = document.getElementById("upload-status");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x10161f);

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.01,
  10000
);
camera.position.set(20, 20, 30);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById("canvas-wrap").appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(30, 50, 20);
scene.add(sun);

let grid = new THREE.GridHelper(200, 40, 0x2a3a4c, 0x1a2634);
scene.add(grid);

// -- Orbit controls (default navigation mode) --
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 0, 0);
orbit.enableDamping = true;

// Real mesh reference (set once loaded, by loadMesh() below) + scale info,
// used by the fly-mode clamp and by DW-S1-09's raycasting/picking.
let pickableMesh = null;
let sceneBounds = { center: new THREE.Vector3(0, 0, 0), maxDim: 50 };

// -- WASD / pointer fly stub (toggled with 'F') --
let flyMode = false;
const flyKeys = { w: false, a: false, s: false, d: false, q: false, e: false };
let flySpeed = 12; // units/sec -- rescaled to mesh size in loadMesh() below
let dragging = false;
let lastPointer = { x: 0, y: 0 };
const yawPitch = { yaw: 0, pitch: 0 };

// Sphere-distance clamp: keeps the camera from getting too close to the
// mesh's overall bounding volume in ANY direction (including straight down,
// straight at a wall, etc). This is the ONLY camera-bounds check in fly mode
// -- verified via automated headless-browser testing (Nirmalya, PR #36
// review): wheel-zoom-in via OrbitControls.minDistance (the same clamp
// value/math) converged exactly to maxDim * 0.15 with no drift, confirming
// this clamp does real, correct work in every direction tested.
//
// A separate downward-raycast "terrain height" clamp was tried and removed:
// heightfield_to_mesh() (the frozen pipeline_to_3d_v1 producer) builds
// vertices as x=col*gsd, y=-row*gsd, z=elevation -- elevation lives in Z,
// not Y, so a raycast fired down -Y wasn't checking real ground clearance.
// Whether the scene should treat Z as the elevation/up axis is a real
// coordinate-frame decision, raised as an open question for whoever owns
// the scene-orientation call, not resolved here. Until that's decided,
// this sphere clamp is the sole verified camera-bounds protection.
function clampDistanceFromMesh() {
  const offset = camera.position.clone().sub(sceneBounds.center);
  const dist = offset.length();
  const minD = sceneBounds.maxDim * 0.15;
  if (dist < minD) {
    camera.position.copy(sceneBounds.center).add(offset.normalize().multiplyScalar(minD));
  }
}

function setMode(fly) {
  if (fly) {
    // Entering fly mode: sync yaw/pitch to the camera's current orbit-facing
    // direction so there's no orientation jump-cut the instant F is pressed.
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    yawPitch.yaw = Math.atan2(-dir.x, -dir.z);
    yawPitch.pitch = Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1));
  }
  flyMode = fly;
  orbit.enabled = !fly;
  hudMode.textContent = fly ? "fly (WASD + drag)" : "orbit";
}

// -- Unit / branch banner (DW-S1-08 Day-3; IF-1 / DW-WP2) --
// Keyed off the loaded sidecar's "branch" field, never inferred. Called only after a
// sidecar has already passed validateSurfaceMeta() - showSidecarFailure() below is the
// path for "no sidecar at all" / "sidecar invalid", kept visually and semantically
// distinct from "sidecar present, branch is relative".
function applyBranchBanner(meta) {
  if (!branchBanner) return;
  if (meta.branch === "relative") {
    branchBanner.textContent = "Uncalibrated surface -- relative units only, not metres";
    branchBanner.classList.add("show");
  } else {
    branchBanner.classList.remove("show");
  }
}

function clearBranchBanner() {
  if (branchBanner) branchBanner.classList.remove("show");
}

// -- IF-1 failure contract (frozen #26, 2026-09-06): missing/invalid metadata,
// unsupported schema, invalid payload dimensions/type, or invalid branch means the
// viewer stays view-only but measurements are disabled AND the UI visibly explains
// why -- never a silent no-op, and never a fall-through to placeholder data. --
function showSidecarFailure(reason) {
  if (sidecarStatus) {
    sidecarStatus.textContent = `No measurement metadata for this mesh: ${reason}`;
    sidecarStatus.classList.add("show");
  }
  clearBranchBanner();
  if (placeholderBadge) placeholderBadge.classList.remove("show");
}

function clearSidecarFailure() {
  if (sidecarStatus) sidecarStatus.classList.remove("show");
}

// -- Mesh / LOD tier handling (DW-S1-08 Day-3) --
// Multi-tier LOD generation itself is out of scope for this viewer (per
// issue #26: a real multi-tier LOD ladder is separate pipeline-side work,
// done by K3_mesh_pipeline.py). This viewer's job is to load and switch
// between whatever tiers already exist, cleanly.
//
// Only "placeholder" is a real file on disk right now. Add real entries
// here once native/medium/small LOD .glb files are generated locally; no
// other code needs to change. Press 'L' to cycle tiers.
const meshTiers = [
  { name: "placeholder", url: "./placeholder_relative.glb" },
  // { name: "native",  url: "./large_area26.glb" },
  // { name: "lod",     url: "./large_LOD_area26.glb" },
];
let tierIndex = 0;

// -- Parameterized loading (DW-S1-08 Day-3, Nirmalya's PR #29 review note) --
// Real run_pipeline.py (#24) output isn't wired up and no run_manifest.json
// exists to build against, so this uses ?glb=<path> rather than inventing a
// manifest schema that might not match what #24 eventually produces. Falls
// back to the placeholder tier list when no param is given.
const urlParams = new URLSearchParams(window.location.search);
const glbParam = urlParams.get("glb");
if (glbParam) {
  meshTiers.unshift({ name: "param:" + glbParam, url: glbParam });
}

function disposeMesh(root) {
  if (!root) return;
  root.traverse((obj) => {
    if (obj.isMesh) {
      obj.geometry?.dispose();
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach((m) => {
        if (!m) return;
        Object.values(m).forEach((v) => {
          if (v && v.isTexture) v.dispose();
        });
        m.dispose();
      });
    }
  });
  scene.remove(root);
}

let currentMesh = null;
let currentGlbUrl = null; // P0-A2: needed to re-derive this mesh's sidecar URL for metricTifUrlFromSidecar()

// #87: monotonic generation token guarding loadMesh()'s two independent async callbacks
// (the glb loader.load() and the separate loadSidecar() fetch) against a stale response
// arriving after a newer loadMesh() call has already started -- without this, rapid
// tier-cycling (L pressed twice quickly) or an upload completing mid-flight could let an
// old callback dispose the newer mesh, or overwrite newer sidecar metadata with older data.
let loadGeneration = 0;
const loader = new GLTFLoader();

// -- DW-S1-09: height-on-click + two-point profile/slope (PR #33) --
// -- DW-WP2 / IF-1 (issues #26/#27): sidecar is resolved PER mesh, from the glb URL
// itself, and loaded fresh every time loadMesh() loads a new tier -- see loadSidecar()
// below. surfaceMeta/elevFlat are only ever set together, from a sidecar that has
// already passed validateSurfaceMeta(); a failed/missing sidecar clears both and shows
// showSidecarFailure(), never leaves stale data from a previously-loaded mesh in place.
let surfaceMeta = null;
let elevFlat = null; // flat Float32Array, row-major (H*W) - index as elevFlat[row*W+col]
let measureMode = false;
const measurePoints = []; // { row, col, world: THREE.Vector3, elev }
const measureGroup = new THREE.Group();
scene.add(measureGroup);
const raycaster = new THREE.Raycaster();

// -- P0-A2: reference-DSM comparison --
// A separate interaction mode from measureMode ('M') -- toggled with 'R'. Only
// meaningful on the metric branch (see compareAtClick below); a reference .tif must
// be uploaded to the server first via the file input, which returns a server-side
// path used in subsequent /api/reference-sample calls.
let referenceTifPath = null;
let referenceMode = false;

function measurementReady() {
  return surfaceMeta !== null && elevFlat !== null;
}

/**
 * Fetch + validate one mesh's viewer_input_v1 sidecar and its raw elevation payload.
 * Resolves to {meta, elevFlat} on success, or null on ANY failure (missing sidecar,
 * network error, malformed JSON, contract validation failure, missing/malformed
 * elevation payload, or a byte length that does not match H*W*4). The caller is
 * responsible for calling showSidecarFailure() with a reason on null - this function
 * never itself falls back to a placeholder or a previous mesh's data.
 */
async function loadSidecar(glbUrl) {
  const surfaceMetaUrl = surfaceMetaUrlFromGlbUrl(glbUrl);
  let meta;
  try {
    const res = await fetch(surfaceMetaUrl, { cache: "no-store" });
    // cache: "no-store" matters here -- without it, a 304 Not Modified response
    // (normal caching, not an error) still needs `res.ok` to correctly read true;
    // no-store sidesteps relying on that ambiguity entirely for a metadata file this
    // small.
    if (!res.ok) {
      return { meta: null, elevFlat: null, reason: `no sidecar at ${surfaceMetaUrl} (HTTP ${res.status})` };
    }
    meta = await res.json();
  } catch (err) {
    return { meta: null, elevFlat: null, reason: `could not fetch/parse ${surfaceMetaUrl}: ${err.message}` };
  }

  const validation = validateSurfaceMeta(meta);
  if (!validation.ok) {
    return { meta: null, elevFlat: null, reason: validation.reason };
  }

  const elevationUrl = elevationUrlFromSurfaceMetaUrl(surfaceMetaUrl, meta.elevation_payload);
  let buffer;
  try {
    const res = await fetch(elevationUrl, { cache: "no-store" });
    if (!res.ok) {
      return { meta: null, elevFlat: null, reason: `elevation payload missing at ${elevationUrl} (HTTP ${res.status})` };
    }
    buffer = await res.arrayBuffer();
  } catch (err) {
    return { meta: null, elevFlat: null, reason: `could not fetch ${elevationUrl}: ${err.message}` };
  }

  const expectedBytes = meta.H * meta.W * 4;
  if (buffer.byteLength !== expectedBytes) {
    return {
      meta: null, elevFlat: null,
      reason: `elevation payload size mismatch: expected ${expectedBytes} bytes for ${meta.H}x${meta.W} float32, got ${buffer.byteLength}`,
    };
  }

  // Explicit little-endian read via DataView, per IF-1 - never a bare `new
  // Float32Array(buffer)` cast, which silently depends on the client's native byte
  // order rather than the frozen "byte_order: little" contract.
  const view = new DataView(buffer);
  const flat = new Float32Array(meta.H * meta.W);
  for (let i = 0; i < flat.length; i++) {
    flat[i] = view.getFloat32(i * 4, /* littleEndian= */ true);
  }
  return { meta, elevFlat: flat, reason: null };
}

function sampleElevBilinear(rowF, colF) {
  const H = surfaceMeta.H;
  const W = surfaceMeta.W;
  const row = Math.min(Math.max(rowF, 0), H - 1);
  const col = Math.min(Math.max(colF, 0), W - 1);
  const r0 = Math.floor(row);
  const c0 = Math.floor(col);
  const r1 = Math.min(r0 + 1, H - 1);
  const c1 = Math.min(c0 + 1, W - 1);
  const fr = row - r0;
  const fc = col - c0;
  const v00 = elevFlat[r0 * W + c0];
  const v01 = elevFlat[r0 * W + c1];
  const v10 = elevFlat[r1 * W + c0];
  const v11 = elevFlat[r1 * W + c1];
  return (
    v00 * (1 - fr) * (1 - fc) +
    v01 * (1 - fr) * fc +
    v10 * fr * (1 - fc) +
    v11 * fr * fc
  );
}

// The producer mesh frame (src/mesh.heightfield_to_mesh, frozen v1) is Z-up: x = col*gsd,
// y = -row*gsd, z = height. three.js is Y-up, so every loaded glb is wrapped in a pivot that
// rotates -90 deg about X (producer y -> world -z, i.e. north stays "into the screen";
// producer z -> world +y, height is up) and scales producer z for display. The scale is
// display-only: every readout samples the sidecar elevation grid, never the geometry.
const VEXAG_TARGET_RELIEF = 0.08; // auto: relief spans ~8% of the longer horizontal side
let meshPivot = null;
let vexagAuto = 1;
let vexag = 1;

function wrapInZUpPivot(root) {
  // Measure the producer-frame extent BEFORE parenting, so the result cannot depend on the
  // pivot's (not yet updated) world matrix or on three.js's Box3 update order (#108 review).
  root.updateMatrixWorld(true);
  const local = new THREE.Box3().setFromObject(root, true);
  const pivot = new THREE.Group();
  pivot.name = "producer-z-up-frame";
  pivot.rotation.x = -Math.PI / 2;
  pivot.add(root);
  const sz = local.getSize(new THREE.Vector3());
  const horiz = Math.max(sz.x, sz.y) || 1;
  vexagAuto = sz.z > 0 && sz.z < VEXAG_TARGET_RELIEF * horiz ? (VEXAG_TARGET_RELIEF * horiz) / sz.z : 1;
  vexag = vexagAuto;
  pivot.scale.set(1, 1, vexag);
  return pivot;
}

function formatVexag() {
  const v = vexag >= 10 ? Math.round(vexag).toString() : vexag.toFixed(1);
  const relative = surfaceMeta && surfaceMeta.branch === "relative";
  return relative
    ? `relief x${v} for display (relative surface, not metres)`
    : `x${v} vertical exaggeration (display only; readouts use true heights)`;
}

function setVexag(v) {
  if (!meshPivot) return;
  vexag = Math.min(Math.max(v, 0.1), 1e6);
  meshPivot.scale.set(1, 1, vexag);
  meshPivot.updateMatrixWorld(true);
  measureGroup.clear(); // markers were placed at the old geometry
  measurePoints.length = 0;
  measurePanel.classList.remove("show");
  hudVexag.textContent = formatVexag();
}

function worldToGrid(point) {
  // World hit -> producer (Z-up) mesh frame first, then invert heightfield_to_mesh's
  // vertex mapping: x = col*gsd, y = -row*gsd.
  if (meshPivot) point = meshPivot.worldToLocal(point.clone());
  // "gsd" here is whichever of gsd/pseudo_gsd the sidecar actually set (see
  // sidecar.js resolveHorizontalScale) - a real map/sensor scale on the metric
  // branch, or the 1-mesh-unit-per-pixel display convention on the relative branch.
  const { value: gsd } = resolveHorizontalScale(surfaceMeta);
  return { row: -point.y / gsd, col: point.x / gsd };
}

function makeMarker(color) {
  const m = new THREE.Mesh(
    // Radius tracks the loaded mesh (0.4 was sized for the ~50-unit placeholder and is
    // invisible on a real upload thousands of units wide).
    new THREE.SphereGeometry(Math.max(0.4, sceneBounds.maxDim * 0.006), 16, 16),
    new THREE.MeshBasicMaterial({ color })
  );
  measureGroup.add(m);
  return m;
}

function renderProfile(a, b) {
  const N = 40;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const row = a.row + (b.row - a.row) * t;
    const col = a.col + (b.col - a.col) * t;
    pts.push(sampleElevBilinear(row, col));
  }

  const ctx = profileCanvas.getContext("2d");
  const w = profileCanvas.width;
  const h = profileCanvas.height;
  ctx.clearRect(0, 0, w, h);
  const min = Math.min(...pts);
  const max = Math.max(...pts);
  const span = max - min || 1;
  ctx.strokeStyle = "#4f8ff7";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  pts.forEach((v, i) => {
    const x = (i / N) * w;
    const y = h - ((v - min) / span) * (h - 8) - 4;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function finishMeasurement() {
  const [a, b] = measurePoints;
  // IF-1: "relative branch ... no metric horizontal distance ... no physical slope %".
  // horizDist is only a real metres figure when the sidecar set a real gsd (metric
  // branch); on the relative branch it is in mesh units via pseudo_gsd, and must never
  // be printed with a metre unit - that was a real bug in the previous version of this
  // function, which always printed "m" regardless of branch, and always labelled
  // Δheight "relative units" even on the metric branch where it IS metres.
  const { value: gsd, isReal } = resolveHorizontalScale(surfaceMeta);
  const horizDist = Math.hypot(b.row - a.row, b.col - a.col) * gsd;
  const deltaHeight = b.elev - a.elev;
  const slopePct = horizDist > 0 ? (deltaHeight / horizDist) * 100 : 0;
  const heightUnit = surfaceMeta.branch === "metric" ? "m" : "relative units";

  if (isReal) {
    mDist.textContent = `${horizDist.toFixed(2)} m`;
    mSlope.textContent = `${slopePct.toFixed(1)}%`;
  } else {
    mDist.textContent = `${horizDist.toFixed(2)} mesh units (uncalibrated, not metres)`;
    mSlope.textContent = `${slopePct.toFixed(1)}% (${heightUnit} / mesh unit, not a physical slope)`;
  }
  mDh.textContent = `${deltaHeight.toFixed(3)} ${heightUnit}`;
  measurePanel.classList.add("show");

  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([a.world, b.world]),
    new THREE.LineBasicMaterial({ color: 0xe0a83c })
  );
  measureGroup.add(line);

  renderProfile(a, b);
}

// -- P0-A2: reference-DSM comparison --

referenceFileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  referenceStatus.textContent = "uploading...";
  referenceResult.style.display = "none";
  try {
    const res = await fetch("/api/upload-reference", {
      method: "POST",
      headers: { "X-Filename": file.name },
      body: await file.arrayBuffer(),
    });
    const result = await res.json();
    if (!result.ok) {
      referenceStatus.textContent = `upload failed: ${result.reason}`;
      referenceTifPath = null;
      return;
    }
    referenceTifPath = result.reference_tif_path;
    referenceStatus.textContent = `loaded: ${file.name} -- press R, then click the surface to compare`;
  } catch (err) {
    referenceStatus.textContent = `upload error: ${err.message}`;
    referenceTifPath = null;
  }
});

async function compareAtClick(row, col) {
  if (!referenceTifPath) {
    referenceStatus.textContent = "load a reference .tif first";
    return;
  }
  if (!surfaceMeta || surfaceMeta.branch !== "metric") {
    referenceStatus.textContent = "reference comparison needs the metric branch (this mesh is relative-only)";
    referenceResult.style.display = "none";
    return;
  }
  const metricTifPath = metricTifUrlFromSidecar(surfaceMetaUrlFromGlbUrl(currentGlbUrl), surfaceMeta);
  if (!metricTifPath) {
    referenceStatus.textContent = "could not resolve this mesh's own metric GeoTIFF path";
    referenceResult.style.display = "none";
    return;
  }
  try {
    const res = await fetch("/api/reference-sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metric_tif_path: metricTifPath, reference_tif_path: referenceTifPath, row, col }),
    });
    const result = await res.json();
    if (!result.ok) {
      referenceStatus.textContent = result.reason;
      referenceResult.style.display = "none";
      return;
    }
    refEstimated.textContent = `${result.estimated_m.toFixed(3)} m`;
    refReference.textContent = `${result.reference_m.toFixed(3)} m`;
    refDelta.textContent = `${result.delta_m.toFixed(3)} m`;
    referenceStatus.textContent = "";
    referenceResult.style.display = "block";
  } catch (err) {
    referenceStatus.textContent = `comparison error: ${err.message}`;
    referenceResult.style.display = "none";
  }
}
// -- P0-A3: real image upload -> real run_pipeline.py -> render result --
// scene = image_path.stem in run_pipeline.py -- the uploaded filename's own stem
// (no extension) is what the pipeline names its outputs after. Served via the
// server's /run/ URL prefix (see server.py -- outputs/run/ is a second, separate
// static root from viewer/, matching this repo's existing canonical output
// location).
//
// #85 Finding A: run_pipeline.py always writes {scene}_relative.glb AND
// {scene}_relative_lod.glb on success; metric/metric_lod only when calibration
// succeeds. The server reports which of these actually landed on disk in
// "available_glbs" -- register every real one as its own tier instead of only
// ever loading the (possibly huge) native mesh. Default the auto-loaded tier to
// relative_lod when present (falls back to relative otherwise) -- avoids loading
// a multi-hundred-MB native mesh by default when a small LOD variant is sitting
// right there unused; full-resolution and metric variants stay reachable via the
// existing L tier-cycle key.
const GLB_TIER_LABELS = {
  relative: "relative",
  relative_lod: "relative-lod",
  metric: "metric",
  metric_lod: "metric-lod",
};
uploadFileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  uploadStatus.textContent = "uploading + running pipeline (this can take a while)...";
  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      headers: { "X-Filename": file.name },
      body: await file.arrayBuffer(),
    });
    const result = await res.json();
    if (result.exit_code !== 0) {
      uploadStatus.textContent = `pipeline failed (exit ${result.exit_code}):\n${result.stderr || result.stdout}`;
      return;
    }
    const newTiers = result.available_glbs.map((tag) => ({
      name: `upload:${result.scene}:${GLB_TIER_LABELS[tag]}`,
      url: `/run/${result.out_dir_name}/${result.scene}_${tag}.glb`,
    }));
    meshTiers.unshift(...newTiers);
    const lodIdx = newTiers.findIndex((t) => t.name.endsWith(":relative-lod"));
    tierIndex = lodIdx !== -1 ? lodIdx : 0;
    uploadStatus.textContent = `done -- loading ${meshTiers[tierIndex].name}`;
    loadMesh(meshTiers[tierIndex]);
  } catch (err) {
    uploadStatus.textContent = `upload error: ${err.message}`;
  }
});

function handlePick(ndcX, ndcY) {
  // IF-1 failure contract: no valid sidecar -> measurements stay disabled, no matter
  // what pickableMesh is loaded. showSidecarFailure() already made this visible; this
  // guard is what actually enforces it against a stray click.
  if (!pickableMesh || !measurementReady()) return;
  raycaster.setFromCamera({ x: ndcX, y: ndcY }, camera);
  const hits = raycaster.intersectObject(pickableMesh, true);
  if (!hits.length) return;
  const point = hits[0].point;
  const { row, col } = worldToGrid(point);
  const elev = sampleElevBilinear(row, col);

  if (referenceMode) {
    compareAtClick(row, col);
    return;
  }

  if (!measureMode) {
    const heightUnit = surfaceMeta.branch === "metric" ? "m" : "relative units";
    pickValue.textContent = `${elev.toFixed(3)} ${heightUnit}`;
    pickWarning.textContent =
      surfaceMeta.branch === "relative"
        ? "uncalibrated surface -- not metres"
        : "";
    return;
  }

  if (measurePoints.length === 0) {
    measureGroup.clear();
    measurePanel.classList.remove("show");
    makeMarker(0x3ecf8e).position.copy(point);
    measurePoints.push({ row, col, world: point.clone(), elev });
  } else if (measurePoints.length === 1) {
    makeMarker(0xe0573c).position.copy(point);
    measurePoints.push({ row, col, world: point.clone(), elev });
    finishMeasurement();
    measurePoints.length = 0; // ready for the next pair; markers/line stay visible until next first click
  }
}

renderer.domElement.addEventListener("click", (e) => {
  if (flyMode) return; // fly mode uses drag-to-look; clicks aren't picks there
  const rect = renderer.domElement.getBoundingClientRect();
  const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  handlePick(ndcX, ndcY);
});

function loadMesh(tier) {
  const loadStart = performance.now();
  currentGlbUrl = tier.url;
  const myGeneration = ++loadGeneration; // #87: this call's identity, checked in both
                                          // async callbacks below before any state mutation

  // Reset any previous mesh's measurement state immediately, so a mesh that fails to
  // load (or has no sidecar) can never leave the PREVIOUS mesh's picking/measurement
  // silently active against a now-wrong mesh.
  surfaceMeta = null;
  elevFlat = null;
  measureMode = false;
  measurePoints.length = 0;
  measureGroup.clear();
  measurePanel.classList.remove("show");
  clearSidecarFailure();
  clearBranchBanner();
  if (placeholderBadge) placeholderBadge.classList.remove("show");
  // P0-A2: a reference comparison result from a previous mesh must not linger once a
  // different mesh is loaded -- referenceTifPath (the uploaded file) is independent of
  // which mesh is loaded, so it deliberately survives a tier switch, but any displayed
  // result/status is mesh-specific and must clear.
  referenceResult.style.display = "none";
  referenceStatus.textContent = referenceTifPath
    ? "reference loaded -- press R, then click the surface to compare"
    : "";

  loader.load(
    tier.url,
    (gltf) => {
      if (myGeneration !== loadGeneration) {
        // A newer loadMesh() call has started since this one was issued -- this response
        // is stale. Dispose what was loaded (it's already fetched/parsed, no point keeping
        // it in memory; disposeMesh()'s scene.remove() is a harmless no-op since this
        // object was never added to the scene) and do not touch any shared scene/HUD state.
        disposeMesh(gltf.scene);
        return;
      }
      disposeMesh(currentMesh);
      // The producer glb sets no metallicFactor, and the glTF default is 1.0 (fully metallic):
      // with no environment map a metallic surface shows almost no diffuse light, which is why
      // uploaded terrain rendered very dark (#108 review). Terrain is a dielectric surface.
      gltf.scene.traverse((obj) => {
        if (obj.isMesh) {
          (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach((m) => {
            if (m && "metalness" in m) {
              m.metalness = 0.0;
              m.roughness = 1.0;
              m.needsUpdate = true;
            }
          });
        }
      });
      meshPivot = wrapInZUpPivot(gltf.scene);
      currentMesh = meshPivot;
      pickableMesh = meshPivot;
      scene.add(meshPivot);
      meshPivot.updateMatrixWorld(true);
      hudVexag.textContent = formatVexag();

      let triCount = 0;
      gltf.scene.traverse((obj) => {
        if (obj.isMesh && obj.geometry) {
          const idx = obj.geometry.index;
          triCount += idx ? idx.count / 3 : obj.geometry.attributes.position.count / 3;
        }
      });
      hudTris.textContent = Math.round(triCount).toLocaleString();

      const box = new THREE.Box3().setFromObject(meshPivot);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      orbit.target.copy(center);
      const maxDim = Math.max(size.x, size.y, size.z) || 1;

      sceneBounds = { center: center.clone(), maxDim };
      orbit.minDistance = maxDim * 0.15;
      orbit.maxDistance = maxDim * 8;

      // Rescale the ground grid and fly speed to this mesh's actual size --
      // both were tuned for the small placeholder and looked wrong (tiny/
      // misplaced grid, sluggish fly movement) on a real, much larger upload.
      scene.remove(grid);
      grid.geometry.dispose();
      grid.material.dispose();
      grid = new THREE.GridHelper(maxDim * 4, 40, 0x2a3a4c, 0x1a2634);
      grid.position.set(center.x, box.min.y, center.z);
      scene.add(grid);

      flySpeed = maxDim * 0.4;

      camera.position.set(
        center.x + maxDim * 0.9,
        center.y + maxDim * 0.9,
        center.z + maxDim * 1.2
      );
      camera.near = maxDim / 1000;
      camera.far = maxDim * 100;
      camera.updateProjectionMatrix();

      const loadMs = (performance.now() - loadStart).toFixed(1);
      hudLoad.textContent = `${loadMs} ms`;
      hudTier.textContent = tier.name;
      hudLoad.classList.remove("err");
    },
    undefined,
    (err) => {
      if (myGeneration !== loadGeneration) return; // stale failure, a newer load superseded it
      hudLoad.textContent = "load failed";
      hudLoad.classList.add("err");
      console.error("GLTF load error:", err);
    }
  );

  // Sidecar load runs independently of the glb load above (they're separate fetches
  // of separate files) - a slow/failed glb load does not block the sidecar result,
  // and vice versa. IF-1: any failure disables measurement and shows why, visibly.
  loadSidecar(tier.url).then((result) => {
    if (myGeneration !== loadGeneration) return; // stale sidecar response, ignore --
                                                   // must never overwrite a newer mesh's metadata
    if (result.meta === null) {
      showSidecarFailure(result.reason);
      return;
    }
    surfaceMeta = result.meta;
    elevFlat = result.elevFlat;
    if (meshPivot) hudVexag.textContent = formatVexag();
    applyBranchBanner(surfaceMeta);
    if (placeholderBadge) placeholderBadge.classList.toggle("show", surfaceMeta.is_placeholder === true);
  });
}

window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "f") {
    setMode(!flyMode);
    return;
  }
  if (e.key === "[" || e.key === "]") {
    setVexag(e.key === "]" ? vexag * 1.5 : vexag / 1.5);
    return;
  }
  if (e.key.toLowerCase() === "l") {
    tierIndex = (tierIndex + 1) % meshTiers.length;
    loadMesh(meshTiers[tierIndex]);
    return;
  }
  if (e.key.toLowerCase() === "m") {
    if (!measurementReady()) return; // IF-1: no sidecar -> measurement stays disabled
    measureMode = !measureMode;
    measurePoints.length = 0;
    if (!measureMode) {
      measureGroup.clear();
      measurePanel.classList.remove("show");
    }
    return;
  }
  if (e.key.toLowerCase() === "r") {
    referenceMode = !referenceMode;
    referenceStatus.textContent = referenceMode
      ? (referenceTifPath ? "reference-compare mode: click the surface" : "load a reference .tif first")
      : "";
    return;
  }
  if (e.key.toLowerCase() in flyKeys) flyKeys[e.key.toLowerCase()] = true;
});
window.addEventListener("keyup", (e) => {
  if (e.key.toLowerCase() in flyKeys) flyKeys[e.key.toLowerCase()] = false;
});
renderer.domElement.addEventListener("pointerdown", (e) => {
  if (!flyMode) return;
  dragging = true;
  lastPointer = { x: e.clientX, y: e.clientY };
});
window.addEventListener("pointerup", () => (dragging = false));
window.addEventListener("pointermove", (e) => {
  if (!flyMode || !dragging) return;
  const dx = e.clientX - lastPointer.x;
  const dy = e.clientY - lastPointer.y;
  lastPointer = { x: e.clientX, y: e.clientY };
  yawPitch.yaw -= dx * 0.003;
  yawPitch.pitch = Math.max(-1.5, Math.min(1.5, yawPitch.pitch - dy * 0.003));
});

function updateFly(dt) {
  if (!flyMode) return;
  camera.rotation.order = "YXZ";
  camera.rotation.y = yawPitch.yaw;
  camera.rotation.x = yawPitch.pitch;

  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  const move = new THREE.Vector3();
  if (flyKeys.w) move.add(forward);
  if (flyKeys.s) move.sub(forward);
  if (flyKeys.d) move.add(right);
  if (flyKeys.a) move.sub(right);
  if (flyKeys.e) move.y += 1;
  if (flyKeys.q) move.y -= 1;
  if (move.lengthSq() > 0) {
    move.normalize().multiplyScalar(flySpeed * dt);
    camera.position.add(move);
  }

  clampDistanceFromMesh();
}

// -- Initial load --
loadMesh(meshTiers[tierIndex]);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

let lastTime = performance.now();

// FPS: rolling average over a short window (500ms), updated a couple times
// per second so the HUD number is readable rather than flickering every
// single frame. Real measured value from requestAnimationFrame timing.
let fpsFrameCount = 0;
let fpsWindowStart = performance.now();

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;

  if (flyMode) updateFly(dt);
  else orbit.update();

  renderer.render(scene, camera);

  fpsFrameCount++;
  const windowElapsed = now - fpsWindowStart;
  if (windowElapsed >= 500) {
    const fps = (fpsFrameCount / windowElapsed) * 1000;
    hudFps.textContent = fps.toFixed(1);
    fpsFrameCount = 0;
    fpsWindowStart = now;
  }
}
animate();
