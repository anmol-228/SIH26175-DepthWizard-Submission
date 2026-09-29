// viewer_input_v1 sidecar resolution + validation - DW-WP2 (issues #26/#27).
//
// Pure logic, no DOM / three.js dependency, so it is testable with Node's built-in
// test runner (see tests/viewer/test_sidecar.mjs) without a browser.
//
// File-layout decision (frozen IF-1 names the field SET, not the exact path layout -
// see run_pipeline.py's write_surface_meta() docstring for the full reasoning): the
// sidecar for a mesh sits FLAT, in the same directory as that mesh, sharing its
// filename stem with a trailing "_lod" stripped - so a native mesh and its LOD sibling
// resolve to the SAME sidecar (the elevation grid is identical; only face count
// differs), and relative/metric meshes in the same output directory can never be
// confused with each other's elevation data.

export const VIEWER_INPUT_SCHEMA = "viewer_input_v1";

/**
 * Given a `.glb` URL (whatever was passed as `?glb=` or a mesh-tier entry), return
 * the base path used to derive that mesh's sidecar files:
 *   `${base}_surface_meta.json` and, once fetched, `meta.elevation_payload` resolved
 *   against the same directory.
 *
 * Strips a query string, the `.glb` extension, and one trailing `_lod` - so
 * `g_relative.glb` and `g_relative_lod.glb` both resolve to base `g_relative`.
 */
export function sidecarBaseFromGlbUrl(glbUrl) {
  if (typeof glbUrl !== "string" || glbUrl.length === 0) {
    throw new TypeError(`sidecarBaseFromGlbUrl expects a non-empty string, got ${JSON.stringify(glbUrl)}`);
  }
  const withoutQuery = glbUrl.split(/[?#]/)[0];
  const withoutExt = withoutQuery.endsWith(".glb")
    ? withoutQuery.slice(0, -".glb".length)
    : withoutQuery;
  const withoutLod = withoutExt.endsWith("_lod") ? withoutExt.slice(0, -"_lod".length) : withoutExt;
  return withoutLod;
}

export function surfaceMetaUrlFromGlbUrl(glbUrl) {
  return `${sidecarBaseFromGlbUrl(glbUrl)}_surface_meta.json`;
}

/**
 * Resolve `meta.elevation_payload` (a bare filename, per IF-1) against the directory
 * the surface_meta.json itself was fetched from - NOT against the glb's directory
 * directly, so a future layout where they differ still works. In the flat, same-
 * directory layout this repo uses, both directories are the same.
 */
export function elevationUrlFromSurfaceMetaUrl(surfaceMetaUrl, elevationPayload) {
  const dir = surfaceMetaUrl.slice(0, surfaceMetaUrl.lastIndexOf("/") + 1);
  return dir + elevationPayload;
}

const REQUIRED_STRING_FIELDS = ["schema", "branch", "dtype", "byte_order", "elevation_payload", "units"];
const REQUIRED_NUMBER_FIELDS = ["H", "W", "z_exaggeration"];

/**
 * Validate a fetched surface_meta.json against the frozen viewer_input_v1 contract,
 * to the extent needed to safely drive measurement math. Returns {ok, reason}:
 * ok=false means the caller must disable picking/measurement and show `reason`
 * verbatim - never fall through to a placeholder or guess a default.
 */
export function validateSurfaceMeta(meta) {
  if (meta === null || typeof meta !== "object") {
    return { ok: false, reason: "sidecar metadata is not a JSON object" };
  }
  if (meta.schema !== VIEWER_INPUT_SCHEMA) {
    return { ok: false, reason: `unsupported sidecar schema: ${JSON.stringify(meta.schema)}` };
  }
  for (const field of REQUIRED_STRING_FIELDS) {
    if (typeof meta[field] !== "string" || meta[field].length === 0) {
      return { ok: false, reason: `sidecar missing required field "${field}"` };
    }
  }
  for (const field of REQUIRED_NUMBER_FIELDS) {
    if (typeof meta[field] !== "number" || !Number.isFinite(meta[field])) {
      return { ok: false, reason: `sidecar missing/invalid required field "${field}"` };
    }
  }
  if (meta.H <= 0 || meta.W <= 0) {
    return { ok: false, reason: `sidecar has invalid payload dimensions H=${meta.H} W=${meta.W}` };
  }
  if (meta.branch !== "relative" && meta.branch !== "metric") {
    return { ok: false, reason: `sidecar has invalid branch: ${JSON.stringify(meta.branch)}` };
  }
  if (meta.dtype !== "float32" || meta.byte_order !== "little") {
    return { ok: false, reason: `sidecar payload type not supported: dtype=${meta.dtype}, byte_order=${meta.byte_order}` };
  }
  const hasGsd = typeof meta.gsd === "number" && Number.isFinite(meta.gsd);
  const hasPseudo = typeof meta.pseudo_gsd === "number" && Number.isFinite(meta.pseudo_gsd);
  if (hasGsd === hasPseudo) {
    return { ok: false, reason: "sidecar must set exactly one of gsd/pseudo_gsd" };
  }
  if (meta.branch === "relative" && meta.crs !== null && meta.crs !== undefined) {
    return { ok: false, reason: "relative branch must not carry a crs value" };
  }
  return { ok: true, reason: null };
}

/**
 * The horizontal pixel spacing to use for distance/slope math, and whether it is a
 * real map/sensor scale (metric branch) or a display convention (relative branch,
 * pseudo_gsd) that must never be reported as metres.
 */
export function resolveHorizontalScale(meta) {
  if (typeof meta.gsd === "number" && Number.isFinite(meta.gsd)) {
    return { value: meta.gsd, isReal: true };
  }
  return { value: meta.pseudo_gsd, isReal: false };
}


/**
 * P0-A2: derive the metric branch's own GeoTIFF path from the loaded sidecar,
 * WITHOUT a new sidecar field -- run_pipeline.py writes metric_tif and metric_glb
 * from the same scene stem and out_dir (`{scene}_metric.glb` / `{scene}_metric.tif`),
 * confirmed directly against run_pipeline.py's code. Only meaningful on the metric
 * branch; the relative branch has no metric GeoTIFF at all (per IF-1, relative
 * carries no crs/georeference). Returns null when not applicable/derivable, rather
 * than guessing a path that may not exist.
 */
export function metricTifUrlFromSidecar(surfaceMetaUrl, meta) {
  if (!meta || meta.branch !== "metric") return null;
  if (typeof meta.mesh_glb !== "string" || !meta.mesh_glb.endsWith(".glb")) return null;
  const dir = surfaceMetaUrl.slice(0, surfaceMetaUrl.lastIndexOf("/") + 1);
  const tifName = meta.mesh_glb.slice(0, -".glb".length) + ".tif";
  return dir + tifName;
}
