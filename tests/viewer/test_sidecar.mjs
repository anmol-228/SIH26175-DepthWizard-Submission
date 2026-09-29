// Pure-logic tests for viewer/sidecar.js - DW-WP2 (issues #26/#27, IF-1).
//
// Uses Node's built-in test runner (node:test) - no browser, no new toolchain
// dependency, matching the viewer's own "no build step" philosophy. Node 18+ has
// node:test built in.
//
//     node tests/viewer/test_sidecar.mjs
//
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  sidecarBaseFromGlbUrl,
  surfaceMetaUrlFromGlbUrl,
  elevationUrlFromSurfaceMetaUrl,
  validateSurfaceMeta,
  resolveHorizontalScale,
} from "../../viewer/sidecar.js";

function validRelativeMeta(overrides = {}) {
  return {
    schema: "viewer_input_v1",
    branch: "relative",
    H: 64,
    W: 64,
    dtype: "float32",
    byte_order: "little",
    elevation_payload: "g_relative_elevation.f32",
    units: "unitless",
    gsd: null,
    pseudo_gsd: 1.0,
    z_exaggeration: 1.0,
    crs: null,
    ...overrides,
  };
}

function validMetricMeta(overrides = {}) {
  return {
    schema: "viewer_input_v1",
    branch: "metric",
    H: 64,
    W: 64,
    dtype: "float32",
    byte_order: "little",
    elevation_payload: "g_metric_elevation.f32",
    units: "metres",
    gsd: 0.09,
    pseudo_gsd: null,
    z_exaggeration: 1.0,
    crs: 32767,
    ...overrides,
  };
}

test("native and LOD glb URLs resolve to the same sidecar base", () => {
  assert.equal(sidecarBaseFromGlbUrl("g_relative.glb"), "g_relative");
  assert.equal(sidecarBaseFromGlbUrl("g_relative_lod.glb"), "g_relative");
});

test("relative and metric glb URLs resolve to DIFFERENT sidecar bases (no cross-branch mismatch)", () => {
  assert.notEqual(sidecarBaseFromGlbUrl("g_relative.glb"), sidecarBaseFromGlbUrl("g_metric.glb"));
});

test("sidecar base strips a query string and preserves a directory prefix", () => {
  assert.equal(sidecarBaseFromGlbUrl("./out/scene_metric_lod.glb?x=1"), "./out/scene_metric");
});

test("surfaceMetaUrlFromGlbUrl appends the frozen suffix", () => {
  assert.equal(surfaceMetaUrlFromGlbUrl("g_relative_lod.glb"), "g_relative_surface_meta.json");
});

test("elevationUrlFromSurfaceMetaUrl resolves the bare payload filename against the sidecar's own directory", () => {
  assert.equal(
    elevationUrlFromSurfaceMetaUrl("./out/g_relative_surface_meta.json", "g_relative_elevation.f32"),
    "./out/g_relative_elevation.f32"
  );
  assert.equal(
    elevationUrlFromSurfaceMetaUrl("g_relative_surface_meta.json", "g_relative_elevation.f32"),
    "g_relative_elevation.f32"
  );
});

test("a well-formed relative sidecar validates ok", () => {
  assert.deepEqual(validateSurfaceMeta(validRelativeMeta()), { ok: true, reason: null });
});

test("a well-formed metric sidecar validates ok", () => {
  assert.deepEqual(validateSurfaceMeta(validMetricMeta()), { ok: true, reason: null });
});

test("missing sidecar (null/undefined) is rejected, not treated as ok", () => {
  assert.equal(validateSurfaceMeta(null).ok, false);
  assert.equal(validateSurfaceMeta(undefined).ok, false);
});

test("wrong schema string is rejected", () => {
  const result = validateSurfaceMeta(validRelativeMeta({ schema: "something_else" }));
  assert.equal(result.ok, false);
  assert.match(result.reason, /schema/);
});

test("missing required field is rejected with a field-naming reason", () => {
  const meta = validRelativeMeta();
  delete meta.dtype;
  const result = validateSurfaceMeta(meta);
  assert.equal(result.ok, false);
  assert.match(result.reason, /dtype/);
});

test("invalid payload dimensions (H<=0) are rejected", () => {
  const result = validateSurfaceMeta(validRelativeMeta({ H: 0 }));
  assert.equal(result.ok, false);
});

test("unsupported dtype/byte_order is rejected rather than assumed compatible", () => {
  assert.equal(validateSurfaceMeta(validRelativeMeta({ dtype: "float64" })).ok, false);
  assert.equal(validateSurfaceMeta(validRelativeMeta({ byte_order: "big" })).ok, false);
});

test("both gsd and pseudo_gsd set is rejected - exactly one must be non-null", () => {
  const result = validateSurfaceMeta(validRelativeMeta({ gsd: 0.09, pseudo_gsd: 1.0 }));
  assert.equal(result.ok, false);
});

test("neither gsd nor pseudo_gsd set is rejected", () => {
  const result = validateSurfaceMeta(validRelativeMeta({ gsd: null, pseudo_gsd: null }));
  assert.equal(result.ok, false);
});

test("a relative branch carrying a non-null crs is rejected - branches must never conflate", () => {
  const result = validateSurfaceMeta(validRelativeMeta({ crs: 32601 }));
  assert.equal(result.ok, false);
});

test("resolveHorizontalScale reports isReal=true only when gsd (not pseudo_gsd) is set", () => {
  assert.deepEqual(resolveHorizontalScale(validMetricMeta()), { value: 0.09, isReal: true });
  assert.deepEqual(resolveHorizontalScale(validRelativeMeta()), { value: 1.0, isReal: false });
});
