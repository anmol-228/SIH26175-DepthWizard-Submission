# Interface contract: CALIBRATION -> PIPELINE / VIEWER

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

Issue: [DW-S1-02] Wire and Validate Calibration Output for Downstream Pipeline (#5)
Owner: Nirmalya (pipeline/integration). Reviewers: Anmol (calibration semantics),
Ashmita/Aishee (downstream fit).
Field list per `(internal note, not included)`, quoted in issue #5.

This document covers **two separate products** that must never be conflated: the
**metric DSM** (real metres, georeferenced) and the **relative DSM** (unitless shape
only). They have different version status. Read both tables before building anything.

---

## Why this document exists

Before this change, the CALIBRATION -> PIPELINE/VIEWER boundary was not one handoff.
It was two disconnected pieces plus a gap, as `fixtures/calibration_output_v0.json`
recorded honestly:

1. `scripts/07_calibrate_and_eval.py` computed the calibrated metric array
   (`p_cal = a * pred + b`) and **never wrote it to disk** - only a metrics report.
   Worse, it flattened every patch (`x.ravel()`) before calibrating, so the 2-D raster
   shape was destroyed and could not have been written even in principle.
2. `PHASE2_RED_FLAGS/scripts/K2_georeferencing_export.py` proved GeoTIFF-with-GeoKeys
   export works, but against the **ground-truth** DSM, with the tag construction inline
   in a one-off script rather than in a reusable function.
3. No relative (non-georeferenced) code path existed at all.

Downstream consumers therefore had to guess. This contract closes (1) and (2), and
explicitly scopes (3) as still unbuilt rather than pretending otherwise.

---

## Product A: metric DSM  -  `calibration_output/v0-metric`

| Field | Value |
|---|---|
| **PRODUCER** | `scripts/07_calibrate_and_eval.py` (per-patch, per test tile) and `PHASE2_RED_FLAGS/scripts/K1_tiling_stitch.py` (full-tile, feather-blended). Both call `geoio.save_geotiff()`. |
| **CONSUMER** | `heightfield_to_mesh()` in `PHASE2_RED_FLAGS/scripts/K3_mesh_pipeline.py` (mesh/`.glb` -> viewer, Ashmita/Aishee); `geoio.load_geotiff()` for any evaluation or inspection (Debopriya). |
| **FILE / OBJECT** | `outputs/predictions/calibrated_dsm_area{N}_{calibration}.tif` (from `07`), `PHASE2_RED_FLAGS/outputs/predictions/K1_calibrated_dsm_area{N}.tif` (from K1). Both directories are gitignored as regenerable bulk; the accompanying manifest is committed. |
| **FORMAT** | GeoTIFF, single band, written by `tifffile` with hand-constructed GeoTIFF tags. Not GDAL/rasterio - see `src/geoio.py`'s module docstring and handbook Part VII. |
| **SHAPE / DIMENSIONS** | 2-D `(H, W)`. From K1: the full tile, e.g. `(1783, 2995)` for area 26. From `07`: the **patch-covered subregion anchored at the tile origin**, `H = max(patch_y) + 512`, `W = max(patch_x) + 512`. `04_make_patches.py` drops patches failing its validity mask, so coverage inside that extent can be sparse; uncovered pixels are nodata. Each raster's manifest entry records `shape`, `n_valid_px` and `n_nodata_px`. |
| **UNITS** | Metres. Real elevation in the source DSM's vertical reference, after `metric = a * relative + b`. |
| **COORDINATE SYSTEM** | **No authority code asserted.** `ProjectedCSTypeGeoKey = 32767` (user-defined). Pixel scale and tiepoint are real and round-trip exactly, so the pixel grid and relative geometry are trustworthy. Earlier exports stamped EPSG:32633, inherited from the dataset's own `dsm.vrt` (`PROJCS["WGS 84 / UTM zone 33N"]`, recorded in `PHASE2_RED_FLAGS/02_DATASET_RESEARCH.md`). That is wrong: easting 496794.825 implies ~8.96E under UTM 32N, which is Vaihingen, while under 33N it implies ~14.96E, roughly 600 km east. We do not substitute 32632/25832 either, since that would assert a code our own arithmetic produced against the source's declaration. Absolute world position is therefore **unverified by design**. |
| **METADATA** | Embedded in the file: `ModelPixelScaleTag` (33550), `ModelTiepointTag` (33922), `GeoKeyDirectoryTag` (34735), `GDAL_NODATA` (42113). Alongside it: a JSON manifest recording `calibration` type, fitted `a`/`b`, `units`, `dtype`, `nodata`, `epsg`, `pixel_size`, `origin`, `shape`, valid/nodata pixel counts, `extent`, and an `IMPORTANT` provenance string. |
| **VERSION** | **v0 / EXPERIMENTAL.** Rolled back from a premature v1: the mechanism is implemented and round-trip tested, but v1 additionally requires a real end-to-end run on model predictions and downstream-consumer review. See `fixtures/calibration_output_v0_metric.json` -> `what_v1_requires`. |
| **ERROR BEHAVIOR** | `save_geotiff` raises `ValueError` on a non-2-D array and requires `epsg` explicitly (no default, so a georeferenced product is never written by accident). `save_dsm_geotiff` raises when the source raster's transform came from `load_geotiff`'s nominal fallback rather than real data (`require_georeferenced=True`). Both producers **skip the export and record the reason** when the Vaihingen `.tfw` geoinfo is absent, rather than writing a guessed georeference. `load_geotiff(..., strict=True)` raises instead of guessing. |
| **VALIDATION** | `scripts/10_verify_calibration_output.py`. Reopens every declared raster and asserts pixel size, origin, shape, EPSG GeoKey, nodata tag, dtype, real-georeference flag and valid-pixel count, then feeds the array into the real `heightfield_to_mesh()` consumer. This is the K2-style round-trip test `role_cards/NIRMALYA.md` requires of any export-path change. |
| **OWNER** | Nirmalya. |

### NoData

`-9999.0`, written as the `GDAL_NODATA` tag and exposed as `geoio.DSM_NODATA`. This is
not a new invention: it is the value the project build guide (not included) section 10 specifies
for the DSM profile, and the default `geoio.dsm_nodata_mask()` already assumed. Previously
the convention existed reader-side only and K2's writer set no tag; that gap is closed.

### What a consumer must still know

- **Elevation is absolute in the GeoTIFF, rebased in the mesh.** `heightfield_to_mesh()`
  computes `Z = (elev - elev.min()) * z_exaggeration`, so the `.glb` loses the absolute
  datum. A viewer needing true height must add `elev.min()` back from the GeoTIFF. See
  `fixtures/pipeline_to_3d_v1.json` -> `z_computation`.
- **Mask nodata before meshing.** `-9999` pixels would otherwise drag the mesh floor down
  through that same `elev.min()` rebase. `10_verify_calibration_output.py` shows the
  pattern.
- **Read `gsd` from the raster, not a constant.** `K3_mesh_pipeline.py` currently hardcodes
  `gsd=0.09`; `GeoRaster.gsd` is the correct source.
- **LOD decimation drops the texture.** Confirmed by reproduction (issue #8 / PR #9):
  `trimesh.simplify_quadric_decimation()` replaces `TextureVisuals` with `ColorVisuals`,
  discarding UVs and the PBR material, because decimation reindexes vertices. An LOD `.glb`
  built that way renders flat grey. This is a property of the post-process, not of
  `heightfield_to_mesh()`, so the frozen `pipeline_to_3d_v1.json` contract is unaffected and
  is deliberately left untouched. Tracked in issue #12 (Aishee), with a verified fix:
  recompute UVs from vertex XY after decimation, valid because `heightfield_to_mesh` derives
  UV purely from grid position.
- **Oracle is not accuracy.** Rasters labelled `per_tile_full_oracle` are calibrated
  against the tile's own dense ground truth: a non-deployable upper bound. `sparse_10` is
  deployment-like but still samples the test tile. Neither is a blind-generalization
  figure. The `IMPORTANT` field in each manifest entry says so; do not strip it when
  quoting a number.

---

## Product B: relative DSM  -  `calibration_output/v0-relative`

| Field | Value |
|---|---|
| **PRODUCER** | `src/relative_dsm.py` - `relative_dsm_from_image()` then `save_relative_dsm()`. Deliberately does **not** import `geoio`; asserted structurally by `tests/test_relative_dsm.py`. |
| **CONSUMER** | Would be the same mesh/viewer path, minus georeferencing. |
| **FILE / OBJECT** | `outputs/predictions/relative_dsm_{scene}.npy` plus `relative_dsm_{scene}.json`. Per the project build guide (not included) section 10.1, a relative DSM is "GeoTIFF if a spatial grid exists; otherwise NPY/PNG" - here one does not. |
| **FORMAT** | float32 array, no GeoTIFF tags, no CRS. |
| **UNITS** | **Unitless.** Internally shape-consistent only (taller reads as taller). No metres, no datum. |
| **COORDINATE SYSTEM** | None, by definition. This branch exists precisely for input that has none. |
| **METADATA** | Sidecar carries `product: "relative_dsm"`, `schema_version`, `units: "UNITLESS"`, the normalization parameters (so the transform is reversible), `depth_sign` and `pseudo_gsd`. It carries **none** of `epsg`, `crs`, `origin`, `pixel_size`, `nodata` - the writer raises if one appears. |
| **VERSION** | **v0 / EXPERIMENTAL - implemented, not frozen.** Buildable, but the field set can still change. v1 additionally needs Part IV item (c), qualitative evaluation on real non-georeferenced photographs, plus consumer review. |
| **ERROR BEHAVIOR** | `RelativeDSMError` on a missing/degenerate image, an invalid `depth_sign`, or a sidecar that somehow carries a georeferencing field. Flat or non-finite input yields a valid flat field rather than inf/nan. |
| **VALIDATION** | `tests/test_relative_dsm.py`, 9 cases: no-geoio-import (AST), grayscale/RGBA handling, outlier robustness, flat/NaN handling, both sign directions, discriminator present, forbidden fields absent, writer refuses a leaked `epsg`, and real consumption by `heightfield_to_mesh()` at the pseudo-GSD. |
| **OWNER** | Nirmalya (handbook Part IV; `role_cards/NIRMALYA.md` starter task 2). |

### Normalization and the sign convention

Normalized to a unitless `[0, 1]` field using a **p2-p98 percentile clip** rather than
min/max, so a single outlier pixel cannot compress the scene into a sliver of the range.
The bounds are recorded in the sidecar, so the transform is auditable and reversible.

`depth_sign` defaults to `+1` (identity). For **nadir** imagery that is correct: DA2's
output is disparity-like, and looking straight down, closer to the camera means higher
elevation - a rooftop is both nearer and taller. Confirmed by this repo's own measurements
(`zeroshot_3A_raw.json` mean_corr +0.731/+0.694/+0.802, `zeroshot_3B` fitted a=+3.898).
`scripts/05_zeroshot.py` previously carried a comment asserting the opposite; that was never
what the code did, and it has been corrected. For oblique or handheld input - which "plain
JPG/PNG" permits - the nadir argument collapses and the sign genuinely inverts, which is why
this is a parameter rather than a constant.

`heightfield_to_mesh()` requires a `gsd` with no default. A non-georeferenced image has
none, so the convention is **`PSEUDO_GSD = 1.0`, one mesh unit per pixel**. Passing a
plausible metric GSD would smuggle a units claim into the mesh.

### What is still missing for v1

The official problem statement makes this a real requirement: non-georeferenced input
(plain JPG/PNG) must produce a relative DSM, and that is a genuinely separate code path,
not something the metric path covers. Handbook Part IV: the relative branch needs "(a) a
clean code path that skips calibration entirely, (b) sensible default normalization ...
(c) its own qualitative testing on non-georeferenced images, none of which exist yet".

Built under issue #15. It stays at v0 because Part IV item (c) - qualitative evaluation on
genuinely non-georeferenced photographs - has not been done: the repo has no licence-clean
sample images, and "does the shape look right" is subjective evidence that belongs with
Debopriya rather than a self-assessed pass. Automated coverage is real but is not the same
claim. The sidecar field set also has not been reviewed by a downstream consumer.

**The failure mode this section exists to prevent** (`role_cards/NIRMALYA.md`): treating
"the metric branch works" as evidence the relative branch works. It does not. Nothing in
this repo currently produces a relative DSM as a tested artifact.

---

## Change log

- **2026-08-29 (b)** - CRS authority code dropped in favour of 32767 user-defined, after the
  PR #13 review; `fixtures/calibration_output_v0.json` marked superseded and `fixtures/README.md`
  rewritten to describe the wired state.
- **2026-08-29 (a)** - metric branch rolled back v1 -> v0 after the Day-1 review: freezing
  on mechanism alone, before a real model-prediction run and consumer review, did not meet
  the policy's v1 bar. The in-place caveat added to the frozen `pipeline_to_3d_v1.json` was
  reverted for the same reason (a v1 fixture needs a v2, not an edit); that caveat now lives
  in this document and in issue #12 instead.
- **2026-08-28** - initial contract written. Wired
  Wired `07_calibrate_and_eval.py` and `K1_tiling_stitch.py` to `geoio.save_geotiff()`;
  added `GDAL_NODATA`; added `epsg` and `georeferenced` to `GeoRaster`; added
  `scripts/10_verify_calibration_output.py`.
