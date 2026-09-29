# DepthWizard interface fixtures

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Doc-integrity note (2026-09-19):** This file cites `CURRENT_STATE.md` and/or `EVIDENCE_LEDGER.md` below as existing sources. Neither file exists anywhere in this repo — confirmed via `git log --all` across every branch (zero commits, ever) and independently cross-checked on GitHub. `CURRENT_STATE.md`'s content (results, known gaps, workspace/repro setup) lives in `BASELINE_v1.0.md`; nothing in this repo currently serves as `EVIDENCE_LEDGER.md`'s claim-to-source-with-status-label table — the closest real, currently-existing equivalents are `EXPERIMENT_INDEX.md`, `FINAL_CLAIM_EVIDENCE_MATRIX.md`, and `PHASE2_RED_FLAGS/17_FINAL_QA.md`. This was already flagged once, for one specific note, in `10_DOCUMENTATION/DW_S1_03_SPARSE10_PROVENANCE.md` (2026-08-29) but not corrected repo-wide. Flagging here rather than silently repointing every individual inline reference below — needs an owner decision: create the real `EVIDENCE_LEDGER.md`, or update each reference to its correct current target.

Frozen (or honestly-not-yet-frozen) example files for the two pipeline boundaries identified as the top structural risk to parallel development: Ashmita/Aishee (3D/UX) need a literal, concrete contract to build against without waiting on Anmol/Nirmalya. Format follows `(internal note, not included)`'s v0/v1 freeze strategy.

Both fixtures were built by reading the actual current code (`src\da2_utils.py`, `src\geoio.py`, `src\metrics.py`, `scripts\07_calibrate_and_eval.py`, `scripts\08_mini_3d_mesh.py`, and the `PHASE2_RED_FLAGS\scripts\K2_georeferencing_export.py` / `K3_mesh_pipeline.py` / `K4_viewer_browser_test.json` evidence), not copied from the handbook's Part X prose. Where the code and the handbook prose disagreed, the code wins and the disagreement is called out explicitly in the fixture file.

## Files

### CALIBRATION → PIPELINE/VIEWER boundary — two fixtures

**`calibration_output_v0_metric.json` — metric DSM. Status: v0 / EXPERIMENTAL.**
**`calibration_output_relative_v0.json` — relative DSM. Status: v0 / EXPERIMENTAL, implemented.**

**`calibration_output_v0.json` is SUPERSEDED** and retained only as a historical record of
the pre-wiring state. Do not build against it.

The boundary described in that older file as "two disconnected pieces" is now wired
(DW-S1-02, issue #5): `geoio.save_geotiff()` writes the calibrated array as a GeoTIFF with
the tag set K2 verified, and both `scripts/07_calibrate_and_eval.py` and
`PHASE2_RED_FLAGS/scripts/K1_tiling_stitch.py` call it. Verified end to end on real model
predictions: 19 rasters, 19/19 passing `scripts/10_verify_calibration_output.py`, including
consumption by the real `heightfield_to_mesh()`. See
`10_DOCUMENTATION/CALIBRATION_OUTPUT_INTERFACE.md` for the full 12-field contract and
`10_DOCUMENTATION/DW_S1_02_RUN_EVIDENCE.md` for the run.

Two things a consumer must know before using it:

- **It stays at v0, not v1.** The mechanism works, but promotion additionally needs Anmol's
  review of calibration semantics and Ashmita/Aishee confirming the field set. The metric
  fixture's `what_v1_requires` lists the conditions explicitly.
- **No CRS authority code is asserted.** Exports carry `ProjectedCSType = 32767`
  (user-defined). The geotransform is real and round-trips exactly, but the dataset's own
  `dsm.vrt` declares EPSG:32633 while the coordinates imply UTM 32N, so no defensible
  authority code exists. Geometry is trustworthy; absolute world position is not.

### `pipeline_to_3d_v1.json` — PIPELINE → 3D boundary (height array + RGB → mesh)

**Status: v1 / FROZEN.**

Why: `heightfield_to_mesh()` in `PHASE2_RED_FLAGS\scripts\K3_mesh_pipeline.py` is a single, stable, self-contained function, demonstrated end-to-end on real Vaihingen data at three scales (small/medium/large — `K3_mesh_pipeline.json`) plus one decimated LOD variant that was also confirmed to load correctly in a three.js browser context (`K4_viewer_browser_test.json`). No roadmap item proposes changing its signature soon. Freezing this means: a future change to `heightfield_to_mesh()`'s inputs/outputs requires a `v2` fixture and a note to Ashmita/Aishee (the declared consumers), not a silent edit — it does not mean the pipeline is finished or that every caveat inside the fixture is resolved.

Two caveats consumers must know even though the contract is frozen:

- **Elevation is rebased, not absolute.** `Z = (elev - elev.min()) * z_exaggeration` — the mesh's vertical origin is the tile's own minimum, not the source GeoTIFF's datum. A viewer needing true absolute height must add `elev.min()` back from outside the `.glb`.
- **"One per requested LOD level" is not real yet.** The handbook's Part X phrasing implies a formal multi-LOD system. Code inspection found only one ad-hoc decimation call (`trimesh.simplify_quadric_decimation`, 10.67M → 500K faces, 15.48s) — `heightfield_to_mesh()` itself has no LOD-level parameter, only a generic integer `downsample` stride. This is flagged as a handbook-vs-code discrepancy inside the fixture; treat "request N LOD levels" as unbuilt when planning FPS-harness work.

Note on provenance: the numbers in this fixture came from a run that fed `heightfield_to_mesh()` from the **ground-truth** DSM. That was the state when the fixture was frozen. Since DW-S1-02 the end-to-end wiring exists and has been exercised on real calibrated predictions — 19 rasters, all consumed by `heightfield_to_mesh()` (`10_DOCUMENTATION/DW_S1_02_RUN_EVIDENCE.md`) — so the gap this paragraph originally described is closed. The mesh numbers themselves are not re-measured here and remain the frozen ground-truth-fed values.

## What was deliberately NOT invented

- No LOD-level parameter was added to the documented contract because none exists in code.
- No single "calibration → GeoTIFF" function was documented as if it existed — the two real, disconnected pieces are documented separately instead.
- No absolute-elevation-preservation behavior was claimed for the mesh output, because the code actively discards it (rebases to tile minimum).
- FPS numbers are marked `fps_measured: false` everywhere, per `CURRENT_STATE.md`'s standing instruction never to round browser-load success up to "smooth" or "fast."

## Freezing this properly (next step, not done here)

**Status update.** All three preconditions this section listed have since been met, under DW-S1-02 (#5) and DW-S1-06 (#15): (1) `geoio.save_geotiff()` is a real reusable function and both producers call it; (2) the relative branch was scoped in and built (`src/relative_dsm.py`); (3) the NoData convention is `-9999.0`, written as the `GDAL_NODATA` tag rather than left reader-side.

Both boundaries nonetheless remain **v0**, for reasons unrelated to the list above: the metric half has not had downstream-consumer review of its field set, and the relative half has not had Part IV item (c), qualitative evaluation on real non-georeferenced photographs. v1 is a review decision now, not an implementation one.
