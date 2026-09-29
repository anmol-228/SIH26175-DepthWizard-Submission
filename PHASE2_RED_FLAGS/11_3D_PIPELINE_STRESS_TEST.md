# 11_3D_PIPELINE_STRESS_TEST.md — Experiment K: 3D/Rendering/Deployment Risk

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

Phase 1 only produced one static sanity-check figure (matplotlib 3D surface plot of a single calibrated patch) — never stress-tested at scale, never a real interactive viewer. This experiment builds and stress-tests a technically representative prototype: GeoTIFF/DSM -> textured mesh -> georeferenced export -> three.js interactive flythrough, using REAL Vaihingen imagery/DSM throughout (area 26, the "typical" tile per Phase-1's own error analysis).

## K1 — Large-image tiling & stitching (Section 33)

Ran the fine-tuned model on a FULL Vaihingen tile (1783x2995px, 160m x 270m — not a pre-cropped 512x512 patch) via sliding-window inference (512px window, 384px stride, 128px overlap), comparing hard-paste (no blending) vs. feather-blended reassembly.

| | Time (32 windows) | Peak mem | Boundary/interior gradient ratio | RMSE (oracle-calibrated) |
|---|---:|---:|---:|---:|
| Hard-paste (no blend) | 1.31s | 136 MB | **5.97x** (visible seams) | 4.16m |
| Feather-blended | (same pass) | 136 MB | **0.83x** (seams eliminated) | 4.75m |

Raw data: `outputs/metrics/K1_tiling_stitch.json`, figure: `outputs/figures/K1_tiling_area26.png`.

**Finding:** naive hard-paste tiling produces a real, measurable seam artifact (gradient magnitude 6x higher at former window boundaries than tile interior — visibly a grid pattern in the reconstructed DSM). Feather-blending with a linear overlap ramp eliminates this almost entirely (boundary gradient actually slightly BELOW interior). **Trade-off, disclosed honestly:** blending costs a small amount of accuracy (RMSE 4.75m vs 4.16m, +14% relative) because averaging two overlapping window predictions smooths some real local variation near boundaries along with the seam. This is a legitimate, worthwhile trade for a system whose output includes a 3D visualization (seams would be visually obvious and worse for demo quality than a small accuracy cost) — recommend blending, disclose the accuracy trade-off. Both memory and runtime are trivial (136MB, 1.3s for a 270m x 160m real scene) — full-scene tiling is not a compute risk at this scale.

## K2 — Georeferencing preservation (Section 34)

Exported a predicted DSM as a proper GeoTIFF (embedded ModelPixelScaleTag/ModelTiepointTag/GeoKeyDirectoryTag declaring EPSG:32633, using Vaihingen's own internally-consistent local coordinate system recovered from `Vaihingen_dsm_tiles_geoinfo.zip`), then reopened it independently and verified every field.

> **ERRATUM (2026-08-30, issue #14).** The `EPSG:32633` recorded above is the value that was actually written at the time, taken from the dataset's own `dsm.vrt` (`PROJCS["WGS 84 / UTM zone 33N"]`). It is wrong: the coordinates are consistent with UTM zone **32N geometry** - easting 496794.825 implies ~8.96E under 32N (Vaihingen), versus ~14.96E under 33N, 6 degrees and roughly 440 km east at this latitude. The datum/authority remains unproven, so current exports write `ProjectedCSTypeGeoKey = 32767` (user-defined) rather than substituting 32632 or 25832. **This is a metadata-assignment correction, not a reprojection:** the coordinates are unchanged, no raster was transformed, and no DSM or calibration metric is affected. The value above is left as recorded, deliberately - it is what the run produced.


**Result: ALL CHECKS PASS** — pixel size, origin, shape, and EPSG GeoKey all round-trip exactly. Raw data: `outputs/metrics/K2_georeferencing_check.json`. Mechanically, the export/reopen pipeline works correctly. (Separate, already-disclosed finding: Vaihingen's own declared absolute coordinates are not trustworthy — see `02_DATASET_RESEARCH.md` — this test validates the EXPORT MECHANISM, not the source data's absolute correctness, which is a different question already answered elsewhere.)

## K3 — Mesh generation at small/medium/large scale (Sections 35-36)

Real textured meshes (heightfield -> triangulated grid -> UV-mapped RGB texture -> glTF/.glb export via `trimesh`) at three real scales from the same area-26 tile:

| Scale | Input | Vertices | Faces | Gen time | Export time | File size |
|---|---|---:|---:|---:|---:|---:|
| small | 512x512 patch | 262,144 | 522,242 | 0.01s | 0.09s | 11.9 MB |
| medium | full tile, 2x downsample | 1,336,216 | 2,667,654 | 0.06s | 0.43s | 61.2 MB |
| large | full tile, native 9cm | 5,340,085 | 10,670,616 | 0.22s | 1.81s | 242.8 MB |

Raw data: `outputs/metrics/K3_mesh_pipeline.json`. Mesh generation itself is fast at every scale (well under 2s even for the largest); no crash, no watertight-mesh requirement violated (open meshes are expected/fine for a heightfield, not a closed solid). **The real risk is not generation time — it's the resulting file being enormous at native resolution** (242.8MB for one 160x270m tile is not viable for a live browser demo, in file size alone, independent of rendering performance).

**LOD mitigation (Section 37), tested and working:** quadric-decimated the "large" mesh from 10.67M -> 500K faces (21.4x reduction) via `trimesh.simplify_quadric_decimation` (fast_simplification backend), taking 15.5s, producing a 9.0MB file (26.9x smaller). This is a genuinely working, fast-enough-to-run-server-side mitigation, not just a theoretical recommendation.

## K4 — Interactive flythrough browser test (Sections 35-37)

Built a real three.js (r160, GLTFLoader + OrbitControls) flythrough page and tested all 4 mesh variants (small/medium/large/large_LOD) in the sandboxed browser tool, served over a local HTTP server.

**Results: no crash, no console error, and successful load for all 4 variants** including the un-decimated 10.67M-triangle mesh — load times (localhost): small 0.05s, medium 0.27s, large 0.63s, large_LOD 0.03s. Full detail: `outputs/metrics/K4_viewer_browser_test.json`.

**Honest limitation, explicitly disclosed:** the sandboxed browser pane never entered a visible/composited state in this session (`document.visibilityState` stayed `"hidden"` throughout), which per browser spec throttles `requestAnimationFrame` — so **actual rendered FPS could not be measured** in this tooling environment. This is a tool limitation, not a claim about the mesh. What WAS measured (load/parse/GPU-upload success, timing, file size) is real. General WebGL practice puts a smooth-60fps interactive budget at roughly 1-3M triangles on typical hardware — well below the "large" variant's 10.67M — so real-time frame rate at that polycount should be assumed poor on typical judge-demo hardware even though it could not be directly confirmed here. **Recommendation for the team: re-run this exact `viewer.html` + GLB files in a normal visible browser window during development to get a real FPS number before the event — this is a 5-minute check, explicitly flagged as not yet done.**

## Answers to Section 47 Questions 7-8

- **Q7 (full-scene without seams)?** YES, with feather-blending (demonstrated, small accuracy trade-off disclosed).
- **Q8 (rendered interactively)?** PARTIAL / UNRESOLVED on the specific FPS question due to a tooling limitation in this session; mesh generation, export, LOD decimation, and browser load/parse all demonstrated working with real timings. The team should independently confirm real FPS before relying on this for a live demo.
