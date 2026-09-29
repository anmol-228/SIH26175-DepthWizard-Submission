# DEPTHWIZARD BASELINE v1.0

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Update 2026-09-29:** this file freezes the **B0 (iter2)** baseline. Since PR #109 the runner default is the v3 full-encoder EMA checkpoint (`da2_small_finetuned_v3_full_encoder_ema.pt`, `a1e3fca3...9ea7`), selected under TRAIN-only protocols and never evaluated on dev/test. **None of the numbers below is a v3 result;** B0 remains the opt-in rollback (`DW_CHECKPOINT` + `DW_CHECKPOINT_SHA256`).

> **Doc-integrity note (2026-09-19):** This file cites `CURRENT_STATE.md` and/or `EVIDENCE_LEDGER.md` below as existing sources. Neither file exists anywhere in this repo — confirmed via `git log --all` across every branch (zero commits, ever) and independently cross-checked on GitHub. `CURRENT_STATE.md`'s content (results, known gaps, workspace/repro setup) lives in `BASELINE_v1.0.md`; nothing in this repo currently serves as `EVIDENCE_LEDGER.md`'s claim-to-source-with-status-label table — the closest real, currently-existing equivalents are `EXPERIMENT_INDEX.md`, `FINAL_CLAIM_EVIDENCE_MATRIX.md`, and `PHASE2_RED_FLAGS/17_FINAL_QA.md`. This was already flagged once, for one specific note, in `10_DOCUMENTATION/DW_S1_03_SPARSE10_PROVENANCE.md` (2026-08-29) but not corrected repo-wide. Flagging here rather than silently repointing every individual inline reference below — needs an owner decision: create the real `EVIDENCE_LEDGER.md`, or update each reference to its correct current target.

Canonical, reproducible snapshot of DepthWizard's current evidence and working paths, frozen at Sprint 0 per `(internal note, not included)`, before any new architecture changes in Sprint 1. Owners: the model and calibration leads.

Every number below was re-read directly from the underlying JSON/report files in this folder on 2026-08-26, not copied from `CURRENT_STATE.md` from memory — see the source path next to each number. Where a value could not be independently re-verified against a raw file, it is marked as such.

**Status labels used below follow the project's status-label convention: VERIFIED / PARTIALLY VERIFIED / PROPOSED / UNKNOWN / FAILED. Oracle, sparse, simulated, and proxy results are never merged into one number without saying which is which.**

---

## 1. Environment

Source: `01_ENVIRONMENT_AUDIT.md`, `requirements.txt`.

- OS: Windows 11 Pro (build 10.0.26200)
- Python: 3.14.6 (system default, no conda/uv/pyenv)
- GPU: NVIDIA GeForce RTX 5060 Ti, 16,311 MiB VRAM, driver CUDA 13.3 runtime, compute capability sm_120 (Blackwell)
- PyTorch: `torch==2.9.0+cu129` (installed from the `cu129` wheel index, not PyPI default — required for Blackwell support)
- No GDAL/rasterio anywhere in the stack — GeoTIFF I/O is a hand-rolled `tifffile` + `.tfw`-parsing module (`src\geoio.py`), deliberately, because no verified GDAL/rasterio build existed for Python 3.14 at experiment time
- Full pinned dependency list: `requirements.txt` (57 packages, e.g. `transformers==5.15.1`, `scikit-learn==1.9.0`, `tifffile==2026.8.16`, `matplotlib==3.11.1`)
- `.venv\` kept installed at the workspace level specifically so this reproduces without a fresh `pip install` on demo day (`CURRENT_STATE.md` workspace map)
- **Reproduction command:**
  ```bash
  python -m venv .venv
  .venv/Scripts/pip install -r requirements.txt
  ```
- **NOT independently re-run in this pass** — this baseline freeze verified the environment audit document and `requirements.txt` exist and are internally consistent; it did not re-execute `pip install` from a clean venv to prove today's `.venv` still matches. STATUS: PARTIALLY VERIFIED (documented, not re-executed this session).

## 2. Model / checkpoints

Source: `07_RESOURCE_PROFILE.md`, `configs\finetune_iter1_frozen_backbone.json`, `configs\finetune_iter2_partial_unfreeze.json`, `EXPERIMENT_INDEX.md`.

- Base model: Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`), 24.8M total params (22.06M frozen DINOv2-Small backbone + 2.73M trainable neck/head)
- Preprocessing bypasses `transformers`' `AutoImageProcessor` (hard-requires torchvision, unverified on this Python/CUDA combo at experiment time) — manual reimplementation in `src\da2_utils.py`: resize 518×518 bicubic, rescale to [0,1], ImageNet mean/std normalize. Matches the model repo's own `preprocessor_config.json`.
- Two checkpoints exist in `checkpoints\`, both confirmed present on disk:
  - `checkpoints\da2_small_finetuned_best.pt` — iteration 1 (frozen backbone, neck+head only trained; 8 epochs, batch 4, lr 5e-5; `configs\finetune_iter1_frozen_backbone.json`; best val SSI-L1 = 1.1041 m; ~205 s wall-clock). Per `EXPERIMENT_INDEX.md`, this file is identical to what the config calls `da2_small_finetuned_iter1_frozen_backbone.pt` (naming discrepancy between the config's recorded checkpoint name and the actual file on disk — noted, not a missing artifact).
  - `checkpoints\da2_small_finetuned_iter2_partial_unfreeze.pt` — iteration 2 (last 2 of 12 DINOv2 blocks unfrozen at lr 5e-6, alongside neck+head at lr 5e-5; 16 epochs; `configs\finetune_iter2_partial_unfreeze.json`; best val SSI-L1 = 1.0001 m; ~440 s wall-clock). Config note: "Reported as FINAL headline result — marginal improvement over iteration 1, diminishing returns, no further iteration attempted."
- Loss: scale-and-shift-invariant L1, closed-form OLS per-sample (`src\metrics.py: fit_scale_shift`). **Real engineering finding kept, not smoothed over:** the naive `torch.linalg.lstsq`-based implementation caused an immediate 268GB CUDA OOM on the first backward pass; fixed by switching to the O(N) closed-form mean/covariance solution.
- Training hardware: local RTX 5060 Ti, peak VRAM ~2.7GB, $0 spent (no cloud compute).

## 3. Datasets / splits

Source: `02_DATA_AUDIT.md`, `data\splits\split_v1.json`, `PHASE2_RED_FLAGS\02_DATASET_RESEARCH.md`, `EXPERIMENT_INDEX.md`.

- **Primary (urban):** ISPRS Vaihingen, 9cm GSD, Germany — **academic-only license, do not redistribute.** Not included in this repo (879MB); re-download per `02_DATA_AUDIT.md`'s verified source link. Split defined in `data\splits\split_v1.json` (keys: `seed`, `method`, `train_areas`, `val_areas`, `test_areas`, `n_train`, `n_val`, `n_test`) — area-disjoint, not just patch-disjoint.
- 513 train patches / 42 val patches / 132 test patches (512×512), per `07_RESOURCE_PROFILE.md`.
- **Forest:** Sentinel-2 + ETH Global Canopy Height, one Pacific Northwest AOI plus a nearby separate AOI for the generalization check (`PHASE2_RED_FLAGS\D0_prepare_forest_data.py` etc.) — one biome only, explicitly not India-tested.
- **Hilly:** USGS 3DEP + NAIP, Mount Adams W flank, WA, 28 patches (`E_hilly_terrain_eval.json`).
- **India sanity check:** public basemap imagery + Copernicus GLO-30 as a rough reference over Bengaluru — qualitative only, not a quantitative validation dataset (`G_india_sanity_check.json`).
- Vaihingen's absolute real-world geolocation is **not confirmed trustworthy** even though its internal local coordinate system is self-consistent (`PHASE2_RED_FLAGS\02_DATASET_RESEARCH.md`) — this matters for the GeoTIFF fixture (see `fixtures\README.md`).

## 4. Urban results (Vaihingen)

Source: `outputs\metrics\baseline_constant.json`, `outputs\metrics\zeroshot_3B_oracle_calibrated.json`, `outputs\metrics\finetuned_calibration_report_iter2_partial_unfreeze.json`.

| Result | RMSE | MAE | Corr | Category | Status |
|---|---|---|---|---|---|
| Global constant-mean baseline | 18.146 m | 15.668 m | n/a | trivial baseline | VERIFIED |
| Global constant-median baseline | 11.473 m | 9.340 m | n/a | trivial baseline | VERIFIED |
| Zero-shot DA2 + one global oracle calibration, test split | 18.505 m | 16.096 m | 0.350 | oracle, global (not per-tile) | VERIFIED |
| **Fine-tuned (iter2) + full per-tile oracle calibration, global-pooled test** | **4.4034 m** | 3.0421 m | 0.9135 | **oracle — non-deployable ceiling** (dense per-tile ground truth from the scored scene) | VERIFIED |
| **Fine-tuned (iter2) + sparse-10-point calibration, global-pooled test** | **5.5006 m** (`finetuned_calibration_report_iter2_partial_unfreeze.json`, matches `CURRENT_STATE.md` exactly) — **see discrepancy note directly below: a second file reports 4.9139 m for the nominally same setup** | 3.803 m | 0.876 | more deployment-like — still samples the test tile's own DSM, not field/SRTM-independent | VERIFIED (headline number); see discrepancy note |

**Discrepancy found during this freeze — update 2026-08-29 (issue #6 / DW-S1-03):** the sparse-10 urban RMSE gap (**5.5006 m** in `finetuned_calibration_report_iter2_partial_unfreeze.json`'s `sparse_stress_test."10".global` block, vs. **4.9139 m** in `PHASE2_RED_FLAGS\outputs\metrics\A1_gsd_measure_no_adaptation.json`'s `"0.09".global_sparse10` block) has a confirmed structural divergence point, though full numeric re-reproduction remains blocked. Every input parameter was checked line-by-line and confirmed identical between `scripts\07_calibrate_and_eval.py` and `PHASE2_RED_FLAGS\scripts\A1_gsd_measure.py`: same checkpoint (`da2_small_finetuned_iter2_partial_unfreeze.pt`), same test manifest/split, same 9 areas in the same sorted order, same RNG seed (42), same RANSAC(min_samples, random_state=42) config, and `PHASE2_RED_FLAGS\scripts\gsd_utils.py` contributes no RNG usage at all. The one thing that differs: **RNG consumption history.** `07`'s shared `rng` object has already drawn 9 areas × (200-point + 50-point) samples before it reaches the `n=10` draw being compared; `A1`'s `n=10` draw at native GSD (0.09) is the very first time its `rng` object is touched. Same seed does not mean same draw once a generator has been advanced — this was verified by actually running an isolated reproduction of the exact call sequences from both real scripts (not guessed): `10_DOCUMENTATION\DW_S1_03_SPARSE10_PROVENANCE.md` and `10_DOCUMENTATION\evidence\DW_S1_03_rng_repro.py`/`_output.txt`. **Not yet closed:** the checkpoint file needed to literally re-run either script and reproduce 5.5006 m / 4.9139 m end-to-end exists only on Anmol's machine (never uploaded externally per `ARTIFACT_MANIFEST.md`) — full DoD item 1 needs that checkpoint shared or the re-run done by whoever has it. Outcome classification (A-E) and final status label are Debopriya's call per this issue's review path, not asserted here. (Note: this note is referenced elsewhere in the repo as living in "`CURRENT_STATE.md`" — no file by that name exists; this is the actual location.)

> **ERRATUM (2026-08-30, issue #6 — Outcome D, resolved).** The paragraph above predates the
> final classification and is left as recorded; this note supersedes its open items without
> rewriting it. PR #18 merged into `main` (merge commit `c618174`, PR head `20dea5b`), with
> both required reviews approved at that head: methodology/experiment-interpretation (Anmol)
> and evidence/release (Debopriya). **Outcome: D — both historical results reproduce; they
> differ because the shared RNG stream is at a different position when each script draws its
> 10 calibration points**, not because of any difference in data, split, checkpoint, area
> set/order, seed value, or RANSAC config (all independently confirmed identical).
>
> The "not yet closed" checkpoint blocker above is resolved: Anmol independently reproduced
> both numbers end-to-end on his own machine against the real checkpoint
> (`da2_small_finetuned_iter2_partial_unfreeze.pt`, SHA-256 `74ccde40…78d8`) and the real 132
> test patches — canonical `07`-order → **5.5006 m**, `A1`-order at native GSD → **4.9139 m**,
> full per-tile oracle (no RNG) → **4.4034 m**, all matching history to high precision. The
> precise mechanism: `07_calibrate_and_eval.py` reaches its N=10 sampling round only after the
> shared generator has already served the N=200 and N=50 rounds — **18 prior per-area
> `choice()` calls** (9 areas × 2 preceding sample sizes) — before the N=10 draw;
> `A1_gsd_measure.py` at native GSD makes its N=10 draw as the generator's first use.
>
> **Patch pooling / array-concatenation order — floated above as a candidate cause — is not
> the cause; that hypothesis is superseded.** Pooling, split, and area order are confirmed
> identical between the two scripts; the isolated RNG-consumption reproduction
> (`10_DOCUMENTATION\DW_S1_03_SPARSE10_PROVENANCE.md`,
> `10_DOCUMENTATION\evidence\DW_S1_03_rng_repro.py`/`_output.txt`) demonstrates the actual
> mechanism directly, not by elimination.
>
> **5.5006 m remains the canonical sparse-10 figure** (canonical calibration script, frozen
> Phase-1 baseline). **4.9139 m remains a separate, legitimate, reproducible result** from a
> distinct RNG-stream position inside the GSD-stress harness at native resolution — not an
> independent accuracy measurement, and not averaged, substituted, or relabelled against the
> canonical figure. Sparse-10 reference points are still sampled from the test tile's own DSM,
> not independent field/SRTM truth — that qualification, from the original paragraph, is
> unchanged.
>
> Issue #6 stays open — not because the classification above is in question, but for
> recipient-side reproducibility/handoff closure: PR #22 (authorized regeneration path for the
> processed test patches) is under review, and Debopriya's own independent recipient-side
> acquisition, regeneration, and rerun are still pending her network access. None of that
> reopens Outcome D.

## 5. Forest results (one Pacific Northwest biome — not India-tested)

Source: `PHASE2_RED_FLAGS\outputs\metrics\D1_baseline_forest_eval.json`, `D1b_constant_baseline_forest.json`, `D2_forest_finetuned_eval.json`, `D2b_generalization_check.json`.

| Result | RMSE | Category | Status |
|---|---|---|---|
| Constant-baseline, forest | 6.937 m | trivial baseline | VERIFIED |
| Zero-shot DA2 (untouched), forest, global-pooled | 6.700 m | zero-shot, no forest adaptation | VERIFIED |
| Vaihingen-finetuned model, zero-shot transfer to forest | 6.806 m | domain transfer without forest-specific tuning | VERIFIED |
| Forest-targeted adaptation, same-AOI held-out, global-pooled | **2.867 m** | adapted, same area held out spatially | QUALIFIED (small sample; same AOI only) |
| Forest-targeted adaptation, nearby separate AOI, global-pooled | **3.703 m**, corr 0.919 (zero-shot-on-forest baseline for comparison: 8.056 m) | adapted, transfer to a different nearby AOI, same biome, no retraining | QUALIFIED (no retraining tested; same biome/source family only) |

## 6. Hilly terrain — confirmed severe failure (load-bearing negative result, not softened)

Source: `PHASE2_RED_FLAGS\outputs\metrics\E_hilly_terrain_eval.json`, `07_HILLY_TERRAIN_TEST.md`, `E_adaptation_attempt1.json`.

- **Zero-shot, whole-scene sparse-10-point calibration (realistic deployment simulation), pooled across 5 seeds:** RMSE mean **69.55 m**, std 5.24 m, range 64.02–78.45 m.
- **Vaihingen-finetuned model, same whole-scene sparse-10 test:** RMSE mean **67.66 m**, std 2.62 m, range 63.49–70.37 m.
- **Per-patch oracle ceiling (dense ground truth per small patch — NOT deployable):** zero-shot mean-of-patches RMSE 11.99 m (std 10.25 m, range 3.03–47.98 m); finetuned mean-of-patches RMSE 11.08 m (std 13.43 m, range 2.98–62.33 m).
- **Full-oracle single global calibration across the whole scene:** RMSE 60.20 m (zero-shot) / 61.94 m (finetuned) — even a single best-fit oracle scale/shift across the whole heterogeneous-relief scene fails badly; this is not just a sparse-sampling artifact.
- **Slope-stratified breakdown (oracle, zero-shot):** 0–5° RMSE 45.49 m → 30–90° RMSE 131.21 m — error grows sharply with slope even under oracle calibration, consistent with a single global linear scale/shift being structurally unable to fit heterogeneous relief.
- **Pooled correlation collapses to 0.05–0.24** despite individual patches showing 0.94–0.97 local correlation — a textbook Simpson's-paradox-like signature: the model tracks *relative* relief well within a patch, but patch-to-patch scale/offset varies enough that pooling destroys the correlation.
- **One adaptation attempt (17 training patches, neck/head only) showed no material improvement** — `E_adaptation_attempt1.json`. VERIFIED (negative result).
- **CURRENT_STATE.md's "~60-70 m RMSE realistic" claim is directly confirmed** by the sparse-10 numbers above (64–78 m range, mean 67.7–69.6 m). Its "~11-12 m" oracle-ceiling claim is also confirmed (11.08–11.99 m mean-of-patches). STATUS: VERIFIED — LOAD-BEARING FAILURE, do not omit from the pitch.

## 7. Calibration point-placement sensitivity

Source: `PHASE2_RED_FLAGS\outputs\metrics\F_calibration_distribution.json`.

At N=10 reference points per tile (urban, Vaihingen):

| Placement strategy | Mean RMSE | Std | Range |
|---|---|---|---|
| Terrain-stratified | **4.70 m** | 0.149 | 4.52–4.99 m |
| Uniform random | 4.86 m | 0.255 | 4.48–5.46 m |
| Edge-biased | 6.69 m | 0.462 | 5.96–7.68 m |
| Clustered | **15.32 m** | 7.124 | 6.75–**28.42 m** |

At N=20, the same ordering holds (stratified 4.55 m best, clustered 10.97 m worst) — **point placement matters far more than point count.** VERIFIED, matches `CURRENT_STATE.md` exactly.

## 8. Simulated GSD (satellite-resolution) stress — not genuine sensor transfer

Source: `PHASE2_RED_FLAGS\outputs\metrics\A1_gsd_measure_no_adaptation.json`, `A2_gsd_adapted_3.0m.json`.

- Native 9cm: oracle 4.403 m / sparse-10 4.914 m (same underlying numbers as Section 4).
- Idealized synthetic downsample to simulated 5m GSD (same imagery/truth, not real satellite physics): oracle degrades, sparse-10 degrades further (see `A1_gsd_measure_no_adaptation.json` for the full per-GSD table).
- Retraining at simulated 3m GSD: oracle 5.089 m, sparse-10 5.919 m (`A2_gsd_adapted_3.0m.json`) — partial recovery via adaptation, still synthetic resampling of the same source imagery, **not** a real lower-resolution sensor. STATUS: QUALIFIED (simulated, not genuine satellite domain transfer).

## 9. India sanity check — qualitative only

Source: `PHASE2_RED_FLAGS\outputs\metrics\G_india_sanity_check.json`.

Rough comparison of model prediction range vs. Copernicus GLO-30 over a Bengaluru bounding box: RMSE 6.02 m against GLO-30 as a rough reference, corr 0.010 (near-zero — the model's relative depth range and GLO-30's absolute elevation range are not expected to align without calibration; this was never intended as a quantitative validation). STATUS: UNRESOLVED (qualitative only), consistent with `CURRENT_STATE.md`.

## 10. Tiling / seam mitigation

Source: `PHASE2_RED_FLAGS\outputs\metrics\K1_tiling_stitch.json`.

- One full Vaihingen tile (area 26, 1783×2995), 512-window/384-stride, 32 blended windows vs. 15 hard-paste windows.
- Seam gradient ratio (boundary-mean / interior-mean gradient): hard-paste **5.969×** → blended **0.832×**.
- Accuracy cost of blending: RMSE 4.161 m (hard paste) → 4.747 m (blended) — a real, disclosed accuracy-for-seam-quality tradeoff, not a free win. VERIFIED, single tile tested.

## 11. GeoTIFF round-trip

Source: `PHASE2_RED_FLAGS\outputs\metrics\K2_georeferencing_check.json`.

Area 26, pixel size (0.09, -0.09) m, origin (496794.825, 5419752.255), shape (1783, 2995), EPSG:32633 GeoKey — all four checks pass (pixel size, origin, shape, EPSG GeoKey preserved). VERIFIED, mechanics only — see `fixtures\calibration_output_v0.json` for why this is not yet wired to the live calibration output, and see Section 3 above for the caveat that Vaihingen's absolute geolocation is not itself trustworthy.

> **ERRATUM (2026-08-30, issue #14).** The `EPSG:32633` recorded above is the value that was actually written at the time, taken from the dataset's own `dsm.vrt` (`PROJCS["WGS 84 / UTM zone 33N"]`). It is wrong: the coordinates are consistent with UTM zone **32N geometry** - easting 496794.825 implies ~8.96E under 32N (Vaihingen), versus ~14.96E under 33N, 6 degrees and roughly 440 km east at this latitude. The datum/authority remains unproven, so current exports write `ProjectedCSTypeGeoKey = 32767` (user-defined) rather than substituting 32632 or 25832. **This is a metadata-assignment correction, not a reprojection:** the coordinates are unchanged, no raster was transformed, and no DSM or calibration metric is affected. The value above is left as recorded, deliberately - it is what the run produced.


## 12. Mesh / LOD

Source: `PHASE2_RED_FLAGS\outputs\metrics\K3_mesh_pipeline.json`, `K4_viewer_browser_test.json`.

| Variant | Faces | Vertices | File size | Gen+export time |
|---|---|---|---|---|
| small (512×512, no downsample) | 522,242 | 262,144 | 11.90 MB | 0.10 s |
| medium (full tile, downsample=2) | 2,667,654 | 1,336,216 | 61.19 MB | 0.50 s |
| large (full tile, native res) | **10,670,616** | 5,340,085 | **242.75 MB** | 2.04 s |
| large_LOD (quadric decimation of large) | 500,000 | 250,472 | 9.01 MB | 15.48 s decimation |

All 4 variants loaded successfully in a three.js `GLTFLoader` + `WebGLRenderer` context with 0 console errors. **True visible-browser FPS was NOT measured** — the sandboxed test pane kept `document.visibilityState` at `hidden`, which throttles `requestAnimationFrame`; only load/parse/GPU-buffer-upload success is a real measured result. Only ONE LOD variant was actually produced (not a formal multi-level LOD ladder) — see `fixtures\pipeline_to_3d_v1.json` for the full handbook-vs-code discrepancy on this point.

## 13. Known gaps (UNKNOWN / UNPROVEN — do not claim solved)

- **Genuine satellite-sensor transfer:** UNKNOWN. DFC2019/US3D access was never accepted (license-gated); no real satellite + independent-elevation-truth run exists anywhere in this repo.
- **Viewer FPS in a real, visible browser:** UNKNOWN. Never benchmarked — see Section 12.
- **India-wide / broader forest generalization:** UNKNOWN beyond the one tested Pacific Northwest biome (Section 5) and the qualitative India check (Section 9).
- **Real external-DEM systematic bias:** UNKNOWN — only IID point noise was tested in calibration-noise experiments (`F_calibration_noise.json`), not spatial/systematic bias of the kind a real SRTM/GEDI reference would actually have.
- **Calibration → GeoTIFF wiring:** not implemented end-to-end — see `fixtures\calibration_output_v0.json`.
- **Formal multi-level LOD:** not implemented — see `fixtures\pipeline_to_3d_v1.json`.

## 14. Reproducible commands

Source: `README.md`, verified consistent with `scripts\` filenames and `EXPERIMENT_INDEX.md`.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
python scripts/01_data_audit.py        # requires data/raw/Vaihingen/{top,dsm}/ — see 02_DATA_AUDIT.md for source
python scripts/02_make_splits.py
python scripts/04_make_patches.py
python scripts/05_zeroshot.py
python scripts/06_finetune.py                                    # iteration 1, frozen backbone
FT_EPOCHS=16 FT_UNFREEZE_LAST_N=2 FT_CKPT_NAME=da2_small_finetuned_iter2_partial_unfreeze.pt \
  python scripts/06_finetune.py                                  # iteration 2, partial-unfreeze
EVAL_CKPT_NAME=da2_small_finetuned_iter2_partial_unfreeze.pt \
  EVAL_OUT_NAME=finetuned_calibration_report_iter2_partial_unfreeze.json \
  python scripts/07_calibrate_and_eval.py
python scripts/08_mini_3d_mesh.py
python scripts/09_error_analysis.py
```

`PHASE2_RED_FLAGS\` scripts (A–K, G) each read Vaihingen/forest/hilly data from their own `PHASE2_RED_FLAGS\data\` locations and are individually runnable (`python PHASE2_RED_FLAGS/scripts/<name>.py`); see `PHASE2_RED_FLAGS\03_PHASE2_EXPERIMENT_PLAN.md` for the full per-experiment command list. **Not re-executed end-to-end in this baseline-freeze pass** — this was a read-and-verify-against-raw-output pass, not a rerun, per the Sprint-0 scope (rerunning everything is Debopriya's standing per-sprint reproduction duty per `CLAIM_RELEASE_POLICY.md`, not a one-time baseline-freeze task).

All scripts are seeded (`SEED = 42`).

---

## Gate

Per `SPRINT_0_DUAL_TRACK_LAUNCH.md`: DepthWizard does not start serious architecture changes in Sprint 1 until this baseline is frozen and reproducible. This document is that freeze as of 2026-08-26. Any future claim of improvement must be measured against the numbers in this file, not against a vaguer prior belief.
