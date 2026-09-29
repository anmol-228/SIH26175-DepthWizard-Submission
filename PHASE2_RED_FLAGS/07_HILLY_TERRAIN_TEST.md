# 07_HILLY_TERRAIN_TEST.md — Experiment E: Genuine Hilly/Mountainous Terrain

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

Answers Red Flag #3 (`01_RED_FLAG_REGISTER.md`): Phase-1 had ZERO evidence either
way about hilly/mountainous terrain (Vaihingen elevation range 240–360m,
flat-to-gently-rolling only). This experiment acquires a real hilly-terrain
RGB+elevation benchmark from scratch and evaluates both the zero-shot
pretrained model and the Phase-1 Vaihingen-fine-tuned checkpoint on it.

**Bottom line up front:** a genuine, correctly-georeferenced hilly-terrain
dataset WAS obtained (no account creation, no fabricated/misaligned data).
The result is not a pass against the pre-registered ~8.6m threshold. Under
the realistic per-scene calibration regime, RMSE is **60–70m — roughly
7–8× the ~8.6m reference point**, and error rises sharply and monotonically
with slope. Under an artificially generous per-patch-oracle calibration
(each 512×512 crop individually calibrated), the picture is less
catastrophic (RMSE ~11–12m, ~2.5× the Vaihingen ceiling) but still clearly
worse than flat terrain, and still rises steeply with slope. Vaihingen
fine-tuning gives no consistent benefit over zero-shot here (both are
essentially untested-domain for this scene). This is reported honestly as a
genuine, load-bearing finding, not adjusted to fit the pre-registered
threshold.

## 1. Region, real coordinates, CRS

**Region:** west flank of Mount Adams, Washington State, USA — forested
volcanic mountain terrain near Trout Lake, WA, including a visible cinder
cone / satellite volcanic feature at the AOI's southwest corner (confirmed
independently by matching a circular DEM elevation feature to a
correspondingly distinct patch in the RGB imagery at the exact same pixel
location — see `outputs/figures/E_alignment_check.png`).

**AOI extent (native CRS, EPSG:26910 — NAD83 / UTM zone 10N):**
X 596700–598900 m, Y 5084100–5088100 m → **2200m × 4000m (8.8 km²)**.

**AOI corners (WGS84, computed via `pyproj`, `EPSG:26910`→`EPSG:4326`):**
| Corner | Lon | Lat |
|---|---|---|
| NW | -121.75252 | 45.93966 |
| NE | -121.72415 | 45.93935 |
| SW | -121.75333 | 45.90367 |
| SE | -121.72497 | 45.90335 |
| **Center** | **-121.73874** | **45.92151** |

Sanity check: this resolves to a real, named place (Mount Adams / Trout
Lake, Skamania–Klickitat county area, WA) — not an arbitrary or nonsensical
coordinate. Both the DEM tile and the NAIP tile independently declare
`NAD83 / UTM zone 10N` (`EPSG:26910`) in their own embedded GeoTIFF
GeoKeyDirectoryTag — same CRS, so **no reprojection was required**, only a
crop + resample onto a shared 1m grid. This sidesteps the exact failure mode
that made Vaihingen's public geolocation untrustworthy (see
`02_DATASET_RESEARCH.md` §1): here, every coordinate came directly from
each file's own embedded tags, cross-checked against pyproj and a known
real-world location, rather than trusted blindly.

## 2. Data sources, exact URLs, license

| | Elevation (truth) | RGB imagery |
|---|---|---|
| Program | USGS 3DEP LiDAR-derived 1m DEM | NAIP 2023, 0.6m GSD |
| Access | `tnmaccess.nationalmap.gov` API → public `prd-tnm` S3 bucket, no auth | Microsoft Planetary Computer STAC API (no account) + free anonymous SAS-token signing endpoint |
| Tile | `USGS_1M_10_x59y509_WA_Mount_Adams_LiDAR_2016_D16.tif` | `m_4512103_sw_10_060_20230714_20240103.tif` |
| URL | `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/WA_Mount_Adams_LiDAR_2016_D16/TIFF/USGS_1M_10_x59y509_WA_Mount_Adams_LiDAR_2016_D16.tif` | `https://naipeuwest.blob.core.windows.net/naip/v002/wa/2023/wa_060cm_2023/45121/m_4512103_sw_10_060_20230714_20240103.tif` (SAS-signed via `https://planetarycomputer.microsoft.com/api/sas/v1/sign`) |
| License | Public domain (USGS 3DEP) | Public domain (USDA Farm Service Agency, NAIP) |
| Vintage | Lidar collected 2016, DEM published via 3DEP | Flown 2023-07-14 |
| Native res | 1m | 0.6m (RGB channels used; NIR dropped) |

Both downloads verified byte-exact against the server's declared
`Content-Length` (`PHASE2_RED_FLAGS/data/hilly_terrain/raw/fetch_manifest.json`).
No account, login, or credential of any kind was created or used for either
source.

**Important caveat on data acquisition (disclosed honestly):** the first
DEM tile fetched, over a different candidate area (Coyote Wall / Catherine
Creek, Columbia River Gorge, WA — the `WA_FEMAHQ_2018` LiDAR project), was
found to be only ~45% valid within its own 10km×10km cell (patchy nodata,
consistent with that project being a narrow flood-study LiDAR corridor
rather than a full-area survey) and was discarded before any evaluation was
run on it. The `WA_Mount_Adams_LiDAR_2016` project used instead is a proper
area-wide topographic survey; the specific AOI chosen inside it is
**100% valid** (verified by direct nodata-mask inspection, not assumed).

## 3. Note on elevation product type (DTM, not DSM)

3DEP's "1 meter DEM" product is a **bare-earth digital terrain model**
(vegetation/canopy removed), unlike Vaihingen's DSM (surface model,
buildings/canopy included). This AOI is forested, so in tree-covered pixels
the ground truth used here is the terrain surface *beneath* the canopy the
RGB image actually shows — a real, disclosed mismatch that likely
contributes some additional error on top of the terrain-relief effect this
experiment targets. Raw 3DEP lidar point clouds could in principle be
processed into a first-return DSM, but that requires point-cloud tooling
(PDAL/laspy) not available in this environment and multi-GB downloads; not
attempted given the time budget. This caveat does not invalidate the
slope-vs-error finding (§6), which is the primary result, but means the
absolute RMSE numbers likely have some forest-canopy contribution mixed in
with the pure-relief effect.

## 4. Alignment validation

`outputs/figures/E_alignment_check.png` shows the NAIP RGB, the DEM
(terrain colormap), and an RGB+DEM-contour overlay side by side. The
overlay's contour lines trace real visible ridgelines/valleys in the
imagery, and — most convincingly — a distinctive circular elevation feature
in the DEM (a small volcanic cone, visible as concentric contour rings)
lines up exactly with a correspondingly distinct light-colored patch in the
RGB at the same pixel location. This was checked visually before any model
evaluation was trusted, matching Phase-1's alignment-check discipline.

Resampling: DEM kept at native 1m (nearest-pixel crop); NAIP bilinear
-resampled from its native 0.6m grid onto the same 1m grid via
`scipy.ndimage.map_coordinates` (both rasters share `EPSG:26910`, so this is
a pure crop+resample, not a reprojection).

## 5. Patches and terrain statistics

28 non-overlapping 512×512 (px, 1m GSD → 512m×512m) patches extracted from
the AOI, all with ≥98% valid DEM coverage (28/28 candidate patches passed;
manifest: `data/hilly_terrain/patches/manifest.json`).

- **Elevation range across AOI:** 908.9m – 1279.2m (**370.4m relief** in an
  8.8 km² area).
- **Per-patch relief:** ranges from 18.8m (flattest patch) to 289.8m
  (steepest patch, contains the cinder-cone feature) — genuinely
  heterogeneous, not uniformly gentle.
- **Slope** (simple finite-difference gradient on the 1m DEM,
  `degrees(atan(|∇z|))`): mean 12.4°, median 10.0°, p90 26.8°, max 69.2°
  across the AOI. Real, substantial slope variation — this is genuinely
  hilly/mountainous terrain, not "not perfectly flat."

## 6. Results

Two models evaluated: **zero-shot** (original pretrained
`depth-anything/Depth-Anything-V2-Small-hf`, no fine-tuning at all) and
**finetuned-vaihingen** (Phase-1's `da2_small_finetuned_iter2_partial_unfreeze.pt`
checkpoint, applied cross-domain with zero adaptation).

Three calibration regimes, all fit ONLY on calibration points (never on the
pixels being scored, except the explicitly-labeled oracle upper bounds):

- **Per-patch oracle** (best-case diagnostic, NOT realistic): each 512×512
  patch gets its own independently-fit scale+shift from its own full dense
  truth. Reported to isolate whether poor whole-scene results are a
  calibration-granularity artifact or a genuine representational limit.
- **Full-scene oracle** (realistic per-scene calibration): ONE scale+shift
  fit from dense truth pooled across the ENTIRE real AOI/scene — matches
  how a real deployment would calibrate once per acquired satellite scene.
- **Sparse 10-point** (realistic + deployable): 10 randomly-drawn
  calibration points from the whole scene, fit via RANSAC (matching
  `scripts/07_calibrate_and_eval.py`'s methodology), 5 random seeds for
  variance.

| Calibration | Model | RMSE (m) | MAE (m) | corr |
|---|---|---|---|---|
| Per-patch oracle (mean of 28 patches) | zero-shot | 11.99 (σ=10.25, range 3.0–48.0) | — | 0.967 (pooled) |
| Per-patch oracle (mean of 28 patches) | finetuned-vaihingen | 11.08 (σ=13.43, range 3.0–62.3) | — | 0.960 (pooled) |
| **Full-scene oracle (realistic)** | zero-shot | **60.20** | 44.43 | **0.241** |
| **Full-scene oracle (realistic)** | finetuned-vaihingen | **61.94** | 45.89 | **0.053** |
| Sparse 10-pt, RANSAC (5-seed mean) | zero-shot | 69.55 (σ=5.24) | — | — |
| Sparse 10-pt, RANSAC (5-seed mean) | finetuned-vaihingen | 67.66 (σ=2.62) | — | — |

**Phase-1 comparison anchors** (from `00_PHASE2_STATE.md`, Vaihingen
flat/urban terrain): fine-tuned + full per-tile oracle = **4.40m** RMSE;
fine-tuned + sparse-10 = **5.50m** RMSE. Pre-registered "reasonably strong"
reference point: calibrated RMSE staying within ~2× that range, i.e.
**roughly under ~8.6m**.

**Against that threshold:** the realistic full-scene-oracle and sparse-10
numbers (60–70m) are **roughly 7–8× the ~8.6m reference point** — this does
NOT clear the pre-registered bar, by a wide margin. The best-case
per-patch-oracle number (11–12m mean) is closer — about **2.5× the
Vaihingen full-oracle ceiling** — still short of "within ~2×," though not
catastrophically so, and several individual low-relief patches (18–33m
local relief) land close to or even inside Vaihingen's normal 3–8m
patch-level error band.

**Why the gap between per-patch and full-scene calibration is so large:**
Depth Anything V2 (like all monocular relative-depth models) only predicts
depth up to an unknown per-image scale and shift — consistent *within* one
inference, not *across* independently-run patches. When 28 patches spanning
370m of real relief (individual-patch relief 19–290m) are pooled and forced
through a SINGLE linear scale+shift, the fit cannot reconcile patches whose
visual appearance is similar but whose true elevation range differs by an
order of magnitude — full-scene pooled correlation collapses to **0.05–0.24**
even under oracle (dense-truth) calibration, vs. 0.96–0.97 per-patch. This
is a genuine, structural, and highly relevant finding: it means the
"per-scene calibration" assumption embedded in the whole pipeline (Red
Flag #8) is far more fragile on heterogeneous hilly terrain than on
Vaihingen's comparatively uniform relief — a full real satellite scene over
mountainous terrain would need either much denser/spatially-distributed
calibration, per-region (not per-whole-scene) calibration, or an absolute-
scale-aware model, none of which the current pipeline has.

Vaihingen fine-tuning gives **no consistent benefit** over zero-shot on
this data (per-patch: fine-tuned slightly better, 11.08m vs 11.99m;
full-scene: fine-tuned slightly worse, 61.94m vs 60.20m) — both are
essentially out-of-domain here, and the small Vaihingen fine-tune did not
teach the model anything that transfers to mountain relief.

## 7. Slope stratification — does error rise with slope?

**Yes, clearly and monotonically, in every regime tested.** (See
`outputs/figures/E_slope_vs_rmse.png`.)

**Per-patch oracle calibration (best-case):**
| Slope bin | n px | zero-shot RMSE (m) | finetuned RMSE (m) |
|---|---|---|---|
| 0–5° | 1.75M | 8.82 | 7.81 |
| 5–15° | 3.13M | 11.92 | 10.60 |
| 15–30° | 2.00M | 18.55 | 20.27 |
| 30–90° | 0.45M | 34.98 | 45.57 |

**Full-scene oracle calibration (realistic):**
| Slope bin | n px | zero-shot RMSE (m) | finetuned RMSE (m) |
|---|---|---|---|
| 0–5° | 1.75M | 45.49 | 44.99 |
| 5–15° | 3.13M | 46.18 | 46.42 |
| 15–30° | 2.00M | 65.05 | 67.10 |
| 30–90° | 0.45M | 131.21 | 139.61 |

Under the best-case per-patch calibration, RMSE roughly **quadruples**
from the flattest bin (0–5°, ~8m) to the steepest (30–90°, ~35–46m). Under
realistic whole-scene calibration the same monotonic trend holds and is
even starker (~45m → ~130–140m). The steepest bin is also where the
fine-tuned model does noticeably worse than zero-shot (45.6m vs 35.0m
per-patch-oracle) — the small Vaihingen fine-tune, done entirely on
gently-rolling terrain, appears to very slightly *hurt* generalization to
the steepest slopes relative to the untouched pretrained backbone, though
sample sizes here (max slope bin = a handful of patches) are small enough
that this specific reversal should not be over-interpreted.

## 8. Adaptation attempt (1 of 2 allowed for this red-flag category)

One lightweight fine-tuning adaptation pass was run: starting from the
Phase-1 Vaihingen checkpoint, backbone frozen, neck+head only, SSI-L1 loss
(same recipe as `scripts/06_finetune.py`), 25 epochs, on a 17/6/5
train/val/test patch split of this hilly AOI (`E_04_adapt_finetune.py`,
`outputs/metrics/E_adaptation_attempt1.json`).

| | Before adaptation | After adaptation (best-val ckpt) |
|---|---|---|
| Full-scene oracle RMSE (test, n=5 patches) | 26.57m | 25.68m |
| Full-scene oracle corr | 0.299 | 0.387 |
| Per-patch oracle mean RMSE | 9.54m | 10.03m |
| Sparse-10pt RMSE (single draw) | 34.83m | 37.80m |

**Result: no material recovery.** Validation SSI-L1 loss plateaued
essentially immediately (best at epoch 9, oscillating ±0.05 around 4.8–5.3
for the remaining 16 epochs) — with only 17 training patches there simply
isn't enough signal for the neck/head to learn anything that generalizes to
the 5 held-out test patches. The full-scene-oracle number improved
marginally (26.6m→25.7m, corr 0.30→0.39) while the per-patch-oracle and
sparse-10 numbers got slightly worse — a mixed, not-clearly-positive
outcome. Given the master-brief's 2-attempt budget for this category and
that the failure mode here (17 patches is far too little data, not a fixable
hyperparameter issue) would not be addressed by a second attempt with the
same data, **a second adaptation attempt was not run**; more data (a
second, independent hilly AOI) would be needed to make a second attempt
meaningfully different rather than a re-roll of the same small-sample noise.

## 9. Honest verdict

- A genuine hilly-terrain RGB+elevation benchmark **was successfully
  obtained and verified** — real US government open data, no accounts
  created, correctly co-registered and visually validated (§4), real
  coordinates independently confirmed to land on a named real-world
  location.
- Under the pre-registered "reasonably strong" threshold (~8.6m, realistic
  calibration), the result is a **clear miss**: realistic per-scene
  calibration gives 60–70m RMSE, roughly 7–8× that threshold.
- Under the most generous possible (non-deployable) per-patch-oracle
  calibration, results are less dramatic (~11–12m, ~2.5× Vaihingen's
  ceiling) but still confirm real, substantial degradation on hilly terrain
  relative to Vaihingen's flat/urban town, and this gap **grows sharply and
  monotonically with slope** in every regime tested — not a borderline or
  ambiguous trend.
- The dominant mechanism identified is **not** simply "the model can't see
  shape on hills" (per-patch relative-depth correlation stays high, 0.96–0.97,
  even on steep patches) but that **a single scale+shift calibration per
  real scene cannot reconcile relative-depth predictions across patches
  with very different local relief** — full-scene pooled correlation
  collapses to 0.05–0.24 even under oracle calibration. This is a structural
  finding about the calibration assumption (Red Flag #8), sharpened
  specifically by hilly terrain (Red Flag #3), not a data-quality artifact.
- One lightweight adaptation attempt on this hilly data gave no material
  improvement, most plausibly because 17 training patches is too little
  data for the neck/head to learn anything transferable — reported
  honestly as a negative result, not spun as a partial success.
- Not marked UNRESOLVED: real data was obtained, a real evaluation was run,
  and a real (negative) result is reported. UNRESOLVED would misrepresent
  this as "could not get data" when the actual finding is "got the data,
  ran the test, the pipeline's per-scene calibration assumption does not
  hold up on genuinely heterogeneous hilly terrain at real GCP-count scale."
