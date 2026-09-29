# 06_VEGETATION_FOREST_TEST.md — Experiment D: Genuine Dense Forest / Vegetation Failure

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

Priority 4 (Red flags #1 and #2). Tests whether DA2-Small's Vaihingen "vegetation failure"
(tile RMSE 10.2m, ~2.3x worse than every other terrain type, patch-level up to 21m,
flagged in Phase 1 as a possible REPRESENTATIONAL limit) reproduces on GENUINE dense
canopy forest — not the street/garden trees that made up Vaihingen's "vegetation" proxy
— and, if it does, whether it is fixable.

**Bottom line up front:** the failure reproduces convincingly on genuine forest
(baseline models are barely better than predicting the per-patch mean height). But
it is **not a fixed representational limit** — a single lightweight fine-tuning
attempt on only 16 real forest training patches (frozen backbone, head+neck only)
cuts RMSE by 57% in-scene and 54% on a separate, non-adjacent held-out forest area
with no retraining. Both clear the pre-registered 25% bar by a wide margin. Attempts
2 and 3 were not needed and were not run.

---

## 1. NEON access blocker (honest account of what happened)

The master brief's plan was to use NEON (National Ecological Observatory Network)
RGB camera orthomosaics + LiDAR Canopy Height Model (DP3.30015.001), reported as
requiring no account/registration. That information is now **out of date**.

What was verified live, this session:
- `GET https://data.neonscience.org/api/v0/products/DP3.30015.001` → **200 OK**
  (product metadata, no auth needed).
- `GET https://data.neonscience.org/api/v0/data/DP3.30015.001/HARV/2018-09` (and
  every other site/month tried) → **403 `{"error":{"status":403,"detail":"Access
  Denied"}}`**, reproduced identically via `curl`, Python `urllib`, and a real
  Chrome browser pane (ruling out a bot-fingerprint block) — the file-listing/
  download endpoint is blocked, only the product-metadata endpoint is open.
- `WebFetch` of NEON's own tutorial page (`neonscience.org/.../create-chm-rasters-r`)
  states directly: **"As of June 2026, NEON requires an API token for data
  downloads."** Tokens are issued through account registration on the NEON data
  portal.

Creating an account/registering for a token is a **prohibited action** under this
agent's operating rules (account creation requires the user to do it themselves),
so NEON was dropped as a data source rather than worked around. This is a genuine,
dated policy change on NEON's side (discovered mid-session, not something the
master brief could have known when it was written) — documented here rather than
silently substituted.

**Substitute real-data pipeline used instead** (below): Sentinel-2 (RGB) + ETH
Global Canopy Height 2020 (elevation truth), both genuinely open, no account, no
token, verified working end-to-end this session.

---

## 2. Dataset actually used

### RGB: Sentinel-2 L2A True Colour (TCI)
- Scene `S2A_10TDT_20210902_0_L2A`, acquired **2021-09-02**, cloud cover **0.169%**
- 10m GSD, CRS **EPSG:32610** (UTM zone 10N), tile MGRS 10TDT
- Source: AWS Open Data `sentinel-cogs` public S3 bucket, queried via the
  Earth Search STAC API (`https://earth-search.aws.element84.com/v1/search`,
  no auth) — asset URL:
  `https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/10/T/DT/2021/9/S2A_10TDT_20210902_0_L2A/TCI.tif`
  (293MB, downloaded whole; also pulled `SCL.tif`, 1.8MB, for cloud/shadow/water
  masking)
- License: Copernicus Sentinel data, free/full/open (ESA/EU Copernicus program)

### Elevation truth: ETH Global Canopy Height 2020
- Lang, Jetz, Schindler, Wegner, *"A high-resolution canopy height model of the
  Earth"*, **Nature Ecology & Evolution, 2023** (DOI 10.3929/ethz-b-000609802)
- 10m nominal GSD (pixel scale 8.3333e-5°, ≈9.26m N-S at this latitude), CRS
  **EPSG:4326**, uint8 (metres, `255` = nodata/masked — confirmed empirically,
  see §3), **CC-BY-4.0**
- Source: ETH Research Collection public mirror (`libdrive.ethz.ch`, no auth),
  tile `N45W126`:
  `https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=%2F3deg_cogs&files=ETH_GlobalCanopyHeight_10m_2020_N45W126_Map.tif`
  (156MB)
- Trained against sparse real spaceborne LiDAR (NASA GEDI) footprints, published
  and peer-reviewed. Independent of Depth Anything V2 (never in DA2's training
  data). **Caveat, disclosed and not hidden:** the ETH map's own training used
  Sentinel-2 spectral features as a predictor, so this is not a fully
  independent-of-imagery ground truth. Vaihingen's own DSM is photogrammetric
  (image-matching, not airborne LiDAR), so it isn't a clean independent-instrument
  baseline either -- both datasets carry some imagery-derived dependency, ETH's
  more explicit and disclosed. It is still real, externally-produced,
  peer-reviewed elevation data — just a weaker independence guarantee than
  Phase-1's setup. Flagged here explicitly (see §7 Limitations).

### Area of interest
Western Olympic National Park lowland river valleys (Hoh / Bogachiel / Queets
drainages), Washington, USA — **genuine dense old-growth temperate rainforest**
(Sitka spruce / western hemlock / Douglas fir, canopy heights commonly 40-60m),
not urban/garden trees.
- Main AOI: lat **47.55–47.83N**, lon **124.05–123.65W**, ≈30.4 × 31.4 km
- Out-of-scene generalization AOI: lat **47.30–47.50N**, lon **123.90–123.55W**
  (≈5.6km buffer south of the main AOI, non-overlapping)
- Working CRS: **EPSG:32610 (UTM 10N), 10m/px** — RGB kept at native Sentinel-2
  grid; CHM nearest-resampled onto this grid via `pyproj` coordinate transforms
  (WGS84 → UTM10N pixel centres → nearest CHM sample)

**Total download: ≈445MB** (well under the 2-3GB budget).

---

## 3. Data validation (same rigor as Phase-1's `02_DATA_AUDIT.md`)

- **CRS/GSD confirmed from embedded GeoTIFF tags**, not assumed: TCI tags read
  `ModelPixelScaleTag=(10,10,0)`, `EPSG:32610`; CHM tags read
  `ModelPixelScaleTag=(8.3333e-5, 8.3333e-5, 0)`, `EPSG:4326` (WGS84 geographic).
  Confirmed genuinely well-georeferenced UTM/WGS84 GeoTIFFs, as expected.
- **Nodata discovery**: CHM value `255` occurs in 3.2% of a preview crop with a
  sharp discontinuity from the 95th percentile (54m) straight to 255 — confirmed
  as the product's nodata/masked-quality code (not a real 255m tree) and excluded
  via a valid-pixel mask.
- **Cloud/shadow/water masking**: Sentinel-2 SCL band used to exclude nodata,
  saturated/defective, cloud shadow, unclassified, and cloud (med/high prob +
  cirrus) pixels. Within the final AOI, cloud contamination is negligible (<0.1%
  combined); 92.2% of the main AOI is SCL-classified vegetation.
- **Height sanity**: valid-pixel CHM stats over the main AOI — mean **40.9m**,
  median **44.0m**, p95 **53.0m**, max **60.0m**. Consistent with published
  descriptions of Olympic old-growth temperate rainforest canopy heights. This is
  taller and denser than anything in Vaihingen's "vegetation" proxy (street trees).
- **Visual RGB/CHM overlay** (`outputs/figures/D0_aoi_rgb_chm_overlay.png`):
  the river channel visible in RGB lines up pixel-for-pixel with the low-height
  (dark) river corridor in the CHM, and the snow-capped peak in the NE corner of
  the RGB lines up with the near-zero-height area in the CHM. **Independently
  confirms correct pixel alignment** between the two separately-sourced,
  separately-reprojected rasters — the same kind of visual check Phase-1 used for
  Vaihingen before trusting the data.
- Patch extraction: AOI sliced into non-overlapping 512×512 px patches (5.12km
  per side at 10m GSD), kept only if ≥98% valid pixels. Main AOI: **25 kept / 30
  candidate** patches → split by spatial row-band (train=16 rows 0-3, val=4 row 4,
  test=5 row 5) so **train/val/test are spatially disjoint, no patch overlap**,
  matching Phase-1's per-area split spirit. OOS AOI: 14 kept patches, held out
  entirely (never trained on).

---

## 4. Baseline results (zero-shot + Vaihingen fine-tuned), full-oracle calibration

Methodology matches Phase-1's `07_calibrate_and_eval.py` exactly: per-unit
`fit_scale_shift` (closed-form OLS) then `all_metrics`, oracle-calibrated on the
SAME pixels being scored (upper-bound / diagnostic only, exactly as Phase-1
disclosed for its "full oracle" number). The "unit" here is each 512×512 patch
(≈5.12km, closer in scale to a Phase-1 *area* than a Phase-1 *patch* — disclosed
mapping), evaluated on the **5 held-out test patches** in the main AOI.

| Model | RMSE (m) | MAE (m) | corr |
|---|---|---|---|
| Zero-shot pretrained DA2-Small (never seen Vaihingen or forest) | **6.70** | 5.33 | 0.436 |
| Vaihingen-fine-tuned checkpoint (cross-domain + vegetation combined) | **6.81** | 5.42 | 0.405 |
| *Context: constant per-patch-mean baseline (trivial, no RGB used)* | *6.94* | *5.52* | *0.363* |

**The failure reproduces, and is worse than Phase-1's proxy suggested in relative
terms.** Both models beat the trivial constant-mean baseline by only **1.8–3.4%**
RMSE — i.e., after full-oracle scale/shift calibration (which can rescue a
correctly-*shaped*-but-mis-scaled prediction), the models are adding almost no
real spatial information on top of "just predict the average height for this
patch." Per-patch correlation is highly unstable: one zero-shot patch has
corr=0.001 (a≈0.04, essentially flat), and the Vaihingen-finetuned checkpoint
produces **negative** oracle-fit slopes (`a=-0.20`, `a=-2.51`) on two patches —
the raw prediction is anti-correlated with real height there, i.e. actively
wrong-direction, not just weak. This is the same "goes nearly flat" symptom
Phase-1 characterized on its vegetation proxy, now confirmed on genuine,
much-taller, much-denser canopy.

---

## 5. Mitigation Attempt 1: fine-tune on real forest data — SUCCEEDS

Recipe: same as Phase-1's `06_finetune.py` (SSI-L1 loss, closed-form per-sample
scale+shift alignment, AdamW, lr=5e-5 head / 5e-6 backbone), starting from the
**original pretrained** DA2-Small weights (not the Vaihingen checkpoint, to avoid
confounding forest-adaptation with a second domain shift). Deviations, disclosed:
backbone **fully frozen** (only 16 train patches — unfreezing risks catastrophic
overfitting), light rot90/flip augmentation (label-preserving for nadir imagery,
needed for epoch diversity with only 4 steps/epoch), 60 epochs (compensating for
the much smaller dataset). **2.7M trainable params** (head+neck) vs **22M frozen**
(backbone). Trained in under a minute on the RTX 5060 Ti.

**Held-out test (same AOI, spatially disjoint row-band, never seen in training):**

| Model | RMSE (m) | MAE (m) | corr |
|---|---|---|---|
| Zero-shot baseline (from §4) | 6.70 | 5.33 | 0.436 |
| **Forest-fine-tuned (Attempt 1)** | **2.87** | **2.24** | **0.923** |
| **RMSE reduction vs. zero-shot baseline** | **57.2%** | | |
| RMSE reduction vs. Vaihingen-finetuned baseline | 57.9% | | |

**Out-of-scene generalization check** (separate, non-adjacent AOI ~5.6km south,
same imagery source, **zero retraining** — a true held-out generalization test,
run specifically to rule out "it only works because train/test share one
contiguous scene"):

| Model | RMSE (m) | MAE (m) | corr |
|---|---|---|---|
| Zero-shot baseline, evaluated on OOS AOI | 8.06 | 6.44 | 0.511 |
| **Forest-fine-tuned (Attempt 1), evaluated on OOS AOI, no retrain** | **3.70** | **2.70** | **0.919** |
| **RMSE reduction vs. OOS zero-shot baseline** | **54.0%** | | |

Both the in-scene and the genuinely-separate out-of-scene test **clear the
pre-registered 25% success bar by more than 2×**, and correlation goes from
"partially/actively flat" (0.36–0.51) to strongly structured (0.92) in both
cases, with no patch showing the near-zero-correlation "flat" signature anymore.

**This is a materially different conclusion than Phase-1's hypothesis.** Phase-1
flagged the vegetation failure as a possible *representational limit* (something
DINOv2's frozen visual features fundamentally cannot resolve). What was actually
tested here — freezing the backbone and only adapting the head/neck — fixed it
convincingly. The likely real mechanism: DINOv2's self-supervised features
already encode plenty of canopy texture/shading information (it was trained on
huge natural-image corpora including forest photos); what was missing was the
**decoder's mapping from those features to *relative height* structure for
this specific overhead-canopy domain**, which a small amount of real paired
forest training data supplies. This reframes the failure as an **adaptation
gap**, not a hard representational ceiling.

### Attempts 2 and 3: not run

Per the pre-registered "up to 3 mitigations, stop early on diminishing returns"
plan: Attempt 1 already exceeds the 25% success bar by more than 2× on both an
in-scene held-out test and a genuinely separate out-of-scene test. There is no
remaining problem for a vegetation-aware calibration split (Attempt 2) or a
canopy-specific loss reweighting (Attempt 3) to solve on this dataset. Running
them would not change the headline finding and would cost iteration budget this
brief explicitly asks to conserve. **Stopped here, by design, not by running out
of time.**

---

## 6. Answer to "can genuine forest/vegetation be handled?"

**Yes, conditionally.** Zero-shot and Vaihingen-only weights fail badly and
predictably on genuine dense forest (near-flat predictions, ≤3.4% improvement
over a trivial constant-height guess). But the failure is **not a fixed
representational limit** — it is fixable with a small amount of real,
target-domain (forest) paired RGB+height training data and a cheap,
low-overfitting-risk fine-tune (frozen backbone, head/neck only, 16 patches,
<1 minute of GPU time), and the fix **generalizes to a genuinely separate,
non-adjacent forest area with zero retraining** (54% RMSE reduction there too).

**What is NOT yet shown**, stated plainly:
- Only **one** forest biome was tested (Pacific Northwest temperate old-growth
  conifer). Generalization to other forest types relevant to an India deployment
  (tropical monsoon forest, mixed deciduous, mangrove, etc.) is **untested** and
  should not be assumed from this result alone.
- The two test AOIs are geographically close (~5.6km apart, same watershed
  complex, same Sentinel-2 scene/epoch) — a stronger test would use a forest on a
  different continent/sensor/epoch. Time/resource budget for this experiment did
  not extend to acquiring a second fully independent forest dataset.
- The GSD here (10m, Sentinel-2) is ~100x coarser than Vaihingen's 9cm — this
  experiment cannot cleanly separate "genuine forest" from "coarser GSD" as the
  cause of the baseline failure or the fix's effectiveness; that confound is
  explicitly flagged (see red flag #4, Experiment A, for the GSD axis tested in
  isolation).
- The elevation ground truth (ETH GCH) has an indirect Sentinel-2 dependency
  (§2), a weaker independence guarantee than Vaihingen's photogrammetric DSM -- though Vaihingen's own DSM is image-matching, not airborne LiDAR, so it isn't the clean separate-sensor case this comparison implies either.

---

## 7. KEEP / MAYBE / REJECT call (master brief Section 44 framework)

**Attempt 1 (forest-domain head/neck fine-tune, frozen backbone): KEEP.**
- Benefit: 54-57% RMSE reduction, corr 0.36→0.92, holds up on a genuinely
  separate held-out area with no retraining.
- Complexity/cost: very low. 2.7M trainable params, <1 minute of training,
  needs only ~15-20 real paired RGB+height forest patches (not thousands),
  no architecture change, no backbone unfreezing (so no added overfitting risk
  or catastrophic-forgetting risk to the rest of the model's behaviour).
- Residual cost this does NOT eliminate: **it requires real target-domain
  training data for whatever forest type will actually be deployed against.**
  For an India deployment this means a genuine India-forest RGB+height dataset
  needs to be sourced/acquired before this fix can be trusted operationally —
  that acquisition is out of scope for this experiment and is a real, concrete
  follow-up item, not a formality.

**Attempts 2 (vegetation-aware calibration split) and 3 (canopy-specific loss
weighting): N/A — not run**, because Attempt 1 already cleared the bar with
large margin on both tests available. No KEEP/MAYBE/REJECT call is being made on
mitigations that were never executed; if a future forest type resists Attempt
1's fix, these remain reasonable next things to try, in that order.

---

## 8. Red-flag register updates

See `01_RED_FLAG_REGISTER.md` rows #1 and #2, updated with these results.
Headline changes:
- Row #1 (dense vegetation failure): severity **CRITICAL → HIGH**. The failure
  is real and reproduces on genuine forest, but Phase-1's "representational
  limit, unfixable by more training" framing is **contradicted** by this
  experiment — a small amount of correctly-targeted training data fixes it and
  the fix generalizes locally. Still HIGH, not LOW/MEDIUM, because the fix
  requires acquiring real target-forest training data that does not yet exist
  for an India deployment, and only one forest biome was tested.
- Row #2 (no true forest validation): **RESOLVED**. A genuine dense-canopy
  RGB+elevation dataset was found, independently verified (§3), and evaluated
  (§4-§5).

---

## 9. Files

- Data prep: `scripts/D0_prepare_forest_data.py` → `data/forest_eth_s2/patches/`
  (25 `.npz` patches + `manifest.json` + `dataset_meta.json`),
  raw source rasters in `data/forest_eth_s2/raw/` (~445MB)
- Baseline eval: `scripts/D1_baseline_eval.py` →
  `outputs/metrics/D1_baseline_forest_eval.json`
- Constant-baseline context: `scripts/D1b_constant_baseline.py` →
  `outputs/metrics/D1b_constant_baseline_forest.json`
- Attempt 1 fine-tune + eval: `scripts/D2_finetune_forest.py` →
  `outputs/checkpoints/D2_da2_small_forest_finetuned.pt`,
  `outputs/metrics/D2_forest_finetuned_eval.json`,
  `logs/D2_finetune_history_D2_da2_small_forest_finetuned.pt.json`
- Out-of-scene generalization check: `scripts/D2b_generalization_check.py` →
  `outputs/metrics/D2b_generalization_check.json`
- Visual validation figure: `outputs/figures/D0_aoi_rgb_chm_overlay.png`

Reproduction: first download the 3 raw files into `data/forest_eth_s2/raw/`
using the exact URLs given in §2 above (`ETH_GlobalCanopyHeight_10m_2020_N45W126_Map.tif`,
`S2A_10TDT_20210902_TCI.tif`, `S2A_10TDT_20210902_SCL.tif` — plain `curl`, no auth,
~445MB total), then run `D0_prepare_forest_data.py` to build the patch set, then
`D1_baseline_eval.py`, `D1b_constant_baseline.py`, `D2_finetune_forest.py`,
`D2b_generalization_check.py` in that order. All scripts use fixed seeds (42).
