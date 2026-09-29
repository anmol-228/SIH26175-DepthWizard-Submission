# 13_INDIA_DOMAIN_AND_UNCERTAINTY.md — Experiments G & U

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

## Experiment G — India Deployment Sanity Check (NOT full ground-truth validation)

Per user decision, no account-gated India sources (Bhoonidhi/CartoDEM, Copernicus Data Space Sentinel-2) were used. Instead: **ESRI World Imagery** (public no-auth XYZ tile service, exact sensor/acquisition date not disclosed by the service — a real, disclosed limitation) for optical RGB, and **Copernicus GLO-30** (public no-auth AWS bucket) for a coarse (30m/px, ~2m-accurate-in-easy-terrain per literature) elevation reference. Three real Indian regions, coordinates verified: Bengaluru urban (12.97°N 77.59°E), Coorg/Western Ghats hilly (12.42°N 75.74°E), Western Ghats forested (10.12°N 77.03°E). Code: `scripts/G_india_sanity_check.py`.

**Real bug found and fixed en route:** the Copernicus DEM bucket's own naming convention has resolution-in-arcseconds, not meters — GLO-30 (30m) tiles are named `COG_10` and GLO-90 (90m) tiles are named `COG_30` (confirmed directly against the bucket's `readme.html` after an initial 404). The earlier research pass's assumption was backwards on this specific point — corrected here after direct verification, not assumed away.

| Region | GLO-30 elev range (m) | Pred std (relative units) | Structural check | Corr vs GLO-30 | Rough RMSE vs GLO-30 (m) |
|---|---|---:|---|---:|---:|
| Urban (Bengaluru) | 898.8-930.3 | 0.54 | **NOT degenerate** — real spatial variance | 0.01 | 6.02 |
| Hilly (Coorg) | 1100.7-1173.1 | 0.41 | **WARNING: nearly flat/degenerate** | 0.25 | 13.04 |
| Forested (W. Ghats) | 1446.2-1817.9 | 0.47 | **WARNING: nearly flat/degenerate** | 0.84 | 31.74 |

Raw data: `outputs/metrics/G_india_sanity_check.json`, figures: `outputs/figures/G_india_*.png`.

### Reading the result — honestly, with every confound stated
**This is explicitly NOT a rigorous benchmark** (Section 28's own framing) — several real confounds stack up: (1) ESRI basemap tiles are of unknown/mixed acquisition date and sensor, especially over rural Coorg/Western Ghats where basemap quality is often lower than urban areas; (2) GLO-30 is 30m/px and only ~2m-accurate in easy terrain (worse in complex terrain per `02_DATASET_RESEARCH.md`'s cited literature) — comparing it pixel-for-pixel against a model operating at ~1.2m/px effective resolution mixes two very different spatial scales, so the "rough RMSE" numbers above conflate real model error with genuine scale-mismatch noise; (3) the Western Ghats tile's huge correlation (0.84) is very likely explained by that region's 372m of real elevation relief within a ~1.2km crop — even a nearly-flat prediction can show high correlation against a single dominant large-scale trend, which is a different, weaker claim than "the model recovered real local relief."

**What IS a genuine, disclosed finding despite those confounds:** the model's prediction was **structurally flat/degenerate on BOTH the hilly and forested Indian scenes** (explicit automated flag, not just visual impression) while remaining non-degenerate on the urban scene. This is directionally consistent with, and adds independent (if imperfect) evidence for, Red Flag #6 (Germany->India domain shift) and the general concern that fine-tuning on flat-to-gently-rolling German urban/suburban terrain does not obviously transfer local-structure recovery to genuinely different terrain types, even before accounting for GSD or sensor differences. **This finding should NOT be read as "the model fails in India" (the confounds above are too large for that claim) — it should be read as "nothing in this rough check contradicts the standing concern, and the flat-prediction signal on 2 of 3 real Indian scenes is a mild independent warning sign worth taking seriously."**

**Status per Section 28's required framing: India Deployment Sanity Check — NOT full ground-truth validation.** Red Flag #6 remains UNRESOLVED at the rigor the master brief originally asked for (that required Bhoonidhi/Sentinel-2, declined per user decision) but is now informed by real, if imperfect, qualitative+rough-quantitative evidence rather than being a total blank.

## Experiment U — Uncertainty/Confidence Validation (Section 39)

Phase 1 never built an uncertainty output at all — there was nothing to validate directly. Built the cheapest defensible proxy: **test-time-augmentation (TTA) prediction variance** across 4 views (identity, horizontal flip, +15% brightness, -15% brightness) on the real 9cm Vaihingen test set, and checked whether that variance actually correlates with real error. Code: `scripts/U_uncertainty_check.py`.

| Metric | Value |
|---|---:|
| Global pixel-wise correlation (TTA std vs. abs error) | 0.190 |
| Mean per-patch correlation | 0.163 |
| Median per-patch correlation | 0.171 |
| Mean abs error, top-10%-uncertainty pixels | 1.84m |
| Mean abs error, bottom-10%-uncertainty pixels | 0.97m |
| **Ratio (top / bottom)** | **1.90x** |

Raw data: `outputs/metrics/U_uncertainty_check.json`.

### Reading the result
**Weak but real, positive signal — earns a place, with an honest label.** TTA variance is NOT random noise (corr ~0.17-0.19, and the top-uncertainty decile has genuinely, consistently ~1.9x higher error than the bottom decile) — but it is also nowhere near strong enough to support a precise per-pixel confidence interval or a claim like "trust predictions with confidence > X to within Y meters." Per the master brief's explicit rule ("if confidence does NOT correlate with error, do not show a fake confidence heatmap; if it DOES, calculate a simple calibration/error relationship"): **it does correlate, so a confidence layer is defensible — but it should be presented as a coarse triage signal ("this region is somewhat less reliable") rather than a calibrated uncertainty estimate.** This is a legitimate, if modest, differentiator, KEEP per Section 44's cost/benefit framework (TTA with 4 views costs ~4x inference time, cheap, and the signal — while weak — is real and free to compute).
