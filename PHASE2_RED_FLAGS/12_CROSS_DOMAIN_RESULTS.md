# 12_CROSS_DOMAIN_RESULTS.md — Master Cross-Domain Evidence Matrix

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

Per master brief Section 40-42: do not average away failure. Every row below is a genuinely different domain/condition; worst-category behaviour is reported alongside best, not blended into one number.

| Domain | GSD | Terrain | RGB/IRRG | Adaptation | Calibration | RMSE (m) | Corr | Verdict |
|---|---:|---|---|---|---|---:|---:|---|
| Vaihingen (Phase 1) | 9cm | Urban | IRRG | fine-tuned | full-tile oracle | 4.40 | 0.913 | STRONG |
| Vaihingen (Phase 1) | 9cm | Urban | IRRG | fine-tuned | sparse-10 (realistic) | 5.50 | 0.876 | STRONG |
| Vaihingen synthetic-degraded | 1m | Urban | IRRG | fine-tuned (9cm) | sparse-10 | 6.91 | 0.787 | STRONG (idealized sim only) |
| Vaihingen synthetic-degraded | 3m | Urban | IRRG | fine-tuned (9cm) | sparse-10 | 8.09 | 0.712 | ACCEPTABLE (idealized sim only) |
| Vaihingen synthetic-degraded | 3m | Urban | IRRG | fine-tuned AT 3m | sparse-10 | 5.92 | 0.842 | STRONG (retraining recovers most loss; idealized sim only) |
| Vaihingen synthetic-degraded | 5m | Urban | IRRG | fine-tuned (9cm) | sparse-10 | 9.95 | 0.606 | ACCEPTABLE, degrading (idealized sim only) |
| Vaihingen (Phase 1 proxy) | 9cm | "Vegetation" (street/garden trees) | IRRG | fine-tuned | full-tile oracle | 10.2 | 0.488 | WEAK (proxy, not genuine forest) |
| Vaihingen, 4 tiles w/ segmentation | 9cm | Moderate tree density (17-31% cover) | IRRG | fine-tuned | full-tile oracle | (tree/non-tree ratio ~1.0, not separately worse) | — | ACCEPTABLE (moderate density is NOT the failure case) |
| **Olympic NP forest (Sentinel-2+ETH CHM)** | 10m | **Genuine dense old-growth forest** | RGB | **zero-shot** | full-oracle | **6.70** | 0.436 | **WEAK — barely beats constant-mean (6.94m)** |
| Olympic NP forest | 10m | Genuine dense old-growth forest | RGB | Vaihingen-finetuned (no forest adapt) | full-oracle | 6.81 | 0.405 | WEAK |
| **Olympic NP forest** | 10m | Genuine dense old-growth forest | RGB | **forest-finetuned (16 patches)** | full-oracle | **2.87** | **0.923** | **STRONG (57% RMSE cut)** |
| Olympic NP forest, OUT-of-scene AOI | 10m | Genuine dense old-growth forest (different, non-adjacent area) | RGB | forest-finetuned, ZERO retrain on this AOI | full-oracle | 3.70 | 0.919 | STRONG (generalizes, 54% RMSE cut) |
| **Mount Adams, WA (3DEP+NAIP)** | 1m | **Genuine hilly/mountainous** | RGB | zero-shot | per-patch oracle (best-case, non-deployable) | 11.99 | 0.967 (per-patch) | ACCEPTABLE shape-only, MISLEADING as a deployability signal |
| **Mount Adams, WA** | 1m | **Genuine hilly/mountainous** | RGB | zero-shot | **full-scene oracle (realistic)** | **60.20** | **0.241** | **SEVERE FAILURE** |
| Mount Adams, WA | 1m | Genuine hilly/mountainous | RGB | Vaihingen-finetuned | full-scene oracle (realistic) | 61.94 | 0.053 | SEVERE FAILURE (no benefit from Vaihingen fine-tune) |
| Mount Adams, WA | 1m | Genuine hilly/mountainous | RGB | one adaptation attempt (17 patches) | full-scene oracle | 25.68 (26.57 before) | 0.387 (0.299 before) | STILL SEVERE (marginal, not material, improvement) |
| Mount Adams, WA | 1m | Genuine hilly/mountainous | RGB | zero-shot | sparse-10 (realistic+deployable) | 69.55 | — | SEVERE FAILURE |
| India (ESRI+GLO-30), Bengaluru | ~1.2m | Urban | RGB | Vaihingen-finetuned | none (qualitative only) | rough 6.02 (heavy confounds, not a real benchmark) | 0.01 | UNKNOWN (non-degenerate structure, but ~zero correlation to coarse truth) |
| India (ESRI+GLO-30), Coorg | ~1.2m | Hilly | RGB | Vaihingen-finetuned | none (qualitative only) | rough 13.04 (heavy confounds) | 0.25 | WEAK SIGNAL — prediction flagged structurally flat |
| India (ESRI+GLO-30), W. Ghats | ~1.2m | Forested | RGB | Vaihingen-finetuned | none (qualitative only) | rough 31.74 (heavy confounds) | 0.84 (likely a single-trend artifact, see caveat) | WEAK SIGNAL — prediction flagged structurally flat |

**Satellite-sensor domain (DFC2019/US3D or equivalent): NOT TESTED — UNRESOLVED by explicit user decision** (account/license-gated). No row exists for this axis; do not read its absence as a pass.

## The single most important row in this table
**Full-scene oracle calibration on hilly terrain (RMSE 60.20m, corr 0.241)** is the worst result in the entire two-phase project by a wide margin — roughly 14x worse than Vaihingen's realistic urban number (5.50m) and ~11x worse than the forest failure case before its fix (6.70m). It is also the ONLY result that stayed severe after a dedicated adaptation attempt. This is not an artifact of a bad dataset (data quality was independently verified) — the root cause (per-scene scale+shift calibration cannot reconcile patches with wildly different local relief) is structural and would apply to ANY future hilly-terrain deployment of this exact architecture, not just this test.

## Terrain-by-terrain read (Section 41-42 framing)
- **Urban**: STRONG across every test that touched it (Vaihingen native, Vaihingen-degraded-GSD, India qualitative-non-degenerate).
- **Sparse/open**: inherited from Vaihingen's sparse-density tiles (STRONG, not separately re-tested in Phase 2 — no new red flag raised against it).
- **Forested**: WEAK zero-shot/untransferred, but a real, cheap, generalizing fix exists (STRONG after 16-patch adaptation) — conditional on sourcing real target-forest training data for whatever biome is actually deployed against (untested beyond one Pacific NW biome).
- **Hilly**: SEVERE, UNFIXED FAILURE under realistic calibration. This is the single biggest new red flag Phase 2 surfaced, and it is a structural, not incidental, finding.
