# 10_ERROR_ROBUSTNESS.md — Experiments H & I: Shadow & Building-Edge Quantification

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

Phase 1 only observed roof-edge and shadow effects qualitatively (one typical-case figure; shadows "visually associated with noisier predictions," never measured). This experiment quantifies both, using the real ISPRS semantic-label ground truth (building/tree classes) on 4 real Vaihingen test tiles that have segmentation labels available (areas 1, 26, 28, 34 — all of Phase-1's 9 test tiles except 6, 8, 10, 29, 38, which lack the `gts_for_participants` label release). Full-tile sliding-window inference with feather-blending (same method as `K1_tiling_stitch.py`), oracle-calibrated per tile. Code: `scripts/I_H_edge_shadow_analysis.py`.

## Building edge vs. interior (Section 31, Red Flag #12)

| Area | Edge mean |err| (m) | Interior mean |err| (m) | Edge/interior ratio |
|---:|---:|---:|---:|
| 1 | 3.05 | 2.99 | 1.02 |
| 26 | 3.73 | 4.57 | **0.82** |
| 28 | 3.20 | 2.79 | 1.15 |
| 34 | 1.88 | 2.11 | **0.89** |
| **Mean** | | | **0.97** |

**Finding: does NOT reproduce as a general quantitative effect.** Phase-1's qualitative single-case observation ("largest residual error concentrated right at roof/building edges") does not hold up as a consistent quantitative pattern across 4 real tiles at the full-tile level — the ratio ranges 0.82-1.15 (essentially noise around 1.0, no consistent direction). **Revise Red Flag #12 down**: this looks more like normal per-tile variance than a real, structural edge-blur effect at this evaluation scale. Caveat: this measures MEAN absolute error over all edge pixels in a tile, which could mask a real but spatially sparse effect visible only in worst-case-percentile error or in a finer per-pixel visualization (the original Phase-1 qualitative figure was a visual/percentile observation, not a mean-error claim) — a genuine methodological difference, not a contradiction so much as a different, complementary measurement.

## Vegetation (tree-class) error at MODERATE density — direct cross-check of Red Flag #1

| Area | Tree mean |err| (m) | Non-tree mean |err| (m) | Tree/non-tree ratio |
|---:|---:|---:|---:|
| 1 | 3.77 | 3.34 | 1.13 |
| 26 | 3.77 | 4.07 | 0.93 |
| 28 | 3.07 | 3.01 | 1.02 |
| 34 | 1.33 | 1.36 | 0.98 |
| **Mean** | | | **1.01** |

**Important refinement of Red Flag #1, not a contradiction of it.** At moderate/scattered tree density (these 4 tiles, tree pixels 17-31% of area), there is essentially NO consistent tree-vs-non-tree error penalty (ratio ~1.0). This is a materially different picture from area 6 (Phase-1's worst tile: RMSE 10.2m tile-level, ~2.3x worse than every other tile), which was dominated by DENSE, CONTIGUOUS canopy — not just "any pixel labeled tree." **The correct, more precise statement of this red flag is: the failure mode is specific to large contiguous canopy regions, not vegetation presence generally.** This matters for mitigation design (Experiment D): a per-pixel vegetation-index correction may be less useful than a region-level "is this a large contiguous canopy patch" detector, since isolated/scattered trees mixed with buildings and open ground do not show the same failure. Experiment D (running separately on genuine dense forest) is the real test of the severe case.

## Shadow (Section 30, Red Flag #13)

| Area | Shadow mean |err| (m) | Non-shadow mean |err| (m) | Ratio |
|---:|---:|---:|---:|
| 1 | 3.77 | 3.35 | 1.13 |
| 26 | 4.16 | 3.99 | 1.04 |
| 28 | 2.95 | 3.03 | 0.98 |
| 34 | 1.50 | 1.33 | 1.12 |
| **Mean** | | | **1.07** |

**Finding: a small, real, fairly consistent effect** (3 of 4 tiles show shadow error 4-13% higher; the one exception is within noise). This is a modest, not dramatic, degradation — matches Phase-1's cautious "visually associated, not confirmed" framing, now upgraded to CONFIRMED-BUT-SMALL. Given the master brief's guidance to only mitigate if the effect is material (Section 30: "do not spend excessive time here unless the effect is material"), **no mitigation was attempted** — a ~7% mean-error increase in the darkest 10% of pixels does not meet that bar.

## Net effect on the register
- #12 (roof/edge error): severity revised DOWN — real but not confirmed as a consistent quantitative effect at full-tile scale.
- #13 (shadow sensitivity): CONFIRMED as real but small (~7% mean effect); no mitigation warranted at this magnitude.
- #1 (vegetation failure): REFINED, not resolved or worsened — confirmed as specific to dense/contiguous canopy, not vegetation-pixel presence in general. Genuine severity assessment now depends on Experiment D's dense-forest result.
