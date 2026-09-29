# 09_RGB_SENSOR_ROBUSTNESS.md — Experiment C: IRRG vs True RGB

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

## Why a direct IRRG-vs-RGB dataset comparison wasn't possible
Vaihingen imagery has NO blue channel at all — it is Infrared-Red-Green (IRRG), not RGB with an extra NIR band. There is no way to "convert it to true RGB" because the blue wavelength was simply never captured by the original camera. A genuine RGB comparison needs a different dataset (attempted separately via Experiment D's NEON acquisition, which is true RGB camera imagery — see `06_VEGETATION_FOREST_TEST.md` once that lands, though that comparison will conflate spectral AND domain shift together, not isolate spectral shift alone).

## What WAS tested: NIR-channel reliance ablation (zero-shot, existing checkpoint)
Directly answers the master brief's real underlying question — "was Phase-1 performance partly benefiting from near-infrared information?" — by ablating the IR channel on the SAME Vaihingen test data, zero-shot (no retraining), full-tile oracle calibration. Code: `scripts/C_irrg_vs_rgb_ablation.py`.

| Variant | RMSE (m) | Δ vs IRRG control | Corr |
|---|---:|---:|---:|
| **irrg (control, = Phase-1)** | 4.403 | — | 0.913 |
| rg_duplicate (IR channel replaced with a copy of G — closest proxy for "a true-RGB-only sensor, no NIR band") | 4.478 | **+1.7%** | 0.910 |
| r_only (all 3 channels = R, grayscale-from-red sanity floor) | 4.542 | +3.2% | 0.908 |
| zero_ir (IR channel zeroed) | 4.654 | +5.7% | 0.903 |

Raw data: `outputs/metrics/C_irrg_vs_rgb_ablation.json`.

## Reading the result
**The effect is real but small.** Even fully removing the near-infrared channel only degrades RMSE by 5.7%, and the most realistic "swap NIR for nothing, keep R+G" proxy (`rg_duplicate`) costs just 1.7%. This is reassuring for Red Flag #7: Phase-1's headline numbers were NOT heavily dependent on having a near-infrared band, so a true-RGB satellite sensor (no NIR) should not, on this evidence, cause a large accuracy collapse from spectral shift alone.

**Caveat, stated plainly:** this ablation is zero-shot on a model trained WITH the IR channel present — it measures "how much does removing IR hurt a model that learned to (possibly) lean on it a little," not "how well would a model trained from scratch without IR perform." Those could differ in either direction (a from-scratch RGB-only model might learn to compensate better, or might lose access to genuinely useful shadow/vegetation-discriminating NIR information that RG alone can't recover). This is a real, disclosed limitation of the zero-shot-ablation method — retraining on each channel variant was not attempted, per the master brief's per-experiment iteration budget (this experiment has no explicit iteration allowance in Section 45's list, so it defaulted to the lightest defensible test).

## Net answer
Was Phase-1 performance partly benefiting from NIR? **YES, modestly (order 2-6% of RMSE), not dominantly.** True-RGB satellite imagery (no NIR band) is not, on this evidence, expected to cause severe degradation from spectral shift alone — real satellite-domain risk (sensor, resolution, atmosphere — Experiment B, UNRESOLVED per user decision) remains a larger and separately-untested concern than the spectral-channel question alone.
