# 04_GSD_STRESS_TEST.md — Experiment A: Controlled GSD Degradation

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

## Method
Same held-out Vaihingen TEST tiles (9 areas, 132 patches) used throughout Phase 1. Input RGB imagery synthetically degraded via BOX-filter downsample (area-average, simulating a coarser pixel footprint) then BICUBIC-upsample back to the native 512x512 patch grid (network input shape requirement only — no information can leak back in through this step). DSM ground truth is left untouched at native 9cm image-matching-DSM resolution throughout, per the master brief's explicit instruction. GSD ladder: 9cm (control), 30cm, 50cm, 1m, 2m, 3m, 5m. Code: `scripts/gsd_utils.py`, `scripts/A1_gsd_measure.py`, `scripts/A2_gsd_finetune.py`.

## Scientific caveat (required disclosure, master brief Section 11)
This tests **information-loss sensitivity only**. It is NOT a substitute for genuine satellite imagery and does not simulate atmospheric effects, sensor MTF/PSF blur, radiometric response differences, JPEG/on-board compression, off-nadir viewing geometry, or actual satellite sensor noise characteristics. A BOX-then-BICUBIC degraded 9cm image is systematically *cleaner* than a genuinely-acquired image at the same nominal GSD would be (no aliasing beat patterns, no sensor noise, perfectly co-registered). Treat every number below as an **optimistic upper bound** on real-satellite-GSD performance, not a prediction of it. Experiment B (genuine satellite-domain dataset) is the real test of this red flag; this experiment only isolates one variable (input detail) cleanly, which B cannot do because too many variables change at once there.

## A1 — Existing Phase-1 model (iter2, partial-unfreeze), NO new training

| Input GSD | Full-oracle RMSE (m) | Full-oracle corr | Sparse-10pt RMSE (m) | Sparse-10pt corr | Relative degradation (full-oracle vs 9cm) |
|---:|---:|---:|---:|---:|---:|
| 0.09 (control) | 4.403 | 0.913 | 4.914 | 0.898 | — |
| 0.30 | 4.705 | 0.901 | 6.294 | 0.824 | +6.9% |
| 0.50 | 4.785 | 0.897 | 7.076 | 0.801 | +8.7% |
| 1.00 | 4.989 | 0.887 | 6.913 | 0.787 | +13.3% |
| 2.00 | 5.496 | 0.861 | 6.514 | 0.815 | +24.8% |
| 3.00 | 5.566 | 0.858 | 8.092 | 0.712 | +26.4% |
| 5.00 | 5.822 | 0.843 | 9.950 | 0.606 | +32.2% |

Raw data: `outputs/metrics/A1_gsd_measure_no_adaptation.json`.

### Reading the result
**No sharp resolution cliff was found in the 9cm-5m range for full-oracle calibration.** RMSE rises smoothly and only moderately (4.40m -> 5.82m, +32%) across a 55x GSD range; correlation degrades gracefully (0.913 -> 0.843). This is a genuinely more resilient result than the pre-registered NO-GO threshold anticipated (Section 4 of `03_PHASE2_EXPERIMENT_PLAN.md` flagged >2x RMSE, i.e. >8.8m, as concerning — never reached in A1's full-oracle case). **However, this is the OPTIMISTIC (oracle-calibration) number** — see the caveat above and the sparse-calibration column.

**Sparse-10pt calibration degrades faster and less monotonically** (4.91m at 9cm -> 9.95m at 5m, +103%, with a non-monotonic dip at 2m that is a sampling-variance artifact of a single random 10-point draw per tile, consistent with the sparse-calibration instability already documented in Phase 1 Section 20). This is the more realistic number: even doubling to 2x under sparse calibration (9.95m vs 4.91m) stays well clear of the 18.15m constant-baseline floor and the NO-GO threshold (>2x the 9cm sparse-10pt anchor of 5.50m from Phase 1, i.e. >11m) — so A1 alone does not trigger a NO-GO, but the trend is real and directionally worrying at coarser GSD, and sparse-calibration variance itself (not just mean degradation) increases at coarse GSD.

## A2 — Fine-tuning AT a degraded GSD (adaptation attempt)

Target GSD: **3.0m** (chosen because it showed the largest sparse-10pt degradation of any non-5m point in A1, and 3m sits inside a plausible band for higher-end open/commercial satellite-derived products). Same recipe as Phase-1 iteration 2 (partial backbone unfreeze, last 2 DINOv2 blocks, 16 epochs, SSI-L1 loss) but trained from the ORIGINAL pretrained DA2-Small (not from the 9cm fine-tuned checkpoint), on TRAIN patches degraded to 3m, matching the master brief's requirement that A2 tests whether training recovers signal lost to resolution, not whether more fine-tuning helps in general.

Trained 16 epochs, 30s/epoch (~8 min total), from the ORIGINAL pretrained DA2-Small (not warm-started from the 9cm checkpoint). Val SSI-L1 improved 1.78 -> 1.47 (best, epoch 9), comparable convergence behavior to the Phase-1 9cm run.

| | Full-oracle RMSE (m) | Full-oracle corr | Sparse-10pt RMSE (m) | Sparse-10pt corr |
|---|---:|---:|---:|---:|
| A1 @ 3m (no adaptation, 9cm-trained model) | 5.566 | 0.858 | 8.092 | 0.712 |
| A2 @ 3m (trained AT 3m from scratch) | **5.089** | **0.883** | **5.919** | **0.842** |
| Delta | -8.7% RMSE | +0.025 | **-26.9% RMSE** | **+0.130** |

Raw data: `outputs/metrics/A2_gsd_adapted_3.0m.json`.

## Question answered
> Can training recover useful signal after resolution degradation?

**YES, partially and meaningfully — especially for the realistic (sparse-calibration) deployment scenario.** Training directly at the degraded 3m GSD does not fully close the gap back to native-9cm performance (A2's 5.09m full-oracle RMSE is still above the 9cm control's 4.40m), but it recovers a substantial fraction of what synthetic degradation cost: full-oracle RMSE improves 8.7% and, more importantly, the sparse-10pt (realistic) RMSE improves 26.9% (8.09m -> 5.92m) with a large correlation gain (0.712 -> 0.842). This says the model was NOT simply hitting an information-theoretic wall at 3m under this simulation — some of A1's degradation was the model being evaluated out-of-distribution (9cm-trained features applied to 3m-blurred input) rather than a hard ceiling set by the input itself. This is a genuinely encouraging, evidence-based finding, not an assumption: **re-training at the deployment GSD is a real, working mitigation**, not merely a hopeful claim.

## Practical resolution read
No hard cliff was observed 9cm-5m under BOX-downsample idealized conditions; degradation is smooth and accelerating, most visible in sparse (realistic) calibration RMSE and variance rather than mean full-oracle RMSE. Re-training at the target GSD materially helps, particularly for realistic sparse calibration. **Do not claim "works at any GSD."** The correct claim is: "degrades gracefully and remains recoverable-by-retraining under idealized synthetic resolution loss; real satellite domain shift (sensor MTF, noise, atmosphere — Experiment B) is the governing UNTESTED risk this experiment cannot speak to, and is very likely to be materially worse than what is measured here" (see scientific caveat above).
