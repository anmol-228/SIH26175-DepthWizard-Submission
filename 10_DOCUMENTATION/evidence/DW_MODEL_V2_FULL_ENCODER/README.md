# DepthWizard model v2 (full-encoder adaptation): evidence summary

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

- **Status:** superseded candidate. The runner default is v3 (`DW_MODEL_V3_FULL_ENCODER_EMA/README.md`); iter2 (B0) remains the verified rollback. The dev/test figures below describe the **v2 recipe** only, on partitions that were already exposed; they are not v3 results.
- **Classification:** internal model-development evidence, 28–29 Sep 2026.
- **Raw artifacts:** the private development record, commit `4a3afa7`, folder `10_DOCUMENTATION/evidence/DW_30H_MODEL_STRIKE_20260928/`.
- **Raw artifacts note (public snapshot):** the raw result files named below are kept in the private development record and are not part of this snapshot.

## Metric used throughout

**M1 = reference-aligned relative-shape RMSE.** For each patch, the prediction is fitted to the reference by least squares (scale and shift), and the RMSE of the fitted surface is reported. M1 measures the *shape* of the relative surface.

- M1 is **not** deployable DSM RMSE and **not** DSM accuracy.
- The reference is the Vaihingen `dsm_09cm_matching` product, which is a dense image-matching DSM.

## Claim boundary (binding)

- **What may be claimed:** v2 has lower reference-aligned relative-shape error than iter2 on Vaihingen, one town and one flight. It was proposed as a reversible runner default and was superseded by v3; it is not the current default. The released file was retrained once on all 21 training areas, so this evidence applies to the recipe; the released file itself has no held-out evaluation.
- **What may not be claimed:**
  - any NAIP, Sentinel-2, mountain, satellite-transfer, metric-DSM or general-generalization improvement;
  - any "no domain degradation" statement;
  - any untouched-test result.

## What changed

| | iter2 (rollback) | v2 |
|---|---|---|
| Architecture / preprocessing / output contract | DA2-Small HF, 518 bicubic + ImageNet norm, raw relative surface | identical |
| Encoder blocks adapted | last 2 of 12 | all 12 (patch and position embeddings still frozen) |
| LRs, loss, epochs, batch, training data | 5e-6 encoder / 5e-5 neck and head, SSI-L1, 16 epochs, batch 4, 21 Vaihingen training areas | identical |
| Seed / epoch selection | 42 / best val SSI-L1 | 42 / best val SSI-L1 (epoch 12) |
| File / SHA-256 | `da2_small_finetuned_iter2_partial_unfreeze.pt` / `74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8` | `da2_small_finetuned_v2_full_encoder.pt` / `0bc0c835120cd78e20e66f2cea2bc4729e8b9513ec2b22e6f2a36ed5af80108f` |

## 1. Primary selection evidence: frozen 7-area, 3-seed lab screen

- **Protocol:** v2, frozen before results; hashes and timestamps are in the evidence folder.
- **Split:** 7 training areas {2, 5, 14, 17, 22, 27, 33} were held out. Every candidate trained on the other 14 areas with seeds 42/43/44, final epoch, and was compared with 3 control seeds of the iter2 recipe. Dev and test were never read by these runs.

| Candidate | Lab M1 (3-seed mean) | vs control 1.6783 (SD 0.012) | Areas better | Gate A |
|---|---|---|---|---|
| Adapt 4 blocks | 1.5760 | −0.102 | 7/7 | pass |
| **Adapt all 12 blocks (v2 recipe)** | **1.4191** | **−0.259** | **7/7** | **pass; displaces the 4-block candidate (−0.157, 7/7)** |
| Weight soup of controls | 1.6647 | −0.014 | 7/7 | fail (margin) |
| Two-view averaging | 1.6760 | −0.002 | 5/7 | fail |
| Axis-consistent augmentation | 1.7057 | +0.027 | 4/7 | fail |
| 12 blocks + annealed 8-epoch tail | 1.3987 | −0.280 | 7/7 | fails the frozen margin against v2 (−0.020, needs −0.03) |
| 12 blocks + doubled encoder LR | 1.4259 | −0.252 | 7/7 | fail (rotated-view error +0.075) |

- An earlier round of four orientation-robustness recipes all raised normal-view error on dev by 0.23–0.35 m, so none was adopted.
- **Limits:** three seeds; seven spatially correlated areas from the same town. The lab had already screened several families before v2, so it is not a pristine benchmark either.

## 2. Supporting diagnostics on previously exposed partitions (not confirmatory)

Dev and historical test were both used in earlier project work. The results below are **supporting diagnostics**, not independent confirmation.

| Partition | iter2 | v2 | Change | Areas |
|---|---|---|---|---|
| Dev (3 areas; also used for epoch selection, as for iter2) | 1.3595 | 1.1772 | −0.182 | 3/3 better |
| Historical test (9 areas) | 1.8240 | 1.5459 | −0.278 | 9/9 better |
| Test, per-area full-reference fit (oracle ceiling, not deployable) | 4.4034 | 4.2404 | −0.163 | — |
| Test, 10 reference points per area, median of 200 paired draws (resampling of the exposed test set; not new evidence) | 5.663 | 5.471 | −0.216 | v2 lower in 70 % of draws |

- **Deviation, retained:** the 4-block candidate was also test-evaluated before v2 was selected. v2's test figure is therefore a **second candidate test look** (`PROTOCOL_ERRATA.md` item 3). Selection used the lab rule only.
- **Two dev looks were pre-registered**, one for each full-data build.

## 3. Post-hoc release-safety assessment (amendment 6)

- **When:** written *after* the first independent review returned NO-GO, and after the hilly numbers below were known. It was frozen before the forest numbers existed.
- **What it is:** a release-safety / robustness screen.
- **What it is not:** it does **not** restore confirmatory status to the historical test or to any other partition.

| Proxy (report-only; neither is a DSM validation) | iter2 | v2 | Reading |
|---|---|---|---|
| Sentinel-2 forest (true colour vs ETH canopy height, 25 patches): M1 / median r | 7.277 / 0.120 | 7.273 / 0.148 | **effectively unchanged; supports no improvement claim** |
| Hilly NAIP vs bare-earth 3DEP DTM (28 patches): M1 | 11.081 | 11.821 | **negative evidence: v2 ~6.7 % worse** |

- On the hilly reference, every fine-tuned model (and iter2) is negatively correlated. That reference is unsuitable for any positive generalization claim, but the v2 regression on it is reported as-is.
- The rule blocks promotion only when a candidate is worse on both proxies; v2 is therefore eligible, and nothing more is concluded from it.

## 4. Operational checks

| Check | Result |
|---|---|
| Forward time (idle RTX 5060 Ti, paired and interleaved) | iter2 19.14 ms, v2 19.15 ms (ratio 1.0006) |
| Strict load / state-dict keys | identical; no missing or unexpected keys |
| Tests with v2 as default | 179 pytest passed; viewer 16/16 (`gates/UA_RELEASE_TESTSUITE_AND_E2E.txt`) |
| Default-checkpoint end-to-end run (NAIP, no environment override) | `run_manifest.json`: checkpoint `da2_small_finetuned_v2_full_encoder.pt`, sha256 `0bc0c835…`, `sha256_verified: true` |
| Independent methodology review | NO-GO, then repairs, then PASS WITH REPAIRS (repairs applied) |

## Rollback

Set `DW_CHECKPOINT=checkpoints/da2_small_finetuned_iter2_partial_unfreeze.pt` and `DW_CHECKPOINT_SHA256=74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8`, or revert this PR.
