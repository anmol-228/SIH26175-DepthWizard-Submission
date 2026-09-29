# DepthWizard model v3 (full-encoder adaptation + EMA weights): release evidence

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

- **Status:** runner default as of PR #109 (v3, TRAIN-only selection evidence). iter2 (B0) remains the verified rollback. Demo-machine checkpoint verification (file hash, no-override run, rollback run) is tracked separately by the team and is not part of the evidence claimed in this note.
- **Classification:** internal model-development evidence, 29 Sep 2026.
- **Raw artifacts:** the private development record, folder `10_DOCUMENTATION/evidence/DW_30H_MODEL_STRIKE_20260928/`. v3 results are at commit `5a11ef3`; release build and release validation at `59817bc`.
- **Raw artifacts note (public snapshot):** the raw result files named below are kept in the private development record and are not part of this snapshot.

## Metric

**M1 = reference-aligned relative-shape RMSE.** For each patch, the prediction is fitted to the reference by least squares (scale and shift), and the RMSE is reported. M1 measures the *shape* of the relative surface.

- M1 is **not** deployable DSM RMSE and **not** DSM accuracy.
- The reference is the Vaihingen `dsm_09cm_matching` product, a dense image-matching DSM.

## Claim boundary (binding)

- **What may be claimed:** on held-out Vaihingen *training* areas, the v3 recipe has lower reference-aligned relative-shape error than the v2 recipe, and v2 lower than iter2. It is a reversible runner default. The released file was retrained once on all 21 training areas, so this evidence applies to the recipe; the released file itself has no held-out evaluation.
- **What may not be claimed:**
  - any NAIP, Sentinel-2, mountain, satellite-transfer, metric-DSM or general-generalization improvement;
  - any "no domain degradation" statement;
  - any dev/test result **for v3**: v3 was never evaluated on dev or test (quarantine).

## What v3 is

| | iter2 (B0, rollback) | v2 (superseded candidate) | **v3 (this default)** |
|---|---|---|---|
| Encoder blocks adapted | last 2 of 12 | all 12 | all 12 |
| Released weights | best-val epoch | best-val epoch (12) | **EMA of the run, decay 0.995 per step, final epoch; no validation-based selection** |
| Architecture / preprocessing / output | DA2-Small HF, 518 + ImageNet norm, raw relative surface | identical | identical |
| LRs / loss / epochs / batch / data | 5e-6 / 5e-5, SSI-L1, 16, 4, 21 Vaihingen training areas | identical | identical |
| SHA-256 | `74ccde40…78d8` | `0bc0c835…108f` | `a1e3fca3…9ea7` (full: `a1e3fca34874856196e53fe9be2be1ee10d1b038c741157dfd6472f74e299ea7`) |

## Selection evidence (in order of weight)

### 1. v2 recipe vs iter2: frozen lab screen (protocol v2)

- **Setup:** 7 held-out training areas {2, 5, 14, 17, 22, 27, 33}, 3 seeds, final epoch.
- **Result:** reference-aligned relative-shape RMSE 1.4191 vs 1.6783, **−0.259 m, 7/7 areas better**.
- The full negative record (A1–A4, S1, T1, T2, X1, UL) is in the model-improvement ledger.

### 2. EMA vs the v2 recipe: final train-only lane (protocol v3)

- **Timing:** frozen 14:41 IST, with addendum 1 at 14:45, before any result.
- **Folds:** F1 {3, 7, 15, 20, 23, 30, 35} and F2 {4, 13, 16, 21, 24, 32, 37}. Neither had been held out before; fold 0 was excluded.
- **Pairing:** each candidate was paired against the v2 recipe run of the same fold and seed.
- **Quarantine:** dev and test were never used.

| Paired over seeds 42/43/44, both folds | Value | Frozen threshold |
|---|---|---|
| Mean area delta (EMA − v2 recipe) | **−0.0258 m** | ≤ −0.020 |
| Median area delta | −0.0162 m | < 0 |
| Areas won | **13 / 14** | ≥ 10 |
| Fold means | F1 −0.024, F2 −0.028 | each < 0 |
| Worst area | +0.0014 m | ≤ +0.10 |
| Rotated-view (D4) mean delta | −0.030 m | ≤ +0.05 |
| Forward time vs v2 recipe (idle, paired) | ratio 1.0007 | ≤ 1.10 |
| State-dict keys / shapes / dtypes | identical (287) | identical |

- The seeds-42/43 provisional pass was confirmed on seed 44, as the protocol required.
- The other frozen candidate, UE (also training the embeddings), **failed**: mean +0.002 m, 5/14 areas.
- **Independent method review before execution:** PROCEED WITH CHANGES, with repairs frozen in addendum 1.

### 3. Supporting diagnostics on previously exposed partitions (v2 recipe only; not confirmatory)

- v2 recipe vs iter2 on dev: 1.3595 → 1.1772. On the historical test: 1.8240 → 1.5459. The test figure was a second candidate test look (a disclosed deviation).
- These numbers describe **v2**. **v3 was deliberately not evaluated on dev or test.**

### 4. Out-of-domain (report-only, v2 recipe; not re-measured for v3)

- Sentinel-2 forest proxy: effectively unchanged against iter2.
- Hilly NAIP vs bare-earth DTM: v2 is ~6.7 % **worse** (negative evidence).
- No out-of-domain claim is made for v3.

## Release build (release engineering, not an experiment)

- **One** deterministic run of the frozen winning recipe: UAE (UA + EMA 0.995), all 21 training areas, seed 42, final epoch. No validation read, no dev or test.
- The release weights are the EMA shadow, saved as `da2_small_finetuned_v3_full_encoder_ema.pt`.
- Release validation (tests, default-checkpoint end-to-end run, rollback end-to-end run, timing, manifest, handoff hash) is recorded in the PR.

## Rollback

Set `DW_CHECKPOINT=checkpoints/da2_small_finetuned_iter2_partial_unfreeze.pt` and `DW_CHECKPOINT_SHA256=74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8`, or revert this change.
