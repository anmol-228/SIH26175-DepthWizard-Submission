# DW-S1-15 / P0-A4 / P-18 — Interpretation of the Official Mojave Sparse/Desert Zero-Shot Evaluation

**Source:** raw report `raw_output/DW_S1_15_P18_MOJAVE_OFFICIAL_ATTEMPT2_20260913T111108Z.json` (SHA-256 `5716d678de3cee3d70a4d659fabcbcaf661c491d49801156f4882fdec715d3e1`), produced by Attempt 2 (the sole successful official run; Attempt 1 produced no scientific output — see `ATTEMPT1_CLASSIFICATION.json`).

## 1. Run validity vs. performance quality (kept strictly separate)

**Run validity: VALID.** The evaluation executed to completion exactly as predeclared and independently verified twice by automated (AI) reviewers: unmodified `scripts/07_calibrate_and_eval.py` at frozen commit `4a442967c9a91361e24241c20a02b3a5bcd83a8e`, the hash-verified handed-off NPZ (`8808d572…b2709a6`), the dual-approved Mojave-only run-local manifest, the accepted iter-2 fine-tuned checkpoint (`74ccde40…eba78d8`), zero-shot with respect to this specific terrain (no training/adaptation on the Mojave sample — confirmed at Gate 4/7 from the checkpoint's own training provenance: 513/42 Vaihingen train/val patches only). No code, parameter, RNG, or calibration-setting changes at any point. Exit code 0, one execution.

**Closure semantic:** Issue #48 explicitly accepts *"any honestly-reported result (PASS/DEGRADED/FAIL/INCONCLUSIVE)"* as sufficient for functional evidence closure — but it does **not** define any numeric threshold for assigning those labels. The project's own dashboard (`PS_REQUIREMENT_CLOSURE_DASHBOARD.md`, P-18 row) records the target status as **`IMPLEMENTED_NOT_VALIDATED (any honest result)`**. Accordingly, **this document does not invent a PASS/DEGRADED/FAIL/INCONCLUSIVE label for the numbers below**, since no numeric bar for assigning one exists anywhere in the project's evidence. This experiment closes the previously-missing P-18 *evidence* gap — it does **not** constitute a *validated* sparse/desert accuracy claim. **Functional/evidence closure and validated performance are two different things, and this document keeps them separate.**

## 2. Raw metrics, exactly as produced — nothing hidden, nothing cherry-picked

| Setting | RMSE (m) | MAE (m) | corr | a | b | n_calib_points | Note |
|---|---|---|---|---|---|---|---|
| Full oracle (dense per-tile calibration — **upper bound, NOT deployable**) | 39.046 | 28.916 | 0.5375 | 46.499 | 612.394 | n/a (all 2,359,296 px) | Oracle only |
| Sparse N=200 | 40.954 | 28.985 | 0.5375 | 28.871 | 655.501 | 200 | |
| Sparse N=50 | 40.039 | 28.727 | 0.5375 | 31.619 | 651.620 | 50 | |
| Sparse N=10 | 39.356 | 29.605 | 0.5375 | 54.374 | 592.133 | 10 | |
| **Sparse N=5** | **88.659** | **70.950** | **−0.5375** | **−96.560** | 1005.240 | 5 | **Sign-inverted — see below** |
| Sparse N=1 | 70.139 | 53.486 | 0.5375 | 1.000 | 691.313 | 1 | shift-only fallback (a=1.0, n<2 branch) |

Global-pooled figures are identical to the per-tile figures above in every row, because the dual-gate-approved run-local manifest contains only this single Mojave tile — there is nothing else to pool with. This is expected and was anticipated at Gate 4/9, not an error.

**The N=5 result is preserved exactly as produced, not smoothed or explained away.** At N=5 calibration points, RANSAC fit a *negative* scale coefficient (`a = −96.560`, vs. positive `a` values of roughly +29 to +54 at every other N). Since Pearson correlation is invariant to positive affine transforms but flips sign under a negative one, and the underlying raw-model-vs-reference-DEM correlation magnitude is consistently ≈0.5375 across every single setting in this table (oracle and every sparse-N alike), the N=5 row's negative correlation and much larger errors are a direct, mechanical consequence of an unstable, sign-flipped 5-point robust fit — not a different underlying phenomenon. **This is a genuine, honestly-reported finding about the fragility of very-low-point sparse calibration on this terrain/tile, not a discarded or cherry-picked-away result.**

## 3. What this run establishes

- One genuinely sourced, provenance-complete, sparse/desert terrain sample (Mojave Desert, southern Nevada, Sentinel-2 `S2A_11SNA_20210927_1_L2A` RGB paired with a USGS 3DEP `NV_Southern_D23` lidar-derived DEM) was evaluated **zero-shot with respect to this terrain** — i.e., using the project's already-selected, already-fine-tuned iter-2 checkpoint, with no training or adaptation performed on this specific sample.
- The model's raw relative-depth signal shows a **moderate, consistent correlation (|r| ≈ 0.54)** with the paired USGS 3DEP reference elevation on this desert tile — noticeably weaker than would indicate strong terrain-height sensitivity, but not zero/random either.
- Absolute error under the *non-deployable* full-oracle calibration is **~39 m RMSE**; under the sparse-reference calibration stress test it ranges **~39–89 m RMSE** depending on how many reference points are available, with a **demonstrated robustness failure at N=5 points** on this tile.
- The P-18 functional gap (zero prior code path, zero prior evidence) is now closed with a genuine, honestly-reported result, per the project's own defined closure criterion for this requirement.

## 4. What this run does NOT establish (explicit claim boundary)

- **Not universal desert-terrain performance.** This is one sample, one tile, one acquisition date, one sensor pairing. No claim is made about sparse/desert terrain generalization beyond this single tested case.
- **Not an India-domain claim.** This tile is in Nevada, USA; nothing here bears on Indian desert/arid terrain specifically.
- **Not a hilly-terrain claim.** P-18 (sparse/desert) is unrelated to the separate, already-documented hilly-terrain calibration failure (~60–70 m RMSE) — the two must not be conflated or averaged.
- **Not a production-wide/general satellite-imagery accuracy claim.** This uses one specific Sentinel-2 scene, not a general validation of satellite-derived RGB across sensors/conditions.
- **The full-oracle number (39.046 m RMSE) is an oracle, not a deployable accuracy figure**, exactly as labeled in the raw report and the evaluator's own inline comments — it uses dense per-tile reference elevation for calibration, which would not be available in real deployment.
- **USGS 3DEP `NV_Southern_D23` is a lidar-derived bare-earth DEM, not a surveyed DSM.** The `dsm` dictionary key is only the evaluator's field-name contract (Gate 6/8 dual-verified); the source product itself must not be described as a "surveyed DSM."
- **This result is not compared to, averaged with, or used to resolve** the unrelated historical Vaihingen sparse-10 RMSE discrepancy (~5.5006 m vs. ~4.9139 m) — those numbers are a different terrain/sensor/resolution domain entirely and remain out of scope here, per explicit instruction.

## 5. Preserved negative finding

The N=5 sparse-calibration sign-flip (corr −0.5375, RMSE 88.66 m, MAE 70.95 m) is flagged here as a **load-bearing negative finding**, consistent with this project's evidence-preservation policy: sparse calibration at very low reference-point counts is demonstrably fragile on this terrain sample and must not be omitted from any downstream summary, slide, or dashboard entry that cites this evaluation.
