# DW-S1-15 / P0-A4 / P-18 — Official Mojave Sparse/Desert Zero-Shot Evaluation Evidence

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

**Issue:** internal tracker item 48 (remains OPEN — not closed by this evidence)
**Status:** Evidence prepared for Debopriya's evidence/release review. NOT yet committed, staged, or pushed.

## 1. What this closes

P-18 ("performance stability... on sparse... landscapes", `PS_REQUIREMENT_CLOSURE_DASHBOARD.md` row) was previously **MISSING** — zero code path, zero evidence, zero prior task. This evidence package provides one genuinely sourced, provenance-complete sparse/desert terrain sample, evaluated zero-shot (with respect to this terrain — no training/adaptation on the sample) through the unmodified accepted evaluation pipeline. Issue #48 explicitly accepts *"any honestly-reported result (PASS/DEGRADED/FAIL/INCONCLUSIVE)"* as sufficient for functional evidence closure, but it does not define a numeric threshold for assigning those labels. The dashboard status remains **`IMPLEMENTED_NOT_VALIDATED (any honest result)`**; therefore this experiment closes the previously-missing P-18 *evidence* gap without claiming *validated* sparse/desert accuracy. Functional/evidence closure and validated performance are kept distinct throughout this package.

## 2. Execution history — both attempts preserved

**ATTEMPT 1 — HARNESS/SETUP FAILURE, NO SCIENTIFIC RESULT.** Exit code 1. The isolated git worktree (correctly pinned to `4a442967c9a91361e24241c20a02b3a5bcd83a8e`) did not contain the gitignored iter-2 checkpoint file — `git worktree add` only materializes tracked files, and `checkpoints/*.pt` is gitignored. The run crashed in `torch.load(CKPT, ...)` before `model.load_state_dict()` completed — zero inference, zero predictions, zero metrics were produced. See `attempt1_stdout.txt`, `attempt1_stderr.txt`, `attempt1_run_metadata.json`, `ATTEMPT1_CLASSIFICATION.json`, `predeclaration_attempt1.json`. This was NOT a hidden dry run — it was a genuine official invocation attempt that failed on setup, preserved exactly as it happened.

**ATTEMPT 2 — AUTHORIZED CORRECTIVE RUN.** After human authorization, the exact already-hash-verified checkpoint (SHA-256 `74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8`) was copied byte-for-byte into the worktree and independently re-verified before proceeding. No other scientific input, code, parameter, RNG, checkpoint selection, or sample changed. Exit code 0. See `stdout_attempt2.txt`, `stderr_attempt2.txt`, `run_metadata_attempt2.json`, `predeclaration_attempt2.json`, `raw_output/`.

**The first official invocation terminated before evaluation because the isolated worktree lacked the gitignored frozen checkpoint. No P-18 metric was produced. The failure was preserved. A human-authorized corrective invocation was then performed after copying the exact previously selected checkpoint byte-for-byte into the worktree and independently re-verifying its SHA-256. No scientific input, model selection, parameter, RNG, sample or evaluation code changed.**

Only Attempt 2's result is used for scientific interpretation, since Attempt 1 produced none.

## 3. Governing provenance chain (full detail in `predeclaration_attempt2.json`)

| Element | Value |
|---|---|
| Frozen repo SHA (= origin/main tip) | `4a442967c9a91361e24241c20a02b3a5bcd83a8e` |
| Producer PR | #63, MERGED, reviewed/approved by Nirmalya (`DevNiru2704`) |
| Handoff | Internal tracker item 48 |
| Artifact | `mojave_sparse_desert_p18.npz`, SHA-256 `8808d5721ae6c2186ccb046965780a586c2af9c973a9bfc15adf19f31b2709a6`, 16,515,570 bytes |
| Run-local manifest | Mojave-only test-split entry (see `predeclaration_attempt2.json` for exact JSON + justification) |
| Checkpoint | `da2_small_finetuned_iter2_partial_unfreeze.pt`, SHA-256 `74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8` — the project's accepted "FINAL headline result" checkpoint, fine-tuned only on standard Vaihingen train/val patches (513/42), never on the Mojave sample |
| Evaluator | `scripts/07_calibrate_and_eval.py`, unmodified, git-blob SHA-256 `eb5609db5634efd611741b9cee8e578e4c7f804a3c289b51fee35ba1f4e004ab` |
| Environment | Windows 11, Python 3.14.6, torch 2.9.0+cu129, CUDA (RTX 5060 Ti), numpy 2.5.2, sklearn 1.9.0, transformers 5.15.1 — see `environment.txt`/`pip_freeze.txt`. NOT claimed identical to the historical Linux/Python 3.13.8 rerun environment. |
| Exact command (Attempt 2) | see `predeclaration_attempt2.json` → `exact_command` |

Sentinel-2/3DEP provenance, CRS/nodata handling, and full handoff text are preserved verbatim in `handoff_snapshot/issue_48_full_exact_text_chronological.txt`.

## 4. Raw result

Raw output file: `raw_output/DW_S1_15_P18_MOJAVE_OFFICIAL_ATTEMPT2_20260913T111108Z.json`, SHA-256 `5716d678de3cee3d70a4d659fabcbcaf661c491d49801156f4882fdec715d3e1`.

See `INTERPRETATION.md` for the full metric table and bounded claim boundary. Headline (full detail, including the preserved N=5 sign-inverted-calibration finding, in `INTERPRETATION.md`):

- Full-oracle (non-deployable upper bound): RMSE 39.05 m, MAE 28.92 m, corr 0.538
- Sparse-reference calibration stress test: RMSE ranges 39.4–88.7 m depending on reference-point count N, with a genuine robustness failure (sign-inverted fit, corr −0.538) at N=5 — preserved, not hidden.

**DSM export transparency note:** metric evaluation completed successfully, but no calibrated DSM GeoTIFF was exported — the exporter found no matching `.tfw` georeference file for this area and correctly refused to invent one (stdout: 0 calibrated DSM rasters written, export skipped with reason logged). This is expected behavior, not an evaluation failure; the metric numbers above are unaffected.

## 5. Claim boundary (see `INTERPRETATION.md` §4 for full text)

One Mojave/southern-Nevada sample only. Not universal desert performance. Not an India-domain claim. Not a hilly-terrain claim. Not a production-wide satellite-accuracy claim. Full-oracle figure is an oracle, not deployable. USGS 3DEP `NV_Southern_D23` is a lidar-derived bare-earth DEM, not a "surveyed DSM" (the `dsm` dict key is only the evaluator's field-name contract). Not compared to or averaged with the unrelated historical Vaihingen sparse-10 discrepancy (~5.5006 m vs. ~4.9139 m).

## 6. Dual-control verification ledger

Every gate in this evidence chain (manifest/evaluator-protocol identity, producer lineage, geospatial/provenance coherence, checkpoint/config lineage, evaluator code audit, isolated worktree, environment freeze, predeclaration, corrective setup, metric extraction/interpretation) was independently verified by an independent read-only automated (AI) reviewer session in addition to the AI author's own check's own analysis. Two real defects were caught and corrected during this process (a bash path-separator bug in the frozen command, and a CRLF-vs-blob hash-recording gap) — see `ORCHESTRATION_LOG.jsonl` for the complete, timestamped record of every gate, every verdict, and every correction.

## 7. What is intentionally NOT included here

The 16.5 MB NPZ, the 99 MB checkpoint, and any raw 3DEP/Sentinel-2 source tiles are NOT copied into this evidence bundle — only their hashes and provenance are recorded, per repository policy against committing large binary datasets. The GitHub attachment (Issue #48 comment `5652139162`) remains the authoritative source for re-obtaining the exact artifact.

## 8. Outstanding before closure

- Debopriya's evidence/release review (not yet requested)
- `PS_REQUIREMENT_CLOSURE_DASHBOARD.md` and `FINAL_CLAIM_EVIDENCE_MATRIX.md` updates (not yet made)
- Human-controlled git staging/commit/push/PR (not performed by this process)
- Issue #48 remains OPEN and must not be closed until the above are complete
- Issue #32 / P0-A5 has NOT been started
