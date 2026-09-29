# Issue #91 Track A — constrained calibration refit: result

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

**Status: FAIL (frozen, honestly reported). Predeclared gate did not pass. No merge, per contract's `SHIP_ONLY_IF_PREDECLARED_GATE_PASSES`.**

Contract: `docs/final_sprint/phase2/HILLY_ACCURACY_RESCUE_EXPERIMENT_CONTRACT.md`
Script: `PHASE2_RED_FLAGS/scripts/E_08_track_a_constrained_calibration_refit.py`
Raw evidence: `RESULTS.json` (25 rows), `DECISION.json` (this run's classification)
Independent audit: an independent read-only automated (AI) reviewer — confirmed the FAIL classification is mechanically correct, arithmetic matches raw rows, no seed/threshold/scene substitution, execution began after the preregistration timestamp, Mount Adams reference-point hashes/baseline `a,b` match frozen #73 exactly.

## What was tested

Positive-scale-constrained RANSAC (via sklearn's `is_model_valid` callback enforcing `a>0` during the consensus search itself, not only after) plus a Huber-IRLS refinement pass on the resulting inliers — targeting the #73 N=100 sign-flip anomaly (`a<0` on all 5 Mount Adams seeds under the existing unconstrained fit).

## Result

| | Requirement | Observed |
|---|---|---|
| Valid-fit seeds (Mount Adams) | ≥4/5 | **3/5** (seeds 0,1,2 valid; seeds 3,4 still negative even after the constrained search) |
| RMSE on valid seeds | ≤23.0 m (2× the ~11.5 m oracle reference) | **60.7 – 67.0 m** — roughly 3× over the fail margin |
| Flat/urban control regression | ≤1 m RMSE increase | **not regressed** (deltas: +0.08, +0.01, +0.03, −0.02 m across the 4 Vaihingen controls; all still 5/5 valid) |

Both the valid-fit-count gate and the RMSE-margin gate failed independently; the control did not regress.

## Interpretation (claim boundary, per contract's `CLAIM_IMPLICATION`)

The constrained search recovered a *positive-scale* fit on 3/5 seeds where the baseline found none — the mechanism is not inert — but the recovered fits are not *accurate*: RMSE of 60–67 m is not meaningfully better than not having a metric fit at all. This narrowly rejects "positive-scale-constrained RANSAC + Huber refinement, as specified, recovers a usable Mount Adams calibration at N=100." It does not prove no constrained-fit variant could ever work, and it does not touch #73's own separate, already-frozen local/windowed finding.

A documentation-accuracy correction (not a result change) was made after an independent automated (AI) audit caught that the code comment overstated the refinement's rollback guarantee for seeds 3–4, where the initial RANSAC fit was already negative before refinement ran — the decision logic (`fit_valid()`) already handled this correctly; only the comment was imprecise.

## Rollback / production impact

None. This is an experiment-only additive function (`fit_scale_shift_constrained`) never called from `run_pipeline.py`. The shipped calibration path (`scripts/07_calibrate_and_eval.py`, `run_pipeline.py`) is untouched.

## Handoff

Preserved as frozen negative evidence, same treatment as #73. No further seed-shopping or threshold-tuning was performed after seeing this result, per the stopping rule.
