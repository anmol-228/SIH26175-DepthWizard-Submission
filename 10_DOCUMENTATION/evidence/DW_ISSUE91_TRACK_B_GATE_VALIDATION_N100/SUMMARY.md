# Issue #91 Track B — reference-quality-gated calibration: result

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

**Status: FAIL against the contract's own failure threshold, but for a reason distinct from a failed implementation — see "What this actually found" below. No production code changed.**

Contract: `docs/final_sprint/phase2/HILLY_RESCUE_EXECUTION_COUNCIL.md` Part 4
Script: `PHASE2_RED_FLAGS/scripts/E_09_track_b_gate_validation_n100.py`
Raw evidence: `RESULTS.json` (25 rows: 5 Mount Adams + 20 Vaihingen-control), `DECISION.json`

## Correction to the contract's premise (code-first, per this workspace's evidence hierarchy)

The contract describes Track B as *adding* a hull-coverage/stratification gate to `src/preflight.py`, alongside the existing 87.5 m relief gate. Reading `src/preflight.py` and its git history (`git blame`) before writing any code showed this is not accurate: `preflight_checks()` has gated on **count, `hull_coverage`, AND `normalized_spread`** — independently of the later relief gate — since commit `24a95b5` ("DW-S1-10: calibration pre-flight gate (count + coverage), wired into the runner", Nirmalya, merged 2026-09-02), three weeks before #91 was opened. It is already wired into `run_pipeline.py:231` and already covered by `tests/test_preflight.py`'s clustered-vs-stratified test pairs at small N (N=10–16).

No new gating code was written. Writing a second, redundant gate would have contradicted this workspace's own code policy ("do not add unnecessary abstractions") and the contract's own merge rule ("do not invent a new interface if an existing one can express it").

## What this actually found

What the contract's own DATASET/METRICS/SUCCESS-THRESHOLD sections call for — real validation evidence at the N=100 geometry #73/#91 Track A actually use, not the small N=10–16 synthetic geometry the existing unit tests use — had never been produced. Running the **existing, unmodified** `preflight_checks()` against real N=100 uniform-random reference sampling (same SEED_BASE=7000 stream, same 5 seeds) on Mount Adams and all four Vaihingen controls surfaced a genuine, previously-undocumented limitation:

| | Contract requirement | Observed |
|---|---|---|
| Mount Adams (known-bad) declines | all 5/5 | **5/5 declined** ✓ |
| Flat/urban control (known-good) false refusals | ≤1 additional vs. existing baseline (0) | **20/20 declined — 100% false-refusal rate** ✗ |

Every Vaihingen-control run was declined, not by the relief gate (their reference relief stays well under the 87.5 m floor, as the existing DW_P0_A5 evidence shows), but by the `spread` check firing on **all** scenes including the known-good ones.

**Mechanism** (confirmed analytically and with a direct test, `tests/test_track_b_gate_validation.py::test_spread_formula_is_mean_nn_distance_over_diagonal_not_scale_invariant`): `normalized_spread()` is mean nearest-neighbour distance ÷ AOI diagonal. Under uniform random sampling, expected nearest-neighbour distance shrinks as ~1/√N while the AOI diagonal is fixed — so the statistic necessarily falls as N grows, regardless of whether the points are genuinely well-spread or not. `MIN_NORMALIZED_SPREAD = 0.05` was set against N=10–16 evidence (`F_calibration_distribution.json`) and is not scale-invariant; at N=100 it flags essentially everything, including geometrically excellent coverage.

## What was NOT done, and why

Per the frozen stopping rule ("no iterative threshold-tuning after seeing results"), `MIN_NORMALIZED_SPREAD` and `normalized_spread()`'s formula were **not** changed to make this pass. That would be exactly the seen-result-driven tuning the contract forbids. This is reported as frozen negative/limitation evidence, the same treatment as #73 and Track A.

## Rollback / production impact

None. `src/preflight.py` was not modified — this is a pure validation-evidence run against already-shipped code. The 100% false-refusal finding is real and already live in production for any real N≈100 reference set on a large AOI, but it predates this session; nothing shipped today changes that exposure.

## Handoff

This is a real, actionable finding distinct from Track A's calibration-accuracy question: the existing hull/spread gate does not scale correctly to realistic reference counts on large scenes. It needs its own follow-up (likely: a scale-aware spread statistic, or re-deriving `MIN_NORMALIZED_SPREAD` per-N rather than as a fixed constant) — out of scope for today's fixed 1-hour Track B budget, and explicitly not attempted here to avoid post-hoc threshold tuning. Flag for Nirmalya (preflight.py co-owner) and Anmol.
