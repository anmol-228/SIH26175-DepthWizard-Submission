# DW-FS-03 / Issue #73 — Confirmatory N=100 Results Summary

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

**Bounded R&D confirmatory pass, NOT part of A5 (#32), NOT merged into it, and
NOT a production change.** Base: `origin/main` @
`e073be25536efc298956b0e38173b42b9cc6f7c4`. Branch:
`experiment/dw-fs-03-hilly-confirmatory`. Run exactly once under the frozen
protocol; one pre-execution hardening pass occurred before any inference ran
(see `PRE_REGISTRATION.md` "Note on v1 → v2"); zero bug-correction reruns were
needed after the scientific run started (clean run, exit code 0).

## 1. FACT

### 1.1 Governance

- GitHub Issue #73 (internal tracker item 73),
  frozen preregistration comment posted 2026-09-19T05:48:18Z, independently
  re-verified against the live GitHub API before this run began. No other
  comment existed on the issue.
- Confirmatory script:
  `PHASE2_RED_FLAGS/scripts/E_07_hilly_regional_calibration_confirmatory_n100.py`,
  final sha256 `f79b229cc62a9a22fc86fc68137e7dab67710ff9b2e955e66bb970df049fdc08`.
- Checkpoint `da2_small_finetuned_iter2_partial_unfreeze.pt`, sha256
  `74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8` (matches
  the authoritative preregistered value, re-verified in this worktree).
- Historical screening script (`E_06_regional_sparse_calibration_probe.py`)
  sha256 `2adbf206ba563d719d091b5c1d702251f070fc9d92f86341a89d089eeca92606`,
  independently re-hashed and confirmed to match the value stated in the
  execution brief; historical evidence directory
  `10_DOCUMENTATION/evidence/DW_HILLY_REGIONAL_CAL_PROBE/` was never accessed
  for writing in this worktree (it does not even exist here — this worktree
  only ever wrote to `DW_FS03_HILLY_CONFIRMATORY_N100/`).
- Run: single execution, `RUN_LOG.txt`, exit code 0, no exceptions. 25 runs
  recorded (5 scenes × 5 seeds × N=100), matching the frozen design exactly.
  Full per-file identities and command in `RUN_COMMAND.txt`, `FILE_HASHES.sha256`,
  `ENVIRONMENT.txt`.
- Independent review: an independent read-only automated (AI) reviewer
  reviewed the script before execution (the private review record) and
  the evidence after execution (the private review record). Pre-run
  review confirmed correct protocol implementation and flagged a legitimate
  pre-execution hardening gap (no no-clobber guard, no base-SHA/shape
  assertion), which was fixed before the scientific run (v1→v2, no scientific
  parameter changed). Post-run audit independently recomputed the decision
  numbers from raw `RESULTS.json` values and confirmed they match
  `DECISION.json` exactly, and confirmed `STOPPING_RULE_PASS = false` is the
  mechanically correct conclusion.

### 1.2 Mount Adams (known-failure scene) — per-seed raw result

| Seed | Global RMSE (m) | Global fit valid | Local RMSE (m) | Local metric coverage | Positive seed? |
|---|---|---|---|---|---|
| 0 | — | **False** (a=-20.45) | 38.50 | 38.7% | No |
| 1 | — | **False** (a=-15.58) | 38.37 | 38.7% | No |
| 2 | — | **False** (a=-29.03) | 55.48 | 44.7% | No |
| 3 | — | **False** (a=-49.16) | 35.45 | 38.7% | No |
| 4 | — | **False** (a=-23.55) | 61.93 | 47.7% | No |

At N=100 the global whole-scene fit was invalid (`a < 0`) in all five frozen
seeds; invalid global fits had also occurred in earlier screening, but not at
this 5/5 frequency (2–3/5 seeds were valid at N=10/N=50). Because
`global_rmse` is undefined when the global fit is invalid,
`RMSE_DELTA` is undefined and every Mount Adams seed is scored as **not
positive** by the frozen per-seed rule. Local metric coverage (38.7–47.7%) was
also below the 80% bar on every seed regardless.

**Positive Mount Adams seeds: 0/5** (required: ≥4/5).

### 1.3 Known-good controls — per-scene aggregate (median across 5 seeds)

| Scene | Median global RMSE (m) | Median local RMSE (m) | Median local coverage | Degraded? | Degradation reason |
|---|---|---|---|---|---|
| vaihingen_area6 | 11.01 | 8.11 | 65.2% | **Yes** | coverage < 80% (local RMSE actually lower) |
| vaihingen_area8 | 2.33 | 2.14 | 65.2% | **Yes** | coverage < 80% (local RMSE actually lower) |
| vaihingen_area34 | 1.93 | 2.29 | 59.1% | **Yes** | both: local RMSE higher AND coverage < 80% |
| vaihingen_area38 | 2.84 | 2.53 | 75.4% | **Yes** | coverage < 80% (local RMSE actually lower) |

**Degraded controls: 4/4** (allowed: 0/4). Three of the four controls
(area6, area8, area38) actually have a *lower* median local RMSE than global
at N=100 — the degradation classification for those three is driven entirely
by the frozen 80% local-metric-coverage safeguard, not by worse point
accuracy. Only area34 is degraded on both the RMSE and coverage criteria.

### 1.4 Mechanical decision

```
positive_seed_count       = 0   (required >= 4)
degraded_control_count    = 4   (required == 0)
STOPPING_RULE_PASS        = FALSE
```

Independently recomputed and confirmed by an independent read-only automated (AI) reviewer session against the raw `RESULTS.json` values (see
the private review record).

## 2. INTERPRETATION

Increasing sparse-calibration support from N=10/50 (historical screening) to
the preregistered N=100 did **not** produce the hoped-for improvement on
Mount Adams, and did **not** avoid degrading the known-good controls. If
anything, the confirmatory result is *more* negative than the historical
screening on both axes: 0/5 positive Mount Adams seeds (vs. 2/5 apparent
benefit at N=50) and 4/4 degraded controls (vs. 2/4 at N=10/N=50).

The dominant mechanism differs by scene:
- On Mount Adams, at N=100 the global whole-scene fit was invalid (`a < 0`)
  in all five frozen seeds; invalid global fits had also occurred in earlier
  screening, but not at this 5/5 frequency (2–3/5 seeds were valid at
  N=10/N=50). This removes the very baseline the local candidate was being
  compared against, and the local candidate's own low coverage (38.7–47.7%)
  never cleared the 80% screening bar in any case.
- On the four Vaihingen controls, the local candidate's point accuracy
  (RMSE) was frequently *as good or better* than global, but its structural
  coverage limitation (a consequence of the fixed K=5 nearest-reference
  RANSAC design on non-overlapping 512px windows) stayed well under the
  frozen 80% bar in every scene — the same limitation already disclosed in
  the historical N=10/N=50 probe, now confirmed to persist unchanged at
  N=100.

This confirmatory pass therefore constitutes a **negative result** against
the preregistered stopping rule: the pass criteria required improvement
*and* the absence of degradation, and neither controlling condition was met.

## 3. LIMITATION

- One known-failure terrain (Mount Adams) — not independent evidence of
  generalization to any other hilly/mountain terrain.
- Reference points are SIMULATED CALIBRATION CONTROLS SAMPLED FROM
  EVALUATION DSM — not field GCPs, not independent field validation.
- Four known-good controls, all drawn from the same flat-to-rolling ISPRS
  Vaihingen benchmark region — no genuinely "moderately hilly" known-good
  control exists in this repository's preserved data.
- K=5 and window=512px are one fixed, previously-chosen configuration, not
  retuned or searched here (per the frozen #73 no-parameter-sweep rule) —
  this result says nothing about whether a different K, window size, or
  blending scheme would behave differently.
- The Mount Adams global-fit-validity failure at N=100 is a new observation
  relative to N=10/N=50 and is reported as raw fact; this evidence does not
  by itself explain *why* larger N pushed the global RANSAC fit to a
  negative slope on every seed — that would require separate investigation,
  out of scope for #73.
- No production integration, no India-specific validation, no claim beyond
  this one scene/protocol.

## 4. CLAIM BOUNDARY

Under this exact frozen #73 protocol (N=100, seeds 0–4, RNG stream
`7000+seed`, Mount Adams + 4 Vaihingen controls, K=5/512px local candidate,
identical global RANSAC baseline), the regional/windowed sparse-calibration
candidate **did not clear the preregistered stopping rule**:
`STOPPING_RULE_PASS = FALSE` (0/5 positive Mount Adams seeds against a
required ≥4/5; 4/4 controls classified degraded against an allowed 0/4).

This is a **negative confirmatory result** under this specific, bounded
protocol. It does **not** prove the candidate can never help hilly-terrain
calibration under a different design; it does not generalize beyond Mount
Adams and these four Vaihingen controls; the reference points remain
simulated, not field GCPs; and no result-shopping, parameter variation, or
second scientific pass is permitted under #73 in response to this outcome.
Hilly-terrain accuracy remains a known, disclosed severe failure mode; A5
(#32) remains the primary shipped hilly response and is unaffected by this
issue's outcome in either direction.

## 5. Outcome classification

**NEGATIVE / FAIL** against the preregistered stopping rule — the candidate
is materially contradicted at N=100 on both the primary-scene and
control-degradation criteria (worse on both axes than the already
INCONCLUSIVE N=10/N=50 historical screening). This is not "INCONCLUSIVE":
the frozen rule's two conditions both failed decisively (0/5, not close to
4/5; 4/4 degraded, not close to 0/4).

## 6. Recommended next action

Do not integrate this candidate into production or into A5. Do not run a
second confirmatory pass, parameter sweep, or alternate design under #73 —
that is explicitly out of scope. If this direction is pursued further, any
next step (e.g. investigating the N=100 global-fit sign flip on Mount Adams,
or a different local design) requires a new, separately-scoped, reviewed
issue — not a continuation of #73.
