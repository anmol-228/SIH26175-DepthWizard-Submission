# DW-P0-A5 (#32) — reference-elevation relief gate: exit evidence

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

**Status: PROPOSED implementation evidence, pending Nirmalya's required technical review
and Debopriya's evidence/release review (per #32's 2026-09-11 Phase-2 activation comment).
Not yet merged.**

Base: `origin/main` @ `24d0e24d5e46243c35aa463f9e4bf95deb88c42b`.
Branch: `feature/dw-p0-a5-hilly-relief-refusal`.

## What this proves

Two real invocations of `run_pipeline.main()` (the actual CLI entry point, not a unit-level
call to `preflight_checks()` directly) — model stages stubbed exactly as
`tests/test_hilly_gate_dw_s1_11.py` already does for this same DoD item under DW-S1-11, so
this reuses an established, already-accepted evidence pattern rather than inventing a new
one. Both cases share identical, well-spread reference-point geometry (5x5 stratified grid,
passes count/hull-coverage/spread on their own) so the only variable between them is the
reference points' own elevation relief:

- `non_trigger_flat_run_manifest.json` — relief 25.92 m (under the 87.5 m provisional
  floor). Gate passes, `branch=metric`, a metric GeoTIFF/mesh is written.
- `trigger_hilly_relief_run_manifest.json` — relief 200.0 m (above the floor). Gate
  declines, `branch=relative`, exit code 0 (not an error), no `*_metric.*` artifact is
  written, `gate_reason` names the relief and its threshold, and `preflight.checks`
  carries `reference_relief_m`, `reference_relief_status`, `reference_relief_threshold_m`,
  and `reference_relief_evidence`.

`SUMMARY.json` is the condensed form of both runs' key fields, plus the base SHA/branch this
evidence was generated from.

This is exactly the "genuinely-hilly-but-well-spread reference set" DW-S1-11's own test
file documented as a gap the old (rejected) model-output signal family could not catch —
this new, separate, geometry-only signal narrows that gap by catching it whenever the
supplied reference points' own elevations happen to read a relief above the provisional
floor. It does not close the gap outright: see "Limitations" below for the disclosed case
(a genuinely hilly scene sampled in a narrow elevation band) this specific check still
cannot catch.

## Threshold provenance

**87.5 m** (`src/preflight.py::HILLY_RELIEF_THRESHOLD_M`), derived directly from the SAME
statistic compared at runtime — not from a different (dense) statistic. Full derivation,
exact per-sample values, and the frozen selection rule are in
[`reference_relief_threshold_analysis.json`](./reference_relief_threshold_analysis.json).

**Primary source**: `PHASE2_RED_FLAGS/outputs/metrics/E_hilly_risk_signal_experiment.json`,
field `scenes.*.signal_B_sparse10_ransac_on_blended[*].sampled_reference_range_m` — per
RANSAC seed, `max - min` over the exact 10 sampled reference points used in that seed's
fit. This is the identical operation `reference_elevation_relief()` performs on a real
`--reference` file's elevations — a **sparse**, runtime-available statistic, not a dense
one. 5 seeds × 5 scenes, all included regardless of downstream fit validity (that is
post-fit information a pre-fit threshold must not condition on):

| Group | Scenes | Samples | Range observed |
|---|---|---|---|
| Known-good | 4 Vaihingen areas (area6, area8, area34, area38) | 20 | 7.774 – 44.355 m |
| Known-failure | Mount Adams | 5 | 130.650 – 266.360 m |

No overlap: `good_max` (44.355 m) < `failure_min` (130.650 m). Frozen rule, declared before
computing these numbers: `threshold = (good_max + failure_min) / 2` = **87.50 m** (exact:
87.50271606445312, rounded for the code constant). In-sample at this threshold: 0
known-good samples above it, 0 known-failure samples at/below it.

**Corroborating context only** (does not derive the threshold — both are DENSE, whole-patch
or full-AOI statistics, a different statistic from the sparse one above):

- `PHASE2_RED_FLAGS/outputs/metrics/E_hilly_terrain_eval.json`: 28 Mount Adams per-patch
  oracle calibrations at the runner's own 512px window scale, both checkpoint variants.
  Observed lower-relief patches (relief_m <= 118.54 m, 26/28 patches) → oracle RMSE
  3.03-23.89 m; observed high-relief/high-error patches (relief_m >= 222.80 m, 2/28
  patches) → oracle RMSE 41.66-62.33 m.
- `(internal note, not included)` §3/§8 (full-AOI scale): Vaihingen area6
  67.16 m vs Mount Adams 370.4 m.

These two dense views are related evidence from the same preserved Mount Adams/Vaihingen
data, not independent confirmations of each other, and neither derives the 87.5 m figure —
they corroborate that the direction (more relief → worse outcome) holds at a different
statistic and scale, nothing more precise than that.

**Provisional, not validated** — `thresholds_provisional=True` on every result, same as
every other threshold in `src/preflight.py`.

## Limitations

- **n=1 known-failure scene.** The 5 Mount Adams samples are 5 different reference-point
  resamples of the SAME underlying terrain, not five independent failure terrains.
  Resampling one scene does not create independent evidence of generalization to other
  hilly terrain.
- **n=4 known-good scenes**, all from the same flat-to-rolling Vaihingen ISPRS benchmark
  region — no genuinely "moderately hilly" known-good control exists in this repository's
  preserved data.
- **Simulated reference points.** The sampled elevations in the source experiment are drawn
  from each scene's evaluation DSM (`y = gt[rr, cc]`), not independently surveyed field
  GCPs — see `E_05_hilly_risk_signal_experiment.py`'s own "SPARSE-REFERENCE PROVENANCE"
  note.
- **In-sample counts, not a validated error rate.** "0 false-refuse / 0 false-accept"
  describes this one preserved dataset (n=25 total samples), not a statistical rate that
  generalizes to unseen scenes.
- **Not that hilly-terrain accuracy is solved**, or that the model performs well in hilly
  terrain. Hilly-terrain accuracy stays a known, disclosed BROKEN failure. When this check
  triggers (a high sparse reading), it declines rather than emitting a confident wrong
  metric answer for that run — it does not fix the underlying calibration, and it does not
  make the hilly failure mode safe in general: a hilly scene the supplied points happen to
  under-sample would not trigger it.
- **Not field-validated.** 87.5 m is a provisional engineering separator computed by one
  frozen, predeclared rule over a small preserved direct-statistic gap, not a number a
  validation study selected after testing candidates.
- **Not a claim that every hilly scene is caught.** This check only sees the elevation
  spread of whatever reference points a caller actually supplies; a genuinely hilly scene
  sampled by reference points that happen to sit within an 87.5 m band would still pass
  this specific check (though it may still be caught by the existing DW-S1-10 geometry
  checks, independently).
- **Not a change to the DW-S1-11 (#32, PR #35) finding** that no defensible signal exists
  in the model's own relative-surface output — `hilly_relief_signature()` and
  `checks["hilly_signature"]` are untouched and still return `None`/"unsupported" for that
  documented, separate reason.

## Reproduce

```
"python" -m pytest tests/test_preflight.py tests/test_hilly_gate_dw_s1_11.py -v
```

The two manifests here were captured from a real `run_pipeline.main()` invocation (model
stages stubbed, per the established `tests/test_hilly_gate_dw_s1_11.py` pattern); the
pytest command above re-derives the same two cases (and more) as durable, CI-checkable
tests rather than a one-off script, which was deleted after generating this evidence.

`reference_relief_threshold_analysis.json` was generated by a one-off script (also
deleted after running) that reads only `E_hilly_risk_signal_experiment.json` — no network,
no checkpoint, no new data — and applies the frozen selection rule described above.
