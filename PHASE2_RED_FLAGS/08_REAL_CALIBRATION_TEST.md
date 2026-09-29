# 08_REAL_CALIBRATION_TEST.md — Experiment F: Realistic Calibration Sources

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

## What could NOT be done, and why (read first)
The master brief's primary design for this experiment (Sections 22-24) is: fetch a genuinely real external DEM (Copernicus GLO-30) over the EXACT geographic footprint of the Vaihingen test tiles, and use it — pixel-aligned — as the calibration source, scored against the independent image-matching DSM. **This is not achievable with legitimately-accessible Vaihingen data.** This session discovered (see `02_DATASET_RESEARCH.md`) that the freely-downloadable Vaihingen tiled benchmark subset's declared coordinates (EPSG:32633/UTM 33N) resolve to 14.96°E, six degrees away from Vaihingen's real longitude (~9°E) — a wrong-UTM-zone artifact indicating the public release's absolute geolocation cannot be trusted. The true absolute-georeferenced raw data is exactly the request-form-gated release Phase-1 already found inaccessible. Fabricating a "real DEM calibration" result against admittedly-wrong coordinates would be worse than not doing it, so this specific sub-experiment is marked **UNRESOLVED for Vaihingen specifically** and answered instead via the literature-grounded fallback the master brief itself sanctions (Section 25: "If actual GCP-quality truth is unavailable, simulate realistic noise levels... derive relevant noise bands from documented source accuracy").

Real, pixel-aligned external-DEM calibration IS attempted on genuinely-geolocated new data in Experiments D/E (vegetation/hilly), where this project controls acquisition and can confirm real coordinates from the start.

## F-noise — calibration-point VALUE noise sensitivity (Section 25)

Method: take the Phase-1 fine-tuned (iter2) model's predictions on the real 9cm Vaihingen test set, sample 10 points/tile uniformly (matching the Phase-1 "realistic" comparator), add i.i.d. Gaussian noise to the TRUE elevation value at each calibration point (not to the model's prediction) before fitting scale+shift, then score on the full tile. 15 seeds/band. Noise magnitudes taken directly from documented, cited real-source accuracy (`02_DATASET_RESEARCH.md` §4) — not invented.

| Band | σ (m) | Mean RMSE (m) | Std | Source |
|---|---:|---:|---:|---|
| Oracle (no noise) | 0.0 | 5.07 | 0.35 | — |
| ICESat-2 ATL08 terrain | 1.5 | 4.92 | 0.22 | Neuenschwander & Magruder 2019 |
| Copernicus GLO-30 (typical) | 1.7 | 4.93 | 0.23 | ISPRS Annals X-4-W1-2022-209-2023 |
| GEDI terrain (low-lat) | 4.0 | 5.17 | 0.28 | cited comparative study |
| Copernicus GLO-30 (rough terrain) | 7.7 | 5.97 | 0.51 | LiDAR-DTM comparison, same research pass |
| Stress test | 10.0 | 6.66 | 0.72 | not source-derived, upper-bound stress only |

Raw data: `outputs/metrics/F_calibration_noise.json`.

**Reading the result:** point-VALUE noise alone is remarkably well-tolerated — even GLO-30's rough-terrain LE90 band (7.7m) only degrades RMSE by ~18% (5.07m -> 5.97m), because least-squares scale+shift fitting over N=10 points averages out i.i.d. noise. **Critical caveat, do not drop this when citing the result:** this simulation is i.i.d. per-point noise only. A REAL external DEM's error is spatially autocorrelated and can carry a regional SYSTEMATIC bias (the whole tile shifted the same direction), which averaging over 10 points does NOT cancel out the way it cancels random noise — real degradation from an actual external DEM is very likely worse than this number suggests. This experiment answers "is the calibration fit fragile to point-level noise" (answer: no, not much) — it does NOT answer "is it fragile to systematic external-DEM bias" (untested, and expected to be the dominant real-world risk based on how least-squares regression responds to correlated error).

## F-distribution — calibration-point SPATIAL DISTRIBUTION sensitivity (Section 26)

Method: same predictions, compare 4 point-selection strategies at fixed N (10, 20), 15 seeds each: **uniform** (Phase-1's original method), **clustered** (all N points from one small contiguous region — simulates a single GCP survey visit or a narrow satellite altimetry swath), **edge-biased** (points concentrated at height discontinuities), **terrain-stratified** (points spread proportionally across the tile's elevation deciles).

| N | Strategy | Mean RMSE (m) | Std | Range |
|---:|---|---:|---:|---|
| 10 | uniform | 4.86 | 0.25 | 4.48-5.46 |
| 10 | **clustered** | **15.32** | **7.12** | **6.75-28.42** |
| 10 | edge-biased | 6.69 | 0.46 | 5.96-7.68 |
| 10 | terrain-stratified | **4.70** | **0.15** | 4.52-4.99 |
| 20 | uniform | 4.67 | 0.14 | 4.51-4.99 |
| 20 | clustered | 10.97 | 3.44 | 6.11-17.92 |
| 20 | edge-biased | 6.04 | 0.26 | 5.69-6.60 |
| 20 | terrain-stratified | **4.55** | **0.08** | 4.46-4.78 |

Raw data: `outputs/metrics/F_calibration_distribution.json`.

**This is the single most important calibration finding of Phase 2.** Point COUNT is not the governing variable — point DISTRIBUTION is, and the effect size is enormous: 10 badly-clustered points (mean RMSE 15.3m, worst-case 28.4m) perform barely better than the 18.15m constant-baseline "do nothing" number, i.e. **10 real reference points can be almost worthless if they all come from one small area** (e.g. a single field survey visit, or a satellite altimetry track that only crosses one corner of the scene) — while the SAME count of terrain-stratified points (4.70m) slightly BEATS uniform-random (4.86m) and has the lowest variance of any strategy tested. Edge-biased calibration (points concentrated at height discontinuities — a plausible real strategy if GCPs are placed at visible landmarks like building corners) is worse than either uniform or stratified, though far better than clustered.

**Practical implication for red flag #9 (real GEDI/ICESat-2/GCP distribution):** real satellite altimetry (GEDI, ICESat-2) tracks are narrow, roughly-linear swaths — closer to "clustered/linear" than "uniform" in this taxonomy for a single pass over one small scene. A single altimetry pass crossing one tile should be treated with real suspicion, not assumed adequate just because it clears a point-count threshold; multiple crossing passes or additional non-collinear points are needed for real spatial coverage.

## Calibration failure policy (Section 27) — design proposed, spot-validated against the evidence above

Proposed rule for a real deployed system:

```
if n_reference_points < 10:
    use shift-only calibration (single offset, matches Phase-1's own finding
    that 1-point shift-only beat unstable 5-point scale+shift fits)
    mark output confidence LOW
elif spatial_coverage_metric(points) < threshold:   # e.g. convex-hull area
    fraction of tile area, or max pairwise distance / tile diagonal
    use shift-only calibration regardless of point COUNT
    mark output confidence LOW
    (this directly follows from the clustered-points result above: point
    count alone is an insufficient gate)
else:
    use full scale+shift calibration
    mark output confidence based on point count + spatial spread jointly
```

This was NOT built and re-run as a separate standalone experiment (time-budget discipline, master brief Section 45) — it is a design recommendation directly justified by two pieces of hard evidence already gathered in this Phase: (1) Phase-1's own 1-pt-vs-5-pt finding, (2) this session's clustered-vs-uniform/stratified finding. A team implementing this system should build the `spatial_coverage_metric` check as a real pre-flight gate before trusting any scale+shift fit, not just count how many points they have.

## Net answer to Red Flag #8/#9/#10/#11

- #8 (real coarse/noisy DEM as calibration source): **UNRESOLVED for Vaihingen** (georeferencing blocker, disclosed above), **point-VALUE noise robustness demonstrated** (F-noise), systematic-bias risk remains untested and is flagged as the more concerning unknown.
- #9 (real sparse GCP/GEDI/ICESat distribution): **PARTIALLY addressed** — the distribution experiment (F-distribution) is a direct, real proxy for this exact risk and shows it is severe; real altimetry-track-shaped point sets should be assumed closer to the "clustered" risk case than "uniform" until proven otherwise on real track geometry.
- #10 (instability below ~10 points): **CONFIRMED, refined** — Phase-1's single-seed non-monotonic finding is now understood as one symptom of a larger distribution-sensitivity effect, not a fixed magic number; N=10 uniform/stratified is stable (std 0.15-0.25m), but N=10 or even N=20 clustered is not (std 3.4-7.1m).
- #11 (spatial-distribution sensitivity): **CONFIRMED as the dominant calibration risk**, more load-bearing than point count.
