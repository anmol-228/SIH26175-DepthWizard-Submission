# 05_REAL_SATELLITE_DOMAIN_TEST.md — Experiment B: NOT RUN (UNRESOLVED)

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

This file exists so the gap in the numbered sequence is explicit rather than silent.

## Status: UNRESOLVED by explicit user decision

Priority 2 of the master brief (Section 12-14): find and evaluate a genuinely satellite-sensor dataset with real elevation truth. `02_DATASET_RESEARCH.md` independently verified one strong real candidate — **DFC2019/US3D** (WorldView-3 satellite, 0.35m pan / ~0.3m pansharpened RGB, real airborne-LiDAR DSM truth, Jacksonville FL & Omaha NE) — currently accessible via a free IEEE DataPort account, but under a **restricted research-use license** that the account holder must accept personally.

Creating accounts and accepting binding license agreements on the user's behalf are both outside what this agent does autonomously (per its own operating rules, and independently flagged as an explicit stop-condition in the master brief itself — "a gated dataset legally requires my personal acceptance"). The user was asked directly (`AskUserQuestion`, mid-session) and chose: **"Proceed without them"** — i.e., continue Phase 2 without this dataset, and mark satellite-domain transfer UNRESOLVED with reduced confidence in the final verdict, rather than skip it silently.

## What this means for the rest of Phase 2
- Red Flag #5 (aerial→satellite sensor/domain shift) and the satellite-sensor half of Red Flag #15 (cross-dataset generalization) remain genuinely untested.
- Every real cross-domain result in Phase 2 (forest, hilly, India) used airborne-camera or basemap-tile RGB imagery, never a genuine satellite sensor. The GSD stress test (Experiment A) tests resolution loss in isolation but explicitly cannot simulate real sensor MTF, radiometric response, or atmospheric effects.
- This is identified in `15_UPDATED_GO_NO_GO.md` as **the single largest remaining unretired risk** across both phases.

## If the team later wants to close this gap
The path is known and verified, not hypothetical: register a free IEEE DataPort account, accept DFC2019's research-use license, download the Jacksonville/Omaha tiles (baseline code at `github.com/pubgeo/dfc2019`), and run the same B0-B5 ladder this file was meant to contain (constant baseline → zero-shot transfer → scene calibration → light fine-tuning → fine-tuning + sparse calibration), using the exact same methodology (`src/metrics.py`, `src/da2_utils.py`, spatially disjoint train/val/test scenes) established throughout this project.
