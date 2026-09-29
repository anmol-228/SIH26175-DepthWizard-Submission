# REPRODUCIBILITY.md

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

**Status:** Corrected after an internal methodology and claim-boundary review. Compiled from verified repository and primary-source evidence.

**REPRODUCIBILITY REFERENCE SHA: `a64b129`**

## Headline status

- **Reproducible inference: SUPPORTED** — see §6.
- **Full training reproducibility: NOT SUPPORTED** — do not claim it. See §7.

## 1. Code

**Reproducibility reference SHA: `a64b129`** — confirmed 2026-09-22 via `git fetch origin && git log --oneline -1 origin/main` in a team member's terminal. `origin/main` advanced `e073be2..a64b129` (PR #79, the #73 confirmatory hilly-failure result), a clean fast-forward with no divergent local commits. This SHA identifies the repository state referenced when this reproducibility document was finalized and verified. It is **not** asserted as the historical producer commit for older saved metric artifacts whose producer commit is not recorded.

## 2. Model checkpoint (accepted, for inference)

**Runner default (v3, from PR #109):**

| Field | Value |
|---|---|
| File | `da2_small_finetuned_v3_full_encoder_ema.pt` |
| SHA-256 | `a1e3fca34874856196e53fe9be2be1ee10d1b038c741157dfd6472f74e299ea7` |
| Size | 99,242,742 bytes |
| Source | Fine-tuned locally from Depth Anything V2 Small on Vaihingen training areas: all 12 encoder blocks adapted, released weights = EMA (decay 0.995) of the run. Selected under frozen, pre-registered **TRAIN-only** protocols; v3 was **not** evaluated on the dev or historical-test partitions. Evidence: `10_DOCUMENTATION/evidence/DW_MODEL_V3_FULL_ENCODER_EMA/README.md` |
| Distribution | **Intentionally gitignored — never committed or uploaded externally** (`ARTIFACT_MANIFEST.md`). Held by the team; an authorized evaluator can request it (see `README.md`). Anyone reproducing inference needs this exact file by hash, not from the repo. |

**Rollback (B0, opt-in, verified earlier baseline):**

| Field | Value |
|---|---|
| File | `da2_small_finetuned_iter2_partial_unfreeze.pt` |
| SHA-256 | `74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8` |
| Size | 95 MB (`ARTIFACT_MANIFEST.md`); ~94.67 MB (`CHECKPOINT_HANDOFF.txt`) |
| Source | Fine-tuned locally from Depth Anything V2 Small on Vaihingen, iteration-2 partial-unfreeze |
| Use | Set `DW_CHECKPOINT=checkpoints/da2_small_finetuned_iter2_partial_unfreeze.pt` and `DW_CHECKPOINT_SHA256=74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8`. |

The B0 hash is cross-confirmed in three independent places: `ARTIFACT_MANIFEST.md`, `CHECKPOINT_HANDOFF.txt`, and issue #40's real A3 end-to-end run (`run_manifest.json`). All frozen baseline figures in this repository (5.50 m sparse-10, 4.40 m oracle, hilly and Mojave failures) were produced with B0, not v3.

## 3. Base backbone model

- `depth-anything/Depth-Anything-V2-Small-hf` — **Apache-2.0** (verified via the HF Hub API, `03_LITERATURE_BENCHMARKS.md`)
- Snapshot hash observed in the local cache used for the recorded A3 run: `5426e4f0f36572d16453bbda7a8389317b1bef99`. This is an **observed local-cache/provenance fact** — the snapshot as it existed on the machine that ran A3 — not a claim that this is the permanent, immutable upstream revision on Hugging Face. Re-downloading today could return a different snapshot hash unless separately re-verified against the HF repo at pin time.
- Offline smoke test passed against that cached snapshot: 287/287 weights matched, no network call made.
- Offline-first policy: backbone download is OFF by default (`--allow-backbone-download` required to opt in) — do not silently fall back to a network download.

## 4. Environment

**Two separate facts — do not conflate them (per the internal review):**

- **Recorded full-stack evidence:** the preserved first non-stubbed end-to-end run was executed under **CPython 3.14.6**, on the recorded Windows/CUDA host described below. This is the actual environment the evidence trail was produced on.
- **Current judge-facing demo launcher:** `viewer/server.py` has its own preflight gate (`SUPPORTED_PYTHON = {(3, 12), (3, 14)}`, PR #108) that constrains the *demo launcher specifically* to Python 3.12.x or 3.14.x. This is a launcher-level constraint, not a project-wide inference/training requirement — `run_pipeline.py` and `scripts/07_calibrate_and_eval.py` do not impose it.
- **Checkpoint identity is enforced at run time:** `run_pipeline.py` verifies the checkpoint SHA-256 against the v3 hash `a1e3fca3...9ea7` by default (or `DW_CHECKPOINT_SHA256` when `DW_CHECKPOINT` points at a deliberately different file, e.g. the B0 rollback `74ccde40...`) before loading, and refuses on mismatch.

**Do not generalize the launcher's version gate to the whole project, to training, or to the reproducible-inference path in general.**

`requirements.txt` separately pins: `numpy==2.5.2`, `scipy==1.18.1`, `tifffile==2026.8.16`, `imagecodecs==2026.8.16` (these require 3.12+ if installing that exact lockset). `torch==2.9.0+cu129` requires the `cu129` wheel index, not plain PyPI.

Training/dev GPU used for the fine-tuning runs: NVIDIA RTX 5060 Ti, 16 GB VRAM, CUDA 13.3 runtime, Windows 11 Pro (`01_ENVIRONMENT_AUDIT.md` / `BASELINE_v1.0.md` §1) — this is the *training* machine; state the demo machine separately if it differs.

## 5. Exact commands (reproducible-inference / held-out calibration-evaluation path)

`scripts/07_calibrate_and_eval.py` is the held-out calibration/evaluation workflow — it requires the appropriate preprocessed inputs already staged, not a generic "run inference on any image" command. Commands below are PowerShell, matching the actual Windows workflow this was run on.

```powershell
# 1. Set up environment
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. Verify the checkpoint hash BEFORE running anything against it
Get-FileHash -Algorithm SHA256 .\checkpoints\da2_small_finetuned_v3_full_encoder_ema.pt
# Expected (runner default, v3): a1e3fca34874856196e53fe9be2be1ee10d1b038c741157dfd6472f74e299ea7
# B0 rollback file: 74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8
# Do not proceed if this does not match.

# 3. Run the held-out calibration/evaluation workflow. This path reproduces the FROZEN
#    B0 baseline (iter2); it does not evaluate v3 (v3 was never evaluated on dev/test).
$env:EVAL_CKPT_NAME = "da2_small_finetuned_iter2_partial_unfreeze.pt"
$env:EVAL_OUT_NAME  = "finetuned_calibration_report_iter2_partial_unfreeze.json"
python scripts\07_calibrate_and_eval.py
```

Full 01–09 pipeline command sequence is in `BASELINE_v1.0.md` §14. All scripts are seeded (`SEED = 42`).

Viewer/demo launch (per #40 / PR #70): `python viewer\server.py`, then open `http://localhost:8793/`. This launcher enforces the Python 3.12.x/3.14.x preflight described in §4 — a demo-launcher constraint, not a general environment requirement.

## 6. What IS reproducible — safe to claim

**"Reproducible inference"**: given the repository state referenced in §1 + the hashed checkpoint (§2) + the documented commands (§5), the same input produces the same output — **on the tested relative branch, on the recorded host/GPU** (§4's training machine: RTX 5060 Ti / CUDA 13.3 / Windows 11). The preserved bit-exact repetitions were recorded with the B0 checkpoint (`74ccde40...`); they have not been repeated for v3.

**Do not generalize this to cross-machine, cross-GPU, CPU-only, or universal deterministic behavior** — those have not been tested and are not claimed here. Only the code/checkpoint/environment/command chain being followable is guaranteed; exact bit-for-bit match on different hardware is not.

## 7. What is NOT reproducible — do not claim this

**"Full training reproducibility" is NOT supported:**

- The fine-tuning checkpoint itself is intentionally gitignored and was never uploaded externally (`ARTIFACT_MANIFEST.md`) — it exists only in the team's private storage.
- Training data splits/preprocessing/config/randomness for the fine-tune runs are documented (`02_DATA_AUDIT.md`, `04_EXPERIMENT_PLAN.md`) but the runs themselves have not been independently re-executed end-to-end by a non-author — `BASELINE_v1.0.md` §1 explicitly flags this as "NOT independently re-run in this pass."
- Determinism (§6) is bounded to one host/GPU — cross-machine/cross-GPU determinism is untested and not claimed.
- Do not claim someone outside the team can retrain from scratch and get the identical checkpoint.

## 8. Dataset licenses / attribution (carried forward from the #76 provenance sweep)

| Dataset | License / access | Attribution required |
|---|---|---|
| ISPRS Vaihingen | Academic benchmark / research-use context | Do not redistribute |
| Sentinel-2 L2A | Copernicus free/open data | "Contains modified Copernicus Sentinel data 2021" (verified against Copernicus's own licence page, per PR #69) |
| ETH Global Canopy Height 2020 | CC BY 4.0 | Citation (verified against the dataset's own project page): Lang, N., Jetz, W., Schindler, K., & Wegner, J. D. (2023). *A high-resolution canopy height model of the Earth.* Nature Ecology & Evolution, 1-12. Also disclose as a GEDI-calibrated model product, not ground survey — never as field truth. |
| USGS 3DEP + NAIP | Public domain | — |
| Copernicus DEM GLO-30 | Copernicus / Airbus–DLR WorldDEM-30 licence | Required notice for adapted/modified use (our case): *"produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."* **Plus the Article 6(c) liability disclaimer, verbatim (previously referenced but not spelled out — added 2026-09-26):** *"The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30."* Both verified directly against the primary licence document (`License-COPDEM-30.pdf`, Copernicus Data Space Ecosystem / Sentinel Hub) — not assumed. |
| Depth Anything V2 Small (base model) | Apache-2.0 | — |

## 9. Open items

- The **reproducibility reference SHA** is pinned (§1, confirmed 2026-09-22). It identifies the repository state referenced for reproducibility verification; it is **not a historical producer SHA for older result artifacts** whose producer commit is not recorded.
- This file replaces `REPRODUCIBILITY_DRAFT.md` as the real, committed `REPRODUCIBILITY.md`.

---
*Compiled by the team from `ARTIFACT_MANIFEST.md`, `CHECKPOINT_HANDOFF.txt`, `BASELINE_v1.0.md`, `03_LITERATURE_BENCHMARKS.md`, `DATASETS.md`, the #40/#69 GitHub evidence trail, and the primary Copernicus DEM licence document. Corrections applied per the internal methodology review.*
