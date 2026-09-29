# DepthWizard: model artifact manifest

Model weights and bulk data are **not** in this repository. This manifest records what exists, its identity and how it is obtained, so results stay traceable. None of the files below is published; an authorized evaluator can request the runner-default checkpoint from the team (see `README.md`).

## Model checkpoints (individually hashed)

| Artifact | File name | SHA-256 | Size | Origin | Terms |
|---|---|---|---|---|---|
| **v3 full-encoder EMA checkpoint (runner default)** | `da2_small_finetuned_v3_full_encoder_ema.pt` | `a1e3fca34874856196e53fe9be2be1ee10d1b038c741157dfd6472f74e299ea7` | 99,242,742 bytes | Depth Anything V2 Small; all 12 encoder blocks adapted (lr 5e-6 / 5e-5), SSI-L1, 16 epochs, batch 4, seed 42, all 21 Vaihingen training areas; released weights are the EMA (decay 0.995) of the run, final epoch, no validation-based selection | Derived from Depth Anything V2 Small (Apache-2.0); trained on ISPRS Vaihingen (academic/research-use terms apply to the training data) |
| Partial-unfreeze iteration-2 checkpoint (**B0, rollback only**) | `da2_small_finetuned_iter2_partial_unfreeze.pt` | `74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8` | 99,266,768 bytes | Same base; last 2 of 12 encoder blocks adapted. Source of all frozen baseline figures in this repository | Same as above |
| v2 full-encoder checkpoint (superseded candidate, not the default) | `da2_small_finetuned_v2_full_encoder.pt` | `0bc0c835120cd78e20e66f2cea2bc4729e8b9513ec2b22e6f2a36ed5af80108f` | about 95 MB | Same recipe as iter2 except all 12 encoder blocks adapted (lr 5e-6), seed 42, best-validation epoch 12 | Same as above |
| Earlier iteration-1 checkpoint | `da2_small_finetuned_best.pt` | `146039bc075b6848c569460a4038e8ed0887bc6b1d404cd8fea77fb9e54c4442` | about 95 MB | Frozen backbone; head and neck only | Same as above |
| Hilly-terrain adaptation attempt 1 | `da2_hilly_adapted_attempt1.pt` | `638f2f0c62cae1ba8d7022f9b286a8a8cc982337b259aa864c8f88d0d2fde0b6` | about 95 MB | 17-patch hill-adaptation experiment; no material improvement (see `PHASE2_RED_FLAGS/07_HILLY_TERRAIN_TEST.md`) | Same as above |
| Simulated-3 m-GSD retrained model | `da2_gsd3.0m_adapted.pt` | `ad046a79ff146b870c0de068a9b4e92f90e3c66cb0e1db17a00b380337d44e98` | about 95 MB | GSD stress-test retrain (see `PHASE2_RED_FLAGS/04_GSD_STRESS_TEST.md`) | Same as above |
| Forest research checkpoint | `D2_da2_small_forest_finetuned.pt` | `109aa7d8b595908cecbce69ed795e6c84bcf616f19a1e18891811ee090487648` | about 95 MB | Forest-adaptation experiment (see `PHASE2_RED_FLAGS/06_VEGETATION_FOREST_TEST.md`); not loaded by the shipped pipeline | Same as above |

`run_pipeline.py` verifies the SHA-256 of the checkpoint it loads and refuses a mismatch. `checkpoints/README.md` explains where to place the files.

## Bulk data and regenerable outputs (not published)

| Artifact | Size | Origin | Terms | How to obtain |
|---|---|---|---|---|
| Vaihingen raw imagery and DSM | about 2.2 GB | ISPRS Vaihingen benchmark | Academic/research use; **not redistributed**, do not publish | Obtain from the ISPRS source listed in `DATA_SOURCES.md` and follow its terms |
| Vaihingen processed tiles | about 0.9 GB | Derived from the raw data by the preprocessing scripts in `scripts/` | Same restriction as the raw data | Regenerate with `scripts/` |
| Prediction rasters and native 3D meshes | hundreds of MB | Model inference output | Regenerable; Vaihingen-derived outputs are not published | Regenerate by running inference with a checkpoint |
| Phase-2 experiment data | varies | Mixed public sources per experiment (see `DATA_SOURCES.md`) | Per source | Per-experiment scripts in `PHASE2_RED_FLAGS/scripts/` |
