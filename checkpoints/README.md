# checkpoints/

Model weights are not part of this repository. To run inference, place the file here under this exact name:

| File | SHA-256 |
|---|---|
| `da2_small_finetuned_v3_full_encoder_ema.pt` (runner default, v3) | `a1e3fca34874856196e53fe9be2be1ee10d1b038c741157dfd6472f74e299ea7` |
| `da2_small_finetuned_iter2_partial_unfreeze.pt` (B0 rollback, opt-in via `DW_CHECKPOINT` and `DW_CHECKPOINT_SHA256`) | `74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8` |

`run_pipeline.py` verifies the SHA-256 before loading and refuses any other file. There is no public download route; an authorized evaluator can request the files from the team (see the top-level README) and verify the hashes above.
