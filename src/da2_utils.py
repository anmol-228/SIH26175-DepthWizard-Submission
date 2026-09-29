"""Manual Depth Anything V2 preprocessing (bypasses transformers' AutoImageProcessor,
which hard-requires torchvision -- and no working torchvision build exists for
Python 3.14 + cu129 at experiment time; see 01_ENVIRONMENT_AUDIT.md).

Preprocessing exactly matches depth-anything/Depth-Anything-V2-Small-hf's
preprocessor_config.json (fetched directly from the model repo): resize to
518x518 bicubic, rescale to [0,1], ImageNet mean/std normalize.
"""
import numpy as np
import torch
from PIL import Image

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 518

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# Revision provenance: REVISION_NOT_ESTABLISHED. The exact HuggingFace snapshot used at
# fine-tune time was never durably recorded (training configs/logs carry only the model
# id). We therefore do NOT pin a revision - the default offline load resolves whatever
# snapshot the local cache's `refs/main` points to, which is byte-identical to what the
# un-pinned call already resolved. Pin MODEL_REVISION here only if an exact revision is
# ever genuinely verified from training evidence.
MODEL_REVISION: str | None = None


class BackboneUnavailableError(RuntimeError):
    """The HF backbone is not available locally and DepthWizard will not download it."""


class BackboneRevisionUnverifiedError(BackboneUnavailableError):
    """An online backbone download was requested for fine-tuned inference, but the
    checkpoint's historical base revision is not durably established."""


def preprocess_rgb(rgb_uint8: np.ndarray) -> torch.Tensor:
    """rgb_uint8: (H,W,3) uint8 -> normalized (3,518,518) float32 tensor."""
    img = Image.fromarray(rgb_uint8).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def preprocess_batch(rgb_batch_uint8: list[np.ndarray]) -> torch.Tensor:
    return torch.stack([preprocess_rgb(r) for r in rgb_batch_uint8])


def load_model(device="cuda", *, allow_backbone_download: bool = False,
               for_finetuned_checkpoint: bool = False):
    """Load the base DA2-Small HF backbone.

    Offline-first and fail-closed by default: `allow_backbone_download=False` passes
    `local_files_only=True` so `from_pretrained` never touches the network, and a missing
    local backbone raises `BackboneUnavailableError` with no online retry. Pass
    `allow_backbone_download=True` (the runner's `--allow-backbone-download`) to permit an
    online resolve/download - the only code path that may reach huggingface.co.

    `for_finetuned_checkpoint=True` means the caller will apply a fine-tuned checkpoint to
    this backbone. Combined with an unpinned MODEL_REVISION, an online download would
    resolve whatever upstream `main` happens to be today - which can stay architecture and
    state-dict compatible, load strictly, and still shift predictions. That combination is
    refused up front, before any online-capable construction.

    No `revision` is pinned (REVISION_NOT_ESTABLISHED - see MODEL_REVISION). Architecture,
    dtype and the subsequent `.to(device)` are unchanged from the previous implementation.
    """
    # Refuse BEFORE importing/constructing anything that could resolve online.
    if allow_backbone_download and for_finetuned_checkpoint and MODEL_REVISION is None:
        raise BackboneRevisionUnverifiedError(
            f"Refusing to download '{MODEL_ID}' for fine-tuned inference.\n"
            f"  The base-model revision used when the fine-tuned checkpoint was trained is "
            f"NOT durably established (REVISION_NOT_ESTABLISHED): the training configs and "
            f"logs record only the model id, never a revision or commit hash.\n"
            f"  Downloading now would fetch whatever upstream 'main' currently is. Such a "
            f"snapshot can remain architecture- and state-dict-compatible, load strictly, "
            f"and still silently change prediction comparability against the DepthWizard "
            f"baseline.\n"
            f"  No revision has been fabricated, and no observed local cache snapshot has "
            f"been promoted to historical provenance.\n"
            f"  - Provision the already-vetted local backbone cache on this machine and "
            f"re-run WITHOUT --allow-backbone-download.\n"
            f"  - --allow-backbone-download remains available for a deliberate "
            f"base-model-only run (--base-model), which applies no fine-tuned checkpoint.")

    from transformers import AutoModelForDepthEstimation

    kwargs: dict = {"local_files_only": not allow_backbone_download}
    if MODEL_REVISION is not None:
        kwargs["revision"] = MODEL_REVISION

    if allow_backbone_download:
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, **kwargs)
    else:
        try:
            model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, **kwargs)
        except Exception as exc:  # local cache miss / corrupt cache - do NOT retry online
            raise BackboneUnavailableError(
                f"The base backbone '{MODEL_ID}' is not available in the local "
                f"HuggingFace cache, and DepthWizard will not download it silently.\n"
                f"  - To provision it offline: copy an existing HF cache containing "
                f"'{MODEL_ID}' to this machine's HF cache dir, or set HF_HOME to one.\n"
                f"  - To allow a one-time online download explicitly: re-run with "
                f"--allow-backbone-download (run_pipeline.py), or call "
                f"load_model(..., allow_backbone_download=True).\n"
                f"  Underlying cause: {type(exc).__name__}: {exc}"
            ) from exc

    return model.to(device)


def resize_pred_to(pred: torch.Tensor, size_hw) -> torch.Tensor:
    """pred: (B, h, w) or (B,1,h,w) -> (B, H, W) resized with bicubic."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    out = torch.nn.functional.interpolate(pred, size=size_hw, mode="bicubic", align_corners=False)
    return out.squeeze(1)
