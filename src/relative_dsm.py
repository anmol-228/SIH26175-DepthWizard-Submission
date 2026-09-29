"""Relative (non-georeferenced) DSM branch - handbook Part IV, issue #15.

The official PS splits on input type: a georeferenced input must produce an absolute
metric DSM (that is the calibration path, issue #5), while a plain JPG/PNG with no
coordinates must still produce a *relative* DSM - internally shape-consistent, taller
reads as taller, no metres and no map location.

That is a genuinely separate requirement, and Part IV is explicit that "the model
produces depth maps" is not evidence it is met. This module is the dedicated path:
plain image in, unitless height field out, calibration skipped entirely.

DELIBERATELY DOES NOT IMPORT src/geoio.py. That module is georeference machinery, and
save_geotiff() makes `epsg` a required argument precisely so a georeferenced product is
never written by accident. Keeping the two writers apart is what stops a relative
product being silently handed off as a metric one - the failure mode
role_cards/NIRMALYA.md names for this lane. There is a test asserting this file has no
geoio import.

Contract: fixtures/calibration_output_relative_v0.json,
10_DOCUMENTATION/CALIBRATION_OUTPUT_INTERFACE.md "Product B".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

SCHEMA_VERSION = "calibration_output/v0-relative"
PRODUCT = "relative_dsm"

# A non-georeferenced image has no ground sample distance. heightfield_to_mesh() requires
# a `gsd` and has no default, so the relative branch needs a convention: 1.0 = one mesh
# unit per pixel. Passing a plausible metric GSD instead would smuggle a units claim into
# the mesh, which is exactly what this branch exists to avoid.
PSEUDO_GSD = 1.0

# Fields a relative product must never carry - anything implying a real-world position.
# Mirrors MUST_NOT_CARRY in fixtures/calibration_output_relative_v0.json.
FORBIDDEN_SIDECAR_FIELDS = ("epsg", "crs", "origin", "origin_x", "origin_y",
                            "pixel_size", "pixel_size_x", "pixel_size_y", "nodata")


class RelativeDSMError(ValueError):
    """Raised when input cannot be turned into an honest relative DSM."""


@dataclass
class RelativeDSM:
    """A unitless height field plus the provenance needed to interpret it.

    `array` is normalized to [0, 1]: higher reads as taller. It is NOT elevation in
    metres and carries no datum, which is the entire point of this branch.
    """
    array: np.ndarray                 # (H, W) float32, normalized, unitless
    source_image: str
    depth_sign: int                   # +1 identity, -1 inverted; see infer_relative_depth
    norm: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape       # type: ignore[return-value]

    def sidecar(self) -> dict:
        """Metadata written beside the .npy. Carries the discriminator, and deliberately
        omits every georeferencing field - see FORBIDDEN_SIDECAR_FIELDS."""
        return {
            "schema_version": SCHEMA_VERSION,
            "product": PRODUCT,
            "units": "UNITLESS",
            "meaning": ("internally shape-consistent only: higher value reads as taller. "
                        "No metres, no datum, no map location."),
            "shape": list(self.array.shape),
            "dtype": "float32",
            "source_image": self.source_image,
            "depth_sign": self.depth_sign,
            "normalization": self.norm,
            "pseudo_gsd": PSEUDO_GSD,
            "pseudo_gsd_note": ("1.0 = one mesh unit per pixel. A non-georeferenced image has "
                                "no real GSD; this is a display convention, not a measurement."),
            "IMPORTANT": ("Relative product. Must never be presented as, compared against, or "
                          "converted into a metric DSM without a real reference. See handbook "
                          "Part IV and fixtures/calibration_output_relative_v0.json."),
        }


def load_plain_image(path: str | Path) -> np.ndarray:
    """Load an arbitrary JPG/PNG as (H, W, 3) uint8 - the input this branch exists for.

    Nothing else in the repo does this: every other reader expects a GeoTIFF or a
    pre-made patch bundle. Grayscale and RGBA are converted rather than rejected.
    """
    path = Path(path)
    if not path.exists():
        raise RelativeDSMError(f"input image not found: {path}")
    try:
        with Image.open(path) as im:
            rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        # A corrupt or non-image file is bad input, not a crash. Without this the caller
        # gets a raw PIL traceback, which reads as a pipeline defect rather than "your
        # file is not an image".
        raise RelativeDSMError(f"cannot read {path} as an image: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RelativeDSMError(f"expected (H, W, 3) after RGB conversion, got {rgb.shape}")
    if min(rgb.shape[:2]) < 2:
        raise RelativeDSMError(f"image too small to form a height field: {rgb.shape[:2]}")
    return rgb


def normalize_relative(depth: np.ndarray, *, depth_sign: int = 1,
                       low_pct: float = 2.0, high_pct: float = 98.0) -> tuple[np.ndarray, dict]:
    """Map a raw relative-depth array onto a unitless [0, 1] height field.

    Percentile-based rather than min/max: a single hot pixel would otherwise compress the
    whole scene into a sliver of the range and flatten the visible terrain. Values outside
    the percentile window are clipped, and the chosen bounds are returned so the transform
    stays auditable and reversible.

    depth_sign: +1 keeps the model's ordering (correct for NADIR imagery - DA2's output is
    disparity-like, and looking straight down, closer to the camera means higher elevation).
    -1 inverts it, which is what an oblique or handheld photo needs. This is a parameter
    rather than a constant precisely because the nadir argument does not generalise; see
    the corrected note in scripts/05_zeroshot.py.
    """
    if depth_sign not in (1, -1):
        raise RelativeDSMError(f"depth_sign must be +1 or -1, got {depth_sign!r}")
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise RelativeDSMError(f"need 0 <= low_pct < high_pct <= 100, got {low_pct}, {high_pct}")

    d = np.asarray(depth, dtype=np.float32) * depth_sign
    finite = np.isfinite(d)
    if not finite.any():
        raise RelativeDSMError("depth array has no finite values")

    lo = float(np.percentile(d[finite], low_pct))
    hi = float(np.percentile(d[finite], high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate (flat or near-flat) input: emit a valid flat field rather than
        # dividing by ~zero and producing inf/nan that would poison the mesh.
        out = np.zeros_like(d, dtype=np.float32)
        return out, {"method": "percentile_clip", "low_pct": low_pct, "high_pct": high_pct,
                     "low_value": lo, "high_value": hi, "degenerate": True,
                     "note": "input had no usable dynamic range; emitted a flat field"}

    out = np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~finite] = 0.0
    return out, {"method": "percentile_clip", "low_pct": low_pct, "high_pct": high_pct,
                 "low_value": lo, "high_value": hi, "degenerate": False,
                 "output_range": [0.0, 1.0], "invertible_as": "raw = value*(hi-lo)+lo, /depth_sign"}


def infer_relative_depth(rgb: np.ndarray, model=None, device: str | None = None) -> np.ndarray:
    """Run the depth model on a plain RGB array, returning raw relative depth at (H, W).

    Calibration is skipped entirely - that is the whole point of this branch. Reuses
    src/da2_utils.py, which has no georeferencing coupling and already takes (H,W,3) uint8.
    Imported lazily so the rest of this module stays usable without torch.
    """
    import torch  # noqa: PLC0415
    from da2_utils import load_model, preprocess_rgb, resize_pred_to  # noqa: PLC0415

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = load_model(device)
    model.eval()
    with torch.no_grad():
        x = preprocess_rgb(rgb).unsqueeze(0).to(device)
        pred = model(pixel_values=x).predicted_depth
        return resize_pred_to(pred, rgb.shape[:2]).squeeze(0).cpu().numpy()


def save_relative_dsm(out_dir: str | Path, scene: str, dsm: RelativeDSM) -> tuple[Path, Path]:
    """Write `relative_dsm_{scene}.npy` plus its sidecar JSON. Returns both paths.

    Not a GeoTIFF, by design: the Build Guide's output table says a relative DSM is
    "GeoTIFF if a spatial grid exists; otherwise NPY/PNG", and here one does not.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / f"relative_dsm_{scene}.npy"
    side = out_dir / f"relative_dsm_{scene}.json"

    meta = dsm.sidecar()
    leaked = sorted(f for f in FORBIDDEN_SIDECAR_FIELDS if f in meta)
    if leaked:
        # Belt and braces: a relative product must never gain a georeference by accident.
        raise RelativeDSMError(f"relative sidecar must not carry georeferencing fields: {leaked}")

    np.save(npy, dsm.array.astype(np.float32))
    side.write_text(json.dumps(meta, indent=2) + "\n")
    return npy, side


def relative_dsm_from_image(image_path: str | Path, *, depth_sign: int = 1,
                            model=None, device: str | None = None) -> RelativeDSM:
    """Full path: plain image file -> normalized, unitless RelativeDSM. No calibration."""
    rgb = load_plain_image(image_path)
    raw = infer_relative_depth(rgb, model=model, device=device)
    arr, norm = normalize_relative(raw, depth_sign=depth_sign)
    return RelativeDSM(array=arr, source_image=str(Path(image_path).name),
                       depth_sign=depth_sign, norm=norm)
