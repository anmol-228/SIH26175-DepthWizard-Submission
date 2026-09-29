"""run_pipeline.py - one command, arbitrary optical image -> labelled height surface.

DW-S1-07 (#24). validate -> tile -> predict -> blend -> calibrate, then the Day-3 output
stages: a relative `.glb` mesh (+ LOD) always, and on the metric branch a GeoTIFF plus its
own `.glb` (+ LOD). `run_manifest.json` records provenance for every product.

Why this exists: nothing in the repo turned an arbitrary image into a height surface.
`scripts/05_zeroshot.py` needs DSM ground truth just to run, and `scripts/08_mini_3d_mesh.py`
calls `fit_scale_shift(pred, dsm)` - both are structurally unable to work without ground
truth. This is the first path that does not need any.

Two output branches, never conflated (handbook Part IV):
  relative - unitless, no CRS, no datum. Always produced.
  metric   - metric elevation from a gated scale/shift fit, on a real affine transform.
             Only with a georeferenced GeoTIFF input AND a valid --reference that passes
             the DW-S1-10 pre-flight gate. Horizontal CRS is the input's embedded code if
             it has one, otherwise geoio.CRS_UNDEFINED - the runner never asserts a CRS.

Usage:
    python run_pipeline.py <image> [--reference <gcps>] [--out <dir>]

--reference is vertical elevation control points only: `.npy` or whitespace/comma text,
shape (N, >=3) as (row, col, elevation_m) in the input pixel grid. Not a DEM raster, not a
horizontal georeference, does not override the input transform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from relative_dsm import (  # noqa: E402
    PSEUDO_GSD, RelativeDSM, RelativeDSMError, load_plain_image, normalize_relative,
    save_relative_dsm,
)
from model_stages import blend_tiles, predict_depth  # noqa: E402
from preflight import preflight_checks  # noqa: E402
from tiling import DEFAULT_STRIDE, DEFAULT_WINDOW, plan_windows  # noqa: E402

MAX_PIXELS = 120_000_000        # ~11k x 11k. Guards against an accidental OOM, not a policy.

# The checkpoint behind BASELINE_v1.0's headline result ("Reported as FINAL headline
# result" per BASELINE_v1.0.md:37) and the one K1_tiling_stitch.py hardcodes. The Day-3
# exit gate compares runner output against that frozen baseline, so this is the default
# rather than 07_calibrate_and_eval.py's `_best` (which is iteration 1).
# v3 (2026-09-29): same DA2-Small architecture, preprocessing and output contract as iter2. All 12
# encoder blocks are adapted (iter2 adapted the last 2), and the released weights are the exponential
# moving average (decay 0.995) of that training run. Both choices were selected under frozen,
# pre-registered protocols on held-out TRAINING areas only; the all-12-block recipe was also evaluated
# once on dev and on the historical test as supporting diagnostics (not confirmatory); the EMA step was
# never evaluated on dev/test. Evidence: 10_DOCUMENTATION/evidence/DW_MODEL_V3_FULL_ENCODER_EMA/README.md.
# Rollback = iter2: DW_CHECKPOINT=checkpoints/da2_small_finetuned_iter2_partial_unfreeze.pt with
# DW_CHECKPOINT_SHA256=74ccde40cf703095ed2adb783394d0dd2692b3ffaaf3b8a9e534e27feeba78d8.
DEFAULT_CHECKPOINT = "da2_small_finetuned_v3_full_encoder_ema.pt"
# SHA-256 of DEFAULT_CHECKPOINT as recorded in ARTIFACT_MANIFEST.md / BASELINE_v1.0.md. A
# file with the right name but other contents (a stale or partial copy) runs without any
# error and silently changes every number, so the runner verifies the hash before loading.
# DW_CHECKPOINT / DW_CHECKPOINT_SHA256 select a different checkpoint and its expected hash.
EXPECTED_CHECKPOINT_SHA256 = "a1e3fca34874856196e53fe9be2be1ee10d1b038c741157dfd6472f74e299ea7"
EXIT_BAD_INPUT = 2              # per #24: bad input -> clean message + non-zero exit


class PipelineError(RuntimeError):
    """A run that cannot honestly continue. Message is shown to the user verbatim."""


# --- stage 1: validate -------------------------------------------------------------
def validate_input(image_path: str | Path) -> np.ndarray:
    """Readable, 3-band after conversion, sane dimensions. Raises PipelineError otherwise.

    Reuses relative_dsm.load_plain_image, which already converts grayscale/RGBA to RGB and
    rejects unreadable or degenerate files - no second image loader.
    """
    path = Path(image_path)
    try:
        rgb = load_plain_image(path)
    except RelativeDSMError as exc:
        raise PipelineError(str(exc)) from exc

    h, w = rgb.shape[:2]
    if h * w > MAX_PIXELS:
        raise PipelineError(
            f"{path.name} is {w}x{h} = {h * w / 1e6:.0f} MP, above the {MAX_PIXELS / 1e6:.0f} MP "
            "guard. Crop or downsample it, or raise MAX_PIXELS if you have the memory.")
    if min(h, w) < DEFAULT_WINDOW:
        raise PipelineError(
            f"{path.name} is {w}x{h}; both sides must be at least {DEFAULT_WINDOW} px so at "
            "least one inference window fits. Supply a larger image.")
    return rgb


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictor(checkpoint: Path | None, *, allow_base_model: bool,
                   allow_backbone_download: bool = False):
    """Load DA2-Small and apply the fine-tuned checkpoint. Returns (model, device, record).

    `da2_utils.load_model()` returns the *base* HF model; every other script in the repo
    applies a checkpoint's state_dict after it. Skipping that step silently produces
    un-finetuned predictions that look completely normal - no error, no warning, just
    quietly wrong numbers. So a missing checkpoint is a hard failure here unless
    --base-model is passed explicitly, and either way the manifest records exactly which
    weights ran.

    The required fine-tuned checkpoint is validated *before* the HF backbone is
    constructed, so a missing checkpoint never triggers a (network-capable) backbone
    load. `load_model` is offline-first: it will not reach the network unless
    `allow_backbone_download=True` (the runner's --allow-backbone-download).
    """
    import torch                                                     # noqa: PLC0415
    from da2_utils import MODEL_ID, load_model                       # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not allow_base_model and (checkpoint is None or not checkpoint.exists()):
        raise PipelineError(
            f"checkpoint not found: {checkpoint}\n"
            "  The fine-tuned weights are gitignored and are not in a fresh clone; obtain "
            "them per ARTIFACT_MANIFEST.md.\n"
            "  Refusing to fall back to the base model silently - that would produce "
            "un-finetuned predictions that look normal. Pass --base-model if you genuinely "
            "want the base model.")

    checkpoint_sha = None
    if not allow_base_model:
        checkpoint_sha = _sha256(checkpoint)
        expected = os.environ.get("DW_CHECKPOINT_SHA256", EXPECTED_CHECKPOINT_SHA256).strip().lower()
        if checkpoint_sha != expected:
            raise PipelineError(
                f"checkpoint hash mismatch: {checkpoint}\n"
                f"  SHA-256 found   : {checkpoint_sha}\n"
                f"  SHA-256 expected: {expected}\n"
                "  This is not the checkpoint the reported results were produced with. Copy the "
                "canonical file (see ARTIFACT_MANIFEST.md) or, for a deliberately different "
                "checkpoint, set DW_CHECKPOINT and DW_CHECKPOINT_SHA256 together.")

    # for_finetuned_checkpoint gates the provenance refusal: downloading an arbitrary
    # current upstream snapshot to pair with our fine-tuned checkpoint can silently change
    # prediction comparability. A deliberate --base-model run applies no checkpoint, so
    # explicit download stays available there.
    model = load_model(device, allow_backbone_download=allow_backbone_download,
                       for_finetuned_checkpoint=not allow_base_model)

    if allow_base_model:
        return model, device, {
            "checkpoint": None, "sha256": None, "base_model": MODEL_ID,
            "WARNING": ("Ran the UN-FINE-TUNED base model (--base-model). Predictions are "
                        "not comparable to any DepthWizard baseline."),
        }

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, device, {
        "checkpoint": checkpoint.name, "sha256": checkpoint_sha, "sha256_verified": True,
        "base_model": MODEL_ID, "WARNING": None,
    }


def load_georaster(path: Path):
    """Return a geoio.GeoRaster for a GeoTIFF input, or None for anything else.

    Only `.tif/.tiff` can carry a geotransform, and geoio.load_geotiff is the one reader
    that resolves it: embedded GeoTIFF tags -> same-basename `.tfw` -> Vaihingen nominal
    fallback (flagged `georeferenced=False`). The metric branch checks
    `source.georeferenced is True`, so the nominal fallback is treated as "no transform" -
    a guessed georeference must never let a relative product pass as a metric one.

    A genuine read/parse failure (corrupt raster, permission error, malformed `.tfw`
    sidecar) is a caller error, not "not georeferenced" - it is raised as
    PipelineError -> exit 2, never silently collapsed into a relative-mode return.
    geoio.load_geotiff() (called here without strict=True) already returns cleanly with
    georeferenced=False for the legitimate "no embedded tags, no .tfw" case, so any
    exception it raises for a `.tif`/`.tiff` path means something is actually broken.
    """
    if path.suffix.lower() not in (".tif", ".tiff"):
        return None
    from geoio import load_geotiff                                  # noqa: PLC0415
    try:
        return load_geotiff(path)
    except Exception as exc:
        raise PipelineError(f"cannot read georaster from {path}: {exc}") from exc


# --- stages 2-4: tile / predict / blend --------------------------------------------
def build_relative_surface(rgb: np.ndarray, *, window: int, stride: int, model, device: str,
                           timings: dict):
    """Tile -> predict -> blend. Returns (surface, TilePlan, per-tile provenance)."""
    t0 = time.perf_counter()
    plan = plan_windows(rgb.shape[0], rgb.shape[1], window=window, stride=stride)
    timings["tile"] = time.perf_counter() - t0
    print(f"  tile     : {plan.n_windows} windows of {window}px, stride {stride}")

    t0 = time.perf_counter()
    depths, tile_provenance = predict_depth(rgb, plan, model=model, device=device)
    timings["predict"] = time.perf_counter() - t0
    print(f"  predict  : {len(depths)} windows in {timings['predict']:.1f}s")

    t0 = time.perf_counter()
    surface = blend_tiles(depths, plan)
    timings["blend"] = time.perf_counter() - t0
    if not np.isfinite(surface).all():
        raise PipelineError("blended surface contains non-finite values - refusing to write it")
    print(f"  blend    : {surface.shape} surface, "
          f"range [{surface.min():.3f}, {surface.max():.3f}] (unitless)")
    return surface, plan, tile_provenance


# --- stage 5: calibrate ------------------------------------------------------------
def calibrate(surface: np.ndarray, reference: Path | None, source) -> dict:
    """Decide the branch, running the DW-S1-10 pre-flight gate before any metric fit.

    Order matters (per #24): an explicitly supplied `--reference` is parsed and validated
    FIRST, so a broken reference is exit 2 even for an input that could never reach the
    metric branch anyway. Only then does eligibility decide the branch.

    The metric branch needs ALL of: a georeferenced GeoTIFF input
    (`source.georeferenced is True`, never the nominal fallback), a valid in-bounds
    reference, a passing pre-flight gate, and a finite scale/shift fit with a > 0.

    A failing gate is NOT an error: the run stays relative-only, records `gate_reason`,
    and exits 0 - a gate firing is the system working. An invalid fit after a passing gate
    IS an error (exit 2, nothing written) - a non-finite or non-positive scale means the
    surface and the reference elevations do not agree, and a believable-but-wrong metric
    DSM is worse than none.
    """
    georeferenced = bool(source is not None and getattr(source, "georeferenced", False))

    if reference is None:
        return {"branch": "relative", "reason": "no --reference supplied",
                "gate": None, "reference": None}

    # --reference was supplied: validate it before deciding anything else.
    points = load_reference_points(reference, surface.shape)

    if not georeferenced:
        return {"branch": "relative",
                "reason": ("--reference is valid but the input carries no real geotransform; "
                           "a metric DSM needs a georeferenced GeoTIFF"),
                "gate": None, "reference": str(reference)}

    gate = preflight_checks(points, surface)
    if not gate.passed:
        return {"branch": "relative",
                "reason": "calibration pre-flight gate declined the metric branch",
                "gate": gate.as_dict(), "gate_reason": gate.reason,
                "reference": str(reference)}

    a, b, fit_meta = fit_scale_shift_ransac(surface, points)
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        raise PipelineError(
            f"scale/shift fit is invalid (a={a}, b={b}) - refusing to write a metric DSM. "
            "A non-finite or non-positive scale means the relative surface and the "
            f"reference elevations disagree in sign or scale ({fit_meta['method']} fit on "
            f"{fit_meta['n_points']} points).")

    assert source is not None            # georeferenced is True -> source was loaded
    from geoio import CRS_UNDEFINED                                 # noqa: PLC0415
    epsg = int(source.epsg) if source.epsg is not None else int(CRS_UNDEFINED)
    dsm_metres = (a * surface + b).astype(np.float32)
    return {"branch": "metric",
            "reason": f"metric fit ok (a={a:.6g}, b={b:.6g}, {fit_meta['method']})",
            "gate": gate.as_dict(), "gate_reason": None, "reference": str(reference),
            "a": float(a), "b": float(b), "epsg": epsg, "fit": fit_meta,
            "dsm_metres": dsm_metres, "source": source,
            "crs_note": ("input's embedded ProjectedCSTypeGeoKey propagated verbatim"
                         if source.epsg is not None else
                         "input has no embedded CRS; horizontal position is geoio.CRS_UNDEFINED")}


def fit_scale_shift_ransac(surface: np.ndarray, points: np.ndarray):
    """Canonical sparse scale/shift, following scripts/07_calibrate_and_eval.py's lineage.

    X = relative surface sampled at the reference (row, col); y = reference elevation_m.
    RANSAC(LinearRegression, min_samples=max(2, n//2), random_state=42) - the exact
    estimator #24's canonical calibration uses. If RANSAC itself raises, fall back to the
    existing least-squares `metrics.fit_scale_shift`; the caller validates the resulting
    (a, b) either way, and an invalid fit is never a reason to silently try a third path.
    """
    rr = points[:, 0].astype(int)
    cc = points[:, 1].astype(int)
    x = np.asarray(surface, dtype=np.float64)[rr, cc]
    y = points[:, 2].astype(np.float64)
    n = int(len(y))

    try:
        from sklearn.linear_model import LinearRegression, RANSACRegressor   # noqa: PLC0415
        reg = RANSACRegressor(LinearRegression(), min_samples=max(2, n // 2), random_state=42)
        reg.fit(x.reshape(-1, 1), y)
        a = float(reg.estimator_.coef_[0])
        b = float(reg.estimator_.intercept_)
        return a, b, {"method": "ransac", "n_points": n,
                      "inliers": int(reg.inlier_mask_.sum()),
                      "a": a, "b": b}
    except Exception as exc:                                        # noqa: BLE001
        from metrics import fit_scale_shift                         # noqa: PLC0415
        a, b = fit_scale_shift(x, y)
        return a, b, {"method": f"ols_fallback ({type(exc).__name__})", "n_points": n,
                      "inliers": None, "a": float(a), "b": float(b)}


def _load_reference_text(path: Path) -> np.ndarray:
    """Parse a reference text file. #24 supports exactly two text layouts: whitespace-
    separated and comma-separated. Try whitespace first, then comma; both failing is a
    caller error (raised by the caller as PipelineError -> exit 2). No other format.
    """
    last_exc: Exception | None = None
    for delimiter in (None, ","):
        try:
            return np.loadtxt(path, delimiter=delimiter, ndmin=2)
        except Exception as exc:                                    # noqa: BLE001
            last_exc = exc
    raise ValueError(f"not whitespace- or comma-separated numeric rows ({last_exc})")


def load_reference_points(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Load and validate reference points as (N, 3) rows of (row, col, elevation_m).

    Accepts `.npy` or whitespace/comma-separated text. Every rejection below is a caller
    error, not a gate decision: missing file, unreadable, wrong shape, non-finite value,
    fractional pixel coordinate, or row/col outside the image -> PipelineError -> exit 2.
    Sub-pixel sampling is out of MVP scope, so a fractional row/col is refused rather than
    silently truncated.
    """
    path = Path(path)
    if not path.exists():
        raise PipelineError(f"reference file not found: {path}")
    try:
        raw = np.load(path) if path.suffix == ".npy" else _load_reference_text(path)
    except Exception as exc:
        raise PipelineError(f"cannot read reference points from {path}: {exc}") from exc

    pts = np.asarray(raw, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise PipelineError(
            f"reference points must be (N, >=3) as (row, col, elevation_m); got shape {pts.shape}")
    pts = pts[:, :3]
    if not np.isfinite(pts).all():
        raise PipelineError(
            "reference points contain non-finite (NaN/Inf) values in row, col or elevation")

    rc = pts[:, :2]
    frac = np.abs(rc - np.round(rc))
    if frac.max() > 1e-6:
        i = int(np.argmax(frac.max(axis=1)))
        raise PipelineError(
            f"reference point {i} has a fractional pixel coordinate (row={pts[i, 0]}, "
            f"col={pts[i, 1]}); row/col must be integer indices - sub-pixel sampling is "
            "out of MVP scope")

    h, w = int(image_shape[0]), int(image_shape[1])
    rr = np.round(pts[:, 0]).astype(int)
    cc = np.round(pts[:, 1]).astype(int)
    oob = (rr < 0) | (rr >= h) | (cc < 0) | (cc >= w)
    if oob.any():
        i = int(np.argmax(oob))
        raise PipelineError(
            f"reference point {i} at (row={rr[i]}, col={cc[i]}) is outside the "
            f"{h}x{w} image bounds")

    return np.column_stack([rr.astype(np.float64), cc.astype(np.float64), pts[:, 2]])


# --- outputs -----------------------------------------------------------------------
# Visualization multiplier applied to the mesh Z. 1.0 = faithful geometry (no vertical
# exaggeration); the runner leaves relief-exaggeration to the viewer. Recorded in the
# manifest so a .glb is never mistaken for a measurement.
MESH_Z_EXAGGERATION = 1.0


def write_preview(surface_norm: np.ndarray, out_path: Path) -> Path:
    """Normalized grayscale PNG. A look-at-it artifact, never an accuracy claim."""
    from PIL import Image                                            # noqa: PLC0415
    Image.fromarray((np.clip(surface_norm, 0, 1) * 255).astype(np.uint8), mode="L").save(out_path)
    return out_path


# --- stages 6-7: mesh / LOD / GeoTIFF ---------------------------------------------
def to_mesh(elev: np.ndarray, rgb: np.ndarray, gsd: float, out_path: Path) -> tuple:
    """Write a textured `.glb` from a height array + RGB. Returns (mesh, path).

    Thin wrapper over the frozen `src/mesh.heightfield_to_mesh` - the runner adds no
    geometry of its own. The mesh is a visualization artifact: Z is rebased to the tile
    minimum and Y is negated for the viewer, so it does not carry an absolute datum.

    #87 NaN/nodata guard: `heightfield_to_mesh()` is FROZEN v1 (see src/mesh.py's module
    docstring - a change there needs a v2 label and a note to Ashmita/Aishee, not a fix
    here), and it has no NaN check of its own: `Z = (elev - elev.min()) * z_exaggeration`
    silently propagates a NaN into vertex data (elev.min() itself returns NaN when any
    element is NaN) rather than failing loudly, producing degenerate/invisible geometry
    with no warning. Guarded HERE, at the call boundary, instead of inside the frozen
    function.

    Review catch (PR #93, Anmol): `dsm_metres` (the metric branch's elev) is finite but
    does not exclude `geoio.DSM_NODATA` (-9999.0) - a nodata sentinel that reaches here
    bypasses the NaN check above and silently produces invalid-but-finite geometry, even
    though the source GeoTIFF marks that value as nodata. Reuses the project's existing
    nodata tolerance (`geoio.dsm_nodata_mask()`), not a new one.
    """
    if not np.isfinite(elev).all():
        n_bad = int((~np.isfinite(elev)).sum())
        raise PipelineError(
            f"elevation array contains {n_bad} non-finite (NaN/Inf) value(s) - refusing to "
            "build a mesh from it rather than silently emitting degenerate geometry"
        )
    from geoio import DSM_NODATA, dsm_nodata_mask                    # noqa: PLC0415
    valid_mask = dsm_nodata_mask(elev, nodata_value=DSM_NODATA)       # True = valid pixel
    if not valid_mask.all():
        n_bad = int((~valid_mask).sum())
        raise PipelineError(
            f"elevation array contains {n_bad} nodata sentinel value(s) (within tolerance of "
            f"{DSM_NODATA}) - refusing to build a mesh from it rather than silently emitting "
            "invalid geometry"
        )
    from mesh import heightfield_to_mesh                            # noqa: PLC0415
    mesh = heightfield_to_mesh(elev, rgb, gsd=gsd, z_exaggeration=MESH_Z_EXAGGERATION)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    return mesh, out_path


def to_lod(mesh, out_path: Path, *, target_faces: int | None = None) -> tuple:
    """Write a decimated LOD `.glb`, texture/UV preserved (issue #12 / PR #17). Returns
    (lod_mesh, path). A mesh already at/below the target is exported unchanged."""
    from mesh import LOD_TARGET_FACES, make_lod                     # noqa: PLC0415
    lod = make_lod(mesh, target_faces if target_faces is not None else LOD_TARGET_FACES)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lod.export(str(out_path))
    return lod, out_path


def to_geotiff(dsm_metres: np.ndarray, source, epsg: int, out_path: Path) -> Path:
    """Write the metric DSM as a GeoTIFF on the input raster's own affine transform.

    `geoio.save_dsm_geotiff` already guards shape and refuses a guessed georeference
    (`require_georeferenced=True`); `epsg` is the input's embedded code or CRS_UNDEFINED,
    decided in `calibrate()`.
    """
    from geoio import save_dsm_geotiff                              # noqa: PLC0415
    return save_dsm_geotiff(Path(out_path), dsm_metres, source, epsg=epsg)


# --- stage 8: viewer_input_v1 sidecar (DW-WP2, IF-1, issues #26/#27) --------------
# Frozen on issue #26 (Anmol + Nirmalya, 2026-09-06): one interface for both branches,
# a small projection of run_manifest.json (not a duplicate), raw little-endian float32
# elevation payload (never a nested JSON array - that does not scale past a placeholder
# tile), and an explicit view-only/measurements-disabled failure contract that the
# viewer must show, never silently no-op.
#
# File-layout call NOT pinned down by the frozen text (flagged on #26 rather than
# guessed silently): the sidecar is written FLAT, next to its own mesh - a scene+branch
# prefix matching the existing .glb/.tif naming convention, in the SAME directory a
# viewer is pointed at via `?glb=<path>` - rather than a separate nested `<out>/viewer/`
# folder. Reasoning: relative and metric are pixel-aligned but numerically different
# arrays, so a single "primary branch" sidecar shared across both .glb files would let a
# viewer sample the wrong array depending on which mesh happens to be loaded - exactly
# the "confident wrong number" failure mode this contract exists to prevent. Flat,
# per-branch, glb-adjacent naming makes that mismatch structurally impossible: the
# viewer derives the sidecar path directly from whichever `.glb` URL it was given.
VIEWER_INPUT_SCHEMA = "viewer_input_v1"


def _elevation_stem(scene: str, branch: str) -> str:
    """The shared filename stem for one branch's mesh + sidecar pair.

    `{stem}.glb` / `{stem}_lod.glb` / `{stem}_surface_meta.json` / `{stem}_elevation.f32`
    all share this stem, so the viewer can derive the sidecar path from any of that
    branch's mesh URLs by stripping `.glb` and a trailing `_lod`.
    """
    return f"{scene}_{branch}"


def write_elevation_f32(elev: np.ndarray, out_path: Path) -> Path:
    """Write a (H, W) array as raw little-endian float32 bytes, row-major.

    Explicit `<f4` rather than a bare `.astype(np.float32).tobytes()` - the latter uses
    the host's native byte order, which is little-endian on every machine this project
    runs on today but is not a contract a browser's `DataView.getFloat32(offset, true)`
    should be trusted to match by coincidence. This is IF-1's replacement for the
    placeholder's `elev.tolist()`, which does not scale past a 64x64 tile (hundreds of MB
    of JSON on a real-resolution surface).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(elev, dtype="<f4")
    out_path.write_bytes(arr.tobytes(order="C"))
    return out_path


def write_surface_meta(
    *,
    out_dir: Path,
    scene: str,
    branch: str,
    elev: np.ndarray,
    mesh_glb: Path,
    lod_glb: Path,
    gsd: float | None,
    pseudo_gsd: float | None,
    z_exaggeration: float,
    crs: int | None,
    normalization: dict | None,
    nodata: float | None,
    is_placeholder: bool = False,
    note: str | None = None,
) -> tuple[Path, Path]:
    """Write one branch's `{scene}_{branch}_surface_meta.json` + `..._elevation.f32`.

    Returns (surface_meta_path, elevation_f32_path). Field set per the frozen
    `viewer_input_v1` (#26, 2026-09-06): schema/branch/H/W/dtype/byte_order/
    elevation_payload/units/gsd-or-pseudo_gsd/z_exaggeration/crs/mesh+LOD references/
    run_manifest reference/is_placeholder/normalization-where-applicable/
    nodata-where-applicable/explicit mesh-frame convention/optional note.

    Relative vs metric are never conflated: relative always has crs=None, gsd=None,
    nodata=None (a relative product carries no georeference or nodata convention at
    all - see relative_dsm.FORBIDDEN_SIDECAR_FIELDS); metric always has
    pseudo_gsd=None. Exactly one of (gsd, pseudo_gsd) is non-None.
    """
    if branch not in ("relative", "metric"):
        raise ValueError(f"branch must be 'relative' or 'metric', got {branch!r}")
    if (gsd is None) == (pseudo_gsd is None):
        raise ValueError(
            f"exactly one of gsd/pseudo_gsd must be set for branch={branch!r}; "
            f"got gsd={gsd!r}, pseudo_gsd={pseudo_gsd!r}")
    units = "unitless" if branch == "relative" else "metres"

    elev = np.asarray(elev, dtype=np.float32)
    h, w = int(elev.shape[0]), int(elev.shape[1])
    stem = _elevation_stem(scene, branch)
    surface_meta_path = out_dir / f"{stem}_surface_meta.json"
    elevation_path = out_dir / f"{stem}_elevation.f32"
    write_elevation_f32(elev, elevation_path)

    meta = {
        "schema": VIEWER_INPUT_SCHEMA,
        "branch": branch,
        "H": h,
        "W": w,
        "dtype": "float32",
        "byte_order": "little",
        "elevation_payload": elevation_path.name,
        "units": units,
        "gsd": float(gsd) if gsd is not None else None,
        "pseudo_gsd": float(pseudo_gsd) if pseudo_gsd is not None else None,
        "z_exaggeration": float(z_exaggeration),
        "crs": int(crs) if crs is not None else None,
        "mesh_glb": Path(mesh_glb).name,
        "lod_glb": Path(lod_glb).name,
        "run_manifest": "run_manifest.json",
        "is_placeholder": bool(is_placeholder),
        "normalization": normalization,
        "nodata": float(nodata) if nodata is not None else None,
        "mesh_frame": {
            "x": "col * gsd",
            "y": "-row * gsd",
            "z": "(elevation - elevation_min) * z_exaggeration",
            "elevation_min": float(elev.min()),
            "note": ("gsd here means whichever of gsd/pseudo_gsd is non-null above - the "
                     "producer mesh frame frozen in src/mesh.heightfield_to_mesh()"),
        },
        "note": note,
    }
    surface_meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return surface_meta_path, elevation_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Turn one optical image into a labelled height surface (DW-S1-07).")
    ap.add_argument("image", help="input image (JPG/PNG, or GeoTIFF for the metric branch)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="vertical elevation control points (.npy or text, (N,>=3) as "
                         "row,col,elevation_m); required for the metric branch")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "run",
                    help="output directory (default: outputs/run)")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--depth-sign", type=int, choices=(1, -1), default=1,
                    help="+1 nadir (default); -1 for oblique/handheld input")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(os.environ.get("DW_CHECKPOINT")
                                 or ROOT / "checkpoints" / DEFAULT_CHECKPOINT),
                    help=f"fine-tuned weights (default: {DEFAULT_CHECKPOINT})")
    ap.add_argument("--base-model", action="store_true",
                    help="run the un-fine-tuned base model instead of a checkpoint; "
                         "recorded loudly in run_manifest.json")
    ap.add_argument("--allow-backbone-download", action="store_true",
                    help="permit an online HuggingFace resolve/download of the base "
                         "backbone. Default: OFF - the backbone loads local-cache-only "
                         "and a missing local backbone fails closed with instructions.")
    args = ap.parse_args(argv)

    started = datetime.now(timezone.utc)
    timings: dict = {}
    image_path = Path(args.image)

    try:
        t0 = time.perf_counter()
        rgb = validate_input(image_path)
        timings["validate"] = time.perf_counter() - t0
        source = load_georaster(image_path)
        georef = bool(source is not None and source.georeferenced)
        print(f"run_pipeline: {image_path.name}  {rgb.shape[1]}x{rgb.shape[0]}  "
              f"georeferenced={georef}")

        t0 = time.perf_counter()
        model, device, ckpt_record = load_predictor(
            args.checkpoint, allow_base_model=args.base_model,
            allow_backbone_download=args.allow_backbone_download)
        timings["load_model"] = time.perf_counter() - t0
        print(f"  model    : {ckpt_record['checkpoint'] or ckpt_record['base_model']} on {device}"
              + (f"  sha {ckpt_record['sha256'][:12]}..." if ckpt_record["sha256"] else ""))
        if ckpt_record["WARNING"]:
            print(f"  WARNING  : {ckpt_record['WARNING']}")

        surface, plan, tile_provenance = build_relative_surface(
            rgb, window=args.window, stride=args.stride, model=model, device=device,
            timings=timings)

        # calibrate() does all reference parsing / validation / gating / fitting up front,
        # so by the time it returns, a broken reference or an invalid fit has already
        # raised (exit 2) and nothing has been written.
        t0 = time.perf_counter()
        cal = calibrate(surface, args.reference, source)
        timings["calibrate"] = time.perf_counter() - t0
        print(f"  calibrate: branch={cal['branch']} ({cal['reason']})")
        if cal.get("gate_reason"):
            print(f"  GATE     : metric calibration declined - {cal['gate_reason']}")
            print("             emitting the relative DSM only; this is a safety decision, "
                  "not a failure")

        norm, norm_meta = normalize_relative(surface, depth_sign=args.depth_sign)
        out_dir = Path(args.out)
        scene = image_path.stem

        # --- relative products: always, unchanged from Day 2 -----------------------
        npy, sidecar = save_relative_dsm(
            out_dir, scene,
            RelativeDSM(array=norm, source_image=image_path.name,
                        depth_sign=args.depth_sign, norm=norm_meta))
        preview = write_preview(norm, out_dir / f"{scene}_preview.png")
        outputs: dict = {"height_relative": npy, "sidecar": sidecar, "preview": preview}
        mesh_meta: dict = {"z_exaggeration": MESH_Z_EXAGGERATION}

        # --- stage 6: relative mesh + LOD (always) --------------------------------
        t0 = time.perf_counter()
        rel_mesh, rel_glb = to_mesh(norm, rgb, PSEUDO_GSD, out_dir / f"{scene}_relative.glb")
        rel_lod, rel_lod_glb = to_lod(rel_mesh, out_dir / f"{scene}_relative_lod.glb")
        outputs["relative_glb"] = rel_glb
        outputs["relative_lod_glb"] = rel_lod_glb
        mesh_meta["relative"] = {
            "gsd": PSEUDO_GSD,
            "gsd_note": "pseudo-GSD: 1 mesh unit per pixel, not a real map/sensor scale",
            "faces": int(len(rel_mesh.faces)),
            "lod_faces": int(len(rel_lod.faces)),
            "lod_decimated": len(rel_lod.faces) < len(rel_mesh.faces),
        }
        print(f"  mesh     : {scene}_relative.glb ({len(rel_mesh.faces):,} faces), "
              f"LOD {len(rel_lod.faces):,} faces")

        rel_surface_meta, rel_elevation_f32 = write_surface_meta(
            out_dir=out_dir, scene=scene, branch="relative", elev=norm,
            mesh_glb=rel_glb, lod_glb=rel_lod_glb,
            gsd=None, pseudo_gsd=PSEUDO_GSD, z_exaggeration=MESH_Z_EXAGGERATION,
            crs=None, normalization=norm_meta, nodata=None,
            note="unitless relative surface; no metres, no datum, no map location")
        outputs["relative_surface_meta"] = rel_surface_meta
        outputs["relative_elevation_f32"] = rel_elevation_f32

        # --- stage 7: metric branch (GeoTIFF + mesh + LOD) ----------------------
        calibration = {"a": cal.get("a"), "b": cal.get("b"), "fit": cal.get("fit"),
                       "reference": cal.get("reference")}
        if cal["branch"] == "metric":
            source_gr = cal["source"]
            metric_tif = to_geotiff(cal["dsm_metres"], source_gr, cal["epsg"],
                                    out_dir / f"{scene}_metric.tif")
            met_mesh, metric_glb = to_mesh(cal["dsm_metres"], rgb, source_gr.gsd,
                                           out_dir / f"{scene}_metric.glb")
            met_lod, metric_lod_glb = to_lod(met_mesh, out_dir / f"{scene}_metric_lod.glb")
            outputs["metric_tif"] = metric_tif
            outputs["metric_glb"] = metric_glb
            outputs["metric_lod_glb"] = metric_lod_glb
            mesh_meta["metric"] = {
                "gsd": float(source_gr.gsd),
                "faces": int(len(met_mesh.faces)),
                "lod_faces": int(len(met_lod.faces)),
                "lod_decimated": len(met_lod.faces) < len(met_mesh.faces),
            }
            dsm = cal["dsm_metres"]
            calibration.update({
                "epsg": cal["epsg"], "crs_note": cal["crs_note"],
                "dsm_metres_stats": {"min": float(dsm.min()), "max": float(dsm.max()),
                                     "mean": float(dsm.mean())},
            })
            print(f"  metric   : {scene}_metric.tif (epsg {cal['epsg']}), "
                  f"{scene}_metric.glb, LOD {len(met_lod.faces):,} faces")

            from geoio import DSM_NODATA                             # noqa: PLC0415
            met_surface_meta, met_elevation_f32 = write_surface_meta(
                out_dir=out_dir, scene=scene, branch="metric", elev=dsm,
                mesh_glb=metric_glb, lod_glb=metric_lod_glb,
                gsd=source_gr.gsd, pseudo_gsd=None, z_exaggeration=MESH_Z_EXAGGERATION,
                crs=cal["epsg"], normalization=None, nodata=DSM_NODATA,
                note=(f"metric elevation in metres from a gated scale/shift fit "
                      f"({cal['fit']['method']}); {cal['crs_note']}; nodata is the "
                      "project convention, this dense array has no masked pixels"))
            outputs["metric_surface_meta"] = met_surface_meta
            outputs["metric_elevation_f32"] = met_elevation_f32
        timings["outputs"] = time.perf_counter() - t0

    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except (RelativeDSMError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    manifest = {
        "schema": "run_pipeline/v0",
        "run_started_utc": started.isoformat(),
        "duration_s": round(sum(timings.values()), 3),
        "input": {"path": str(image_path), "name": image_path.name,
                  "sha256": _sha256(image_path),
                  "shape": [int(rgb.shape[0]), int(rgb.shape[1])],
                  "georeferenced": georef},
        "model": ckpt_record,
        "device": device,
        "branch": cal["branch"],
        "branch_reason": cal["reason"],
        "gate_reason": cal.get("gate_reason"),
        "preflight": cal.get("gate"),
        # metric branch only: the input's embedded ProjectedCSTypeGeoKey if it has one,
        # else geoio.CRS_UNDEFINED. Relative products carry no CRS.
        "crs": cal.get("epsg"),
        # #24 locked runner->manifest contract: crs / a / b / reference / preflight are
        # top-level. On the relative branch a and b are None; the nested `calibration`
        # object mirrors them plus the fit provenance.
        "a": cal.get("a"),
        "b": cal.get("b"),
        "calibration": calibration,
        "reference": cal.get("reference"),
        "tiling": plan.as_manifest(),
        "tiles": tile_provenance,
        "stage_timings_s": {k: round(v, 3) for k, v in timings.items()},
        "surface_stats": {"shape": list(surface.shape),
                          "min": float(surface.min()), "max": float(surface.max()),
                          "mean": float(surface.mean())},
        "normalization": norm_meta,
        "depth_sign": args.depth_sign,
        "pseudo_gsd": PSEUDO_GSD,
        "mesh": mesh_meta,
        "outputs": {k: str(Path(v).relative_to(out_dir)) for k, v in outputs.items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "IMPORTANT": (
            "The relative height surface is UNITLESS and uncalibrated - no metres, no "
            "datum, no map location, no RMSE. On the metric branch, the vertical scale "
            "comes from a sparse gated scale/shift fit (not field-validated accuracy), the "
            "GeoTIFF's horizontal transform is the input raster's own, and any .glb is a "
            "visualization mesh (Z rebased to the tile minimum, Y negated) - not a "
            "measurement product."),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    written = ", ".join(sorted(Path(v).name for v in outputs.values()))
    print(f"  wrote    : {written}, run_manifest.json")
    print(f"  total    : {manifest['duration_s']}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
