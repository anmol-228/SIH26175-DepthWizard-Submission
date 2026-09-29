"""Tests for the relative (non-georeferenced) DSM branch - handbook Part IV, issue #15.

Mirrors tests/test_geoio_roundtrip.py: no checkpoint, no Vaihingen data, no network, so it
runs on any clone. The depth model is never loaded - normalization, packaging and the sign
convention are all testable without it, which keeps these unit tests rather than a GPU job.

Run either way:
    python tests/test_relative_dsm.py     # plain, no pytest needed
    pytest tests/test_relative_dsm.py     # if pytest is installed
"""
import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relative_dsm import (  # noqa: E402
    FORBIDDEN_SIDECAR_FIELDS, PRODUCT, PSEUDO_GSD, SCHEMA_VERSION, RelativeDSM,
    RelativeDSMError, load_plain_image, normalize_relative, save_relative_dsm,
)


def _ramp(h=40, w=60, tall_at_top=True):
    """Synthetic depth with a known-tall region, for the sign test."""
    d = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    return d if tall_at_top else d[::-1].copy()


def test_relative_module_never_imports_geoio():
    """The discriminator is enforced by keeping the two writers apart, so assert it
    structurally. Parses the AST rather than grepping, because the module docstring
    deliberately mentions geoio to explain why it is absent."""
    tree = ast.parse((ROOT / "src" / "relative_dsm.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("geoio" in m for m in imported), f"relative path imported geoio: {imported}"


def test_load_plain_image_handles_grayscale_and_rgba():
    """The branch exists for arbitrary consumer images, not well-formed RGB only."""
    with tempfile.TemporaryDirectory() as d:
        for mode, name in (("L", "gray.png"), ("RGBA", "alpha.png"), ("RGB", "rgb.png")):
            p = Path(d) / name
            Image.new(mode, (32, 24)).save(p)
            rgb = load_plain_image(p)
            assert rgb.shape == (24, 32, 3), f"{mode} -> {rgb.shape}"
            assert rgb.dtype == np.uint8


def test_load_plain_image_rejects_missing_and_degenerate():
    with tempfile.TemporaryDirectory() as d:
        try:
            load_plain_image(Path(d) / "nope.png")
        except RelativeDSMError:
            pass
        else:
            raise AssertionError("missing file was accepted")
        tiny = Path(d) / "tiny.png"
        Image.new("RGB", (1, 1)).save(tiny)
        try:
            load_plain_image(tiny)
        except RelativeDSMError:
            pass
        else:
            raise AssertionError("1x1 image was accepted")


def test_normalization_range_and_robustness_to_outliers():
    """Percentile clipping, so a single hot pixel cannot flatten the scene."""
    d = _ramp()
    d[0, 0] = 1e6
    out, meta = normalize_relative(d)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0, "must normalize to [0,1]"
    assert not meta["degenerate"]
    body = out[1:, :]
    assert float(body.max() - body.min()) > 0.5, "outlier flattened the real terrain"


def test_normalization_handles_flat_and_nonfinite_input():
    flat, meta = normalize_relative(np.full((8, 8), 3.0, np.float32))
    assert meta["degenerate"] and np.all(flat == 0.0), "flat input must not divide by ~0"
    assert np.isfinite(flat).all()

    d = _ramp(8, 8)
    d[0, 0] = np.nan
    out, _ = normalize_relative(d)
    assert np.isfinite(out).all(), "NaN must not propagate into the height field"


def test_depth_sign_is_explicit_and_inverts_ordering():
    """The trap this branch had to dodge. scripts/05_zeroshot.py's comment claimed DA2's
    sign was inverted for nadir imagery; it is not (measured corr +0.73/+0.69/+0.80, fitted
    a=+3.898), so +1 is identity. Oblique input genuinely does invert, so the sign is a
    parameter and both directions are pinned here."""
    d = _ramp(tall_at_top=True)
    up, _ = normalize_relative(d, depth_sign=1)
    assert up[0, 0] > up[-1, 0], "depth_sign=+1 must preserve ordering (nadir: higher = taller)"

    down, _ = normalize_relative(d, depth_sign=-1)
    assert down[0, 0] < down[-1, 0], "depth_sign=-1 must invert ordering (oblique case)"

    for bad in (0, 2, "up", None):
        try:
            normalize_relative(d, depth_sign=bad)
        except RelativeDSMError:
            pass
        else:
            raise AssertionError(f"depth_sign={bad!r} was accepted")


def test_sidecar_carries_discriminator_and_no_georeference():
    """A relative product must be impossible to mistake for a metric one."""
    arr, norm = normalize_relative(_ramp())
    meta = RelativeDSM(array=arr, source_image="synthetic.png",
                       depth_sign=1, norm=norm).sidecar()
    assert meta["product"] == PRODUCT
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["units"] == "UNITLESS"
    leaked = sorted(f for f in FORBIDDEN_SIDECAR_FIELDS if f in meta)
    assert not leaked, f"sidecar leaked georeferencing fields: {leaked}"


def test_save_writes_npy_plus_sidecar_and_refuses_georeference():
    arr, norm = normalize_relative(_ramp())
    d = RelativeDSM(array=arr, source_image="synthetic.png", depth_sign=1, norm=norm)
    with tempfile.TemporaryDirectory() as tmp:
        npy, side = save_relative_dsm(tmp, "scene01", d)
        assert npy.name == "relative_dsm_scene01.npy" and side.exists()
        back = np.load(npy)
        assert back.dtype == np.float32 and np.array_equal(back, arr)
        meta = json.loads(side.read_text())
        assert meta["product"] == PRODUCT
        assert not any(f in meta for f in FORBIDDEN_SIDECAR_FIELDS)

        class Sneaky(RelativeDSM):
            def sidecar(self):
                m = super().sidecar()
                m["epsg"] = 32632
                return m

        try:
            save_relative_dsm(tmp, "bad", Sneaky(array=arr, source_image="x",
                                                 depth_sign=1, norm=norm))
        except RelativeDSMError:
            pass
        else:
            raise AssertionError("writer accepted a sidecar carrying epsg")


def test_downstream_mesh_consumes_it_with_the_pseudo_gsd():
    """Same consumer the metric branch feeds, minus any georeference."""
    try:
        sys.path.insert(0, str(ROOT / "PHASE2_RED_FLAGS" / "scripts"))
        from K3_mesh_pipeline import heightfield_to_mesh
    except ImportError:
        return  # trimesh absent; every assertion above still ran
    arr, _ = normalize_relative(_ramp(64, 64))
    rgb = np.zeros(arr.shape + (3,), np.uint8)
    mesh = heightfield_to_mesh(arr, rgb, gsd=PSEUDO_GSD, downsample=2)
    assert len(mesh.faces) > 0 and np.isfinite(mesh.vertices).all()
    span = float(mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min())
    assert 0 < span <= arr.shape[1] * PSEUDO_GSD, "one mesh unit per pixel"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:              # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
