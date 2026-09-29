"""Guard against EPSG:32633 being silently reintroduced - issue #14.

32633 (WGS 84 / UTM zone 33N) came from the Vaihingen dataset's own dsm.vrt and
propagated through Phase 1/2 before anyone checked it against the coordinates. It is
wrong: easting 496794.825 implies ~8.96E under UTM 32N (Vaihingen), versus ~14.96E under
33N - 6 degrees, roughly 440 km east at this latitude. The datum/authority is a separate
question and remains unproven, so exports assert no authority code (geoio.CRS_UNDEFINED,
32767) rather than substituting 32632 or 25832.

This test exists so the value cannot creep back into a producer unnoticed. It scans
executable code only: prose, historical fixtures and frozen evidence deliberately still
record 32633, because that is what those runs actually wrote and rewriting them would be
falsifying the record. Those carry errata instead.

Run either way:
    python tests/test_no_hardcoded_crs.py
    pytest tests/test_no_hardcoded_crs.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Executable code that can write a CRS. Docs, fixtures and outputs are excluded on
# purpose - see the module docstring.
SCANNED_DIRS = ("src", "scripts", "PHASE2_RED_FLAGS/scripts", "tests")

BANNED = {
    32633: ("wrong for Vaihingen: coordinates are consistent with UTM 32N geometry, "
            "not 33N. Use geoio.CRS_UNDEFINED unless the authority is actually proven."),
}


# This file necessarily names the banned value in BANNED above, so it excludes itself.
SELF = Path(__file__).resolve()


def _python_files():
    for d in SCANNED_DIRS:
        for f in sorted((ROOT / d).rglob("*.py")):
            if f.resolve() != SELF:
                yield f


def test_no_banned_crs_literal_in_executable_code():
    """Numeric literals only, via AST - so the explanatory comments and docstrings that
    legitimately name 32633 do not trip the guard, but `EPSG = 32633` would."""
    offenders = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:                       # pragma: no cover
            raise AssertionError(f"could not parse {path}: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in BANNED:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} -> "
                                 f"{node.value} ({BANNED[node.value]})")
    assert not offenders, "banned CRS literal reintroduced:\n  " + "\n  ".join(offenders)


def test_producers_use_crs_undefined():
    """The two calibrated-DSM producers and the K2 exporter must route through the shared
    constant rather than any inline code, so there is exactly one place to change."""
    required = [
        Path("scripts/07_calibrate_and_eval.py"),
        Path("PHASE2_RED_FLAGS/scripts/K1_tiling_stitch.py"),
        Path("PHASE2_RED_FLAGS/scripts/K2_georeferencing_export.py"),
    ]
    missing = [str(p) for p in required
               if "CRS_UNDEFINED" not in (ROOT / p).read_text()]
    assert not missing, f"producer(s) not using geoio.CRS_UNDEFINED: {missing}"


def test_crs_undefined_is_the_geotiff_user_defined_code():
    sys.path.insert(0, str(ROOT / "src"))
    from geoio import CRS_UNDEFINED
    assert CRS_UNDEFINED == 32767, (
        "32767 is the GeoTIFF spec's 'user-defined' code; changing it would silently turn "
        "an honest non-assertion into a claim about some real CRS")


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
