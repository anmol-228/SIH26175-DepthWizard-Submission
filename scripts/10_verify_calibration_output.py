"""Issue #5 (DW-S1-02): prove a downstream consumer can read the calibration
stage's saved output without manual patching.

Two independent checks per raster:
  1. K2-style georeferencing round-trip -- reopen the file with tifffile and
     assert pixel size, origin, shape, EPSG GeoKey and the nodata tag all
     survived the write. This is the check NIRMALYA.md requires of any
     export-path change.
  2. Real downstream consumption -- feed the reopened array into
     heightfield_to_mesh() from K3_mesh_pipeline.py, the actual PIPELINE -> 3D
     consumer, sourcing gsd from the raster rather than hardcoding 0.09.
     Skipped (not failed) when trimesh is absent, since trimesh is not in
     requirements.txt.

Reads the export manifests written by scripts/07_calibrate_and_eval.py and
PHASE2_RED_FLAGS/scripts/K1_tiling_stitch.py. Writes its own report; changes
nothing.
"""
import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import tifffile
from geoio import load_geotiff, read_geotiff_tags, dsm_nodata_mask, DSM_NODATA

METRICS_DIR = ROOT / "outputs" / "metrics"
P2_METRICS_DIR = ROOT / "PHASE2_RED_FLAGS" / "outputs" / "metrics"


def collect_records():
    """Every calibrated raster any producer has declared, from its own manifest."""
    records = []
    for mf in sorted(METRICS_DIR.glob("*_dsm_export_manifest.json")):
        for r in json.loads(mf.read_text()).get("rasters", []):
            records.append((mf.name, r))
    # glob, so a K1_OUT_NAME re-run is picked up as well as the default filename
    for k1 in sorted(P2_METRICS_DIR.glob("K1_tiling_stitch*.json")):
        exp = json.loads(k1.read_text()).get("calibrated_dsm_export")
        if exp and exp.get("written"):
            records.append((k1.name, exp))
    return records


def check_roundtrip(rec):
    """Assert the georeference declared in the manifest is what the file carries."""
    path = ROOT / rec["path"]
    if not path.exists():
        return {"ok": False, "error": f"declared raster missing on disk: {rec['path']}"}

    tags = read_geotiff_tags(path)
    raster = load_geotiff(path)
    px, py = rec["pixel_size"]
    ox, oy = rec["origin"]
    checks = {
        "pixel_size_preserved": abs(raster.pixel_size_x - px) < 1e-9
                                and abs(raster.pixel_size_y - py) < 1e-9,
        "origin_preserved": abs(raster.origin_x - ox) < 1e-6
                            and abs(raster.origin_y - oy) < 1e-6,
        "shape_preserved": list(raster.array.shape) == rec["shape"],
        # Round-trip only: asserts the code we WROTE came back, not that it is the
# correct CRS for this scene. Vaihingen exports use CRS_UNDEFINED (32767)
# precisely because no authority code can be defended - see geoio.CRS_UNDEFINED.
        "epsg_geokey_roundtripped": raster.epsg == rec["epsg"],
        "nodata_tag_preserved": tags.get("nodata") == rec["nodata"],
        "dtype_float32": raster.array.dtype == np.float32,
        # georeference came from the file, not from load_geotiff's nominal fallback
        "georeference_is_real": raster.georeferenced is True,
        "valid_pixel_count_matches": int(dsm_nodata_mask(raster.array, rec["nodata"]).sum())
                                     == rec["n_valid_px"],
    }
    valid = dsm_nodata_mask(raster.array, rec["nodata"])
    stats = {}
    if valid.any():
        v = raster.array[valid]
        stats = {"min_m": float(v.min()), "max_m": float(v.max()), "mean_m": float(v.mean())}
        # a metric DSM in metres should not be silently unitless or wildly out of range
        checks["values_look_metric"] = bool(-500.0 < v.min() and v.max() < 9000.0)
    return {"ok": all(checks.values()), "checks": checks, "stats": stats,
            "gsd_from_raster": raster.gsd}


def check_consumer(rec):
    """Feed the saved raster into the real PIPELINE -> 3D consumer."""
    try:
        import trimesh  # noqa: F401
    except ImportError:
        return {"skipped": "trimesh not installed (absent from requirements.txt)"}
    sys.path.insert(0, str(ROOT / "PHASE2_RED_FLAGS" / "scripts"))
    from K3_mesh_pipeline import heightfield_to_mesh

    raster = load_geotiff(ROOT / rec["path"])
    elev = raster.array.astype(np.float32)
    valid = dsm_nodata_mask(elev, rec["nodata"])
    if not valid.any():
        return {"ok": False, "error": "raster is entirely nodata"}
    # nodata would otherwise drag the mesh floor to -9999 via the elev.min() rebase
    elev = np.where(valid, elev, float(elev[valid].min()))
    rgb = np.zeros(elev.shape + (3,), dtype=np.uint8)   # geometry check only
    # gsd read from the raster, not hardcoded as K3 currently does
    mesh = heightfield_to_mesh(elev, rgb, gsd=raster.gsd, downsample=8)
    return {"ok": bool(len(mesh.faces) > 0 and np.isfinite(mesh.vertices).all()),
            "n_faces": int(len(mesh.faces)), "n_vertices": int(len(mesh.vertices)),
            "gsd_used": raster.gsd,
            "z_range_m": [float(mesh.vertices[:, 2].min()), float(mesh.vertices[:, 2].max())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mesh", action="store_true", help="skip the trimesh consumer check")
    args = ap.parse_args()

    records = collect_records()
    if not records:
        print("No calibrated-DSM export manifests found. Run scripts/07_calibrate_and_eval.py "
              "or PHASE2_RED_FLAGS/scripts/K1_tiling_stitch.py first.")
        return 1

    report, failures = [], 0
    for source, rec in records:
        print(f"\n{rec['path']}  [{rec.get('calibration', '?')}, from {source}]")
        rt = check_roundtrip(rec)
        if "error" in rt:
            print(f"  ERROR {rt['error']}"); failures += 1
        else:
            for k, v in rt["checks"].items():
                print(f"  {'PASS' if v else 'FAIL':4}  {k}")
            if rt["stats"]:
                print(f"        elevation {rt['stats']['min_m']:.2f}..{rt['stats']['max_m']:.2f} m")
            failures += (not rt["ok"])
        cons = {} if args.no_mesh else check_consumer(rec)
        if cons.get("skipped"):
            print(f"  SKIP  consumer mesh check: {cons['skipped']}")
        elif cons:
            print(f"  {'PASS' if cons.get('ok') else 'FAIL':4}  "
                  f"heightfield_to_mesh consumed it ({cons.get('n_faces', 0):,} faces, "
                  f"gsd={cons.get('gsd_used')})")
            failures += (not cons.get("ok"))
        report.append({"source_manifest": source, "record": rec,
                       "roundtrip": rt, "consumer": cons})

    out = METRICS_DIR / "calibration_output_consumer_check.json"
    out.write_text(json.dumps({"n_rasters": len(records), "n_failures": failures,
                               "results": report}, indent=2))
    print(f"\n{len(records) - failures}/{len(records)} raster(s) passed. Report -> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
