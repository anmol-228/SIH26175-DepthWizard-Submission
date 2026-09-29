"""PR #99 (issue #98) reproducibility artifact for the real Vaihingen N=100 controls.

Anmol's PR #99 review: "the new MIN_NORMALIZED_SPREAD threshold is not yet fully
reproducible from committed evidence for the real Vaihingen N=100 controls." This
script produces exactly that evidence: per scene/seed, the raw tile shape, the seed,
the actual sampled (row, col) reference points, and the resulting normalized_spread
value under the CURRENT (PR #99, median-based) formula in src/preflight.py.

This is READ-ONLY validation evidence for src/preflight.py's already-fixed
normalized_spread() - it does not modify production code, and it does not itself
decide anything; a reviewer can independently recompute normalized_spread() from the
recorded (row, col) points and shape and confirm the recorded value matches, or rerun
this script from a clean checkout to reproduce the same points from the same seeds.

Scope: Vaihingen controls only (real DSM data available in this repo). Mount Adams
is NOT included here - no real Mount Adams DEM data exists on this machine (see
10_DOCUMENTATION/evidence/DW_ISSUE75_PRECOMPUTED_GOLDEN_PATH/SUMMARY.md for the same
disclosed gap in a different context). Track B's own DECISION.json/RESULTS.json
already carry Mount Adams' real spread values under the OLD formula; this artifact
does not attempt to reproduce those under the new formula without the underlying data.

REPRODUCE

    python PHASE2_RED_FLAGS/scripts/E_10_track_b_spread_reproducibility_pr99.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

P2ROOT = Path(__file__).resolve().parents[1]
ROOT = P2ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from preflight import MIN_NORMALIZED_SPREAD, normalized_spread  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
N = 100
SEED_BASE = 7000  # identical stream to #73/#91 Track A and Track B's E_09

OUT_DIR = ROOT / "10_DOCUMENTATION" / "evidence" / "DW_ISSUE91_TRACK_B_GATE_VALIDATION_N100"
OUT_PATH = OUT_DIR / "REPRODUCIBILITY_PR99_MEDIAN_SPREAD.json"
THIS_SCRIPT = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_hash(path: Path) -> str:
    """The blob hash git itself would record for this exact content -- reproducible by
    `git hash-object <path>` independent of line-ending/checkout differences (the
    concern Anmol's control-tower pass flagged for the sibling E_08/E_09 scripts)."""
    return subprocess.check_output(
        ["git", "hash-object", str(path)], cwd=ROOT, text=True
    ).strip()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:                                        # noqa: BLE001
        return f"unknown ({exc})"


def git_provenance() -> dict:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    porcelain = git("status", "--porcelain=v1", "--untracked-files=all")
    lines = [l for l in porcelain.splitlines() if l.strip()]
    return {"branch": branch, "head": head,
            "tree_state": "clean" if not lines else "dirty",
            "status_lines": lines}


def sample_references(h: int, w: int, n: int, seed: int):
    """Identical sampling to #73/#91 Track A/B's sample_references."""
    rng = np.random.default_rng(SEED_BASE + seed)
    rr = rng.integers(0, h, size=n)
    cc = rng.integers(0, w, size=n)
    return rr, cc


def main():
    if OUT_PATH.exists():
        raise RuntimeError(f"Refusing to run: {OUT_PATH} already exists.")

    start_time_utc = datetime.now(timezone.utc).isoformat()
    rows = []

    va_dsm_dir = ROOT / "data" / "raw" / "Vaihingen" / "dsm"
    for area in ["6", "8", "34", "38"]:
        dsm = tifffile.imread(va_dsm_dir / f"dsm_09cm_matching_area{area}.tif")
        h, w = dsm.shape[:2]
        for seed in SEEDS:
            rr, cc = sample_references(h, w, N, seed)
            pts_rc = np.stack([rr, cc], axis=1)
            spread = normalized_spread(pts_rc.astype(np.float64), (h, w))
            rows.append({
                "scene": f"vaihingen_area{area}",
                "raw_tile_shape_hw": [int(h), int(w)],
                "seed": seed,
                "seed_stream": SEED_BASE + seed,
                "n": N,
                "sampled_row_col_points": pts_rc.tolist(),
                "normalized_spread_current_formula": round(float(spread), 6),
                "min_normalized_spread_threshold": MIN_NORMALIZED_SPREAD,
                "spread_gate_would_pass": bool(spread >= MIN_NORMALIZED_SPREAD),
            })
            print(f"  vaihingen_area{area} seed={seed}: spread={spread:.4f} "
                  f"(floor {MIN_NORMALIZED_SPREAD}) -> "
                  f"{'PASS' if spread >= MIN_NORMALIZED_SPREAD else 'FAIL'}")

    end_time_utc = datetime.now(timezone.utc).isoformat()

    report = {
        "schema": "dw-fs-pr99-track-b-spread-reproducibility/v1",
        "purpose": ("Reproducibility artifact requested in PR #99 review: real Vaihingen "
                    "N=100 sampled points and the resulting normalized_spread value under "
                    "the CURRENT (PR #99, median-based) formula, so the re-derived "
                    "MIN_NORMALIZED_SPREAD threshold can be independently recomputed."),
        "pr_url": "internal tracker item 99",
        "issue_url": "internal tracker item 98",
        "scope_note": ("Vaihingen controls only -- no real Mount Adams DEM data exists on "
                       "this machine. See DW_ISSUE91_TRACK_B_GATE_VALIDATION_N100/RESULTS.json "
                       "for Mount Adams' real spread values under the OLD (pre-#99) formula."),
        "run_start_utc": start_time_utc,
        "run_end_utc": end_time_utc,
        "git": git_provenance(),
        "generator_script": {
            "path": str(THIS_SCRIPT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(THIS_SCRIPT),
            "git_blob_hash": git_blob_hash(THIS_SCRIPT),
        },
        "current_formula_note": ("normalized_spread() = MEDIAN nearest-neighbour distance / "
                                 "expected nearest-neighbour distance for N uniformly-"
                                 "distributed points over the AOI area (~0.5*sqrt(area/N)). "
                                 "See src/preflight.py's normalized_spread() docstring and "
                                 "MIN_NORMALIZED_SPREAD's comment for the full derivation."),
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_PATH}")
    n_pass = sum(1 for r in rows if r["spread_gate_would_pass"])
    print(f"{n_pass}/{len(rows)} rows pass the spread gate under the current formula")


if __name__ == "__main__":
    main()
