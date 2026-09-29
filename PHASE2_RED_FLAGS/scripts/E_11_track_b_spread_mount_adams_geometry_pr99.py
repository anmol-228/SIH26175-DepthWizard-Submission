"""PR #99 (issue #98) reproducibility artifact for the Mount Adams N=100 geometry rows.

Anmol's second PR #99 review: the MIN_NORMALIZED_SPREAD comment says the 0.7389 floor was
re-derived across all 25 real-geometry runs (20 Vaihingen + 5 Mount Adams), but the committed
artifact (E_10 / REPRODUCIBILITY_PR99_MEDIAN_SPREAD.json) holds only the 20 Vaihingen rows.
This script produces the missing 5 Mount Adams rows.

GEOMETRY ONLY: normalized_spread() reads (row, col) and the raw tile shape, never elevation, so
the missing Mount Adams DEM does not block this. The raw shape (4000, 2200) is the already
documented Mount Adams shape (E_09's MOUNT_ADAMS_SHAPE, #73 / #91 Track A). Sampling is identical
to E_09 / E_10: default_rng(7000 + seed), rows then cols via integers(0, h) / integers(0, w).

READ-ONLY validation evidence: src/preflight.py is not modified, no threshold or formula change.
The Vaihingen artifact and E_10 are left untouched (this writes a sibling file).

REPRODUCE

    python PHASE2_RED_FLAGS/scripts/E_11_track_b_spread_mount_adams_geometry_pr99.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

P2ROOT = Path(__file__).resolve().parents[1]
ROOT = P2ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from preflight import MIN_NORMALIZED_SPREAD, normalized_spread  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
N = 100
SEED_BASE = 7000  # identical stream to #73 / #91 Track A and Track B's E_09 / E_10
MOUNT_ADAMS_SHAPE = (4000, 2200)  # (rows, cols), documented raw shape, geometry only

OUT_DIR = ROOT / "10_DOCUMENTATION" / "evidence" / "DW_ISSUE91_TRACK_B_GATE_VALIDATION_N100"
OUT_PATH = OUT_DIR / "REPRODUCIBILITY_PR99_MOUNT_ADAMS_GEOMETRY.json"
THIS_SCRIPT = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:                                        # noqa: BLE001
        return f"unknown ({exc})"


def git_provenance() -> dict:
    porcelain = git("status", "--porcelain=v1", "--untracked-files=all")
    lines = [l for l in porcelain.splitlines() if l.strip()]
    return {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
            "tree_state": "clean" if not lines else "dirty", "status_lines": lines}


def sample_references(h: int, w: int, n: int, seed: int):
    """Identical sampling to E_09 / E_10 / #73 / #91 Track A/B."""
    rng = np.random.default_rng(SEED_BASE + seed)
    rr = rng.integers(0, h, size=n)
    cc = rng.integers(0, w, size=n)
    return rr, cc


def main():
    if OUT_PATH.exists():
        raise RuntimeError(f"Refusing to run: {OUT_PATH} already exists.")

    start_time_utc = datetime.now(timezone.utc).isoformat()
    h, w = MOUNT_ADAMS_SHAPE
    rows = []
    for seed in SEEDS:
        rr, cc = sample_references(h, w, N, seed)
        pts_rc = np.stack([rr, cc], axis=1)
        spread = normalized_spread(pts_rc.astype(np.float64), (h, w))
        rows.append({
            "scene": "mount_adams",
            "raw_tile_shape_hw": [int(h), int(w)],
            "seed": seed,
            "seed_stream": SEED_BASE + seed,
            "n": N,
            "sampled_row_col_points": pts_rc.tolist(),
            "normalized_spread_current_formula": round(float(spread), 6),
            "min_normalized_spread_threshold": MIN_NORMALIZED_SPREAD,
            "spread_gate_would_pass": bool(spread >= MIN_NORMALIZED_SPREAD),
        })
        print(f"  mount_adams seed={seed}: spread={spread:.4f} (floor {MIN_NORMALIZED_SPREAD}) -> "
              f"{'PASS' if spread >= MIN_NORMALIZED_SPREAD else 'FAIL'}")

    report = {
        "schema": "dw-fs-pr99-track-b-spread-mount-adams-geometry/v1",
        "purpose": ("Second PR #99 review request: the five Mount Adams geometry-only rows "
                    "behind the '25 real-geometry runs, min 0.7389' provenance claim in "
                    "src/preflight.py's MIN_NORMALIZED_SPREAD comment (the 20 Vaihingen rows "
                    "are in REPRODUCIBILITY_PR99_MEDIAN_SPREAD.json)."),
        "pr_url": "internal tracker item 99",
        "issue_url": "internal tracker item 98",
        "scope_note": ("Geometry only: normalized_spread() takes (row, col) points and the raw "
                       "tile shape, never elevation. No Mount Adams DEM was read or needed. "
                       "Shape (4000, 2200) is the already-documented raw shape from E_09."),
        "run_start_utc": start_time_utc,
        "run_end_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_provenance(),
        "generator_script": {
            "path": str(THIS_SCRIPT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(THIS_SCRIPT),
            "git_blob_hash": git("hash-object", str(THIS_SCRIPT)),
        },
        "current_formula_note": ("normalized_spread() = MEDIAN nearest-neighbour distance / "
                                 "expected nearest-neighbour distance for N uniformly-"
                                 "distributed points, ~0.5*sqrt(area/N). See src/preflight.py."),
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    n_pass = sum(1 for r in rows if r["spread_gate_would_pass"])
    print(f"\nWrote {OUT_PATH}\n{n_pass}/{len(rows)} rows pass the spread gate under the current formula")


if __name__ == "__main__":
    main()
