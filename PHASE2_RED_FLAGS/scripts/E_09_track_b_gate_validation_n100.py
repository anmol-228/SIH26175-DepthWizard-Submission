"""DW-FS Sep23 / GitHub Issue #91, Track B: reference-quality-gated calibration validation.

Pre-registered contract (Track B / Candidate 1): docs/final_sprint/phase2/
HILLY_RESCUE_EXECUTION_COUNCIL.md Part 4.

IMPORTANT FINDING, established by reading src/preflight.py + its git history before writing
any code (git blame: commit 24a95b5, "DW-S1-10: calibration pre-flight gate (count +
coverage), wired into the runner", authored by Nirmalya, merged 2026-09-02 - three weeks
before issue #91 was opened): the hull-coverage/normalized-spread gate Track B's contract
describes as new work ("extends the existing diagnostic machinery to also gate on the
already-proven distribution finding... if the supplied reference points fail a
stratification/hull-coverage threshold, refuse metric calibration - instead of only gating
on relief span") is NOT new. `preflight_checks()` in src/preflight.py already gates on
count, hull_coverage, AND normalized_spread, independently of and in addition to the later
(#32/DW-P0-A5) relief-span gate - see preflight.py:394-411, wired into run_pipeline.py:231,
and already covered by tests/test_preflight.py's clustered/stratified test pairs
(test_clustered_reference_set_fails_with_a_reason, test_stratified_reference_set_passes,
test_hull_coverage_and_spread_behave_monotonically).

An internal planning document's claim ("Currently these are informational/gating only for the 87.5m
relief threshold") does not match the current source tree. Per the evidence hierarchy in
the project's evidence hierarchy (code/config beats internal summaries beats reasoning), the
code is treated as authoritative here, and this is reported as a correction rather than
silently acted on as if it were a gap to fill.

WHAT THIS SCRIPT ACTUALLY DOES, GIVEN THAT FINDING

Rather than reimplementing already-shipped, already-tested gating logic (which the Track B
contract's own MERGE RULE and general project policy - "do not add unnecessary
abstractions", "do not invent a new interface if an existing one can express it" - argue
against), this produces the genuinely missing piece the contract's own DATASET/SCENES/
METRICS/SUCCESS-THRESHOLD sections call for: real validation evidence of the EXISTING,
UNMODIFIED `preflight_checks()` against the actual large-N (N=100) uniform-random reference
geometry used by #73 and #91 Track A on Mount Adams (known-failure) and all four Vaihingen
controls (known-good) - not the small synthetic N=10-16 geometries the existing unit tests
already use. This is additive evidence generation only; src/preflight.py is not modified.

METRICS RECORDED per scene/seed: count, hull_coverage, normalized_spread, reference_relief_m,
which specific check(s) triggered a decline (if any), and the overall gate decision.

SUCCESS THRESHOLD (from the contract, evaluated honestly against this evidence): the gate
correctly declines on the known-bad case (Mount Adams) while not increasing refusal rate on
the known-good flat/urban stratified cases versus the existing DW_P0_A5 evidence baseline
(0 false refusals recorded there).

REPRODUCE

    "<worktree>" \\
        PHASE2_RED_FLAGS/scripts/E_09_track_b_gate_validation_n100.py
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

from preflight import preflight_checks  # noqa: E402

ISSUE_NUMBER = 91
ISSUE_URL = "internal tracker item 91"
CONTRACT_PATH = "docs/final_sprint/phase2/HILLY_RESCUE_EXECUTION_COUNCIL.md#part-4"
PREREGISTRATION_TIMESTAMP_UTC = "2026-09-23T15:51:11Z"

SEEDS = [0, 1, 2, 3, 4]              # identical to #73/#91 Track A, for direct comparability
N = 100
SEED_BASE = 7000                     # identical stream to #73/#91 Track A

MOUNT_ADAMS_SHAPE = (4000, 2200)     # EXPECTED_MOUNT_ADAMS_SHAPE, from #73/#91 Track A
VAIHINGEN_SHAPES = {                  # (rows, cols) per area, read from the real GeoTIFFs
    "6": None, "8": None, "34": None, "38": None,
}

OUT_DIR = P2ROOT.parent / "10_DOCUMENTATION" / "evidence" / "DW_ISSUE91_TRACK_B_GATE_VALIDATION_N100"
RESULTS_PATH = OUT_DIR / "RESULTS.json"
DECISION_PATH = OUT_DIR / "DECISION.json"
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
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    porcelain = git("status", "--porcelain=v1", "--untracked-files=all")
    lines = [l for l in porcelain.splitlines() if l.strip()]
    return {"branch": branch, "head": head,
            "tree_state": "clean" if not lines else "dirty",
            "status_lines": lines}


def sample_references(gt: np.ndarray, h: int, w: int, n: int, seed: int):
    """Identical sampling to #73/#91 Track A's sample_references - scene-wide uniform
    random, dedicated 7000+seed RNG stream."""
    rng = np.random.default_rng(SEED_BASE + seed)
    rr = rng.integers(0, h, size=n)
    cc = rng.integers(0, w, size=n)
    elev = gt[rr, cc].astype(np.float64)
    return rr, cc, elev


def evaluate_scene(name: str, role: str, gt: np.ndarray, results: list):
    h, w = gt.shape
    dummy_surface = np.zeros((h, w), dtype=np.float32)  # preflight_checks only reads .shape
    for seed in SEEDS:
        rr, cc, elev = sample_references(gt, h, w, N, seed)
        ref_points = np.stack([rr.astype(np.float64), cc.astype(np.float64), elev], axis=1)
        result = preflight_checks(ref_points, dummy_surface)
        triggered = []
        c = result.checks
        if c["count"] < c["thresholds"]["min_points"]:
            triggered.append("count")
        if c["hull_coverage"] < c["thresholds"]["min_hull_coverage"]:
            triggered.append("hull_coverage")
        if c["spread"] < c["thresholds"]["min_spread"]:
            triggered.append("spread")
        if c["reference_relief_m"] is not None and c["reference_relief_m"] > c["thresholds"]["max_reference_relief_m"]:
            triggered.append("reference_relief")
        results.append({
            "scene": name, "role": role, "seed": seed, "N": N,
            "gate_passed": result.passed,
            "gate_reason": result.reason,
            "count": c["count"], "hull_coverage": c["hull_coverage"], "spread": c["spread"],
            "reference_relief_m": c["reference_relief_m"],
            "triggered_checks": triggered,
        })
        print(f"  {name} seed={seed}: passed={result.passed} triggered={triggered or 'none'}")


def compute_decision(results: list) -> dict:
    mount_adams = [r for r in results if r["scene"] == "mount_adams"]
    controls = [r for r in results if r["scene"] != "mount_adams"]

    mount_adams_declined = sum(1 for r in mount_adams if not r["gate_passed"])
    control_false_refusals = sum(1 for r in controls if not r["gate_passed"])

    # Baseline false-refusal count on known-good scenes, from the existing shipped
    # DW_P0_A5 evidence (non_trigger_flat_run_manifest.json records a PASS on a
    # well-spread, low-relief synthetic reference set - 0 false refusals recorded there).
    baseline_control_false_refusals = 0
    control_regression = control_false_refusals - baseline_control_false_refusals

    success = bool(
        mount_adams_declined == len(mount_adams)
        and control_regression <= 1  # contract: "no more than 1 additional false refusal"
    )

    return {
        "gate_source": CONTRACT_PATH,
        "mount_adams_declined_count": mount_adams_declined,
        "mount_adams_total": len(mount_adams),
        "control_false_refusal_count": control_false_refusals,
        "control_total": len(controls),
        "baseline_control_false_refusals": baseline_control_false_refusals,
        "control_regression": control_regression,
        "SUCCESS_THRESHOLD_MET": success,
        "note": ("This validates the EXISTING, unmodified preflight_checks() gate "
                "(count + hull_coverage + normalized_spread + reference_relief, shipped "
                "since commit 24a95b5 / #25 and 09413db / #32 respectively) against real "
                "N=100 uniform-random reference geometry, not new gating logic."),
    }


def main():
    if RESULTS_PATH.exists() or DECISION_PATH.exists():
        raise RuntimeError(
            f"Refusing to run: Track B evidence already exists. "
            f"({RESULTS_PATH} exists={RESULTS_PATH.exists()}, "
            f"{DECISION_PATH} exists={DECISION_PATH.exists()})")

    start_time_utc = datetime.now(timezone.utc).isoformat()
    results: list = []

    ma_dir = P2ROOT / "data" / "hilly_terrain" / "aoi"
    ma_dem = np.load(ma_dir / "dem_1m.npy")
    assert ma_dem.shape == MOUNT_ADAMS_SHAPE, f"unexpected Mount Adams shape {ma_dem.shape}"
    evaluate_scene("mount_adams", "known_failure", ma_dem, results)

    import tifffile  # noqa: PLC0415 - only needed for the Vaihingen control DSMs
    va_dsm_dir = ROOT / "data" / "raw" / "Vaihingen" / "dsm"
    for area in ["6", "8", "34", "38"]:
        dsm = tifffile.imread(va_dsm_dir / f"dsm_09cm_matching_area{area}.tif")
        evaluate_scene(f"vaihingen_area{area}", "known_good", dsm, results)

    end_time_utc = datetime.now(timezone.utc).isoformat()
    decision = compute_decision(results)

    report = {
        "schema": "dw-fs-issue91-track-b-gate-validation-n100/v1",
        "issue_number": ISSUE_NUMBER,
        "issue_url": ISSUE_URL,
        "contract_path": CONTRACT_PATH,
        "preregistration_timestamp_utc": PREREGISTRATION_TIMESTAMP_UTC,
        "run_start_utc": start_time_utc,
        "run_end_utc": end_time_utc,
        "git": git_provenance(),
        "experiment_script": {"path": str(THIS_SCRIPT.relative_to(ROOT)).replace("\\", "/"),
                              "sha256": sha256_file(THIS_SCRIPT)},
        "preflight_module_note": ("src/preflight.py NOT modified by this script - this is a "
                                  "validation-evidence run of already-shipped gating logic."),
        "runs": results,
        "decision": decision,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    DECISION_PATH.write_text(json.dumps(decision, indent=2))
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote {DECISION_PATH}")
    print(f"\nSUCCESS_THRESHOLD_MET = {decision['SUCCESS_THRESHOLD_MET']}")
    print(f"mount_adams_declined = {decision['mount_adams_declined_count']}/{decision['mount_adams_total']}")
    print(f"control_false_refusals = {decision['control_false_refusal_count']}/{decision['control_total']}")


if __name__ == "__main__":
    main()
