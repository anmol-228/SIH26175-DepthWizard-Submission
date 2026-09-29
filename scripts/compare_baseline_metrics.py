"""Regression check for DepthWizard calibration-report JSONs.

Compares two reports produced by `scripts/07_calibrate_and_eval.py` (same shape:
`global_pooled.finetuned_full_per_tile_oracle_calibration` + `sparse_stress_test.<N>.global`)
and prints a per-metric delta table for the oracle and sparse categories.

Purpose: let a future PR answer "did this change move the frozen baseline?" without
re-deriving the ML experiment. It only *reads* existing metric fields; it never writes
to either input and never touches a checkpoint or the dataset.

Exit code:
  0  every compared metric within tolerance
  1  a metric drifted past --tol, OR a required field is missing/malformed

Usage:
  python scripts/compare_baseline_metrics.py BASELINE.json CANDIDATE.json [--tol 1e-6] [--sparse 10 ...]

All error metrics (rmse, mae, median_abs_err, abs_err_p*) are metres; corr and bias
are reported with the same tolerance for simplicity (they are not re-unit-ed).
"""
import argparse
import json
import sys
from pathlib import Path

ORACLE_PATH = ("global_pooled", "finetuned_full_per_tile_oracle_calibration")
METRICS = ("rmse", "mae", "corr", "bias", "median_abs_err",
           "abs_err_p50", "abs_err_p75", "abs_err_p90", "abs_err_p95", "abs_err_p99")
UNIT = {"corr": "", "bias": " m"}  # everything else is metres


class MetricError(Exception):
    pass


def _load(path):
    p = Path(path)
    if not p.is_file():
        raise MetricError(f"not a file: {path}")
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise MetricError(f"invalid JSON in {path}: {e}")


def _dig(report, keys, label):
    node = report
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise MetricError(f"{label}: missing key path {' -> '.join(map(str, keys))}")
        node = node[k]
    if not isinstance(node, dict):
        raise MetricError(f"{label}: {' -> '.join(map(str, keys))} is not an object")
    return node


def _block(report, category, sparse_n, label):
    if category == "oracle":
        return _dig(report, ORACLE_PATH, label)
    return _dig(report, ("sparse_stress_test", str(sparse_n), "global"), label)


def _metric(block, name, label):
    if name not in block:
        raise MetricError(f"{label}: metric '{name}' absent")
    v = block[name]
    if not isinstance(v, (int, float)):
        raise MetricError(f"{label}: metric '{name}' is not numeric ({v!r})")
    return float(v)


def compare(baseline, candidate, categories, tol):
    """Yield (category, metric, base_val, cand_val, abs_diff, ok). Raises MetricError on bad input."""
    for cat_label, category, sparse_n in categories:
        b = _block(baseline, category, sparse_n, f"baseline/{cat_label}")
        c = _block(candidate, category, sparse_n, f"candidate/{cat_label}")
        for m in METRICS:
            bv = _metric(b, m, f"baseline/{cat_label}")
            cv = _metric(c, m, f"candidate/{cat_label}")
            d = abs(bv - cv)
            yield cat_label, m, bv, cv, d, (d <= tol)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare two DepthWizard calibration-report JSONs.")
    ap.add_argument("baseline", help="frozen/reference metrics JSON")
    ap.add_argument("candidate", help="new metrics JSON to check against the baseline")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max allowed abs difference per metric (default 1e-6)")
    ap.add_argument("--sparse", type=int, nargs="*", default=[200, 50, 10, 5, 1],
                    help="sparse point-counts to compare (default: 200 50 10 5 1)")
    args = ap.parse_args(argv)

    categories = [("full_oracle", "oracle", None)]
    categories += [(f"sparse_{n}", "sparse", n) for n in args.sparse]

    try:
        baseline = _load(args.baseline)
        candidate = _load(args.candidate)
        rows = list(compare(baseline, candidate, categories, args.tol))
    except MetricError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"baseline : {args.baseline}")
    print(f"candidate: {args.candidate}")
    print(f"tolerance: {args.tol:g} per metric\n")
    print(f"{'category':<13} {'metric':<15} {'baseline':>20} {'candidate':>20} {'abs_diff':>13}  status")
    print("-" * 90)
    regressions = 0
    for cat, m, bv, cv, d, ok in rows:
        u = UNIT.get(m, " m")
        flag = "ok" if ok else "DRIFT"
        if not ok:
            regressions += 1
        print(f"{cat:<13} {m:<15} {bv:>20.12g} {cv:>20.12g} {d:>13.3e}  {flag}{u if not ok else ''}")

    print()
    if regressions:
        print(f"RESULT: {regressions} metric(s) exceeded tolerance {args.tol:g}.")
        return 1
    print(f"RESULT: all {len(rows)} compared metrics within tolerance {args.tol:g}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
