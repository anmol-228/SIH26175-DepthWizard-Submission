"""Deterministic self-test for scripts/compare_baseline_metrics.py.

Run:  python scripts/test_compare_baseline_metrics.py
Exit 0 = all assertions passed.  No pytest dependency (repo has no test runner configured).
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_baseline_metrics as cbm

# Minimal fixture with the two blocks the harness reads.
def _report(oracle_rmse, sparse10_rmse):
    def block(rmse):
        return {"rmse": rmse, "mae": 3.0, "corr": 0.9, "bias": 0.0, "median_abs_err": 2.0,
                "abs_err_p50": 2.0, "abs_err_p75": 4.0, "abs_err_p90": 6.0,
                "abs_err_p95": 8.0, "abs_err_p99": 15.0}
    return {
        "global_pooled": {"finetuned_full_per_tile_oracle_calibration": block(oracle_rmse)},
        "sparse_stress_test": {"10": {"global": block(sparse10_rmse)}},
    }


def _write(tmp, name, obj):
    p = Path(tmp) / name
    p.write_text(json.dumps(obj))
    return str(p)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        base = _write(tmp, "base.json", _report(4.4034, 5.5006))
        same = _write(tmp, "same.json", _report(4.4034, 5.5006))
        drift = _write(tmp, "drift.json", _report(4.4034, 5.9000))
        bad = _write(tmp, "bad.json", {"global_pooled": {}})

        # 1. identical -> exit 0
        rc = cbm.main([base, same, "--sparse", "10"])
        assert rc == 0, f"identical reports should pass, got rc={rc}"

        # 2. drift beyond tol -> exit 1
        rc = cbm.main([base, drift, "--sparse", "10"])
        assert rc == 1, f"drifted report should fail, got rc={rc}"

        # 3. drift within a loose tol -> exit 0
        rc = cbm.main([base, drift, "--sparse", "10", "--tol", "1.0"])
        assert rc == 0, f"drift under loose tol should pass, got rc={rc}"

        # 4. malformed / missing block -> exit 1 (not a crash)
        rc = cbm.main([base, bad, "--sparse", "10"])
        assert rc == 1, f"malformed candidate should return 1, got rc={rc}"

        # 5. compare() yields the expected number of rows (1 oracle + 1 sparse) * METRICS
        rows = list(cbm.compare(json.loads(Path(base).read_text()),
                                json.loads(Path(same).read_text()),
                                [("full_oracle", "oracle", None), ("sparse_10", "sparse", 10)],
                                1e-6))
        assert len(rows) == 2 * len(cbm.METRICS), f"unexpected row count {len(rows)}"
        assert all(ok for *_, ok in rows), "identical fixture rows should all be ok"

    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
