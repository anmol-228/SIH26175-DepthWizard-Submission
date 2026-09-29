"""
tests/test_server_upload_timeout.py -- #87 [DW-FS-11] subprocess timeout.

viewer/server.py::run_pipeline_on_upload() invokes run_pipeline.py as a subprocess with
no timeout, so a hung pipeline run (bad GPU state, a pathological input) would freeze the
demo UI indefinitely. These tests mock subprocess.run directly (no real GPU/model
invocation) to verify the timeout is actually wired in, and that a TimeoutExpired is
converted into the SAME result-dict shape a normal run returns -- not a differently-shaped
error, since main.js's `if (result.exit_code !== 0)` branch already renders
result.stdout/result.stderr cleanly and expects those keys to exist.

Run: pytest tests/test_server_upload_timeout.py -v
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viewer"))

import server as server_module  # noqa: E402


def test_timeout_is_actually_passed_to_subprocess_run():
    """The fix must reach subprocess.run's own timeout= kwarg, not just exist as an
    unused constant -- confirmed by inspecting the actual call, not just the return
    value of a successful run."""
    captured_kwargs = {}

    def _fake_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    with tempfile.TemporaryDirectory() as d, patch.object(subprocess, "run", _fake_run):
        server_module.run_pipeline_on_upload(Path(d) / "in.png", Path(d) / "out")

    assert captured_kwargs.get("timeout") == server_module.UPLOAD_SUBPROCESS_TIMEOUT_S


def test_timeout_expired_returns_the_same_result_shape_as_a_normal_run():
    """The core fix: a TimeoutExpired must not propagate as a raised exception (which
    would hit /api/upload's generic except-handler and return a DIFFERENTLY-shaped
    {"ok": false, "reason": ...} body) -- it must be caught here and returned in the same
    shape a successful/failed run already uses, so main.js's existing exit_code check
    renders it correctly with no frontend change needed."""
    cmd = ["python", "run_pipeline.py"]

    def _fake_run(cmd_arg, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0), output=b"partial stdout", stderr=b"partial stderr")

    with tempfile.TemporaryDirectory() as d, patch.object(subprocess, "run", _fake_run):
        result = server_module.run_pipeline_on_upload(Path(d) / "in.png", Path(d) / "out")

    # Same keys a normal run/failure returns -- this is what main.js's response parsing needs.
    # available_glbs joined this set via #85/PR #97 (merged after this test was written) --
    # computed unconditionally after the try/except, so it's present here too, correctly [].
    assert set(result.keys()) == {
        "exit_code", "stdout", "stderr", "out_dir", "out_dir_name", "scene", "available_glbs",
    }
    assert result["exit_code"] != 0, "a timeout must never look like exit_code=0 (success)"
    assert "did not finish within" in result["stderr"] and "terminated" in result["stderr"]
    assert "partial stderr" in result["stderr"], "any output captured before the timeout should be preserved"
    assert result["available_glbs"] == [], "a timed-out run writes no output files"


def test_successful_run_is_unaffected_by_the_timeout_change():
    """Regression check: adding the timeout kwarg must not change behavior on the
    existing successful-run path."""
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="all good", stderr="")

    with tempfile.TemporaryDirectory() as d, patch.object(subprocess, "run", _fake_run):
        result = server_module.run_pipeline_on_upload(Path(d) / "scene.png", Path(d) / "out")

    assert result["exit_code"] == 0
    assert result["stdout"] == "all good"
    assert result["stderr"] == ""
    assert result["scene"] == "scene"


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
