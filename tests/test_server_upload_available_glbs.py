"""
tests/test_server_upload_available_glbs.py -- #85 Finding A regression coverage.

run_pipeline_on_upload() must report which of the four possible mesh outputs
(relative, relative_lod, metric, metric_lod) actually landed on disk, so the
client can register every real tier instead of hardcoding just
"{scene}_relative.glb". subprocess.run is monkeypatched (the real
run_pipeline.py invocation is exercised elsewhere, e.g. tests/test_run_pipeline*.py
and the live #74/#85 Playwright passes) -- this test is only about the
file-detection logic added on top of it.

Run: pytest tests/test_server_upload_available_glbs.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "viewer"))

import server  # noqa: E402


def _stub_successful_subprocess(monkeypatch, exit_code: int = 0) -> None:
    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, exit_code, stdout="ok\n", stderr="")

    monkeypatch.setattr(server.subprocess, "run", _fake_run)


def test_reports_only_relative_when_calibration_did_not_run(tmp_path, monkeypatch):
    """Metric calibration failing/not running is the common case -- relative and
    relative_lod are always written, metric variants are not."""
    _stub_successful_subprocess(monkeypatch)
    out_dir = tmp_path / "run" / "scene1"
    out_dir.mkdir(parents=True)
    (out_dir / "scene1_relative.glb").write_bytes(b"glb")
    (out_dir / "scene1_relative_lod.glb").write_bytes(b"glb")

    result = server.run_pipeline_on_upload(tmp_path / "scene1.png", out_dir)

    assert result["available_glbs"] == ["relative", "relative_lod"]


def test_reports_all_four_when_calibration_succeeded(tmp_path, monkeypatch):
    _stub_successful_subprocess(monkeypatch)
    out_dir = tmp_path / "run" / "scene2"
    out_dir.mkdir(parents=True)
    for suffix in ("_relative.glb", "_relative_lod.glb", "_metric.glb", "_metric_lod.glb"):
        (out_dir / f"scene2{suffix}").write_bytes(b"glb")

    result = server.run_pipeline_on_upload(tmp_path / "scene2.png", out_dir)

    assert result["available_glbs"] == ["relative", "relative_lod", "metric", "metric_lod"]


def test_reports_empty_list_on_total_pipeline_failure(tmp_path, monkeypatch):
    """A failed run (nonzero exit) writes no mesh files at all -- available_glbs
    must be empty, not crash on a missing directory/files."""
    _stub_successful_subprocess(monkeypatch, exit_code=2)
    out_dir = tmp_path / "run" / "scene3"

    result = server.run_pipeline_on_upload(tmp_path / "scene3.png", out_dir)

    assert result["exit_code"] == 2
    assert result["available_glbs"] == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
