"""DW-S1-14: the HF backbone must load offline-first and fail closed.

`da2_utils.load_model()` used to call `from_pretrained(MODEL_ID)` with no
`local_files_only`, so a cold cache silently reached huggingface.co at inference time -
a hidden network dependency that a warm dev cache hides and a fresh demo machine
discovers at the worst moment. These tests pin the hardened contract:

  - default  -> local_files_only=True, no online access, no retry after a local miss;
  - opt-in   -> the ONLY path allowed to resolve/download online;
  - ordering -> a missing fine-tuned checkpoint is refused BEFORE the (network-capable)
                backbone is constructed;
  - the runner CLI flag actually reaches the loader.

`transformers.AutoModelForDepthEstimation` is mocked throughout: no network, no weights.

    pytest tests/test_backbone_offline_dw_s1_14.py
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import da2_utils  # noqa: E402


class _FakeModel:
    """Stands in for the HF model; records the device handed to .to()."""

    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class _Recorder:
    """Fake AutoModelForDepthEstimation.from_pretrained with a call log."""

    def __init__(self, raises=None):
        self.calls = []
        self.raises = raises
        self.model = _FakeModel()

    def from_pretrained(self, model_id, **kwargs):
        self.calls.append((model_id, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.model


def _patched_transformers(rec):
    """Patch the lazily-imported transformers symbol load_model() pulls in."""
    import types
    mod = types.SimpleNamespace(AutoModelForDepthEstimation=rec)
    return patch.dict(sys.modules, {"transformers": mod})


# --------------------------------------------------------------- default = offline
def test_default_load_is_local_files_only_and_pins_no_invented_revision():
    """The default path must be local-cache-only, and must not fabricate a revision.

    The exact HF snapshot used at fine-tune time was never durably recorded
    (REVISION_NOT_ESTABLISHED), so pinning a guessed revision here would be a fake
    provenance claim. Leaving it unset resolves the same local snapshot the un-pinned
    call already resolved.
    """
    rec = _Recorder()
    with _patched_transformers(rec):
        da2_utils.load_model("cpu")

    assert len(rec.calls) == 1
    model_id, kwargs = rec.calls[0]
    assert model_id == da2_utils.MODEL_ID
    assert kwargs["local_files_only"] is True
    assert da2_utils.MODEL_REVISION is None, "no revision is established; do not invent one"
    assert "revision" not in kwargs, "must not pass an unverified revision"


def test_device_transfer_is_unchanged():
    """.to(device) must still happen, with the caller's device, exactly as before."""
    rec = _Recorder()
    with _patched_transformers(rec):
        out = da2_utils.load_model("cpu")
    assert rec.model.to_calls == ["cpu"]
    assert out is rec.model


# ------------------------------------------------------- local miss = fail closed
def test_local_miss_fails_closed_with_no_online_retry():
    """A local cache miss must raise an actionable error and must NOT retry online.

    The whole point of the change: local failure must never silently become an online
    download. Exactly one from_pretrained call, and it was the offline one.
    """
    rec = _Recorder(raises=OSError("not found in local cache"))
    with _patched_transformers(rec):
        try:
            da2_utils.load_model("cpu")
        except da2_utils.BackboneUnavailableError as exc:
            msg = str(exc)
            assert da2_utils.MODEL_ID in msg, "error must name the required model"
            assert "will not download it silently" in msg
            assert "--allow-backbone-download" in msg, "error must say how to opt in"
            assert "not found in local cache" in msg, "underlying cause must be preserved"
        else:
            raise AssertionError("a missing local backbone was silently accepted")

    assert len(rec.calls) == 1, f"expected no retry, got {len(rec.calls)} calls"
    assert rec.calls[0][1]["local_files_only"] is True


def test_opt_in_is_the_only_path_that_permits_online():
    """local_files_only=False must be reachable ONLY via the explicit opt-in."""
    rec = _Recorder()
    with _patched_transformers(rec):
        da2_utils.load_model("cpu", allow_backbone_download=True)

    assert rec.calls[0][1]["local_files_only"] is False

    # ...and the opt-in is keyword-only, so it can never be set by a stray positional.
    import inspect
    param = inspect.signature(da2_utils.load_model).parameters["allow_backbone_download"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False


# --------------------------------------- unverified revision + download = fail closed
def test_finetuned_download_is_refused_while_revision_provenance_is_unestablished():
    """Review blocker (DW-S1-14): an explicit download for FINE-TUNED inference would fetch
    whatever upstream `main` is today. Such a snapshot can stay architecture- and
    state-dict-compatible, load strictly, and still shift predictions against the
    baseline. With MODEL_REVISION unpinned that combination must fail closed, and the
    online-capable loader must never be reached.
    """
    assert da2_utils.MODEL_REVISION is None, "precondition: revision provenance unestablished"

    rec = _Recorder()
    with _patched_transformers(rec):
        try:
            da2_utils.load_model("cpu", allow_backbone_download=True,
                                 for_finetuned_checkpoint=True)
        except da2_utils.BackboneRevisionUnverifiedError as exc:
            msg = str(exc)
            assert "NOT durably established" in msg
            assert "prediction comparability" in msg
            assert "No revision has been fabricated" in msg
            assert "--base-model" in msg, "must point at the still-allowed base-model route"
        else:
            raise AssertionError("an unpinned upstream download was allowed for a "
                                 "fine-tuned run")

    assert rec.calls == [], (
        "from_pretrained was reached despite unverified revision provenance; the refusal "
        "must happen before any online-capable construction")


def test_no_cached_snapshot_is_promoted_to_historical_provenance():
    """The one locally cached snapshot is circumstantial, not recorded training evidence.
    Hard-coding it here would manufacture provenance that does not exist.
    """
    src = Path(da2_utils.__file__).read_text(encoding="utf-8")
    assert "5426e4f0" not in src, "a cached snapshot hash was promoted to provenance"
    assert da2_utils.MODEL_REVISION is None


def test_base_model_only_download_stays_available():
    """The refusal is scoped to fine-tuned inference. A deliberate base-model run applies
    no checkpoint, so an explicit download must still be permitted there.
    """
    rec = _Recorder()
    with _patched_transformers(rec):
        da2_utils.load_model("cpu", allow_backbone_download=True,
                             for_finetuned_checkpoint=False)
    assert len(rec.calls) == 1
    assert rec.calls[0][1]["local_files_only"] is False


def test_runner_marks_the_finetuned_path_and_exempts_base_model():
    """The runner must tell the loader which mode it is in, or the refusal never fires."""
    from run_pipeline import load_predictor

    seen = []

    def _spy(device, *, allow_backbone_download=False, for_finetuned_checkpoint=False):
        seen.append(for_finetuned_checkpoint)
        raise _Sentinel()

    with patch.object(da2_utils, "load_model", _spy):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "ft.pt"
            ckpt.write_bytes(b"stub")
            stub_sha = hashlib.sha256(b"stub").hexdigest()
            with patch.dict(os.environ, {"DW_CHECKPOINT_SHA256": stub_sha}):
                for kwargs in ({"allow_base_model": False}, {"allow_base_model": True}):
                    try:
                        load_predictor(ckpt, allow_backbone_download=True, **kwargs)
                    except _Sentinel:
                        pass

    assert seen == [True, False], (
        "fine-tuned run must set for_finetuned_checkpoint=True, --base-model must not")


# ------------------------------------------------- ordering: checkpoint before backbone
def test_missing_checkpoint_is_refused_before_the_backbone_is_constructed():
    """Regression for the ordering defect: the runner used to build the HF backbone
    (a network-capable call) before checking that the required checkpoint exists. A
    missing checkpoint must now fail without ever touching the loader.
    """
    from run_pipeline import PipelineError, load_predictor

    called = []
    with patch.object(da2_utils, "load_model", lambda *a, **k: called.append(k) or object()):
        try:
            load_predictor(Path("/nonexistent/checkpoint.pt"), allow_base_model=False)
        except PipelineError as exc:
            assert "base model" in str(exc).lower()
        else:
            raise AssertionError("a missing checkpoint was silently accepted")

    assert called == [], "backbone was constructed before the checkpoint was validated"


# --------------------------------------------------------- runner CLI propagation
def _tiny_image(d):
    p = Path(d) / "in.png"
    Image.new("RGB", (600, 600)).save(p)
    return p


class _Sentinel(Exception):
    pass


def _run_main_capturing_load_predictor(argv_extra):
    """Run main() far enough to capture load_predictor's kwargs, then abort."""
    import run_pipeline as rp

    seen = {}

    def _spy(checkpoint, *, allow_base_model, allow_backbone_download=False):
        seen["allow_backbone_download"] = allow_backbone_download
        raise _Sentinel()

    with tempfile.TemporaryDirectory() as d:
        img = _tiny_image(d)
        with patch.object(rp, "load_predictor", _spy):
            try:
                rp.main([str(img), "--out", str(Path(d) / "out"), *argv_extra])
            except _Sentinel:
                pass
    return seen


def test_runner_defaults_to_no_download_and_propagates_the_opt_in():
    """The CLI flag must actually reach the loader - a declared-but-unwired flag would
    leave the download path unreachable and the default silently unchanged.
    """
    assert _run_main_capturing_load_predictor([]) == {"allow_backbone_download": False}
    assert _run_main_capturing_load_predictor(
        ["--allow-backbone-download"]) == {"allow_backbone_download": True}


# ------------------------------------------- other production call sites stay closed
def test_other_production_call_sites_inherit_the_offline_default():
    """predict_depth() and infer_relative_depth() load the backbone themselves when no
    model is injected. Neither may silently download.
    """
    import inspect

    import model_stages
    import relative_dsm

    for fn in (model_stages.predict_depth, relative_dsm.infer_relative_depth,
               relative_dsm.relative_dsm_from_image):
        src = inspect.getsource(fn)
        if "load_model(" in src:
            assert "allow_backbone_download=True" not in src, (
                f"{fn.__name__} must not hard-enable downloads")

    # And the shared default they all inherit is offline.
    assert inspect.signature(
        da2_utils.load_model).parameters["allow_backbone_download"].default is False


# ------------------------------------------------- checkpoint identity: hash is verified
def test_checkpoint_with_wrong_hash_is_refused_before_the_backbone_is_constructed():
    """A correctly named but different checkpoint file (e.g. a stale copy on a demo machine)
    must fail loudly with both hashes, not run and silently change every number."""
    from run_pipeline import EXPECTED_CHECKPOINT_SHA256, PipelineError, load_predictor

    called = []
    with patch.object(da2_utils, "load_model", lambda *a, **k: called.append(1)):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "da2_small_finetuned_iter2_partial_unfreeze.pt"
            ckpt.write_bytes(b"not the canonical weights")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DW_CHECKPOINT_SHA256", None)
                try:
                    load_predictor(ckpt, allow_base_model=False)
                except PipelineError as exc:
                    msg = str(exc)
                else:
                    raise AssertionError("a checkpoint with the wrong hash was accepted")
    assert "hash mismatch" in msg and EXPECTED_CHECKPOINT_SHA256 in msg
    assert hashlib.sha256(b"not the canonical weights").hexdigest() in msg
    assert called == [], "backbone was constructed before the checkpoint hash was verified"


def test_checkpoint_env_override_is_the_runner_default(monkeypatch):
    import importlib
    import run_pipeline as rp
    monkeypatch.setenv("DW_CHECKPOINT", "Z:/elsewhere/candidate.pt")
    seen = {}

    def _spy(checkpoint, *, allow_base_model, allow_backbone_download=False):
        seen["ckpt"] = checkpoint
        raise rp.PipelineError("stop")

    with tempfile.TemporaryDirectory() as d:
        img = Path(d) / "in.png"
        Image.new("RGB", (600, 600)).save(img)
        with patch.object(rp, "load_predictor", _spy):
            rp.main([str(img), "--out", str(Path(d) / "out")])
    assert seen["ckpt"] == Path("Z:/elsewhere/candidate.pt")
