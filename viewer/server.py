"""
viewer/server.py -- shared local server for P0-A2 (reference-DSM validation) and
P0-A3 (image upload), issue #40.

Stdlib only for the SERVER PROCESS ITSELF (http.server) -- deliberately no
Flask/FastAPI/etc, since requirements.txt has no web framework today and
P0-B3's standalone launcher must not add a new dependency just to serve a
page + two small endpoints. This process DOES import geoio (numpy, tifffile)
for A2's reference-DSM comparison -- see viewer/requirements.txt and the
_preflight_check() function below for how that's now validated up front
with an actionable message, instead of a raw crash (B3 fix, 2026-09-14).

Endpoints:
  GET  /                          -> serves viewer/ static files (replaces the old
                                     `python -m http.server --directory viewer` one-liner)
  GET  /run/<path>                -> serves files from outputs/run/ (real pipeline
                                     results, per P0-A3 -- kept as a SEPARATE root
                                     from viewer/, matching this repo's existing
                                     outputs/run/<scene>/ convention used by the
                                     non-author evidence run, DW-P1c)
  POST /api/upload                -> P0-A3: receives an image, runs run_pipeline.py::main()
                                     for real, returns the output dir + enough info for
                                     the caller to build a /run/... URL for ?glb=-style
                                     loading
  POST /api/upload-reference      -> P0-A2: receives a reference .tif, saves it, returns
                                     a server-side path for use with /api/reference-sample
  POST /api/reference-sample      -> P0-A2: given a metric run's output dir + a reference
                                     .tif path + a clicked (row, col) into that run's own
                                     metric grid, returns {estimated_m, reference_m, delta_m}
                                     or an explicit "not comparable" reason -- never a
                                     fabricated number.

Design notes (grounded in the real code, not assumed):
  - run_pipeline.py's metric branch writes <scene>_metric.tif via
    to_geotiff(cal["dsm_metres"], source_gr, cal["epsg"], ...) -- this is a REAL,
    independently georeferenced GeoTIFF; geoio.load_geotiff() reads its actual
    geotransform back, so we do not need to add anything to the frozen
    viewer_input_v1 sidecar to know where a pixel sits in real-world space.
  - A clicked (row, col) is assumed to index the SAME grid as <scene>_metric.tif,
    since both the mesh's elevation.f32 sidecar and the metric .tif are written from
    the same `dsm_metres`/`dsm` array in run_pipeline.py's metric branch. This is a
    real assumption worth confirming with Nirmalya if reference-sampling in practice
    looks wrong -- flag it, don't silently paper over a mismatch.
  - A reference .tif with georeferenced=False (geoio's guessed-fallback case) is
    never compared -- per A2's DoD ("invalid comparison -> view-only/disabled, never
    fabricated"), this returns an explicit reason, not a number computed from a guess.
  - A clicked point that falls outside the reference raster's real-world footprint
    is reported as such, not silently clamped into a wrong answer.
  - CRS check (added per Nirmalya's PR #62 review, same class of finding as his PR #54
    review): metric_gr and reference_gr's real-world coordinates are only comparable
    if they're actually in the same coordinate system. epsg is compared explicitly;
    CRS_UNDEFINED (geoio's "transform is real but the authority code isn't defensible"
    marker) and None are both treated as "unknown, refuse" on EITHER side, per this
    repo's existing "don't assert what we can't defend" convention -- two
    CRS_UNDEFINED rasters might coincidentally be the same real system, but nothing
    here can defend that, so the comparison is refused rather than guessed.
  - Uploaded filenames (both /api/upload and /api/upload-reference) are sanitized to
    their basename before being used as a path component -- a client-controlled
    X-Filename header used unsanitized would be a path-traversal write primitive
    (Nirmalya, PR #62 review, finding 2).
  - GET serving deliberately covers TWO SEPARATE roots (viewer/ and outputs/run/),
    each independently guarded against path traversal, rather than one shared root --
    outputs/run/ is this repo's existing canonical location for real pipeline output
    (used elsewhere by the non-author evidence run, DW-P1c), so P0-A3's uploaded-image
    results are written there too, not redirected somewhere viewer-local just to dodge
    the static-file-serving boundary. Caught before shipping: an earlier version of
    this file only ever served viewer/, which would have returned 403 for every
    uploaded result -- never actually tested end-to-end before this was fixed.
  - P-14 fix (Anmol, live browser reproduction 2026-09-14): reference_sample()'s
    metric_tif_path/reference_tif_path arguments may arrive as a browser-facing
    "/run/..." web path (the ONLY shape the real viewer ever sends for the metric
    side, derived from ?glb=/currentGlbUrl) rather than a plain absolute filesystem
    path. _resolve_metric_path() mirrors do_GET's existing /run/ prefix-stripping
    and RUN_OUTPUT_DIR containment logic so POST /api/reference-sample resolves the
    SAME way GET /run/... already does. Before this fix, every genuine metric-branch
    reference-sample click failed "metric output not found" even though the file
    was real and present -- because Path("/run/...") on Windows resolves against
    the current drive root, not RUN_OUTPUT_DIR.
  - B3 fix (Anmol/Nirmalya/Ashmita, ASSIST-8 clean-checkout finding 2026-09-14):
    this file's own module-level `from geoio import ...` (below) transitively
    imports numpy/tifffile with NO error handling -- on a genuinely clean
    environment (Nirmalya's isolated venv, zero third-party packages) this crashed
    with a raw ModuleNotFoundError before the server could even bind a port,
    despite the file's own "Stdlib only" docstring claim (true when only static
    serving existed, stale since A2 added the geoio import). _preflight_check()
    now validates the bounded-supported Python version (3.12.x/3.14.x) and these two
    startup dependencies BEFORE the geoio import runs, printing a concise,
    actionable message and exiting non-zero instead of a raw traceback. Per
    Anmol's explicit scope: never installs anything automatically, never
    contacts the network, and does NOT pull the full A3/model runtime
    (torch/transformers/trimesh/etc.) into this minimal check -- those remain
    required only inside the run_pipeline.py subprocess /api/upload spawns,
    confirmed via dependency-graph trace to NOT be imported into this process
    at all. See viewer/requirements.txt for the minimal startup pin set, and
    the root requirements.txt for the full A3/model runtime.
"""

from __future__ import annotations

import sys


# Interpreters with real evidence behind them: 3.14 runs the full model pipeline on the
# primary development machine (01_ENVIRONMENT_AUDIT.md, BASELINE_v1.0.md; the /api/upload
# subprocess uses this same interpreter), 3.12 is the isolated demo-fallback venv.
SUPPORTED_PYTHON = {(3, 12), (3, 14)}


def _preflight_check() -> None:
    """
    B3 fix: validate the bounded-supported Python version and the minimal
    startup dependencies (numpy, tifffile -- see viewer/requirements.txt)
    BEFORE importing geoio below, which is what previously crashed with a
    raw, undocumented ModuleNotFoundError on a genuinely clean environment
    (Nirmalya, ASSIST-8 clean-checkout test, 2026-09-14). Exits non-zero
    with a concise, actionable message instead. Never installs anything
    automatically, never contacts the network -- per Anmol's explicit
    scope for this fix.

    #72 fix: also probes for the HF backbone model cache, offline and
    filesystem-only. `da2_utils.load_model()` already fails closed correctly
    today (local_files_only=True by default, --allow-backbone-download off
    end-to-end) -- a cold cache does not hang or silently download. What was
    missing is WHEN that failure surfaces: today it's mid-upload, in front of
    a judge, instead of at server startup before anyone is watching. This
    check moves the failure earlier; it changes no runtime behavior on any
    currently-passing path. No torch/transformers import and no network call
    here -- pure filesystem existence checks against the same HF hub cache
    layout `transformers` itself uses.
    """
    if sys.version_info[:2] not in SUPPORTED_PYTHON:
        print(
            "ERROR: DepthWizard viewer requires Python 3.12.x or 3.14.x "
            f"(this interpreter is {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}).\n"
            "Re-run with the project .venv interpreter (Python 3.12 or 3.14).",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = []
    for pkg in ("numpy", "tifffile"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(
            "ERROR: missing required viewer startup package(s): " + ", ".join(missing) + "\n"
            "Install them with:\n"
            "    pip install -r viewer/requirements.txt\n"
            "then re-run:\n"
            "    python viewer/server.py",
            file=sys.stderr,
        )
        sys.exit(1)

    import os
    from pathlib import Path

    # Kept as a local, preflight-only constant rather than importing
    # da2_utils.MODEL_ID -- da2_utils imports numpy/torch/PIL at module level,
    # which this check must not pull in (see docstring above). Must stay in
    # sync with da2_utils.MODEL_ID by hand.
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    repo_dirname = "models--" + model_id.replace("/", "--")

    def _expand(raw: str) -> Path:
        # Match huggingface_hub's own resolution ORDER exactly (verified against our
        # pinned huggingface_hub==1.28.0, constants.py): expanduser() runs FIRST, then
        # expandvars() SECOND -- os.path.expandvars(os.path.expanduser(getenv(...))).
        # Getting this order backwards is not just cosmetic: for a nested value like
        # HF_HUB_CACHE=$CACHE_ROOT with CACHE_ROOT=~/hf-cache, expandvars-before-
        # expanduser would substitute $CACHE_ROOT into a literal "~/hf-cache" string
        # and then (wrongly) expand that leftover "~", resolving/accepting a location
        # the real loader does not -- the real loader's expanduser already ran BEFORE
        # the "$CACHE_ROOT" substitution happened, so it never sees a leading "~" to
        # expand and treats "~" as a literal directory name instead (Anmol, PR #80
        # review, round 2).
        return Path(os.path.expandvars(os.path.expanduser(raw)))

    # Precedence mirrors huggingface_hub.constants exactly, including the nested
    # default-chaining: HF_HUB_CACHE, else the legacy HUGGINGFACE_HUB_CACHE, else
    # HF_HOME/hub, else XDG_CACHE_HOME/huggingface/hub, else ~/.cache/huggingface/hub.
    # (HF_HOME's own default already folds in XDG_CACHE_HOME before appending
    # "huggingface"; omitting that step here would reject a valid cache the real
    # loader would still find via XDG_CACHE_HOME alone.)
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    huggingface_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")  # legacy env var
    hf_home = os.environ.get("HF_HOME")
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if hf_hub_cache:
        hub_cache_dir = _expand(hf_hub_cache)
    elif huggingface_hub_cache:
        hub_cache_dir = _expand(huggingface_hub_cache)
    elif hf_home:
        hub_cache_dir = _expand(hf_home) / "hub"
    elif xdg_cache_home:
        hub_cache_dir = _expand(xdg_cache_home) / "huggingface" / "hub"
    else:
        hub_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    repo_dir = hub_cache_dir / repo_dirname
    refs_main = repo_dir / "refs" / "main"

    cache_problem = None
    if not refs_main.is_file():
        cache_problem = f"'{refs_main}' not found"
    else:
        snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_dir / "snapshots" / snapshot_hash
        config_path = snapshot_dir / "config.json"
        weights_paths = [snapshot_dir / "model.safetensors", snapshot_dir / "pytorch_model.bin"]
        if not config_path.is_file():
            cache_problem = f"snapshot config missing at '{config_path}'"
        elif not any(p.is_file() for p in weights_paths):
            cache_problem = ("snapshot weights missing (checked "
                             + " and ".join(str(p) for p in weights_paths) + ")")

    if cache_problem is not None:
        print(
            f"ERROR: HF backbone model cache not found or incomplete for '{model_id}'.\n"
            f"  Problem: {cache_problem}\n"
            f"  Expected cache directory: {hub_cache_dir}\n"
            f"  Fix: copy the vetted HF cache for this model onto this machine (the "
            f"'{repo_dirname}' folder under a HF hub cache dir), or set HF_HOME/"
            "HF_HUB_CACHE to point at one that already has it.\n"
            "  The pipeline itself already fails closed and never downloads silently "
            "-- this check only surfaces that earlier, at startup.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    _preflight_check()

import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geoio import CRS_UNDEFINED, load_geotiff  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT / "viewer"
UPLOAD_DIR = ROOT / "outputs" / "uploads"
REFERENCE_DIR = ROOT / "outputs" / "references"
RUN_OUTPUT_DIR = ROOT / "outputs" / "run"

# Serve static files with sane content types; three.js ES modules need
# text/javascript, not the default text/plain some stdlib versions pick.
_EXTRA_MIME_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".glb": "model/gltf-binary",
    ".f32": "application/octet-stream",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _safe_filename(raw: str, default: str) -> str:
    """
    Reduce a client-supplied filename to just its basename, refusing to let it
    escape the intended upload directory via '..' or an absolute path. See
    module docstring -- Nirmalya, PR #62 review, finding 2.
    """
    name = Path(raw or default).name
    return name if name else default


def _resolve_metric_path(path_str: str) -> Path:
    """
    Resolve a metric-output path that may be either:
      - a browser-facing web path in /run/... URL-space, as the viewer actually
        sends it (derived from ?glb=/currentGlbUrl -- see
        sidecar.js::metricTifUrlFromSidecar, main.js::compareAtClick), or
      - a plain absolute filesystem path, as every existing caller/test in this
        module uses directly.

    P-14 bug (Anmol, live browser reproduction 2026-09-14): reference_sample()
    previously called Path(metric_tif_path) directly on a "/run/..." web path.
    On Windows this resolves against the CURRENT DRIVE ROOT, not RUN_OUTPUT_DIR
    -- unlike do_GET's sibling handling of the exact same /run/ URL space
    (_serve_file_from_root), which correctly strips the prefix and resolves
    under RUN_OUTPUT_DIR. The mesh/sidecar therefore loaded fine (GET), but
    every reference-sample click against a genuine metric output failed
    .exists() and refused cleanly -- refusing safely, but refusing on the ONLY
    flow that can ever reach the metric branch (the upload flow never passes
    --reference, so it only ever produces branch=relative).

    Mirrors _serve_file_from_root's containment guard: a resolved /run/ path
    must stay under RUN_OUTPUT_DIR, so "/run/../../<anything>" cannot escape
    it. Raises ValueError on an attempted escape -- callers must turn this
    into a clean {"ok": False, ...} refusal, never let it become an uncaught
    500/400 that leaks a stack trace to the client.
    """
    if path_str.startswith("/run/"):
        rel = path_str[len("/run/"):]
        candidate = (RUN_OUTPUT_DIR / rel).resolve()
        root = RUN_OUTPUT_DIR.resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"path escapes RUN_OUTPUT_DIR: {path_str!r}")
        return candidate
    return Path(path_str)


def _pixel_to_world(row: float, col: float, gr) -> tuple[float, float]:
    """Convert a raster's own (row, col) to real-world (x, y) using its geotransform."""
    x = gr.origin_x + col * gr.pixel_size_x
    y = gr.origin_y + row * gr.pixel_size_y  # pixel_size_y is already negative, north-up
    return x, y


def _world_to_pixel(x: float, y: float, gr) -> tuple[float, float]:
    """Inverse of _pixel_to_world -- real-world (x, y) into this raster's (row, col)."""
    col = (x - gr.origin_x) / gr.pixel_size_x
    row = (y - gr.origin_y) / gr.pixel_size_y
    return row, col


def _bilinear_sample(array, row: float, col: float) -> float | None:
    """Bilinear sample a (H, W) array at fractional (row, col). None if out of bounds."""
    h, w = array.shape[:2]
    if row < 0 or col < 0 or row > h - 1 or col > w - 1:
        return None
    r0, c0 = int(row), int(col)
    r1, c1 = min(r0 + 1, h - 1), min(c0 + 1, w - 1)
    fr, fc = row - r0, col - c0
    v00, v01 = array[r0, c0], array[r0, c1]
    v10, v11 = array[r1, c0], array[r1, c1]
    return float(
        v00 * (1 - fr) * (1 - fc)
        + v01 * (1 - fr) * fc
        + v10 * fr * (1 - fc)
        + v11 * fr * fc
    )


def _crs_comparable(metric_epsg, reference_epsg) -> tuple[bool, str | None]:
    """
    True only if both sides carry a real, defensible, MATCHING EPSG code.
    CRS_UNDEFINED or None on either side means "unknown, refuse" -- per this repo's
    existing convention of not asserting what we can't defend (Nirmalya, PR #62
    review). Two CRS_UNDEFINED rasters might coincidentally share a real system,
    but nothing here can prove that, so refuse rather than guess.
    """
    if metric_epsg is None or metric_epsg == CRS_UNDEFINED:
        return False, "estimated output has no defensible CRS (undefined/unknown)"
    if reference_epsg is None or reference_epsg == CRS_UNDEFINED:
        return False, "reference file has no defensible CRS (undefined/unknown)"
    if metric_epsg != reference_epsg:
        return False, f"CRS mismatch: estimated output is EPSG:{metric_epsg}, reference is EPSG:{reference_epsg}"
    return True, None


def reference_sample(metric_tif_path: str, reference_tif_path: str, row: float, col: float) -> dict:
    """
    Core A2 logic: sample the pipeline's own metric output and a user-supplied
    reference raster at the same real-world location, return both plus delta_m --
    or an explicit reason why the comparison is not valid. Never a fabricated number.
    """
    try:
        metric_path = _resolve_metric_path(metric_tif_path)
        reference_path = _resolve_metric_path(reference_tif_path)
    except ValueError as exc:
        return {"ok": False, "reason": str(exc)}

    if not metric_path.exists():
        return {"ok": False, "reason": f"metric output not found: {metric_path}"}
    if not reference_path.exists():
        return {"ok": False, "reason": f"reference file not found: {reference_path}"}

    metric_gr = load_geotiff(metric_path)
    estimated_m = _bilinear_sample(metric_gr.array, row, col)
    if estimated_m is None:
        return {"ok": False, "reason": "clicked point is outside the estimated output's own grid"}

    reference_gr = load_geotiff(reference_path)
    if not reference_gr.georeferenced:
        return {
            "ok": False,
            "reason": (
                "reference file has no real georeference (embedded GeoTIFF tags or a "
                ".tfw sidecar) -- comparison disabled rather than guessed"
            ),
        }

    comparable, reason = _crs_comparable(metric_gr.epsg, reference_gr.epsg)
    if not comparable:
        return {"ok": False, "reason": f"{reason} -- comparison disabled rather than guessed"}

    world_x, world_y = _pixel_to_world(row, col, metric_gr)
    ref_row, ref_col = _world_to_pixel(world_x, world_y, reference_gr)
    reference_m = _bilinear_sample(reference_gr.array, ref_row, ref_col)
    if reference_m is None:
        return {
            "ok": False,
            "reason": "clicked point's real-world location falls outside the reference raster's footprint",
        }

    return {
        "ok": True,
        "estimated_m": estimated_m,
        "reference_m": reference_m,
        "delta_m": estimated_m - reference_m,
        "units": "m",
    }


# #87: hard cap on the upload subprocess. Without this, a hung run_pipeline.py (bad GPU
# state, a pathological input) freezes the demo UI indefinitely with no recovery path --
# real observed runs finish in well under a minute (51.951s for a full metric run, per the
# DW_TASK2 evidence), so 5 minutes is generous headroom, not a tight bound likely to
# false-trigger on a legitimate input.
UPLOAD_SUBPROCESS_TIMEOUT_S = 300


def run_pipeline_on_upload(image_path: Path, out_dir: Path) -> dict:
    """
    A3: invoke the REAL run_pipeline.py::main() as a subprocess (not re-implemented
    inline) against an uploaded image, return its exit code + output dir + the scene
    stem so the caller can build a /run/<scene_dir>/<scene>_relative.glb URL (the
    relative branch is ALWAYS written by run_pipeline.py; metric only when
    calibration succeeds -- confirmed directly against its code, not assumed).

    Note (B3 scope): this subprocess needs the FULL project requirements.txt
    (torch, transformers, trimesh, etc.) -- confirmed via dependency-graph trace
    that none of these are imported into THIS process directly, only invoked as
    a separate `python run_pipeline.py ...` subprocess. The minimal
    viewer/requirements.txt (numpy, tifffile) validated by _preflight_check()
    above covers only what THIS server process needs to start and serve A2;
    it deliberately does not and should not cover A3's model runtime.

    #87: a timeout is enforced (UPLOAD_SUBPROCESS_TIMEOUT_S). On expiry, the timeout is
    caught HERE rather than left to propagate to the generic /api/upload exception
    handler, so the response keeps the SAME shape callers already expect
    (exit_code/stdout/stderr/out_dir/out_dir_name/scene) instead of a differently-shaped
    {"ok": false, "reason": ...} error body -- main.js's existing
    `if (result.exit_code !== 0)` branch already renders stdout/stderr cleanly; keeping the
    shape consistent means that path handles a timeout correctly with no frontend change,
    rather than rendering "pipeline failed (exit undefined): undefined".
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "run_pipeline.py"), str(image_path), "--out", str(out_dir)]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=UPLOAD_SUBPROCESS_TIMEOUT_S,
        )
        exit_code = result.returncode
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = -1
        # exc.stdout/exc.stderr are normally str here (text=True was passed to the timed-out
        # run), but defensively decode if bytes ever surface -- cheap, and avoids a
        # str/bytes TypeError on the concatenation below masking the real timeout error.
        def _as_text(v) -> str:
            if v is None:
                return ""
            return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
        stdout = _as_text(exc.stdout)
        stderr = (
            _as_text(exc.stderr)
            + f"\nERROR: pipeline did not finish within {UPLOAD_SUBPROCESS_TIMEOUT_S}s and was "
            "terminated. This may indicate a stuck GPU/model state -- consider restarting the "
            "server before the next attempt."
        )

    # #85 Finding A: relative_lod.glb is ALWAYS written alongside relative.glb on success;
    # metric variants only when the calibration branch succeeds. Report which actually landed
    # on disk so the client can register every real tier instead of hardcoding just
    # "{scene}_relative.glb" -- confirmed against run_pipeline.py's own output naming, not
    # assumed. Computed unconditionally, whether or not the subprocess above timed out --
    # a timeout normally means no output files exist yet, so this correctly reports [] in
    # that case, same shape either way.
    scene = image_path.stem
    available_glbs = [
        tag for tag, suffix in (
            ("relative", "_relative.glb"),
            ("relative_lod", "_relative_lod.glb"),
            ("metric", "_metric.glb"),
            ("metric_lod", "_metric_lod.glb"),
        )
        if (out_dir / f"{scene}{suffix}").exists()
    ]
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "out_dir": str(out_dir),
        "out_dir_name": out_dir.name,
        "scene": scene,
        "available_glbs": available_glbs,
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _serve_file_from_root(self, root: Path, rel_path: str) -> None:
        """
        Shared static-file serving logic for a given root directory, with a
        path-traversal guard specific to THAT root. Used for both viewer/ (the app
        itself) and outputs/run/ (real pipeline results, P0-A3) as two independent,
        separately-guarded roots -- see module docstring.
        """
        file_path = (root / rel_path).resolve()
        if root.resolve() not in file_path.parents and file_path != root.resolve():
            self.send_response(403)
            self.end_headers()
            return
        if not file_path.is_file():
            self.send_response(404)
            self.end_headers()
            return

        suffix = file_path.suffix.lower()
        content_type = _EXTRA_MIME_TYPES.get(suffix, "application/octet-stream")
        if suffix in (".html", ".htm"):
            content_type = "text/html"
        elif suffix == ".json":
            content_type = "application/json"

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 (stdlib method name)
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/run/"):
            self._serve_file_from_root(RUN_OUTPUT_DIR, path[len("/run/"):])
            return

        rel_path = path.lstrip("/") or "index.html"
        self._serve_file_from_root(VIEWER_DIR, rel_path)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/api/reference-sample":
            try:
                body = self._read_json_body()
                result = reference_sample(
                    metric_tif_path=body["metric_tif_path"],
                    reference_tif_path=body["reference_tif_path"],
                    row=float(body["row"]),
                    col=float(body["col"]),
                )
                self._send_json(200, result)
            except Exception as exc:  # noqa: BLE001 -- surface the real error to the caller
                self._send_json(400, {"ok": False, "reason": str(exc)})
            return

        if parsed.path == "/api/upload-reference":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                filename = _safe_filename(self.headers.get("X-Filename"), "reference.tif")
                REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
                ref_path = REFERENCE_DIR / filename
                ref_path.write_bytes(raw)
                self._send_json(200, {"ok": True, "reference_tif_path": str(ref_path)})
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"ok": False, "reason": str(exc)})
            return

        if parsed.path == "/api/upload":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                filename = _safe_filename(self.headers.get("X-Filename"), "upload.png")
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                image_path = UPLOAD_DIR / filename
                image_path.write_bytes(raw)

                out_dir = RUN_OUTPUT_DIR / image_path.stem
                result = run_pipeline_on_upload(image_path, out_dir)
                self._send_json(200, result)
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"ok": False, "reason": str(exc)})
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):  # keep default stderr logging, just labeled
        sys.stderr.write(f"[viewer-server] {self.address_string()} - {fmt % args}\n")


def serve(port: int = 8793) -> None:
    """
    B3 fix: bind the port explicitly with a try/except, so an already-in-use
    port 8793 produces a concise, actionable message instead of an ambiguous
    traceback (Anmol's explicit spec: "friendly port-8793 conflict handling").
    Deliberately does NOT identify or kill any owning process -- per Anmol's
    explicit instruction, the tester must independently identify stale
    processes (netstat -ano | findstr :8793), this function only refuses to
    silently hang or crash unclearly.
    """
    try:
        # Explicitly disable SO_REUSEADDR (Windows quirk: with it on, binding
        # can silently succeed even when another process is actively
        # LISTENING on the same port, unlike POSIX -- meaning a genuine
        # stale-process conflict would go undetected instead of failing
        # clearly, defeating the whole point of this check).
        ThreadingHTTPServer.allow_reuse_address = False
        # #87: bind localhost only, not "" (all interfaces). This server has no auth on
        # /api/upload (spawns run_pipeline.py) or /api/upload-reference, so binding "" would
        # expose both to anyone else on the same network (venue WiFi during the demo is
        # exactly this scenario) -- a real unauthenticated-exposure risk, not just style.
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        print(
            f"ERROR: could not start the server on port {port} -- {exc}\n"
            "Something else may already be using this port. Check with:\n"
            f"    netstat -ano | findstr :{port}\n"
            "and stop that process manually, or restart once the port is free.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Serving DepthWizard viewer + API on http://localhost:{port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    serve()
