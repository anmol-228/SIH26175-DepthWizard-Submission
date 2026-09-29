"""
tests/test_server_preflight.py -- P0-B3 fix tests.

Covers Anmol's explicit B3 requirements:
  - missing prerequisites fail clearly/actionably rather than a raw traceback
  - the bounded-supported Python version (3.12.x) is validated explicitly
  - port-8793-in-use is handled with a clear message, no process is killed

These tests run server.py as a real subprocess (not importing it directly into
the test process) so we can control sys.version_info / block imports cleanly
without contaminating the actual test runner's own numpy/tifffile imports
(this test file itself, and the rest of the suite, needs numpy -- we are
testing server.py's behavior in ISOLATION, not removing our own dependencies).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"
SERVER_PY = VIEWER_DIR / "server.py"


def _run_server_subprocess(env_overrides: dict[str, str] | None = None, timeout: float = 5.0) -> subprocess.CompletedProcess:
    """
    Run `python viewer/server.py` as an isolated subprocess and return
    immediately (the preflight check exits before serve_forever() would
    ever block, so this never actually needs to be killed for the failure
    cases -- only the success-path port test below needs a running process).
    """
    return subprocess.run(
        [sys.executable, str(SERVER_PY)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(SERVER_PY.parent.parent),
        env=env_overrides,
    )


class TestPreflightPythonVersion:
    def test_wrong_python_version_fails_with_clear_message_not_traceback(self, tmp_path):
        """
        B3 requirement: 'print a concise, actionable setup message... exit
        non-zero... do not produce the current raw ModuleNotFoundError
        traceback.'

        Hermetic by construction (Ashmita, PR #70 review): does NOT rely on
        the ambient interpreter's real version/installed packages. The
        earlier version of this test just ran `python viewer/server.py` and
        hoped it failed naturally based on whatever this machine happens to
        be -- on a correctly-provisioned Python 3.12 environment (exactly
        what Nirmalya's non-author retest uses), the preflight check would
        actually SUCCEED and fall through into the real serve_forever(),
        which never returns on its own. subprocess.run's timeout would then
        raise TimeoutExpired (an uncaught exception, not a clean test
        failure) AND leak an orphaned server process still bound to port
        8793 -- breaking every subsequent test or launch attempt.

        Fixed by faking sys.version_info to an unsupported value (3.11)
        INSIDE an isolated wrapper subprocess, before server.py's own
        top-level code runs -- deterministic regardless of the host's real
        Python version, same technique as TestPreflightDependencyCheck
        below. A Popen + timeout + kill() safety net guarantees no process
        can ever leak from this test, even if the fix itself regresses.
        """
        wrapper = tmp_path / "run_wrong_version_isolated.py"
        wrapper.write_text(
            textwrap.dedent(
                f"""
                import sys
                sys.version_info = (3, 11, 9, "final", 0)
                import runpy
                runpy.run_path(r"{SERVER_PY}", run_name="__main__")
                """
            )
        )
        proc = subprocess.Popen(
            [sys.executable, str(wrapper)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise AssertionError(
                "server.py did not exit within 5s -- the Python-version "
                "preflight gate may not be firing; a real server process "
                "would have leaked and stayed bound to port 8793"
            )

        assert proc.returncode == 1
        assert "Traceback" not in stderr
        assert "ERROR:" in stderr
        assert "Python 3.12" in stderr and "3.14" in stderr


class TestPreflightDependencyCheck:
    def test_missing_numpy_tifffile_gives_actionable_message(self, tmp_path):
        """
        Isolate JUST the dependency-check branch by running a small wrapper
        script that fakes sys.version_info to 3.12 (so the version gate
        passes) and blocks numpy/tifffile imports specifically, then
        executes server.py's actual preflight logic. This is the same
        technique used interactively to verify the fix, now captured as a
        real regression test.
        """
        wrapper = tmp_path / "run_preflight_isolated.py"
        wrapper.write_text(
            textwrap.dedent(
                f"""
                import sys, builtins, importlib.util

                sys.version_info = (3, 12, 0, "final", 0)

                real_import = builtins.__import__
                def blocking_import(name, *args, **kwargs):
                    if name in ("numpy", "tifffile"):
                        raise ImportError(f"simulated missing: {{name}}")
                    return real_import(name, *args, **kwargs)
                builtins.__import__ = blocking_import

                spec = importlib.util.spec_from_file_location(
                    "__main__", r"{SERVER_PY}"
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                """
            )
        )
        result = subprocess.run(
            [sys.executable, str(wrapper)],
            capture_output=True,
            text=True,
            timeout=5.0,
        )

        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "numpy" in result.stderr
        assert "tifffile" in result.stderr
        assert "pip install -r viewer/requirements.txt" in result.stderr


class TestPreflightHFCache:
    """
    #72: offline, filesystem-only probe for the HF backbone model cache.
    `da2_utils.load_model()` already fails closed correctly today
    (local_files_only=True by default) -- this only moves that failure from
    mid-upload to server startup.

    Hermetic by construction, same technique as TestPreflightDependencyCheck
    above: a wrapper script fakes sys.version_info to 3.12 (so the Python
    gate passes) and points HF_HUB_CACHE at a real, throwaway tmp_path
    directory built to look like (or deliberately not look like) a complete
    HF hub cache -- never touches the real machine's actual HF cache.

    Loads server.py under a module name other than "__main__" (so neither of
    its own `if __name__ == "__main__":` guards auto-fire) and then calls
    `_preflight_check()` explicitly -- this exercises the exact function
    under test without ever falling through to `serve()`'s real
    `serve_forever()` (which never returns on its own) or binding a real
    port, which the miss/hit cases here have no need to touch.
    """

    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
    REPO_DIRNAME = "models--depth-anything--Depth-Anything-V2-Small-hf"

    def _run_preflight_isolated(self, tmp_path: Path, hf_cache_dir: Path) -> subprocess.CompletedProcess:
        wrapper = tmp_path / "run_hf_cache_isolated.py"
        wrapper.write_text(
            textwrap.dedent(
                f"""
                import os, sys
                sys.version_info = (3, 12, 0, "final", 0)
                os.environ["HF_HUB_CACHE"] = r"{hf_cache_dir}"
                os.environ.pop("HF_HOME", None)

                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "server_under_test", r"{SERVER_PY}"
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)   # __name__ != "__main__": neither
                                                # top nor bottom guard fires here
                mod._preflight_check()
                print("PREFLIGHT_OK")
                """
            )
        )
        return subprocess.run(
            [sys.executable, str(wrapper)],
            capture_output=True,
            text=True,
            timeout=5.0,
        )

    def _build_complete_cache(self, cache_root: Path, snapshot_hash: str = "deadbeef" * 5) -> None:
        repo_dir = cache_root / self.REPO_DIRNAME
        (repo_dir / "refs").mkdir(parents=True)
        (repo_dir / "refs" / "main").write_text(snapshot_hash)
        snapshot_dir = repo_dir / "snapshots" / snapshot_hash
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text("{}")
        (snapshot_dir / "model.safetensors").write_bytes(b"\x00")

    def test_missing_cache_dir_gives_actionable_message(self, tmp_path):
        empty_cache = tmp_path / "empty_hf_cache"
        empty_cache.mkdir()

        result = self._run_preflight_isolated(tmp_path, empty_cache)

        assert result.returncode == 1
        assert "PREFLIGHT_OK" not in result.stdout
        assert "Traceback" not in result.stderr
        assert self.MODEL_ID in result.stderr
        assert "HF_HOME" in result.stderr and "HF_HUB_CACHE" in result.stderr
        assert str(empty_cache) in result.stderr

    def test_missing_snapshot_config_gives_actionable_message(self, tmp_path):
        cache_root = tmp_path / "hf_cache"
        repo_dir = cache_root / self.REPO_DIRNAME
        snapshot_hash = "cafebabe" * 5
        (repo_dir / "refs").mkdir(parents=True)
        (repo_dir / "refs" / "main").write_text(snapshot_hash)
        (repo_dir / "snapshots" / snapshot_hash).mkdir(parents=True)  # no config.json, no weights

        result = self._run_preflight_isolated(tmp_path, cache_root)

        assert result.returncode == 1
        assert "PREFLIGHT_OK" not in result.stdout
        assert "Traceback" not in result.stderr
        assert "config" in result.stderr.lower()

    def test_missing_weights_gives_actionable_message(self, tmp_path):
        cache_root = tmp_path / "hf_cache"
        repo_dir = cache_root / self.REPO_DIRNAME
        snapshot_hash = "01234567" * 5
        (repo_dir / "refs").mkdir(parents=True)
        (repo_dir / "refs" / "main").write_text(snapshot_hash)
        snapshot_dir = repo_dir / "snapshots" / snapshot_hash
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text("{}")  # no model.safetensors / pytorch_model.bin

        result = self._run_preflight_isolated(tmp_path, cache_root)

        assert result.returncode == 1
        assert "PREFLIGHT_OK" not in result.stdout
        assert "Traceback" not in result.stderr
        assert "weights" in result.stderr.lower()

    def test_stale_refs_main_pointing_at_missing_snapshot_gives_actionable_message(self, tmp_path):
        """refs/main present but the snapshot directory it names doesn't exist --
        a partially-deleted/corrupted cache, not just a totally absent one."""
        cache_root = tmp_path / "hf_cache"
        repo_dir = cache_root / self.REPO_DIRNAME
        (repo_dir / "refs").mkdir(parents=True)
        (repo_dir / "refs" / "main").write_text("nonexistent_snapshot_hash")
        # no snapshots/ dir created at all

        result = self._run_preflight_isolated(tmp_path, cache_root)

        assert result.returncode == 1
        assert "PREFLIGHT_OK" not in result.stdout
        assert "Traceback" not in result.stderr

    def test_complete_cache_passes_silently(self, tmp_path):
        cache_root = tmp_path / "hf_cache"
        self._build_complete_cache(cache_root)

        result = self._run_preflight_isolated(tmp_path, cache_root)

        assert result.returncode == 0
        assert "ERROR" not in result.stderr
        assert "Traceback" not in result.stderr
        assert "PREFLIGHT_OK" in result.stdout

    def _run_wrapper(self, wrapper_path: Path, body: str) -> subprocess.CompletedProcess:
        wrapper_path.write_text(textwrap.dedent(body))
        return subprocess.run(
            [sys.executable, str(wrapper_path)], capture_output=True, text=True, timeout=5.0,
        )

    def test_complete_cache_via_legacy_huggingface_hub_cache_env_passes_silently(self, tmp_path):
        """PR #80 review (Anmol): our pinned huggingface_hub==1.28.0 resolves
        HF_HUB_CACHE -> HUGGINGFACE_HUB_CACHE (legacy) -> HF_HOME/hub -> default
        (constants.py, verified against the installed package). The original
        implementation stopped at HF_HUB_CACHE/HF_HOME, which would false-reject a
        cache huggingface_hub itself validly resolves via the legacy env var alone."""
        cache_root = tmp_path / "hf_cache"
        self._build_complete_cache(cache_root)

        result = self._run_wrapper(tmp_path / "run_legacy_env_isolated.py", f"""
            import os, sys
            sys.version_info = (3, 12, 0, "final", 0)
            os.environ.pop("HF_HUB_CACHE", None)
            os.environ.pop("HF_HOME", None)
            os.environ["HUGGINGFACE_HUB_CACHE"] = r"{cache_root}"

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "server_under_test", r"{SERVER_PY}"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod._preflight_check()
            print("PREFLIGHT_OK")
            """)

        assert result.returncode == 0
        assert "ERROR" not in result.stderr
        assert "Traceback" not in result.stderr
        assert "PREFLIGHT_OK" in result.stdout

    def test_cache_root_requiring_env_var_expansion_resolves_correctly(self, tmp_path):
        """PR #80 review (Anmol): huggingface_hub applies expandvars() to the resolved
        cache path, so a value like "$MY_CACHE_ROOT/hub" (a real pattern for a shared/
        provisioned cache location, e.g. on the primary demo machine) must resolve the
        same way here it would for the real model loader -- not be treated literally
        (which would false-reject a genuinely complete, valid cache)."""
        cache_root = tmp_path / "hf_cache"
        self._build_complete_cache(cache_root)

        result = self._run_wrapper(tmp_path / "run_expandvars_isolated.py", f"""
            import os, sys
            sys.version_info = (3, 12, 0, "final", 0)
            os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
            os.environ.pop("HF_HOME", None)
            os.environ["DW_TEST_CACHE_ROOT"] = r"{cache_root}"
            os.environ["HF_HUB_CACHE"] = "$DW_TEST_CACHE_ROOT"

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "server_under_test", r"{SERVER_PY}"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod._preflight_check()
            print("PREFLIGHT_OK")
            """)

        assert result.returncode == 0
        assert "ERROR" not in result.stderr
        assert "Traceback" not in result.stderr
        assert "PREFLIGHT_OK" in result.stdout

    def test_cache_root_requiring_user_expansion_resolves_correctly(self, tmp_path, monkeypatch):
        """PR #80 review (Anmol): huggingface_hub also applies expanduser() to the
        resolved cache path, so a "~/..." value must resolve against the real home
        directory the same way it would for the real model loader.

        Sets BOTH HOME and USERPROFILE, not just HOME (Anmol, PR #80 review, round 2):
        os.path.expanduser() resolves "~" from HOME on POSIX but from USERPROFILE (or
        HOMEDRIVE+HOMEPATH) on Windows -- our actual primary demo machine is Windows, so
        a HOME-only test would silently pass here while proving nothing there."""
        fake_home = tmp_path / "fake_home"
        cache_root = fake_home / "hf_cache_under_home"
        self._build_complete_cache(cache_root)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("USERPROFILE", str(fake_home))

        result = self._run_wrapper(tmp_path / "run_expanduser_isolated.py", f"""
            import os, sys
            sys.version_info = (3, 12, 0, "final", 0)
            os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
            os.environ.pop("HF_HOME", None)
            os.environ["HOME"] = r"{fake_home}"
            os.environ["USERPROFILE"] = r"{fake_home}"
            os.environ["HF_HUB_CACHE"] = "~/hf_cache_under_home"

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "server_under_test", r"{SERVER_PY}"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod._preflight_check()
            print("PREFLIGHT_OK")
            """)

        assert result.returncode == 0
        assert "ERROR" not in result.stderr
        assert "Traceback" not in result.stderr
        assert "PREFLIGHT_OK" in result.stdout

    def test_complete_cache_via_xdg_cache_home_only_passes_silently(self, tmp_path):
        """PR #80 review (Anmol), round 2: with none of HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE/
        HF_HOME set, huggingface_hub derives HF_HOME's own default from XDG_CACHE_HOME
        (join(XDG_CACHE_HOME, "huggingface")) before falling back further to ~/.cache --
        the original chain skipped straight to ~/.cache and would reject a cache that is
        only reachable through XDG_CACHE_HOME."""
        xdg = tmp_path / "xdg_cache"
        cache_root = xdg / "huggingface" / "hub"
        self._build_complete_cache(cache_root)

        result = self._run_wrapper(tmp_path / "run_xdg_isolated.py", f"""
            import os, sys
            sys.version_info = (3, 12, 0, "final", 0)
            os.environ.pop("HF_HUB_CACHE", None)
            os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
            os.environ.pop("HF_HOME", None)
            os.environ["XDG_CACHE_HOME"] = r"{xdg}"

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "server_under_test", r"{SERVER_PY}"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod._preflight_check()
            print("PREFLIGHT_OK")
            """)

        assert result.returncode == 0
        assert "ERROR" not in result.stderr
        assert "Traceback" not in result.stderr
        assert "PREFLIGHT_OK" in result.stdout

    def test_nested_var_pointing_at_tilde_matches_library_order_not_falsely_accepted(
        self, tmp_path, monkeypatch,
    ):
        """PR #80 review (Anmol), round 2, the core finding: expandvars-before-expanduser
        (the bug in round 1) would resolve HF_HUB_CACHE=$CACHE_ROOT with
        CACHE_ROOT=~/hf-cache by substituting $CACHE_ROOT into a literal "~/hf-cache"
        string and THEN expanding that leftover "~" -- accepting a cache location the
        real huggingface_hub loader does not resolve the same way. The real loader's
        expanduser() already ran BEFORE the "$CACHE_ROOT" substitution happened (there is
        no leading "~" in the literal string "$CACHE_ROOT" for it to act on), so
        expandvars() then substitutes in a literal, never-expanded "~" that Path treats
        as a directory named "~", not the home directory.

        A complete cache placed at the REAL expanded location must therefore NOT be
        found through this nested pattern -- if it were, preflight would be accepting a
        cache the actual model loader would not."""
        fake_home = tmp_path / "fake_home"
        real_expanded_root = fake_home / "hf-cache"
        self._build_complete_cache(real_expanded_root)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("USERPROFILE", str(fake_home))

        result = self._run_wrapper(tmp_path / "run_nested_var_isolated.py", f"""
            import os, sys
            sys.version_info = (3, 12, 0, "final", 0)
            os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
            os.environ.pop("HF_HOME", None)
            os.environ["HOME"] = r"{fake_home}"
            os.environ["USERPROFILE"] = r"{fake_home}"
            os.environ["CACHE_ROOT"] = "~/hf-cache"
            os.environ["HF_HUB_CACHE"] = "$CACHE_ROOT"

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "server_under_test", r"{SERVER_PY}"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod._preflight_check()
            print("PREFLIGHT_OK")
            """)

        assert result.returncode == 1, (
            "nested $CACHE_ROOT (pointing at ~/hf-cache) was falsely accepted -- this "
            "means expand order regressed back to expandvars-before-expanduser"
        )
        assert "PREFLIGHT_OK" not in result.stdout
        assert "Traceback" not in result.stderr

    def test_complete_cache_via_hf_home_passes_silently(self, tmp_path):
        """HF_HOME (not HF_HUB_CACHE) is the more commonly-set env var in
        practice -- confirm the HF_HOME/hub fallback resolution works too."""
        hf_home = tmp_path / "hf_home"
        hub_cache = hf_home / "hub"
        self._build_complete_cache(hub_cache)

        wrapper = tmp_path / "run_hf_home_isolated.py"
        wrapper.write_text(
            textwrap.dedent(
                f"""
                import os, sys
                sys.version_info = (3, 12, 0, "final", 0)
                os.environ.pop("HF_HUB_CACHE", None)
                os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
                os.environ["HF_HOME"] = r"{hf_home}"

                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "server_under_test", r"{SERVER_PY}"
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod._preflight_check()
                print("PREFLIGHT_OK")
                """
            )
        )
        result = subprocess.run(
            [sys.executable, str(wrapper)], capture_output=True, text=True, timeout=5.0,
        )

        assert result.returncode == 0
        assert "ERROR" not in result.stderr
        assert "PREFLIGHT_OK" in result.stdout


class TestPortConflictHandling:
    def test_port_already_in_use_gives_clear_message_not_traceback(self, monkeypatch):
        """
        B3 requirement: 'friendly port-8793 conflict handling... do not
        automatically kill any owning process.' Rather than racing a real
        OS-level port conflict (unreliable on Windows -- a wildcard "" bind
        and a specific 127.0.0.1 bind are not always treated as conflicting,
        even with allow_reuse_address disabled), mocks ThreadingHTTPServer's
        constructor to raise the exact OSError a real port conflict would,
        and confirms serve() catches it cleanly rather than letting it
        propagate as a raw traceback.
        """
        sys.path.insert(0, str(VIEWER_DIR))
        import server as server_module
        import pytest

        def _raise_address_in_use(*args, **kwargs):
            raise OSError(10048, "Only one usage of each socket address is normally permitted")

        monkeypatch.setattr(server_module, "ThreadingHTTPServer", _raise_address_in_use)

        with pytest.raises(SystemExit) as exc_info:
            server_module.serve(port=8793)

        assert exc_info.value.code == 1


class TestBindAddress:
    def test_serve_binds_localhost_only_not_all_interfaces(self, monkeypatch):
        """
        #87: serve() must bind 127.0.0.1, not "" (all interfaces). /api/upload
        (spawns run_pipeline.py) and /api/upload-reference have no auth -- binding ""
        would expose both to anyone else on the same network, a real risk on venue WiFi
        during the demo. Mocks ThreadingHTTPServer's constructor to capture the exact
        address tuple passed, rather than actually binding a socket.
        """
        sys.path.insert(0, str(VIEWER_DIR))
        import server as server_module

        captured = {}

        class _FakeServer:
            def __init__(self, addr, handler):
                captured["addr"] = addr
            def serve_forever(self):
                raise KeyboardInterrupt  # exit serve() cleanly right after bind
            def shutdown(self):
                pass

        monkeypatch.setattr(server_module, "ThreadingHTTPServer", _FakeServer)
        server_module.serve(port=8793)

        assert captured["addr"] == ("127.0.0.1", 8793), (
            f"expected a localhost-only bind, got {captured['addr']!r}"
        )