# -*- coding: utf-8 -*-
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from openjudge.harness import base as base_module
from openjudge.harness.base import RESULT_FILENAME, SPEC_FILENAME, BaseHarness
from openjudge.harness.sandbox import ProcessSandbox


class DummyHarness(BaseHarness):
    """Minimal concrete BaseHarness for exercising run()'s shared protocol."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_command_calls: List[tuple] = []

    @property
    def default_binary(self) -> str:
        return "dummy-cli"

    def build_command(self, sandbox_dir: Path, prompt: str, model: Optional[str]) -> List[str]:
        self.build_command_calls.append((sandbox_dir, prompt, model))
        return [self.binary, "-p", prompt]


def _write_result(cwd: str, payload: Dict[str, Any]) -> None:
    Path(cwd, RESULT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
class TestBaseHarnessSuccess:
    def test_successful_run_reads_back_result_file(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            _write_result(cwd, {"c1": {"passed": True, "reason": "ok"}})
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="done", stderr="")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="judge this", schema={"dimensions": {"c1": {}}})

        assert result.available is True
        assert result.result == {"c1": {"passed": True, "reason": "ok"}}
        assert result.exit_code == 0
        assert result.raw_stdout == "done"

    def test_writes_spec_file_with_instructions_and_schema(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            _write_result(cwd, {"c1": {"passed": True, "reason": "ok"}})
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness()
        harness.run(tmp_path, prompt="judge this", schema={"dimensions": {"c1": {}}})

        spec = json.loads((tmp_path / SPEC_FILENAME).read_text(encoding="utf-8"))
        assert spec == {"instructions": "judge this", "output_schema": {"dimensions": {"c1": {}}}}

    def test_removes_stale_result_file_before_running(self, tmp_path, monkeypatch):
        (tmp_path / RESULT_FILENAME).write_text(
            json.dumps({"c1": {"passed": False, "reason": "stale"}}), encoding="utf-8"
        )

        def fake_run(cmd, cwd, timeout, cancel_event=None):
            # Simulate a CLI that exits cleanly but crashes before writing a fresh result file.
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="judge this", schema={})

        assert result.available is False

    def test_passes_sandbox_dir_prompt_and_model_to_build_command(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            _write_result(cwd, {"c1": {"passed": True, "reason": "ok"}})
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness()
        harness.run(tmp_path, prompt="judge this", schema={}, model="gpt-5")

        assert harness.build_command_calls == [(tmp_path, "judge this", "gpt-5")]


@pytest.mark.unit
class TestBaseHarnessFailureGates:
    @pytest.mark.parametrize("payload", [[], None, "not an object", 1])
    def test_non_object_json_result_is_unavailable(self, tmp_path, monkeypatch, payload):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            Path(cwd, RESULT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        assert DummyHarness().run(tmp_path, prompt="p", schema={}).available is False

    def test_nonzero_returncode_rejects_even_with_valid_result_file(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            _write_result(cwd, {"c1": {"passed": True, "reason": "ok"}})
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False
        assert result.exit_code == 1
        assert result.raw_stderr == "boom"

    def test_file_not_found_error_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            raise FileNotFoundError("dummy-cli: command not found")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False

    def test_os_error_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            raise OSError("permission denied")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False

    def test_missing_result_file_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="no result written", stderr="")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False
        assert result.raw_stdout == "no result written"

    def test_malformed_json_result_file_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            Path(cwd, RESULT_FILENAME).write_text("{not valid json", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False

    def test_spec_write_oserror_normalised_to_unavailable(self, tmp_path, monkeypatch):
        """OSError raised during spec-file write (before launching the subprocess) must not escape run()."""
        nonexistent_sandbox = tmp_path / "does_not_exist"
        harness = DummyHarness()
        # sandbox_dir does not exist → write_text raises FileNotFoundError (subclass of OSError)
        result = harness.run(nonexistent_sandbox, prompt="p", schema={})

        assert result.available is False


@pytest.mark.unit
class TestBaseHarnessTimeout:
    def test_timeout_with_result_file_already_written_is_rescued(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            _write_result(cwd, {"c1": {"passed": True, "reason": "finished just before being killed"}})
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness(timeout_s=1)
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is True
        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.result == {"c1": {"passed": True, "reason": "finished just before being killed"}}

    def test_timeout_without_result_file_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, timeout, cancel_event=None):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(base_module, "_run_command", fake_run)
        harness = DummyHarness(timeout_s=1)
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False
        assert result.timed_out is True


class PythonHarness(BaseHarness):
    """Run local Python scripts through the same process lifecycle as agent CLIs."""

    @property
    def default_binary(self) -> str:
        return sys.executable

    def build_command(self, sandbox_dir: Path, prompt: str, model: Optional[str]) -> List[str]:
        return [self.binary, "-c", prompt]


@pytest.mark.unit
class TestHarnessProcessLifecycle:
    @pytest.mark.parametrize("parent_input", [None, "another evaluation row"], ids=["open-pipe", "piped-data"])
    def test_cli_does_not_read_parent_stdin(self, tmp_path, parent_input):
        program = (
            "import json,sys\n"
            "from pathlib import Path\n"
            "from openjudge.harness.base import _run_command\n"
            "result = _run_command([sys.executable, '-c', 'import sys; print(repr(sys.stdin.read()))'], "
            "Path(sys.argv[1]), 1)\n"
            "remaining = sys.stdin.read() if sys.argv[2] == 'read' else ''\n"
            "print(json.dumps([result.stdout.strip(), remaining]))\n"
        )
        cmd = [sys.executable, "-c", program, str(tmp_path), "read" if parent_input is not None else "skip"]
        with subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as p:
            try:
                if parent_input is None:
                    # Keep the input pipe open: the judge must still finish without waiting for EOF.
                    p.wait(timeout=5)
                stdout, stderr = p.communicate(input=parent_input, timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                pytest.fail("The CLI waited on its caller's stdin")
        assert p.returncode == 0, stderr
        assert json.loads(stdout) == ["''", parent_input or ""]

    @pytest.mark.parametrize("exit_code", [0, 2])
    def test_real_process_result_protocol_and_failure_gate(self, tmp_path, exit_code):
        payload = {"c1": {"passed": True, "reason": "local test"}}
        script = (
            "import pathlib,sys; "
            f"pathlib.Path({RESULT_FILENAME!r}).write_text({json.dumps(payload)!r}); "
            "print('stdout evidence'); print('stderr evidence', file=sys.stderr); "
            f"sys.exit({exit_code})"
        )
        result = PythonHarness(timeout_s=5).run(tmp_path, script, {})
        assert result.available is (exit_code == 0)
        assert result.exit_code == exit_code
        assert result.raw_stdout.strip() == "stdout evidence"
        assert result.raw_stderr.strip() == "stderr evidence"
        assert result.result == (payload if exit_code == 0 else {})

    def test_real_timeout_preserves_completed_result_and_logs(self, tmp_path):
        payload = {"c1": {"passed": False, "reason": "verified"}}
        script = (
            "import pathlib,time; "
            f"pathlib.Path({RESULT_FILENAME!r}).write_text({json.dumps(payload)!r}); "
            "print('before timeout', flush=True); time.sleep(30)"
        )
        result = PythonHarness(timeout_s=1).run(tmp_path, script, {})
        assert result.available is True
        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.result == payload
        assert result.raw_stdout.strip() == "before timeout"

    @pytest.mark.skipif(os.name != "posix", reason="POSIX process group lifecycle")
    @pytest.mark.parametrize("parent_exits", [False, True], ids=["running-parent", "exited-parent"])
    @pytest.mark.parametrize("detached", [False, True], ids=["same-session", "detached-session"])
    def test_stops_descendants_before_sandbox_cleanup(self, tmp_path, parent_exits, detached):
        heartbeat = tmp_path / "heartbeat.txt"
        pid_file = tmp_path / "child.pid"
        child_script = (
            "import pathlib,time\n"
            f"heartbeat = pathlib.Path({str(heartbeat)!r})\n"
            "for tick in range(3000):\n"
            "    heartbeat.write_text(str(tick))\n"
            "    time.sleep(0.01)\n"
        )
        script = (
            "import pathlib,subprocess,sys,time\n"
            f"child = subprocess.Popen([sys.executable, '-c', {child_script!r}], start_new_session={detached})\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
            f"while not pathlib.Path({str(heartbeat)!r}).exists(): time.sleep(0.01)\n"
            + ("time.sleep(0.3)" if parent_exits else "time.sleep(30)")
        )
        cleanup_needed = True
        try:
            with ProcessSandbox() as sandbox_dir:
                result = PythonHarness(timeout_s=1).run(sandbox_dir, script, {})
                assert result.timed_out is (not parent_exits)
                assert result.available is False
                assert result.duration < 5, "Output collection must not wait for the child to close inherited handles"
                assert heartbeat.exists(), "The child must have started before the timeout"
                last_write = heartbeat.stat().st_mtime_ns
            assert not sandbox_dir.exists()
            time.sleep(0.15)
            assert heartbeat.stat().st_mtime_ns == last_write, "Child still writes after sandbox cleanup"
            cleanup_needed = False
        finally:
            # A regression must not leave the deliberately long-running child alive.
            if cleanup_needed and pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
