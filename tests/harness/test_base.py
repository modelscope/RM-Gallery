# -*- coding: utf-8 -*-
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from openjudge.harness import base as base_module
from openjudge.harness.base import BaseHarness, RESULT_FILENAME, SPEC_FILENAME


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
        def fake_run(cmd, cwd, capture_output, text, timeout):
            _write_result(cwd, {"c1": {"passed": True, "reason": "ok"}})
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="done", stderr="")

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="judge this", schema={"dimensions": {"c1": {}}})

        assert result.available is True
        assert result.result == {"c1": {"passed": True, "reason": "ok"}}
        assert result.exit_code == 0
        assert result.raw_stdout == "done"

    def test_writes_spec_file_with_instructions_and_schema(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, capture_output, text, timeout):
            _write_result(cwd, {"c1": {"passed": True, "reason": "ok"}})
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness()
        harness.run(tmp_path, prompt="judge this", schema={"dimensions": {"c1": {}}})

        spec = json.loads((tmp_path / SPEC_FILENAME).read_text(encoding="utf-8"))
        assert spec == {"instructions": "judge this", "output_schema": {"dimensions": {"c1": {}}}}

    def test_removes_stale_result_file_before_running(self, tmp_path, monkeypatch):
        (tmp_path / RESULT_FILENAME).write_text(
            json.dumps({"c1": {"passed": False, "reason": "stale"}}), encoding="utf-8"
        )

        def fake_run(cmd, cwd, capture_output, text, timeout):
            # Simulate a CLI that exits cleanly but crashes before writing a fresh result file.
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="judge this", schema={})

        assert result.available is False

    def test_passes_sandbox_dir_prompt_and_model_to_build_command(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, capture_output, text, timeout):
            _write_result(cwd, {"c1": {"passed": True, "reason": "ok"}})
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness()
        harness.run(tmp_path, prompt="judge this", schema={}, model="gpt-5")

        assert harness.build_command_calls == [(tmp_path, "judge this", "gpt-5")]


@pytest.mark.unit
class TestBaseHarnessFailureGates:
    def test_nonzero_returncode_rejects_even_with_valid_result_file(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, capture_output, text, timeout):
            _write_result(cwd, {"c1": {"passed": True, "reason": "ok"}})
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False
        assert result.exit_code == 1
        assert result.raw_stderr == "boom"

    def test_file_not_found_error_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, capture_output, text, timeout):
            raise FileNotFoundError("dummy-cli: command not found")

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False

    def test_os_error_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, capture_output, text, timeout):
            raise OSError("permission denied")

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False

    def test_missing_result_file_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, capture_output, text, timeout):
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="no result written", stderr="")

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False
        assert result.raw_stdout == "no result written"

    def test_malformed_json_result_file_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, capture_output, text, timeout):
            Path(cwd, RESULT_FILENAME).write_text("{not valid json", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness()
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False

    def test_spec_write_oserror_normalised_to_unavailable(self, tmp_path, monkeypatch):
        """OSError raised during spec-file write (before subprocess.run) must not escape run()."""
        nonexistent_sandbox = tmp_path / "does_not_exist"
        harness = DummyHarness()
        # sandbox_dir does not exist → write_text raises FileNotFoundError (subclass of OSError)
        result = harness.run(nonexistent_sandbox, prompt="p", schema={})

        assert result.available is False


@pytest.mark.unit
class TestBaseHarnessTimeout:
    def test_timeout_with_result_file_already_written_is_rescued(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, capture_output, text, timeout):
            _write_result(cwd, {"c1": {"passed": True, "reason": "finished just before being killed"}})
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness(timeout_s=1)
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is True
        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.result == {"c1": {"passed": True, "reason": "finished just before being killed"}}

    def test_timeout_without_result_file_rejects(self, tmp_path, monkeypatch):
        def fake_run(cmd, cwd, capture_output, text, timeout):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        monkeypatch.setattr(base_module.subprocess, "run", fake_run)
        harness = DummyHarness(timeout_s=1)
        result = harness.run(tmp_path, prompt="p", schema={})

        assert result.available is False
        assert result.timed_out is True
