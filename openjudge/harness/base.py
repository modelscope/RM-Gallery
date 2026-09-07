# -*- coding: utf-8 -*-
"""Base abstraction for external coding-agent CLI harnesses.

A harness wraps a specific CLI (Claude Code, Codex, Cursor CLI, ...) so that
AgenticGrader can hand it a judging task without knowing which concrete CLI
is behind it. The execution protocol is harness-agnostic: write a spec file
into the sandbox, invoke the CLI non-interactively, and read back a result
file the CLI's agent loop is instructed to write. This never depends on
parsing any CLI's own `--output-format`/`--json` stdout schema (the three
CLIs format that output differently and that surface is far more prone to
change than a small on-disk JSON contract).

Failure gate policy (never raises -- callers get `HarnessResult(available=False)`):
    - Clean exit but `returncode != 0` -> rejected even if the result file parses.
    - `FileNotFoundError`/`OSError` (binary missing, permissions, ...) -> rejected.
    - `subprocess.TimeoutExpired` -> NOT auto-rejected: the subprocess may have
      finished writing the result file just before being killed, so the result
      file is still checked; only rejected if that file is missing/unparsable.
    - Missing or unparsable result file -> rejected.
    - Explicit cancellation -> rejected, even if a result file was already written.
"""
import json
import os
import signal
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional

import psutil
from pydantic import BaseModel, Field

__all__ = ["HarnessResult", "BaseHarness", "RESULT_FILENAME", "SPEC_FILENAME"]

RESULT_FILENAME = "_judge_result.json"
SPEC_FILENAME = "_judge_spec.json"


def _remember_descendants(parent: Optional[psutil.Process], descendants: Dict[int, psutil.Process]) -> None:
    """Retain process identities so detached children can be stopped after reparenting."""
    if parent is not None:
        try:
            descendants.update((child.pid, child) for child in parent.children(recursive=True))
        except psutil.Error:
            pass


def _terminate_process_tree(process: subprocess.Popen, descendants: Dict[int, psutil.Process]) -> None:
    """Stop the CLI's process group and observed descendants, with bounded waits."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in descendants.values():
        try:
            child.kill()
        except psutil.Error:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=1)
    psutil.wait_procs(list(descendants.values()), timeout=1)


def _run_command(
    cmd: List[str], sandbox_dir: Path, timeout_s: float, cancel_event: Optional[Event] = None
) -> subprocess.CompletedProcess:
    """Run a CLI with isolated stdin, cancellable waits, and file-backed output.

    Regular files avoid waiting for EOF on pipes inherited by detached children.
    Only the output already written at cleanup is read back.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError()
    deadline = time.monotonic() + timeout_s
    descendants: Dict[int, psutil.Process] = {}
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        # Manage cleanup explicitly: Popen.__exit__ waits without a timeout.
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            cmd,
            cwd=str(sandbox_dir),
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=os.name == "posix",
        )
        try:
            parent = psutil.Process(process.pid)
        except psutil.Error:
            parent = None
        timed_out = False
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError()
                _remember_descendants(parent, descendants)
                remaining = deadline - time.monotonic()
                try:
                    process.wait(timeout=max(0, min(0.1, remaining)))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
        finally:
            _remember_descendants(parent, descendants)
            _terminate_process_tree(process, descendants)

        stdout_size = os.fstat(stdout_file.fileno()).st_size
        stderr_size = os.fstat(stderr_file.fileno()).st_size
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(stdout_size).decode("utf-8", errors="replace")
        stderr = stderr_file.read(stderr_size).decode("utf-8", errors="replace")
        if timed_out:
            raise subprocess.TimeoutExpired(cmd, timeout_s, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


class HarnessResult(BaseModel):
    """Outcome of a single harness invocation.

    Attributes:
        available: Whether a trustworthy result was produced. When False,
            `result` is empty and callers must not treat this sample as a vote.
        result: Parsed contents of the sandbox's `_judge_result.json` (empty if unavailable).
        raw_stdout: Captured subprocess stdout, for optional trajectory logging only --
            never used to derive the verdict.
        raw_stderr: Captured subprocess stderr.
        exit_code: Subprocess exit code, or -1 if the process never produced one
            (e.g. it timed out, or failed to start).
        timed_out: Whether the subprocess hit `timeout_s`.
        duration: Wall-clock seconds spent in this invocation.
    """

    available: bool = False
    result: Dict[str, Any] = Field(default_factory=dict)
    raw_stdout: str = ""
    raw_stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    duration: float = 0.0


class BaseHarness(ABC):
    """Abstract base for a specific coding-agent CLI harness.

    Subclasses implement only `default_binary` and `build_command`; the
    shared sandboxed-execution + file-result-protocol + failure-gate logic
    lives in `run()` and is identical across all harnesses.

    Attributes:
        binary: The CLI executable to invoke (defaults to `default_binary`).
        timeout_s: Per-invocation subprocess timeout in seconds. Process cleanup
            has separate bounded waits, so total duration can exceed this limit.
    """

    def __init__(self, binary: Optional[str] = None, timeout_s: float = 90.0):
        self.binary = binary or self.default_binary
        self.timeout_s = timeout_s

    @property
    @abstractmethod
    def default_binary(self) -> str:
        """Default CLI executable name for this harness (e.g. `"claude"`)."""

    @abstractmethod
    def build_command(self, sandbox_dir: Path, prompt: str, model: Optional[str]) -> List[str]:
        """Build the argv list to run this harness non-interactively.

        Args:
            sandbox_dir: The sandbox working directory (also the subprocess cwd).
            prompt: The full instructions text passed to the CLI (also written into
                the sandbox's spec file, so the agent can re-read it from disk too).
            model: Optional model name override for this invocation.

        Returns:
            The argv list to pass to `subprocess.Popen`, e.g. `["claude", "-p", prompt, ...]`.
        """

    def run(
        self,
        sandbox_dir: Path,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        cancel_event: Optional[Event] = None,
    ) -> HarnessResult:
        """Run this harness once against an already-built sandbox.

        Writes the spec file, invokes the CLI via `build_command()`, and reads
        back the result file. Never raises -- any failure path is normalized
        into `HarnessResult(available=False)` per the module-level gate policy.

        Args:
            sandbox_dir: An existing sandbox directory (e.g. from `ProcessSandbox`),
                used as both the spec/result file location and the subprocess cwd.
            prompt: Full judging instructions to write into the spec file and pass
                to the CLI.
            schema: The `output_schema` to write into the spec file, describing the
                shape the agent must write to `_judge_result.json`.
            model: Optional model name override for this invocation.
            cancel_event: Per-invocation cancellation signal. Overrides of `run()`
                must accept this argument and stop promptly when it is set.

        Returns:
            A `HarnessResult`; check `.available` before trusting `.result`.
        """
        spec_path = sandbox_dir / SPEC_FILENAME
        result_path = sandbox_dir / RESULT_FILENAME
        cmd = self.build_command(sandbox_dir, prompt, model)
        start = time.time()
        proc: Optional[subprocess.CompletedProcess] = None
        timed_out = False
        try:
            spec_path.write_text(
                json.dumps({"instructions": prompt, "output_schema": schema}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if result_path.exists():
                result_path.unlink()
            proc = _run_command(cmd, sandbox_dir, self.timeout_s, cancel_event=cancel_event)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            proc = subprocess.CompletedProcess(cmd, -1, stdout=exc.stdout, stderr=exc.stderr)
        except (CancelledError, FileNotFoundError, OSError):
            return HarnessResult(available=False, duration=time.time() - start)

        duration = time.time() - start

        if proc is not None and proc.returncode != 0 and not timed_out:
            return HarnessResult(
                available=False,
                raw_stdout=proc.stdout or "",
                raw_stderr=proc.stderr or "",
                exit_code=proc.returncode,
                duration=duration,
            )

        if not result_path.exists():
            return HarnessResult(
                available=False,
                raw_stdout=(proc.stdout if proc else "") or "",
                raw_stderr=(proc.stderr if proc else "") or "",
                timed_out=timed_out,
                duration=duration,
            )

        try:
            parsed = json.loads(result_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return HarnessResult(available=False, timed_out=timed_out, duration=duration)
        if not isinstance(parsed, dict):
            return HarnessResult(available=False, timed_out=timed_out, duration=duration)

        return HarnessResult(
            available=True,
            result=parsed,
            raw_stdout=(proc.stdout if proc else "") or "",
            raw_stderr=(proc.stderr if proc else "") or "",
            exit_code=(proc.returncode if proc else -1),
            timed_out=timed_out,
            duration=duration,
        )
