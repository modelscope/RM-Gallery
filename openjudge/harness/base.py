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
"""
import json
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = ["HarnessResult", "BaseHarness", "RESULT_FILENAME", "SPEC_FILENAME"]

RESULT_FILENAME = "_judge_result.json"
SPEC_FILENAME = "_judge_spec.json"


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
        timeout_s: Per-invocation subprocess timeout in seconds.
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
            The argv list to pass to `subprocess.run`, e.g. `["claude", "-p", prompt, ...]`.
        """

    def run(
        self,
        sandbox_dir: Path,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
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
            proc = subprocess.run(
                cmd,
                cwd=str(sandbox_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
        except (FileNotFoundError, OSError):
            return HarnessResult(available=False, duration=time.time() - start)

        duration = time.time() - start

        if proc is not None and proc.returncode != 0:
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

        return HarnessResult(
            available=True,
            result=parsed,
            raw_stdout=(proc.stdout if proc else "") or "",
            raw_stderr=(proc.stderr if proc else "") or "",
            exit_code=(proc.returncode if proc else -1),
            timed_out=timed_out,
            duration=duration,
        )
