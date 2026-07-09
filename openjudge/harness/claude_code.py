# -*- coding: utf-8 -*-
"""Claude Code CLI harness (`claude -p ...`)."""
from pathlib import Path
from typing import List, Optional

from openjudge.harness.base import BaseHarness

__all__ = ["ClaudeCodeHarness"]


class ClaudeCodeHarness(BaseHarness):
    """Runs the Claude Code CLI (`claude`) non-interactively as a judge.

    Requires the `claude` CLI to be installed and authenticated (e.g. via the
    `ANTHROPIC_API_KEY` environment variable, inherited from the current
    process environment) in the current environment; this class only shells
    out to it.

    Claude Code has no `--force`/`--trust` flag like Cursor CLI does --
    non-interactive auto-approval is `--permission-mode bypassPermissions`
    instead. `--output-format stream-json` additionally requires `--verbose`
    when combined with `-p`, or the CLI rejects the flag combination.
    """

    @property
    def default_binary(self) -> str:
        return "claude"

    def build_command(self, sandbox_dir: Path, prompt: str, model: Optional[str]) -> List[str]:
        cmd = [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
        ]
        if model:
            cmd += ["--model", model]
        return cmd
