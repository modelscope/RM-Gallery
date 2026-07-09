# -*- coding: utf-8 -*-
"""Cursor CLI agent harness (`cursor-agent -p ...`)."""
from pathlib import Path
from typing import List, Optional

from openjudge.harness.base import BaseHarness

__all__ = ["CursorAgentHarness"]


class CursorAgentHarness(BaseHarness):
    """Runs the Cursor CLI agent (`cursor-agent`) non-interactively as a judge.

    Requires the `cursor-agent` CLI to be installed and authenticated (e.g.
    via the `CURSOR_API_KEY` environment variable or an interactive login,
    inherited from the current process environment) in the current
    environment; this class only shells out to it.
    """

    @property
    def default_binary(self) -> str:
        return "cursor-agent"

    def build_command(self, sandbox_dir: Path, prompt: str, model: Optional[str]) -> List[str]:
        cmd = [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--force",
        ]
        if model:
            cmd += ["--model", model]
        return cmd
