# -*- coding: utf-8 -*-
"""OpenAI Codex CLI harness (`codex exec ...`)."""
from pathlib import Path
from typing import List, Optional

from openjudge.harness.base import BaseHarness

__all__ = ["CodexHarness"]


class CodexHarness(BaseHarness):
    """Runs the Codex CLI (`codex exec`) non-interactively as a judge.

    Requires the `codex` CLI to be installed and authenticated (e.g. via the
    `OPENAI_API_KEY` environment variable or a ChatGPT login, inherited from
    the current process environment) in the current environment; this class
    only shells out to it.

    Codex's command structure differs from Claude Code/Cursor CLI in three
    ways: it requires the `exec` subcommand, uses `--json` instead of
    `--output-format`, and uses `--sandbox`/`--ask-for-approval` instead of a
    single force/trust flag. Codex's `exec` sandbox defaults to read-only,
    which would prevent the agent from writing `_judge_result.json` into the
    sandbox directory -- `--sandbox workspace-write` is therefore always
    passed explicitly and is not optional. The approval option is global and
    must precede `exec`; the temporary sandbox is not a Git repository, so
    `--skip-git-repo-check` is required as well.
    """

    @property
    def default_binary(self) -> str:
        return "codex"

    def build_command(self, sandbox_dir: Path, prompt: str, model: Optional[str]) -> List[str]:
        cmd = [
            self.binary,
            "--ask-for-approval",
            "never",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
        ]
        if model:
            cmd += ["--model", model]
        cmd.append(prompt)
        return cmd
