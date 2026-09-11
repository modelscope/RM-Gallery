# -*- coding: utf-8 -*-
"""Harness layer for wrapping external coding-agent CLIs as agent-as-judge backends.

Each harness (Claude Code / Codex / Cursor CLI) implements the same
sandboxed, file-based judging protocol (see `openjudge.harness.base`) --
only the CLI-specific command-line flags differ, encapsulated in each
harness's `build_command()`.
"""
from openjudge.harness.base import BaseHarness, HarnessResult
from openjudge.harness.claude_code import ClaudeCodeHarness
from openjudge.harness.codex import CodexHarness
from openjudge.harness.cursor_agent import CursorAgentHarness
from openjudge.harness.sandbox import ProcessSandbox

__all__ = [
    "BaseHarness",
    "HarnessResult",
    "ProcessSandbox",
    "ClaudeCodeHarness",
    "CodexHarness",
    "CursorAgentHarness",
]
