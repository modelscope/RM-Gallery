# -*- coding: utf-8 -*-
import pytest


@pytest.mark.unit
class TestHarnessPackageExports:
    def test_top_level_imports_resolve(self):
        from openjudge.harness import (
            BaseHarness,
            ClaudeCodeHarness,
            CodexHarness,
            CursorAgentHarness,
            HarnessResult,
            ProcessSandbox,
        )

        assert ClaudeCodeHarness().binary == "claude"
        assert CodexHarness().binary == "codex"
        assert CursorAgentHarness().binary == "cursor-agent"
        assert issubclass(ClaudeCodeHarness, BaseHarness)
        assert HarnessResult().available is False
        assert ProcessSandbox is not None
