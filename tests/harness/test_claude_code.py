# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from openjudge.harness.claude_code import ClaudeCodeHarness


@pytest.mark.unit
class TestClaudeCodeHarness:
    def test_default_binary_is_claude(self):
        harness = ClaudeCodeHarness()
        assert harness.binary == "claude"

    def test_custom_binary_override(self):
        harness = ClaudeCodeHarness(binary="/usr/local/bin/claude")
        assert harness.binary == "/usr/local/bin/claude"

    def test_build_command_uses_bypass_permissions_not_force_or_trust(self, tmp_path):
        harness = ClaudeCodeHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)

        assert cmd[0] == "claude"
        assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "judge this"
        assert "--permission-mode" in cmd
        assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"
        assert "--force" not in cmd
        assert "--trust" not in cmd

    def test_build_command_without_model_omits_model_flag(self, tmp_path):
        harness = ClaudeCodeHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert "--model" not in cmd

    def test_build_command_with_model_includes_model_flag(self, tmp_path):
        harness = ClaudeCodeHarness()
        cmd = harness.build_command(tmp_path, "judge this", "claude-opus-4")
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "claude-opus-4"

    def test_build_command_uses_stream_json_with_verbose(self, tmp_path):
        harness = ClaudeCodeHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert "--output-format" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in cmd
