# -*- coding: utf-8 -*-

import pytest

from openjudge.harness.cursor_agent import CursorAgentHarness


@pytest.mark.unit
class TestCursorAgentHarness:
    def test_default_binary_is_cursor_agent(self):
        harness = CursorAgentHarness()
        assert harness.binary == "cursor-agent"

    def test_build_command_uses_force_flag(self, tmp_path):
        harness = CursorAgentHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert cmd[0] == "cursor-agent"
        assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "judge this"
        assert "--force" in cmd

    def test_build_command_uses_stream_json_output_format(self, tmp_path):
        harness = CursorAgentHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert "--output-format" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "stream-json"

    def test_build_command_without_model_omits_model_flag(self, tmp_path):
        harness = CursorAgentHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert "--model" not in cmd

    def test_build_command_with_model_includes_model_flag(self, tmp_path):
        harness = CursorAgentHarness()
        cmd = harness.build_command(tmp_path, "judge this", "sonnet-4.5")
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "sonnet-4.5"
