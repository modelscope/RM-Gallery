# -*- coding: utf-8 -*-

import pytest

from openjudge.harness.codex import CodexHarness


@pytest.mark.unit
class TestCodexHarness:
    def test_default_binary_is_codex(self):
        harness = CodexHarness()
        assert harness.binary == "codex"

    def test_build_command_uses_exec_subcommand(self, tmp_path):
        harness = CodexHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"

    def test_build_command_always_overrides_sandbox_to_workspace_write(self, tmp_path):
        harness = CodexHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert "--sandbox" in cmd
        assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"

    def test_build_command_uses_ask_for_approval_never_not_force(self, tmp_path):
        harness = CodexHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert "--ask-for-approval" in cmd
        assert cmd[cmd.index("--ask-for-approval") + 1] == "never"
        assert "--force" not in cmd
        assert "--trust" not in cmd

    def test_build_command_uses_json_flag_not_output_format(self, tmp_path):
        harness = CodexHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert "--json" in cmd
        assert "--output-format" not in cmd

    def test_build_command_prompt_is_trailing_positional_argument(self, tmp_path):
        harness = CodexHarness()
        cmd = harness.build_command(tmp_path, "judge this", None)
        assert cmd[-1] == "judge this"

    def test_build_command_with_model_includes_model_flag(self, tmp_path):
        harness = CodexHarness()
        cmd = harness.build_command(tmp_path, "judge this", "gpt-5-codex")
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "gpt-5-codex"
        assert cmd[-1] == "judge this"
