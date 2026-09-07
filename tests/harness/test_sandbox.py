# -*- coding: utf-8 -*-
import json
import shutil

import pytest

from openjudge.harness.sandbox import ProcessSandbox


@pytest.mark.unit
class TestProcessSandboxNoEvidence:
    def test_creates_and_removes_empty_sandbox(self):
        with ProcessSandbox() as sandbox_dir:
            assert sandbox_dir.is_dir()
            created_path = sandbox_dir
        assert not created_path.exists()

    def test_keep_on_exit_preserves_directory(self):
        with ProcessSandbox(keep_on_exit=True) as sandbox_dir:
            created_path = sandbox_dir
        assert created_path.exists()
        shutil.rmtree(created_path, ignore_errors=True)


@pytest.mark.unit
class TestProcessSandboxWorkspaceCopy:
    @pytest.mark.parametrize("is_file", [False, True], ids=["missing", "file"])
    def test_rejects_invalid_workspace_and_cleans_up(self, tmp_path, is_file):
        candidate = tmp_path / "candidate"
        if is_file:
            candidate.write_text("not a directory", encoding="utf-8")
        sandbox = ProcessSandbox(workspace_path=str(candidate))
        error_type = NotADirectoryError if is_file else FileNotFoundError
        with pytest.raises(error_type, match="workspace path"):
            with sandbox:
                pytest.fail("Invalid workspace must not be exposed to the judge")
        assert sandbox.sandbox_dir is not None
        assert not sandbox.sandbox_dir.exists()

    def test_copies_workspace_files_into_workspace_subdir(self, tmp_path):
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        (candidate / "answer.txt").write_text("42", encoding="utf-8")
        nested = candidate / "nested"
        nested.mkdir()
        (nested / "note.md").write_text("hello", encoding="utf-8")

        with ProcessSandbox(workspace_path=str(candidate)) as sandbox_dir:
            copied = sandbox_dir / "workspace"
            assert (copied / "answer.txt").read_text(encoding="utf-8") == "42"
            assert (copied / "nested" / "note.md").read_text(encoding="utf-8") == "hello"

    def test_skips_symlinks_without_dereferencing_or_preserving(self, tmp_path):
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        target = tmp_path / "outside_secret.txt"
        target.write_text("should not leak", encoding="utf-8")
        (candidate / "link_to_outside").symlink_to(target)

        with ProcessSandbox(workspace_path=str(candidate)) as sandbox_dir:
            copied = sandbox_dir / "workspace"
            assert not (copied / "link_to_outside").exists()
            assert not (copied / "link_to_outside").is_symlink()

    def test_reports_number_of_symlinks_skipped(self, tmp_path):
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        (candidate / "real.txt").write_text("ok", encoding="utf-8")
        (candidate / "link1").symlink_to(candidate / "real.txt")
        (candidate / "link2").symlink_to(candidate / "real.txt")

        sandbox = ProcessSandbox(workspace_path=str(candidate))
        with sandbox:
            pass
        assert sandbox.symlinks_skipped == 2


@pytest.mark.unit
class TestProcessSandboxTranscript:
    def test_materializes_transcript_from_message_list(self):
        transcript = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        with ProcessSandbox(transcript=transcript) as sandbox_dir:
            lines = (sandbox_dir / "transcript.jsonl").read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 2
            assert json.loads(lines[0]) == transcript[0]
            assert json.loads(lines[1]) == transcript[1]

    def test_materializes_transcript_from_existing_jsonl_path(self, tmp_path):
        src = tmp_path / "existing_transcript.jsonl"
        src.write_text('{"role": "user", "content": "hi"}\n', encoding="utf-8")

        with ProcessSandbox(transcript=str(src)) as sandbox_dir:
            dest = sandbox_dir / "transcript.jsonl"
            assert dest.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")

    def test_raises_file_not_found_for_missing_transcript_path(self):
        with pytest.raises(FileNotFoundError, match="transcript path not found"):
            with ProcessSandbox(transcript="/nonexistent/path/transcript.jsonl"):
                pass

    def test_cleans_up_sandbox_dir_when_enter_raises(self):
        sandbox = ProcessSandbox(transcript="/nonexistent/path/transcript.jsonl")
        with pytest.raises(FileNotFoundError, match="transcript path not found"):
            with sandbox:
                pass
        assert sandbox.sandbox_dir is not None
        assert not sandbox.sandbox_dir.exists()

    def test_both_workspace_and_transcript_can_be_given_together(self, tmp_path):
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        (candidate / "out.txt").write_text("result", encoding="utf-8")
        transcript = [{"role": "user", "content": "go"}]

        with ProcessSandbox(workspace_path=str(candidate), transcript=transcript) as sandbox_dir:
            assert (sandbox_dir / "workspace" / "out.txt").exists()
            assert (sandbox_dir / "transcript.jsonl").exists()
