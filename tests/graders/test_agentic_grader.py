# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from openjudge.graders.agentic_grader import (
    AgenticGrader,
    _aggregate_overall,
    _aggregate_rubric,
    _build_output_schema,
    _build_prompt,
    _normalize_sample,
)
from openjudge.graders.schema import (
    Checkpoint,
    CheckpointResult,
    GraderError,
    GraderMode,
    GraderScore,
    Rubric,
    RubricResult,
)
from openjudge.harness.base import BaseHarness, HarnessResult


class FakeHarness(BaseHarness):
    """Test double that skips subprocess/sandbox execution and returns a pre-baked HarnessResult."""

    def __init__(self, results: List[HarnessResult], **kwargs):
        super().__init__(**kwargs)
        self._results = results
        self.run_calls: List[Dict[str, Any]] = []

    @property
    def default_binary(self) -> str:
        return "fake-cli"

    def build_command(self, sandbox_dir: Path, prompt: str, model: Optional[str]) -> List[str]:
        return [self.binary, prompt]

    def run(
        self, sandbox_dir: Path, prompt: str, schema: Dict[str, Any], model: Optional[str] = None, cancel_event=None
    ) -> HarnessResult:
        idx = len(self.run_calls)
        self.run_calls.append({"sandbox_dir": sandbox_dir, "prompt": prompt, "schema": schema, "model": model})
        return self._results[idx % len(self._results)]


def _rubrics() -> List[Rubric]:
    return [
        Rubric(
            name="correctness",
            weight=1.0,
            checkpoints=[
                Checkpoint(id="c1", description="Answer matches expected value", weight=2.0),
                Checkpoint(id="c2", description="Explanation is clear", weight=1.0),
            ],
        )
    ]


@pytest.mark.unit
class TestAgenticGraderConstruction:
    @pytest.mark.parametrize("mode", [GraderMode.LISTWISE, "listwise", "unknown"])
    def test_rejects_unsupported_modes(self, mode):
        with pytest.raises(ValueError, match="only supports POINTWISE"):
            AgenticGrader(harness=FakeHarness([]), rubrics=_rubrics(), mode=mode)

    def test_requires_harness(self):
        with pytest.raises(ValueError, match="harness is required"):
            AgenticGrader(harness=None, rubrics=_rubrics())

    def test_requires_rubrics(self):
        with pytest.raises(ValueError, match="rubrics is required"):
            AgenticGrader(harness=FakeHarness([]), rubrics=[])

    def test_defaults(self):
        grader = AgenticGrader(harness=FakeHarness([]), rubrics=_rubrics())
        assert grader.name == "agentic_grader"
        assert grader.model is None

    @pytest.mark.parametrize("same_rubric", [False, True])
    def test_rejects_duplicate_checkpoint_ids(self, same_rubric):
        rubrics = _rubrics()
        duplicate = Checkpoint(id="c1", description="A different criterion")
        if same_rubric:
            rubrics[0].checkpoints.append(duplicate)
        else:
            rubrics.append(Rubric(name="safety", checkpoints=[duplicate]))
        with pytest.raises(ValueError, match="Duplicate checkpoint ID.*c1"):
            AgenticGrader(harness=FakeHarness([]), rubrics=rubrics)

    def test_rejects_duplicate_rubric_names(self):
        rubrics = _rubrics() + [Rubric(name="correctness", checkpoints=[])]
        with pytest.raises(ValueError, match="Duplicate rubric name"):
            AgenticGrader(harness=FakeHarness([]), rubrics=rubrics)


@pytest.mark.unit
class TestBuildOutputSchemaAndPrompt:
    def test_build_output_schema_flattens_all_checkpoint_ids(self):
        schema = _build_output_schema(_rubrics())
        assert schema == {"c1": {}, "c2": {}}

    def test_build_prompt_includes_query_response_and_checkpoints(self):
        prompt = _build_prompt("What is 2+2?", "4", _rubrics())
        assert "What is 2+2?" in prompt
        assert "<response>4</response>" in prompt
        assert "c1" in prompt and "c2" in prompt
        assert "_judge_result.json" in prompt

    def test_build_prompt_includes_checkpoint_content_when_present(self):
        rubrics = [
            Rubric(
                name="tests",
                checkpoints=[Checkpoint(id="t1", description="Passes unit test", content="assert 1 + 1 == 2")],
            )
        ]
        prompt = _build_prompt("q", "r", rubrics)
        assert "assert 1 + 1 == 2" in prompt


@pytest.mark.unit
class TestNormalizeSample:
    @pytest.mark.parametrize("passed", ["false", "true", "0", 1, 0, None, [], {"value": False}])
    def test_rejects_non_boolean_verdicts(self, passed):
        normalized = _normalize_sample({"c1": {"passed": passed}}, ["c1"])
        assert normalized == {}

    @pytest.mark.parametrize("execution_log", [["pytest: passed"], {"command": "pytest"}, 42])
    def test_drops_malformed_logs_without_losing_valid_checkpoints(self, execution_log):
        parsed = {"c1": {"passed": True, "execution_log": execution_log}, "c2": {"passed": False}}
        normalized = _normalize_sample(parsed, ["c1", "c2"])
        assert set(normalized) == {"c2"}
        assert normalized["c2"].passed is False

    def test_extracts_known_checkpoint_ids_only(self):
        parsed = {"c1": {"passed": True, "reason": "ok"}, "unknown_id": {"passed": True, "reason": "ignored"}}
        normalized = _normalize_sample(parsed, ["c1", "c2"])
        assert set(normalized.keys()) == {"c1"}
        assert normalized["c1"].passed is True

    def test_drops_entries_missing_passed_key(self):
        parsed = {"c1": {"reason": "no passed field"}}
        normalized = _normalize_sample(parsed, ["c1"])
        assert normalized == {}

    def test_captures_execution_log_when_present(self):
        parsed = {"c1": {"passed": False, "reason": "failed", "execution_log": "$ pytest\n1 failed"}}
        normalized = _normalize_sample(parsed, ["c1"])
        assert normalized["c1"].execution_log == "$ pytest\n1 failed"


@pytest.mark.unit
class TestAggregateRubric:
    def test_large_finite_weights_do_not_overflow(self):
        rubric = _rubrics()[0]
        for cp in rubric.checkpoints:
            cp.weight = 1e308
        result = _aggregate_rubric(rubric, {"c1": CheckpointResult(checkpoint_id="c1", passed=True)})
        assert result.score == 0.5

    def test_weighted_mean_of_checkpoints(self):
        rubric = _rubrics()[0]  # c1 weight=2.0, c2 weight=1.0
        results = {
            "c1": CheckpointResult(checkpoint_id="c1", passed=True, reason="right"),
            "c2": CheckpointResult(checkpoint_id="c2", passed=False, reason="unclear"),
        }
        rubric_result = _aggregate_rubric(rubric, results)
        assert rubric_result.score == pytest.approx(2.0 / 3.0)

    def test_all_checkpoints_pass_scores_full(self):
        rubric = _rubrics()[0]
        results = {
            "c1": CheckpointResult(checkpoint_id="c1", passed=True, reason="right"),
            "c2": CheckpointResult(checkpoint_id="c2", passed=True, reason="clear"),
        }
        rubric_result = _aggregate_rubric(rubric, results)
        assert rubric_result.score == 1.0

    def test_missing_checkpoint_result_counts_as_failed(self):
        rubric = _rubrics()[0]
        results = {"c2": CheckpointResult(checkpoint_id="c2", passed=True, reason="clear")}
        rubric_result = _aggregate_rubric(rubric, results)
        assert rubric_result.score == pytest.approx(1.0 / 3.0)

    def test_no_checkpoints_scores_zero(self):
        rubric = Rubric(name="empty", checkpoints=[])
        rubric_result = _aggregate_rubric(rubric, {})
        assert rubric_result.score == 0.0


@pytest.mark.unit
class TestAggregateOverall:
    def test_large_finite_weights_do_not_overflow(self):
        rubrics = [Rubric(name="a", weight=1e308, checkpoints=[]), Rubric(name="b", weight=1e308, checkpoints=[])]
        results = [
            RubricResult(name="a", score=0.5, checkpoint_results=[]),
            RubricResult(name="b", score=1.0, checkpoint_results=[]),
        ]
        assert _aggregate_overall(rubrics, results) == 0.75

    def test_weighted_mean_across_rubrics(self):
        rubrics = [Rubric(name="a", weight=1.0, checkpoints=[]), Rubric(name="b", weight=3.0, checkpoints=[])]
        rubric_results = [
            RubricResult(name="a", score=1.0, checkpoint_results=[]),
            RubricResult(name="b", score=0.0, checkpoint_results=[]),
        ]
        overall = _aggregate_overall(rubrics, rubric_results)
        assert overall == pytest.approx(0.25)


@pytest.mark.unit
class TestAgenticGraderEvaluateFullFlow:
    @pytest.mark.parametrize("checkpoint_weight", [False, True], ids=["rubric", "checkpoint"])
    @pytest.mark.parametrize("use_override", [False, True], ids=["mutated-default", "override"])
    async def test_revalidates_weights_before_running(self, checkpoint_weight, use_override):
        harness = FakeHarness([])
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        rubrics = _rubrics() if use_override else grader.rubrics
        target = rubrics[0].checkpoints[0] if checkpoint_weight else rubrics[0]
        target.weight = float("nan")
        kwargs = {"rubrics": rubrics} if use_override else {}
        result = await grader.aevaluate(transcript=[], **kwargs)
        assert isinstance(result, GraderError)
        assert result.error == "invalid_rubrics"
        assert harness.run_calls == []

    @pytest.mark.parametrize("invalid_result", [{"passed": "false"}, {"passed": True, "execution_log": ["ok"]}])
    async def test_malformed_verdicts_do_not_raise_or_award_credit(self, invalid_result):
        harness = FakeHarness([HarnessResult(available=True, result={"c1": invalid_result, "c2": {"passed": True}})])
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(transcript=[{"role": "assistant", "content": "answer"}])
        assert isinstance(result, GraderScore)
        assert result.score == pytest.approx(1.0 / 3.0)

    @pytest.mark.parametrize("is_file", [False, True], ids=["missing", "file"])
    async def test_invalid_workspace_returns_error_before_invoking_harness(self, tmp_path, is_file):
        candidate = tmp_path / "candidate"
        if is_file:
            candidate.write_text("not a directory", encoding="utf-8")
        harness = FakeHarness([])
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(workspace_path=str(candidate))
        assert isinstance(result, GraderError)
        assert result.error == "unavailable"
        assert harness.run_calls == []

    @pytest.mark.parametrize("use_override", [False, True], ids=["mutated-default", "override"])
    async def test_validates_duplicate_checkpoint_ids_on_every_evaluation(self, use_override):
        harness = FakeHarness([])
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        rubrics = _rubrics() if use_override else grader.rubrics
        rubrics.append(Rubric(name="safety", checkpoints=[Checkpoint(id="c1", description="Another criterion")]))
        kwargs = {"rubrics": rubrics} if use_override else {}
        result = await grader.aevaluate(transcript=[], **kwargs)
        assert isinstance(result, GraderError)
        assert result.error == "invalid_rubrics"
        assert "c1" in result.reason
        assert harness.run_calls == []

    async def test_returns_grader_error_when_no_evidence_given(self):
        grader = AgenticGrader(harness=FakeHarness([]), rubrics=_rubrics())
        result = await grader.aevaluate(query="q", response="r")
        assert isinstance(result, GraderError)
        assert result.error == "no_evidence"

    async def test_returns_grader_error_when_harness_unavailable(self, tmp_path):
        harness = FakeHarness([HarnessResult(available=False)])
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(query="q", response="r", workspace_path=str(tmp_path))
        assert isinstance(result, GraderError)
        assert result.error == "unavailable"

    async def test_grader_error_metadata_carries_harness_diagnostics(self, tmp_path):
        """A caller must be able to tell "harness CLI exited non-zero after 12s" apart from
        "the agent decided the checkpoints failed" without digging into private internals --
        the diagnostics HarnessResult already carries (exit_code/timed_out/duration/stderr)
        must survive into GraderError.metadata rather than being discarded."""
        harness = FakeHarness(
            [
                HarnessResult(
                    available=False,
                    exit_code=1,
                    timed_out=False,
                    duration=12.5,
                    raw_stderr="claude: command failed: permission denied",
                )
            ]
        )
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(query="q", response="r", workspace_path=str(tmp_path))

        assert isinstance(result, GraderError)
        assert result.metadata["exit_code"] == 1
        assert result.metadata["timed_out"] is False
        assert result.metadata["duration"] == pytest.approx(12.5)
        assert "permission denied" in result.metadata["raw_stderr"]
        assert result.metadata["harness_type"] == "FakeHarness"

    async def test_grader_error_metadata_reports_timeout(self, tmp_path):
        harness = FakeHarness([HarnessResult(available=False, timed_out=True, duration=900.0)])
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(query="q", response="r", workspace_path=str(tmp_path))

        assert isinstance(result, GraderError)
        assert result.metadata["timed_out"] is True
        assert result.metadata["duration"] == pytest.approx(900.0)

    async def test_returns_grader_score_when_all_checkpoints_pass(self, tmp_path):
        harness = FakeHarness(
            [
                HarnessResult(
                    available=True,
                    exit_code=0,
                    duration=42.0,
                    result={"c1": {"passed": True, "reason": "matches"}, "c2": {"passed": True, "reason": "clear"}},
                )
            ]
        )
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(query="q", response="r", workspace_path=str(tmp_path))

        assert isinstance(result, GraderScore)
        assert result.score == 1.0
        assert result.metadata["rubric_results"][0]["score"] == 1.0
        assert len(harness.run_calls) == 1
        # Harness-level diagnostics must also survive on the success path, not just on failure.
        assert result.metadata["exit_code"] == 0
        assert result.metadata["timed_out"] is False
        assert result.metadata["duration"] == pytest.approx(42.0)

    async def test_returns_partial_score_when_one_checkpoint_fails(self, tmp_path):
        harness = FakeHarness(
            [
                HarnessResult(
                    available=True,
                    result={"c1": {"passed": False, "reason": "wrong"}, "c2": {"passed": True, "reason": "clear"}},
                )
            ]
        )
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(query="q", response="r", workspace_path=str(tmp_path))

        assert isinstance(result, GraderScore)
        assert result.score == pytest.approx(1.0 / 3.0)  # only c2 (weight=1.0 of total 3.0) passed

    async def test_rubrics_kwarg_overrides_constructor_default(self, tmp_path):
        override_rubrics = [Rubric(name="other", checkpoints=[Checkpoint(id="o1", description="Different checkpoint")])]
        harness = FakeHarness([HarnessResult(available=True, result={"o1": {"passed": True, "reason": "ok"}})])
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(query="q", response="r", workspace_path=str(tmp_path), rubrics=override_rubrics)

        assert isinstance(result, GraderScore)
        assert harness.run_calls[0]["schema"] == {"o1": {}}

    async def test_transcript_alone_is_sufficient_evidence(self):
        harness = FakeHarness(
            [
                HarnessResult(
                    available=True,
                    result={"c1": {"passed": True, "reason": "a"}, "c2": {"passed": True, "reason": "b"}},
                )
            ]
        )
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(query="q", response="r", transcript=[{"role": "user", "content": "hi"}])
        assert isinstance(result, GraderScore)

    async def test_returns_grader_error_unavailable_when_sandbox_setup_raises(self):
        """ProcessSandbox.__enter__ raises FileNotFoundError for a nonexistent transcript path.
        _run_one must catch this and return None so _aevaluate surfaces GraderError(unavailable),
        and the exception itself (not just a generic "unavailable" label) must be visible in
        metadata -- a caller misconfiguring workspace_path/transcript deserves a config-error
        message, not the same opaque "harness CLI unavailable" reason as a real CLI crash.
        """
        harness = FakeHarness([HarnessResult(available=True, result={})])
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(
            query="q",
            response="r",
            transcript="/nonexistent/path/transcript.jsonl",
        )
        assert isinstance(result, GraderError)
        assert result.error == "unavailable"
        assert "FileNotFoundError" in result.metadata["setup_error"]
        assert "transcript path not found" in result.metadata["setup_error"]
        assert "harness_error" not in result.metadata
        assert not harness.run_calls
        # The harness subprocess was never reached, so there is no exit_code/duration to report.
        assert "exit_code" not in result.metadata

    async def test_invalid_result_is_not_a_sandbox_setup_error(self):
        """Invalid result types are rejected by the harness after sandbox setup succeeds."""
        sandbox_dirs = []

        class InvalidResultHarness(BaseHarness):
            @property
            def default_binary(self):
                return sys.executable

            def build_command(self, sandbox_dir, prompt, model):
                sandbox_dirs.append(sandbox_dir)
                return [
                    self.binary,
                    "-c",
                    "from pathlib import Path; Path('_judge_result.json').write_text('[]', encoding='utf-8')",
                ]

        grader = AgenticGrader(harness=InvalidResultHarness(), rubrics=_rubrics())
        result = await grader.aevaluate(transcript=[])

        assert isinstance(result, GraderError)
        assert result.error == "unavailable"
        assert "harness_error" not in result.metadata
        assert "setup_error" not in result.metadata
        assert result.metadata["timed_out"] is False
        assert len(sandbox_dirs) == 1
        assert not sandbox_dirs[0].exists()

    async def test_harness_exception_is_not_a_sandbox_setup_error(self, monkeypatch):
        """Unexpected harness exceptions retain their stage and still clean up the sandbox."""
        harness = FakeHarness([])
        sandbox_dirs = []

        def fail(sandbox_dir, *args, **kwargs):
            sandbox_dirs.append(sandbox_dir)
            raise RuntimeError("harness failed")

        monkeypatch.setattr(harness, "run", fail)
        grader = AgenticGrader(harness=harness, rubrics=_rubrics())
        result = await grader.aevaluate(transcript=[])

        assert isinstance(result, GraderError)
        assert result.metadata["harness_error"] == "RuntimeError: harness failed"
        assert "setup_error" not in result.metadata
        assert len(sandbox_dirs) == 1
        assert not sandbox_dirs[0].exists()


@pytest.mark.unit
class TestAgenticGraderGetMetadata:
    def test_get_metadata_describes_protocol(self):
        metadata = AgenticGrader.get_metadata()
        assert "protocol" in metadata
        assert "sandbox" in metadata["protocol"]
