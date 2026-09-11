#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for schema types: EvalSuggestion, EvalFeedback,
and the eval_feedback field on GraderScore and GraderScoreCallback.
"""

import pytest
from pydantic import ValidationError

from openjudge.graders.schema import (
    Checkpoint,
    CheckpointResult,
    EvalFeedback,
    EvalSuggestion,
    GraderScore,
    GraderScoreCallback,
    Rubric,
    RubricResult,
)


@pytest.mark.unit
@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("checkpoint", [False, True], ids=["rubric", "checkpoint"])
def test_judge_weights_must_be_finite_and_non_negative(weight, checkpoint):
    with pytest.raises(ValidationError):
        if checkpoint:
            Checkpoint(id="c", description="criterion", weight=weight)
        else:
            Rubric(name="r", checkpoints=[], weight=weight)


@pytest.mark.unit
def test_zero_judge_weights_remain_supported():
    checkpoint = Checkpoint(id="c", description="criterion", weight=0)
    assert Rubric(name="r", checkpoints=[checkpoint], weight=0).weight == checkpoint.weight == 0


@pytest.mark.unit
class TestEvalSuggestion:
    def test_create_with_reason_only(self):
        s = EvalSuggestion(reason="No assertion checks correctness")
        assert s.reason == "No assertion checks correctness"
        assert s.assertion is None

    def test_create_with_assertion(self):
        s = EvalSuggestion(
            assertion="The output includes 'John Smith'",
            reason="A hallucinated document would also pass",
        )
        assert s.assertion == "The output includes 'John Smith'"

    def test_serialization_roundtrip(self):
        s = EvalSuggestion(reason="Improve coverage")
        data = s.model_dump()
        s2 = EvalSuggestion(**data)
        assert s2 == s


@pytest.mark.unit
class TestEvalFeedback:
    def test_create_defaults(self):
        f = EvalFeedback()
        assert f.suggestions == []
        assert f.overall == "No suggestions, evals look solid"

    def test_create_with_suggestions(self):
        f = EvalFeedback(
            suggestions=[EvalSuggestion(reason="Add check")],
            overall="Assertions check presence but not correctness",
        )
        assert len(f.suggestions) == 1
        assert f.suggestions[0].reason == "Add check"
        assert f.overall == "Assertions check presence but not correctness"

    def test_from_dict(self):
        f = EvalFeedback(
            **{
                "suggestions": [{"reason": "Weak assertion"}],
                "overall": "Needs improvement",
            }
        )
        assert len(f.suggestions) == 1
        assert isinstance(f.suggestions[0], EvalSuggestion)

    def test_serialization_roundtrip(self):
        f = EvalFeedback(
            suggestions=[EvalSuggestion(assertion="A", reason="B")],
            overall="Fair",
        )
        data = f.model_dump()
        f2 = EvalFeedback(**data)
        assert f2.overall == f.overall
        assert len(f2.suggestions) == len(f.suggestions)


@pytest.mark.unit
class TestGraderScoreEvalFeedback:
    def test_score_defaults_none_feedback(self):
        s = GraderScore(name="test", score=3.0, reason="OK")
        assert s.eval_feedback is None

    def test_score_with_eval_feedback(self):
        fb = EvalFeedback(
            suggestions=[EvalSuggestion(reason="Add check")],
            overall="Needs work",
        )
        s = GraderScore(name="test", score=2.0, reason="Weak", eval_feedback=fb)
        assert s.eval_feedback is not None
        assert s.eval_feedback.overall == "Needs work"

    def test_score_serialization_includes_feedback(self):
        s = GraderScore(
            name="test",
            score=5.0,
            reason="Perfect",
            eval_feedback=EvalFeedback(overall="Solid"),
        )
        data = s.model_dump()
        assert "eval_feedback" in data
        assert data["eval_feedback"]["overall"] == "Solid"

    def test_score_deserialization_with_feedback(self):
        data = {
            "name": "test",
            "score": 3.0,
            "reason": "Mid",
            "eval_feedback": {
                "suggestions": [{"assertion": "A", "reason": "R"}],
                "overall": "OK",
            },
        }
        s = GraderScore(**data)
        assert s.eval_feedback is not None
        assert isinstance(s.eval_feedback, EvalFeedback)
        assert len(s.eval_feedback.suggestions) == 1
        assert isinstance(s.eval_feedback.suggestions[0], EvalSuggestion)


@pytest.mark.unit
class TestGraderScoreCallbackExcludesEvalFeedback:
    def test_callback_has_no_eval_feedback_field(self):
        schema = GraderScoreCallback.model_json_schema()
        assert "eval_feedback" not in schema.get("properties", {})

    def test_callback_only_has_reason_score_metadata(self):
        schema = GraderScoreCallback.model_json_schema()
        props = set(schema.get("properties", {}).keys())
        assert props == {"reason", "score", "metadata"}


@pytest.mark.unit
class TestCheckpointAndRubric:
    def test_checkpoint_defaults(self):
        cp = Checkpoint(id="c1", description="Must return valid JSON")
        assert cp.id == "c1"
        assert cp.content is None
        assert cp.weight == 1.0

    def test_checkpoint_with_content_and_weight(self):
        cp = Checkpoint(
            id="c1",
            description="Output parses as JSON",
            content="import json; json.loads(response)",
            weight=2.0,
        )
        assert cp.content == "import json; json.loads(response)"
        assert cp.weight == 2.0

    def test_rubric_defaults_and_checkpoints(self):
        rubric = Rubric(name="correctness", checkpoints=[Checkpoint(id="c1", description="d")])
        assert rubric.name == "correctness"
        assert rubric.description is None
        assert rubric.weight == 1.0
        assert len(rubric.checkpoints) == 1

    def test_rubric_serialization_roundtrip(self):
        rubric = Rubric(
            name="safety",
            description="No unsafe actions",
            weight=2.0,
            checkpoints=[Checkpoint(id="s1", description="No destructive commands")],
        )
        data = rubric.model_dump()
        rubric2 = Rubric(**data)
        assert rubric2 == rubric


@pytest.mark.unit
class TestCheckpointResultAndRubricResult:
    def test_checkpoint_result_defaults(self):
        r = CheckpointResult(checkpoint_id="c1", passed=True, reason="matched expected output")
        assert r.execution_log is None

    def test_checkpoint_result_with_execution_log(self):
        r = CheckpointResult(
            checkpoint_id="c1",
            passed=False,
            reason="test script failed",
            execution_log="$ python test.py\nAssertionError: expected 4 got 5",
        )
        assert "AssertionError" in r.execution_log

    def test_rubric_result_serialization_roundtrip(self):
        result = RubricResult(
            name="correctness",
            score=0.5,
            checkpoint_results=[CheckpointResult(checkpoint_id="c1", passed=True, reason="ok")],
        )
        data = result.model_dump()
        result2 = RubricResult(**data)
        assert result2 == result
