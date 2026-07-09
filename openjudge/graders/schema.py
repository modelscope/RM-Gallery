# -*- coding: utf-8 -*-
"""
Schemas for grading tasks.

This module defines the data schemas used in grading tasks, including grader modes,
result structures, eval feedback, and error handling.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class GraderMode(str, Enum):
    """Grader modes for grader functions.

    This enum defines the two primary modes that graders can operate in:
    pointwise (evaluating individual samples) and listwise (ranking multiple samples).

    Attributes:
        POINTWISE: Pointwise grader mode.
        LISTWISE: Listwise grader mode.

    Example:
        >>> mode = GraderMode.POINTWISE
        >>> print(mode.value)
        pointwise
        >>>
        >>> mode = GraderMode.LISTWISE
        >>> print(mode.value)
        listwise
    """

    POINTWISE = "pointwise"
    LISTWISE = "listwise"


class EvalSuggestion(BaseModel):
    """A suggestion for improving the evaluation itself.

    Used when the grader detects weak assertions, missing coverage,
    or assertions that would pass for clearly wrong outputs.

    Attributes:
        assertion: The original assertion text this relates to (optional).
        reason: Why this suggestion is needed.

    Example:
        >>> s = EvalSuggestion(
        ...     assertion="The output includes the name 'John Smith'",
        ...     reason="A hallucinated document would also pass this check"
        ... )
    """

    assertion: Optional[str] = Field(default=None, description="The assertion this relates to")
    reason: str = Field(description="Why this suggestion is needed")


class EvalFeedback(BaseModel):
    """Feedback on the quality of the evaluation itself.

    Follows the principle that a passing grade on a weak assertion is worse
    than useless — it creates false confidence.

    Attributes:
        suggestions: List of concrete improvement suggestions.
        overall: Brief assessment of the eval quality.

    Example:
        >>> f = EvalFeedback(
        ...     suggestions=[EvalSuggestion(reason="No assertion checks correctness")],
        ...     overall="Assertions check presence but not correctness"
        ... )
    """

    suggestions: List[EvalSuggestion] = Field(default_factory=list, description="Improvement suggestions")
    overall: str = Field(default="No suggestions, evals look solid", description="Brief assessment")


class GraderResult(BaseModel):
    """Base class for grader results.

    This Pydantic model defines the structure for grader results,
    which include a reason and optional metadata.

    Attributes:
        name (str): The name of the grader.
        reason (str): The reason for the result.
        metadata (Dict[str, Any]): The metadata of the grader result.

    Example:
        >>> result = GraderResult(
        ...     name="test_grader",
        ...     reason="Test evaluation completed",
        ...     metadata={"duration": 0.1}
        ... )
        >>> print(result.name)
        test_grader
        >>> result.model_dump()
        {'name': 'test_grader', 'reason': 'Test evaluation completed', 'metadata': {'duration': 0.1}}
    """

    name: str = Field(description="The name of the grader")
    reason: str = Field(default="", description="The reason for the result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the grader result",
    )


class GraderScore(GraderResult):
    """Grader score result.

    Represents a numerical score assigned by a grader along with a reason.

    Attributes:
        reason (str): Explanation of how the score was determined.
        score (float): A numerical score assigned by the grader.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> score_result = GraderScore(
        ...     name="accuracy_grader",
        ...     score=0.85,
        ...     reason="Answer is mostly accurate",
        ...     metadata={"confidence": 0.9}
        ... )
        >>> print(score_result.score)
        0.85
    """

    reason: str = Field(description="reason")
    score: float = Field(description="score")
    eval_feedback: Optional[EvalFeedback] = Field(
        default=None, description="Feedback on the quality of the evaluation itself"
    )


class GraderScoreCallback(BaseModel):
    """Callback for grader score result.

    Represents a numerical score assigned by a grader along with a reason.

    Attributes:
        reason (str): Explanation of how the score was determined.
        score (float): A numerical score assigned by the grader.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> callback = GraderScoreCallback(
        ...     score=0.9,
        ...     reason="High confidence in evaluation",
        ...     metadata={"model_used": ""}
        ... )
        >>> print(callback.score)
        0.9
    """

    reason: str = Field(description="reason")
    score: float = Field(description="score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the grader result",
    )


class RankValidation:
    """This class provides a field validator that can be inherited or mixed into Pydantic models
    to validate rank-related fields. The validator ensures that rank lists represent proper
    rankings according to standard ranking conventions:

    - Ranks must be positive integers starting from 1
    - All ranks in the list must be unique (no ties/duplicates)
    - The list must contain exactly all integers from 1 to n (where n is the list length)
    - Empty rank lists are not allowed

    This validation is particularly useful for listwise evaluation scenarios where
    LLMs or other systems output rankings of items, ensuring the output conforms to
    expected ranking formats before further processing.

    The validation will automatically be applied to the "rank" field during model
    instantiation and validation.
    """

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, rank: List[int]) -> List[int]:
        """Validate that the rank list is a valid permutation of consecutive positive integers starting from 1.

        This validator ensures that the rank list meets all requirements for a proper ranking:
        - Cannot be empty
        - Contains only positive integers (≥ 1)
        - All values are unique (no duplicates)
        - Forms a complete permutation of integers from 1 to n (where n is the list length)

        Args:
            rank: A list of integers representing ranks to be validated.

        Returns:
            The validated rank list unchanged if all validation checks pass.

        Raises:
            ValueError: If any of the following conditions are violated:
                - The rank list is empty
                - Any rank value is not a positive integer (≤ 0)
                - The rank list contains duplicate values
                - The rank list is not a complete permutation of [1, 2, ..., n]

        Examples:
            >>> validate_rank([1, 2, 3])  # Valid - sequential ranks
            [1, 2, 3]
            >>> validate_rank([3, 1, 2])  # Valid - permuted ranks
            [3, 1, 2]
            >>> validate_rank([1, 1, 2])  # Invalid - duplicates
            ValueError: Ranks should be unique
            >>> validate_rank([1, 3])     # Invalid - missing rank 2
            ValueError: Ranks should be a permutation of [1, 2]
            >>> validate_rank([0, 1])     # Invalid - contains zero
            ValueError: All ranks should be positive integers
        """
        if not rank:
            raise ValueError("Rank list cannot be empty")
        if any(x <= 0 for x in rank):
            raise ValueError("All ranks should be positive integers")
        if len(rank) != len(set(rank)):
            raise ValueError("Ranks should be unique")
        expected = set(range(1, len(rank) + 1))
        if set(rank) != expected:
            raise ValueError(f"Ranks should be a permutation of {sorted(expected)}")
        return rank


class GraderRank(GraderResult, RankValidation):
    """Grader rank result.

    Represents a ranking of items assigned by a grader along with a reason.

    Attributes:
        rank (List[int]): The ranking of items.
        reason (str): Explanation of how the ranking was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> rank_result = GraderRank(
        ...     name="relevance_ranker",
        ...     rank=[1, 3, 2],
        ...     reason="First response is most relevant",
        ...     metadata={"criteria": "relevance"}
        ... )
        >>> print(rank_result.rank)
        [1, 3, 2]
    """

    rank: List[int] = Field(description="rank")
    reason: str = Field(description="reason")


class GraderRankCallback(BaseModel, RankValidation):
    """Callback schema for LLM structured output in listwise grading.

    Used as the structured_model parameter in LLMGrader for LISTWISE mode.
    The LLM returns this schema which is then converted to GraderRank.

    Attributes:
        rank (List[int]): The ranking of items.
        reason (str): Explanation of how the ranking was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> callback = GraderRankCallback(
        ...     rank=[2, 1],
        ...     reason="Second response is more relevant",
        ...     metadata={"criteria": "clarity"}
        ... )
        >>> print(callback.rank)
        [2, 1]
    """

    rank: List[int] = Field(description="rank")
    reason: str = Field(description="reason")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the grader result",
    )


class GraderError(GraderResult):
    """Grader error result.

    Represents an error encountered during evaluation.

    Attributes:
        error (str): The error message.
        reason (str): Description of the error encountered during evaluation.
        metadata (Dict[str, Any]): Optional additional error information.

    Example:
        >>> error_result = GraderError(
        ...     name="test_grader",
        ...     error="Timeout occurred",
        ...     reason="Model took too long to respond",
        ...     metadata={"timeout_seconds": 30}
        ... )
        >>> print(error_result.error)
        Timeout occurred
    """

    error: str = Field(description="error")


class Checkpoint(BaseModel):
    """A single verifiable judging criterion within a Rubric.

    `content` is intentionally generic free text rather than a typed union:
    it may hold executable code/test scripts, natural-language judging
    criteria, or a reference answer. The agentic judge decides at runtime
    whether to execute it (if it looks like code) or reason about it (if
    it is a natural-language criterion).

    Attributes:
        id: Unique identifier within the parent Rubric's checkpoints list.
        description: Human-readable description of what this checkpoint verifies.
        content: Optional freeform content (code/test script/criteria/reference answer).
        weight: Relative weight among checkpoints in the same rubric.

    Example:
        >>> Checkpoint(id="c1", description="Response includes a citation", weight=1.0)
        >>> Checkpoint(
        ...     id="c2",
        ...     description="Code compiles and passes the provided test",
        ...     content="assert add(2, 2) == 4",
        ... )
    """

    id: str = Field(description="Unique identifier for this checkpoint within its rubric")
    description: str = Field(description="Human-readable description of what this checkpoint verifies")
    content: Optional[str] = Field(
        default=None,
        description="Freeform content: code/test script, judging criteria, or reference answer",
    )
    weight: float = Field(default=1.0, description="Relative weight among checkpoints in the same rubric")


class Rubric(BaseModel):
    """A named evaluation dimension made up of one or more Checkpoints.

    Attributes:
        name: Rubric dimension name (e.g. "correctness", "safety").
        description: Optional human-readable description of this dimension.
        weight: Relative weight of this rubric among all rubrics passed to AgenticGrader.
        checkpoints: The checkpoints belonging to this rubric.

    Example:
        >>> Rubric(
        ...     name="correctness",
        ...     checkpoints=[Checkpoint(id="c1", description="Answer matches ground truth")],
        ... )
    """

    name: str = Field(description="Rubric dimension name")
    description: Optional[str] = Field(default=None, description="Human-readable description of this dimension")
    weight: float = Field(default=1.0, description="Relative weight of this rubric among all rubrics")
    checkpoints: List[Checkpoint] = Field(description="Checkpoints belonging to this rubric")


class CheckpointResult(BaseModel):
    """The judged outcome of a single Checkpoint.

    Attributes:
        checkpoint_id: Matches the `id` of the Checkpoint this result is for.
        passed: Whether this checkpoint was judged as passed.
        reason: Evidence/explanation for the pass/fail judgment.
        execution_log: If the checkpoint's content was executable, the actual command(s)
            run and their output — evidence that the agent really executed it rather than
            guessing.

    Example:
        >>> CheckpointResult(checkpoint_id="c1", passed=True, reason="Matches reference answer verbatim")
    """

    checkpoint_id: str = Field(description="Matches Checkpoint.id")
    passed: bool = Field(description="Whether this checkpoint was judged as passed")
    reason: str = Field(default="", description="Evidence/explanation for the pass/fail judgment")
    execution_log: Optional[str] = Field(
        default=None,
        description="Command(s) actually executed and their output, if applicable",
    )


class RubricResult(BaseModel):
    """The aggregated outcome of a single Rubric across all its Checkpoints.

    Attributes:
        name: Matches the `name` of the Rubric this result is for.
        score: Weighted mean of this rubric's checkpoints' pass rates (each checkpoint
            contributes `weight * (1.0 if passed else 0.0)`).
        checkpoint_results: Per-checkpoint results that fed into this aggregation.

    Example:
        >>> RubricResult(
        ...     name="correctness",
        ...     score=1.0,
        ...     checkpoint_results=[CheckpointResult(checkpoint_id="c1", passed=True, reason="ok")],
        ... )
    """

    name: str = Field(description="Matches Rubric.name")
    score: float = Field(description="Aggregated score for this rubric")
    checkpoint_results: List[CheckpointResult] = Field(description="Per-checkpoint results")
