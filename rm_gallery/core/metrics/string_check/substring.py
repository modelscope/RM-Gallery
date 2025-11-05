"""
Substring Match Metrics

Substring matching metrics for checking text containment relationships.
"""

from typing import List

from pydantic import Field

from rm_gallery.core.metrics.base import BaseStringMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("substring_match")
class SubstringMatchMetric(BaseStringMetric):
    """
    Substring Match Metric

    Checks if the candidate text contains the reference text (or reference contains candidate).
    Similar to OpenAI Evals' Includes metric.

    Attributes:
        name: Metric name
        case_sensitive: Whether to perform case-sensitive matching
        bidirectional: Whether to check bidirectionally (candidate contains reference OR reference contains candidate)

    Example:
        >>> metric = SubstringMatchMetric(case_sensitive=False)
        >>> input_data = ComparisonInput(
        ...     reference="cat",
        ...     candidate="The cat sat on the mat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Contains: {result.score}")
        Contains: 1.0
    """

    name: str = "substring_match"
    case_sensitive: bool = Field(
        default=False, description="Whether to perform case-sensitive matching"
    )
    bidirectional: bool = Field(
        default=False, description="Whether to check bidirectionally"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute substring match

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Preprocess
        if not self.case_sensitive:
            candidate = candidate.lower()

        # Handle multiple reference texts
        if isinstance(reference, list):
            references = [
                ref if self.case_sensitive else ref.lower() for ref in reference
            ]
            matches = [self._check_substring(candidate, ref) for ref in references]
            matched = any(matches)
            details = {
                "num_references": len(reference),
                "matches_per_reference": matches,
                "matched_reference_indices": [i for i, m in enumerate(matches) if m],
            }
        else:
            if not self.case_sensitive:
                reference = reference.lower()
            matched = self._check_substring(candidate, reference)
            details = {}

        details.update(
            {
                "matched": matched,
                "case_sensitive": self.case_sensitive,
                "bidirectional": self.bidirectional,
            }
        )

        return MetricResult(
            name=self.name,
            score=1.0 if matched else 0.0,
            details=details,
        )

    def _check_substring(self, candidate: str, reference: str) -> bool:
        """
        Check substring relationship

        Args:
            candidate: Candidate text
            reference: Reference text

        Returns:
            bool: Whether containment relationship exists
        """
        if self.bidirectional:
            # Bidirectional: candidate contains reference OR reference contains candidate
            return reference in candidate or candidate in reference
        else:
            # Unidirectional: candidate contains reference
            return reference in candidate


@register_metric("contains_all")
class ContainsAllMetric(BaseStringMetric):
    """
    Contains All Metric

    Checks if the candidate text contains all specified substrings.

    Attributes:
        name: Metric name
        substrings: List of substrings that must be contained
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> metric = ContainsAllMetric(substrings=["cat", "mat"])
        >>> input_data = ComparisonInput(
        ...     reference="",  # substrings already specified at initialization
        ...     candidate="The cat sat on the mat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Contains all: {result.score}")
    """

    name: str = "contains_all"
    substrings: List[str] = Field(
        default=[], description="List of substrings that must be contained"
    )
    case_sensitive: bool = Field(
        default=False, description="Whether to perform case-sensitive matching"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute contains all

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        candidate = input_data.candidate

        # If no preset substrings, use reference
        substrings = self.substrings if self.substrings else []
        if not substrings and input_data.reference:
            if isinstance(input_data.reference, list):
                substrings = input_data.reference
            else:
                substrings = [input_data.reference]

        # Preprocess
        if not self.case_sensitive:
            candidate = candidate.lower()
            substrings = [s.lower() for s in substrings]

        # Check each substring
        contains = [substring in candidate for substring in substrings]
        matched = all(contains)

        details = {
            "matched": matched,
            "num_substrings": len(substrings),
            "contains_per_substring": contains,
            "missing_substrings": [s for s, c in zip(substrings, contains) if not c],
        }

        # Calculate score: proportion of contained substrings
        score = sum(contains) / len(contains) if contains else 0.0

        return MetricResult(
            name=self.name,
            score=score,
            details=details,
        )


@register_metric("contains_any")
class ContainsAnyMetric(BaseStringMetric):
    """
    Contains Any Metric

    Checks if the candidate text contains at least one of the specified substrings.

    Attributes:
        name: Metric name
        substrings: List of candidate substrings
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> metric = ContainsAnyMetric(substrings=["cat", "dog"])
        >>> input_data = ComparisonInput(
        ...     reference="",
        ...     candidate="The cat sat on the mat"
        ... )
        >>> result = metric.compute(input_data)
    """

    name: str = "contains_any"
    substrings: List[str] = Field(
        default=[], description="List of candidate substrings"
    )
    case_sensitive: bool = Field(
        default=False, description="Whether to perform case-sensitive matching"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute contains any

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        candidate = input_data.candidate

        # If no preset substrings, use reference
        substrings = self.substrings if self.substrings else []
        if not substrings and input_data.reference:
            if isinstance(input_data.reference, list):
                substrings = input_data.reference
            else:
                substrings = [input_data.reference]

        # Preprocess
        if not self.case_sensitive:
            candidate = candidate.lower()
            substrings = [s.lower() for s in substrings]

        # Check each substring
        contains = [substring in candidate for substring in substrings]
        matched = any(contains)

        details = {
            "matched": matched,
            "num_substrings": len(substrings),
            "contains_per_substring": contains,
            "matched_substrings": [s for s, c in zip(substrings, contains) if c],
        }

        return MetricResult(
            name=self.name,
            score=1.0 if matched else 0.0,
            details=details,
        )


@register_metric("word_overlap")
class WordOverlapMetric(BaseStringMetric):
    """
    Word Overlap Metric

    Calculates the proportion of word overlap between candidate and reference texts.

    Attributes:
        name: Metric name
        case_sensitive: Whether to perform case-sensitive matching
        normalize_text: Whether to normalize text

    Example:
        >>> metric = WordOverlapMetric()
        >>> input_data = ComparisonInput(
        ...     reference="the cat sat on the mat",
        ...     candidate="the dog sat on the rug"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Overlap: {result.score:.2f}")
    """

    name: str = "word_overlap"
    case_sensitive: bool = Field(
        default=False, description="Whether to perform case-sensitive matching"
    )
    normalize_text: bool = Field(default=True, description="Whether to normalize text")

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute word overlap

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result, score is the overlap ratio
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Normalize
        if input_data.normalize and self.normalize_text:
            candidate = self._normalize(candidate)
            if isinstance(reference, list):
                reference = [self._normalize(ref) for ref in reference]
            else:
                reference = self._normalize(reference)
        elif not self.case_sensitive:
            candidate = candidate.lower()
            if isinstance(reference, list):
                reference = [ref.lower() for ref in reference]
            else:
                reference = reference.lower()

        # Handle multiple reference texts
        if isinstance(reference, list):
            scores = [self._compute_overlap(candidate, ref) for ref in reference]
            score = max(scores)
            details = {
                "scores_per_reference": scores,
                "max_score": score,
            }
        else:
            score = self._compute_overlap(candidate, reference)
            details = {}

        return MetricResult(
            name=self.name,
            score=score,
            details=details,
        )

    def _compute_overlap(self, candidate: str, reference: str) -> float:
        """
        Calculate word overlap between two texts

        Args:
            candidate: Candidate text
            reference: Reference text

        Returns:
            float: Overlap ratio
        """
        cand_words = set(candidate.split())
        ref_words = set(reference.split())

        if len(ref_words) == 0:
            return 0.0

        overlap = cand_words & ref_words
        return len(overlap) / len(ref_words)


@register_metric("char_overlap")
class CharacterOverlapMetric(BaseStringMetric):
    """
    Character Overlap Metric

    Calculates the proportion of character overlap between candidate and reference texts.

    Attributes:
        name: Metric name
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> metric = CharacterOverlapMetric()
        >>> input_data = ComparisonInput(
        ...     reference="hello",
        ...     candidate="helo"
        ... )
        >>> result = metric.compute(input_data)
    """

    name: str = "char_overlap"
    case_sensitive: bool = Field(
        default=False, description="Whether to perform case-sensitive matching"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute character overlap

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Preprocess
        if not self.case_sensitive:
            candidate = candidate.lower()

        # Handle multiple reference texts
        if isinstance(reference, list):
            references = [
                ref if self.case_sensitive else ref.lower() for ref in reference
            ]
            scores = [self._compute_char_overlap(candidate, ref) for ref in references]
            score = max(scores)
            details = {"scores_per_reference": scores}
        else:
            if not self.case_sensitive:
                reference = reference.lower()
            score = self._compute_char_overlap(candidate, reference)
            details = {}

        return MetricResult(
            name=self.name,
            score=score,
            details=details,
        )

    def _compute_char_overlap(self, candidate: str, reference: str) -> float:
        """
        Calculate character overlap ratio

        Args:
            candidate: Candidate text
            reference: Reference text

        Returns:
            float: Overlap ratio
        """
        cand_chars = set(candidate)
        ref_chars = set(reference)

        if len(ref_chars) == 0:
            return 0.0

        overlap = cand_chars & ref_chars
        return len(overlap) / len(ref_chars)


__all__ = [
    "SubstringMatchMetric",
    "ContainsAllMetric",
    "ContainsAnyMetric",
    "WordOverlapMetric",
    "CharacterOverlapMetric",
]
