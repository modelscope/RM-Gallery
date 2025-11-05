"""
Fuzzy Match Metric

Fuzzy matching metrics based on Levenshtein Distance.
Supports multiple matching modes: ratio, partial_ratio, token_sort_ratio.
"""

import Levenshtein
from pydantic import Field

from rm_gallery.core.metrics.base import BaseMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("fuzzy_match")
class FuzzyMatchMetric(BaseMetric):
    """
    Fuzzy Match Metric

    Calculates text similarity using Levenshtein edit distance.
    Supports three matching modes:
    - ratio: Standard similarity ratio
    - partial_ratio: Partial string matching
    - token_sort_ratio: Token order-independent matching

    Attributes:
        name: Metric name
        method: Matching method (ratio/partial_ratio/token_sort_ratio)
        threshold: Match threshold, score >= threshold is considered a match

    Example:
        >>> metric = FuzzyMatchMetric(method="ratio", threshold=0.8)
        >>> input_data = ComparisonInput(
        ...     reference="hello world",
        ...     candidate="hello worl"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Score: {result.score}, Matched: {result.details['matched']}")
        Score: 0.95, Matched: True
    """

    name: str = "fuzzy_match"
    method: str = Field(default="ratio", description="Matching method")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Match threshold")
    normalize_text: bool = Field(default=True, description="Whether to normalize text")

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute fuzzy match score

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result containing score and match status
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Normalize text if enabled
        if input_data.normalize and self.normalize_text:
            if isinstance(reference, list):
                reference = [self._normalize(r) for r in reference]
            else:
                reference = self._normalize(reference)
            candidate = self._normalize(candidate)

        # Handle multiple reference texts
        if isinstance(reference, list):
            scores = [self._compute_single(candidate, ref) for ref in reference]
            score = max(scores)
            details = {
                "scores_per_reference": scores,
                "num_references": len(reference),
            }
        else:
            score = self._compute_single(candidate, reference)
            details = {}

        # Add common details
        details.update(
            {
                "method": self.method,
                "threshold": self.threshold,
                "matched": score >= self.threshold,
            }
        )

        return MetricResult(
            name=self.name,
            score=score,
            details=details,
            metadata={
                "method": self.method,
                "normalize": input_data.normalize and self.normalize_text,
            },
        )

    def _compute_single(self, candidate: str, reference: str) -> float:
        """
        Compute fuzzy match score for a single reference text

        Args:
            candidate: Candidate text
            reference: Reference text

        Returns:
            float: Similarity score [0, 1]
        """
        if self.method == "ratio":
            return Levenshtein.ratio(candidate, reference)
        elif self.method == "partial_ratio":
            return self._partial_ratio(candidate, reference)
        elif self.method == "token_sort_ratio":
            return self._token_sort_ratio(candidate, reference)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Use 'ratio', 'partial_ratio', or 'token_sort_ratio'"
            )

    def _partial_ratio(self, s1: str, s2: str) -> float:
        """
        Partial string matching

        Finds the best matching position of the shorter text within the longer text.

        Args:
            s1: First text
            s2: Second text

        Returns:
            float: Best partial match score
        """
        if len(s1) == 0 or len(s2) == 0:
            return 0.0 if s1 != s2 else 1.0

        # Determine shorter and longer texts
        shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)

        # Sliding window to find best match
        m = len(shorter)
        max_ratio = 0.0

        for i in range(len(longer) - m + 1):
            ratio = Levenshtein.ratio(shorter, longer[i : i + m])
            max_ratio = max(max_ratio, ratio)

        return max_ratio

    def _token_sort_ratio(self, s1: str, s2: str) -> float:
        """
        Token order-independent fuzzy matching

        Sorts tokens of both texts before comparison, ignoring word order differences.

        Args:
            s1: First text
            s2: Second text

        Returns:
            float: Token order-independent similarity score
        """
        tokens1 = sorted(s1.split())
        tokens2 = sorted(s2.split())
        return Levenshtein.ratio(" ".join(tokens1), " ".join(tokens2))


@register_metric("fuzzy_match_simple")
class SimpleFuzzyMatchMetric(BaseMetric):
    """
    Simple Fuzzy Match

    Based on OpenAI Evals implementation, using normalization + substring containment check.

    Attributes:
        name: Metric name

    Example:
        >>> metric = SimpleFuzzyMatchMetric()
        >>> input_data = ComparisonInput(
        ...     reference="the cat",
        ...     candidate="cat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(result.score)
        1.0
    """

    name: str = "fuzzy_match_simple"
    normalize_text: bool = Field(
        default=True, description="Normalization enabled by default"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute simple fuzzy match

        Uses bidirectional substring containment check after normalization.

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Normalize text
        if input_data.normalize and self.normalize_text:
            if isinstance(reference, list):
                reference = [self._normalize(r) for r in reference]
            else:
                reference = self._normalize(reference)
            candidate = self._normalize(candidate)

        # Handle multiple reference texts
        if isinstance(reference, list):
            matches = [self._fuzzy_match_simple(candidate, ref) for ref in reference]
            score = 1.0 if any(matches) else 0.0
            details = {
                "matches_per_reference": matches,
                "any_match": any(matches),
            }
        else:
            matched = self._fuzzy_match_simple(candidate, reference)
            score = 1.0 if matched else 0.0
            details = {"matched": matched}

        return MetricResult(name=self.name, score=score, details=details)

    def _fuzzy_match_simple(self, s1: str, s2: str) -> bool:
        """
        Simple fuzzy match: bidirectional substring check

        Args:
            s1: First text
            s2: Second text

        Returns:
            bool: Whether texts match
        """
        # Handle empty strings
        if s1 == "" or s2 == "":
            return s1 == s2

        # Bidirectional containment check (based on OpenAI Evals implementation)
        return s1 in s2 or s2 in s1


@register_metric("edit_distance")
class EditDistanceMetric(BaseMetric):
    """
    Edit Distance Metric

    Returns normalized Levenshtein edit distance.

    Attributes:
        name: Metric name
        normalize_by_length: Whether to normalize by length

    Example:
        >>> metric = EditDistanceMetric()
        >>> input_data = ComparisonInput(
        ...     reference="kitten",
        ...     candidate="sitting"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Distance: {result.raw_score}, Normalized: {result.score}")
    """

    name: str = "edit_distance"
    normalize_by_length: bool = Field(
        default=True, description="Whether to normalize by length"
    )
    normalize_text: bool = Field(
        default=False, description="Edit distance typically does not normalize text"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute edit distance

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result, score is normalized similarity (1 - normalized distance)
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Handle multiple reference texts (take minimum distance)
        if isinstance(reference, list):
            distances = [Levenshtein.distance(candidate, ref) for ref in reference]
            raw_distance = min(distances)
            max_len = max(len(candidate), max(len(ref) for ref in reference))
            details = {
                "distances_per_reference": distances,
                "min_distance": raw_distance,
            }
        else:
            raw_distance = Levenshtein.distance(candidate, reference)
            max_len = max(len(candidate), len(reference))
            details = {}

        # Normalize to similarity score
        if self.normalize_by_length and max_len > 0:
            normalized_score = 1.0 - (raw_distance / max_len)
        else:
            normalized_score = 1.0 / (1.0 + raw_distance)

        details.update(
            {
                "raw_distance": raw_distance,
                "max_length": max_len,
                "normalize_by_length": self.normalize_by_length,
            }
        )

        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, normalized_score)),  # Ensure within [0, 1] range
            raw_score=float(raw_distance),
            details=details,
        )


__all__ = [
    "FuzzyMatchMetric",
    "SimpleFuzzyMatchMetric",
    "EditDistanceMetric",
]
