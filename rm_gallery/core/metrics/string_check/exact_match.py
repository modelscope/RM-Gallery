"""
Exact Match Metrics

Exact matching metrics for checking if candidate text matches reference text exactly.
"""

from pydantic import Field

from rm_gallery.core.metrics.base import BaseStringMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("exact_match")
class ExactMatchMetric(BaseStringMetric):
    """
    Exact Match Metric

    Checks if the candidate text exactly matches the reference text.

    Attributes:
        name: Metric name
        case_sensitive: Whether to perform case-sensitive matching
        ignore_whitespace: Whether to ignore whitespace differences

    Example:
        >>> metric = ExactMatchMetric(case_sensitive=False)
        >>> input_data = ComparisonInput(
        ...     reference="Hello World",
        ...     candidate="hello world"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Matched: {result.score}")
        Matched: 1.0
    """

    name: str = "exact_match"
    case_sensitive: bool = Field(
        default=True, description="Whether to perform case-sensitive matching"
    )
    ignore_whitespace: bool = Field(
        default=False, description="Whether to ignore whitespace"
    )
    normalize_text: bool = Field(
        default=False, description="Exact match typically does not normalize"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute exact match

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result, score is 1.0 (matched) or 0.0 (not matched)
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Preprocess
        candidate_processed = self._preprocess(candidate)

        # Handle multiple reference texts
        if isinstance(reference, list):
            references_processed = [self._preprocess(ref) for ref in reference]
            matched = any(
                candidate_processed == ref_proc for ref_proc in references_processed
            )
            details = {
                "num_references": len(reference),
                "matched_reference_index": (
                    next(
                        (
                            i
                            for i, ref_proc in enumerate(references_processed)
                            if candidate_processed == ref_proc
                        ),
                        None,
                    )
                ),
            }
        else:
            reference_processed = self._preprocess(reference)
            matched = candidate_processed == reference_processed
            details = {}

        details.update(
            {
                "matched": matched,
                "case_sensitive": self.case_sensitive,
                "ignore_whitespace": self.ignore_whitespace,
            }
        )

        return MetricResult(
            name=self.name,
            score=1.0 if matched else 0.0,
            details=details,
            metadata={
                "case_sensitive": self.case_sensitive,
                "ignore_whitespace": self.ignore_whitespace,
            },
        )

    def _preprocess(self, text: str) -> str:
        """
        Preprocess text

        Args:
            text: Text to be processed

        Returns:
            str: Processed text
        """
        if not self.case_sensitive:
            text = text.lower()

        if self.ignore_whitespace:
            text = "".join(text.split())

        return text


@register_metric("prefix_match")
class PrefixMatchMetric(BaseStringMetric):
    """
    Prefix Match Metric

    Checks if the candidate text starts with the reference text.
    Similar to OpenAI Evals' Match metric.

    Attributes:
        name: Metric name
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> metric = PrefixMatchMetric()
        >>> input_data = ComparisonInput(
        ...     reference="Hello",
        ...     candidate="Hello World"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Matched: {result.score}")
        Matched: 1.0
    """

    name: str = "prefix_match"
    case_sensitive: bool = Field(
        default=True, description="Whether to perform case-sensitive matching"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute prefix match

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
            matched = any(candidate.startswith(ref) for ref in references)
            details = {
                "num_references": len(reference),
                "matched_reference_index": (
                    next(
                        (
                            i
                            for i, ref in enumerate(references)
                            if candidate.startswith(ref)
                        ),
                        None,
                    )
                ),
            }
        else:
            if not self.case_sensitive:
                reference = reference.lower()
            matched = candidate.startswith(reference)
            details = {}

        details.update({"matched": matched, "case_sensitive": self.case_sensitive})

        return MetricResult(
            name=self.name,
            score=1.0 if matched else 0.0,
            details=details,
        )


@register_metric("suffix_match")
class SuffixMatchMetric(BaseStringMetric):
    """
    Suffix Match Metric

    Checks if the candidate text ends with the reference text.

    Attributes:
        name: Metric name
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> metric = SuffixMatchMetric()
        >>> input_data = ComparisonInput(
        ...     reference="World",
        ...     candidate="Hello World"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Matched: {result.score}")
        Matched: 1.0
    """

    name: str = "suffix_match"
    case_sensitive: bool = Field(
        default=True, description="Whether to perform case-sensitive matching"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute suffix match

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
            matched = any(candidate.endswith(ref) for ref in references)
            details = {"num_references": len(reference)}
        else:
            if not self.case_sensitive:
                reference = reference.lower()
            matched = candidate.endswith(reference)
            details = {}

        details.update({"matched": matched, "case_sensitive": self.case_sensitive})

        return MetricResult(
            name=self.name,
            score=1.0 if matched else 0.0,
            details=details,
        )


@register_metric("regex_match")
class RegexMatchMetric(BaseStringMetric):
    """
    Regular Expression Match Metric

    Matches candidate text using regular expression patterns.

    Attributes:
        name: Metric name
        pattern: Regular expression pattern (if not provided, uses reference as pattern)
        case_sensitive: Whether to perform case-sensitive matching

    Example:
        >>> metric = RegexMatchMetric(pattern=r"\\d{3}-\\d{4}")
        >>> input_data = ComparisonInput(
        ...     reference="",  # pattern already specified at initialization
        ...     candidate="My phone is 123-4567"
        ... )
        >>> result = metric.compute(input_data)
    """

    name: str = "regex_match"
    pattern: str = Field(default="", description="Regular expression pattern")
    case_sensitive: bool = Field(
        default=True, description="Whether to perform case-sensitive matching"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute regular expression match

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        import re

        candidate = input_data.candidate

        # If no preset pattern, use reference as pattern
        pattern = self.pattern if self.pattern else input_data.reference

        # Compile regular expression
        flags = 0 if self.case_sensitive else re.IGNORECASE

        try:
            if isinstance(pattern, list):
                # Multiple patterns: any match is acceptable
                matches = []
                for pat in pattern:
                    regex = re.compile(pat, flags)
                    match = regex.search(candidate)
                    matches.append(match is not None)

                matched = any(matches)
                details = {
                    "num_patterns": len(pattern),
                    "matches_per_pattern": matches,
                }
            else:
                regex = re.compile(pattern, flags)
                match = regex.search(candidate)
                matched = match is not None
                details = {"match_groups": match.groups() if match else None}

            details.update(
                {
                    "matched": matched,
                    "pattern": pattern
                    if not isinstance(pattern, list)
                    else f"{len(pattern)} patterns",
                    "case_sensitive": self.case_sensitive,
                }
            )

            return MetricResult(
                name=self.name,
                score=1.0 if matched else 0.0,
                details=details,
            )
        except re.error as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": f"Invalid regex pattern: {str(e)}"},
            )


__all__ = [
    "ExactMatchMetric",
    "PrefixMatchMetric",
    "SuffixMatchMetric",
    "RegexMatchMetric",
]
