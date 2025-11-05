"""
Custom ROUGE N-gram Metrics

Custom implementation of ROUGE-3, ROUGE-4, and ROUGE-5 metrics.
Since the standard rouge-score library only supports ROUGE-1, ROUGE-2, and ROUGE-L,
these metrics implement n-gram overlap calculation following the ROUGE methodology.
"""

from collections import Counter
from typing import Dict, List, Tuple

from pydantic import Field

from rm_gallery.core.metrics.base import BaseNLPMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


class ROUGENGramMetric(BaseNLPMetric):
    """
    Base class for ROUGE N-gram metrics

    ROUGE-N measures n-gram overlap between candidate and reference texts.
    It calculates precision, recall, and F1-score based on n-gram matching.

    Attributes:
        name: Metric name
        n: N-gram size
        use_stemming: Whether to use stemming (currently not implemented)
        score_type: Which score to return (precision/recall/fmeasure)

    References:
        Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
        Text Summarization Branches Out.
    """

    name: str = "rouge_ngram"
    n: int = Field(default=3, ge=1, description="N-gram size")
    use_stemming: bool = Field(default=False, description="Whether to use stemming")
    score_type: str = Field(
        default="fmeasure", description="Score type: precision/recall/fmeasure"
    )
    normalize_text: bool = Field(
        default=False, description="ROUGE typically does not normalize"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute ROUGE N-gram score

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result containing precision, recall, and F1-score
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Handle multiple reference texts
        if isinstance(reference, list):
            # Compute for each reference and take the best score
            all_scores = []
            for ref in reference:
                scores = self._compute_rouge_n(candidate, ref)
                all_scores.append(scores)

            # Take maximum F1-score
            max_idx = max(
                range(len(all_scores)), key=lambda i: all_scores[i]["fmeasure"]
            )
            best_scores = all_scores[max_idx]

            details = {
                "num_references": len(reference),
                "scores_per_reference": all_scores,
                "best_reference_index": max_idx,
            }
        else:
            best_scores = self._compute_rouge_n(candidate, reference)
            details = {}

        # Add detailed scores
        details.update(best_scores)

        # Select score based on score_type
        if self.score_type == "precision":
            score = best_scores["precision"]
        elif self.score_type == "recall":
            score = best_scores["recall"]
        else:  # fmeasure (default)
            score = best_scores["fmeasure"]

        return MetricResult(
            name=self.name,
            score=score,
            details=details,
            metadata={
                "n": self.n,
                "score_type": self.score_type,
                "use_stemming": self.use_stemming,
            },
        )

    def _compute_rouge_n(self, candidate: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE-N scores

        Args:
            candidate: Candidate text
            reference: Reference text

        Returns:
            Dict containing precision, recall, and fmeasure
        """
        # Get n-grams
        ref_ngrams = self._get_ngrams(reference, self.n)
        cand_ngrams = self._get_ngrams(candidate, self.n)

        # Count matches
        ref_counter = Counter(ref_ngrams)
        cand_counter = Counter(cand_ngrams)

        # Calculate overlap
        overlap = cand_counter & ref_counter
        overlap_count = sum(overlap.values())

        # Calculate total counts
        ref_count = sum(ref_counter.values())
        cand_count = sum(cand_counter.values())

        # Calculate precision, recall, and F1
        if cand_count == 0:
            precision = 0.0
        else:
            precision = overlap_count / cand_count

        if ref_count == 0:
            recall = 0.0
        else:
            recall = overlap_count / ref_count

        if precision + recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "fmeasure": fmeasure,
            "overlap_count": overlap_count,
            "reference_count": ref_count,
            "candidate_count": cand_count,
        }

    def _get_ngrams(self, text: str, n: int) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from text

        Args:
            text: Input text
            n: N-gram size

        Returns:
            List of n-grams as tuples
        """
        # Tokenize
        tokens = text.split()

        # Generate n-grams
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams.append(ngram)

        return ngrams


@register_metric("rouge3")
class ROUGE3Metric(ROUGENGramMetric):
    """
    ROUGE-3 Metric (3-gram overlap)

    Measures 3-gram overlap between candidate and reference texts.
    Useful for evaluating longer phrase matching.

    Attributes:
        name: Metric name
        n: Fixed at 3 for ROUGE-3

    Example:
        >>> metric = ROUGE3Metric()
        >>> input_data = ComparisonInput(
        ...     reference="the quick brown fox jumps over the lazy dog",
        ...     candidate="the quick brown fox jumps over the lazy cat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"ROUGE-3: {result.score:.4f}")

    References:
        Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
    """

    name: str = "rouge3"
    n: int = Field(default=3, frozen=True, description="Fixed at 3 for ROUGE-3")


@register_metric("rouge4")
class ROUGE4Metric(ROUGENGramMetric):
    """
    ROUGE-4 Metric (4-gram overlap)

    Measures 4-gram overlap between candidate and reference texts.
    Useful for evaluating longer phrase and sentence structure matching.

    Attributes:
        name: Metric name
        n: Fixed at 4 for ROUGE-4

    Example:
        >>> metric = ROUGE4Metric()
        >>> input_data = ComparisonInput(
        ...     reference="the quick brown fox jumps over the lazy dog today",
        ...     candidate="the quick brown fox jumps over the lazy dog yesterday"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"ROUGE-4: {result.score:.4f}")
    """

    name: str = "rouge4"
    n: int = Field(default=4, frozen=True, description="Fixed at 4 for ROUGE-4")


@register_metric("rouge5")
class ROUGE5Metric(ROUGENGramMetric):
    """
    ROUGE-5 Metric (5-gram overlap)

    Measures 5-gram overlap between candidate and reference texts.
    Useful for evaluating very long phrase and sentence structure matching.

    Attributes:
        name: Metric name
        n: Fixed at 5 for ROUGE-5

    Example:
        >>> metric = ROUGE5Metric()
        >>> input_data = ComparisonInput(
        ...     reference="the quick brown fox jumps over the lazy dog every single day",
        ...     candidate="the quick brown fox jumps over the lazy dog every single time"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"ROUGE-5: {result.score:.4f}")
    """

    name: str = "rouge5"
    n: int = Field(default=5, frozen=True, description="Fixed at 5 for ROUGE-5")


@register_metric("rouge_n")
class CustomROUGENMetric(ROUGENGramMetric):
    """
    Custom ROUGE-N Metric

    Allows specifying custom n-gram size for ROUGE evaluation.

    Attributes:
        name: Metric name
        n: Custom n-gram size (can be any positive integer)

    Example:
        >>> # ROUGE-6
        >>> metric = CustomROUGENMetric(n=6)
        >>> result = metric.compute(input_data)

        >>> # ROUGE-10
        >>> metric = CustomROUGENMetric(n=10)
        >>> result = metric.compute(input_data)
    """

    name: str = "rouge_n"
    # n is inherited from parent class and can be set to any value


__all__ = [
    "ROUGENGramMetric",
    "ROUGE3Metric",
    "ROUGE4Metric",
    "ROUGE5Metric",
    "CustomROUGENMetric",
]
