"""
F1 Score Metric

Token-based F1 score calculation, following OpenAI Evals implementation.
"""

from collections import Counter

from pydantic import Field

from rm_gallery.core.metrics.base import BaseMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("f1_score")
class F1ScoreMetric(BaseMetric):
    """
    Token-based F1 Score Metric

    Calculates F1 score based on token overlap between candidate and reference texts.
    Based on OpenAI Evals fuzzy_match implementation.

    The metric:
    1. Normalizes and tokenizes both texts
    2. Calculates token overlap using Counter
    3. Computes precision, recall, and F1 score
    4. For multiple references, returns the maximum F1 score

    Attributes:
        name: Metric name
        normalize_text: Whether to normalize text before tokenization

    Example:
        >>> metric = F1ScoreMetric()
        >>> input_data = ComparisonInput(
        ...     reference="the cat is on the mat",
        ...     candidate="cat on mat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"F1 Score: {result.score:.4f}")
        F1 Score: 0.6667

    Example with multiple references:
        >>> metric = F1ScoreMetric()
        >>> input_data = ComparisonInput(
        ...     reference=["the quick brown fox", "a fast brown dog"],
        ...     candidate="quick brown fox"
        ... )
        >>> result = metric.compute(input_data)
        >>> # Takes maximum F1 across all references
    """

    name: str = "f1_score"
    normalize_text: bool = Field(default=True, description="Whether to normalize text")

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute F1 score

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result containing F1 score and details
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Handle multiple reference texts
        if isinstance(reference, list):
            scores = []
            precision_list = []
            recall_list = []

            for ref in reference:
                score, precision, recall = self._compute_single_f1(
                    candidate, ref, input_data.normalize
                )
                scores.append(score)
                precision_list.append(precision)
                recall_list.append(recall)

            # Take maximum F1 score (following OpenAI Evals implementation)
            max_idx = scores.index(max(scores)) if scores else 0
            final_score = scores[max_idx] if scores else 0.0

            details = {
                "f1_scores_per_reference": scores,
                "precision_scores": precision_list,
                "recall_scores": recall_list,
                "best_reference_idx": max_idx,
                "num_references": len(reference),
            }
        else:
            final_score, precision, recall = self._compute_single_f1(
                candidate, reference, input_data.normalize
            )
            details = {
                "precision": precision,
                "recall": recall,
            }

        return MetricResult(
            name=self.name,
            score=final_score,
            details=details,
            metadata={
                "normalize": input_data.normalize and self.normalize_text,
            },
        )

    def _compute_single_f1(
        self, candidate: str, reference: str, normalize: bool = True
    ) -> tuple[float, float, float]:
        """
        Compute F1 score for a single reference text

        Following OpenAI Evals implementation:
        https://github.com/openai/evals/blob/main/evals/elsuite/utils.py#L75-L88

        Args:
            candidate: Candidate text
            reference: Reference text
            normalize: Whether to normalize text

        Returns:
            tuple[float, float, float]: (f1_score, precision, recall)
        """
        # Normalize and tokenize
        if normalize and self.normalize_text:
            candidate_norm = self._normalize(candidate)
            reference_norm = self._normalize(reference)
        else:
            candidate_norm = candidate
            reference_norm = reference

        # Tokenize by splitting on whitespace
        candidate_tokens = candidate_norm.split()
        reference_tokens = reference_norm.split()

        # Handle empty cases
        if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
            if len(candidate_tokens) == 0 and len(reference_tokens) == 0:
                return 1.0, 1.0, 1.0  # Both empty - perfect match
            else:
                return 0.0, 0.0, 0.0  # One empty - no match

        # Calculate token overlap using Counter (following OpenAI Evals)
        candidate_counter = Counter(candidate_tokens)
        reference_counter = Counter(reference_tokens)
        common = candidate_counter & reference_counter
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0, 0.0, 0.0

        # Calculate precision and recall
        precision = 1.0 * num_same / len(candidate_tokens)
        recall = 1.0 * num_same / len(reference_tokens)

        # Calculate F1 score
        f1 = (2 * precision * recall) / (precision + recall)

        return f1, precision, recall


@register_metric("token_f1")
class TokenF1Metric(F1ScoreMetric):
    """
    Alias for F1ScoreMetric

    Provides a more descriptive name emphasizing token-based calculation.
    """

    name: str = "token_f1"


__all__ = ["F1ScoreMetric", "TokenF1Metric"]
