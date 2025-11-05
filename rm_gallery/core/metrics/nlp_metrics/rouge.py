"""
ROUGE Metric

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric,
primarily used for automatic summarization evaluation. Supports ROUGE-1, ROUGE-2, ROUGE-L and other variants.
"""

from typing import List

import numpy as np
from pydantic import Field, PrivateAttr
from rouge_score import rouge_scorer

from rm_gallery.core.metrics.base import BaseNLPMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("rouge")
class ROUGEMetric(BaseNLPMetric):
    """
    ROUGE Metric (supports multiple variants)

    Implemented using rouge_score library. Supports:
    - ROUGE-1: Word overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence
    - ROUGE-Lsum: Sentence-level longest common subsequence

    Attributes:
        name: Metric name
        rouge_types: List of ROUGE types
        use_stemmer: Whether to use stemming
        use_aggregator: Whether to aggregate multiple ROUGE scores

    Example:
        >>> metric = ROUGEMetric(rouge_types=["rouge1", "rouge2", "rougeL"])
        >>> input_data = ComparisonInput(
        ...     reference="the cat is on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"ROUGE: {result.score:.4f}")

    References:
        Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
    """

    name: str = "rouge"
    rouge_types: List[str] = Field(
        default=["rouge1", "rouge2", "rougeL"], description="List of ROUGE types"
    )
    use_stemmer: bool = Field(default=True, description="Whether to use stemming")
    use_aggregator: bool = Field(
        default=True, description="Whether to aggregate scores"
    )
    score_key: str = Field(
        default="fmeasure", description="Which score to use: precision/recall/fmeasure"
    )
    normalize_text: bool = Field(
        default=False, description="ROUGE typically does not normalize"
    )

    # Private attribute for ROUGE scorer instance
    _scorer: rouge_scorer.RougeScorer = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize ROUGE scorer
        self._scorer = rouge_scorer.RougeScorer(
            self.rouge_types, use_stemmer=self.use_stemmer
        )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute ROUGE score

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Handle multiple reference texts
        if isinstance(reference, list):
            # Compute ROUGE for each reference, take maximum value
            all_scores = []
            for ref in reference:
                scores = self._scorer.score(ref, candidate)
                all_scores.append(scores)

            # Aggregate: take maximum F1 for each ROUGE type
            aggregated = {}
            for rouge_type in self.rouge_types:
                max_score = max(
                    self._get_score_value(scores[rouge_type]) for scores in all_scores
                )
                aggregated[rouge_type] = max_score

            details = {
                "num_references": len(reference),
                "aggregation_method": "max",
            }
        else:
            scores = self._scorer.score(reference, candidate)
            aggregated = {
                rouge_type: self._get_score_value(scores[rouge_type])
                for rouge_type in self.rouge_types
            }
            details = {}

        # Add detailed scores
        details.update(aggregated)

        # Compute overall score (average of all ROUGE types)
        if self.use_aggregator:
            avg_score = sum(aggregated.values()) / len(aggregated)
        else:
            # Use only the first ROUGE type score
            avg_score = aggregated[self.rouge_types[0]]

        return MetricResult(
            name=self.name,
            score=avg_score,
            details=details,
            metadata={
                "rouge_types": self.rouge_types,
                "use_stemmer": self.use_stemmer,
                "score_key": self.score_key,
            },
        )

    def _get_score_value(self, score_obj) -> float:
        """
        Extract score value from Score object

        Args:
            score_obj: rouge_score.scoring.Score object

        Returns:
            float: Score value
        """
        if self.score_key == "precision":
            return score_obj.precision
        elif self.score_key == "recall":
            return score_obj.recall
        else:  # fmeasure (default)
            return score_obj.fmeasure


@register_metric("rouge1")
class ROUGE1Metric(ROUGEMetric):
    """ROUGE-1 Metric (word-level overlap)"""

    name: str = "rouge1"
    rouge_types: List[str] = Field(default=["rouge1"], description="Use ROUGE-1 only")


@register_metric("rouge2")
class ROUGE2Metric(ROUGEMetric):
    """ROUGE-2 Metric (bigram overlap)"""

    name: str = "rouge2"
    rouge_types: List[str] = Field(default=["rouge2"], description="Use ROUGE-2 only")


@register_metric("rougeL")
class ROUGELMetric(ROUGEMetric):
    """ROUGE-L Metric (longest common subsequence)"""

    name: str = "rougeL"
    rouge_types: List[str] = Field(default=["rougeL"], description="Use ROUGE-L only")


@register_metric("rouge_multi")
class MultiROUGEMetric(BaseNLPMetric):
    """
    Multi-ROUGE Evaluation

    Computes and returns detailed scores for all ROUGE types separately.

    Attributes:
        name: Metric name
        include_rouge_types: ROUGE types to include

    Example:
        >>> metric = MultiROUGEMetric()
        >>> result = metric.compute(input_data)
        >>> print(result.details)  # Contains detailed scores for all ROUGE types
    """

    name: str = "rouge_multi"
    include_rouge_types: List[str] = Field(
        default=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        description="ROUGE types to include",
    )
    use_stemmer: bool = Field(default=True, description="Whether to use stemming")

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute multi-ROUGE scores

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Detailed results containing all ROUGE variants
        """
        # Create separate metric for each ROUGE type
        results = {}
        scores = []

        for rouge_type in self.include_rouge_types:
            metric = ROUGEMetric(rouge_types=[rouge_type], use_stemmer=self.use_stemmer)
            result = metric.compute(input_data)
            results[rouge_type] = {
                "score": result.score,
                "details": result.details.get(rouge_type, {}),
            }
            scores.append(result.score)

        # Compute average score
        avg_score = np.mean(scores) if scores else 0.0

        return MetricResult(
            name=self.name,
            score=avg_score,
            details=results,
            metadata={
                "rouge_types": self.include_rouge_types,
                "use_stemmer": self.use_stemmer,
            },
        )


__all__ = [
    "ROUGEMetric",
    "ROUGE1Metric",
    "ROUGE2Metric",
    "ROUGELMetric",
    "MultiROUGEMetric",
]
