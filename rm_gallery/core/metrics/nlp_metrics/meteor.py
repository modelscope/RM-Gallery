"""
METEOR Metric

METEOR (Metric for Evaluation of Translation with Explicit ORdering),
a translation evaluation metric that comprehensively considers precision, recall,
morphological variations, and semantic information.
"""

from pydantic import Field

from rm_gallery.core.metrics.base import BaseNLPMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("meteor")
class METEORMetric(BaseNLPMetric):
    """
    METEOR Metric

    Implemented using NLTK's meteor_score.
    METEOR has the following improvements over BLEU:
    1. Considers both precision and recall
    2. Supports stemming and synonym matching
    3. Considers word order (via fragmentation penalty)

    Attributes:
        name: Metric name
        alpha: Precision weight parameter
        beta: Recall weight parameter
        gamma: Fragmentation penalty weight parameter

    Example:
        >>> metric = METEORMetric()
        >>> input_data = ComparisonInput(
        ...     reference="the cat sat on the mat",
        ...     candidate="on the mat sat the cat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"METEOR: {result.score:.4f}")

    References:
        Banerjee & Lavie (2005). METEOR: An Automatic Metric for MT Evaluation
        with Improved Correlation with Human Judgments.
    """

    name: str = "meteor"
    alpha: float = Field(default=0.9, ge=0.0, le=1.0, description="Precision weight")
    beta: float = Field(default=3.0, ge=0.0, description="Recall weight")
    gamma: float = Field(
        default=0.5, ge=0.0, description="Fragmentation penalty weight"
    )
    normalize_text: bool = Field(
        default=False, description="METEOR typically does not normalize"
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Attempt to download necessary NLTK data
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Ensure NLTK data is downloaded"""
        try:
            import nltk

            # Attempt to download necessary data packages
            for package in ["wordnet", "punkt", "omw-1.4"]:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    # Best-effort; missing data will be surfaced during compute()
                    pass
        except ImportError:
            pass

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute METEOR score

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        try:
            from nltk.translate.meteor_score import meteor_score
        except ImportError:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "error": "NLTK not installed or missing dependencies",
                    "message": "Please install: pip install nltk",
                },
            )

        reference = input_data.reference
        candidate = input_data.candidate

        # Tokenization
        candidate_tokens = candidate.split()

        # Handle multiple reference texts
        if isinstance(reference, list):
            scores = []
            for ref in reference:
                ref_tokens = ref.split()
                try:
                    score = meteor_score(
                        [ref_tokens],
                        candidate_tokens,
                        alpha=self.alpha,
                        beta=self.beta,
                        gamma=self.gamma,
                    )
                    scores.append(score)
                except Exception as e:
                    # Skip if calculation fails for a reference text
                    continue

            if not scores:
                return MetricResult(
                    name=self.name,
                    score=0.0,
                    details={"error": "Failed to compute METEOR for all references"},
                )

            # Take the maximum score
            final_score = max(scores)
            details = {
                "scores_per_reference": scores,
                "max_score": final_score,
                "min_score": min(scores),
                "avg_score": sum(scores) / len(scores),
                "num_references": len(reference),
            }
        else:
            reference_tokens = reference.split()
            try:
                final_score = meteor_score(
                    [reference_tokens],
                    candidate_tokens,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=self.gamma,
                )
                details = {}
            except Exception as e:
                return MetricResult(
                    name=self.name,
                    score=0.0,
                    details={"error": str(e)},
                )

        details.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            }
        )

        return MetricResult(
            name=self.name,
            score=final_score,
            details=details,
            metadata={
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
        )


@register_metric("meteor_simple")
class SimpleMETEORMetric(METEORMetric):
    """
    Simplified METEOR

    Uses default parameters, does not support synonym and stemming matching (faster).

    Example:
        >>> metric = SimpleMETEORMetric()
        >>> result = metric.compute(input_data)
    """

    name: str = "meteor_simple"
    # Use default parameters


__all__ = ["METEORMetric", "SimpleMETEORMetric"]
