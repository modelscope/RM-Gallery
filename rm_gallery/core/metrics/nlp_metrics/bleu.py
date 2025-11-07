"""
BLEU Metric

BLEU (Bilingual Evaluation Understudy) metric, the standard metric for machine translation evaluation.
Implemented based on the sacrebleu library, supporting standard machine translation evaluation.
"""

from pydantic import Field, PrivateAttr
from sacrebleu.metrics import BLEU

from rm_gallery.core.metrics.base import BaseNLPMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("bleu")
class BLEUMetric(BaseNLPMetric):
    """
    BLEU Metric

    Standard BLEU scoring implemented using the sacrebleu library.
    BLEU evaluates translation quality through n-gram precision and brevity penalty.

    Attributes:
        name: Metric name
        max_ngram_order: Maximum n-gram order (typically 4)
        smooth_method: Smoothing method (exp/floor/add-k/none)
        effective_order: Whether to use effective order

    Example:
        >>> metric = BLEUMetric(max_ngram_order=4)
        >>> input_data = ComparisonInput(
        ...     reference="the cat is on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"BLEU: {result.score:.4f}")
        BLEU: 1.0000

    References:
        Papineni et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation.
    """

    name: str = "bleu"
    max_ngram_order: int = Field(
        default=4, ge=1, le=4, description="Maximum n-gram order"
    )
    smooth_method: str = Field(default="exp", description="Smoothing method")
    effective_order: bool = Field(
        default=True, description="Whether to use effective order"
    )
    normalize_text: bool = Field(
        default=False, description="BLEU typically does not normalize text"
    )

    # Private attribute for sacrebleu BLEU instance
    _bleu: BLEU = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize sacrebleu BLEU object
        self._bleu = BLEU(
            max_ngram_order=self.max_ngram_order,
            smooth_method=self.smooth_method,
            effective_order=self.effective_order,
        )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute BLEU score

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result, score is normalized BLEU score [0, 1]
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # sacrebleu requires reference text format: List[List[str]]
        if isinstance(reference, str):
            refs = [[reference]]
        else:
            # Multiple references: each reference as a list
            refs = [[ref] for ref in reference]

        # Compute BLEU score
        try:
            result = self._bleu.corpus_score([candidate], refs)

            # sacrebleu returns scores in 0-100 range
            normalized_score = result.score / 100.0
            # Clamp to [0, 1] range to handle floating point precision issues
            normalized_score = max(0.0, min(1.0, normalized_score))

            details = {
                "precisions": [
                    p / 100.0 for p in result.precisions
                ],  # Normalize precisions to [0, 1]
                "bp": result.bp,  # Brevity penalty
                "sys_len": result.sys_len,  # System (candidate) length
                "ref_len": result.ref_len,  # Reference length
                "ratio": result.sys_len / result.ref_len if result.ref_len > 0 else 0,
            }

            return MetricResult(
                name=self.name,
                score=normalized_score,
                raw_score=result.score,
                details=details,
                metadata={
                    "max_ngram_order": self.max_ngram_order,
                    "smooth_method": self.smooth_method,
                    "effective_order": self.effective_order,
                },
            )
        except Exception as e:
            # Handle exceptions (e.g., empty text)
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": str(e)},
            )


@register_metric("sentence_bleu")
class SentenceBLEUMetric(BaseNLPMetric):
    """
    Sentence-level BLEU Metric

    Implemented using NLTK's sentence_bleu, suitable for single sentence evaluation.

    Attributes:
        name: Metric name
        weights: Weights for each n-gram order
        smoothing_function: Smoothing function type (1-7)

    Example:
        >>> metric = SentenceBLEUMetric()
        >>> input_data = ComparisonInput(
        ...     reference="the cat sat on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>> result = metric.compute(input_data)
    """

    name: str = "sentence_bleu"
    weights: tuple[float, ...] = Field(
        default=(0.25, 0.25, 0.25, 0.25), description="N-gram weights"
    )
    smoothing_function: int = Field(
        default=1, ge=1, le=7, description="Smoothing function"
    )
    normalize_text: bool = Field(
        default=False, description="Typically no normalization"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute sentence-level BLEU score

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        except ImportError:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "error": "NLTK not installed. Please install: pip install nltk"
                },
            )

        reference = input_data.reference
        candidate = input_data.candidate

        # Tokenize
        candidate_tokens = candidate.split()

        # Handle multiple references
        if isinstance(reference, list):
            reference_tokens = [ref.split() for ref in reference]
        else:
            reference_tokens = [reference.split()]

        # Select smoothing function
        smoothing = SmoothingFunction()
        smooth_func = getattr(smoothing, f"method{self.smoothing_function}")

        # Compute BLEU
        try:
            score = sentence_bleu(
                reference_tokens,
                candidate_tokens,
                weights=self.weights,
                smoothing_function=smooth_func,
            )

            return MetricResult(
                name=self.name,
                score=score,
                details={
                    "weights": self.weights,
                    "smoothing_method": self.smoothing_function,
                    "num_references": len(reference_tokens),
                },
                metadata={
                    "weights": str(self.weights),
                    "smoothing": self.smoothing_function,
                },
            )
        except Exception as e:
            return MetricResult(name=self.name, score=0.0, details={"error": str(e)})


@register_metric("self_bleu")
class SelfBLEUMetric(BaseNLPMetric):
    """
    Self-BLEU Metric

    Used to evaluate the diversity of generated text.
    Computes the average BLEU score of each candidate text against other candidate texts.
    Lower scores indicate higher diversity.

    Note:
        This metric requires multiple candidate texts and is used differently from other metrics.

    Attributes:
        name: Metric name
    """

    name: str = "self_bleu"

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute Self-BLEU

        Note: This metric requires special input format (multiple candidate texts)

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        return MetricResult(
            name=self.name,
            score=0.0,
            details={
                "error": "Self-BLEU requires special handling with multiple candidates",
                "message": "Use a dedicated Self-BLEU evaluator for diversity assessment",
            },
        )


__all__ = ["BLEUMetric", "SentenceBLEUMetric", "SelfBLEUMetric"]
