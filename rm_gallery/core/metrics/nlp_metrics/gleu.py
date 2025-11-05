"""
GLEU Metric

GLEU (Google-BLEU) is a variant of BLEU proposed by Google.
Optimized for sentence-level evaluation, particularly suitable for grammatical error correction tasks.
"""

from pydantic import Field

from rm_gallery.core.metrics.base import BaseNLPMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("gleu")
class GLEUMetric(BaseNLPMetric):
    """
    GLEU Metric

    Implemented using NLTK's sentence_gleu.
    GLEU is a sentence-level variant of BLEU with the following improvements:
    1. Better suited for sentence-level evaluation
    2. More friendly to short sentences
    3. Takes recall into account

    Attributes:
        name: Metric name
        min_len: Minimum n-gram length
        max_len: Maximum n-gram length

    Example:
        >>> metric = GLEUMetric()
        >>> input_data = ComparisonInput(
        ...     reference="the cat sat on the mat",
        ...     candidate="the cat is on the mat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"GLEU: {result.score:.4f}")

    References:
        Wu et al. (2016). Google's Neural Machine Translation System.
    """

    name: str = "gleu"
    min_len: int = Field(default=1, ge=1, description="Minimum n-gram length")
    max_len: int = Field(default=4, ge=1, description="Maximum n-gram length")
    normalize_text: bool = Field(
        default=False, description="GLEU typically does not normalize"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute GLEU score

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        try:
            from nltk.translate.gleu_score import sentence_gleu
        except ImportError:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "error": "NLTK not installed",
                    "message": "Please install: pip install nltk",
                },
            )

        reference = input_data.reference
        candidate = input_data.candidate

        # Tokenization
        candidate_tokens = candidate.split()

        # Handle multiple reference texts
        if isinstance(reference, list):
            reference_tokens = [ref.split() for ref in reference]
        else:
            reference_tokens = [reference.split()]

        # Compute GLEU
        try:
            score = sentence_gleu(
                reference_tokens,
                candidate_tokens,
                min_len=self.min_len,
                max_len=self.max_len,
            )

            details = {
                "min_len": self.min_len,
                "max_len": self.max_len,
                "num_references": len(reference_tokens),
                "candidate_length": len(candidate_tokens),
            }

            return MetricResult(
                name=self.name,
                score=score,
                details=details,
                metadata={
                    "min_len": self.min_len,
                    "max_len": self.max_len,
                },
            )
        except Exception as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": str(e)},
            )


@register_metric("corpus_gleu")
class CorpusGLEUMetric(BaseNLPMetric):
    """
    Corpus-level GLEU Metric

    Used for evaluating translation quality of an entire corpus.

    Note:
        This metric requires special batch processing logic.

    Attributes:
        name: Metric name
    """

    name: str = "corpus_gleu"

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute corpus-level GLEU

        Note: This metric requires batch input

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        try:
            from nltk.translate.gleu_score import corpus_gleu  # noqa: F401
        except ImportError:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "NLTK not installed"},
            )

        # For single sample, fall back to sentence-level GLEU
        gleu_metric = GLEUMetric()
        return gleu_metric.compute(input_data)


@register_metric("chrf")
class ChrFMetric(BaseNLPMetric):
    """
    ChrF Metric

    Character n-gram F-score, an F-score based on character-level n-grams.
    Particularly suitable for morphologically rich languages and low-resource languages.

    Attributes:
        name: Metric name
        n: Character n-gram size
        beta: Beta parameter for F-score (beta=1 for F1, beta=2 for F2, etc.)

    Example:
        >>> metric = ChrFMetric(n=6, beta=2)
        >>> result = metric.compute(input_data)
    """

    name: str = "chrf"
    n: int = Field(default=6, ge=1, description="Character n-gram size")
    beta: float = Field(default=2.0, ge=0.0, description="F-score beta parameter")
    normalize_text: bool = Field(
        default=False, description="Typically does not normalize"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute ChrF score

        Note: Requires sacrebleu library support

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        try:
            from sacrebleu.metrics import CHRF
        except ImportError:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "error": "sacrebleu not installed",
                    "message": "Please install: pip install sacrebleu",
                },
            )

        reference = input_data.reference
        candidate = input_data.candidate

        # sacrebleu ChrF requires reference text format
        if isinstance(reference, str):
            refs = [[reference]]
        else:
            refs = [[ref] for ref in reference]

        # Compute ChrF
        try:
            chrf = CHRF(char_order=self.n, beta=self.beta)
            result = chrf.corpus_score([candidate], refs)

            # sacrebleu returns scores in 0-100 range
            normalized_score = result.score / 100.0

            return MetricResult(
                name=self.name,
                score=normalized_score,
                raw_score=result.score,
                details={
                    "char_order": self.n,
                    "beta": self.beta,
                },
                metadata={
                    "n": self.n,
                    "beta": self.beta,
                },
            )
        except Exception as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": str(e)},
            )


__all__ = ["GLEUMetric", "CorpusGLEUMetric", "ChrFMetric"]
