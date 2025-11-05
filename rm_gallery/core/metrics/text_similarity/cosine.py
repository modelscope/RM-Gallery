"""
Cosine Similarity Metric

Cosine similarity metrics based on TF-IDF or term frequency vectors for calculating text similarity.
"""

from collections import Counter
from typing import Optional

import numpy as np
from pydantic import Field
from sklearn.feature_extraction.text import TfidfVectorizer

from rm_gallery.core.metrics.base import BaseMetric
from rm_gallery.core.metrics.registry import register_metric
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


@register_metric("cosine")
class CosineSimilarityMetric(BaseMetric):
    """
    Cosine Similarity Metric (TF-IDF based)

    Converts text to TF-IDF vectors and then calculates cosine similarity between vectors.

    Attributes:
        name: Metric name
        use_tfidf: Whether to use TF-IDF (otherwise uses simple term frequency)
        ngram_range: N-gram range
        max_features: Maximum number of features

    Example:
        >>> metric = CosineSimilarityMetric(use_tfidf=True)
        >>> input_data = ComparisonInput(
        ...     reference="the cat sat on the mat",
        ...     candidate="the dog sat on the mat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Cosine Similarity: {result.score:.4f}")
    """

    name: str = "cosine"
    use_tfidf: bool = Field(default=True, description="Whether to use TF-IDF")
    ngram_range: tuple[int, int] = Field(default=(1, 2), description="N-gram range")
    max_features: Optional[int] = Field(
        default=None, description="Maximum number of features"
    )
    normalize_text: bool = Field(default=True, description="Whether to normalize text")

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute cosine similarity

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Normalize
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
                "max_score": score,
                "avg_score": np.mean(scores),
            }
        else:
            score = self._compute_single(candidate, reference)
            details = {}

        details.update(
            {
                "use_tfidf": self.use_tfidf,
                "ngram_range": self.ngram_range,
            }
        )

        return MetricResult(
            name=self.name,
            score=score,
            details=details,
            metadata={
                "use_tfidf": self.use_tfidf,
                "ngram_range": str(self.ngram_range),
            },
        )

    def _compute_single(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Cosine similarity [0, 1]
        """
        if self.use_tfidf:
            return self._cosine_tfidf(text1, text2)
        else:
            return self._cosine_simple(text1, text2)

    def _cosine_tfidf(self, text1: str, text2: str) -> float:
        """
        TF-IDF based cosine similarity

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Cosine similarity
        """
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range, max_features=self.max_features
            )
            vectors = vectorizer.fit_transform([text1, text2])
            vec1 = vectors[0].toarray().flatten()
            vec2 = vectors[1].toarray().flatten()
        except Exception as e:
            # Handle empty text or other exceptions
            return 0.0

        return self._cosine_similarity_vectors(vec1, vec2)

    def _cosine_simple(self, text1: str, text2: str) -> float:
        """
        Simple term frequency based cosine similarity

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Cosine similarity
        """
        words1 = text1.split()
        words2 = text2.split()

        # Calculate term frequencies
        counter1 = Counter(words1)
        counter2 = Counter(words2)

        # Build vocabulary
        all_words = set(counter1.keys()) | set(counter2.keys())
        if not all_words:
            return 0.0

        # Build vectors
        vec1 = np.array([counter1.get(word, 0) for word in all_words])
        vec2 = np.array([counter2.get(word, 0) for word in all_words])

        return self._cosine_similarity_vectors(vec1, vec2)

    def _cosine_similarity_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity [0, 1]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        # Clamp to [0, 1] range to handle floating point precision issues
        return max(0.0, min(similarity, 1.0))


__all__ = [
    "CosineSimilarityMetric",
    "CosineEmbeddingSimilarityMetric",
    "JaccardSimilarityMetric",
]


@register_metric("cosine_embedding")
class CosineEmbeddingSimilarityMetric(BaseMetric):
    """
    Embedding-based Cosine Similarity

    Note: Requires external embedding vectors or pretrained models.
    This implementation is a placeholder and requires an embedding function to be provided for actual use.

    Attributes:
        name: Metric name
        embedding_model: Embedding model name (placeholder)

    Example:
        >>> # This metric requires an external embedding model
        >>> # metric = CosineEmbeddingSimilarityMetric()
        >>> # Need to provide embedding_fn
    """

    name: str = "cosine_embedding"
    embedding_model: str = Field(
        default="placeholder", description="Embedding model name"
    )
    normalize_text: bool = Field(
        default=False, description="Typically no normalization before embedding"
    )

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute embedding-based cosine similarity

        Note: This is a placeholder implementation. Requires integration with an embedding model for actual use.

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        # Placeholder implementation - requires external embedding function
        return MetricResult(
            name=self.name,
            score=0.0,
            details={
                "error": "Embedding model not configured",
                "message": "This metric requires an embedding function to be provided",
            },
            metadata={"embedding_model": self.embedding_model},
        )


@register_metric("jaccard")
class JaccardSimilarityMetric(BaseMetric):
    """
    Jaccard Similarity Metric

    Calculates Jaccard similarity of word sets between two texts.

    Attributes:
        name: Metric name
        use_ngrams: Whether to use n-grams instead of words
        n: Size of n-grams (when use_ngrams=True)

    Example:
        >>> metric = JaccardSimilarityMetric()
        >>> input_data = ComparisonInput(
        ...     reference="the cat sat on the mat",
        ...     candidate="the dog sat on the mat"
        ... )
        >>> result = metric.compute(input_data)
        >>> print(f"Jaccard Similarity: {result.score:.4f}")
    """

    name: str = "jaccard"
    use_ngrams: bool = Field(default=False, description="Whether to use n-grams")
    n: int = Field(default=2, description="N-gram size")
    normalize_text: bool = Field(default=True, description="Whether to normalize text")

    def compute(self, input_data: ComparisonInput) -> MetricResult:
        """
        Compute Jaccard similarity

        Args:
            input_data: Comparison input data

        Returns:
            MetricResult: Evaluation result
        """
        reference = input_data.reference
        candidate = input_data.candidate

        # Normalize
        if input_data.normalize and self.normalize_text:
            if isinstance(reference, list):
                reference = [self._normalize(r) for r in reference]
            else:
                reference = self._normalize(reference)
            candidate = self._normalize(candidate)

        # Handle multiple reference texts
        if isinstance(reference, list):
            scores = [self._compute_jaccard(candidate, ref) for ref in reference]
            score = max(scores)
            details = {
                "scores_per_reference": scores,
                "max_score": score,
            }
        else:
            score = self._compute_jaccard(candidate, reference)
            details = {}

        details.update(
            {
                "use_ngrams": self.use_ngrams,
                "n": self.n if self.use_ngrams else None,
            }
        )

        return MetricResult(name=self.name, score=score, details=details)

    def _compute_jaccard(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Jaccard similarity
        """
        if self.use_ngrams:
            from rm_gallery.core.metrics.utils.tokenization import ngram_tokenize

            tokens1 = set(ngram_tokenize(text1, n=self.n, char_level=False))
            tokens2 = set(ngram_tokenize(text2, n=self.n, char_level=False))
        else:
            tokens1 = set(text1.split())
            tokens2 = set(text2.split())

        if len(tokens1) == 0 and len(tokens2) == 0:
            return 1.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)
