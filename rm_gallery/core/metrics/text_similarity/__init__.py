"""
Text Similarity Metrics

Text similarity metrics module containing various string-based and vector-based similarity calculation methods.
"""

from rm_gallery.core.metrics.text_similarity.cosine import (
    CosineEmbeddingSimilarityMetric,
    CosineSimilarityMetric,
    JaccardSimilarityMetric,
)
from rm_gallery.core.metrics.text_similarity.f1_score import (
    F1ScoreMetric,
    TokenF1Metric,
)
from rm_gallery.core.metrics.text_similarity.fuzzy import (
    EditDistanceMetric,
    FuzzyMatchMetric,
    SimpleFuzzyMatchMetric,
)

__all__ = [
    # Fuzzy matching
    "FuzzyMatchMetric",
    "SimpleFuzzyMatchMetric",
    "EditDistanceMetric",
    # F1 Score
    "F1ScoreMetric",
    "TokenF1Metric",
    # Cosine similarity
    "CosineSimilarityMetric",
    "CosineEmbeddingSimilarityMetric",
    # Jaccard similarity
    "JaccardSimilarityMetric",
]
