"""
RM-Gallery Metrics Module

Text similarity and string checking evaluation metrics module.

This module provides a comprehensive collection of text evaluation metrics, including:
- Text similarity metrics (Fuzzy Match, Cosine Similarity, Jaccard, etc.)
- NLP evaluation metrics (BLEU, ROUGE, METEOR, GLEU, etc.)
- String check metrics (Exact Match, Substring, Overlap, etc.)

Quick Start:
    >>> from rm_gallery.core.metrics import get_metric, list_available_metrics
    >>>
    >>> # List all available metrics
    >>> metrics = list_available_metrics()
    >>> print(f"Available: {', '.join(metrics)}")
    >>>
    >>> # Use a metric
    >>> bleu = get_metric("bleu")
    >>> from rm_gallery.core.metrics.schema import ComparisonInput
    >>> input_data = ComparisonInput(
    ...     reference="the cat is on the mat",
    ...     candidate="the cat is on the mat"
    ... )
    >>> result = bleu.compute(input_data)
    >>> print(f"BLEU: {result.score:.4f}")
"""

# Base classes and schemas
from rm_gallery.core.metrics.base import BaseMetric, BaseNLPMetric, BaseStringMetric

# Evaluator
from rm_gallery.core.metrics.evaluator import TextSimilarityEvaluator

# Format Check
from rm_gallery.core.metrics.format_check import (
    JsonMatchMetric,
    JsonSchemaValidatorMetric,
    JsonValidatorMetric,
)

# NLP Metrics
from rm_gallery.core.metrics.nlp_metrics import (
    BLEUMetric,
    ChrFMetric,
    GLEUMetric,
    METEORMetric,
    ROUGE1Metric,
    ROUGE2Metric,
    ROUGE3Metric,
    ROUGE4Metric,
    ROUGE5Metric,
    ROUGELMetric,
    ROUGEMetric,
    SentenceBLEUMetric,
)
from rm_gallery.core.metrics.registry import (
    get_metric,
    list_available_metrics,
    metric_registry,
    register_metric,
)
from rm_gallery.core.metrics.schema import (
    AggregatedMetricResult,
    BatchComparisonInput,
    ComparisonInput,
    MetricConfig,
    MetricResult,
)

# String Check
from rm_gallery.core.metrics.string_check import (
    CharacterOverlapMetric,
    ContainsAllMetric,
    ContainsAnyMetric,
    ExactMatchMetric,
    PrefixMatchMetric,
    RegexMatchMetric,
    SubstringMatchMetric,
    SuffixMatchMetric,
    WordOverlapMetric,
)

# Import all metrics to trigger registration
# Text Similarity
from rm_gallery.core.metrics.text_similarity import (
    CosineSimilarityMetric,
    EditDistanceMetric,
    F1ScoreMetric,
    FuzzyMatchMetric,
    JaccardSimilarityMetric,
    SimpleFuzzyMatchMetric,
    TokenF1Metric,
)

__version__ = "0.1.0"

__all__ = [
    # Base classes
    "BaseMetric",
    "BaseNLPMetric",
    "BaseStringMetric",
    # Schema
    "MetricResult",
    "ComparisonInput",
    "BatchComparisonInput",
    "MetricConfig",
    "AggregatedMetricResult",
    # Registry
    "metric_registry",
    "register_metric",
    "get_metric",
    "list_available_metrics",
    # Evaluator
    "TextSimilarityEvaluator",
    # Text Similarity Metrics
    "FuzzyMatchMetric",
    "SimpleFuzzyMatchMetric",
    "EditDistanceMetric",
    "F1ScoreMetric",
    "TokenF1Metric",
    "CosineSimilarityMetric",
    "JaccardSimilarityMetric",
    # NLP Metrics
    "BLEUMetric",
    "SentenceBLEUMetric",
    "ROUGEMetric",
    "ROUGE1Metric",
    "ROUGE2Metric",
    "ROUGE3Metric",
    "ROUGE4Metric",
    "ROUGE5Metric",
    "ROUGELMetric",
    "METEORMetric",
    "GLEUMetric",
    "ChrFMetric",
    # String Check Metrics
    "ExactMatchMetric",
    "PrefixMatchMetric",
    "SuffixMatchMetric",
    "RegexMatchMetric",
    "SubstringMatchMetric",
    "ContainsAllMetric",
    "ContainsAnyMetric",
    "WordOverlapMetric",
    "CharacterOverlapMetric",
    # Format Check Metrics
    "JsonMatchMetric",
    "JsonValidatorMetric",
    "JsonSchemaValidatorMetric",
]
