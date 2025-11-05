"""
String Check Metrics

String checking metrics module containing exact match, substring match,
and other string operation-based metrics.
"""

from rm_gallery.core.metrics.string_check.exact_match import (
    ExactMatchMetric,
    PrefixMatchMetric,
    RegexMatchMetric,
    SuffixMatchMetric,
)
from rm_gallery.core.metrics.string_check.substring import (
    CharacterOverlapMetric,
    ContainsAllMetric,
    ContainsAnyMetric,
    SubstringMatchMetric,
    WordOverlapMetric,
)

__all__ = [
    # Exact matching
    "ExactMatchMetric",
    "PrefixMatchMetric",
    "SuffixMatchMetric",
    "RegexMatchMetric",
    # Substring matching
    "SubstringMatchMetric",
    "ContainsAllMetric",
    "ContainsAnyMetric",
    # Overlap metrics
    "WordOverlapMetric",
    "CharacterOverlapMetric",
]
