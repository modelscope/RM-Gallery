"""
NLP Metrics

Natural Language Processing evaluation metrics module, containing standard NLP evaluation metrics.
"""

from rm_gallery.core.metrics.nlp_metrics.bleu import (
    BLEUMetric,
    SelfBLEUMetric,
    SentenceBLEUMetric,
)
from rm_gallery.core.metrics.nlp_metrics.gleu import (
    ChrFMetric,
    CorpusGLEUMetric,
    GLEUMetric,
)
from rm_gallery.core.metrics.nlp_metrics.meteor import METEORMetric, SimpleMETEORMetric
from rm_gallery.core.metrics.nlp_metrics.rouge import (
    MultiROUGEMetric,
    ROUGE1Metric,
    ROUGE2Metric,
    ROUGELMetric,
    ROUGEMetric,
)
from rm_gallery.core.metrics.nlp_metrics.rouge_ngram import (
    CustomROUGENMetric,
    ROUGE3Metric,
    ROUGE4Metric,
    ROUGE5Metric,
    ROUGENGramMetric,
)

__all__ = [
    # BLEU variants
    "BLEUMetric",
    "SentenceBLEUMetric",
    "SelfBLEUMetric",
    # ROUGE variants
    "ROUGEMetric",
    "ROUGE1Metric",
    "ROUGE2Metric",
    "ROUGELMetric",
    "MultiROUGEMetric",
    # ROUGE N-gram variants (3, 4, 5)
    "ROUGENGramMetric",
    "ROUGE3Metric",
    "ROUGE4Metric",
    "ROUGE5Metric",
    "CustomROUGENMetric",
    # METEOR
    "METEORMetric",
    "SimpleMETEORMetric",
    # GLEU
    "GLEUMetric",
    "CorpusGLEUMetric",
    # ChrF
    "ChrFMetric",
]
