"""
Unit Tests for BLEU Metric

Test BLEU score calculation for machine translation evaluation.
"""

import pytest

from rm_gallery.core.metrics.nlp_metrics.bleu import BLEUMetric, SentenceBLEUMetric
from rm_gallery.core.metrics.schema import ComparisonInput


class TestBLEUBasic:
    """Basic BLEU functionality tests"""

    def test_perfect_match(self):
        """Test perfect match returns score of 1.0"""
        metric = BLEUMetric()
        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="the cat is on the mat"
        )
        result = metric.compute(input_data)

        assert result.score == 1.0
        assert result.name == "bleu"
        assert "precisions" in result.details

    def test_complete_mismatch(self):
        """Test completely different sentences"""
        metric = BLEUMetric()
        input_data = ComparisonInput(
            reference="the cat is on the mat",
            candidate="hello world foo bar baz qux",
        )
        result = metric.compute(input_data)

        assert result.score < 0.1  # Very low score for completely different text

    def test_partial_match(self):
        """Test partial matching"""
        metric = BLEUMetric()
        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="the dog is on the mat"
        )
        result = metric.compute(input_data)

        # Should have some overlap but not perfect
        assert 0.3 < result.score < 0.9

    def test_word_order_matters(self):
        """Test that word order affects BLEU score"""
        metric = BLEUMetric()
        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="the mat on is cat the"
        )
        result = metric.compute(input_data)

        # Same words but different order should give lower score
        assert result.score < 1.0


class TestBLEUParameters:
    """Test BLEU with different parameters"""

    def test_different_max_ngram_orders(self):
        """Test different n-gram orders"""
        input_data = ComparisonInput(
            reference="the quick brown fox jumps over the lazy dog",
            candidate="the quick brown fox jumps over the lazy dog",
        )

        # Test with different max n-gram orders
        for n in [1, 2, 3, 4]:
            metric = BLEUMetric(max_ngram_order=n)
            result = metric.compute(input_data)
            assert result.score == 1.0
            assert len(result.details["precisions"]) == n

    def test_smoothing_methods(self):
        """Test different smoothing methods"""
        input_data = ComparisonInput(
            reference="the cat sat on the mat", candidate="the cat"
        )

        # Test different smoothing methods
        for method in ["none", "floor", "add-k", "exp"]:
            metric = BLEUMetric(smooth_method=method)
            result = metric.compute(input_data)
            assert 0.0 <= result.score <= 1.0


class TestBLEUMultipleReferences:
    """Test BLEU with multiple reference translations"""

    def test_multiple_references_best_match(self):
        """Test that best matching reference is used"""
        metric = BLEUMetric()
        input_data = ComparisonInput(
            reference=[
                "the cat is on the mat",
                "a cat sits on the mat",
                "there is a cat on the mat",
            ],
            candidate="the cat is on the mat",
        )
        result = metric.compute(input_data)

        # Should match first reference perfectly
        assert result.score == 1.0

    def test_multiple_references_partial_match(self):
        """Test partial matching with multiple references"""
        metric = BLEUMetric()
        input_data = ComparisonInput(
            reference=["the cat is here", "the dog is here"],
            candidate="the bird is here",
        )
        result = metric.compute(input_data)

        # Should have some match due to common words
        assert result.score > 0.0


class TestBLEUEdgeCases:
    """Test edge cases"""

    def test_empty_candidate(self):
        """Test handling of empty candidate"""
        metric = BLEUMetric()
        input_data = ComparisonInput(reference="the cat is on the mat", candidate="")
        result = metric.compute(input_data)

        # Empty candidate should give zero score
        assert result.score == 0.0

    def test_empty_reference(self):
        """Test handling of empty reference"""
        metric = BLEUMetric()
        input_data = ComparisonInput(reference="", candidate="the cat")
        result = metric.compute(input_data)

        # Empty reference should give zero score
        assert result.score == 0.0

    def test_single_word_sentences(self):
        """Test single word sentences"""
        metric = BLEUMetric()
        input_data = ComparisonInput(reference="cat", candidate="cat")
        result = metric.compute(input_data)

        assert result.score == 1.0

    def test_very_long_sentences(self):
        """Test with very long sentences"""
        metric = BLEUMetric()
        long_sentence = " ".join(["word"] * 500)
        input_data = ComparisonInput(reference=long_sentence, candidate=long_sentence)
        result = metric.compute(input_data)

        assert result.score == 1.0


class TestBLEUDetails:
    """Test BLEU result details"""

    def test_precision_details(self):
        """Test that precision details are included"""
        metric = BLEUMetric(max_ngram_order=4)
        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="the cat is on the mat"
        )
        result = metric.compute(input_data)

        assert "precisions" in result.details
        assert len(result.details["precisions"]) == 4
        # Perfect match should have all precisions = 1.0
        for prec in result.details["precisions"]:
            assert prec == pytest.approx(1.0, abs=0.01)

    def test_brevity_penalty(self):
        """Test brevity penalty calculation"""
        metric = BLEUMetric()

        # Short candidate
        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="the cat"
        )
        result = metric.compute(input_data)

        assert "bp" in result.details
        # Brevity penalty should be < 1.0 for shorter candidate
        assert result.details["bp"] < 1.0

    def test_length_information(self):
        """Test that length information is included"""
        metric = BLEUMetric()
        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="the dog is on the rug"
        )
        result = metric.compute(input_data)

        assert "sys_len" in result.details  # System (candidate) length
        assert "ref_len" in result.details  # Reference length
        assert result.details["sys_len"] == result.details["ref_len"]


class TestSentenceBLEU:
    """Test sentence-level BLEU variant"""

    def test_sentence_bleu_basic(self):
        """Test basic sentence BLEU"""
        metric = SentenceBLEUMetric()
        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="the cat is on the mat"
        )
        result = metric.compute(input_data)

        assert result.score == 1.0
        assert result.name == "sentence_bleu"

    def test_sentence_bleu_vs_corpus_bleu(self):
        """Compare sentence-level and corpus-level BLEU"""
        sentence_metric = SentenceBLEUMetric()
        corpus_metric = BLEUMetric()

        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="the dog is on the mat"
        )

        sentence_result = sentence_metric.compute(input_data)
        corpus_result = corpus_metric.compute(input_data)

        # Scores may differ slightly due to different calculation methods
        assert 0.0 <= sentence_result.score <= 1.0
        assert 0.0 <= corpus_result.score <= 1.0
