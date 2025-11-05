"""
Unit Tests for Fuzzy Match Metric

Test fuzzy matching functionality including exact match, partial match, and token sorting.
"""

from rm_gallery.core.metrics.schema import ComparisonInput
from rm_gallery.core.metrics.text_similarity.fuzzy import FuzzyMatchMetric


class TestFuzzyMatchBasic:
    """Basic fuzzy match functionality tests"""

    def test_exact_match(self):
        """Test exact match returns perfect score"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(reference="hello world", candidate="hello world")
        result = metric.compute(input_data)

        assert result.score == 1.0
        assert result.name == "fuzzy_match"
        assert "matched" in result.details
        assert result.details["matched"] is True

    def test_complete_mismatch(self):
        """Test completely different strings return low score"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(
            reference="hello world", candidate="goodbye universe"
        )
        result = metric.compute(input_data)

        assert 0.0 <= result.score < 0.5
        assert result.details["matched"] is False

    def test_partial_match(self):
        """Test partial matching"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(
            reference="hello world", candidate="hello worl"  # Missing 'd'
        )
        result = metric.compute(input_data)

        assert 0.9 < result.score < 1.0

    def test_case_insensitive(self):
        """Test case insensitive matching"""
        metric = FuzzyMatchMetric(normalize_text=True)
        input_data = ComparisonInput(reference="Hello World", candidate="hello world")
        result = metric.compute(input_data)

        assert result.score == 1.0


class TestFuzzyMatchMethods:
    """Test different fuzzy matching methods"""

    def test_ratio_method(self):
        """Test standard ratio method"""
        metric = FuzzyMatchMetric(method="ratio")
        input_data = ComparisonInput(
            reference="the quick brown fox", candidate="the quick brown fox"
        )
        result = metric.compute(input_data)

        assert result.score == 1.0
        assert result.details["method"] == "ratio"

    def test_partial_ratio_method(self):
        """Test partial ratio for substring matching"""
        metric = FuzzyMatchMetric(method="partial_ratio")
        input_data = ComparisonInput(
            reference="the quick brown fox jumps", candidate="quick brown fox"
        )
        result = metric.compute(input_data)

        # Partial ratio should give high score for substring match
        assert result.score > 0.8

    def test_token_sort_ratio_method(self):
        """Test token sort ratio for order-independent matching"""
        metric = FuzzyMatchMetric(method="token_sort_ratio")
        input_data = ComparisonInput(
            reference="brown quick the fox", candidate="the quick brown fox"
        )
        result = metric.compute(input_data)

        # Token sort should give perfect score for same words different order
        assert result.score == 1.0


class TestFuzzyMatchMultipleReferences:
    """Test fuzzy matching with multiple reference texts"""

    def test_multiple_references_exact_match(self):
        """Test exact match with multiple references"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(
            reference=["hello world", "hi there", "greetings"],
            candidate="hello world",
        )
        result = metric.compute(input_data)

        assert result.score == 1.0
        assert "scores_per_reference" in result.details
        assert len(result.details["scores_per_reference"]) == 3

    def test_multiple_references_best_match(self):
        """Test that best matching reference is selected"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(
            reference=["completely different text", "hello world", "another text"],
            candidate="hello world",
        )
        result = metric.compute(input_data)

        # Should match the second reference perfectly
        assert result.score == 1.0
        scores = result.details["scores_per_reference"]
        assert max(scores) == 1.0

    def test_multiple_references_partial_match(self):
        """Test partial matching with multiple references"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(
            reference=["hello", "world", "foo bar"],
            candidate="hello world",
        )
        result = metric.compute(input_data)

        # Should have some match
        assert result.score > 0.0


class TestFuzzyMatchEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_strings(self):
        """Test handling of empty strings"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(reference="", candidate="")
        result = metric.compute(input_data)

        # Empty strings should match perfectly
        assert result.score == 1.0

    def test_empty_reference_non_empty_candidate(self):
        """Test empty reference with non-empty candidate"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(reference="", candidate="hello")
        result = metric.compute(input_data)

        assert result.score == 0.0

    def test_very_long_strings(self):
        """Test performance with long strings"""
        metric = FuzzyMatchMetric()
        long_text = "word " * 1000  # 1000 words
        input_data = ComparisonInput(reference=long_text, candidate=long_text)
        result = metric.compute(input_data)

        assert result.score == 1.0

    def test_special_characters(self):
        """Test handling of special characters"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(
            reference="hello@world#2024!", candidate="hello@world#2024!"
        )
        result = metric.compute(input_data)

        assert result.score == 1.0

    def test_unicode_characters(self):
        """Test handling of Unicode characters"""
        metric = FuzzyMatchMetric()
        input_data = ComparisonInput(reference="你好世界", candidate="你好世界")
        result = metric.compute(input_data)

        assert result.score == 1.0


class TestFuzzyMatchThreshold:
    """Test threshold functionality"""

    def test_threshold_matching(self):
        """Test threshold-based matching decision"""
        metric = FuzzyMatchMetric(threshold=0.9)

        # High similarity - should match
        input_data = ComparisonInput(
            reference="the quick brown fox", candidate="the quick brown fo"
        )
        result = metric.compute(input_data)

        if result.score >= 0.9:
            assert result.details["matched"] is True
        else:
            assert result.details["matched"] is False

    def test_different_thresholds(self):
        """Test different threshold values"""
        input_data = ComparisonInput(reference="hello world", candidate="hello worl")

        # Strict threshold
        metric_strict = FuzzyMatchMetric(threshold=0.99)
        result_strict = metric_strict.compute(input_data)

        # Lenient threshold
        metric_lenient = FuzzyMatchMetric(threshold=0.80)
        result_lenient = metric_lenient.compute(input_data)

        # Scores should be the same, but matching decision may differ
        assert result_strict.score == result_lenient.score
