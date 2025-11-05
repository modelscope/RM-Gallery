"""
Unit Tests for F1 Score Metric

Test token-based F1 score calculation following OpenAI Evals implementation.
"""

from rm_gallery.core.metrics.schema import ComparisonInput
from rm_gallery.core.metrics.text_similarity.f1_score import F1ScoreMetric


class TestF1ScoreBasic:
    """Basic F1 score functionality tests"""

    def test_exact_match(self):
        """Test exact match returns perfect F1 score"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(reference="hello world", candidate="hello world")
        result = metric.compute(input_data)

        assert result.score == 1.0
        assert result.details["precision"] == 1.0
        assert result.details["recall"] == 1.0

    def test_no_overlap(self):
        """Test completely different strings return 0 F1 score"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(
            reference="hello world", candidate="goodbye universe"
        )
        result = metric.compute(input_data)

        assert result.score == 0.0
        assert result.details["precision"] == 0.0
        assert result.details["recall"] == 0.0

    def test_partial_overlap(self):
        """Test partial token overlap"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="cat on mat"
        )
        result = metric.compute(input_data)

        # After normalization:
        # Reference: "cat is on mat" (removed "the") = 4 tokens
        # Candidate: "cat on mat" = 3 tokens
        # Common: cat, on, mat = 3 tokens
        # precision = 3/3 = 1.0
        # recall = 3/4 = 0.75
        # f1 = 2 * 1.0 * 0.75 / (1.0 + 0.75) = 0.857
        assert abs(result.score - 0.857) < 0.01
        assert result.details["precision"] == 1.0
        assert abs(result.details["recall"] - 0.75) < 0.01

    def test_openai_evals_example_1(self):
        """Test OpenAI Evals example from fuzzy_match_test.py"""
        # From evals fuzzy_match_test.py line 18:
        # ("world", ["some foo world", "dummy"], dict(accuracy=1.0, f1_score=0.5))
        metric = F1ScoreMetric()
        input_data = ComparisonInput(
            reference=["some foo world", "dummy"], candidate="world"
        )
        result = metric.compute(input_data)

        # Should get f1_score of 0.5 from the first reference
        assert abs(result.score - 0.5) < 0.001

    def test_word_order_matters(self):
        """Test that word order doesn't affect F1 (token-based)"""
        metric = F1ScoreMetric()

        input_data1 = ComparisonInput(
            reference="the quick brown fox", candidate="fox brown quick the"
        )
        result1 = metric.compute(input_data1)

        input_data2 = ComparisonInput(
            reference="the quick brown fox", candidate="the quick brown fox"
        )
        result2 = metric.compute(input_data2)

        # Both should have perfect F1 (same tokens)
        assert result1.score == result2.score == 1.0


class TestF1ScoreMultipleReferences:
    """Test F1 score with multiple reference texts"""

    def test_multiple_references_max_score(self):
        """Test that maximum F1 score is returned"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(
            reference=["completely different", "world", "another text"],
            candidate="world",
        )
        result = metric.compute(input_data)

        # Should match the second reference perfectly
        assert result.score == 1.0
        assert result.details["best_reference_idx"] == 1

    def test_multiple_references_partial_match(self):
        """Test partial matching with multiple references"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(
            reference=["hello", "world", "foo bar"],
            candidate="hello world",
        )
        result = metric.compute(input_data)

        # Candidate: "hello world" (2 tokens)
        # Ref 1: "hello" - common: 1, precision=0.5, recall=1.0, f1=0.667
        # Ref 2: "world" - common: 1, precision=0.5, recall=1.0, f1=0.667
        # Ref 3: "foo bar" - common: 0, f1=0.0
        # Should get best match (0.667)
        assert abs(result.score - 0.667) < 0.01
        assert "f1_scores_per_reference" in result.details
        assert len(result.details["f1_scores_per_reference"]) == 3


class TestF1ScoreNormalization:
    """Test normalization effects on F1 score"""

    def test_case_normalization(self):
        """Test case insensitive matching with normalization"""
        metric = F1ScoreMetric(normalize_text=True)
        input_data = ComparisonInput(reference="Hello World", candidate="hello world")
        result = metric.compute(input_data)

        assert result.score == 1.0

    def test_punctuation_removal(self):
        """Test punctuation is removed with normalization"""
        metric = F1ScoreMetric(normalize_text=True)
        input_data = ComparisonInput(reference="hello, world!", candidate="hello world")
        result = metric.compute(input_data)

        assert result.score == 1.0

    def test_article_removal(self):
        """Test articles (a, an, the) are removed"""
        metric = F1ScoreMetric(normalize_text=True)
        input_data = ComparisonInput(
            reference="the quick brown fox", candidate="quick brown fox"
        )
        result = metric.compute(input_data)

        # After normalization, both become "quick brown fox"
        assert result.score == 1.0

    def test_no_normalization(self):
        """Test that disabling normalization preserves case and punctuation"""
        metric = F1ScoreMetric(normalize_text=False)
        input_data = ComparisonInput(
            reference="Hello, World!", candidate="hello world", normalize=False
        )
        result = metric.compute(input_data)

        # Without normalization, tokens don't match
        assert result.score == 0.0


class TestF1ScoreEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_strings(self):
        """Test handling of empty strings"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(reference="", candidate="")
        result = metric.compute(input_data)

        # Both empty - perfect match
        assert result.score == 1.0

    def test_empty_reference(self):
        """Test empty reference with non-empty candidate"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(reference="", candidate="hello")
        result = metric.compute(input_data)

        assert result.score == 0.0

    def test_empty_candidate(self):
        """Test non-empty reference with empty candidate"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(reference="hello", candidate="")
        result = metric.compute(input_data)

        assert result.score == 0.0

    def test_single_token(self):
        """Test single token matching"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(reference="hello", candidate="hello")
        result = metric.compute(input_data)

        assert result.score == 1.0

    def test_duplicate_tokens(self):
        """Test handling of duplicate tokens"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(
            reference="hello hello world", candidate="hello world world"
        )
        result = metric.compute(input_data)

        # Candidate: [hello(1), world(2)] vs Reference: [hello(2), world(1)]
        # Common: hello(1), world(1) = 2 tokens
        # precision = 2/3, recall = 2/3, f1 = 2/3
        assert abs(result.score - 0.6667) < 0.001


class TestF1ScorePrecisionRecall:
    """Test precision and recall calculations"""

    def test_high_precision_low_recall(self):
        """Test case with high precision but low recall"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(
            reference="the quick brown fox jumps over the lazy dog",
            candidate="quick brown fox",
        )
        result = metric.compute(input_data)

        # After normalization (removes "the"):
        # Reference: "quick brown fox jumps over lazy dog" = 7 tokens
        # Candidate: "quick brown fox" = 3 tokens
        # Common: quick, brown, fox = 3 tokens
        # All candidate tokens are in reference (precision = 1.0)
        # But only 3 out of 7 reference tokens are in candidate (recall = 3/7 ≈ 0.429)
        assert result.details["precision"] == 1.0
        assert abs(result.details["recall"] - 0.429) < 0.01

    def test_low_precision_high_recall(self):
        """Test case with low precision but high recall"""
        metric = F1ScoreMetric()
        input_data = ComparisonInput(
            reference="quick brown fox",
            candidate="the quick brown fox jumps over the lazy dog",
        )
        result = metric.compute(input_data)

        # After normalization (removes "the"):
        # Reference: "quick brown fox" = 3 tokens
        # Candidate: "quick brown fox jumps over lazy dog" = 7 tokens
        # Common: quick, brown, fox = 3 tokens
        # All reference tokens are in candidate (recall = 1.0)
        # But only 3 out of 7 candidate tokens are in reference (precision = 3/7 ≈ 0.429)
        assert abs(result.details["precision"] - 0.429) < 0.01
        assert result.details["recall"] == 1.0


class TestTokenF1Alias:
    """Test TokenF1Metric alias"""

    def test_alias_works(self):
        """Test that TokenF1Metric is an alias for F1ScoreMetric"""
        from rm_gallery.core.metrics.text_similarity.f1_score import TokenF1Metric

        metric = TokenF1Metric()
        input_data = ComparisonInput(reference="hello world", candidate="hello world")
        result = metric.compute(input_data)

        assert result.score == 1.0
        assert result.name == "token_f1"
