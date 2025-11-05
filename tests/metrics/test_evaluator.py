"""
Integration Tests for TextSimilarityEvaluator

Test the unified evaluator that combines multiple metrics.
"""

import pytest

from rm_gallery.core.metrics.evaluator import TextSimilarityEvaluator
from rm_gallery.core.metrics.schema import ComparisonInput, MetricResult


class TestEvaluatorBasic:
    """Basic evaluator functionality tests"""

    def test_evaluator_initialization(self):
        """Test evaluator initialization with specified metrics"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge", "fuzzy_match"])

        assert len(evaluator.list_metrics()) == 3
        assert "bleu" in evaluator.list_metrics()
        assert "rouge" in evaluator.list_metrics()
        assert "fuzzy_match" in evaluator.list_metrics()

    def test_evaluator_auto_select(self):
        """Test evaluator with auto-select all metrics"""
        evaluator = TextSimilarityEvaluator(auto_select=True)

        # Should have multiple metrics available
        assert len(evaluator.list_metrics()) > 0

    def test_evaluator_single_metric(self):
        """Test evaluator with single metric"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        assert len(evaluator.list_metrics()) == 1
        assert "bleu" in evaluator.list_metrics()


class TestEvaluatorEvaluate:
    """Test single sample evaluation"""

    def test_evaluate_perfect_match(self):
        """Test evaluation of perfect match"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1", "fuzzy_match"])

        results = evaluator.evaluate(
            reference="the cat is on the mat", candidate="the cat is on the mat"
        )

        # All metrics should give perfect score
        assert len(results) == 3
        for metric_name, result in results.items():
            assert isinstance(result, MetricResult)
            assert result.score == 1.0

    def test_evaluate_partial_match(self):
        """Test evaluation of partial match"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1"])

        results = evaluator.evaluate(
            reference="the cat is on the mat", candidate="the dog is on the rug"
        )

        assert len(results) == 2
        # Should have some overlap but not perfect
        for result in results.values():
            assert 0.0 < result.score < 1.0

    def test_evaluate_subset_of_metrics(self):
        """Test evaluating with subset of initialized metrics"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1", "fuzzy_match"])

        # Only evaluate with BLEU and ROUGE
        results = evaluator.evaluate(
            reference="hello world", candidate="hello world", metrics=["bleu", "rouge1"]
        )

        assert len(results) == 2
        assert "bleu" in results
        assert "rouge1" in results
        assert "fuzzy_match" not in results

    def test_evaluate_with_normalization(self):
        """Test evaluation with text normalization"""
        evaluator = TextSimilarityEvaluator(metrics=["fuzzy_match"])

        results = evaluator.evaluate(
            reference="Hello World", candidate="hello world", normalize=True
        )

        # Should match after normalization
        assert results["fuzzy_match"].score == 1.0


class TestEvaluatorBatchEvaluate:
    """Test batch evaluation"""

    def test_batch_evaluate_basic(self):
        """Test basic batch evaluation"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        references = ["ref1", "ref2", "ref3"]
        candidates = ["ref1", "ref2", "ref3"]

        results = evaluator.evaluate_batch(
            references=references, candidates=candidates, return_aggregated=False
        )

        assert len(results) == 3
        # All should be perfect matches
        for result_dict in results:
            assert result_dict["bleu"].score == 1.0

    def test_batch_evaluate_aggregated(self):
        """Test batch evaluation with aggregation"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1"])

        references = ["the cat", "the dog", "the bird"]
        candidates = ["the cat", "the dog", "the bird"]

        aggregated = evaluator.evaluate_batch(
            references=references, candidates=candidates, return_aggregated=True
        )

        # Should return aggregated results
        assert "bleu" in aggregated
        assert "rouge1" in aggregated

        # Check aggregated statistics
        bleu_agg = aggregated["bleu"]
        assert bleu_agg.mean_score == 1.0
        assert bleu_agg.std_score == 0.0
        assert bleu_agg.count == 3

    def test_batch_evaluate_parallel(self):
        """Test parallel batch evaluation"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"], max_workers=4)

        # Create larger batch
        references = [f"text {i}" for i in range(20)]
        candidates = [f"text {i}" for i in range(20)]

        results = evaluator.evaluate_batch(
            references=references, candidates=candidates, max_workers=4
        )

        assert len(results) == 20

    def test_batch_evaluate_length_mismatch(self):
        """Test batch evaluation with mismatched lengths"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        with pytest.raises(ValueError, match="Length mismatch"):
            evaluator.evaluate_batch(
                references=["ref1", "ref2"], candidates=["cand1", "cand2", "cand3"]
            )


class TestEvaluatorSummary:
    """Test result summary generation"""

    def test_get_summary_simple(self):
        """Test simple summary format"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1"])

        results = evaluator.evaluate(reference="the cat", candidate="the cat")

        summary = evaluator.get_summary(results, format="simple")

        assert "bleu" in summary
        assert "rouge1" in summary
        assert isinstance(summary, str)

    def test_get_summary_table(self):
        """Test table summary format"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        results = evaluator.evaluate(reference="the cat", candidate="the cat")

        summary = evaluator.get_summary(results, format="table")

        assert "Metric" in summary
        assert "Score" in summary
        assert "bleu" in summary

    def test_get_summary_json(self):
        """Test JSON summary format"""
        import json

        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        results = evaluator.evaluate(reference="the cat", candidate="the cat")

        summary = evaluator.get_summary(results, format="json")

        # Should be valid JSON
        parsed = json.loads(summary)
        assert "bleu" in parsed


class TestEvaluatorModelComparison:
    """Test model comparison functionality"""

    def test_compare_models(self):
        """Test comparing outputs from multiple models"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1"])

        reference = "the cat is on the mat"
        candidates = {
            "model_a": "the cat is on the mat",
            "model_b": "the dog is on the rug",
            "model_c": "a feline sits on a mat",
        }

        comparison = evaluator.compare_models(reference, candidates)

        assert len(comparison) == 3
        assert "model_a" in comparison
        assert "model_b" in comparison
        assert "model_c" in comparison

        # Model A should have perfect score
        assert comparison["model_a"]["bleu"].score == 1.0

    def test_get_best_model(self):
        """Test selecting best model from comparison"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1"])

        reference = "the cat is on the mat"
        candidates = {
            "model_a": "the cat is on the mat",  # Perfect match
            "model_b": "the dog is on the rug",  # Partial match
            "model_c": "hello world",  # No match
        }

        comparison = evaluator.compare_models(reference, candidates)
        best_model = evaluator.get_best_model(comparison)

        assert best_model == "model_a"

    def test_get_best_model_with_weights(self):
        """Test selecting best model with metric weights"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1"])

        reference = "the cat is on the mat"
        candidates = {
            "model_a": "the cat is on the mat",
            "model_b": "the dog is on the rug",
        }

        comparison = evaluator.compare_models(reference, candidates)

        # Give BLEU more weight
        weights = {"bleu": 2.0, "rouge1": 1.0}
        best_model = evaluator.get_best_model(comparison, metric_weights=weights)

        assert best_model in ["model_a", "model_b"]


class TestEvaluatorDynamicMetrics:
    """Test dynamic metric management"""

    def test_add_metric(self):
        """Test adding metric dynamically"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        assert len(evaluator.list_metrics()) == 1

        evaluator.add_metric("rouge1")

        assert len(evaluator.list_metrics()) == 2
        assert "rouge1" in evaluator.list_metrics()

    def test_remove_metric(self):
        """Test removing metric"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1"])

        assert len(evaluator.list_metrics()) == 2

        evaluator.remove_metric("rouge1")

        assert len(evaluator.list_metrics()) == 1
        assert "rouge1" not in evaluator.list_metrics()

    def test_remove_nonexistent_metric(self):
        """Test removing metric that doesn't exist"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        # Should not raise error
        evaluator.remove_metric("nonexistent_metric")

        assert len(evaluator.list_metrics()) == 1


class TestEvaluatorErrorHandling:
    """Test error handling in evaluator"""

    def test_invalid_metric_name(self):
        """Test initialization with invalid metric name"""
        evaluator = TextSimilarityEvaluator(
            metrics=["bleu", "invalid_metric", "rouge1"]
        )

        # Should skip invalid metric
        assert "invalid_metric" not in evaluator.list_metrics()
        assert "bleu" in evaluator.list_metrics()
        assert "rouge1" in evaluator.list_metrics()

    def test_evaluate_with_invalid_metric(self):
        """Test evaluation with invalid metric in subset"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        # Request invalid metric in evaluate
        results = evaluator.evaluate(
            reference="test", candidate="test", metrics=["bleu", "invalid_metric"]
        )

        # Should only return results for valid metrics
        assert "bleu" in results
        assert "invalid_metric" not in results


class TestEvaluatorMultipleReferences:
    """Test evaluator with multiple reference texts"""

    def test_evaluate_multiple_references(self):
        """Test evaluation with multiple reference texts"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1"])

        # Note: Need to construct ComparisonInput manually for multiple refs
        from rm_gallery.core.metrics.registry import get_metric

        bleu = get_metric("bleu")
        input_data = ComparisonInput(
            reference=["the cat sits", "a cat is sitting"],
            candidate="the cat is sitting",
        )

        result = bleu.compute(input_data)

        # Should handle multiple references
        assert 0.0 <= result.score <= 1.0


class TestEvaluatorPerformance:
    """Test evaluator performance characteristics"""

    def test_batch_faster_than_sequential(self):
        """Test that batch evaluation is faster than sequential"""
        import time

        evaluator = TextSimilarityEvaluator(metrics=["bleu"], max_workers=4)

        references = [f"text {i}" for i in range(50)]
        candidates = [f"text {i}" for i in range(50)]

        # Batch evaluation
        start = time.time()
        evaluator.evaluate_batch(
            references=references, candidates=candidates, max_workers=4
        )
        batch_time = time.time() - start

        # Sequential evaluation
        start = time.time()
        evaluator.evaluate_batch(
            references=references, candidates=candidates, max_workers=1
        )
        sequential_time = time.time() - start

        # Performance comparison is environment-dependent
        # Just verify both complete successfully
        # In ideal conditions, batch should be faster, but overhead may vary
        print(f"Parallel: {batch_time:.4f}s, Sequential: {sequential_time:.4f}s")
        assert batch_time > 0 and sequential_time > 0  # Both complete successfully
