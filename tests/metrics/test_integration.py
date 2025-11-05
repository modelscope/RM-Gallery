"""
Integration Tests for Metrics Module

Test complete workflows and integration between different components.
"""

from rm_gallery.core.metrics import (
    ComparisonInput,
    MetricResult,
    TextSimilarityEvaluator,
)
from rm_gallery.core.metrics.registry import (
    get_metric,
    list_available_metrics,
    metric_registry,
)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""

    def test_simple_evaluation_workflow(self):
        """Test simple evaluation workflow"""
        # Step 1: Get a metric
        bleu = get_metric("bleu")
        assert bleu is not None

        # Step 2: Create input
        input_data = ComparisonInput(
            reference="the cat is on the mat", candidate="the cat is on the mat"
        )

        # Step 3: Compute score
        result = bleu.compute(input_data)

        # Step 4: Verify result
        assert isinstance(result, MetricResult)
        assert result.score == 1.0

    def test_multi_metric_workflow(self):
        """Test workflow with multiple metrics"""
        # Initialize evaluator
        evaluator = TextSimilarityEvaluator(
            metrics=["bleu", "rouge1", "rouge2", "fuzzy_match"]
        )

        # Evaluate
        results = evaluator.evaluate(
            reference="the quick brown fox jumps over the lazy dog",
            candidate="the quick brown fox jumps over the lazy dog",
        )

        # Verify all metrics computed
        assert len(results) == 4
        for metric_name, result in results.items():
            assert result.score == 1.0

    def test_batch_evaluation_workflow(self):
        """Test batch evaluation workflow"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1"])

        # Prepare batch data
        references = [
            "the cat is on the mat",
            "the dog runs in the park",
            "birds fly in the sky",
        ]
        candidates = [
            "the cat is on the mat",
            "the dog runs in the park",
            "birds fly in the sky",
        ]

        # Batch evaluate
        results = evaluator.evaluate_batch(references=references, candidates=candidates)

        # Verify results
        assert len(results) == 3
        for result_dict in results:
            for result in result_dict.values():
                assert result.score == 1.0

    def test_aggregated_batch_workflow(self):
        """Test batch evaluation with aggregation"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        references = ["text"] * 10
        candidates = ["text"] * 10

        aggregated = evaluator.evaluate_batch(
            references=references, candidates=candidates, return_aggregated=True
        )

        assert "bleu" in aggregated
        assert aggregated["bleu"].mean_score == 1.0
        assert aggregated["bleu"].count == 10


class TestMetricDiscovery:
    """Test metric discovery and registry"""

    def test_list_all_metrics(self):
        """Test listing all available metrics"""
        metrics = list_available_metrics()

        # Should have all implemented metrics
        assert len(metrics) > 0
        assert "bleu" in metrics
        assert "rouge" in metrics or "rouge1" in metrics

    def test_get_metric_by_name(self):
        """Test getting metric by name"""
        bleu = get_metric("bleu")
        assert bleu is not None
        assert bleu.name == "bleu"

    def test_get_nonexistent_metric(self):
        """Test getting metric that doesn't exist"""
        metric = get_metric("nonexistent_metric")
        assert metric is None

    def test_metric_categories(self):
        """Test listing metrics by category"""
        categories = metric_registry.list_metrics_by_category()

        # Should have categories
        assert isinstance(categories, dict)
        assert len(categories) > 0


class TestCrossMetricConsistency:
    """Test consistency across different metrics"""

    def test_perfect_match_all_metrics(self):
        """Test that perfect match gives score 1.0 for all metrics"""
        evaluator = TextSimilarityEvaluator(auto_select=True)

        results = evaluator.evaluate(
            reference="the cat is on the mat", candidate="the cat is on the mat"
        )

        # Metrics that are placeholder implementations or have special behavior
        skip_metrics = {
            "cosine_embedding",  # Placeholder implementation without embedding model
            "self_bleu",  # Self-BLEU measures diversity across multiple candidates, returns 0 for single text
            # JSON/structured format validators - expect specific formats
            "json_match",
            "json_validator",
            "json_schema_validator",
        }

        # All metrics should give perfect or near-perfect score (except known placeholders)
        for metric_name, result in results.items():
            if metric_name in skip_metrics:
                continue
            # Use approximate equality for floating point comparisons
            assert (
                result.score >= 0.99
            ), f"{metric_name} didn't give perfect score (got {result.score})"

    def test_complete_mismatch_all_metrics(self):
        """Test that complete mismatch gives low scores"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1", "fuzzy_match"])

        results = evaluator.evaluate(
            reference="the cat is on the mat", candidate="xyz abc def ghi jkl"
        )

        # All metrics should give very low scores
        # Note: fuzzy_match may give slightly higher scores (~0.25) for similar-length strings
        for metric_name, result in results.items():
            threshold = 0.3 if metric_name == "fuzzy_match" else 0.2
            assert (
                result.score < threshold
            ), f"{metric_name} gave unexpectedly high score: {result.score}"


class TestNormalizationConsistency:
    """Test text normalization consistency"""

    def test_normalization_across_metrics(self):
        """Test that normalization works consistently across metrics"""
        # Note: BLEU does not support normalization by design (case matters in MT evaluation)
        # Only test metrics that support normalization
        evaluator = TextSimilarityEvaluator(metrics=["rouge1", "fuzzy_match"])

        # Test with different cases
        results = evaluator.evaluate(
            reference="The Cat Is On The Mat",
            candidate="the cat is on the mat",
            normalize=True,
        )

        # All tested metrics should handle normalization
        for result in results.values():
            assert result.score > 0.9

    def test_case_sensitivity(self):
        """Test case sensitivity handling"""
        evaluator = TextSimilarityEvaluator(metrics=["fuzzy_match"])

        # Without normalization
        results_sensitive = evaluator.evaluate(
            reference="HELLO WORLD", candidate="hello world", normalize=False
        )

        # With normalization
        results_normalized = evaluator.evaluate(
            reference="HELLO WORLD", candidate="hello world", normalize=True
        )

        # Normalized should give higher score
        assert (
            results_normalized["fuzzy_match"].score
            >= results_sensitive["fuzzy_match"].score
        )


class TestLanguageSupport:
    """Test multi-language support"""

    def test_english_text(self):
        """Test English text evaluation"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        results = evaluator.evaluate(
            reference="the cat is on the mat",
            candidate="the cat is on the mat",
            language="en",
        )

        assert results["bleu"].score == 1.0

    def test_chinese_text(self):
        """Test Chinese text evaluation"""
        evaluator = TextSimilarityEvaluator(metrics=["fuzzy_match"])

        results = evaluator.evaluate(
            reference="猫在垫子上", candidate="猫在垫子上", language="zh"
        )

        assert results["fuzzy_match"].score == 1.0

    def test_mixed_language(self):
        """Test mixed language text"""
        evaluator = TextSimilarityEvaluator(metrics=["fuzzy_match"])

        results = evaluator.evaluate(reference="Hello 世界", candidate="Hello 世界")

        assert results["fuzzy_match"].score == 1.0


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    def test_machine_translation_evaluation(self):
        """Test evaluating machine translation"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1", "rougeL"])

        reference = "The cat sat on the mat"
        candidate = "A cat was sitting on the mat"

        results = evaluator.evaluate(reference, candidate)

        # Should have reasonable scores
        # Note: BLEU can be lower (~0.26) due to word changes (sat->sitting)
        for result in results.values():
            assert 0.25 < result.score < 1.0

    def test_summarization_evaluation(self):
        """Test evaluating text summarization"""
        evaluator = TextSimilarityEvaluator(metrics=["rouge1", "rouge2", "rougeL"])

        reference = "The quick brown fox jumps over the lazy dog"
        candidate = "A fox jumps over a dog"

        results = evaluator.evaluate(reference, candidate)

        # Should detect some overlap
        for result in results.values():
            assert result.score > 0.0

    def test_paraphrase_detection(self):
        """Test detecting paraphrases"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1", "fuzzy_match"])

        reference = "The weather is nice today"
        candidate = "Today the weather is pleasant"

        results = evaluator.evaluate(reference, candidate)

        # Should have some similarity
        for result in results.values():
            assert result.score > 0.0

    def test_question_answering_evaluation(self):
        """Test evaluating question answering"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu", "rouge1", "exact_match"])

        reference = "Paris"
        candidate = "Paris"

        results = evaluator.evaluate(reference, candidate)

        # Exact match should be perfect
        if "exact_match" in results:
            assert results["exact_match"].score == 1.0


class TestPerformanceAndScalability:
    """Test performance and scalability"""

    def test_large_batch_evaluation(self):
        """Test evaluation with large batch"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"], max_workers=8)

        # Create large batch
        size = 100
        references = [f"text number {i}" for i in range(size)]
        candidates = [f"text number {i}" for i in range(size)]

        results = evaluator.evaluate_batch(references=references, candidates=candidates)

        assert len(results) == size

    def test_long_text_evaluation(self):
        """Test evaluation with very long text"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        # Create long text (1000 words)
        long_text = " ".join([f"word{i}" for i in range(1000)])

        results = evaluator.evaluate(reference=long_text, candidate=long_text)

        assert results["bleu"].score == 1.0

    def test_parallel_speedup(self):
        """Test that parallel processing provides speedup"""
        import time

        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        references = [f"text {i}" for i in range(50)]
        candidates = [f"text {i}" for i in range(50)]

        # Sequential
        start = time.time()
        evaluator.evaluate_batch(references, candidates, max_workers=1)
        seq_time = time.time() - start

        # Parallel
        start = time.time()
        evaluator.evaluate_batch(references, candidates, max_workers=4)
        par_time = time.time() - start

        # Parallel should be faster or comparable
        assert par_time <= seq_time * 1.2  # Allow 20% overhead


class TestErrorRecovery:
    """Test error handling and recovery"""

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        # Empty strings
        results = evaluator.evaluate(reference="", candidate="")
        assert "bleu" in results

    def test_metric_computation_error(self):
        """Test handling when metric computation fails"""
        evaluator = TextSimilarityEvaluator(metrics=["bleu"])

        # This should not crash even with edge cases
        results = evaluator.evaluate(reference="test", candidate="test")

        assert "bleu" in results
