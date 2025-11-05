"""
Test Runner for Phase 4

Quick test to verify core functionality works before running full test suite.
"""


def test_basic_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    # Test imports - variables used in verification logic
    from rm_gallery.core.metrics import (  # noqa: F401
        ComparisonInput,
        MetricResult,
        TextSimilarityEvaluator,
    )
    from rm_gallery.core.metrics.registry import (  # noqa: F401
        get_metric,
        list_available_metrics,
    )

    print("✓ All imports successful")


def test_fuzzy_match():
    """Test fuzzy match metric"""
    print("\nTesting Fuzzy Match...")

    from rm_gallery.core.metrics.registry import get_metric
    from rm_gallery.core.metrics.schema import ComparisonInput

    fuzzy = get_metric("fuzzy_match")
    if fuzzy is None:
        print("✗ Fuzzy match metric not found")
        return False

    input_data = ComparisonInput(reference="hello world", candidate="hello world")

    result = fuzzy.compute(input_data)

    if result.score == 1.0:
        print(f"✓ Fuzzy match works: {result.score}")
        return True
    else:
        print(f"✗ Unexpected score: {result.score}")
        return False


def test_evaluator():
    """Test unified evaluator"""
    print("\nTesting TextSimilarityEvaluator...")

    from rm_gallery.core.metrics.evaluator import TextSimilarityEvaluator

    # Try with a safe subset of metrics
    evaluator = TextSimilarityEvaluator(metrics=["fuzzy_match", "exact_match"])

    results = evaluator.evaluate(reference="test", candidate="test")

    if len(results) > 0:
        print(f"✓ Evaluator works with {len(results)} metrics")
        for name, result in results.items():
            print(f"  - {name}: {result.score:.4f}")
        return True
    else:
        print("✗ No results returned")
        return False


def test_batch_evaluation():
    """Test batch evaluation"""
    print("\nTesting Batch Evaluation...")

    from rm_gallery.core.metrics.evaluator import TextSimilarityEvaluator

    evaluator = TextSimilarityEvaluator(metrics=["fuzzy_match"])

    references = ["test1", "test2", "test3"]
    candidates = ["test1", "test2", "test3"]

    try:
        results = evaluator.evaluate_batch(references=references, candidates=candidates)

        if len(results) == 3:
            print(f"✓ Batch evaluation works: {len(results)} samples processed")
            return True
        else:
            print(f"✗ Expected 3 results, got {len(results)}")
            return False
    except Exception as e:
        print(f"✗ Batch evaluation failed: {e}")
        return False


def main():
    """Run all quick tests"""
    print("=" * 60)
    print("Phase 4 Quick Test Runner")
    print("=" * 60)

    tests = [
        test_basic_imports,
        test_fuzzy_match,
        test_evaluator,
        test_batch_evaluation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test() is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} raised exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
