#!/usr/bin/env python3
"""
Test 10 samples for both pairwise and pointwise modes
"""

import json
import os
import sys

# Set environment variables
os.environ["OPENAI_API_KEY"] = "***REMOVED***"
os.environ["BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from rm_gallery.gallery.evaluation.conflict_detector import main


def print_comparison_summary(pairwise_result: dict, pointwise_result: dict):
    """Print comparison summary between two modes"""
    print("\n" + "=" * 80)
    print("MODE COMPARISON SUMMARY (10 Samples)")
    print("=" * 80)

    # Accuracy comparison
    print("\n📊 Accuracy Metrics:")
    pw_acc = pairwise_result["accuracy_metrics"]["accuracy"]
    pt_acc = pointwise_result["accuracy_metrics"]["accuracy"]
    print(f"  {'Metric':<30} {'Pairwise':<15} {'Pointwise':<15} {'Difference':<15}")
    print("  " + "-" * 75)
    print(
        f"  {'Accuracy':<30} {pw_acc:<15.2%} {pt_acc:<15.2%} {(pw_acc-pt_acc)*100:+.1f}%"
    )

    pw_dom = pairwise_result["accuracy_metrics"]["chosen_dominance_rate"]
    pt_dom = pointwise_result["accuracy_metrics"]["chosen_dominance_rate"]
    print(
        f"  {'Chosen dominance rate':<30} {pw_dom:<15.2%} {pt_dom:<15.2%} {(pw_dom-pt_dom)*100:+.1f}%"
    )

    pw_ties = pairwise_result["accuracy_metrics"]["chosen_tie_rate"]
    pt_ties = pointwise_result["accuracy_metrics"]["chosen_tie_rate"]
    print(
        f"  {'Tie rate':<30} {pw_ties:<15.2%} {pt_ties:<15.2%} {(pw_ties-pt_ties)*100:+.1f}%"
    )

    # Conflict comparison
    print("\n🔍 Conflict Metrics:")
    pw_conf = pairwise_result["conflict_metrics"]["overall_conflict_rate"]
    pt_conf = pointwise_result["conflict_metrics"]["overall_conflict_rate"]
    print(
        f"  {'Overall conflict rate':<30} {pw_conf:<15.2f}% {pt_conf:<15.2f}% {(pw_conf-pt_conf):+.2f}%"
    )

    pw_conf_samples = pairwise_result["conflict_metrics"]["samples_with_conflicts"]
    pt_conf_samples = pointwise_result["conflict_metrics"]["samples_with_conflicts"]
    print(
        f"  {'Samples with conflicts':<30} {pw_conf_samples:<15} {pt_conf_samples:<15}"
    )

    # Conclusion
    print("\n💡 Conclusion:")
    if pw_acc > pt_acc:
        print(f"  ✓ Pairwise mode has higher accuracy (+{(pw_acc - pt_acc)*100:.1f}%)")
    elif pt_acc > pw_acc:
        print(f"  ✓ Pointwise mode has higher accuracy (+{(pt_acc - pw_acc)*100:.1f}%)")
    else:
        print("  ✓ Both modes have equal accuracy")

    if pw_conf < pt_conf:
        print(
            f"  ✓ Pairwise mode has lower conflict rate ({pw_conf:.2f}% vs {pt_conf:.2f}%)"
        )
    elif pt_conf < pw_conf:
        print(
            f"  ✓ Pointwise mode has lower conflict rate ({pt_conf:.2f}% vs {pw_conf:.2f}%)"
        )
    else:
        print("  ✓ Both modes have equal conflict rate")

    if pw_ties < pt_ties:
        print(f"  ✓ Pairwise mode has fewer ties ({pw_ties:.1%} vs {pt_ties:.1%})")
    elif pt_ties < pw_ties:
        print(f"  ✓ Pointwise mode has fewer ties ({pt_ties:.1%} vs {pw_ties:.1%})")


if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("TESTING 10 SAMPLES - PAIRWISE MODE")
        print("=" * 80 + "\n")

        # Test Pairwise mode
        main(
            data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
            result_path="data/results/test_10_pairwise.json",
            max_samples=100,
            model="qwen3-32b",
            max_workers=60,
            comparison_mode="pairwise",
            save_detailed_outputs=False,
            random_seed=42,
        )

        print("\n\n" + "=" * 80)
        print("TESTING 10 SAMPLES - POINTWISE MODE")
        print("=" * 80 + "\n")

        # Test Pointwise mode
        main(
            data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
            result_path="data/results/test_10_pointwise.json",
            max_samples=100,
            model="qwen3-32b",
            max_workers=60,
            comparison_mode="pointwise",
            save_detailed_outputs=False,
            random_seed=42,
        )

        # Load results and compare
        with open("data/results/test_10_pairwise.json", "r") as f:
            pairwise_result = json.load(f)

        with open("data/results/test_10_pointwise.json", "r") as f:
            pointwise_result = json.load(f)

        # Print comparison
        print_comparison_summary(pairwise_result, pointwise_result)

        print("\n" + "=" * 80)
        print("✅ All tests completed successfully!")
        print("=" * 80)
        print("\nResult files:")
        print("  - Pairwise: data/results/test_10_pairwise.json")
        print("  - Pointwise: data/results/test_10_pointwise.json")
        print()

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
