#!/usr/bin/env python3
"""
全面测试 Conflict Detector 的脚本
测试更多样本，验证冲突检测和指标计算
"""

import json
import os
import sys

# 设置环境变量
os.environ["OPENAI_API_KEY"] = "***REMOVED***"
os.environ["BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from rm_gallery.gallery.evaluation.conflict_detector import main


def print_result_summary(result_path: str, mode: str):
    """打印结果摘要"""
    with open(result_path, "r") as f:
        results = json.load(f)

    print(f"\n{'='*80}")
    print(f"{mode.upper()} 模式结果摘要")
    print(f"{'='*80}")
    print(f"模型: {results['model']}")
    print(f"比较模式: {results['comparison_mode']}")

    # 准确率指标
    acc = results["accuracy_metrics"]
    print("\n✅ 准确率指标:")
    print(f"  - 准确率: {acc['accuracy']:.2%}")
    print(f"  - Chosen主导率: {acc['chosen_dominance_rate']:.2%}")
    print(
        f"  - Chosen获胜: {acc['total_chosen_wins']}/{acc['total_chosen_vs_rejected_comparisons']}"
    )
    print(f"  - Chosen失败: {acc['total_chosen_losses']}")
    print(f"  - 平局: {acc['total_chosen_ties']}")

    # 冲突指标
    conflict = results["conflict_metrics"]
    print("\n🔍 冲突指标:")
    print(f"  - 总体冲突率: {conflict['overall_conflict_rate']:.2f}%")
    print(f"  - 平均每样本冲突节点数: {conflict['conflict_density_rate']:.2f}")
    print(f"  - 平均每比较冲突节点数: {conflict['conflicts_per_comparison']:.4f}")
    print(f"  - 总比较次数: {conflict['total_comparisons']}")

    # 评估摘要
    summary = results["evaluation_summary"]
    print("\n📊 评估摘要:")
    print(f"  - 成功样本: {summary['successful_samples']}/{results['total_count']}")
    print(f"  - 成功率: {summary['success_rate']:.2%}")


def test_comprehensive():
    """全面测试：使用更多样本"""
    print("\n" + "=" * 80)
    print("全面测试 - Pairwise 模式 (20个样本)")
    print("=" * 80 + "\n")

    main(
        data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
        result_path="data/results/conflict_pairwise_comprehensive.json",
        max_samples=20,
        model="qwen2.5-32b-instruct",
        max_workers=8,
        comparison_mode="pairwise",
        save_detailed_outputs=False,  # 节省空间
        random_seed=42,
    )

    print_result_summary(
        "data/results/conflict_pairwise_comprehensive.json", "pairwise"
    )

    print("\n\n" + "=" * 80)
    print("全面测试 - Pointwise 模式 (20个样本)")
    print("=" * 80 + "\n")

    main(
        data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
        result_path="data/results/conflict_pointwise_comprehensive.json",
        max_samples=20,
        model="qwen2.5-32b-instruct",
        max_workers=8,
        comparison_mode="pointwise",
        save_detailed_outputs=False,
        random_seed=42,
    )

    print_result_summary(
        "data/results/conflict_pointwise_comprehensive.json", "pointwise"
    )


def compare_modes():
    """比较两种模式的性能"""
    print("\n" + "=" * 80)
    print("模式比较分析")
    print("=" * 80)

    with open("data/results/conflict_pairwise_comprehensive.json", "r") as f:
        pairwise = json.load(f)

    with open("data/results/conflict_pointwise_comprehensive.json", "r") as f:
        pointwise = json.load(f)

    print("\n性能对比 (基于20个样本):")
    print(f"\n{'指标':<30} {'Pairwise':<20} {'Pointwise':<20}")
    print("-" * 70)

    pw_acc = pairwise["accuracy_metrics"]["accuracy"]
    pt_acc = pointwise["accuracy_metrics"]["accuracy"]
    print(f"{'准确率':<30} {pw_acc:<20.2%} {pt_acc:<20.2%}")

    pw_dom = pairwise["accuracy_metrics"]["chosen_dominance_rate"]
    pt_dom = pointwise["accuracy_metrics"]["chosen_dominance_rate"]
    print(f"{'Chosen主导率':<30} {pw_dom:<20.2%} {pt_dom:<20.2%}")

    pw_conf = pairwise["conflict_metrics"]["overall_conflict_rate"]
    pt_conf = pointwise["conflict_metrics"]["overall_conflict_rate"]
    print(f"{'冲突率':<30} {pw_conf:<20.2f}% {pt_conf:<20.2f}%")

    pw_comp = pairwise["conflict_metrics"]["total_comparisons"]
    pt_comp = pointwise["conflict_metrics"]["total_comparisons"]
    print(f"{'总比较次数':<30} {pw_comp:<20} {pt_comp:<20}")

    print("\n结论:")
    if pw_acc > pt_acc:
        print(f"  ✓ Pairwise模式准确率更高 (+{(pw_acc - pt_acc)*100:.1f}%)")
    else:
        print(f"  ✓ Pointwise模式准确率更高 (+{(pt_acc - pw_acc)*100:.1f}%)")

    if pw_conf < pt_conf:
        print("  ✓ Pairwise模式冲突率更低")
    elif pt_conf < pw_conf:
        print("  ✓ Pointwise模式冲突率更低")
    else:
        print("  ✓ 两种模式冲突率相同")


if __name__ == "__main__":
    try:
        # 运行全面测试
        test_comprehensive()

        # 比较两种模式
        compare_modes()

        print("\n\n" + "=" * 80)
        print("✅ 全面测试完成！")
        print("=" * 80)
        print("\n结果文件:")
        print("  - Pairwise 全面测试: data/results/conflict_pairwise_comprehensive.json")
        print("  - Pointwise 全面测试: data/results/conflict_pointwise_comprehensive.json")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
