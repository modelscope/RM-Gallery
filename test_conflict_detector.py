#!/usr/bin/env python3
"""
测试 Conflict Detector 的脚本
测试 pairwise 和 pointwise 两种模式
"""

import os
import sys

# 设置环境变量
os.environ["OPENAI_API_KEY"] = "sk-8aea8510fde3483981256623c393018e"
os.environ["BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from rm_gallery.gallery.evaluation.conflict_detector import main


def test_pairwise_mode():
    """测试 pairwise 模式"""
    print("\n" + "=" * 80)
    print("测试 Pairwise 模式")
    print("=" * 80 + "\n")

    main(
        data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
        result_path="data/results/conflict_pairwise_test.json",
        max_samples=5,  # 先用5个样本快速测试
        model="qwen2.5-32b-instruct",  # 使用qwen模型
        max_workers=4,  # 使用4个线程
        comparison_mode="pairwise",
        save_detailed_outputs=True,
        random_seed=42,
    )


def test_pointwise_mode():
    """测试 pointwise 模式"""
    print("\n" + "=" * 80)
    print("测试 Pointwise 模式")
    print("=" * 80 + "\n")

    main(
        data_path="data/benchmarks/reward-bench-2/data/test-00000-of-00001.parquet",
        result_path="data/results/conflict_pointwise_test.json",
        max_samples=5,  # 先用5个样本快速测试
        model="qwen2.5-32b-instruct",
        max_workers=4,
        comparison_mode="pointwise",
        save_detailed_outputs=True,
        random_seed=42,
    )


if __name__ == "__main__":
    try:
        # 测试 pairwise 模式
        test_pairwise_mode()

        print("\n\n" + "=" * 80)
        print("Pairwise 模式测试完成！开始 Pointwise 模式测试...")
        print("=" * 80 + "\n")

        # 测试 pointwise 模式
        test_pointwise_mode()

        print("\n\n" + "=" * 80)
        print("✅ 所有测试完成！")
        print("=" * 80)
        print("\n结果文件:")
        print("  - Pairwise 模式: data/results/conflict_pairwise_test.json")
        print("  - Pointwise 模式: data/results/conflict_pointwise_test.json")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
