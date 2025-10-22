#!/usr/bin/env python3
"""
Test script for RM-Gallery LLM Judge evaluation modes

Tests all LLM as Judge evaluation modes:
- pairwise-winrate: Simple win rate calculation
- pairwise-copeland: Copeland method (net wins)
- pairwise-dgr: DGR (TFAS) algorithm
- pairwise-elo: ELO Rating algorithm
- pointwise: Direct scoring (1-10)
- listwise: Ranking
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
sys.path.insert(0, project_root)

from rm_gallery.gallery.evaluation.llm_judge import AlignmentReward

# Test data
TEST_CASES = [
    {
        "name": "Python斐波那契函数",
        "prompt": "请帮我写一个Python函数来计算斐波那契数列。",
        "responses": [
            # Response 1: 简单但不完整
            "好的，这里是一个递归实现：\n\n```python\ndef fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n```\n\n这个函数使用递归计算。",
            # Response 2: 详细且提供多种方案
            "我来提供两种实现方式：\n\n**方法1：递归实现（简单但效率低）**\n```python\ndef fibonacci_recursive(n):\n    if n <= 1:\n        return n\n    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)\n```\n\n**方法2：迭代实现（推荐，效率高）**\n```python\ndef fibonacci_iterative(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```\n\n推荐使用迭代版本，因为它的时间复杂度是O(n)，而递归版本是O(2^n)。",
            # Response 3: 中等质量
            "这是一个斐波那契函数的实现：\n\n```python\ndef fibonacci(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n```",
        ],
        "reference": "我很乐意帮你写一个计算斐波那契数列的Python函数！\n\n这里有两种常见的实现方式：\n\n**方法1：递归实现（简单但效率较低）**\n```python\ndef fibonacci_recursive(n):\n    if n <= 1:\n        return n\n    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)\n```\n\n**方法2：迭代实现（推荐，效率高）**\n```python\ndef fibonacci_iterative(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```\n\n推荐使用迭代版本，因为它的时间复杂度是O(n)，而递归版本是O(2^n)。",
    },
    {
        "name": "机器学习解释",
        "prompt": "什么是机器学习？请简单解释一下。",
        "responses": [
            # Response 1: 简单准确
            "机器学习是人工智能的一个分支，它让计算机通过数据和经验自动学习和改进，而不需要明确编程。主要包括监督学习、无监督学习和强化学习三种类型。",
            # Response 2: 详细但有些冗长
            "机器学习（Machine Learning）是人工智能的核心领域之一。它的基本思想是让计算机系统能够从数据中学习模式和规律，从而在没有明确编程指令的情况下完成特定任务。\n\n主要分为三大类：\n1. 监督学习：使用标注数据训练模型\n2. 无监督学习：从未标注数据中发现模式\n3. 强化学习：通过与环境交互学习最优策略\n\n常见应用包括图像识别、自然语言处理、推荐系统等。",
            # Response 3: 过于简单
            "机器学习就是让计算机自己学习的技术。",
        ],
        "reference": "机器学习是人工智能的一个分支，它让计算机通过数据和经验自动学习和改进，而不需要明确编程。主要包括监督学习、无监督学习和强化学习三种类型。",
    },
]


def test_evaluation_mode(
    mode_name: str,
    eval_mode: str,
    pairwise_mode: str = None,
    api_key: str = None,
    base_url: str = None,
    model_name: str = "qwen3-32b",
    verbose: bool = False,
):
    """
    Test a specific evaluation mode

    Args:
        mode_name: Display name for this mode
        eval_mode: 'pairwise', 'pointwise', or 'listwise'
        pairwise_mode: 'winrate' or 'dgr' (only for pairwise)
        api_key: OpenAI API key
        base_url: API base URL
        model_name: Model name
        verbose: Print detailed info
    """
    print(f"\n{'='*80}")
    print(f"Testing: {mode_name}")
    print(f"{'='*80}\n")

    # Create reward instance
    kwargs = {
        "model_name": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "eval_mode": eval_mode,
        "max_workers": 5,
        "verbose": verbose,
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    if pairwise_mode:
        kwargs["pairwise_mode"] = pairwise_mode

    try:
        # All modes use AlignmentReward (including ELO)
        reward = AlignmentReward(**kwargs)
    except Exception as e:
        print(f"❌ Failed to create reward instance: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test each case
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        print(f"Prompt: {test_case['prompt'][:50]}...")
        print(f"Number of responses: {len(test_case['responses'])}")

        start_time = time.time()

        try:
            result = reward.compute_score(
                data_source="test",
                solution_str=test_case["responses"],
                ground_truth={},
                extra_info={
                    "x": [{"role": "user", "content": test_case["prompt"]}],
                    "chosen": [
                        {"role": "user", "content": test_case["prompt"]},
                        {"role": "assistant", "content": test_case["reference"]},
                    ],
                },
                group_evaluation=True,
            )

            elapsed = time.time() - start_time

            # Display results
            scores = result.get("group_scores", [])
            print(f"\n✓ Evaluation completed in {elapsed:.2f}s")
            print(f"Mode: {result.get('mode', 'unknown')}")
            print(f"Scores: {[f'{s:.3f}' for s in scores]}")

            # Show comparisons for pairwise mode
            if eval_mode == "pairwise" and "comparisons" in result:
                comparisons = result["comparisons"]
                print(f"Comparisons: {len(comparisons)} pairs evaluated")
                winners = [c.get("winner") for c in comparisons]
                ties = sum(1 for w in winners if w is None)
                print(f"  Ties: {ties}/{len(comparisons)}")
                if ties < len(comparisons):
                    print(
                        f"  Example comparison: {comparisons[0].get('result', 'N/A')}"
                    )

            if "conflicts_removed" in result:
                print(f"Conflicts removed (DGR): {result['conflicts_removed']}")

            if "net_wins" in result:
                print(f"Net wins: {[f'{nw:.2f}' for nw in result['net_wins']]}")

            # Rank responses by score
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            print("\nRanking (best to worst):")
            for rank, (idx, score) in enumerate(ranked, 1):
                preview = test_case["responses"][idx][:80].replace("\n", " ")
                print(f"  {rank}. Response {idx+1} (score: {score:.3f}): {preview}...")

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Main test function"""
    print("\n" + "=" * 80)
    print("RM-Gallery LLM Judge - Reward Mode Testing")
    print("=" * 80)

    # Get API configuration from environment
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "qwen3-32b")

    if not api_key:
        print("❌ Error: OPENAI_API_KEY not set in environment")
        print("Please set OPENAI_API_KEY environment variable")
        return 1

    print(f"API Base URL: {base_url}")
    print(f"Model: {model_name}")
    print(f"Number of test cases: {len(TEST_CASES)}")

    # Test modes
    test_modes = [
        ("Pairwise - Winrate", "winrate"),
        ("Pairwise - Copeland (Net Wins)", "copeland"),
        ("Pairwise - DGR (TFAS)", "dgr"),
        ("Pairwise - ELO Rating", "elo"),
        ("Pointwise (1-10 scoring)", "pointwise"),
        ("Listwise (Ranking)", "listwise"),
    ]

    for mode_name, mode_type in test_modes:
        # Determine if it's pairwise or other modes
        if mode_type in ["winrate", "copeland", "dgr", "elo"]:
            test_evaluation_mode(
                mode_name=mode_name,
                eval_mode="pairwise",
                pairwise_mode=mode_type,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                verbose=False,
            )
        else:  # pointwise or listwise
            test_evaluation_mode(
                mode_name=mode_name,
                eval_mode=mode_type,
                pairwise_mode=None,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                verbose=False,
            )

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
