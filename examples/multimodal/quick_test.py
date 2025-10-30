#!/usr/bin/env python3
"""
快速测试视觉Reward模型

这是一个简单的测试脚本，快速验证视觉reward模型是否工作正常。
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rm_gallery.core.data.multimodal_content import ImageContent, MultimodalContent
from rm_gallery.core.data.multimodal_message import MultimodalChatMessage
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, MessageRole
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.gallery.rm.multimodal import (
    QwenImageTextAlignmentReward,
    QwenMultimodalRankingReward,
    QwenVisualHelpfulnessReward,
)


def test_alignment():
    """测试图像-文本对齐reward"""
    print("\n" + "=" * 60)
    print("测试 1: 图像-文本对齐 (Image-Text Alignment)")
    print("=" * 60)

    # 检查API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        return False

    try:
        # 初始化
        vlm_api = QwenVLAPI(
            api_key=api_key, model_name="qwen-vl-plus", enable_cache=True
        )

        reward = QwenImageTextAlignmentReward(vlm_api=vlm_api)

        # 创建测试样本
        sample = DataSample(
            unique_id="test_alignment",
            input=[
                MultimodalChatMessage(
                    role=MessageRole.USER,
                    content=MultimodalContent(
                        text="描述这张图片",
                        images=[
                            ImageContent(
                                type="url",
                                data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
                            )
                        ],
                    ),
                )
            ],
            output=[
                DataOutput(
                    answer=ChatMessage(
                        role=MessageRole.ASSISTANT, content="一个女孩和一只狗在草地上玩耍"
                    )
                )
            ],
        )

        # 评估
        result = reward.evaluate(sample)
        score = result.output[0].answer.reward.score

        print(f"✓ 对齐得分: {score:.3f}")
        print("✓ 测试通过!")

        # 显示统计
        stats = reward.get_cost_stats()
        print(
            f"\n统计: 请求={stats['total_requests']}, 成本=${stats['estimated_cost_usd']:.4f}"
        )

        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_helpfulness():
    """测试视觉有帮助性reward"""
    print("\n" + "=" * 60)
    print("测试 2: 视觉有帮助性 (Visual Helpfulness)")
    print("=" * 60)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        return False

    try:
        # 初始化
        vlm_api = QwenVLAPI(
            api_key=api_key,
            model_name="qwen-vl-max",  # 使用max获得更好的评估质量
            enable_cache=True,
        )

        reward = QwenVisualHelpfulnessReward(vlm_api=vlm_api, use_detailed_rubric=True)

        # 创建测试样本
        sample = DataSample(
            unique_id="test_helpfulness",
            input=[
                MultimodalChatMessage(
                    role=MessageRole.USER,
                    content=MultimodalContent(
                        text="图片中的人在做什么？",
                        images=[
                            ImageContent(
                                type="url",
                                data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
                            )
                        ],
                    ),
                )
            ],
            output=[
                DataOutput(
                    answer=ChatMessage(
                        role=MessageRole.ASSISTANT, content="一个女孩正在草地上和一只金毛犬愉快地玩耍"
                    )
                )
            ],
        )

        # 评估
        result = reward.evaluate(sample)
        score = result.output[0].answer.reward.score

        print(f"✓ 有用性得分: {score:.3f}")
        print("✓ 测试通过!")

        # 显示统计
        stats = reward.get_cost_stats()
        print(
            f"\n统计: 请求={stats['total_requests']}, 成本=${stats['estimated_cost_usd']:.4f}"
        )

        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_ranking():
    """测试多模态排序reward"""
    print("\n" + "=" * 60)
    print("测试 3: 多模态排序 (Multimodal Ranking)")
    print("=" * 60)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        return False

    try:
        # 初始化
        vlm_api = QwenVLAPI(
            api_key=api_key, model_name="qwen-vl-plus", enable_cache=True
        )

        reward = QwenMultimodalRankingReward(
            vlm_api=vlm_api, ranking_metric="combined", use_parallel_evaluation=True
        )

        # 创建测试样本（多个候选答案）
        sample = DataSample(
            unique_id="test_ranking",
            input=[
                MultimodalChatMessage(
                    role=MessageRole.USER,
                    content=MultimodalContent(
                        text="描述这张图片",
                        images=[
                            ImageContent(
                                type="url",
                                data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
                            )
                        ],
                    ),
                )
            ],
            output=[
                DataOutput(answer=ChatMessage(content="一只狗")),
                DataOutput(answer=ChatMessage(content="一个女孩和一只金毛犬在草地上")),
                DataOutput(answer=ChatMessage(content="动物")),
            ],
        )

        # 评估
        result = reward.evaluate(sample)
        ranks = result.output[0].answer.reward.details[0].rank

        # 找到最佳答案
        best_idx = ranks.index(max(ranks))
        candidates = [o.answer.content for o in sample.output]

        print("排序结果:")
        for i, (rank, answer) in enumerate(zip(ranks, candidates)):
            marker = "★" if i == best_idx else " "
            print(f"  {marker} [{rank:.3f}] {answer}")

        print(f"\n✓ 最佳答案: {candidates[best_idx]}")
        print("✓ 测试通过!")

        # 显示统计
        stats = reward.get_cost_stats()
        print(
            f"\n统计: 请求={stats['total_requests']}, 成本=${stats['estimated_cost_usd']:.4f}"
        )

        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("视觉Reward模型快速测试")
    print("=" * 60)

    # 检查API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n❌ 错误: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("\n请运行以下命令设置API key:")
        print("  export DASHSCOPE_API_KEY='your_api_key_here'")
        print("\n或者在Python中:")
        print("  import os")
        print("  os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'")
        return

    # 运行测试
    results = []

    print("\n开始测试...\n")

    results.append(("图像-文本对齐", test_alignment()))
    results.append(("视觉有帮助性", test_helpfulness()))
    results.append(("多模态排序", test_ranking()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, passed in results:
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\n总计: {passed}/{total} 个测试通过")

    if passed == total:
        print("\n🎉 所有测试通过!")
        print("\n接下来:")
        print("  - 运行 python examples/multimodal/demo_vlm_rewards.py 查看完整演示")
        print("  - 查看 examples/multimodal/README.md 了解更多用法")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
