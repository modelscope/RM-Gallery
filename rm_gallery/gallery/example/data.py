# DataSample 使用示例：从数据到评估的完整流程

import asyncio

from rm_gallery.core.data import DataSample, DataSampleMapping
from rm_gallery.core.grader import evaluate
from rm_gallery.gallery.example.llm import FactualGrader


# 示例1: 数据已经匹配reward关键字，直接进行评估
async def example_evaluate_without_mapping():
    """
    示例1: 数据已经匹配reward关键字，直接进行评估

    展示从准备数据到使用FactualGrader进行评估的完整流程。
    数据键名已经与评分器期望的参数名匹配，无需映射。
    """

    print("=== 示例1: 无需映射的数据评估流程 ===")

    # 1. 创建FactualGrader评分器
    grader = FactualGrader()
    print("1. 创建评分器: {grader.name} ({grader.__class__.__name__})")

    # 2. 准备数据（键名已匹配）
    data_sample = DataSample(
        data={"query": "What is the capital of France?"},
        samples=[
            {"answer": "Paris"},  # 应该得分高
            {"answer": "London"},  # 应该得分低
            {"answer": "Berlin"},  # 应该得分低
        ],
    )
    print("2. 准备数据:")
    print(f"   查询问题: {data_sample.data['query']}")
    print(f"   回答选项: {[s['answer'] for s in data_sample.samples]}")

    # 3. 执行评估过程
    print("3. 执行评估:")
    try:
        results = await evaluate(
            grader=grader, mapping=None, data_sample=data_sample  # 无需映射
        )
        print("4. 评估结果:")
        for i, result in enumerate(results):
            print(
                f"   回答 '{data_sample.samples[i]['answer']}': Score={result.score:.2f}, Reason='{result.reason}'"
            )
    except Exception as e:
        print(f"   评估出错: {e}")

    print()


# 示例2: 数据不匹配reward关键字，需要映射后进行评估
async def example_evaluate_with_mapping():
    """
    示例2: 数据不匹配reward关键字，需要映射后进行评估

    展示从准备不匹配的数据，通过映射，再到使用FactualGrader进行评估的完整流程。
    """

    print("=== 示例2: 需要映射的数据评估流程 ===")

    # 1. 创建FactualGrader评分器
    grader = FactualGrader()
    print(f"1. 创建评分器: {grader.name} ({grader.__class__.__name__})")

    # 2. 准备数据（键名不匹配）
    raw_data_sample = DataSample(
        data={"question": "What is the capital of France?"},
        samples=[
            {"response": "Paris"},  # 应该得分高
            {"response": "London"},  # 应该得分低
            {"response": "Berlin"},  # 应该得分低
        ],
    )
    print("2. 准备原始数据（键名不匹配）:")
    print(f"   查询问题: {raw_data_sample.data['question']}")
    print(f"   回答选项: {[s['response'] for s in raw_data_sample.samples]}")

    # 3. 定义映射规则
    mapping = DataSampleMapping(
        data_mapping={"query": "question"},  # 将query映射到question
        sample_mapping={"answer": "response"},  # 将answer映射到response
    )
    print("3. 定义映射规则:")
    print(f"   数据映射: {mapping.data_mapping}")
    print(f"   样本映射: {mapping.sample_mapping}")

    # 4. 执行评估过程（映射会在evaluate函数内部自动应用）
    print("4. 执行评估（映射会自动应用）:")
    try:
        results = await evaluate(
            grader=grader, mapping=mapping, data_sample=raw_data_sample  # 提供映射规则
        )
        print("5. 评估结果:")
        for i, result in enumerate(results):
            print(
                f"   回答 '{raw_data_sample.samples[i]['response']}': Score={result.score:.2f}, Reason='{result.reason}'"
            )
    except Exception as e:
        print(f"   评估出错: {e}")

    print()


if __name__ == "__main__":
    # 运行两个示例
    asyncio.run(example_evaluate_without_mapping())
    asyncio.run(example_evaluate_with_mapping())
