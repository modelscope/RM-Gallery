"""
Qwen VL API Usage Examples.

This script demonstrates how to use the Qwen VL API integration
for multimodal reward modeling tasks.

Prerequisites:
    1. Set DASHSCOPE_API_KEY environment variable
    2. Install dependencies: pip install openai httpx loguru
"""

import asyncio
import os

from loguru import logger

from rm_gallery.core.data.multimodal_content import ImageContent
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI


async def example_basic_generation():
    """Example 1: Basic text generation with image."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Generation with Image")
    logger.info("=" * 60)

    # Initialize API
    api = QwenVLAPI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_name="qwen-vl-plus",  # or "qwen-vl-max" for higher accuracy
        enable_cache=True,
    )

    # Create image content
    image = ImageContent(
        type="url",
        data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
    )

    # Generate description
    response = await api.generate(
        text="请详细描述这张图片的内容。", images=[image], temperature=0.7, max_tokens=200
    )

    logger.info(f"Generated Description:\n{response.content}")
    logger.info(f"Token Usage: {response.token_usage}")
    logger.info("")


async def example_similarity_computation():
    """Example 2: Compute image-text similarity."""
    logger.info("=" * 60)
    logger.info("Example 2: Image-Text Similarity Computation")
    logger.info("=" * 60)

    api = QwenVLAPI(api_key=os.getenv("DASHSCOPE_API_KEY"), model_name="qwen-vl-plus")

    image = ImageContent(
        type="url",
        data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
    )

    # Test different descriptions
    descriptions = ["一个女孩和一只狗在草地上玩耍", "一只金色的拉布拉多犬", "一辆汽车在高速公路上行驶", "一座高楼大厦"]

    logger.info("Computing similarity scores...")
    for desc in descriptions:
        score = await api.compute_similarity(image, desc)
        logger.info(f"  '{desc}': {score:.3f}")

    logger.info("")


async def example_quality_evaluation():
    """Example 3: Evaluate response quality."""
    logger.info("=" * 60)
    logger.info("Example 3: Response Quality Evaluation")
    logger.info("=" * 60)

    api = QwenVLAPI(api_key=os.getenv("DASHSCOPE_API_KEY"), model_name="qwen-vl-plus")

    image = ImageContent(
        type="url",
        data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
    )

    # Different quality responses
    responses = {
        "Detailed": "这张图片展示了一个穿着蓝色裙子的小女孩和一只金色的拉布拉多犬在绿色的草地上玩耍。女孩笑容灿烂，狗狗看起来很友好，整个画面充满了温馨和快乐的氛围。",
        "Brief": "一个女孩和一只狗。",
        "Vague": "这是一张照片。",
        "Inaccurate": "一只猫在树上睡觉。",
    }

    logger.info("Evaluating response quality...")
    for label, response_text in responses.items():
        score = await api.evaluate_quality(image, response_text)
        logger.info(f"  {label:12s}: {score:.3f} - '{response_text[:40]}...'")

    logger.info("")


async def example_batch_processing():
    """Example 4: Batch processing multiple requests."""
    logger.info("=" * 60)
    logger.info("Example 4: Batch Processing")
    logger.info("=" * 60)

    api = QwenVLAPI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_name="qwen-vl-plus",
        enable_cache=True,
    )

    image = ImageContent(
        type="url",
        data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
    )

    # Create batch of requests
    prompts = ["描述图片中的人物。", "描述图片中的动物。", "描述图片的背景环境。", "描述图片的整体氛围。", "这张图片适合什么场景使用？"]

    logger.info(f"Processing {len(prompts)} requests concurrently...")

    # Execute concurrently
    tasks = [
        api.generate(text=prompt, images=[image], max_tokens=100) for prompt in prompts
    ]

    responses = await asyncio.gather(*tasks)

    logger.info("\nResults:")
    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        logger.info(f"{i}. Q: {prompt}")
        logger.info(f"   A: {response.content}\n")

    # Show statistics
    cost_stats = api.get_cost_stats()
    cache_stats = await api.get_cache_stats()

    logger.info(f"Cost Statistics: {cost_stats}")
    logger.info(f"Cache Statistics: {cache_stats}")
    logger.info("")


async def example_with_system_prompt():
    """Example 5: Using system prompts for specialized evaluation."""
    logger.info("=" * 60)
    logger.info("Example 5: System Prompt Usage")
    logger.info("=" * 60)

    api = QwenVLAPI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_name="qwen-vl-max",  # Use Max for better instruction following
    )

    image = ImageContent(
        type="url",
        data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
    )

    system_prompt = """你是一个专业的图片质量评估专家。
请从以下几个维度评估图片：
1. 构图 (0-10分)
2. 光线 (0-10分)
3. 清晰度 (0-10分)
4. 情感表达 (0-10分)

输出格式：
构图：X分 - 简短评语
光线：X分 - 简短评语
清晰度：X分 - 简短评语
情感表达：X分 - 简短评语
总体评分：X分"""

    response = await api.generate(
        text="请评估这张图片。",
        images=[image],
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=300,
    )

    logger.info(f"Evaluation Result:\n{response.content}")
    logger.info("")


async def example_cost_tracking():
    """Example 6: Monitor API costs."""
    logger.info("=" * 60)
    logger.info("Example 6: Cost Tracking")
    logger.info("=" * 60)

    api = QwenVLAPI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_name="qwen-vl-plus",
        enable_cache=True,
    )

    image = ImageContent(
        type="url",
        data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
    )

    # Make several requests
    for i in range(5):
        await api.generate(text=f"请简短描述这张图片。（请求 {i+1}）", images=[image], max_tokens=50)

    # Make same request again (should hit cache)
    for i in range(3):
        await api.generate(text="请简短描述这张图片。（请求 1）", images=[image], max_tokens=50)

    # Get statistics
    cost_stats = api.get_cost_stats()
    cache_stats = await api.get_cache_stats()

    logger.info("Final Statistics:")
    logger.info(f"  Total Requests: {cost_stats['total_requests']}")
    logger.info(f"  Cache Hits: {cost_stats['cache_hits']}")
    logger.info(f"  Cache Misses: {cost_stats['cache_misses']}")
    logger.info(f"  Cache Hit Rate: {cost_stats['cache_rate']}")
    logger.info(f"  Total Tokens: {cost_stats['total_tokens']}")
    logger.info(f"  Estimated Cost: {cost_stats['estimated_cost_usd']}")
    logger.info(f"  Saved Cost (from cache): {cost_stats['saved_cost_usd']}")
    logger.info(f"  Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
    logger.info("")


async def example_error_handling():
    """Example 7: Robust error handling."""
    logger.info("=" * 60)
    logger.info("Example 7: Error Handling")
    logger.info("=" * 60)

    api = QwenVLAPI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_name="qwen-vl-plus",
        max_retries=3,  # Will retry on transient errors
    )

    # Test with potentially problematic inputs
    test_cases = [
        (
            "Valid request",
            ImageContent(
                type="url",
                data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
            ),
            "描述图片",
        ),
        (
            "Empty text",
            ImageContent(
                type="url",
                data="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
            ),
            "",
        ),
    ]

    for label, image, text in test_cases:
        try:
            logger.info(f"Testing: {label}")
            response = await api.generate(text=text, images=[image], max_tokens=50)
            logger.info(f"  ✓ Success: {response.content[:50]}...")
        except Exception as e:
            logger.error(f"  ✗ Error: {type(e).__name__}: {str(e)}")

    logger.info("")


async def example_health_check():
    """Example 8: API health check."""
    logger.info("=" * 60)
    logger.info("Example 8: Health Check")
    logger.info("=" * 60)

    api = QwenVLAPI(api_key=os.getenv("DASHSCOPE_API_KEY"), model_name="qwen-vl-plus")

    logger.info("Performing health check...")
    is_healthy = await api.health_check()

    if is_healthy:
        logger.info("✓ API is healthy and responding normally")
    else:
        logger.error("✗ API health check failed")

    # Get circuit breaker state
    cb_state = api.get_circuit_breaker_state()
    logger.info(f"Circuit Breaker State: {cb_state['state']}")
    logger.info("")


async def main():
    """Run all examples."""
    # Check if API key is set
    if not os.getenv("DASHSCOPE_API_KEY"):
        logger.error("DASHSCOPE_API_KEY environment variable not set!")
        logger.error("Please set it before running examples:")
        logger.error("  export DASHSCOPE_API_KEY='your-api-key-here'")
        return

    logger.info("\n" + "=" * 60)
    logger.info("Qwen VL API Examples")
    logger.info("=" * 60 + "\n")

    try:
        # Run examples
        await example_basic_generation()
        await example_similarity_computation()
        await example_quality_evaluation()
        await example_batch_processing()
        await example_with_system_prompt()
        await example_cost_tracking()
        await example_error_handling()
        await example_health_check()

        logger.info("=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nExamples interrupted by user")
    except Exception as e:
        logger.error(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
