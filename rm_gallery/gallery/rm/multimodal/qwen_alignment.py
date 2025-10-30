"""
Qwen VL Image-Text Alignment Reward Model.

Evaluates the alignment/similarity between image content and text descriptions
using Qwen VL API. Suitable for tasks like image captioning, visual QA, and
multimodal content validation.
"""

import asyncio
import os
from typing import List, Optional

from loguru import logger
from pydantic import Field

from rm_gallery.core.data.multimodal_content import ImageContent
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.reward.vlm_reward import BasePointWiseVLMReward


@RewardRegistry.register("qwen_image_text_alignment")
class QwenImageTextAlignmentReward(BasePointWiseVLMReward):
    """
    Qwen VL Image-Text Alignment Reward.

    Evaluates how well a text description matches the content of an image
    using Qwen VL models (qwen-vl-plus or qwen-vl-max).

    Key Features:
    - Point-wise evaluation (single response scoring)
    - 0-1 normalized score output
    - Built-in caching to reduce API costs
    - Async API calls for better performance
    - Automatic cost tracking

    Attributes:
        name: Reward model identifier
        vlm_api: Qwen VL API client
        similarity_prompt_template: Prompt template for similarity evaluation
        fallback_score: Default score when evaluation fails

    Examples:
        >>> # Initialize with API key
        >>> reward = QwenImageTextAlignmentReward(
        ...     vlm_api=QwenVLAPI(
        ...         api_key=os.getenv("DASHSCOPE_API_KEY"),
        ...         model_name="qwen-vl-plus"
        ...     )
        ... )
        >>>
        >>> # Evaluate a sample
        >>> sample = DataSample(
        ...     unique_id="sample_001",
        ...     input=[MultimodalChatMessage(
        ...         role=MessageRole.USER,
        ...         content=MultimodalContent(
        ...             text="Describe this image",
        ...             images=[ImageContent(type="url", data="https://...")]
        ...         )
        ...     )],
        ...     output=[DataOutput(
        ...         answer=ChatMessage(
        ...             role=MessageRole.ASSISTANT,
        ...             content="A cat sitting on a couch"
        ...         )
        ...     )]
        ... )
        >>> result = reward.evaluate(sample)
        >>> print(f"Alignment score: {result.output[0].answer.reward.score:.3f}")
        >>>
        >>> # Check cost statistics
        >>> stats = reward.get_cost_stats()
        >>> print(f"Total requests: {stats['total_requests']}")
        >>> print(f"Cache rate: {stats['cache_rate']}")
        >>> print(f"Estimated cost: {stats['estimated_cost_usd']}")
    """

    name: str = Field(
        default="qwen_image_text_alignment", description="Reward model name"
    )

    vlm_api: QwenVLAPI = Field(
        default_factory=lambda: QwenVLAPI(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            model_name="qwen-vl-plus",  # Default to plus (faster, cheaper)
            enable_cache=True,
        ),
        description="Qwen VL API client",
    )

    similarity_prompt_template: str = Field(
        default=(
            "请仔细观察图片内容，评估以下描述与图片的匹配程度（0-10分）：\n\n"
            "描述：{text}\n\n"
            "评分标准：\n"
            "- 10分：完全准确，描述了图片的所有关键信息\n"
            "- 7-9分：大部分准确，捕捉了主要内容\n"
            "- 4-6分：部分准确，遗漏了一些重要信息\n"
            "- 1-3分：基本不准确，与图片内容不符\n"
            "- 0分：完全错误\n\n"
            "只回答一个0-10之间的数字，不要有任何其他内容。"
        ),
        description="Prompt template for similarity evaluation",
    )

    # Use English prompt for better performance with multilingual content
    similarity_prompt_template_en: str = Field(
        default=(
            "Carefully observe the image content and rate how well the following description "
            "matches the image (0-10 score):\n\n"
            "Description: {text}\n\n"
            "Scoring criteria:\n"
            "- 10: Perfectly accurate, describes all key information\n"
            "- 7-9: Mostly accurate, captures main content\n"
            "- 4-6: Partially accurate, misses some important details\n"
            "- 1-3: Mostly inaccurate, doesn't match image content\n"
            "- 0: Completely wrong\n\n"
            "Only respond with a single number between 0-10, no other content."
        ),
        description="English prompt template for better multilingual support",
    )

    use_english_prompt: bool = Field(
        default=False,
        description="Whether to use English prompt (recommended for multilingual content)",
    )

    async def _compute_reward(
        self, images: List[ImageContent], texts: List[str], **kwargs
    ) -> float:
        """
        Compute image-text alignment score using Qwen VL API.

        Args:
            images: List of images (uses first image)
            texts: List of text descriptions (uses first text)
            **kwargs: Additional parameters

        Returns:
            Alignment score in range [0, 1]
        """
        # Validate inputs
        if not images:
            logger.warning("No images provided for alignment evaluation")
            return self.fallback_score

        if not texts:
            logger.warning("No text provided for alignment evaluation")
            return self.fallback_score

        # Use first image and first text
        image = images[0]
        text = texts[0]

        # Handle empty text
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return self.fallback_score

        try:
            # Select prompt template
            prompt_template = (
                self.similarity_prompt_template_en
                if self.use_english_prompt
                else self.similarity_prompt_template
            )

            # Compute similarity using VLM API
            score = await self.vlm_api.compute_similarity(
                image=image, text=text, prompt_template=prompt_template
            )

            # Ensure score is in valid range
            score = max(0.0, min(1.0, float(score)))

            logger.debug(
                f"Computed alignment score: {score:.3f} for text length: {len(text)}"
            )

            return score

        except Exception as e:
            logger.error(f"Failed to compute alignment score: {str(e)}")
            return self.fallback_score

    async def get_detailed_stats_async(self) -> dict:
        """
        Get detailed statistics including API usage and cache performance (async version).

        Returns:
            Dictionary with comprehensive statistics
        """
        cost_stats = self.get_cost_stats()

        # Add model-specific information
        stats = {
            "reward_model": self.name,
            "vlm_model": self.vlm_api.model_name,
            "api_base_url": self.vlm_api.base_url,
            **cost_stats,
        }

        # Add cache stats if available
        if self.vlm_api.cache:
            cache_stats = await self.vlm_api.get_cache_stats()
            stats["api_cache_stats"] = cache_stats

        # Add circuit breaker state
        if self.vlm_api.circuit_breaker:
            stats["circuit_breaker"] = self.vlm_api.circuit_breaker.get_state()

        return stats

    def get_detailed_stats(self) -> dict:
        """
        Get detailed statistics (sync wrapper).

        Returns:
            Dictionary with comprehensive statistics
        """
        try:
            # Try to get running loop
            loop = asyncio.get_running_loop()
            # We're in an async context, use create_task
            logger.warning(
                "get_detailed_stats() called in async context. "
                "Consider using get_detailed_stats_async() instead."
            )
            # Return basic stats without cache info
            cost_stats = self.get_cost_stats()
            return {
                "reward_model": self.name,
                "vlm_model": self.vlm_api.model_name,
                "api_base_url": self.vlm_api.base_url,
                **cost_stats,
                "note": "Cache stats unavailable in async context. Use get_detailed_stats_async()",
            }
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self.get_detailed_stats_async())


# Convenience factory functions
def create_qwen_alignment_reward(
    api_key: Optional[str] = None,
    model_name: str = "qwen-vl-plus",
    enable_cache: bool = True,
    cache_ttl: int = 3600,
    **kwargs,
) -> QwenImageTextAlignmentReward:
    """
    Factory function to create Qwen alignment reward with custom configuration.

    Args:
        api_key: DashScope API key (defaults to env var)
        model_name: Qwen model to use ("qwen-vl-plus" or "qwen-vl-max")
        enable_cache: Whether to enable response caching
        cache_ttl: Cache time-to-live in seconds
        **kwargs: Additional parameters for QwenImageTextAlignmentReward

    Returns:
        Configured QwenImageTextAlignmentReward instance

    Examples:
        >>> # Create with defaults
        >>> reward = create_qwen_alignment_reward()
        >>>
        >>> # Create with custom model
        >>> reward = create_qwen_alignment_reward(
        ...     model_name="qwen-vl-max",  # More accurate
        ...     cache_ttl=7200,  # 2 hour cache
        ...     use_english_prompt=True
        ... )
    """
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Set DASHSCOPE_API_KEY environment variable "
            "or pass api_key parameter."
        )

    vlm_api = QwenVLAPI(
        api_key=api_key,
        model_name=model_name,
        enable_cache=enable_cache,
        cache_ttl=cache_ttl,
    )

    return QwenImageTextAlignmentReward(
        vlm_api=vlm_api, enable_cache=enable_cache, cache_ttl=cache_ttl, **kwargs
    )
