"""
VLM (Vision-Language Model) Reward Base Classes.

This module provides base classes for reward models that use VLM APIs to evaluate
multimodal content (text + images). Designed for API-based inference with built-in
caching, cost tracking, and error handling.

Key Features:
- API-based VLM evaluation (Qwen VL, GPT-4V, etc.)
- Response caching to reduce costs
- Cost tracking and statistics
- Async/concurrent evaluation support
- Graceful error handling and fallbacks
"""

import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import Field

from rm_gallery.core.data.multimodal_content import ImageContent, MultimodalContent
from rm_gallery.core.data.multimodal_message import MultimodalChatMessage
from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.api_utils import CostTracker, TTLCache
from rm_gallery.core.model.vlm_api_base import BaseVLMAPI
from rm_gallery.core.reward.base import (
    BaseListWiseReward,
    BasePairWiseReward,
    BasePointWiseReward,
)
from rm_gallery.core.reward.schema import (
    RewardDimensionWithRank,
    RewardDimensionWithScore,
    RewardResult,
)


class BaseVLMReward:
    """
    Base class for VLM-based reward models.

    Provides common functionality for reward models that use Vision-Language Model APIs
    to evaluate multimodal content. Includes caching, cost tracking, and robust error handling.

    This base class should not be used directly. Instead, use one of the concrete subclasses:
    - BasePointWiseVLMReward: For single response evaluation
    - BaseListWiseVLMReward: For ranking multiple responses
    - BasePairWiseVLMReward: For pairwise comparison

    Attributes:
        vlm_api: VLM API client instance (e.g., QwenVLAPI, GPT4VisionAPI)
        enable_cache: Whether to enable response caching
        cache_ttl: Cache time-to-live in seconds
        cache_max_size: Maximum cache size
        enable_cost_tracking: Whether to track API costs
        fallback_score: Default score when evaluation fails
        prompt_template: Template for generating evaluation prompts

    Example:
        >>> class MyVLMReward(BasePointWiseVLMReward):
        ...     async def _compute_reward(self, images, texts, **kwargs):
        ...         # Custom reward computation logic (async)
        ...         score = await self.vlm_api.compute_similarity(images[0], texts[0])
        ...         return score
        ...
        >>> reward = MyVLMReward(vlm_api=QwenVLAPI(api_key="xxx"))
        >>> result = reward.evaluate(sample)
    """

    # VLM API client
    vlm_api: BaseVLMAPI = Field(..., description="VLM API client instance")

    # Cache configuration
    enable_cache: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=10000, description="Maximum cache size")

    # Cost tracking
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")

    # Error handling
    fallback_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Fallback score on errors"
    )
    max_api_retries: int = Field(default=3, description="Maximum API retry attempts")

    # Prompt configuration
    prompt_template: Optional[str] = Field(
        default=None, description="Custom prompt template"
    )

    # Internal state (not serialized)
    _cache: Optional[TTLCache] = None
    _cost_tracker: Optional[CostTracker] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **data):
        """Initialize VLM reward with cache and cost tracker."""
        super().__init__(**data)

        # Initialize cache
        if self.enable_cache:
            self._cache = TTLCache(max_size=self.cache_max_size, ttl=self.cache_ttl)
            logger.info(
                f"Initialized cache with TTL={self.cache_ttl}s, max_size={self.cache_max_size}"
            )

        # Initialize cost tracker
        if self.enable_cost_tracking:
            self._cost_tracker = CostTracker()
            logger.info("Initialized cost tracker")

    @abstractmethod
    async def _compute_reward(
        self, images: List[ImageContent], texts: List[str], **kwargs
    ) -> float:
        """
        Compute reward score from images and texts (async).

        This is the core logic that subclasses must implement.
        Should call VLM API asynchronously and return a normalized score in [0, 1].

        Args:
            images: List of images to evaluate
            texts: List of text content
            **kwargs: Additional parameters

        Returns:
            Reward score in range [0, 1]

        Example:
            async def _compute_reward(self, images, texts, **kwargs):
                score = await self.vlm_api.compute_similarity(images[0], texts[0])
                return float(score)
        """
        pass

    def _extract_multimodal_content(
        self, sample: DataSample
    ) -> Tuple[List[str], List[ImageContent]]:
        """
        Extract text and image content from DataSample.

        Handles various content types:
        - Plain text strings
        - MultimodalContent objects
        - MultimodalChatMessage objects

        Args:
            sample: Input data sample

        Returns:
            Tuple of (texts, images) lists
        """
        texts = []
        images = []

        # Process input messages
        for msg in sample.input:
            if isinstance(msg, MultimodalChatMessage):
                # Extract text
                text_content = msg.get_text()
                if text_content and text_content.strip():
                    texts.append(text_content)

                # Extract images
                msg_images = msg.get_images()
                if msg_images:
                    images.extend(msg_images)

            else:
                # Fallback for regular ChatMessage
                if hasattr(msg, "content"):
                    content = msg.content
                    if isinstance(content, str) and content.strip():
                        texts.append(content)
                    elif isinstance(content, MultimodalContent):
                        if content.text and content.text.strip():
                            texts.append(content.text)
                        if content.images:
                            images.extend(content.images)

        # Also check output for multimodal content (in case needed)
        if sample.output and len(sample.output) > 0:
            output = sample.output[0]
            if hasattr(output.answer, "content"):
                content = output.answer.content
                if isinstance(content, MultimodalContent):
                    # Usually we don't extract images from answers, but text might be useful
                    if content.text and content.text.strip():
                        texts.append(content.text)

        logger.debug(f"Extracted {len(texts)} text(s) and {len(images)} image(s)")
        return texts, images

    def _make_cache_key(
        self, images: List[ImageContent], texts: List[str], **kwargs
    ) -> str:
        """
        Generate cache key for request deduplication.

        Args:
            images: List of images
            texts: List of texts
            **kwargs: Additional parameters

        Returns:
            Cache key string
        """
        key_parts = []

        # Add model name
        key_parts.append(self.vlm_api.model_name)

        # Add text content
        for text in texts:
            key_parts.append(text)

        # Add image keys
        for img in images:
            key_parts.append(img.get_cache_key())

        # Add reward name
        key_parts.append(self.name)

        # Combine into cache key
        import hashlib

        key_string = "|".join(str(p) for p in key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def _call_api_with_cache(
        self, cache_key: str, api_call_func: callable, estimate_tokens: int = 100
    ) -> float:
        """
        Call API with caching support.

        Args:
            cache_key: Cache key for this request
            api_call_func: Async function that makes the API call
            estimate_tokens: Estimated token count for cost tracking

        Returns:
            Reward score
        """
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key[:16]}...")

                # Track cache hit
                if self._cost_tracker:
                    self._cost_tracker.track_request(tokens=0, cached=True)

                return cached_result

        # Cache miss - call API
        logger.debug(f"Cache miss for key: {cache_key[:16]}...")

        try:
            result = await api_call_func()

            # Store in cache
            if self._cache:
                await self._cache.set(cache_key, result)

            # Track cost
            if self._cost_tracker:
                self._cost_tracker.track_request(tokens=estimate_tokens, cached=False)

            return result

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")

            # Track failed request
            if self._cost_tracker:
                self._cost_tracker.track_request(tokens=0, cached=False)

            # Return fallback score
            return self.fallback_score

    def _handle_missing_image(
        self, sample: DataSample
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Handle case when no images are present.

        Args:
            sample: Input sample

        Returns:
            Default reward result
        """
        logger.warning(f"No images found in sample {sample.unique_id}")

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=self.fallback_score,
                    reason="No image content found in sample",
                )
            ],
        )

    def _handle_api_error(
        self, sample: DataSample, error: Exception
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Handle API errors gracefully.

        Args:
            sample: Input sample
            error: Exception that occurred

        Returns:
            Fallback reward result
        """
        logger.error(f"VLM API error for sample {sample.unique_id}: {str(error)}")

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=self.fallback_score,
                    reason=f"API error: {str(error)[:100]}",
                )
            ],
        )

    def _estimate_tokens(self, images: List[ImageContent], texts: List[str]) -> int:
        """
        Estimate token count for API call.

        Args:
            images: List of images
            texts: List of texts

        Returns:
            Estimated token count
        """
        # Text tokens (rough estimate: 1 token ≈ 4 chars)
        text_tokens = sum(len(text) // 4 for text in texts)

        # Image tokens (varies by model, rough estimate)
        # GPT-4V: ~85-170 tokens per image depending on detail
        # Qwen-VL: ~256-1024 tokens per image
        image_tokens = len(images) * 256  # Conservative estimate

        # Add prompt overhead
        prompt_tokens = 50

        return text_tokens + image_tokens + prompt_tokens

    def get_cost_stats(self) -> Dict[str, Any]:
        """
        Get cost tracking statistics.

        Returns:
            Dictionary with cost and cache statistics
        """
        stats = {
            "reward_name": self.name,
            "cache_enabled": self.enable_cache,
            "cost_tracking_enabled": self.enable_cost_tracking,
        }

        if self._cost_tracker:
            stats.update(self._cost_tracker.get_stats())

        if self._cache:
            # Get cache stats synchronously by checking if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._cache.get_stats())
                    cache_stats = future.result(timeout=5)
                    stats["cache_stats"] = cache_stats
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                cache_stats = asyncio.run(self._cache.get_stats())
                stats["cache_stats"] = cache_stats
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")
                stats["cache_stats"] = {"error": str(e)}

        return stats

    def reset_stats(self):
        """Reset cost and cache statistics."""
        if self._cost_tracker:
            self._cost_tracker.reset()
            logger.info("Reset cost tracker statistics")

        if self._cache:
            # Clear cache safely
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._cache.clear())
                    future.result(timeout=5)
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                asyncio.run(self._cache.clear())
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
            logger.info("Cleared cache")


class BasePointWiseVLMReward(BaseVLMReward, BasePointWiseReward):
    """
    Point-wise VLM reward for evaluating individual responses.

    Evaluates each response independently using VLM API.
    Suitable for tasks like:
    - Image-text alignment scoring
    - Quality assessment
    - Compliance checking

    Example:
        >>> class ImageTextAlignmentReward(BasePointWiseVLMReward):
        ...     async def _compute_reward(self, images, texts, **kwargs):
        ...         score = await self.vlm_api.compute_similarity(images[0], texts[0])
        ...         return score
    """

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Evaluate a single response using VLM API.

        Args:
            sample: Input data sample
            **kwargs: Additional parameters

        Returns:
            Reward result with score
        """
        # Extract multimodal content
        texts, images = self._extract_multimodal_content(sample)

        # Handle missing images
        if not images:
            return self._handle_missing_image(sample)

        # Get answer text
        if sample.output and len(sample.output) > 0:
            answer = sample.output[0].answer.content
            if isinstance(answer, str):
                answer_text = answer
            elif isinstance(answer, MultimodalContent):
                answer_text = answer.text or ""
            else:
                answer_text = str(answer)

            # Add answer to texts if needed
            if answer_text.strip():
                texts = [answer_text] + texts

        try:
            # Compute reward - handle async safely
            try:
                # Try to get running loop
                loop = asyncio.get_running_loop()
                # We're in an async context, create task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._async_compute_reward(images, texts, **kwargs)
                    )
                    score = future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                score = asyncio.run(self._async_compute_reward(images, texts, **kwargs))

            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithScore(
                        name=self.name,
                        score=float(score),
                        reason=f"VLM-based evaluation score: {score:.3f}",
                    )
                ],
            )

        except Exception as e:
            return self._handle_api_error(sample, e)

    async def _async_compute_reward(
        self, images: List[ImageContent], texts: List[str], **kwargs
    ) -> float:
        """
        Async wrapper for _compute_reward with caching.

        Args:
            images: List of images
            texts: List of texts
            **kwargs: Additional parameters

        Returns:
            Reward score
        """
        # Generate cache key
        cache_key = self._make_cache_key(images, texts, **kwargs)

        # Estimate tokens
        token_estimate = self._estimate_tokens(images, texts)

        # Define API call function - now directly await the async method
        async def api_call():
            return await self._compute_reward(images, texts, **kwargs)

        # Call with caching
        score = await self._call_api_with_cache(
            cache_key=cache_key, api_call_func=api_call, estimate_tokens=token_estimate
        )

        return score


class BaseListWiseVLMReward(BaseVLMReward, BaseListWiseReward):
    """
    List-wise VLM reward for ranking multiple responses.

    Evaluates and ranks multiple candidate responses using VLM API.
    Suitable for:
    - Best-of-N selection
    - Response ranking
    - Comparative evaluation

    Example:
        >>> class MultimodalRankingReward(BaseListWiseVLMReward):
        ...     async def _compute_reward(self, images, texts, **kwargs):
        ...         # Compute score for one candidate
        ...         score = await self.vlm_api.compute_similarity(images[0], texts[0])
        ...         return score
    """

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithRank]:
        """
        Evaluate and rank multiple responses.

        Args:
            sample: Input data sample with multiple outputs
            **kwargs: Additional parameters

        Returns:
            Reward result with rankings
        """
        # Extract multimodal content from input
        input_texts, images = self._extract_multimodal_content(sample)

        # Handle missing images
        if not images:
            n = len(sample.output)
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithRank(
                        name=self.name,
                        rank=[1.0 / n] * n,  # Equal ranking
                        reason="No image content found",
                    )
                ],
            )

        # Extract all candidate answers
        candidate_texts = []
        for output in sample.output:
            answer = output.answer.content
            if isinstance(answer, str):
                answer_text = answer
            elif isinstance(answer, MultimodalContent):
                answer_text = answer.text or ""
            else:
                answer_text = str(answer)
            candidate_texts.append(answer_text)

        try:
            # Compute scores for all candidates - handle async safely
            scores = []
            for candidate_text in candidate_texts:
                texts = [candidate_text] + input_texts
                try:
                    # Try to get running loop
                    loop = asyncio.get_running_loop()
                    # We're in an async context, use executor
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._async_compute_reward(images, texts, **kwargs),
                        )
                        score = future.result()
                except RuntimeError:
                    # No running loop, safe to use asyncio.run
                    score = asyncio.run(
                        self._async_compute_reward(images, texts, **kwargs)
                    )
                scores.append(score)

            # Convert scores to ranks
            ranks = self._scores_to_ranks(scores)

            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithRank(
                        name=self.name,
                        rank=ranks,
                        reason=f"VLM-based ranking scores: {[f'{s:.3f}' for s in scores]}",
                    )
                ],
            )

        except Exception as e:
            # Return equal ranking on error
            n = len(sample.output)
            logger.error(f"List-wise evaluation error: {str(e)}")
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithRank(
                        name=self.name,
                        rank=[1.0 / n] * n,
                        reason=f"Error during ranking: {str(e)[:100]}",
                    )
                ],
            )

    async def _async_compute_reward(
        self, images: List[ImageContent], texts: List[str], **kwargs
    ) -> float:
        """
        Async wrapper for _compute_reward with caching.

        Args:
            images: List of images
            texts: List of texts
            **kwargs: Additional parameters

        Returns:
            Reward score for one candidate
        """
        # Generate cache key
        cache_key = self._make_cache_key(images, texts, **kwargs)

        # Estimate tokens
        token_estimate = self._estimate_tokens(images, texts)

        # Define API call function - now directly await the async method
        async def api_call():
            return await self._compute_reward(images, texts, **kwargs)

        # Call with caching
        score = await self._call_api_with_cache(
            cache_key=cache_key, api_call_func=api_call, estimate_tokens=token_estimate
        )

        return score

    def _scores_to_ranks(self, scores: List[float]) -> List[float]:
        """
        Convert scores to normalized ranks.

        Higher scores get higher ranks (closer to 1.0).

        Args:
            scores: List of scores

        Returns:
            List of normalized ranks in [0, 1]
        """
        if len(scores) <= 1:
            return [1.0] * len(scores)

        # Get sorted indices (highest score first)
        import numpy as np

        sorted_indices = np.argsort(scores)[::-1]

        # Assign ranks (best = 1.0, worst approaches 0)
        ranks = [0.0] * len(scores)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = 1.0 - (rank / (len(scores) - 1))

        return ranks


class BasePairWiseVLMReward(BaseVLMReward, BasePairWiseReward):
    """
    Pair-wise VLM reward for comparing two responses.

    Compares pairs of responses using VLM API.
    Suitable for:
    - Preference learning
    - Pairwise ranking
    - RLHF data collection

    Example:
        >>> class PairwisePreferenceReward(BasePairWiseVLMReward):
        ...     async def _compute_reward(self, images, texts, **kwargs):
        ...         # texts[0] is response A, texts[1] is response B
        ...         score_a = await self.vlm_api.compute_similarity(images[0], texts[0])
        ...         score_b = await self.vlm_api.compute_similarity(images[0], texts[1])
        ...         # Return preference score for A
        ...         return 1.0 if score_a > score_b else 0.0
    """

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithRank]:
        """
        Evaluate pairwise preference between two responses.

        This method is called by BasePairWiseReward._parallel which handles
        the pair iteration logic.

        Args:
            sample: Input data sample with exactly 2 outputs to compare
            **kwargs: Additional parameters

        Returns:
            Reward result with pairwise ranking
        """
        # Extract multimodal content from input
        input_texts, images = self._extract_multimodal_content(sample)

        # Handle missing images
        if not images:
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithRank(
                        name=self.name,
                        rank=[0.5, 0.5],  # Equal preference
                        reason="No image content found",
                    )
                ],
            )

        # Extract the two candidate answers
        if len(sample.output) != 2:
            logger.error(
                f"Pairwise reward expects exactly 2 outputs, got {len(sample.output)}"
            )
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithRank(
                        name=self.name,
                        rank=[0.5] * len(sample.output),
                        reason="Invalid number of outputs for pairwise comparison",
                    )
                ],
            )

        candidate_texts = []
        for output in sample.output:
            answer = output.answer.content
            if isinstance(answer, str):
                answer_text = answer
            elif isinstance(answer, MultimodalContent):
                answer_text = answer.text or ""
            else:
                answer_text = str(answer)
            candidate_texts.append(answer_text)

        try:
            # Combine input and candidate texts
            texts = candidate_texts + input_texts

            # Compute preference score - handle async safely
            try:
                # Try to get running loop
                loop = asyncio.get_running_loop()
                # We're in an async context, use executor
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._async_compute_reward(images, texts, **kwargs)
                    )
                    preference_score = future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                preference_score = asyncio.run(
                    self._async_compute_reward(images, texts, **kwargs)
                )

            # Convert preference score to ranks
            # preference_score = 1.0 means A is preferred
            # preference_score = 0.0 means B is preferred
            # preference_score = 0.5 means equal
            ranks = [preference_score, 1.0 - preference_score]

            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithRank(
                        name=self.name,
                        rank=ranks,
                        reason=f"Pairwise preference score: {preference_score:.3f}",
                    )
                ],
            )

        except Exception as e:
            logger.error(f"Pairwise evaluation error: {str(e)}")
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithRank(
                        name=self.name,
                        rank=[0.5, 0.5],
                        reason=f"Error during pairwise comparison: {str(e)[:100]}",
                    )
                ],
            )

    async def _async_compute_reward(
        self, images: List[ImageContent], texts: List[str], **kwargs
    ) -> float:
        """
        Async wrapper for _compute_reward with caching.

        Args:
            images: List of images
            texts: List of texts (texts[0] is response A, texts[1] is response B, rest are input)
            **kwargs: Additional parameters

        Returns:
            Preference score: 1.0 if A preferred, 0.0 if B preferred, 0.5 if equal
        """
        # Generate cache key
        cache_key = self._make_cache_key(images, texts, **kwargs)

        # Estimate tokens
        token_estimate = self._estimate_tokens(images, texts)

        # Define API call function - now directly await the async method
        async def api_call():
            return await self._compute_reward(images, texts, **kwargs)

        # Call with caching
        score = await self._call_api_with_cache(
            cache_key=cache_key, api_call_func=api_call, estimate_tokens=token_estimate
        )

        return score
