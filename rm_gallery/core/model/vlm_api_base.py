"""
Base classes for Vision-Language Model API integrations.

This module provides abstract base classes for integrating VLM APIs (Qwen VL, GPT-4V, etc.)
into the RM-Gallery reward modeling framework. Designed for API-based inference only.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.data.multimodal_content import ImageContent
from rm_gallery.core.data.multimodal_message import MultimodalChatMessage


class VLMResponse(BaseModel):
    """
    Response from VLM API calls.

    Attributes:
        content: Generated text content
        score: Optional similarity or quality score (0-1)
        raw_response: Raw API response for debugging
        token_usage: Token consumption statistics
        metadata: Additional response metadata
    """

    content: str = Field(..., description="Generated text content from VLM")
    score: Optional[float] = Field(
        default=None, description="Optional similarity/quality score"
    )
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw API response"
    )
    token_usage: Optional[Dict[str, int]] = Field(
        default=None, description="Token usage statistics"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "examples": [
                {
                    "content": "这张图片展示了一只猫坐在沙发上",
                    "score": 0.85,
                    "token_usage": {"prompt_tokens": 100, "completion_tokens": 20},
                }
            ]
        }


class BaseVLMAPI(BaseModel, ABC):
    """
    Abstract base class for Vision-Language Model API clients.

    Provides common interface and utilities for VLM API integrations.
    Subclasses should implement specific API calling logic.

    Attributes:
        api_key: API authentication key
        model_name: Name/ID of the VLM model to use
        base_url: API endpoint base URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens to generate
    """

    api_key: str = Field(..., description="API authentication key")
    model_name: str = Field(..., description="VLM model name/ID")
    base_url: Optional[str] = Field(default=None, description="API endpoint base URL")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=2048, gt=0, description="Maximum tokens to generate"
    )

    # Internal state
    _request_count: int = 0
    _total_tokens: int = 0

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        extra = "allow"

    @abstractmethod
    async def call_api(self, messages: List[Dict[str, Any]], **kwargs) -> VLMResponse:
        """
        Call the VLM API with formatted messages.

        Args:
            messages: List of message dictionaries in API-specific format
            **kwargs: Additional API-specific parameters

        Returns:
            VLMResponse containing generated content and metadata

        Raises:
            Exception: If API call fails after retries
        """
        pass

    @abstractmethod
    def format_messages(
        self,
        text: Optional[str] = None,
        images: Optional[List[ImageContent]] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[MultimodalChatMessage]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format input data into API-specific message format.

        Args:
            text: Text content
            images: List of images
            system_prompt: System instruction
            history: Conversation history

        Returns:
            List of formatted message dictionaries
        """
        pass

    async def generate(
        self,
        text: Optional[str] = None,
        images: Optional[List[ImageContent]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> VLMResponse:
        """
        Generate response from text and/or images.

        Args:
            text: Input text
            images: Input images
            system_prompt: System instruction
            **kwargs: Additional parameters

        Returns:
            VLMResponse with generated content
        """
        messages = self.format_messages(
            text=text, images=images, system_prompt=system_prompt
        )

        response = await self.call_api(messages, **kwargs)

        # Track usage
        self._request_count += 1
        if response.token_usage:
            # Use total_tokens to avoid double counting
            # (total_tokens already includes prompt_tokens + completion_tokens)
            total_tokens = response.token_usage.get(
                "total_tokens",
                response.token_usage.get("prompt_tokens", 0)
                + response.token_usage.get("completion_tokens", 0),
            )
            self._total_tokens += total_tokens

        return response

    async def compute_similarity(
        self, image: ImageContent, text: str, prompt_template: Optional[str] = None
    ) -> float:
        """
        Compute similarity score between image and text.

        Uses the VLM to evaluate how well the text matches the image content.

        Args:
            image: Image to evaluate
            text: Text description to match
            prompt_template: Optional custom prompt template

        Returns:
            Similarity score in range [0, 1]
        """
        if prompt_template is None:
            prompt_template = "请评估图片与以下描述的匹配程度（0-10分）：{text}\n" "只回答一个数字，不要有其他内容。"

        prompt = prompt_template.format(text=text)

        response = await self.generate(
            text=prompt,
            images=[image],
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=10,
        )

        # Parse score from response
        score = self._parse_score(response.content)
        return score

    async def evaluate_quality(
        self, image: ImageContent, text: str, criteria: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate quality of text response given an image.

        Args:
            image: Reference image
            text: Text response to evaluate
            criteria: Optional evaluation criteria

        Returns:
            Quality score in range [0, 1]
        """
        if criteria is None:
            criteria = ["准确性", "详细性", "相关性", "有用性"]

        criteria_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(criteria)])

        prompt = f"""基于图片内容，评估以下回答的质量（0-10分）：

回答：{text}

评估标准：
{criteria_str}

只回答一个0-10的数字。"""

        response = await self.generate(
            text=prompt, images=[image], temperature=0.1, max_tokens=10
        )

        score = self._parse_score(response.content)
        return score

    def _parse_score(self, content: str) -> float:
        """
        Parse numerical score from model response.

        Args:
            content: Model response text

        Returns:
            Score normalized to [0, 1]
        """
        try:
            # Extract first number from response
            import re

            numbers = re.findall(r"\d+\.?\d*", content)
            if not numbers:
                logger.warning(f"No number found in response: {content}")
                return 0.5  # Default to neutral score

            score = float(numbers[0])

            # Normalize to 0-1 range
            if score <= 1.0:
                return max(0.0, min(1.0, score))
            elif score <= 10.0:
                return score / 10.0
            elif score <= 100.0:
                return score / 100.0
            else:
                logger.warning(f"Unexpected score range: {score}")
                return 0.5

        except Exception as e:
            logger.error(f"Failed to parse score from '{content}': {e}")
            return 0.5

    def get_cache_key(
        self, text: Optional[str] = None, images: Optional[List[ImageContent]] = None
    ) -> str:
        """
        Generate cache key for request deduplication.

        Args:
            text: Text content
            images: Image content

        Returns:
            SHA256 hash as cache key
        """
        key_parts = []

        if text:
            key_parts.append(text)

        if images:
            for img in images:
                key_parts.append(img.get_cache_key())

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.

        Returns:
            Dictionary with request count, token usage, etc.
        """
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "model_name": self.model_name,
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self._request_count = 0
        self._total_tokens = 0
