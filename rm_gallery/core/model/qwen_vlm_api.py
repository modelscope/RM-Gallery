"""
Qwen VL API implementation for multimodal reward modeling.

Supports Qwen-VL-Plus and Qwen-VL-Max models through:
1. OpenAI-compatible API (recommended)
2. DashScope API (阿里云灵积平台)

Both methods provide the same interface for seamless integration.
"""

from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from openai import AsyncOpenAI
from pydantic import Field, model_validator

from rm_gallery.core.data.multimodal_content import ImageContent
from rm_gallery.core.data.multimodal_message import MultimodalChatMessage
from rm_gallery.core.model.api_utils import (
    APIAuthenticationError,
    APIConnectionError,
    APIError,
    APIRateLimitError,
    APITimeoutError,
    CircuitBreaker,
    CostTracker,
    RateLimiter,
    RetryConfig,
    TTLCache,
    retry_with_exponential_backoff,
)
from rm_gallery.core.model.base import get_from_dict_or_env
from rm_gallery.core.model.vlm_api_base import BaseVLMAPI, VLMResponse


class QwenVLAPI(BaseVLMAPI):
    """
    Qwen VL API client using OpenAI-compatible interface.

    Supports both Qwen-VL-Plus (faster, cheaper) and Qwen-VL-Max (more accurate).
    Uses OpenAI SDK for robust error handling and streaming support.

    Attributes:
        model_name: Model to use ("qwen-vl-plus", "qwen-vl-max", "qwen-vl-max-0201")
        api_key: DashScope API key or OpenAI API key
        base_url: API endpoint (defaults to DashScope compatible mode)
        client: AsyncOpenAI client instance
        rate_limiter: Request rate limiter
        cache: Response cache
        cost_tracker: API cost tracker
        circuit_breaker: Circuit breaker for resilience
        enable_cache: Whether to enable response caching

    Examples:
        >>> # Initialize with DashScope API key
        >>> api = QwenVLAPI(
        ...     api_key=os.getenv("DASHSCOPE_API_KEY"),
        ...     model_name="qwen-vl-plus"
        ... )
        >>>
        >>> # Generate response with image
        >>> image = ImageContent(type="url", data="https://example.com/cat.jpg")
        >>> response = await api.generate(
        ...     text="Describe this image",
        ...     images=[image]
        ... )
        >>> print(response.content)
        >>>
        >>> # Compute similarity
        >>> score = await api.compute_similarity(image, "A cute cat")
        >>> print(f"Similarity: {score:.2f}")
    """

    # Model configurations
    model_name: str = Field(default="qwen-vl-plus", description="Qwen VL model name")
    api_key: str = Field(..., description="DashScope or OpenAI API key")
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="API endpoint URL",
    )

    # Client and utilities
    client: Optional[AsyncOpenAI] = Field(default=None, exclude=True)
    rate_limiter: Optional[RateLimiter] = Field(default=None, exclude=True)
    cache: Optional[TTLCache] = Field(default=None, exclude=True)
    cost_tracker: CostTracker = Field(default_factory=CostTracker, exclude=True)
    circuit_breaker: Optional[CircuitBreaker] = Field(default=None, exclude=True)

    # Configuration
    enable_cache: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_requests_per_minute: int = Field(default=60, description="Rate limit")
    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def validate_client(cls, data: Dict) -> Dict:
        """
        Initialize and validate API client.

        Args:
            data: Configuration dictionary

        Returns:
            Updated configuration with initialized client
        """
        # Get API key from data or environment
        api_key = data.get("api_key")
        if not api_key:
            api_key = get_from_dict_or_env(
                data=data,
                key="dashscope_api_key",
                default=get_from_dict_or_env(
                    data=data, key="openai_api_key", default=None
                ),
            )

        if not api_key:
            raise ValueError(
                "API key not found. Please set DASHSCOPE_API_KEY or OPENAI_API_KEY "
                "environment variable, or pass api_key parameter."
            )

        data["api_key"] = api_key

        # Get base URL
        if "base_url" not in data or not data["base_url"]:
            data["base_url"] = get_from_dict_or_env(
                data=data,
                key="base_url",
                default="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

        return data

    def model_post_init(self, __context: Any) -> None:
        """Initialize client and utilities after model creation."""
        # Initialize AsyncOpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,  # We handle retries ourselves
        )

        # Sync max_retries to retry_config if specified
        if self.max_retries != 3:  # 3 is the default from BaseVLMAPI
            self.retry_config.max_attempts = self.max_retries

        # Initialize utilities
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=self.max_requests_per_minute, max_concurrent=10
        )

        if self.enable_cache:
            self.cache = TTLCache(max_size=10000, ttl=self.cache_ttl)

        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5, recovery_timeout=60.0
        )

        # Set cost per 1k tokens based on model
        if "plus" in self.model_name.lower():
            self.cost_tracker.cost_per_1k_tokens = 0.008  # Qwen-VL-Plus: ¥0.008/千tokens
        else:
            self.cost_tracker.cost_per_1k_tokens = 0.02  # Qwen-VL-Max: ¥0.02/千tokens

        logger.info(
            f"Initialized QwenVLAPI: model={self.model_name}, "
            f"base_url={self.base_url}, cache_enabled={self.enable_cache}, "
            f"max_retries={self.retry_config.max_attempts}"
        )

    def format_messages(
        self,
        text: Optional[str] = None,
        images: Optional[List[ImageContent]] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[MultimodalChatMessage]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format input into OpenAI-compatible message format.

        Args:
            text: Text content
            images: List of images
            system_prompt: System instruction
            history: Conversation history

        Returns:
            List of formatted message dictionaries
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history if provided
        if history:
            for msg in history:
                messages.append(msg.to_api_format(api_type="openai"))

        # Build current user message content
        content_parts = []

        # Add text
        if text:
            content_parts.append({"type": "text", "text": text})

        # Add images
        if images:
            for image in images:
                content_parts.append(image.to_api_format(api_type="openai"))

        # Add user message
        if content_parts:
            messages.append(
                {
                    "role": "user",
                    "content": content_parts
                    if len(content_parts) > 1
                    else (
                        content_parts[0]["text"]
                        if content_parts[0]["type"] == "text"
                        else content_parts
                    ),
                }
            )

        return messages

    async def call_api(self, messages: List[Dict[str, Any]], **kwargs) -> VLMResponse:
        """
        Call Qwen VL API with formatted messages.

        Includes rate limiting, caching, retry logic, and circuit breaker.

        Args:
            messages: Formatted message list
            **kwargs: Additional API parameters

        Returns:
            VLMResponse with generated content

        Raises:
            APIError: If API call fails after retries
        """
        # Generate cache key
        cache_key = None
        if self.enable_cache and self.cache:
            import json

            cache_key = self.get_cache_key(text=json.dumps(messages, sort_keys=True))

            # Check cache
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                self.cost_tracker.track_request(tokens=0, cached=True)
                return cached_response

        # Prepare API call parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Remove None values
        api_params = {k: v for k, v in api_params.items() if v is not None}

        # Define API call function
        async def _api_call():
            async with self.rate_limiter:
                try:
                    response = await self.client.chat.completions.create(**api_params)
                    return response

                except Exception as e:
                    # Categorize exception
                    error_msg = str(e).lower()

                    if "rate" in error_msg or "429" in error_msg:
                        raise APIRateLimitError(f"Rate limit exceeded: {e}")
                    elif "auth" in error_msg or "401" in error_msg:
                        raise APIAuthenticationError(f"Authentication failed: {e}")
                    elif "timeout" in error_msg:
                        raise APITimeoutError(f"Request timeout: {e}")
                    elif "connect" in error_msg or "network" in error_msg:
                        raise APIConnectionError(f"Connection failed: {e}")
                    else:
                        raise APIError(f"API call failed: {e}")

        # Execute with retry and circuit breaker
        try:
            response = await self.circuit_breaker.call(
                lambda: retry_with_exponential_backoff(
                    _api_call, config=self.retry_config
                )
            )

        except Exception as e:
            logger.error(f"API call failed after retries: {e}")
            # Track failed request
            self.cost_tracker.track_request(tokens=0, cached=False)
            raise

        # Parse response
        vlm_response = self._parse_response(response)

        # Track cost
        if vlm_response.token_usage:
            # Use total_tokens if available, otherwise sum prompt + completion
            # Note: total_tokens already includes both, so don't double count
            total_tokens = vlm_response.token_usage.get(
                "total_tokens",
                vlm_response.token_usage.get("prompt_tokens", 0)
                + vlm_response.token_usage.get("completion_tokens", 0),
            )
            self.cost_tracker.track_request(tokens=total_tokens, cached=False)

        # Cache response
        if self.enable_cache and self.cache and cache_key:
            await self.cache.set(cache_key, vlm_response)

        return vlm_response

    def _parse_response(self, response: Any) -> VLMResponse:
        """
        Parse API response into VLMResponse.

        Args:
            response: Raw API response

        Returns:
            Parsed VLMResponse
        """
        try:
            message = response.choices[0].message
            content = message.content or ""

            # Handle case where content is a list (multimodal response format)
            # OpenAI/DashScope may return: [{"type": "text", "text": "..."}, ...]
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        # Extract text from dict items
                        if "text" in item:
                            text_parts.append(item["text"])
                        elif "content" in item:
                            text_parts.append(str(item["content"]))
                    else:
                        # Handle other types by converting to string
                        text_parts.append(str(item))
                content = "".join(text_parts)

            # Ensure content is a string
            content = str(content) if content else ""

            # Extract token usage
            token_usage = None
            if hasattr(response, "usage") and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            # Extract metadata
            metadata = {
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            }

            if hasattr(response, "id"):
                metadata["id"] = response.id

            return VLMResponse(
                content=content,
                token_usage=token_usage,
                raw_response=response.model_dump()
                if hasattr(response, "model_dump")
                else None,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to parse API response: {e}")
            raise APIError(f"Response parsing failed: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if self.cache:
            return await self.cache.get_stats()
        return {"enabled": False}

    def get_cost_stats(self) -> Dict[str, Any]:
        """
        Get cost statistics.

        Returns:
            Dictionary with cost and usage stats
        """
        return self.cost_tracker.get_stats()

    def get_circuit_breaker_state(self) -> Dict[str, Any]:
        """
        Get circuit breaker state.

        Returns:
            Dictionary with circuit breaker state
        """
        if self.circuit_breaker:
            return self.circuit_breaker.get_state()
        return {"enabled": False}

    async def health_check(self) -> bool:
        """
        Perform health check on API.

        Returns:
            True if API is healthy
        """
        try:
            response = await self.generate(text="Hello", temperature=0.1, max_tokens=5)
            return bool(response.content)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


class QwenVLDashScopeAPI(BaseVLMAPI):
    """
    Qwen VL API client using native DashScope API.

    Direct integration with阿里云灵积平台 DashScope API.
    Use this if OpenAI-compatible mode is not available.

    Attributes:
        model_name: Model to use
        api_key: DashScope API key
        base_url: DashScope API endpoint

    Note:
        This is an alternative implementation using httpx for direct API calls.
        The OpenAI-compatible QwenVLAPI is recommended for most use cases.
    """

    model_name: str = Field(default="qwen-vl-plus")
    api_key: str = Field(...)
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    )

    async_client: Optional[httpx.AsyncClient] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, data: Dict) -> Dict:
        """Validate configuration."""
        if "api_key" not in data:
            data["api_key"] = get_from_dict_or_env(
                data=data, key="dashscope_api_key", default=None
            )

        if not data.get("api_key"):
            raise ValueError("DASHSCOPE_API_KEY is required")

        return data

    def model_post_init(self, __context: Any) -> None:
        """Initialize HTTP client."""
        self.async_client = httpx.AsyncClient(timeout=self.timeout)

    def format_messages(
        self,
        text: Optional[str] = None,
        images: Optional[List[ImageContent]] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[MultimodalChatMessage]] = None,
    ) -> List[Dict[str, Any]]:
        """Format messages for DashScope API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": [{"text": system_prompt}]})

        if history:
            for msg in history:
                messages.append(msg.to_api_format(api_type="qwen"))

        # Build current message
        content = []
        if text:
            content.append({"text": text})
        if images:
            for img in images:
                content.append(img.to_api_format(api_type="qwen"))

        if content:
            messages.append({"role": "user", "content": content})

        return messages

    async def call_api(self, messages: List[Dict[str, Any]], **kwargs) -> VLMResponse:
        """Call DashScope API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "input": {"messages": messages},
            "parameters": {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        try:
            response = await self.async_client.post(
                self.base_url, headers=headers, json=payload
            )
            response.raise_for_status()

            result = response.json()

            # Parse DashScope response format
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]

                # Extract text from content array
                if isinstance(content, list):
                    content = " ".join(
                        [item.get("text", "") for item in content if "text" in item]
                    )

                token_usage = result.get("usage", {})

                return VLMResponse(
                    content=content,
                    token_usage=token_usage if token_usage else None,
                    raw_response=result,
                    metadata={"request_id": result.get("request_id")},
                )
            else:
                raise APIError(f"Unexpected response format: {result}")

        except httpx.HTTPStatusError as e:
            raise APIError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise APIError(f"API call failed: {e}")
