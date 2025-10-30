"""
Core model interfaces and implementations.

This module provides base classes and implementations for:
- Large Language Models (LLMs)
- Vision-Language Models (VLMs)
- Message handling and formatting
- API clients and utilities
"""

from rm_gallery.core.model.base import BaseLLM, get_from_dict_or_env
from rm_gallery.core.model.huggingface_llm import HuggingFaceLLM
from rm_gallery.core.model.message import (
    ChatMessage,
    ChatResponse,
    GeneratorChatResponse,
    MessageRole,
)
from rm_gallery.core.model.openai_llm import OpenaiLLM

# Backward compatibility alias
HuggingfaceLLM = HuggingFaceLLM

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
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI, QwenVLDashScopeAPI

# VLM API components
from rm_gallery.core.model.vlm_api_base import BaseVLMAPI, VLMResponse

__all__ = [
    # Base classes
    "BaseLLM",
    "BaseVLMAPI",
    # LLM implementations
    "OpenaiLLM",
    "HuggingfaceLLM",
    # VLM implementations
    "QwenVLAPI",
    "QwenVLDashScopeAPI",
    # Message handling
    "ChatMessage",
    "ChatResponse",
    "GeneratorChatResponse",
    "MessageRole",
    "VLMResponse",
    # API utilities
    "APIError",
    "APIRateLimitError",
    "APIConnectionError",
    "APIAuthenticationError",
    "APITimeoutError",
    "RateLimiter",
    "RetryConfig",
    "TTLCache",
    "CostTracker",
    "CircuitBreaker",
    "retry_with_exponential_backoff",
    # Utilities
    "get_from_dict_or_env",
]
