"""
Utility functions and classes for API operations.

Provides error handling, retry logic, rate limiting, and caching mechanisms
for robust VLM API integrations.
"""

import asyncio
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIError(Exception):
    """Base exception for API errors."""

    pass


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    pass


class APIConnectionError(APIError):
    """Raised when API connection fails."""

    pass


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""

    pass


class APITimeoutError(APIError):
    """Raised when API request times out."""

    pass


class RateLimiter:
    """
    Rate limiter to control API request frequency.

    Implements token bucket algorithm for smooth rate limiting.

    Attributes:
        max_requests_per_minute: Maximum requests allowed per minute
        max_concurrent: Maximum concurrent requests
    """

    def __init__(self, max_requests_per_minute: int = 60, max_concurrent: int = 10):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_minute: Maximum requests per minute
            max_concurrent: Maximum concurrent requests
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_concurrent = max_concurrent

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.min_interval = 60.0 / max_requests_per_minute
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire permission to make an API request.

        Blocks until rate limit allows the request.
        """
        # Acquire semaphore for concurrency control
        await self.semaphore.acquire()

        # Enforce minimum interval between requests
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()

    def release(self):
        """Release the semaphore."""
        self.semaphore.release()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()


class RetryConfig(BaseModel):
    """
    Configuration for retry logic.

    Attributes:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
    """

    max_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts")
    initial_delay: float = Field(default=1.0, gt=0, description="Initial retry delay")
    max_delay: float = Field(default=60.0, gt=0, description="Maximum retry delay")
    exponential_base: float = Field(
        default=2.0, gt=1, description="Exponential backoff base"
    )
    jitter: bool = Field(default=True, description="Add random jitter to delays")


async def retry_with_exponential_backoff(
    func: Callable,
    config: Optional[RetryConfig] = None,
    retryable_exceptions: tuple = (APIError, asyncio.TimeoutError, ConnectionError),
) -> Any:
    """
    Execute function with exponential backoff retry logic.

    Args:
        func: Async function to execute
        config: Retry configuration
        retryable_exceptions: Tuple of exceptions to retry on

    Returns:
        Function result

    Raises:
        Exception: Last exception if all retries fail
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            result = await func()
            return result

        except retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_attempts - 1:
                # Last attempt failed
                logger.error(
                    f"All {config.max_attempts} retry attempts failed. "
                    f"Last error: {str(e)}"
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(
                config.initial_delay * (config.exponential_base**attempt),
                config.max_delay,
            )

            # Add jitter if enabled
            if config.jitter:
                import random

                delay *= 0.5 + random.random()

            logger.warning(
                f"Attempt {attempt + 1}/{config.max_attempts} failed: {str(e)}. "
                f"Retrying in {delay:.2f}s..."
            )

            await asyncio.sleep(delay)

        except Exception as e:
            # Non-retryable exception
            logger.error(f"Non-retryable error occurred: {str(e)}")
            raise

    # Should not reach here, but just in case
    raise last_exception


class TTLCache:
    """
    Simple TTL (Time-To-Live) cache implementation.

    Cached items expire after a specified duration.
    Uses OrderedDict for LRU eviction when size limit is reached.

    Attributes:
        max_size: Maximum number of items to cache
        ttl: Time-to-live in seconds
    """

    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        """
        Initialize TTL cache.

        Args:
            max_size: Maximum cache size
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self.lock:
            if key not in self.cache:
                return None

            value, timestamp = self.cache[key]

            # Check if expired
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            return value

    async def set(self, key: str, value: Any):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self.lock:
            # Remove oldest item if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            self.cache[key] = (value, time.time())

    async def delete(self, key: str):
        """
        Delete key from cache.

        Args:
            key: Cache key
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]

    async def clear(self):
        """Clear all cached items."""
        async with self.lock:
            self.cache.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        async with self.lock:
            # Clean up expired items
            current_time = time.time()
            expired_keys = [
                k for k, (_, ts) in self.cache.items() if current_time - ts > self.ttl
            ]
            for k in expired_keys:
                del self.cache[k]

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "expired_count": len(expired_keys),
            }


class CostTracker(BaseModel):
    """
    Track API usage costs.

    Attributes:
        total_requests: Total number of API requests
        total_tokens: Total tokens consumed
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        cost_per_1k_tokens: Cost per 1000 tokens in USD
    """

    total_requests: int = Field(default=0, description="Total API requests")
    total_tokens: int = Field(default=0, description="Total tokens consumed")
    cache_hits: int = Field(default=0, description="Cache hits")
    cache_misses: int = Field(default=0, description="Cache misses")
    cost_per_1k_tokens: float = Field(
        default=0.02, description="Cost per 1000 tokens (USD)"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def track_request(self, tokens: int, cached: bool = False):
        """
        Track an API request.

        Args:
            tokens: Number of tokens consumed
            cached: Whether response was from cache
        """
        self.total_requests += 1

        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.total_tokens += tokens

    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with usage and cost statistics
        """
        cache_rate = (
            self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0
        )

        estimated_cost = (self.total_tokens / 1000) * self.cost_per_1k_tokens

        # Calculate savings from cache
        if self.cache_hits > 0:
            avg_tokens_per_request = (
                self.total_tokens / self.cache_misses if self.cache_misses > 0 else 0
            )
            saved_tokens = self.cache_hits * avg_tokens_per_request
            saved_cost = (saved_tokens / 1000) * self.cost_per_1k_tokens
        else:
            saved_cost = 0.0

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_rate": f"{cache_rate:.2%}",
            "estimated_cost_usd": f"${estimated_cost:.4f}",
            "saved_cost_usd": f"${saved_cost:.4f}",
            "cost_per_1k_tokens": f"${self.cost_per_1k_tokens:.4f}",
        }

    def reset(self):
        """Reset all statistics."""
        self.total_requests = 0
        self.total_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API resilience.

    Prevents cascading failures by temporarily blocking requests
    when error rate exceeds threshold.

    States:
        - CLOSED: Normal operation
        - OPEN: Blocking requests due to high error rate
        - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Max calls to allow in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0
        self.lock = asyncio.Lock()

    async def call(self, func: Callable) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute

        Returns:
            Function result

        Raises:
            APIError: If circuit is open
        """
        async with self.lock:
            # Check circuit state
            if self.state == "OPEN":
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                else:
                    raise APIError("Circuit breaker is OPEN - service unavailable")

            if self.state == "HALF_OPEN":
                if self.half_open_calls >= self.half_open_max_calls:
                    raise APIError("Circuit breaker HALF_OPEN limit reached")
                self.half_open_calls += 1

        # Execute function
        try:
            result = await func()

            # Success - reset failure count
            async with self.lock:
                if self.state == "HALF_OPEN":
                    logger.info("Circuit breaker transitioning to CLOSED")
                    self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception as e:
            # Record failure
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    logger.error(
                        f"Circuit breaker OPEN after {self.failure_count} failures"
                    )
                    self.state = "OPEN"

            raise

    def get_state(self) -> Dict[str, Any]:
        """
        Get circuit breaker state.

        Returns:
            Dictionary with state information
        """
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
        }

    def reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.half_open_calls = 0
