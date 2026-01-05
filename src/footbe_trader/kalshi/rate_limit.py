"""Rate limiting and retry utilities for API calls."""

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

from footbe_trader.common.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_second: float = 10.0
    burst_size: int = 10


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Uses a sliding window approach to enforce rate limits.
    """

    def __init__(self, requests_per_second: float = 10.0, burst_size: int = 10):
        """Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second.
            burst_size: Maximum burst size (token bucket capacity).
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Acquire a token, waiting if necessary.

        Returns:
            Time waited in seconds.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            # Add tokens based on elapsed time
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.requests_per_second,
            )

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            # Calculate wait time
            wait_time = (1.0 - self._tokens) / self.requests_per_second
            self._tokens = 0.0

        # Wait outside the lock
        await asyncio.sleep(wait_time)
        return wait_time

    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        self._tokens = float(self.burst_size)
        self._last_update = time.monotonic()


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay in seconds.
            max_delay: Maximum delay in seconds.
            exponential_base: Base for exponential backoff.
            jitter: Whether to add random jitter.
            retryable_status_codes: HTTP status codes that trigger retry.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_status_codes = retryable_status_codes

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt.

        Args:
            attempt: Attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% jitter
            delay = delay * (0.75 + random.random() * 0.5)

        return delay


@dataclass
class RetryResult:
    """Result of a retried operation."""

    success: bool
    result: Any = None
    error: Exception | None = None
    attempts: int = 0
    total_wait_time: float = 0.0


class RequestLogger:
    """Logger for HTTP requests and responses."""

    def __init__(self, name: str = "kalshi_api"):
        """Initialize request logger.

        Args:
            name: Logger name.
        """
        self.logger = get_logger(name)
        self._request_times: deque[float] = deque(maxlen=100)

    def log_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> float:
        """Log an outgoing request.

        Args:
            method: HTTP method.
            url: Request URL.
            params: Query parameters.

        Returns:
            Start time for timing.
        """
        start_time = time.monotonic()
        self.logger.debug(
            "api_request",
            method=method,
            url=url,
            params=params,
        )
        return start_time

    def log_response(
        self,
        method: str,
        url: str,
        status_code: int,
        start_time: float,
        error: str | None = None,
    ) -> None:
        """Log a response.

        Args:
            method: HTTP method.
            url: Request URL.
            status_code: HTTP status code.
            start_time: Request start time.
            error: Error message if any.
        """
        duration_ms = (time.monotonic() - start_time) * 1000
        self._request_times.append(duration_ms)

        log_data = {
            "method": method,
            "url": url,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
        }

        if error:
            log_data["error"] = error
            self.logger.warning("api_response_error", **log_data)
        elif status_code >= 400:
            self.logger.warning("api_response_error", **log_data)
        else:
            self.logger.info("api_response", **log_data)

    def log_retry(
        self,
        method: str,
        url: str,
        attempt: int,
        delay: float,
        reason: str,
    ) -> None:
        """Log a retry attempt.

        Args:
            method: HTTP method.
            url: Request URL.
            attempt: Attempt number.
            delay: Delay before next attempt.
            reason: Reason for retry.
        """
        self.logger.info(
            "api_retry",
            method=method,
            url=url,
            attempt=attempt,
            delay_seconds=round(delay, 2),
            reason=reason,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get request statistics.

        Returns:
            Dictionary with request timing stats.
        """
        if not self._request_times:
            return {"count": 0}

        times = list(self._request_times)
        return {
            "count": len(times),
            "avg_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
        }
