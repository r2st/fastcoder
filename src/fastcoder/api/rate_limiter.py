"""Rate limiting middleware using token bucket algorithm."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import structlog
from fastapi import FastAPI, Request, HTTPException, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_minute: Number of requests allowed per minute.
        burst_size: Maximum burst size (tokens in bucket). If None, defaults to requests_per_minute.
    """

    requests_per_minute: int = 60
    burst_size: Optional[int] = None

    def __post_init__(self) -> None:
        """Set burst_size to requests_per_minute if not specified."""
        if self.burst_size is None:
            self.burst_size = self.requests_per_minute


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Attributes:
        capacity: Maximum number of tokens in the bucket.
        refill_rate: Tokens to add per second.
        tokens: Current number of tokens.
        last_refill: Timestamp of last refill.
        lock: Asyncio lock for thread-safe access.
    """

    capacity: float
    refill_rate: float
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def try_consume(self, amount: float = 1.0) -> bool:
        """Try to consume tokens from the bucket.

        Args:
            amount: Number of tokens to consume.

        Returns:
            True if tokens were available and consumed, False otherwise.
        """
        async with self.lock:
            await self._refill()
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    async def get_retry_after(self) -> float:
        """Get seconds to wait before next token is available.

        Returns:
            Seconds until next token is available.
        """
        async with self.lock:
            await self._refill()
            if self.tokens >= 1.0:
                return 0.0
            # Calculate how long until next token
            return (1.0 - self.tokens) / self.refill_rate


class RateLimiter:
    """Rate limiter using token bucket algorithm.

    Tracks per-IP rate limits with configurable rates. Includes automatic
    cleanup of stale buckets (not accessed for 10+ minutes).
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize the rate limiter.

        Args:
            config: RateLimitConfig with rate limits.
        """
        self.config = config
        self.buckets: dict[str, TokenBucket] = {}
        self.lock = asyncio.Lock()
        # Stale bucket cleanup: track last access time
        self.bucket_access_times: dict[str, float] = {}
        self.stale_bucket_ttl = 600.0  # 10 minutes
        logger.info(
            "rate_limiter_initialized",
            requests_per_minute=config.requests_per_minute,
            burst_size=config.burst_size,
        )

    async def is_allowed(self, client_ip: str) -> bool:
        """Check if a request from client_ip is allowed.

        Args:
            client_ip: Client IP address.

        Returns:
            True if request is allowed, False if rate limit exceeded.
        """
        # Clean stale buckets periodically
        await self._cleanup_stale_buckets()

        bucket = await self._get_or_create_bucket(client_ip)
        self.bucket_access_times[client_ip] = time.time()

        allowed = await bucket.try_consume(1.0)
        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                limit=self.config.requests_per_minute,
            )
        return allowed

    async def get_retry_after(self, client_ip: str) -> float:
        """Get retry-after seconds for a client.

        Args:
            client_ip: Client IP address.

        Returns:
            Seconds to wait before next request.
        """
        bucket = await self._get_or_create_bucket(client_ip)
        return await bucket.get_retry_after()

    async def _get_or_create_bucket(self, client_ip: str) -> TokenBucket:
        """Get or create a token bucket for a client.

        Args:
            client_ip: Client IP address.

        Returns:
            TokenBucket for the client.
        """
        async with self.lock:
            if client_ip not in self.buckets:
                # Calculate refill rate: tokens per second
                refill_rate = self.config.requests_per_minute / 60.0
                bucket = TokenBucket(
                    capacity=float(self.config.burst_size),
                    refill_rate=refill_rate,
                    tokens=float(self.config.burst_size),
                )
                self.buckets[client_ip] = bucket
                self.bucket_access_times[client_ip] = time.time()
                logger.debug(
                    "token_bucket_created",
                    client_ip=client_ip,
                    capacity=bucket.capacity,
                    refill_rate=refill_rate,
                )
            return self.buckets[client_ip]

    async def _cleanup_stale_buckets(self) -> None:
        """Remove buckets not accessed in the last 10 minutes."""
        now = time.time()
        async with self.lock:
            stale_ips = [
                ip
                for ip, last_access in self.bucket_access_times.items()
                if now - last_access > self.stale_bucket_ttl
            ]
            for ip in stale_ips:
                del self.buckets[ip]
                del self.bucket_access_times[ip]
                logger.debug("stale_bucket_removed", client_ip=ip)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app: FastAPI, rate_limiter: RateLimiter) -> None:
        """Initialize the middleware.

        Args:
            app: FastAPI application.
            rate_limiter: RateLimiter instance.
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Rate limit incoming requests.

        Args:
            request: Incoming request.
            call_next: Next middleware/route handler.

        Returns:
            Response (429 if rate limited, otherwise normal response).
        """
        client_ip = request.client.host if request.client else "unknown"

        allowed = await self.rate_limiter.is_allowed(client_ip)
        if not allowed:
            retry_after = await self.rate_limiter.get_retry_after(client_ip)
            logger.warning(
                "rate_limit_exceeded_response",
                client_ip=client_ip,
                retry_after=retry_after,
            )
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        response = await call_next(request)
        return response


async def rate_limit(
    request: Request, rate_limiter: RateLimiter = Depends()
) -> bool:
    """Dependency for per-route rate limiting.

    Can be used as a dependency on routes that need stricter limits
    than the global middleware. Should be combined with a tighter
    RateLimitConfig passed as the Depends() argument.

    Args:
        request: Current request.
        rate_limiter: RateLimiter instance (from dependency).

    Returns:
        True if request is allowed.

    Raises:
        HTTPException: 429 if rate limit exceeded.
    """
    client_ip = request.client.host if request.client else "unknown"

    allowed = await rate_limiter.is_allowed(client_ip)
    if not allowed:
        retry_after = await rate_limiter.get_retry_after(client_ip)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    return True


def create_rate_limit_middleware(
    app: FastAPI,
    config: RateLimitConfig,
) -> RateLimiter:
    """Create and attach rate limit middleware to FastAPI app.

    Args:
        app: FastAPI application instance.
        config: RateLimitConfig with rate limit settings.

    Returns:
        RateLimiter instance that was added to the app.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> config = RateLimitConfig(requests_per_minute=60)
        >>> rate_limiter = create_rate_limit_middleware(app, config)
    """
    rate_limiter = RateLimiter(config)
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    logger.info(
        "rate_limit_middleware_added",
        requests_per_minute=config.requests_per_minute,
    )
    return rate_limiter
