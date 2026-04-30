"""Abstract base class for LLM providers."""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator

from fastcoder.types.llm import (
    CompletionRequest,
    CompletionResponse,
    HealthStatus,
    StreamChunk,
    UsageMetrics,
)


@dataclass
class UsageTracker:
    """Internal usage tracking."""

    provider: str
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    error_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across calls."""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.error_count / self.total_calls) * 100.0

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record a completed call."""
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd
        self.total_latency_ms += latency_ms
        if error:
            self.error_count += 1

    def to_metrics(self) -> UsageMetrics:
        """Convert to UsageMetrics."""
        return UsageMetrics(
            provider=self.provider,
            total_calls=self.total_calls,
            total_tokens=self.total_input_tokens + self.total_output_tokens,
            total_cost_usd=self.total_cost_usd,
            avg_latency_ms=self.avg_latency_ms,
            error_rate=self.error_rate,
        )


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, provider_type: str):
        """Initialize provider.

        Args:
            name: Human-readable provider name
            provider_type: Type identifier (anthropic, openai, ollama, etc.)
        """
        self._name = name
        self._provider_type = provider_type
        self._usage = UsageTracker(provider=provider_type)

    @property
    def name(self) -> str:
        """Provider name."""
        return self._name

    @property
    def provider_type(self) -> str:
        """Provider type identifier."""
        return self._provider_type

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Get a single completion from the provider.

        Args:
            request: Completion request

        Returns:
            Completion response with content, tool calls, and usage

        Raises:
            ProviderError: If provider call fails
        """

    @abstractmethod
    async def stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion from the provider.

        Args:
            request: Completion request

        Yields:
            Stream chunks with partial content and final status

        Raises:
            ProviderError: If provider call fails
        """

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check provider health with minimal API call.

        Returns:
            Health status with latency and error info
        """

    def get_usage(self) -> UsageMetrics:
        """Get usage metrics since initialization.

        Returns:
            Accumulated usage statistics
        """
        return self._usage.to_metrics()

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Estimate cost for a completion.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            model: Model identifier

        Returns:
            Estimated cost in USD (default: $0, override in subclass)
        """
        return 0.0

    def _record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record usage internally.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            cost_usd: Cost in USD
            latency_ms: Latency in milliseconds
            error: Whether this call errored
        """
        self._usage.record_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            error=error,
        )

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique ID for responses."""
        return str(uuid.uuid4())

    @staticmethod
    async def _measure_time(coro):
        """Measure execution time of a coroutine.

        Args:
            coro: Coroutine to measure

        Returns:
            Tuple of (result, elapsed_ms)
        """
        start = time.perf_counter()
        result = await coro
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
