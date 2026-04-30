"""OpenAI API provider implementation."""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator, Optional

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from fastcoder.types.llm import (
    CompletionRequest,
    CompletionResponse,
    HealthStatus,
    StreamChunk,
    ToolCallRequest,
    TokenUsage,
)

from .base import LLMProvider

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    # Pricing per 1M tokens
    MODEL_PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.0},
        "gpt-4o-2024-05-13": {"input": 2.50, "output": 10.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4-turbo-2024-04-09": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize OpenAI provider.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (optional, for Azure or proxy)
        """
        super().__init__("OpenAI", "openai")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._base_url = base_url

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Get a single completion from OpenAI.

        Args:
            request: Completion request

        Returns:
            Completion response

        Raises:
            ProviderError: If API call fails
        """
        start_time = time.perf_counter()
        try:
            # Build messages list
            messages = []

            # Add system prompt if provided
            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt,
                })

            # Add user messages
            for msg in request.messages:
                if msg.role != "system":
                    messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            # Map tool definitions to OpenAI function calling format
            tools = None
            if request.tools:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": tool.parameters.get("properties", {}),
                                "required": tool.parameters.get("required", []),
                            },
                        },
                    }
                    for tool in request.tools
                ]

            # Make API call
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                tools=tools,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract content and tool calls
            content = response.choices[0].message.content or ""
            tool_calls = None

            if response.choices[0].message.tool_calls:
                tool_calls = [
                    ToolCallRequest(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                    for tc in response.choices[0].message.tool_calls
                ]

            # Calculate tokens and cost
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
            cost = self.estimate_cost(usage.input_tokens, usage.output_tokens, request.model)

            # Record usage
            self._record_usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                latency_ms=elapsed_ms,
            )

            return CompletionResponse(
                id=response.id,
                content=content,
                tool_calls=tool_calls,
                model=request.model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason or "stop",
                latency_ms=elapsed_ms,
            )

        except APIStatusError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_usage(0, 0, 0.0, elapsed_ms, error=True)
            raise ProviderError(f"OpenAI API error: {e.message}") from e
        except APIConnectionError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_usage(0, 0, 0.0, elapsed_ms, error=True)
            raise ProviderError(f"OpenAI connection error: {str(e)}") from e
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_usage(0, 0, 0.0, elapsed_ms, error=True)
            raise ProviderError(f"Unexpected error: {str(e)}") from e

    async def stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion from OpenAI.

        Args:
            request: Completion request

        Yields:
            Stream chunks with partial content

        Raises:
            ProviderError: If API call fails
        """
        try:
            # Build messages list
            messages = []

            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt,
                })

            for msg in request.messages:
                if msg.role != "system":
                    messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            # Map tools
            tools = None
            if request.tools:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                "type": "object",
                                "properties": tool.parameters.get("properties", {}),
                                "required": tool.parameters.get("required", []),
                            },
                        },
                    }
                    for tool in request.tools
                ]

            start_time = time.perf_counter()
            input_tokens = 0
            output_tokens = 0

            with await self._client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                tools=tools,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield StreamChunk(content=chunk.choices[0].delta.content)

                    # Track usage if available
                    if hasattr(chunk, "usage") and chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            cost = self.estimate_cost(input_tokens, output_tokens, request.model)
            self._record_usage(input_tokens, output_tokens, cost, elapsed_ms)

            yield StreamChunk(done=True)

        except APIStatusError as e:
            self._record_usage(0, 0, 0.0, 0.0, error=True)
            raise ProviderError(f"OpenAI API error: {e.message}") from e
        except Exception as e:
            self._record_usage(0, 0, 0.0, 0.0, error=True)
            raise ProviderError(f"Stream error: {str(e)}") from e

    async def health_check(self) -> HealthStatus:
        """Check OpenAI API health.

        Returns:
            Health status
        """
        start_time = time.perf_counter()
        try:
            # Minimal API call to check connectivity
            response = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return HealthStatus(
                provider="openai",
                healthy=True,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return HealthStatus(
                provider="openai",
                healthy=False,
                latency_ms=elapsed_ms,
                error=str(e),
            )

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Estimate cost for a completion.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            model: Model identifier

        Returns:
            Cost in USD
        """
        pricing = self.MODEL_PRICING.get(model)
        if not pricing:
            # Default pricing if model not found
            pricing = {"input": 0.003, "output": 0.015}

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
