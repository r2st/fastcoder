"""Anthropic API provider implementation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic

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


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    # Pricing per 1M tokens
    MODEL_PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
        "claude-3-opus-20250219": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize Anthropic provider.

        Args:
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Base URL for API (optional)
        """
        super().__init__("Anthropic", "anthropic")
        self._client = AsyncAnthropic(api_key=api_key)
        self._base_url = base_url

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Get a single completion from Anthropic Claude.

        Args:
            request: Completion request

        Returns:
            Completion response

        Raises:
            ProviderError: If API call fails
        """
        start_time = time.perf_counter()
        try:
            # Separate system prompt from messages
            system_prompt = request.system_prompt or ""
            messages = []

            for msg in request.messages:
                if msg.role == "system":
                    system_prompt = msg.content
                else:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            # Map tool definitions to Anthropic format
            tools = None
            if request.tools:
                tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": {
                            "type": "object",
                            "properties": tool.parameters.get("properties", {}),
                            "required": tool.parameters.get("required", []),
                        },
                    }
                    for tool in request.tools
                ]

            # Make API call
            response = await self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools,
                temperature=request.temperature,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract content and tool calls
            content = ""
            tool_calls = None

            for block in response.content:
                if hasattr(block, "text"):
                    content = block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(
                        ToolCallRequest(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )

            # Calculate cost
            usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=(
                    response.usage.input_tokens + response.usage.output_tokens
                ),
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
                finish_reason=response.stop_reason or "stop",
                latency_ms=elapsed_ms,
            )

        except APIStatusError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_usage(0, 0, 0.0, elapsed_ms, error=True)
            raise ProviderError(f"Anthropic API error: {e.message}") from e
        except APIConnectionError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_usage(0, 0, 0.0, elapsed_ms, error=True)
            raise ProviderError(f"Anthropic connection error: {str(e)}") from e
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_usage(0, 0, 0.0, elapsed_ms, error=True)
            raise ProviderError(f"Unexpected error: {str(e)}") from e

    async def stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion from Anthropic Claude.

        Args:
            request: Completion request

        Yields:
            Stream chunks with partial content

        Raises:
            ProviderError: If API call fails
        """
        try:
            # Separate system prompt from messages
            system_prompt = request.system_prompt or ""
            messages = []

            for msg in request.messages:
                if msg.role == "system":
                    system_prompt = msg.content
                else:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            # Map tool definitions
            tools = None
            if request.tools:
                tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": {
                            "type": "object",
                            "properties": tool.parameters.get("properties", {}),
                            "required": tool.parameters.get("required", []),
                        },
                    }
                    for tool in request.tools
                ]

            start_time = time.perf_counter()
            input_tokens = 0
            output_tokens = 0
            accumulated_content = ""
            tool_calls_buffer = {}

            async with self._client.messages.stream(
                model=request.model,
                max_tokens=request.max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools,
                temperature=request.temperature,
            ) as stream:
                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_start":
                            if hasattr(event.content_block, "type"):
                                if event.content_block.type == "text":
                                    pass
                                elif event.content_block.type == "tool_use":
                                    tool_calls_buffer[event.content_block.id] = {
                                        "id": event.content_block.id,
                                        "name": event.content_block.name,
                                        "arguments": "",
                                    }

                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                accumulated_content += event.delta.text
                                yield StreamChunk(content=event.delta.text)
                            elif hasattr(event.delta, "input"):
                                # Tool use input being streamed
                                if hasattr(event.delta, "input"):
                                    pass

                        elif event.type == "message_delta":
                            if hasattr(event, "usage"):
                                output_tokens = event.usage.output_tokens

                        elif event.type == "message_start":
                            if hasattr(event.message, "usage"):
                                input_tokens = event.message.usage.input_tokens

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            cost = self.estimate_cost(input_tokens, output_tokens, request.model)
            self._record_usage(input_tokens, output_tokens, cost, elapsed_ms)

            yield StreamChunk(done=True)

        except APIStatusError as e:
            self._record_usage(0, 0, 0.0, 0.0, error=True)
            raise ProviderError(f"Anthropic API error: {e.message}") from e
        except Exception as e:
            self._record_usage(0, 0, 0.0, 0.0, error=True)
            raise ProviderError(f"Stream error: {str(e)}") from e

    async def health_check(self) -> HealthStatus:
        """Check Anthropic API health.

        Returns:
            Health status
        """
        start_time = time.perf_counter()
        try:
            # Minimal API call to check connectivity
            response = await self._client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return HealthStatus(
                provider="anthropic",
                healthy=True,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return HealthStatus(
                provider="anthropic",
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
