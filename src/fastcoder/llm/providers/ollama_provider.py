"""Ollama local provider implementation."""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator, Optional

from ollama import AsyncClient

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


class OllamaProvider(LLMProvider):
    """Ollama local provider for running models locally."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama provider.

        Args:
            base_url: Base URL for Ollama service (default: http://localhost:11434)
        """
        super().__init__("Ollama", "ollama")
        self._client = AsyncClient(host=base_url)
        self._base_url = base_url

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Get a single completion from Ollama.

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

            # Make API call
            response = await self._client.chat(
                model=request.model,
                messages=messages,
                stream=False,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract content
            content = response.get("message", {}).get("content", "")

            # Ollama doesn't natively support tool calling, but we can parse
            # tool calls from structured responses if formatted correctly
            tool_calls = None
            try:
                # Attempt to parse tool calls from response if present
                if "tool_calls" in response.get("message", {}):
                    tool_call_data = response["message"]["tool_calls"]
                    if isinstance(tool_call_data, str):
                        tool_call_data = json.loads(tool_call_data)
                    if isinstance(tool_call_data, list):
                        tool_calls = [
                            ToolCallRequest(
                                id=tc.get("id", ""),
                                name=tc.get("name", ""),
                                arguments=tc.get("arguments", {}),
                            )
                            for tc in tool_call_data
                        ]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

            # Ollama doesn't return token counts, estimate based on text length
            # Rough estimate: ~4 chars per token
            estimated_input_tokens = len("".join(m["content"] for m in messages)) // 4
            estimated_output_tokens = len(content) // 4

            usage = TokenUsage(
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                total_tokens=estimated_input_tokens + estimated_output_tokens,
            )

            # Ollama is local, so cost is always $0
            cost = 0.0
            self._record_usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                latency_ms=elapsed_ms,
            )

            return CompletionResponse(
                id=self._generate_id(),
                content=content,
                tool_calls=tool_calls,
                model=request.model,
                usage=usage,
                finish_reason="stop",
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_usage(0, 0, 0.0, elapsed_ms, error=True)
            raise ProviderError(f"Ollama error: {str(e)}") from e

    async def stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion from Ollama.

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

            start_time = time.perf_counter()
            accumulated_content = ""

            # Use streaming API
            async for chunk in await self._client.chat(
                model=request.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    accumulated_content += content
                    yield StreamChunk(content=content)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Estimate tokens
            estimated_input_tokens = len("".join(m["content"] for m in messages)) // 4
            estimated_output_tokens = len(accumulated_content) // 4

            self._record_usage(
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                cost_usd=0.0,  # Local model
                latency_ms=elapsed_ms,
            )

            yield StreamChunk(done=True)

        except Exception as e:
            self._record_usage(0, 0, 0.0, 0.0, error=True)
            raise ProviderError(f"Stream error: {str(e)}") from e

    async def health_check(self) -> HealthStatus:
        """Check Ollama service health.

        Returns:
            Health status
        """
        start_time = time.perf_counter()
        try:
            # List available models to check connectivity
            models = await self._client.list()
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if models and "models" in models and len(models["models"]) > 0:
                return HealthStatus(
                    provider="ollama",
                    healthy=True,
                    latency_ms=elapsed_ms,
                )
            else:
                return HealthStatus(
                    provider="ollama",
                    healthy=False,
                    latency_ms=elapsed_ms,
                    error="No models available",
                )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return HealthStatus(
                provider="ollama",
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
            Cost in USD (always $0 for local models)
        """
        return 0.0
