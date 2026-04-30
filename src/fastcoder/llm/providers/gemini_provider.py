"""Google Gemini API provider implementation."""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator, Optional

from google import genai
from google.genai import types

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


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    # Pricing per 1M tokens
    MODEL_PRICING = {
        "gemini-2-5-pro": {"input": 1.25, "output": 10.0},
        "gemini-2-5-flash": {"input": 0.15, "output": 0.60},
        "gemini-2-0-flash": {"input": 0.10, "output": 0.40},
        "gemini-1-5-pro": {"input": 1.25, "output": 5.0},
        "gemini-1-5-flash": {"input": 0.075, "output": 0.30},
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini provider.

        Args:
            api_key: API key (defaults to GOOGLE_API_KEY env var)
        """
        super().__init__("Google", "gemini")
        self._client = genai.Client(api_key=api_key)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Get a single completion from Google Gemini.

        Args:
            request: Completion request

        Returns:
            Completion response

        Raises:
            ProviderError: If API call fails
        """
        start_time = time.perf_counter()
        try:
            # Build system instruction
            system_instruction = request.system_prompt or ""

            # Map messages to Gemini format
            contents = []
            for msg in request.messages:
                if msg.role == "system":
                    system_instruction = msg.content
                else:
                    # Map assistant to model for Gemini
                    role = "model" if msg.role == "assistant" else msg.role
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part(text=msg.content)],
                        )
                    )

            # Map tool definitions to Gemini format
            tools = None
            if request.tools:
                functions = [
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=types.Schema(
                            type="object",
                            properties={
                                k: types.Schema(
                                    type=v.get("type", "string"),
                                    description=v.get("description", ""),
                                )
                                for k, v in tool.parameters.get("properties", {}).items()
                            },
                            required=tool.parameters.get("required", []),
                        ),
                    )
                    for tool in request.tools
                ]
                tools = [types.Tool(function_declarations=functions)]

            # Make API call
            response = await self._client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                system_instruction=system_instruction or None,
                config=types.GenerateContentConfig(
                    max_output_tokens=request.max_tokens,
                    temperature=request.temperature,
                    tools=tools,
                ),
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract content and tool calls
            content = ""
            tool_calls = None

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            content = part.text
                        elif hasattr(part, "function_call") and part.function_call:
                            if tool_calls is None:
                                tool_calls = []
                            tool_calls.append(
                                ToolCallRequest(
                                    id=part.function_call.name,
                                    name=part.function_call.name,
                                    arguments=dict(part.function_call.args),
                                )
                            )

            # Extract usage information
            usage_data = response.usage_metadata if hasattr(response, "usage_metadata") else None
            input_tokens = usage_data.prompt_token_count if usage_data else 0
            output_tokens = usage_data.candidates_token_count if usage_data else 0

            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

            cost = self.estimate_cost(usage.input_tokens, usage.output_tokens, request.model)

            # Record usage
            self._record_usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                latency_ms=elapsed_ms,
            )

            # Determine finish reason
            finish_reason = "stop"
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason_enum = response.candidates[0].finish_reason
                if hasattr(finish_reason_enum, "name"):
                    finish_reason = finish_reason_enum.name.lower()

            return CompletionResponse(
                id=self._generate_id(),
                content=content,
                tool_calls=tool_calls,
                model=request.model,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_usage(0, 0, 0.0, elapsed_ms, error=True)
            raise ProviderError(f"Gemini API error: {str(e)}") from e

    async def stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion from Google Gemini.

        Args:
            request: Completion request

        Yields:
            Stream chunks with partial content

        Raises:
            ProviderError: If API call fails
        """
        try:
            # Build system instruction
            system_instruction = request.system_prompt or ""

            # Map messages to Gemini format
            contents = []
            for msg in request.messages:
                if msg.role == "system":
                    system_instruction = msg.content
                else:
                    # Map assistant to model for Gemini
                    role = "model" if msg.role == "assistant" else msg.role
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part(text=msg.content)],
                        )
                    )

            # Map tool definitions to Gemini format
            tools = None
            if request.tools:
                functions = [
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=types.Schema(
                            type="object",
                            properties={
                                k: types.Schema(
                                    type=v.get("type", "string"),
                                    description=v.get("description", ""),
                                )
                                for k, v in tool.parameters.get("properties", {}).items()
                            },
                            required=tool.parameters.get("required", []),
                        ),
                    )
                    for tool in request.tools
                ]
                tools = [types.Tool(function_declarations=functions)]

            start_time = time.perf_counter()
            input_tokens = 0
            output_tokens = 0
            accumulated_content = ""

            async with await self._client.aio.models.generate_content_stream(
                model=request.model,
                contents=contents,
                system_instruction=system_instruction or None,
                config=types.GenerateContentConfig(
                    max_output_tokens=request.max_tokens,
                    temperature=request.temperature,
                    tools=tools,
                ),
            ) as stream:
                async for chunk in stream:
                    if chunk.candidates and len(chunk.candidates) > 0:
                        candidate = chunk.candidates[0]
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    accumulated_content += part.text
                                    yield StreamChunk(content=part.text)

                    # Extract usage if available
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        input_tokens = chunk.usage_metadata.prompt_token_count
                        output_tokens = chunk.usage_metadata.candidates_token_count

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            cost = self.estimate_cost(input_tokens, output_tokens, request.model)
            self._record_usage(input_tokens, output_tokens, cost, elapsed_ms)

            yield StreamChunk(done=True)

        except Exception as e:
            self._record_usage(0, 0, 0.0, 0.0, error=True)
            raise ProviderError(f"Stream error: {str(e)}") from e

    async def health_check(self) -> HealthStatus:
        """Check Gemini API health.

        Returns:
            Health status
        """
        start_time = time.perf_counter()
        try:
            # Minimal API call to check connectivity
            response = await self._client.aio.models.generate_content(
                model="gemini-2-0-flash",
                contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
                config=types.GenerateContentConfig(max_output_tokens=10),
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return HealthStatus(
                provider="gemini",
                healthy=True,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return HealthStatus(
                provider="gemini",
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
            pricing = {"input": 0.15, "output": 0.60}

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
