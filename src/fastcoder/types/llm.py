"""LLM integration types — model router, providers, and call tracking."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    LOW = "low"
    MID = "mid"
    TOP = "top"


class TaskPurpose(str, Enum):
    STORY_ANALYSIS = "story_analysis"
    PLANNING = "planning"
    CODE_GENERATION = "code_generation"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    ERROR_ANALYSIS = "error_analysis"
    DOCUMENTATION = "documentation"
    COMMIT_MESSAGE = "commit_message"
    SELF_REFLECTION = "self_reflection"


class ReasoningMode(str, Enum):
    DIRECT = "direct"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_REFLECT = "self_reflect"
    TREE_OF_THOUGHT = "tree_of_thought"


class Message(BaseModel):
    role: str  # system, user, assistant, tool
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list["ToolCallRequest"]] = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class CompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: Optional[list[ToolDefinition]] = None
    max_tokens: int = 4096
    temperature: float = 0.3
    system_prompt: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    id: str = ""
    content: str = ""
    tool_calls: Optional[list[ToolCallRequest]] = None
    model: str = ""
    usage: "TokenUsage" = Field(default_factory=lambda: TokenUsage())
    finish_reason: str = "stop"  # stop, tool_calls, length, error
    latency_ms: float = 0


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class StreamChunk(BaseModel):
    content: Optional[str] = None
    tool_call: Optional[ToolCallRequest] = None
    done: bool = False


class ModelCall(BaseModel):
    id: str = ""
    model: str = ""
    provider: str = ""
    purpose: TaskPurpose = TaskPurpose.CODE_GENERATION
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthStatus(BaseModel):
    provider: str
    healthy: bool = True
    latency_ms: float = 0
    error: Optional[str] = None
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class UsageMetrics(BaseModel):
    provider: str
    total_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0


class RoutingContext(BaseModel):
    purpose: TaskPurpose
    complexity_score: float = 5.0
    token_budget: int = 4096
    cost_remaining_usd: float = 5.0
    preferred_tier: Optional[ModelTier] = None
    exclude_providers: list[str] = Field(default_factory=list)


class SelectedModel(BaseModel):
    provider: str
    model: str
    tier: ModelTier
    estimated_cost_per_1k_tokens: float = 0.0
    max_context_tokens: int = 128000


class PromptSection(BaseModel):
    role: str  # system, user, assistant
    template: str
    priority: int = 0
    required: bool = True
    cache_control: Optional[str] = None


class PromptTemplate(BaseModel):
    id: str
    version: str = "1.0"
    sections: list[PromptSection] = Field(default_factory=list)
    variables: dict[str, dict[str, Any]] = Field(default_factory=dict)
    output_schema: Optional[dict[str, Any]] = None
    reasoning_mode: ReasoningMode = ReasoningMode.CHAIN_OF_THOUGHT
    max_output_tokens: int = 4096


Message.model_rebuild()
