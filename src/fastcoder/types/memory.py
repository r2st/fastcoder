"""Memory store types — for RAG-based self-improvement."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    PATTERN = "pattern"
    ERROR_FIX = "error_fix"
    CONVENTION = "convention"
    ANTI_PATTERN = "anti_pattern"


class MemoryTier(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PROJECT = "project"


class MemoryEntry(BaseModel):
    id: str
    type: MemoryType
    tier: MemoryTier
    context: str = ""
    content: str = ""
    source_story_id: str = ""
    effectiveness_score: float = 0.5
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: datetime = Field(default_factory=datetime.utcnow)
    use_count: int = 0
    transferability_score: float = 0.5
    project_id: Optional[str] = None


class MemoryQuery(BaseModel):
    query: str
    tier: Optional[MemoryTier] = None
    type: Optional[MemoryType] = None
    project_id: Optional[str] = None
    max_results: int = 10
    min_effectiveness: float = 0.0


class MemoryConsolidationResult(BaseModel):
    new_memories: list[MemoryEntry] = Field(default_factory=list)
    updated_memories: list[MemoryEntry] = Field(default_factory=list)
    evicted_memory_ids: list[str] = Field(default_factory=list)
    merged_count: int = 0
