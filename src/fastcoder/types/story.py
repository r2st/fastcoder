"""Story types — the primary input entity for the agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StoryState(str, Enum):
    RECEIVED = "RECEIVED"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    CODING = "CODING"
    REVIEWING = "REVIEWING"
    TESTING = "TESTING"
    DEPLOYING = "DEPLOYING"
    VERIFYING = "VERIFYING"
    DONE = "DONE"
    FAILED = "FAILED"


class StoryType(str, Enum):
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    INFRA = "infra"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AcceptanceCriterion(BaseModel):
    id: str
    description: str
    testable: bool = True
    given: Optional[str] = None
    when: Optional[str] = None
    then: Optional[str] = None
    verified: bool = False
    linked_test_ids: list[str] = Field(default_factory=list)


class FileDependency(BaseModel):
    file_path: str
    relationship: str  # imports, imported_by, modifies, creates, deletes
    confidence: float = 1.0


class StorySpec(BaseModel):
    title: str
    description: str
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    story_type: StoryType = StoryType.FEATURE
    complexity_score: int = Field(default=5, ge=1, le=10)
    dependencies: list[FileDependency] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)


class StoryConstraints(BaseModel):
    max_iterations: int = 10
    approval_gates: list[str] = Field(default_factory=lambda: ["pre_deploy"])
    target_branch: Optional[str] = None
    deploy_target: Optional[str] = None
    cost_budget_usd: float = 5.0


class StoryMetadata(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    model_usage: dict[str, int] = Field(default_factory=dict)


class Story(BaseModel):
    id: str
    raw_text: str
    project_id: str
    priority: Priority = Priority.MEDIUM
    spec: Optional[StorySpec] = None
    plan: Optional["ExecutionPlan"] = None  # Forward ref resolved at runtime
    state: StoryState = StoryState.RECEIVED
    iterations: list["Iteration"] = Field(default_factory=list)
    constraints: StoryConstraints = Field(default_factory=StoryConstraints)
    metadata: StoryMetadata = Field(default_factory=StoryMetadata)


class StorySubmission(BaseModel):
    story: str
    project_id: str
    priority: Priority = Priority.MEDIUM
    constraints: Optional[StoryConstraints] = None


# Deferred model rebuild for forward references
from fastcoder.types.plan import ExecutionPlan  # noqa: E402
from fastcoder.types.iteration import Iteration  # noqa: E402

Story.model_rebuild()
