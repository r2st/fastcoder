"""Post-mortem learning types."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FailureClass(str, Enum):
    """Classification of failure modes."""

    GATE_FAILURE = "gate_failure"
    REVIEWER_REJECTION = "reviewer_rejection"
    TEST_REGRESSION = "test_regression"
    CONVENTION_VIOLATION = "convention_violation"
    SECURITY_FINDING = "security_finding"
    PERFORMANCE_REGRESSION = "performance_regression"
    BUILD_FAILURE = "build_failure"
    DEPLOYMENT_FAILURE = "deployment_failure"


class PostMortemEntry(BaseModel):
    """Record of a failure and its analysis."""

    id: str
    story_id: str
    project_id: str
    failure_class: FailureClass
    failure_details: str = ""
    root_cause: str = ""
    error_fingerprint: str = ""
    gate_type: Optional[str] = None
    reviewer_comment: Optional[str] = None
    resolution: str = ""
    heuristic_update: Optional[str] = None
    memory_entry_id: Optional[str] = None
    iteration_count: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    effectiveness_measured: bool = False
    prevented_future_failures: int = 0

    model_config = {"frozen": False}


class HeuristicRule(BaseModel):
    """A learned rule for preventing future failures."""

    id: str
    project_id: str
    rule_type: str
    trigger_pattern: str = ""
    action: str = ""
    source_post_mortem_id: str = ""
    times_applied: int = 0
    times_effective: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True

    model_config = {"frozen": False}


class LearningStats(BaseModel):
    """Aggregate statistics on learning and improvements."""

    total_post_mortems: int = 0
    total_heuristics: int = 0
    active_heuristics: int = 0
    failure_reduction_rate: float = 0.0
    top_failure_classes: dict[str, int] = Field(default_factory=dict)

    model_config = {"frozen": False}
