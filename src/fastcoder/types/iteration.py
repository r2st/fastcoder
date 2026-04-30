"""Iteration types — tracking each attempt within a story lifecycle."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from fastcoder.types.task import (
    DeployReport,
    FileChange,
    ReviewReport,
    TestReport,
)


class Iteration(BaseModel):
    number: int
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    stage: str = ""
    changes: list[FileChange] = Field(default_factory=list)
    test_results: Optional[TestReport] = None
    review_results: Optional[ReviewReport] = None
    deploy_results: Optional[DeployReport] = None
    error_context: Optional["ErrorContext"] = None
    model_calls: list["ModelCall"] = Field(default_factory=list)


# Deferred imports for forward references
from fastcoder.types.errors import ErrorContext  # noqa: E402
from fastcoder.types.llm import ModelCall  # noqa: E402

Iteration.model_rebuild()
