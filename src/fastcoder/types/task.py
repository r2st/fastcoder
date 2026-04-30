"""Task types — individual units of work within an execution plan."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileChange(BaseModel):
    file_path: str
    change_type: str  # created, modified, deleted
    diff: Optional[str] = None
    content: Optional[str] = None
    previous_content: Optional[str] = None


class TestFailure(BaseModel):
    suite: str
    test: str
    error: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    stack_trace: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None


class SuiteResult(BaseModel):
    name: str
    file: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: float = 0


class TestReport(BaseModel):
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: float = 0
    coverage_percent: Optional[float] = None
    failures: list[TestFailure] = Field(default_factory=list)
    suite_results: list[SuiteResult] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    severity: str  # blocking, suggestion, nit
    category: str  # security, performance, correctness, style, maintainability
    file: str
    line: Optional[int] = None
    description: str
    suggested_fix: Optional[str] = None


class ReviewReport(BaseModel):
    approved: bool = False
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str = ""
    reviewer_model: str = ""


class DeployReport(BaseModel):
    success: bool = False
    environment: str = ""
    url: Optional[str] = None
    health_check_passed: bool = False
    smoke_tests_passed: bool = False
    rollback_triggered: bool = False
    error: Optional[str] = None


class TaskResult(BaseModel):
    status: TaskStatus = TaskStatus.PENDING
    changes: list[FileChange] = Field(default_factory=list)
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0
