"""Tool & Environment layer types."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    FILE_SYSTEM = "file_system"
    SHELL = "shell"
    GIT = "git"
    PACKAGE_MANAGER = "package_manager"
    BUILD_TOOLS = "build_tools"
    TEST_RUNNER = "test_runner"
    DATABASE = "database"
    HTTP_CLIENT = "http_client"
    CONTAINER_RUNTIME = "container_runtime"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolCall(BaseModel):
    tool: ToolName
    operation: str
    args: dict[str, Any] = Field(default_factory=dict)
    working_dir: Optional[str] = None
    timeout_ms: Optional[int] = None
    env: Optional[dict[str, str]] = None


class SideEffects(BaseModel):
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    packages_added: list[str] = Field(default_factory=list)
    packages_removed: list[str] = Field(default_factory=list)


class LintError(BaseModel):
    file: str
    line: int
    column: int = 0
    rule: str = ""
    message: str = ""
    severity: str = "error"
    fixable: bool = False


class TypeCheckError(BaseModel):
    file: str
    line: int
    column: int = 0
    code: str = ""
    message: str = ""


class ParsedToolOutput(BaseModel):
    type: str  # test_report, lint_report, type_check, build_output, coverage_report, generic
    data: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: ToolName
    operation: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0
    side_effects: SideEffects = Field(default_factory=SideEffects)
    parsed: Optional[ParsedToolOutput] = None


class SandboxConfig(BaseModel):
    project_dir: str
    cpu_limit: str = "2"
    memory_limit_mb: int = 2048
    disk_limit_mb: int = 10240
    timeout_ms: int = 300000
    network_whitelist: list[str] = Field(
        default_factory=lambda: ["registry.npmjs.org", "pypi.org", "github.com"]
    )
    env_vars: dict[str, str] = Field(default_factory=dict)


class ToolPolicy(BaseModel):
    tool: ToolName
    allowed_operations: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    requires_approval: list[str] = Field(default_factory=list)
    max_calls_per_minute: int = 60
    max_calls_per_story: int = 500
