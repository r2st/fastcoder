"""Quality gate types."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class GateType(str, Enum):
    """Types of quality gates."""
    LINT = "lint"
    TYPE_CHECK = "type_check"
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    E2E_TEST = "e2e_test"
    SAST = "sast"
    DEPENDENCY_AUDIT = "dependency_audit"
    SECRET_DETECTION = "secret_detection"
    COVERAGE_DELTA = "coverage_delta"
    PERFORMANCE_BUDGET = "performance_budget"
    MIGRATION_SAFETY = "migration_safety"


class GateOutcome(str, Enum):
    """Result of running a quality gate."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class EnforcementLevel(str, Enum):
    """How strictly to enforce a quality gate."""
    REQUIRED = "required"      # Failure blocks merge
    WARNING_ONLY = "warning"   # Failure warns but doesn't block
    OPTIONAL = "optional"      # Run but never block
    DISABLED = "disabled"      # Don't run


class GateThreshold(BaseModel):
    """Configurable threshold for a quality gate."""
    gate_type: GateType
    enforcement: EnforcementLevel = EnforcementLevel.REQUIRED
    min_coverage: Optional[float] = None        # For coverage gates (0-100)
    max_findings: Optional[int] = None           # For SAST/lint findings
    max_severity: Optional[str] = None           # e.g., "medium" - block if higher
    custom_command: Optional[str] = None         # For custom gates
    custom_args: list[str] = Field(default_factory=list)
    timeout_seconds: int = 300

    class Config:
        """Pydantic config."""
        use_enum_values = False


class GateResult(BaseModel):
    """Result of running a single quality gate."""
    gate_type: GateType
    outcome: GateOutcome
    enforcement: EnforcementLevel
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    findings_count: int = 0
    duration_ms: int = 0
    executed_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config."""
        use_enum_values = False


class QualityGatePolicy(BaseModel):
    """A complete quality gate policy configuration."""
    name: str = "default"
    description: str = ""
    gates: list[GateThreshold] = Field(default_factory=list)

    # Overall policy settings
    fail_fast: bool = False  # Stop on first required gate failure
    parallel_execution: bool = True  # Run independent gates concurrently

    class Config:
        """Pydantic config."""
        use_enum_values = False


class PolicyEvaluationResult(BaseModel):
    """Result of evaluating all gates in a policy."""
    policy_name: str
    results: list[GateResult] = Field(default_factory=list)
    all_required_passed: bool = True
    has_warnings: bool = False
    total_duration_ms: int = 0
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    recommended_action: str = "merge"  # merge, review, block

    class Config:
        """Pydantic config."""
        use_enum_values = False
