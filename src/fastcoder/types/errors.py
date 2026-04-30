"""Error taxonomy types — for classifying and recovering from errors."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    IMPORT_ERROR = "import_error"
    LOGIC_ERROR = "logic_error"
    INTEGRATION_ERROR = "integration_error"
    ENVIRONMENT_ERROR = "environment_error"
    FLAKY_ERROR = "flaky_error"
    ARCHITECTURAL_ERROR = "architectural_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    DIRECT_FIX = "direct_fix"
    INCLUDE_TYPES = "include_types"
    CONSULT_SYMBOL_TABLE = "consult_symbol_table"
    INCLUDE_BROAD_CONTEXT = "include_broad_context"
    LOAD_API_SPECS = "load_api_specs"
    ENVIRONMENT_REPAIR = "environment_repair"
    RERUN = "rerun"
    REPLAN = "replan"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class ErrorClassification(BaseModel):
    category: ErrorCategory = ErrorCategory.UNKNOWN
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.DIRECT_FIX
    typical_fix_attempts: int = 1
    fingerprint: str = ""
    confidence: float = 0.0


class ErrorDetail(BaseModel):
    type: str = ""
    message: str = ""
    stack_trace: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None


class PreviousAttempt(BaseModel):
    attempt_number: int
    code_diff: str = ""
    error_message: str = ""
    error_fingerprint: str = ""


class ErrorContext(BaseModel):
    attempt: int = 1
    previous_code: Optional[str] = None
    error: ErrorDetail = Field(default_factory=ErrorDetail)
    classification: ErrorClassification = Field(default_factory=ErrorClassification)
    previous_attempts: list[PreviousAttempt] = Field(default_factory=list)
    instruction: Optional[str] = None


class ErrorFingerprint(BaseModel):
    hash: str
    category: ErrorCategory
    pattern: str
    file_location: Optional[str] = None
    known_fix: Optional[str] = None
    occurrences: int = 0
    last_seen: datetime = Field(default_factory=datetime.utcnow)
