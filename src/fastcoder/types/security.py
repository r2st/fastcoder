"""Security scanning types."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Security finding severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(str, Enum):
    """Categories of security findings."""

    SECRET_LEAK = "secret_leak"
    INJECTION = "injection"
    XSS = "xss"
    INSECURE_CRYPTO = "insecure_crypto"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    PATH_TRAVERSAL = "path_traversal"
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    INSECURE_PERMISSIONS = "insecure_permissions"
    DEPENDENCY_VULNERABILITY = "dependency_vulnerability"
    OTHER = "other"


class SecurityFinding(BaseModel):
    """A single security finding from a scanner."""

    id: str
    severity: Severity
    category: FindingCategory
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    cwe_id: Optional[str] = None
    scanner: str = ""  # bandit, eslint, semgrep, secret-detection
    rule_id: str = ""
    snippet: str = ""
    fix_suggestion: Optional[str] = None


class SecretFinding(SecurityFinding):
    """A secret detection finding (extends SecurityFinding)."""

    secret_type: str = ""  # api_key, password, token, private_key, etc.
    entropy: float = 0.0


class SASTReport(BaseModel):
    """Report from a single SAST scanner."""

    scanner: str
    findings: list[SecurityFinding] = Field(default_factory=list)
    scanned_files: int = 0
    scan_duration_ms: int = 0
    passed: bool = True  # True if no critical/high findings


class SecurityScanResult(BaseModel):
    """Complete security scan result."""

    sast_reports: list[SASTReport] = Field(default_factory=list)
    secret_findings: list[SecretFinding] = Field(default_factory=list)
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    passed: bool = True  # True if no critical/high findings
    scanned_at: datetime = Field(default_factory=datetime.utcnow)
