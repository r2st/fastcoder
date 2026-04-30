"""Error Handling System — classification, recovery, and learning."""

from fastcoder.errors.classifier import ErrorClassifier
from fastcoder.errors.recovery import (
    ErrorRecoveryCoordinator,
    RecoveryAction,
    RecoveryManager,
)

__all__ = [
    "ErrorClassifier",
    "RecoveryManager",
    "RecoveryAction",
    "ErrorRecoveryCoordinator",
]
