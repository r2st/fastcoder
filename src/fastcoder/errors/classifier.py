"""Error Classifier — categorizes errors and maps to recovery strategies."""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from fastcoder.types.errors import ErrorCategory, ErrorClassification, RecoveryStrategy


class ErrorClassifier:
    """Classifies errors into categories and determines recovery strategies."""

    # Error patterns for classification
    PATTERN_MAP = {
        ErrorCategory.SYNTAX_ERROR: [
            r"SyntaxError",
            r"IndentationError",
            r"invalid syntax",
            r"unexpected",
            r"expected.*received",
            r"Missing closing",
        ],
        ErrorCategory.TYPE_ERROR: [
            r"TypeError",
            r"TS2\d{3}",  # TypeScript errors
            r"is not compatible with",
            r"Argument of type",
            r"Property.*does not exist",
            r"Cannot assign",
            r"Type.*is not assignable",
        ],
        ErrorCategory.IMPORT_ERROR: [
            r"ImportError",
            r"ModuleNotFoundError",
            r"Cannot find module",
            r"No module named",
            r"cannot find name",
            r"import.*not found",
        ],
        ErrorCategory.LOGIC_ERROR: [
            r"AssertionError",
            r"Expected.*but got",
            r"assertion failed",
            r"test failed",
            r"expected value",
        ],
        ErrorCategory.ENVIRONMENT_ERROR: [
            r"ENOENT",  # File not found
            r"EACCES",  # Permission denied
            r"EADDRINUSE",  # Port in use
            r"FileNotFoundError",
            r"PermissionError",
            r"OSError",
            r"Environment variable",
        ],
        ErrorCategory.FLAKY_ERROR: [
            r"timeout",
            r"network",
            r"connection refused",
            r"temporary failure",
            r"rate limited",
        ],
        ErrorCategory.INTEGRATION_ERROR: [
            r"connection error",
            r"database error",
            r"API error",
            r"service unavailable",
            r"health check failed",
        ],
        ErrorCategory.ARCHITECTURAL_ERROR: [
            r"circular dependency",
            r"incompatible design",
            r"breaking change",
            r"cannot modify",
        ],
    }

    def classify(
        self,
        error_type: str,
        message: str,
        stack_trace: str = "",
    ) -> ErrorClassification:
        """Classify an error and determine recovery strategy.

        Args:
            error_type: Type of error (e.g., "SyntaxError", "TypeError")
            message: Error message
            stack_trace: Full stack trace if available

        Returns:
            ErrorClassification with category and recovery strategy
        """
        full_text = f"{error_type} {message} {stack_trace}".lower()

        # Find matching category
        category = ErrorCategory.UNKNOWN
        for cat, patterns in self.PATTERN_MAP.items():
            if any(re.search(pattern, full_text, re.IGNORECASE) for pattern in patterns):
                category = cat
                break

        # Map category to recovery strategy
        strategy = self._get_recovery_strategy(category)

        # Generate fingerprint
        fingerprint = self.generate_fingerprint(error_type, message)

        # Determine confidence based on pattern match
        confidence = self._calculate_confidence(error_type, category)

        return ErrorClassification(
            category=category,
            recovery_strategy=strategy,
            typical_fix_attempts=self._get_fix_attempts(category),
            fingerprint=fingerprint,
            confidence=confidence,
        )

    def generate_fingerprint(self, error_type: str, message: str, file: str = "") -> str:
        """Generate a unique fingerprint for this error.

        Strips dynamic values (line numbers, IDs, etc.) for consistency.

        Args:
            error_type: Type of error
            message: Error message (dynamic values stripped)
            file: File path where error occurred

        Returns:
            SHA-256 hash fingerprint
        """
        # Normalize the message by removing dynamic values
        normalized = self._normalize_message(message)

        # Create fingerprint components
        fingerprint_parts = [error_type, normalized]

        if file:
            # Only include directory, not full path
            fingerprint_parts.append(file.split("/")[-1])

        fingerprint_str = "|".join(fingerprint_parts)
        hash_obj = hashlib.sha256(fingerprint_str.encode())

        return hash_obj.hexdigest()[:16]  # Use first 16 chars

    def _normalize_message(self, message: str) -> str:
        """Normalize message by removing dynamic values.

        Examples:
            "Line 42: unexpected token" -> "unexpected token"
            "Expected string, got 123" -> "Expected string, got <value>"
        """
        # Remove line numbers
        normalized = re.sub(r"(?:line|at|:)\s*\d+", "", message)

        # Remove actual values in comparisons
        normalized = re.sub(r"got\s+['\"]?[^'\"\s]+['\"]?", "got <value>", normalized)
        normalized = re.sub(r"expected\s+['\"]?[^'\"\s]+['\"]?", "expected <type>", normalized)

        # Remove file paths and URLs
        normalized = re.sub(r"(/[^\s]+)+", "<path>", normalized)
        normalized = re.sub(r"https?://[^\s]+", "<url>", normalized)

        # Remove UUID/hash-like values
        normalized = re.sub(r"[a-f0-9]{8,}", "<hash>", normalized)

        return normalized.strip()

    def _get_recovery_strategy(self, category: ErrorCategory) -> RecoveryStrategy:
        """Map error category to recovery strategy."""
        strategy_map = {
            ErrorCategory.SYNTAX_ERROR: RecoveryStrategy.DIRECT_FIX,
            ErrorCategory.TYPE_ERROR: RecoveryStrategy.INCLUDE_TYPES,
            ErrorCategory.IMPORT_ERROR: RecoveryStrategy.CONSULT_SYMBOL_TABLE,
            ErrorCategory.LOGIC_ERROR: RecoveryStrategy.INCLUDE_BROAD_CONTEXT,
            ErrorCategory.INTEGRATION_ERROR: RecoveryStrategy.LOAD_API_SPECS,
            ErrorCategory.ENVIRONMENT_ERROR: RecoveryStrategy.ENVIRONMENT_REPAIR,
            ErrorCategory.FLAKY_ERROR: RecoveryStrategy.RERUN,
            ErrorCategory.ARCHITECTURAL_ERROR: RecoveryStrategy.REPLAN,
            ErrorCategory.UNKNOWN: RecoveryStrategy.DIRECT_FIX,
        }
        return strategy_map[category]

    def _get_fix_attempts(self, category: ErrorCategory) -> int:
        """Get typical number of fix attempts for this error category."""
        attempts_map = {
            ErrorCategory.SYNTAX_ERROR: 1,
            ErrorCategory.TYPE_ERROR: 2,
            ErrorCategory.IMPORT_ERROR: 2,
            ErrorCategory.LOGIC_ERROR: 3,
            ErrorCategory.ENVIRONMENT_ERROR: 2,
            ErrorCategory.FLAKY_ERROR: 3,
            ErrorCategory.INTEGRATION_ERROR: 2,
            ErrorCategory.ARCHITECTURAL_ERROR: 1,  # Escalate, don't retry
            ErrorCategory.UNKNOWN: 2,
        }
        return attempts_map[category]

    def _calculate_confidence(self, error_type: str, category: ErrorCategory) -> float:
        """Calculate confidence in the classification."""
        if category == ErrorCategory.UNKNOWN:
            return 0.3

        # Higher confidence for standard error types
        standard_types = {"SyntaxError", "TypeError", "ImportError", "ValueError"}
        if error_type in standard_types:
            return 0.95

        return 0.75
