"""Recovery Manager — handles error recovery with progressive context enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fastcoder.types.errors import ErrorClassification, ErrorCategory, RecoveryStrategy


@dataclass
class RecoveryAction:
    """Action to take to recover from an error."""

    strategy: RecoveryStrategy
    additional_context: dict = field(default_factory=dict)
    switch_to_top_tier: bool = False  # Switch to more powerful LLM model
    replan: bool = False  # Escalate to planner
    escalate: bool = False  # Escalate to human
    max_retries: int = 1


class RecoveryManager:
    """Manages error recovery strategies and learning from fixes."""

    def __init__(self) -> None:
        """Initialize the recovery manager."""
        # Store known fixes: fingerprint -> (fix_code, story_id)
        self.fingerprint_db: dict[str, tuple[str, str]] = {}

    def get_strategy(
        self,
        classification: ErrorClassification,
        attempt: int,
    ) -> RecoveryAction:
        """Determine recovery action for an error.

        Args:
            classification: ErrorClassification from classifier
            attempt: Current attempt number (1-based)

        Returns:
            RecoveryAction with strategy and context
        """
        category = classification.category
        strategy = classification.recovery_strategy
        confidence = classification.confidence

        # Check if we've seen this fingerprint before
        if classification.fingerprint in self.fingerprint_db:
            known_fix, _ = self.fingerprint_db[classification.fingerprint]
            return RecoveryAction(
                strategy=RecoveryStrategy.DIRECT_FIX,
                additional_context={"known_fix": known_fix},
                max_retries=1,
            )

        # Escalate if we're beyond typical attempts
        max_typical_attempts = classification.typical_fix_attempts
        if attempt > max_typical_attempts:
            # Escalate based on category
            if category == ErrorCategory.ARCHITECTURAL_ERROR:
                return RecoveryAction(
                    strategy=RecoveryStrategy.REPLAN,
                    escalate=True,
                    replan=True,
                )
            elif category == ErrorCategory.INTEGRATION_ERROR:
                return RecoveryAction(
                    strategy=RecoveryStrategy.LOAD_API_SPECS,
                    switch_to_top_tier=True,
                    max_retries=2,
                )
            else:
                return RecoveryAction(
                    strategy=RecoveryStrategy.ESCALATE_TO_HUMAN,
                    escalate=True,
                )

        # Progressive context enrichment based on attempt
        context = {}

        if attempt == 1:
            # First attempt: simple fix with error message
            context = self._get_level_1_context()

        elif attempt == 2:
            # Second attempt: include type definitions and symbols
            context = self._get_level_2_context()
            if confidence < 0.7:
                # Low confidence: switch to stronger model
                return RecoveryAction(
                    strategy=strategy,
                    additional_context=context,
                    switch_to_top_tier=True,
                    max_retries=1,
                )

        elif attempt >= 3:
            # Third attempt: broad context, codebase patterns
            context = self._get_level_3_context()
            # Use top-tier model for complex issues
            if category in [
                ErrorCategory.LOGIC_ERROR,
                ErrorCategory.INTEGRATION_ERROR,
            ]:
                return RecoveryAction(
                    strategy=strategy,
                    additional_context=context,
                    switch_to_top_tier=True,
                    max_retries=1,
                )

        return RecoveryAction(
            strategy=strategy,
            additional_context=context,
            switch_to_top_tier=(
                attempt >= 2 and confidence < 0.7
            ),  # Escalate model if low confidence
            max_retries=1,
        )

    def record_fix(
        self,
        fingerprint: str,
        fix: str,
        story_id: str,
    ) -> None:
        """Record a successful fix for future reference.

        Args:
            fingerprint: Error fingerprint
            fix: The code fix that resolved the error
            story_id: Story ID for tracking
        """
        self.fingerprint_db[fingerprint] = (fix, story_id)

    def lookup_fix(self, fingerprint: str) -> Optional[str]:
        """Look up a known fix for an error fingerprint.

        Args:
            fingerprint: Error fingerprint

        Returns:
            Fix code if found, None otherwise
        """
        if fingerprint in self.fingerprint_db:
            fix, _ = self.fingerprint_db[fingerprint]
            return fix
        return None

    def _get_level_1_context(self) -> dict:
        """First-level context: minimal, just the error."""
        return {
            "context_level": 1,
            "guidance": "Analyze the error message and provide a direct fix",
        }

    def _get_level_2_context(self) -> dict:
        """Second-level context: include type information."""
        return {
            "context_level": 2,
            "include_types": True,
            "include_symbols": True,
            "guidance": "Use type definitions and symbol information to diagnose the issue",
        }

    def _get_level_3_context(self) -> dict:
        """Third-level context: broad codebase patterns."""
        return {
            "context_level": 3,
            "include_types": True,
            "include_symbols": True,
            "include_patterns": True,
            "include_similar_code": True,
            "guidance": "Review similar patterns in the codebase to find the correct approach",
        }


class ErrorRecoveryCoordinator:
    """Coordinates error classification and recovery."""

    def __init__(self, classifier, recovery_manager: RecoveryManager):
        """Initialize coordinator.

        Args:
            classifier: ErrorClassifier instance
            recovery_manager: RecoveryManager instance
        """
        self.classifier = classifier
        self.recovery_manager = recovery_manager

    def handle_error(
        self,
        error_type: str,
        message: str,
        stack_trace: str = "",
        attempt: int = 1,
    ) -> tuple[ErrorClassification, RecoveryAction]:
        """Handle an error end-to-end.

        Args:
            error_type: Type of error
            message: Error message
            stack_trace: Full stack trace
            attempt: Current attempt number

        Returns:
            Tuple of (classification, recovery_action)
        """
        # Classify the error
        classification = self.classifier.classify(
            error_type=error_type,
            message=message,
            stack_trace=stack_trace,
        )

        # Check for known fix
        known_fix = self.recovery_manager.lookup_fix(classification.fingerprint)
        if known_fix:
            classification.confidence = 0.95  # Known fix = high confidence
            return classification, RecoveryAction(
                strategy=RecoveryStrategy.DIRECT_FIX,
                additional_context={"known_fix": known_fix},
            )

        # Get recovery strategy
        action = self.recovery_manager.get_strategy(classification, attempt)

        return classification, action
