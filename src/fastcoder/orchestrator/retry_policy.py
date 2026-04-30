"""Retry policy with exponential backoff and circuit breaker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import structlog

from fastcoder.types.errors import ErrorContext
from fastcoder.types.story import StoryState

logger = structlog.get_logger(__name__)


@dataclass
class RetryAttempt:
    """Record of a retry attempt."""

    story_id: str
    stage: StoryState
    attempt_number: int
    success: bool
    error_fingerprint: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    backoff_seconds: float = 0.0


@dataclass
class CircuitBreakerState:
    """State of circuit breaker for a stage."""

    story_id: str
    stage: StoryState
    consecutive_failures: int = 0
    last_error_fingerprint: str = ""
    last_failure_time: Optional[datetime] = None
    is_broken: bool = False


class RetryPolicy:
    """Manages retries with exponential backoff and circuit breaker."""

    def __init__(
        self,
        max_retries_per_stage: int = 5,
        base_backoff_seconds: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        circuit_breaker_threshold: int = 3,
    ):
        """
        Initialize retry policy.

        Args:
            max_retries_per_stage: Maximum retry attempts per stage
            base_backoff_seconds: Base backoff duration
            backoff_factor: Exponential backoff multiplier
            max_backoff_seconds: Maximum backoff duration
            circuit_breaker_threshold: Consecutive failures before circuit opens
        """
        self.max_retries_per_stage = max_retries_per_stage
        self.base_backoff_seconds = base_backoff_seconds
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds
        self.circuit_breaker_threshold = circuit_breaker_threshold

        # Track attempts per (story_id, stage)
        self._attempts: dict[tuple[str, str], list[RetryAttempt]] = {}

        # Track circuit breaker state per (story_id, stage)
        self._circuit_breakers: dict[tuple[str, str], CircuitBreakerState] = {}

    def should_retry(
        self,
        story_id: str,
        stage: StoryState,
        error_context: Optional[ErrorContext] = None,
    ) -> bool:
        """
        Determine if we should retry the current stage.

        Args:
            story_id: Story ID
            stage: Current stage
            error_context: Error context with classification

        Returns:
            True if should retry, False otherwise
        """
        key = (story_id, stage.value)

        # Check circuit breaker first
        if self._is_circuit_broken(key):
            logger.warning(
                "circuit_breaker_open",
                story_id=story_id,
                stage=stage.value,
            )
            return False

        # Check max retries
        attempt_count = self.get_retry_count(story_id, stage)
        if attempt_count >= self.max_retries_per_stage:
            logger.info(
                "max_retries_exceeded",
                story_id=story_id,
                stage=stage.value,
                attempt_count=attempt_count,
            )
            return False

        return True

    def get_backoff_seconds(self, attempt: int) -> float:
        """
        Calculate backoff duration using exponential backoff.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Backoff duration in seconds
        """
        backoff = self.base_backoff_seconds * (self.backoff_factor ** attempt)
        return min(backoff, self.max_backoff_seconds)

    def record_attempt(
        self,
        story_id: str,
        stage: StoryState,
        success: bool,
        error_fingerprint: str = "",
    ) -> RetryAttempt:
        """
        Record a retry attempt.

        Args:
            story_id: Story ID
            stage: Current stage
            success: Whether attempt succeeded
            error_fingerprint: Error fingerprint if failed

        Returns:
            Recorded attempt
        """
        key = (story_id, stage.value)

        # Get current attempt count
        attempts = self._attempts.get(key, [])
        attempt_number = len(attempts)

        # Calculate backoff for next attempt
        backoff = self.get_backoff_seconds(attempt_number) if not success else 0.0

        # Record attempt
        attempt = RetryAttempt(
            story_id=story_id,
            stage=stage,
            attempt_number=attempt_number,
            success=success,
            error_fingerprint=error_fingerprint,
            backoff_seconds=backoff,
        )

        if key not in self._attempts:
            self._attempts[key] = []
        self._attempts[key].append(attempt)

        # Update circuit breaker
        self._update_circuit_breaker(key, success, error_fingerprint)

        logger.info(
            "retry_attempt_recorded",
            story_id=story_id,
            stage=stage.value,
            attempt_number=attempt_number,
            success=success,
            backoff_seconds=backoff,
        )

        return attempt

    def get_retry_count(self, story_id: str, stage: StoryState) -> int:
        """Get number of retry attempts for a stage."""
        key = (story_id, stage.value)
        return len(self._attempts.get(key, []))

    def get_attempts(self, story_id: str, stage: StoryState) -> list[RetryAttempt]:
        """Get all attempts for a stage."""
        key = (story_id, stage.value)
        return self._attempts.get(key, [])

    def reset(self, story_id: str, stage: StoryState) -> None:
        """Reset retry state for a stage."""
        key = (story_id, stage.value)
        if key in self._attempts:
            del self._attempts[key]
        if key in self._circuit_breakers:
            del self._circuit_breakers[key]
        logger.info("retry_state_reset", story_id=story_id, stage=stage.value)

    def _is_circuit_broken(self, key: tuple[str, str]) -> bool:
        """Check if circuit breaker is open."""
        if key not in self._circuit_breakers:
            return False

        breaker = self._circuit_breakers[key]
        return breaker.is_broken

    def _update_circuit_breaker(
        self,
        key: tuple[str, str],
        success: bool,
        error_fingerprint: str,
    ) -> None:
        """Update circuit breaker state."""
        if key not in self._circuit_breakers:
            self._circuit_breakers[key] = CircuitBreakerState(
                story_id=key[0],
                stage=StoryState(key[1]),
            )

        breaker = self._circuit_breakers[key]

        if success:
            # Reset on success
            breaker.consecutive_failures = 0
            breaker.is_broken = False
        else:
            # Increment failures
            breaker.consecutive_failures += 1
            breaker.last_error_fingerprint = error_fingerprint
            breaker.last_failure_time = datetime.utcnow()

            # Open circuit if same error repeats too many times
            if (
                breaker.consecutive_failures >= self.circuit_breaker_threshold
                and error_fingerprint == breaker.last_error_fingerprint
            ):
                breaker.is_broken = True
                logger.warning(
                    "circuit_breaker_opened",
                    story_id=key[0],
                    stage=key[1],
                    error_fingerprint=error_fingerprint,
                )

    def get_circuit_breaker_state(self, story_id: str, stage: StoryState) -> Optional[CircuitBreakerState]:
        """Get circuit breaker state."""
        key = (story_id, stage.value)
        return self._circuit_breakers.get(key)

    def clear_all(self) -> None:
        """Clear all retry state."""
        self._attempts.clear()
        self._circuit_breakers.clear()
        logger.info("all_retry_state_cleared")
