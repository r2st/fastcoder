"""Convergence detection to identify progress stalls and oscillations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from fastcoder.types.story import Story

logger = structlog.get_logger(__name__)


@dataclass
class ConvergenceResult:
    """Result of convergence check."""

    status: str  # "progressing" | "oscillating" | "stuck" | "diverging"
    confidence: float  # 0.0 to 1.0
    reason: str
    recommended_action: str  # "continue" | "enrich_context" | "replan" | "escalate"
    metrics: dict = None  # Additional diagnostic metrics


class ConvergenceDetector:
    """Detects progress stalls, oscillations, and divergence in story iterations."""

    def __init__(self, window_size: int = 4, stuck_threshold: int = 3):
        """
        Initialize convergence detector.

        Args:
            window_size: Number of recent iterations to analyze
            stuck_threshold: Iterations without progress before "stuck"
        """
        self.window_size = window_size
        self.stuck_threshold = stuck_threshold

    def check(self, story: Story) -> ConvergenceResult:
        """
        Check convergence status of a story.

        Args:
            story: Story with iteration history

        Returns:
            Convergence result with status, confidence, and action
        """
        if not story.iterations:
            return ConvergenceResult(
                status="progressing",
                confidence=1.0,
                reason="No iterations yet",
                recommended_action="continue",
                metrics={},
            )

        # Analyze recent iterations
        recent = story.iterations[-self.window_size :]

        # Collect error fingerprints from recent iterations
        error_fingerprints = []
        error_counts = []

        for iteration in recent:
            if iteration.error_context:
                fingerprint = iteration.error_context.classification.fingerprint
                if fingerprint:
                    error_fingerprints.append(fingerprint)
                error_counts.append(1)
            else:
                error_counts.append(0)

        # Check for oscillation (same error appearing 2+ times in recent iterations)
        oscillation_result = self._check_oscillation(error_fingerprints)
        if oscillation_result:
            return oscillation_result

        # Check for stuck (no progress for 3+ iterations)
        stuck_result = self._check_stuck(error_counts)
        if stuck_result:
            return stuck_result

        # Check for diverging (error count increasing)
        diverging_result = self._check_diverging(error_counts)
        if diverging_result:
            return diverging_result

        # Default: progressing
        return ConvergenceResult(
            status="progressing",
            confidence=0.8,
            reason="Story progressing toward completion",
            recommended_action="continue",
            metrics={
                "recent_errors": len(error_fingerprints),
                "iterations_analyzed": len(recent),
            },
        )

    def _check_oscillation(self, error_fingerprints: list[str]) -> Optional[ConvergenceResult]:
        """Check if same error is recurring (oscillation)."""
        if len(error_fingerprints) < 2:
            return None

        # Count occurrences of each fingerprint
        fingerprint_counts = {}
        for fp in error_fingerprints:
            fingerprint_counts[fp] = fingerprint_counts.get(fp, 0) + 1

        # If any fingerprint appears 2+ times, it's oscillating
        for fp, count in fingerprint_counts.items():
            if count >= 2:
                logger.warning(
                    "oscillation_detected",
                    error_fingerprint=fp,
                    occurrences=count,
                )
                return ConvergenceResult(
                    status="oscillating",
                    confidence=0.85 + min(0.15, count * 0.05),  # Higher confidence with more repeats
                    reason=f"Error '{fp}' appeared {count} times in recent iterations",
                    recommended_action="enrich_context",
                    metrics={
                        "oscillating_fingerprint": fp,
                        "occurrence_count": count,
                    },
                )

        return None

    def _check_stuck(self, error_counts: list[int]) -> Optional[ConvergenceResult]:
        """Check if story is stuck (no progress for N iterations)."""
        if len(error_counts) < self.stuck_threshold:
            return None

        # Check if last N iterations all had errors
        recent_errors = error_counts[-self.stuck_threshold :]
        if all(recent_errors):
            logger.warning(
                "story_stuck",
                error_iterations=self.stuck_threshold,
            )
            return ConvergenceResult(
                status="stuck",
                confidence=0.9,
                reason=f"Story failed to make progress for {self.stuck_threshold} consecutive iterations",
                recommended_action="replan",
                metrics={
                    "stuck_iterations": self.stuck_threshold,
                },
            )

        return None

    def _check_diverging(self, error_counts: list[int]) -> Optional[ConvergenceResult]:
        """Check if error count is increasing (diverging)."""
        if len(error_counts) < 2:
            return None

        # Compare first half vs second half
        mid = len(error_counts) // 2
        first_half = error_counts[:mid]
        second_half = error_counts[mid:]

        if not first_half or not second_half:
            return None

        first_half_errors = sum(first_half)
        second_half_errors = sum(second_half)

        # If error count is increasing, diverging
        if second_half_errors > first_half_errors:
            logger.warning(
                "story_diverging",
                first_half_errors=first_half_errors,
                second_half_errors=second_half_errors,
            )
            return ConvergenceResult(
                status="diverging",
                confidence=0.7,
                reason=f"Error count increased from {first_half_errors} to {second_half_errors}",
                recommended_action="escalate",
                metrics={
                    "first_half_errors": first_half_errors,
                    "second_half_errors": second_half_errors,
                    "error_trend": "increasing",
                },
            )

        return None
