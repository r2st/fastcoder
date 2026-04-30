"""Timeout management for approval gates with escalation support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GateTimeout:
    """Tracked timeout for a pending approval gate."""

    story_id: str
    gate_name: str
    timeout_at: datetime
    escalation_action: str  # "auto_reject", "auto_approve", "escalate"
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_expired(self) -> bool:
        """Check if timeout has expired."""
        return datetime.utcnow() >= self.timeout_at

    @property
    def remaining(self) -> Optional[timedelta]:
        """
        Get time remaining before timeout.

        Returns:
            Remaining timedelta, or None if expired
        """
        now = datetime.utcnow()
        if now >= self.timeout_at:
            return None
        return self.timeout_at - now


class GateTimeoutManager:
    """Manages timeouts for approval gates with escalation."""

    # Escalation actions
    ESCALATION_AUTO_REJECT = "auto_reject"
    ESCALATION_AUTO_APPROVE = "auto_approve"
    ESCALATION_ESCALATE = "escalate"

    VALID_ESCALATION_ACTIONS = {
        ESCALATION_AUTO_REJECT,
        ESCALATION_AUTO_APPROVE,
        ESCALATION_ESCALATE,
    }

    def __init__(
        self,
        default_timeout_minutes: int = 60,
        escalation_action: str = "auto_reject",
    ):
        """
        Initialize timeout manager.

        Args:
            default_timeout_minutes: Default timeout in minutes
            escalation_action: Default escalation action on timeout

        Raises:
            ValueError: If escalation_action is invalid
        """
        if escalation_action not in self.VALID_ESCALATION_ACTIONS:
            raise ValueError(
                f"Invalid escalation_action: {escalation_action}. "
                f"Must be one of {self.VALID_ESCALATION_ACTIONS}"
            )

        self.default_timeout_minutes = default_timeout_minutes
        self.escalation_action = escalation_action
        self._timeouts: dict[tuple[str, str], GateTimeout] = {}

        logger.info(
            "GateTimeoutManager initialized",
            default_timeout_minutes=default_timeout_minutes,
            escalation_action=escalation_action,
        )

    def start_timeout(
        self,
        story_id: str,
        gate_name: str,
        timeout_minutes: Optional[int] = None,
    ) -> GateTimeout:
        """
        Register a timeout for a pending gate.

        Args:
            story_id: Story ID
            gate_name: Name of gate
            timeout_minutes: Override default timeout (optional)

        Returns:
            GateTimeout instance
        """
        minutes = timeout_minutes or self.default_timeout_minutes
        timeout_at = datetime.utcnow() + timedelta(minutes=minutes)

        timeout = GateTimeout(
            story_id=story_id,
            gate_name=gate_name,
            timeout_at=timeout_at,
            escalation_action=self.escalation_action,
        )

        key = (story_id, gate_name)
        self._timeouts[key] = timeout

        logger.info(
            "Gate timeout started",
            story_id=story_id,
            gate_name=gate_name,
            timeout_minutes=minutes,
            escalation_action=self.escalation_action,
        )

        return timeout

    def check_timeouts(self) -> list[tuple[str, str]]:
        """
        Check for expired timeouts.

        Returns:
            List of (story_id, gate_name) tuples that have expired
        """
        expired = []

        for key, timeout in list(self._timeouts.items()):
            if timeout.is_expired:
                expired.append(key)
                logger.warning(
                    "Gate timeout expired",
                    story_id=timeout.story_id,
                    gate_name=timeout.gate_name,
                    escalation_action=timeout.escalation_action,
                )

        return expired

    def cancel_timeout(self, story_id: str, gate_name: str) -> bool:
        """
        Cancel a tracked timeout.

        Args:
            story_id: Story ID
            gate_name: Gate name

        Returns:
            True if timeout was found and removed
        """
        key = (story_id, gate_name)
        if key in self._timeouts:
            del self._timeouts[key]
            logger.info(
                "Gate timeout cancelled",
                story_id=story_id,
                gate_name=gate_name,
            )
            return True
        return False

    def get_remaining(
        self, story_id: str, gate_name: str
    ) -> Optional[timedelta]:
        """
        Get time remaining before timeout.

        Args:
            story_id: Story ID
            gate_name: Gate name

        Returns:
            timedelta remaining, or None if not found or expired
        """
        key = (story_id, gate_name)
        timeout = self._timeouts.get(key)
        if timeout:
            return timeout.remaining
        return None

    def get_timeout(
        self, story_id: str, gate_name: str
    ) -> Optional[GateTimeout]:
        """
        Get timeout object for a gate.

        Args:
            story_id: Story ID
            gate_name: Gate name

        Returns:
            GateTimeout instance, or None if not found
        """
        key = (story_id, gate_name)
        return self._timeouts.get(key)

    def get_escalation_action(self, story_id: str, gate_name: str) -> str:
        """
        Get escalation action for an expired timeout.

        Args:
            story_id: Story ID
            gate_name: Gate name

        Returns:
            Escalation action string
        """
        timeout = self.get_timeout(story_id, gate_name)
        if timeout:
            return timeout.escalation_action
        return self.escalation_action

    def get_all_timeouts(self) -> list[GateTimeout]:
        """
        Get all active timeouts.

        Returns:
            List of all GateTimeout instances
        """
        return list(self._timeouts.values())

    def get_story_timeouts(self, story_id: str) -> list[GateTimeout]:
        """
        Get all timeouts for a specific story.

        Args:
            story_id: Story ID

        Returns:
            List of GateTimeout instances for the story
        """
        return [
            timeout
            for timeout in self._timeouts.values()
            if timeout.story_id == story_id
        ]

    def clear_story_timeouts(self, story_id: str) -> int:
        """
        Cancel all timeouts for a specific story.

        Args:
            story_id: Story ID

        Returns:
            Number of timeouts removed
        """
        keys_to_remove = [
            key for key in self._timeouts.keys()
            if key[0] == story_id
        ]

        for key in keys_to_remove:
            del self._timeouts[key]

        if keys_to_remove:
            logger.info(
                "Story timeouts cleared",
                story_id=story_id,
                count=len(keys_to_remove),
            )

        return len(keys_to_remove)
