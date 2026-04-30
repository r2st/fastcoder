"""Approval gates for controlling risky operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import structlog

from fastcoder.types.config import SafetyConfig
from fastcoder.types.story import Story

logger = structlog.get_logger(__name__)


class GateStatus(str, Enum):
    """Status of a gate check."""

    PASS = "pass"
    PENDING = "pending"
    BLOCKED = "blocked"


@dataclass
class GateResult:
    """Result of a gate check."""

    gate_name: str
    status: GateStatus
    message: str = ""
    required_approval: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class PendingApproval:
    """Pending approval for a gate."""

    story_id: str
    gate_name: str
    requested_at: datetime = field(default_factory=datetime.utcnow)
    requested_by: str = "system"
    decision_reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ApprovalDecision:
    """Decision on a gate approval."""

    story_id: str
    gate_name: str
    approved: bool
    decided_at: datetime = field(default_factory=datetime.utcnow)
    decided_by: str = ""
    comment: str = ""


class ApprovalGateManager:
    """Manages approval gates for story progression."""

    VALID_GATES = {
        "pre_code",
        "pre_deploy",
        "pre_production",
        "budget_exceeded",
        "ambiguity_detected",
    }

    def __init__(self, safety_config: SafetyConfig):
        """
        Initialize approval gate manager.

        Args:
            safety_config: Safety configuration with gate settings
        """
        self.safety_config = safety_config
        self._pending_approvals: dict[tuple[str, str], PendingApproval] = {}
        self._decisions: dict[tuple[str, str], ApprovalDecision] = {}
        self._callbacks: list[Callable[[PendingApproval], None]] = []

    async def check_gate(self, gate_name: str, story: Story) -> GateResult:
        """
        Check if a gate passes.

        Args:
            gate_name: Name of gate to check
            story: Story to check

        Returns:
            Gate result with status
        """
        if gate_name not in self.VALID_GATES:
            raise ValueError(f"Unknown gate: {gate_name}")

        # Check if gate is enabled
        is_enabled = self.safety_config.approval_gates.get(gate_name, False)
        if not is_enabled:
            logger.info("gate_disabled", gate_name=gate_name, story_id=story.id)
            return GateResult(
                gate_name=gate_name,
                status=GateStatus.PASS,
                message="Gate is disabled",
            )

        # Check if already decided
        key = (story.id, gate_name)
        if key in self._decisions:
            decision = self._decisions[key]
            status = GateStatus.PASS if decision.approved else GateStatus.BLOCKED
            return GateResult(
                gate_name=gate_name,
                status=status,
                message=f"Previously decided: {decision.comment}",
                metadata={"decided_at": decision.decided_at.isoformat()},
            )

        # Check for pending approval
        if key in self._pending_approvals:
            return GateResult(
                gate_name=gate_name,
                status=GateStatus.PENDING,
                message="Awaiting approval",
                required_approval=True,
                metadata={"pending_since": self._pending_approvals[key].requested_at.isoformat()},
            )

        # Gate needs approval - create pending
        pending = PendingApproval(
            story_id=story.id,
            gate_name=gate_name,
            requested_by="orchestrator",
            metadata={"gate_enabled": is_enabled},
        )

        self._pending_approvals[key] = pending

        # Notify via callbacks
        for callback in self._callbacks:
            try:
                callback(pending)
            except Exception as e:
                logger.warning("approval_callback_error", error=str(e))

        logger.info(
            "approval_gate_pending",
            story_id=story.id,
            gate_name=gate_name,
        )

        return GateResult(
            gate_name=gate_name,
            status=GateStatus.PENDING,
            message="Approval required",
            required_approval=True,
        )

    def approve(
        self,
        story_id: str,
        gate_name: str,
        decided_by: str = "human",
        comment: str = "",
    ) -> None:
        """
        Approve a gate.

        Args:
            story_id: Story ID
            gate_name: Gate name
            decided_by: Who approved (e.g., "human", "automation")
            comment: Comment on approval
        """
        key = (story_id, gate_name)

        # Record decision
        decision = ApprovalDecision(
            story_id=story_id,
            gate_name=gate_name,
            approved=True,
            decided_by=decided_by,
            comment=comment,
        )
        self._decisions[key] = decision

        # Clear pending
        if key in self._pending_approvals:
            del self._pending_approvals[key]

        logger.info(
            "gate_approved",
            story_id=story_id,
            gate_name=gate_name,
            decided_by=decided_by,
        )

    def reject(
        self,
        story_id: str,
        gate_name: str,
        reason: str = "",
        decided_by: str = "human",
    ) -> None:
        """
        Reject a gate.

        Args:
            story_id: Story ID
            gate_name: Gate name
            reason: Reason for rejection
            decided_by: Who rejected
        """
        key = (story_id, gate_name)

        # Record decision
        decision = ApprovalDecision(
            story_id=story_id,
            gate_name=gate_name,
            approved=False,
            decided_by=decided_by,
            comment=reason,
        )
        self._decisions[key] = decision

        # Clear pending
        if key in self._pending_approvals:
            del self._pending_approvals[key]

        logger.info(
            "gate_rejected",
            story_id=story_id,
            gate_name=gate_name,
            reason=reason,
            decided_by=decided_by,
        )

    def get_pending_approvals(self, story_id: str) -> list[PendingApproval]:
        """Get all pending approvals for a story."""
        return [
            pending
            for (sid, _), pending in self._pending_approvals.items()
            if sid == story_id
        ]

    def register_callback(self, callback: Callable[[PendingApproval], None]) -> None:
        """Register callback for when approval is needed."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[PendingApproval], None]) -> None:
        """Unregister callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def has_pending_approvals(self, story_id: str) -> bool:
        """Check if story has any pending approvals."""
        return any(
            pending.story_id == story_id
            for pending in self._pending_approvals.values()
        )

    def get_decision(self, story_id: str, gate_name: str) -> Optional[ApprovalDecision]:
        """Get decision for a gate."""
        key = (story_id, gate_name)
        return self._decisions.get(key)

    def clear_pending(self, story_id: str, gate_name: str) -> None:
        """Clear pending approval for a gate."""
        key = (story_id, gate_name)
        if key in self._pending_approvals:
            del self._pending_approvals[key]

    def clear_all_for_story(self, story_id: str) -> None:
        """Clear all approvals (pending and decisions) for a story."""
        keys_to_remove = [
            key for key in self._pending_approvals
            if key[0] == story_id
        ]
        for key in keys_to_remove:
            del self._pending_approvals[key]

        keys_to_remove = [
            key for key in self._decisions
            if key[0] == story_id
        ]
        for key in keys_to_remove:
            del self._decisions[key]

        logger.info("approvals_cleared_for_story", story_id=story_id)
