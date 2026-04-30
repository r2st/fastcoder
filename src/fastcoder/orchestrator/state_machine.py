"""State machine for managing story lifecycle transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import structlog

from fastcoder.types.story import Story, StoryState

logger = structlog.get_logger(__name__)


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: StoryState
    to_state: StoryState
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""
    metadata: dict = field(default_factory=dict)


class StateMachine:
    """Manages story state transitions with validation and history tracking."""

    def __init__(self):
        """Initialize the state machine with valid transitions."""
        self._valid_transitions: dict[StoryState, list[StoryState]] = {
            StoryState.RECEIVED: [StoryState.ANALYZING],
            StoryState.ANALYZING: [StoryState.PLANNING, StoryState.FAILED],
            StoryState.PLANNING: [StoryState.CODING, StoryState.FAILED],
            StoryState.CODING: [StoryState.REVIEWING, StoryState.FAILED],
            StoryState.REVIEWING: [StoryState.CODING, StoryState.TESTING, StoryState.FAILED],
            StoryState.TESTING: [StoryState.CODING, StoryState.REVIEWING, StoryState.DEPLOYING, StoryState.FAILED],
            StoryState.DEPLOYING: [StoryState.VERIFYING, StoryState.CODING, StoryState.FAILED],
            StoryState.VERIFYING: [StoryState.CODING, StoryState.DONE, StoryState.FAILED],
            StoryState.DONE: [],
            StoryState.FAILED: [],
        }

        self._transition_history: dict[str, list[StateTransition]] = {}
        self._callbacks: list[Callable[[Story, StateTransition], None]] = []

    def is_valid_transition(self, from_state: StoryState, to_state: StoryState) -> bool:
        """Check if transition is valid."""
        return to_state in self._valid_transitions.get(from_state, [])

    def transition(
        self,
        story: Story,
        target_state: StoryState,
        reason: str = "",
        metadata: dict | None = None,
    ) -> Story:
        """
        Execute a state transition.

        Args:
            story: The story to transition
            target_state: Target state
            reason: Reason for transition
            metadata: Additional context

        Returns:
            Updated story with new state

        Raises:
            ValueError: If transition is invalid
        """
        if not self.is_valid_transition(story.state, target_state):
            raise ValueError(
                f"Invalid transition from {story.state} to {target_state}. "
                f"Valid targets: {self._valid_transitions[story.state]}"
            )

        # Create transition record
        transition = StateTransition(
            from_state=story.state,
            to_state=target_state,
            reason=reason,
            metadata=metadata or {},
        )

        # Update story
        old_state = story.state
        story.state = target_state
        story.metadata.updated_at = datetime.utcnow()

        # Track history
        if story.id not in self._transition_history:
            self._transition_history[story.id] = []
        self._transition_history[story.id].append(transition)

        # Execute callbacks
        for callback in self._callbacks:
            try:
                callback(story, transition)
            except Exception as e:
                logger.warning("callback_error", callback=callback, error=str(e))

        logger.info(
            "state_transition",
            story_id=story.id,
            from_state=old_state,
            to_state=target_state,
            reason=reason,
        )

        return story

    def register_callback(self, callback: Callable[[Story, StateTransition], None]) -> None:
        """Register a callback to be called on every transition."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[Story, StateTransition], None]) -> None:
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_history(self, story_id: str) -> list[StateTransition]:
        """Get transition history for a story."""
        return self._transition_history.get(story_id, [])

    def get_current_state(self, story: Story) -> StoryState:
        """Get current state of story."""
        return story.state

    def get_valid_next_states(self, story: Story) -> list[StoryState]:
        """Get list of valid next states."""
        return self._valid_transitions.get(story.state, [])

    def can_rollback_to(self, story: Story, target_state: StoryState) -> bool:
        """
        Check if story can rollback to target state.

        Valid rollbacks: TESTING→CODING, REVIEWING→CODING, VERIFYING→CODING, any→FAILED
        """
        valid_rollbacks = {
            StoryState.TESTING: [StoryState.CODING],
            StoryState.REVIEWING: [StoryState.CODING],
            StoryState.VERIFYING: [StoryState.CODING],
        }

        # Any state can transition to FAILED
        if target_state == StoryState.FAILED:
            return True

        return target_state in valid_rollbacks.get(story.state, [])

    def rollback(
        self,
        story: Story,
        target_state: StoryState,
        reason: str = "",
    ) -> Story:
        """
        Rollback to a previous state.

        Args:
            story: The story to rollback
            target_state: Target state for rollback
            reason: Reason for rollback

        Returns:
            Updated story

        Raises:
            ValueError: If rollback is invalid
        """
        if not self.can_rollback_to(story, target_state):
            raise ValueError(
                f"Cannot rollback from {story.state} to {target_state}"
            )

        return self.transition(story, target_state, reason=f"Rollback: {reason}")
