"""Persistent task checkpointing for story execution recovery.

Provides checkpoint save/load functionality with atomic writes to enable
recovery from failures and resumption of interrupted stories.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import structlog

if TYPE_CHECKING:
    from fastcoder.types.story import Story, StoryState

logger = structlog.get_logger(__name__)


@dataclass
class StoryCheckpoint:
    """Snapshot of a story's execution state."""

    story_id: str
    state: str  # StoryState enum value
    iteration: int
    timestamp: datetime
    plan_snapshot: Optional[dict] = None
    error_context: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert checkpoint to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> StoryCheckpoint:
        """Create checkpoint from dictionary (reverse of to_dict)."""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)


class CheckpointManager:
    """Manages story execution checkpoints."""

    def __init__(self, checkpoint_dir: str = ".agent_checkpoints") -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints in. Defaults to .agent_checkpoints/.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("checkpoint_manager_initialized", checkpoint_dir=str(self.checkpoint_dir))

    def _get_checkpoint_path(self, story_id: str) -> Path:
        """Get the filesystem path for a story's checkpoint.

        Args:
            story_id: The story ID.

        Returns:
            Path to the checkpoint file.
        """
        # Sanitize story_id to prevent directory traversal
        safe_id = story_id.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"{safe_id}.checkpoint.json"

    def save_checkpoint(self, story: Story) -> None:
        """Save a story's current state to a checkpoint file.

        Uses atomic writes (temp file + rename) to ensure data integrity.

        Args:
            story: The story to checkpoint.
        """
        try:
            # Build checkpoint from story state
            plan_snapshot = None
            if story.plan:
                plan_snapshot = story.plan.model_dump()

            error_context = None
            if story.iterations:
                last_iter = story.iterations[-1]
                if last_iter.error_context:
                    error_context = last_iter.error_context.model_dump()

            checkpoint = StoryCheckpoint(
                story_id=story.id,
                state=story.state.value,
                iteration=len(story.iterations),
                timestamp=datetime.utcnow(),
                plan_snapshot=plan_snapshot,
                error_context=error_context,
                metadata={
                    "title": story.spec.title if story.spec else None,
                    "project_id": story.project_id,
                    "priority": story.priority.value,
                    "total_tokens_used": story.metadata.total_tokens_used,
                    "total_cost_usd": story.metadata.total_cost_usd,
                },
            )

            # Write atomically using temp file + rename
            checkpoint_path = self._get_checkpoint_path(story.id)
            checkpoint_data = json.dumps(checkpoint.to_dict(), indent=2, default=str)

            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(self.checkpoint_dir),
                delete=False,
                suffix=".tmp",
            ) as tmp_file:
                tmp_file.write(checkpoint_data)
                tmp_path = Path(tmp_file.name)

            # Atomic rename
            tmp_path.replace(checkpoint_path)

            logger.info(
                "checkpoint_saved",
                story_id=story.id,
                state=story.state.value,
                iteration=len(story.iterations),
            )

        except Exception as e:
            logger.error("checkpoint_save_failed", story_id=story.id, error=str(e))
            raise

    def load_checkpoint(self, story_id: str) -> Optional[StoryCheckpoint]:
        """Load a story's checkpoint from disk.

        Args:
            story_id: The story ID to load checkpoint for.

        Returns:
            StoryCheckpoint if found, None otherwise.
        """
        try:
            checkpoint_path = self._get_checkpoint_path(story_id)
            if not checkpoint_path.exists():
                logger.debug("checkpoint_not_found", story_id=story_id)
                return None

            with open(checkpoint_path) as f:
                data = json.load(f)

            checkpoint = StoryCheckpoint.from_dict(data)
            logger.info("checkpoint_loaded", story_id=story_id, state=checkpoint.state)
            return checkpoint

        except Exception as e:
            logger.error("checkpoint_load_failed", story_id=story_id, error=str(e))
            return None

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints with summaries.

        Returns:
            List of checkpoint summaries (dicts with key metadata).
        """
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("*.checkpoint.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)
                checkpoints.append({
                    "story_id": data["story_id"],
                    "state": data["state"],
                    "iteration": data["iteration"],
                    "timestamp": data["timestamp"],
                    "title": data.get("metadata", {}).get("title"),
                })
            except Exception as e:
                logger.warning(
                    "checkpoint_list_read_failed",
                    file=str(checkpoint_file),
                    error=str(e),
                )
        return sorted(checkpoints, key=lambda c: c["timestamp"], reverse=True)

    def recover_incomplete(self) -> list[StoryCheckpoint]:
        """Find all checkpoints for stories in non-terminal states.

        Terminal states are: DONE, FAILED.

        Returns:
            List of recoverable checkpoints.
        """
        recoverable = []
        terminal_states = {"DONE", "FAILED"}

        for checkpoint_file in self.checkpoint_dir.glob("*.checkpoint.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)

                if data["state"] not in terminal_states:
                    checkpoint = StoryCheckpoint.from_dict(data)
                    recoverable.append(checkpoint)
            except Exception as e:
                logger.warning(
                    "checkpoint_recovery_read_failed",
                    file=str(checkpoint_file),
                    error=str(e),
                )

        logger.info("incomplete_checkpoints_found", count=len(recoverable))
        return recoverable

    def delete_checkpoint(self, story_id: str) -> None:
        """Delete a checkpoint file.

        Args:
            story_id: The story ID to delete checkpoint for.
        """
        try:
            checkpoint_path = self._get_checkpoint_path(story_id)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("checkpoint_deleted", story_id=story_id)
            else:
                logger.debug("checkpoint_not_found_for_deletion", story_id=story_id)
        except Exception as e:
            logger.error("checkpoint_delete_failed", story_id=story_id, error=str(e))
            raise

    def cleanup_completed(self, max_age_hours: int = 24) -> int:
        """Remove checkpoints for completed stories older than N hours.

        Args:
            max_age_hours: Age threshold in hours. Defaults to 24.

        Returns:
            Number of checkpoints deleted.
        """
        deleted_count = 0
        threshold = datetime.utcnow().timestamp() - (max_age_hours * 3600)

        terminal_states = {"DONE", "FAILED"}

        for checkpoint_file in self.checkpoint_dir.glob("*.checkpoint.json"):
            try:
                if checkpoint_file.stat().st_mtime < threshold:
                    with open(checkpoint_file) as f:
                        data = json.load(f)

                    if data["state"] in terminal_states:
                        checkpoint_file.unlink()
                        deleted_count += 1
                        logger.debug(
                            "cleanup_deleted_checkpoint",
                            story_id=data["story_id"],
                            state=data["state"],
                        )
            except Exception as e:
                logger.warning(
                    "cleanup_failed_for_checkpoint",
                    file=str(checkpoint_file),
                    error=str(e),
                )

        logger.info("cleanup_completed", deleted_count=deleted_count, max_age_hours=max_age_hours)
        return deleted_count
