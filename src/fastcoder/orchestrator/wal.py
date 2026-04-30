"""Write-Ahead Log (WAL) for story state transitions with integrity verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from fastcoder.types.story import StoryState

logger = structlog.get_logger(__name__)


@dataclass
class WALEntry:
    """Single write-ahead log entry for story state transition."""

    story_id: str
    timestamp: datetime
    state_from: StoryState
    state_to: StoryState
    metadata: dict = field(default_factory=dict)
    checksum: str = ""

    def calculate_checksum(self) -> str:
        """
        Calculate SHA-256 checksum for integrity verification.

        Returns:
            Hex digest of SHA-256 hash
        """
        data = (
            f"{self.story_id}|"
            f"{self.timestamp.isoformat()}|"
            f"{self.state_from.value}|"
            f"{self.state_to.value}|"
            f"{json.dumps(self.metadata, sort_keys=True)}"
        ).encode()
        return hashlib.sha256(data).hexdigest()

    def verify_checksum(self) -> bool:
        """
        Verify checksum integrity.

        Returns:
            True if stored checksum matches calculated
        """
        return self.checksum == self.calculate_checksum()

    def to_json_line(self) -> str:
        """
        Serialize entry to JSON line format.

        Returns:
            JSON string representation
        """
        return json.dumps({
            "story_id": self.story_id,
            "timestamp": self.timestamp.isoformat(),
            "state_from": self.state_from.value,
            "state_to": self.state_to.value,
            "metadata": self.metadata,
            "checksum": self.checksum,
        })

    @staticmethod
    def from_json_dict(data: dict) -> WALEntry:
        """
        Deserialize entry from JSON dictionary.

        Args:
            data: Dictionary from JSON

        Returns:
            WALEntry instance
        """
        return WALEntry(
            story_id=data["story_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state_from=StoryState(data["state_from"]),
            state_to=StoryState(data["state_to"]),
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )


class WriteAheadLog:
    """Persistent write-ahead log for story state transitions."""

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize write-ahead log.

        Args:
            file_path: Path to WAL file (default: .agent_wal.jsonl)
        """
        self.file_path = Path(file_path or ".agent_wal.jsonl")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointed_stories = set()
        logger.info("WAL initialized", path=str(self.file_path))

    def append(self, entry: WALEntry) -> None:
        """
        Append entry to WAL file with fsync.

        Args:
            entry: WALEntry to append

        Raises:
            IOError: If write fails
        """
        try:
            # Calculate checksum before writing
            entry.checksum = entry.calculate_checksum()

            # Append to file
            with open(self.file_path, "a") as f:
                f.write(entry.to_json_line() + "\n")
                f.flush()
                # Force sync to disk
                import os
                os.fsync(f.fileno())

            logger.debug(
                "WAL entry appended",
                story_id=entry.story_id,
                state_from=entry.state_from.value,
                state_to=entry.state_to.value,
            )

        except IOError as e:
            logger.error("Failed to append WAL entry", error=str(e))
            raise

    def recover(self) -> list[WALEntry]:
        """
        Recover incomplete stories from WAL.

        Reads WAL and returns entries for stories that haven't reached terminal state
        (DONE or FAILED). Handles corrupt/truncated lines gracefully.

        Returns:
            List of WALEntry objects for incomplete stories
        """
        if not self.file_path.exists():
            return []

        all_entries: dict[str, list[WALEntry]] = {}
        line_num = 0

        try:
            with open(self.file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        entry = WALEntry.from_json_dict(data)

                        # Verify checksum
                        if not entry.verify_checksum():
                            logger.warning(
                                "WAL entry has invalid checksum",
                                line=line_num,
                                story_id=entry.story_id,
                            )
                            continue

                        if entry.story_id not in all_entries:
                            all_entries[entry.story_id] = []
                        all_entries[entry.story_id].append(entry)

                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(
                            "Skipping corrupt WAL line",
                            line=line_num,
                            error=str(e),
                        )
                        continue

        except IOError as e:
            logger.error("Error reading WAL file", error=str(e))
            return []

        # Filter to incomplete stories
        incomplete_entries = []
        for story_id, entries in all_entries.items():
            if entries:
                last_entry = entries[-1]
                # If not in terminal state, include all entries for this story
                if last_entry.state_to not in (StoryState.DONE, StoryState.FAILED):
                    incomplete_entries.extend(entries)

        logger.info(
            "WAL recovery completed",
            incomplete_stories=len(incomplete_entries),
            total_stories=len(all_entries),
        )
        return incomplete_entries

    def checkpoint(self, story_id: str) -> None:
        """
        Mark story as checkpointed (completed).

        Args:
            story_id: Story ID to checkpoint
        """
        self._checkpointed_stories.add(story_id)
        logger.debug("Story checkpointed", story_id=story_id)

    def truncate(self) -> int:
        """
        Remove checkpointed entries to keep WAL small.

        Returns:
            Number of entries removed
        """
        if not self.file_path.exists():
            return 0

        remaining_entries = []
        removed_count = 0

        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        entry = WALEntry.from_json_dict(data)

                        # Keep if not checkpointed
                        if entry.story_id not in self._checkpointed_stories:
                            remaining_entries.append(line)
                        else:
                            removed_count += 1

                    except (json.JSONDecodeError, ValueError):
                        # Keep lines that can't be parsed
                        remaining_entries.append(line)

            # Rewrite file with remaining entries
            if removed_count > 0:
                with open(self.file_path, "w") as f:
                    for entry_line in remaining_entries:
                        f.write(entry_line + "\n")
                    # Sync to disk
                    import os
                    os.fsync(f.fileno())

                logger.info("WAL truncated", removed_count=removed_count)

        except IOError as e:
            logger.error("Failed to truncate WAL", error=str(e))

        return removed_count

    def get_entries(self, story_id: str) -> list[WALEntry]:
        """
        Get all WAL entries for a specific story.

        Args:
            story_id: Story ID to retrieve entries for

        Returns:
            List of WALEntry objects for the story
        """
        if not self.file_path.exists():
            return []

        entries = []

        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        entry = WALEntry.from_json_dict(data)

                        if entry.story_id == story_id:
                            # Verify checksum
                            if entry.verify_checksum():
                                entries.append(entry)
                            else:
                                logger.warning(
                                    "Skipping entry with invalid checksum",
                                    story_id=story_id,
                                )

                    except (json.JSONDecodeError, ValueError):
                        continue

        except IOError as e:
            logger.error("Error reading WAL entries", story_id=story_id, error=str(e))

        return entries
