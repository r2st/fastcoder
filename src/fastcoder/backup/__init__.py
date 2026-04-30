"""Backup manager for periodic state file backups with atomic operations."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class BackupEntry:
    """Metadata about a backup."""

    def __init__(self, timestamp: datetime, file_sizes: dict[str, int]):
        """
        Initialize backup entry.

        Args:
            timestamp: When backup was created
            file_sizes: Dict of filename -> size in bytes
        """
        self.timestamp = timestamp
        self.file_sizes = file_sizes

    @property
    def total_size_mb(self) -> float:
        """Get total backup size in MB."""
        return sum(self.file_sizes.values()) / (1024 * 1024)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_size_mb": self.total_size_mb,
            "files": self.file_sizes,
        }


class BackupManager:
    """Manages periodic backups of agent state files."""

    # State files to backup
    STATE_FILES = {
        ".agent_memory.json",
        ".agent_learnings.json",
        ".agent_cross_repo_index.json",
        ".agent_admin.db",
        ".agent.json",
    }

    def __init__(self, project_dir: str, backup_dir: str):
        """
        Initialize backup manager.

        Args:
            project_dir: Root project directory containing state files
            backup_dir: Directory to store backups
        """
        self.project_dir = Path(project_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._backup_task: Optional[asyncio.Task] = None
        logger.info(
            "BackupManager initialized",
            project_dir=str(self.project_dir),
            backup_dir=str(self.backup_dir),
        )

    def create_backup(self) -> datetime:
        """
        Create a backup of all state files.

        Copies state files to backup_dir with timestamp suffix using atomic operations
        (write to temp, then rename).

        Returns:
            Timestamp of backup created
        """
        backup_time = datetime.utcnow()
        timestamp_str = backup_time.strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / timestamp_str
        backup_subdir.mkdir(parents=True, exist_ok=True)

        file_sizes = {}
        files_backed_up = []

        try:
            for state_file in self.STATE_FILES:
                src_path = self.project_dir / state_file
                if not src_path.exists():
                    continue

                dest_path = backup_subdir / state_file
                temp_path = dest_path.with_suffix(".tmp")

                try:
                    # Atomic copy: write to temp then rename
                    shutil.copy2(src_path, temp_path)
                    temp_path.rename(dest_path)

                    file_sizes[state_file] = dest_path.stat().st_size
                    files_backed_up.append(state_file)
                except Exception as e:
                    logger.error(
                        "Failed to backup file",
                        file=state_file,
                        error=str(e),
                    )
                    # Clean up temp file if it exists
                    if temp_path.exists():
                        temp_path.unlink()

            if files_backed_up:
                logger.info(
                    "Backup created successfully",
                    timestamp=timestamp_str,
                    files_count=len(files_backed_up),
                    total_size_mb=sum(file_sizes.values()) / (1024 * 1024),
                )
            else:
                logger.warning("No state files found to backup")
                # Remove empty backup directory
                backup_subdir.rmdir()

            return backup_time

        except Exception as e:
            logger.error("Backup creation failed", error=str(e))
            # Clean up incomplete backup
            if backup_subdir.exists() and not any(backup_subdir.iterdir()):
                backup_subdir.rmdir()
            raise

    def list_backups(self) -> list[BackupEntry]:
        """
        List all available backups.

        Returns:
            List of BackupEntry objects sorted by timestamp (newest first)
        """
        backups = []

        if not self.backup_dir.exists():
            return backups

        for item in self.backup_dir.iterdir():
            if not item.is_dir():
                continue

            try:
                timestamp = datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                file_sizes = {}

                for file_path in item.iterdir():
                    if file_path.is_file():
                        file_sizes[file_path.name] = file_path.stat().st_size

                backups.append(BackupEntry(timestamp, file_sizes))
            except (ValueError, OSError):
                logger.warning("Skipping invalid backup directory", dir=item.name)

        # Sort by timestamp, newest first
        backups.sort(key=lambda b: b.timestamp, reverse=True)
        return backups

    def restore_backup(self, timestamp: datetime) -> int:
        """
        Restore a specific backup by copying files back to project directory.

        Args:
            timestamp: Timestamp of backup to restore

        Returns:
            Number of files restored

        Raises:
            FileNotFoundError: If backup not found
        """
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / timestamp_str

        if not backup_subdir.exists():
            raise FileNotFoundError(f"Backup not found: {timestamp_str}")

        files_restored = 0

        try:
            for backup_file in backup_subdir.iterdir():
                if not backup_file.is_file():
                    continue

                dest_path = self.project_dir / backup_file.name
                temp_path = dest_path.with_suffix(".restore_tmp")

                try:
                    # Atomic restore: copy to temp, then rename
                    shutil.copy2(backup_file, temp_path)
                    temp_path.replace(dest_path)
                    files_restored += 1
                except Exception as e:
                    logger.error(
                        "Failed to restore file",
                        file=backup_file.name,
                        error=str(e),
                    )
                    if temp_path.exists():
                        temp_path.unlink()

            logger.info(
                "Backup restored",
                timestamp=timestamp_str,
                files_restored=files_restored,
            )
            return files_restored

        except Exception as e:
            logger.error("Backup restore failed", timestamp=timestamp_str, error=str(e))
            raise

    def cleanup_old_backups(self, max_count: int = 10) -> int:
        """
        Remove old backups, keeping only the N most recent.

        Args:
            max_count: Maximum number of backups to keep

        Returns:
            Number of backups deleted
        """
        backups = self.list_backups()

        if len(backups) <= max_count:
            return 0

        backups_to_delete = backups[max_count:]
        deleted_count = 0

        for backup in backups_to_delete:
            backup_subdir = self.backup_dir / backup.timestamp.strftime("%Y%m%d_%H%M%S")
            try:
                shutil.rmtree(backup_subdir)
                deleted_count += 1
                logger.info("Backup deleted", timestamp=backup.timestamp.isoformat())
            except Exception as e:
                logger.error(
                    "Failed to delete backup",
                    timestamp=backup.timestamp.isoformat(),
                    error=str(e),
                )

        logger.info("Cleanup completed", deleted_count=deleted_count)
        return deleted_count

    def schedule_periodic_backup(self, interval_minutes: int = 30) -> asyncio.Task:
        """
        Start periodic backup task that runs in background.

        Creates an asyncio task that runs backups at specified interval.

        Args:
            interval_minutes: Minutes between backups

        Returns:
            asyncio.Task for the periodic backup loop
        """
        async def backup_loop():
            """Run backups at regular intervals."""
            while True:
                try:
                    await asyncio.sleep(interval_minutes * 60)
                    self.create_backup()
                    # Cleanup old backups after creating new one
                    self.cleanup_old_backups(max_count=10)
                except asyncio.CancelledError:
                    logger.info("Periodic backup task cancelled")
                    break
                except Exception as e:
                    logger.error(
                        "Periodic backup failed",
                        error=str(e),
                        interval_minutes=interval_minutes,
                    )
                    # Continue trying on next interval

        if self._backup_task and not self._backup_task.done():
            logger.warning("Backup task already running")
            return self._backup_task

        self._backup_task = asyncio.create_task(backup_loop())
        logger.info("Periodic backup scheduled", interval_minutes=interval_minutes)
        return self._backup_task

    def cancel_periodic_backup(self) -> None:
        """Cancel the periodic backup task if running."""
        if self._backup_task and not self._backup_task.done():
            self._backup_task.cancel()
            logger.info("Periodic backup task cancelled")
