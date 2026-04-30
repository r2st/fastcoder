"""Simple SQLite migration system for the fastcoder project.

Provides a lightweight, dependency-free migration framework that avoids the
overhead of Alembic or SQLAlchemy, keeping the project's footprint minimal.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Migration:
    """A single migration with up and down SQL."""

    version: int
    name: str
    up_sql: str
    down_sql: str


class MigrationManager:
    """Manages SQLite schema migrations."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the migration manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrations: dict[int, Migration] = {}
        self._init_migrations_table()

    def _connect(self) -> sqlite3.Connection:
        """Get a connection to the database."""
        return sqlite3.connect(str(self.db_path))

    def _init_migrations_table(self) -> None:
        """Create the migrations tracking table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    applied_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def register(self, migration: Migration) -> None:
        """Register a migration in the registry.

        Args:
            migration: The migration to register.

        Raises:
            ValueError: If a migration with the same version is already registered.
        """
        if migration.version in self._migrations:
            raise ValueError(f"Migration version {migration.version} already registered")
        self._migrations[migration.version] = migration
        logger.debug(
            "registered_migration",
            version=migration.version,
            name=migration.name,
        )

    def get_current_version(self) -> int:
        """Get the latest applied migration version.

        Returns:
            The version number of the latest migration, or 0 if none applied.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(version) FROM _migrations"
            ).fetchone()
            return row[0] or 0

    def get_pending(self) -> list[Migration]:
        """Get all unapplied migrations in order.

        Returns:
            List of pending migrations sorted by version.
        """
        current = self.get_current_version()
        pending = [m for v, m in sorted(self._migrations.items()) if v > current]
        return pending

    def get_history(self) -> list[dict]:
        """Get the complete migration history.

        Returns:
            List of applied migrations with metadata.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT version, name, applied_at FROM _migrations ORDER BY version"
            ).fetchall()
        return [
            {
                "version": row[0],
                "name": row[1],
                "applied_at": row[2],
            }
            for row in rows
        ]

    def migrate(self) -> None:
        """Apply all pending migrations in order within a transaction.

        Raises:
            RuntimeError: If a pending migration is not registered.
        """
        pending = self.get_pending()
        if not pending:
            logger.info("no_pending_migrations")
            return

        with self._connect() as conn:
            try:
                for migration in pending:
                    logger.info(
                        "applying_migration",
                        version=migration.version,
                        name=migration.name,
                    )
                    conn.executescript(migration.up_sql)
                    conn.execute(
                        "INSERT INTO _migrations (version, name, applied_at) VALUES (?, ?, ?)",
                        (migration.version, migration.name, datetime.utcnow().isoformat()),
                    )
                conn.commit()
                logger.info("migrations_applied", count=len(pending))
            except Exception as e:
                conn.rollback()
                logger.error("migration_failed", error=str(e))
                raise RuntimeError(f"Migration failed: {e}") from e

    def rollback(self, target_version: int) -> None:
        """Roll back migrations to a target version.

        Rolls back all migrations greater than target_version. The target version
        itself is NOT rolled back.

        Args:
            target_version: The version to roll back to (inclusive).

        Raises:
            RuntimeError: If a rollback fails.
        """
        current = self.get_current_version()
        if target_version >= current:
            logger.info("no_rollback_needed", target=target_version, current=current)
            return

        # Collect migrations to rollback in reverse order
        to_rollback = [
            m for v, m in sorted(self._migrations.items(), reverse=True)
            if v > target_version
        ]

        if not to_rollback:
            logger.info("no_migrations_to_rollback")
            return

        with self._connect() as conn:
            try:
                for migration in to_rollback:
                    logger.info(
                        "rolling_back_migration",
                        version=migration.version,
                        name=migration.name,
                    )
                    conn.executescript(migration.down_sql)
                    conn.execute(
                        "DELETE FROM _migrations WHERE version = ?",
                        (migration.version,),
                    )
                conn.commit()
                logger.info("migrations_rolled_back", count=len(to_rollback))
            except Exception as e:
                conn.rollback()
                logger.error("rollback_failed", error=str(e))
                raise RuntimeError(f"Rollback failed: {e}") from e


def create_initial_migration() -> Migration:
    """Create the initial migration that sets up the llm_provider_keys table.

    This migration mirrors the schema from LLMKeyStore._init_db().
    """
    return Migration(
        version=1,
        name="create_llm_provider_keys_table",
        up_sql="""
            CREATE TABLE IF NOT EXISTS llm_provider_keys (
                provider_name TEXT PRIMARY KEY,
                api_key TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """,
        down_sql="""
            DROP TABLE IF EXISTS llm_provider_keys
        """,
    )
