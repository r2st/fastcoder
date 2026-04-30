"""Persistent LLM API key storage for admin-managed configuration."""

from __future__ import annotations

import os
import re
import sqlite3
import stat
from pathlib import Path
from typing import Optional

# Whitelist of valid provider names to prevent injection via URL path params
_VALID_PROVIDERS = frozenset({"anthropic", "openai", "google", "ollama"})


def _validate_provider_name(name: str) -> str:
    """Validate and return provider name, or raise ValueError."""
    if name not in _VALID_PROVIDERS:
        raise ValueError(
            f"Invalid provider name: {name!r}. "
            f"Must be one of: {', '.join(sorted(_VALID_PROVIDERS))}"
        )
    return name


class LLMKeyStore:
    """Simple SQLite-backed storage for provider API keys."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # Restrict file permissions to owner-only (rw-------)
        self._secure_db_permissions()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _secure_db_permissions(self) -> None:
        """Restrict database file permissions to owner read/write only."""
        try:
            if self.db_path.exists():
                os.chmod(str(self.db_path), stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass  # Best-effort on platforms that don't support chmod

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_provider_keys (
                    provider_name TEXT PRIMARY KEY,
                    api_key TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_key(self, provider_name: str) -> Optional[str]:
        _validate_provider_name(provider_name)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT api_key FROM llm_provider_keys WHERE provider_name = ?",
                (provider_name,),
            ).fetchone()
            return row[0] if row else None

    def get_all_keys(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT provider_name, api_key FROM llm_provider_keys"
            ).fetchall()
        return {name: api_key for name, api_key in rows}

    def set_key(self, provider_name: str, api_key: str) -> None:
        _validate_provider_name(provider_name)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_provider_keys(provider_name, api_key, updated_at)
                VALUES(?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(provider_name)
                DO UPDATE SET
                    api_key = excluded.api_key,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (provider_name, api_key),
            )
            conn.commit()

    def clear_key(self, provider_name: str) -> None:
        _validate_provider_name(provider_name)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM llm_provider_keys WHERE provider_name = ?",
                (provider_name,),
            )
            conn.commit()


def resolve_admin_db_path(project_dir: Optional[str] = None) -> Path:
    """Resolve admin DB path from env or project directory."""
    explicit = os.getenv("AGENT_ADMIN_DB_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()

    base_dir = Path(project_dir or ".").expanduser().resolve()
    return base_dir / ".agent_admin.db"


def get_llm_key_store(project_dir: Optional[str] = None) -> LLMKeyStore:
    """Create a key store instance using resolved admin DB path."""
    return LLMKeyStore(resolve_admin_db_path(project_dir))
