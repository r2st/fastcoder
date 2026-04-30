"""Git client for version control operations."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

# Prevent GitPython from crashing import-time when git executable is misconfigured.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

try:
    from git import InvalidGitRepositoryError, Repo
except Exception as exc:  # pragma: no cover - environment-dependent fallback
    InvalidGitRepositoryError = Exception
    Repo = None
    _GITPYTHON_IMPORT_ERROR: Optional[Exception] = exc
else:
    _GITPYTHON_IMPORT_ERROR = None

from fastcoder.types.tools import SideEffects, ToolName, ToolResult


class GitClient:
    """Git operations wrapper around GitPython."""

    def __init__(self, project_dir: str):
        """Initialize git client."""
        self.project_dir = Path(project_dir).resolve()
        self.repo = None
        self._git_unavailable_reason: Optional[str] = None

        if _GITPYTHON_IMPORT_ERROR is not None:
            self._git_unavailable_reason = str(_GITPYTHON_IMPORT_ERROR)
            return

        try:
            self.repo = Repo(self.project_dir)
        except InvalidGitRepositoryError:
            self.repo = Repo.init(self.project_dir)
        except Exception as exc:
            self._git_unavailable_reason = str(exc)

    def _ensure_repo(self) -> None:
        if self.repo is not None:
            return

        reason = self._git_unavailable_reason or "unknown error"
        raise RuntimeError(
            "Git is unavailable in this environment. "
            f"Reason: {reason}. "
            "On macOS, run `sudo xcodebuild -license` once to enable /usr/bin/git."
        )

    def _slugify(self, text: str) -> str:
        """Convert text to slug format."""
        text = text.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[-\s]+", "-", text)
        return text.strip("-")

    def create_branch(self, story_id: str, description: str) -> str:
        """Create feature branch with conventional naming."""
        self._ensure_repo()
        try:
            slug = self._slugify(description)
            branch_name = f"feature/STORY-{story_id}/{slug}"

            self.repo.create_head(branch_name)
            return branch_name
        except Exception as e:
            raise RuntimeError(f"Failed to create branch: {e}")

    def commit_changes(
        self, message: str, files: Optional[list[str]] = None
    ) -> ToolResult:
        """Commit changes with conventional commit format."""
        start = time.time()
        try:
            self._ensure_repo()
            if files:
                self.repo.index.add(files)
            else:
                self.repo.index.add(A=True)

            if not self.repo.index.diff("HEAD"):
                return ToolResult(
                    tool=ToolName.GIT,
                    operation="commit_changes",
                    exit_code=0,
                    stdout="No changes to commit",
                    duration_ms=(time.time() - start) * 1000,
                )

            commit = self.repo.index.commit(message)
            modified = files if files else [f for f in self.repo.untracked_files]

            return ToolResult(
                tool=ToolName.GIT,
                operation="commit_changes",
                exit_code=0,
                stdout=f"Committed {commit.hexsha[:7]}: {message}",
                side_effects=SideEffects(files_modified=modified),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.GIT,
                operation="commit_changes",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def get_diff(self, base: Optional[str] = None) -> ToolResult:
        """Get diff between branches/commits."""
        start = time.time()
        try:
            self._ensure_repo()
            if base:
                diff_index = self.repo.commit(base).diff()
            else:
                diff_index = self.repo.head.commit.diff()

            diff_text = "\n".join(str(d) for d in diff_index)

            return ToolResult(
                tool=ToolName.GIT,
                operation="get_diff",
                exit_code=0,
                stdout=diff_text or "No differences",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.GIT,
                operation="get_diff",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def get_status(self) -> ToolResult:
        """Get repository status."""
        start = time.time()
        try:
            self._ensure_repo()
            status_dict = {
                "current_branch": self.repo.active_branch.name,
                "untracked_files": self.repo.untracked_files,
                "modified_files": [i[0] for i in self.repo.index.diff(None)],
                "staged_files": [i[0] for i in self.repo.index.diff("HEAD")],
                "is_dirty": self.repo.is_dirty(),
            }

            output = "\n".join(
                f"{k}: {v}" for k, v in status_dict.items()
            )

            return ToolResult(
                tool=ToolName.GIT,
                operation="get_status",
                exit_code=0,
                stdout=output,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.GIT,
                operation="get_status",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def get_log(self, count: int = 10) -> ToolResult:
        """Get recent commit log."""
        start = time.time()
        try:
            self._ensure_repo()
            commits = list(self.repo.iter_commits(max_count=count))
            lines = []
            for commit in commits:
                lines.append(
                    f"{commit.hexsha[:7]} - {commit.message.split(chr(10))[0]} ({commit.author})"
                )

            return ToolResult(
                tool=ToolName.GIT,
                operation="get_log",
                exit_code=0,
                stdout="\n".join(lines),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.GIT,
                operation="get_log",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def get_current_branch(self) -> str:
        """Get current branch name."""
        try:
            self._ensure_repo()
            return self.repo.active_branch.name
        except Exception:
            return "HEAD"

    def push(self, branch: Optional[str] = None, force: bool = False) -> ToolResult:
        """Push changes to remote."""
        start = time.time()
        try:
            self._ensure_repo()
            branch_name = branch or self.get_current_branch()

            if force and branch_name in ["main", "develop", "master"]:
                return ToolResult(
                    tool=ToolName.GIT,
                    operation="push",
                    exit_code=1,
                    stderr=f"Force push to {branch_name} is not allowed",
                    duration_ms=(time.time() - start) * 1000,
                )

            if not self.repo.remotes:
                return ToolResult(
                    tool=ToolName.GIT,
                    operation="push",
                    exit_code=1,
                    stderr="No remote configured",
                    duration_ms=(time.time() - start) * 1000,
                )

            origin = self.repo.remote("origin")
            push_kwargs = {"force": force} if force else {}
            origin.push(branch_name, **push_kwargs)

            return ToolResult(
                tool=ToolName.GIT,
                operation="push",
                exit_code=0,
                stdout=f"Pushed {branch_name}",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.GIT,
                operation="push",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def checkout(self, branch: str) -> ToolResult:
        """Checkout a branch."""
        start = time.time()
        try:
            self._ensure_repo()
            self.repo.heads[branch].checkout()

            return ToolResult(
                tool=ToolName.GIT,
                operation="checkout",
                exit_code=0,
                stdout=f"Checked out {branch}",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.GIT,
                operation="checkout",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
