"""File system operations tool."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import aiofiles
import aiofiles.os

from fastcoder.types.tools import SideEffects, ToolName, ToolResult


class FileSystemTool:
    """Async file system operations with path validation."""

    def __init__(self, project_dir: str):
        """Initialize with project directory for security validation."""
        self.project_dir = Path(project_dir).resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, path: str) -> Path:
        """Validate path doesn't escape project directory."""
        target = (self.project_dir / path).resolve()
        if not str(target).startswith(str(self.project_dir)):
            raise ValueError(f"Path escape attempt: {path}")
        return target

    async def read_file(self, path: str) -> ToolResult:
        """Read file contents."""
        start = time.time()
        try:
            target = self._validate_path(path)
            if not target.exists():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="read_file",
                    exit_code=1,
                    stderr=f"File not found: {path}",
                    duration_ms=(time.time() - start) * 1000,
                )

            async with aiofiles.open(target, "r") as f:
                content = await f.read()

            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="read_file",
                exit_code=0,
                stdout=content,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="read_file",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def write_file(self, path: str, content: str) -> ToolResult:
        """Write to existing file."""
        start = time.time()
        try:
            target = self._validate_path(path)
            if not target.exists():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="write_file",
                    exit_code=1,
                    stderr=f"File not found: {path}",
                    duration_ms=(time.time() - start) * 1000,
                )

            async with aiofiles.open(target, "w") as f:
                await f.write(content)

            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="write_file",
                exit_code=0,
                stdout=f"Written {len(content)} bytes to {path}",
                side_effects=SideEffects(files_modified=[str(target.relative_to(self.project_dir))]),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="write_file",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def create_file(self, path: str, content: str = "") -> ToolResult:
        """Create new file with optional content."""
        start = time.time()
        try:
            target = self._validate_path(path)
            if target.exists():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="create_file",
                    exit_code=1,
                    stderr=f"File already exists: {path}",
                    duration_ms=(time.time() - start) * 1000,
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(target, "w") as f:
                await f.write(content)

            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="create_file",
                exit_code=0,
                stdout=f"Created {path}",
                side_effects=SideEffects(files_created=[str(target.relative_to(self.project_dir))]),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="create_file",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def delete_file(self, path: str) -> ToolResult:
        """Delete a file."""
        start = time.time()
        try:
            target = self._validate_path(path)
            if not target.exists():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="delete_file",
                    exit_code=1,
                    stderr=f"File not found: {path}",
                    duration_ms=(time.time() - start) * 1000,
                )

            if not target.is_file():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="delete_file",
                    exit_code=1,
                    stderr=f"Not a file: {path}",
                    duration_ms=(time.time() - start) * 1000,
                )

            await aiofiles.os.remove(target)

            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="delete_file",
                exit_code=0,
                stdout=f"Deleted {path}",
                side_effects=SideEffects(files_deleted=[str(target.relative_to(self.project_dir))]),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="delete_file",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def move_file(self, src: str, dst: str) -> ToolResult:
        """Move/rename a file."""
        start = time.time()
        try:
            src_path = self._validate_path(src)
            dst_path = self._validate_path(dst)

            if not src_path.exists():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="move_file",
                    exit_code=1,
                    stderr=f"Source not found: {src}",
                    duration_ms=(time.time() - start) * 1000,
                )

            if dst_path.exists():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="move_file",
                    exit_code=1,
                    stderr=f"Destination already exists: {dst}",
                    duration_ms=(time.time() - start) * 1000,
                )

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            src_path.rename(dst_path)

            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="move_file",
                exit_code=0,
                stdout=f"Moved {src} to {dst}",
                side_effects=SideEffects(
                    files_created=[str(dst_path.relative_to(self.project_dir))],
                    files_deleted=[str(src_path.relative_to(self.project_dir))],
                ),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="move_file",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def list_directory(self, path: str = ".", recursive: bool = False) -> ToolResult:
        """List directory contents."""
        start = time.time()
        try:
            target = self._validate_path(path)
            if not target.exists():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="list_directory",
                    exit_code=1,
                    stderr=f"Directory not found: {path}",
                    duration_ms=(time.time() - start) * 1000,
                )

            if not target.is_dir():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="list_directory",
                    exit_code=1,
                    stderr=f"Not a directory: {path}",
                    duration_ms=(time.time() - start) * 1000,
                )

            if recursive:
                items = sorted(target.glob("**/*"))
            else:
                items = sorted(target.iterdir())

            output_lines = []
            for item in items:
                rel_path = item.relative_to(self.project_dir)
                if item.is_dir():
                    output_lines.append(f"{rel_path}/")
                else:
                    output_lines.append(str(rel_path))

            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="list_directory",
                exit_code=0,
                stdout="\n".join(output_lines),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="list_directory",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def search_files(self, pattern: str, directory: str = ".") -> ToolResult:
        """Search files by glob pattern."""
        start = time.time()
        try:
            base_dir = self._validate_path(directory)
            if not base_dir.exists():
                return ToolResult(
                    tool=ToolName.FILE_SYSTEM,
                    operation="search_files",
                    exit_code=1,
                    stderr=f"Directory not found: {directory}",
                    duration_ms=(time.time() - start) * 1000,
                )

            matches = sorted(base_dir.glob(pattern))
            output_lines = [str(m.relative_to(self.project_dir)) for m in matches]

            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="search_files",
                exit_code=0,
                stdout="\n".join(output_lines),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.FILE_SYSTEM,
                operation="search_files",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
