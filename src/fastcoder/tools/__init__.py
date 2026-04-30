"""Tool layer - unified interface for all tools."""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastcoder.tools.build_runner import BuildRunner
from fastcoder.tools.file_system import FileSystemTool
from fastcoder.tools.git_client import GitClient
from fastcoder.tools.package_manager import PackageManagerTool
from fastcoder.tools.shell_executor import ShellExecutor
from fastcoder.tools.test_runner import TestRunnerTool
from fastcoder.types.tools import ToolCall, ToolName, ToolPolicy, ToolResult

logger = logging.getLogger(__name__)


class ToolLayer:
    """Unified tool execution layer with validation and metrics."""

    def __init__(
        self,
        project_dir: str,
        policies: Optional[dict[ToolName, ToolPolicy]] = None,
        command_allowlist: Optional[list[str]] = None,
    ):
        """Initialize tool layer."""
        self.project_dir = project_dir
        self.policies = policies or {}
        self.call_counts = {}
        self.call_metrics = {}

        self.shell = ShellExecutor(
            project_dir, command_allowlist=command_allowlist
        )
        self.file_system = FileSystemTool(project_dir)
        self.git = GitClient(project_dir)
        self.package_manager = PackageManagerTool(project_dir, self.shell)
        self.test_runner = TestRunnerTool(project_dir, self.shell)
        self.build_runner = BuildRunner(project_dir, self.shell)

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        start = time.time()

        logger.info(f"Executing {call.tool}.{call.operation} with args {call.args}")

        try:
            self._validate_call(call)

            result = await self._dispatch(call)

            duration = time.time() - start
            self._record_metric(call.tool, call.operation, duration, result.exit_code)

            logger.info(
                f"{call.tool}.{call.operation} completed in {duration:.2f}s "
                f"(exit code: {result.exit_code})"
            )

            return result

        except ValueError as e:
            logger.warning(f"Policy violation: {e}")
            return ToolResult(
                tool=call.tool,
                operation=call.operation,
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return ToolResult(
                tool=call.tool,
                operation=call.operation,
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _validate_call(self, call: ToolCall) -> None:
        """Validate tool call against policies."""
        policy = self.policies.get(call.tool)
        if not policy:
            return

        if policy.allowed_operations and call.operation not in policy.allowed_operations:
            raise ValueError(
                f"Operation {call.operation} not allowed for {call.tool}"
            )

        count_key = f"{call.tool.value}:{call.operation}"
        current_count = self.call_counts.get(count_key, 0)

        if current_count >= policy.max_calls_per_minute:
            raise ValueError(
                f"Rate limit exceeded for {call.tool}.{call.operation} "
                f"({policy.max_calls_per_minute} calls/min)"
            )

        self.call_counts[count_key] = current_count + 1

    async def _dispatch(self, call: ToolCall) -> ToolResult:
        """Dispatch to appropriate tool."""
        if call.tool == ToolName.FILE_SYSTEM:
            return await self._file_system_dispatch(call)
        elif call.tool == ToolName.SHELL:
            return await self.shell.execute(
                call.args.get("command", ""),
                working_dir=call.working_dir,
                timeout_ms=call.timeout_ms,
                env=call.env,
            )
        elif call.tool == ToolName.GIT:
            return await self._git_dispatch(call)
        elif call.tool == ToolName.PACKAGE_MANAGER:
            return await self._package_manager_dispatch(call)
        elif call.tool == ToolName.TEST_RUNNER:
            return await self._test_runner_dispatch(call)
        elif call.tool == ToolName.BUILD_TOOLS:
            return await self._build_runner_dispatch(call)
        else:
            return ToolResult(
                tool=call.tool,
                operation=call.operation,
                exit_code=1,
                stderr=f"Unknown tool: {call.tool}",
            )

    async def _file_system_dispatch(self, call: ToolCall) -> ToolResult:
        """Dispatch file system operations."""
        op = call.operation
        args = call.args

        if op == "read_file":
            return await self.file_system.read_file(args.get("path", ""))
        elif op == "write_file":
            return await self.file_system.write_file(
                args.get("path", ""), args.get("content", "")
            )
        elif op == "create_file":
            return await self.file_system.create_file(
                args.get("path", ""), args.get("content", "")
            )
        elif op == "delete_file":
            return await self.file_system.delete_file(args.get("path", ""))
        elif op == "move_file":
            return await self.file_system.move_file(
                args.get("src", ""), args.get("dst", "")
            )
        elif op == "list_directory":
            return await self.file_system.list_directory(
                args.get("path", "."), args.get("recursive", False)
            )
        elif op == "search_files":
            return await self.file_system.search_files(
                args.get("pattern", ""), args.get("directory", ".")
            )
        else:
            return ToolResult(
                tool=call.tool,
                operation=op,
                exit_code=1,
                stderr=f"Unknown operation: {op}",
            )

    async def _git_dispatch(self, call: ToolCall) -> ToolResult:
        """Dispatch git operations."""
        op = call.operation
        args = call.args

        if op == "create_branch":
            try:
                branch = self.git.create_branch(
                    args.get("story_id", ""), args.get("description", "")
                )
                return ToolResult(
                    tool=call.tool,
                    operation=op,
                    exit_code=0,
                    stdout=f"Created branch: {branch}",
                )
            except Exception as e:
                return ToolResult(
                    tool=call.tool,
                    operation=op,
                    exit_code=1,
                    stderr=str(e),
                )
        elif op == "commit_changes":
            return await self.git.commit_changes(
                args.get("message", ""), args.get("files")
            )
        elif op == "get_diff":
            return self.git.get_diff(args.get("base"))
        elif op == "get_status":
            return self.git.get_status()
        elif op == "get_log":
            return self.git.get_log(args.get("count", 10))
        elif op == "get_current_branch":
            return ToolResult(
                tool=call.tool,
                operation=op,
                exit_code=0,
                stdout=self.git.get_current_branch(),
            )
        elif op == "push":
            return self.git.push(args.get("branch"), args.get("force", False))
        elif op == "checkout":
            return self.git.checkout(args.get("branch", ""))
        else:
            return ToolResult(
                tool=call.tool,
                operation=op,
                exit_code=1,
                stderr=f"Unknown operation: {op}",
            )

    async def _package_manager_dispatch(self, call: ToolCall) -> ToolResult:
        """Dispatch package manager operations."""
        op = call.operation
        args = call.args

        if op == "install":
            return await self.package_manager.install(args.get("packages"))
        elif op == "update":
            return await self.package_manager.update(args.get("packages"))
        elif op == "remove":
            return await self.package_manager.remove(args.get("packages", []))
        elif op == "audit":
            return await self.package_manager.audit()
        else:
            return ToolResult(
                tool=call.tool,
                operation=op,
                exit_code=1,
                stderr=f"Unknown operation: {op}",
            )

    async def _test_runner_dispatch(self, call: ToolCall) -> ToolResult:
        """Dispatch test runner operations."""
        op = call.operation
        args = call.args

        if op == "run_all":
            return await self.test_runner.run_all()
        elif op == "run_file":
            return await self.test_runner.run_file(args.get("file", ""))
        elif op == "run_single":
            return await self.test_runner.run_single(
                args.get("file", ""), args.get("test_name", "")
            )
        elif op == "get_coverage":
            return await self.test_runner.get_coverage()
        else:
            return ToolResult(
                tool=call.tool,
                operation=op,
                exit_code=1,
                stderr=f"Unknown operation: {op}",
            )

    async def _build_runner_dispatch(self, call: ToolCall) -> ToolResult:
        """Dispatch build runner operations."""
        op = call.operation

        if op == "build":
            return await self.build_runner.build()
        elif op == "lint":
            return await self.build_runner.lint()
        elif op == "format_code":
            return await self.build_runner.format_code()
        elif op == "type_check":
            return await self.build_runner.type_check()
        else:
            return ToolResult(
                tool=call.tool,
                operation=op,
                exit_code=1,
                stderr=f"Unknown operation: {op}",
            )

    def _record_metric(
        self, tool: ToolName, operation: str, duration: float, exit_code: int
    ) -> None:
        """Record execution metrics."""
        key = f"{tool.value}:{operation}"
        if key not in self.call_metrics:
            self.call_metrics[key] = {
                "count": 0,
                "total_duration": 0.0,
                "success_count": 0,
                "error_count": 0,
            }

        metrics = self.call_metrics[key]
        metrics["count"] += 1
        metrics["total_duration"] += duration
        if exit_code == 0:
            metrics["success_count"] += 1
        else:
            metrics["error_count"] += 1

    def get_metrics(self) -> dict:
        """Get execution metrics."""
        return {
            key: {
                **metrics,
                "avg_duration": metrics["total_duration"] / metrics["count"],
                "success_rate": metrics["success_count"] / metrics["count"]
                if metrics["count"] > 0
                else 0,
            }
            for key, metrics in self.call_metrics.items()
        }


__all__ = [
    "ToolLayer",
    "FileSystemTool",
    "ShellExecutor",
    "GitClient",
    "PackageManagerTool",
    "TestRunnerTool",
    "BuildRunner",
]
