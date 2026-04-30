"""Shell command execution tool."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import time
from pathlib import Path
from typing import Optional

import structlog

from fastcoder.tools.resource_limiter import ResourceLimiter, ResourceUsage
from fastcoder.types.tools import SideEffects, ToolName, ToolResult

logger = structlog.get_logger(__name__)

# Environment variables that are NEVER allowed to be overridden
_BLOCKED_ENV_VARS = frozenset({
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "PYTHONSTARTUP",
    "PYTHONPATH",
    "NODE_OPTIONS",
    "BASH_ENV",
    "ENV",
    "PROMPT_COMMAND",
})


class ShellExecutor:
    """Execute shell commands with safety controls."""

    def __init__(
        self,
        project_dir: str,
        command_allowlist: Optional[list[str]] = None,
        max_timeout_ms: int = 300000,
        resource_limiter: Optional[ResourceLimiter] = None,
    ):
        """Initialize shell executor.

        Args:
            project_dir: Root directory for all command execution.
            command_allowlist: List of allowed commands. Uses default if None.
            max_timeout_ms: Maximum timeout for any single command.
            resource_limiter: Optional ResourceLimiter for enforcing resource limits.
                If provided, all commands will be executed with resource limits.
        """
        self.project_dir = Path(project_dir).resolve()
        self.command_allowlist = command_allowlist or self._default_allowlist()
        self.max_timeout_ms = max_timeout_ms
        self.resource_limiter = resource_limiter

    @staticmethod
    def _default_allowlist() -> list[str]:
        """Default command allowlist."""
        return [
            "npm",
            "npx",
            "yarn",
            "pnpm",
            "pip",
            "python",
            "node",
            "pytest",
            "jest",
            "vitest",
            "eslint",
            "ruff",
            "mypy",
            "tsc",
            "cargo",
            "go",
            "make",
            "docker",
            "git",
            "cat",
            "ls",
            "find",
            "grep",
            "wc",
            "head",
            "tail",
            "mkdir",
            "cp",
            "mv",
            "rm",
            "touch",
            "chmod",
            "curl",
            "gh",
        ]

    def _validate_command(self, command: str) -> bool:
        """Check if command is in allowlist.

        Validates the base command (first token, stripping any path prefix)
        against the allowlist. Rejects commands with absolute/relative path
        prefixes to prevent allowlist bypass via PATH manipulation.
        """
        try:
            parts = shlex.split(command)
        except ValueError:
            return False
        if not parts:
            return False
        cmd = parts[0]
        # Reject commands specified with path (e.g. /tmp/evil, ./hack)
        # to prevent allowlist bypass via attacker-controlled binaries
        if "/" in cmd:
            return False
        return cmd in self.command_allowlist

    def _validate_command_args(self, command: str) -> tuple[bool, Optional[str]]:
        """Validate command arguments for dangerous patterns.

        Checks for destructive argument patterns:
        - rm: blocks -rf /, --no-preserve-root, paths outside project_dir
        - chmod: blocks recursive chmod on root or parent dirs
        - mv/cp: ensures target is within project_dir

        Args:
            command: The shell command to validate

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        try:
            parts = shlex.split(command)
        except ValueError:
            return False, "Failed to parse command"

        if not parts:
            return False, "Empty command"

        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        # Validate rm command
        if cmd == "rm":
            for arg in args:
                if arg == "-rf" and "/" in args:
                    return False, "Blocked: rm -rf / is dangerous"
                if arg == "--no-preserve-root":
                    return False, "Blocked: --no-preserve-root flag is dangerous"

            # Check paths are within project_dir
            for arg in args:
                if arg.startswith("-"):
                    continue
                try:
                    arg_path = Path(arg).resolve()
                    if not arg_path.is_relative_to(self.project_dir):
                        return False, f"Blocked: rm path outside project: {arg}"
                except (ValueError, RuntimeError):
                    return False, f"Invalid path for rm: {arg}"

        # Validate chmod command
        elif cmd == "chmod":
            if "-R" in args or "--recursive" in args:
                for arg in args:
                    if arg.startswith("-"):
                        continue
                    try:
                        arg_path = Path(arg).resolve()
                        # Block recursive chmod on root or parents of project_dir
                        if arg_path == Path("/") or arg_path == self.project_dir.parent:
                            return False, f"Blocked: recursive chmod on {arg}"
                    except (ValueError, RuntimeError):
                        pass

        # Validate mv/cp command
        elif cmd in ("mv", "cp"):
            # Target is typically the last argument
            if args:
                target = args[-1]
                if not target.startswith("-"):
                    try:
                        target_path = Path(target).resolve()
                        if not target_path.is_relative_to(self.project_dir):
                            return False, f"Blocked: {cmd} target outside project: {target}"
                    except (ValueError, RuntimeError):
                        return False, f"Invalid path for {cmd} target: {target}"

        return True, None

    def _sanitize_env(self, env: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
        """Build a safe environment dict.

        Starts from a minimal copy of the current environment, then merges
        caller-provided overrides — but blocks any variable in the deny list.
        """
        # Start from a minimal base so we inherit PATH etc.
        safe_env = {
            k: v for k, v in os.environ.items()
            if k not in _BLOCKED_ENV_VARS
        }
        if env:
            for k, v in env.items():
                # Check both original and uppercased key against blocklist
                # to prevent bypass via lowercase variants on case-sensitive systems
                if k.upper() in _BLOCKED_ENV_VARS:
                    logger.debug("blocked_env_var_override", key=k)
                    continue  # silently drop dangerous overrides
                safe_env[k] = v
        return safe_env

    async def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ToolResult:
        """Execute shell command with safety controls."""
        start = time.time()

        if not self._validate_command(command):
            cmd_name = command.split()[0] if command.split() else "<empty>"
            return ToolResult(
                tool=ToolName.SHELL,
                operation="execute",
                exit_code=1,
                stderr=f"Command not in allowlist: {cmd_name}",
                duration_ms=(time.time() - start) * 1000,
            )

        # Validate command arguments for dangerous patterns
        args_valid, args_error = self._validate_command_args(command)
        if not args_valid:
            return ToolResult(
                tool=ToolName.SHELL,
                operation="execute",
                exit_code=1,
                stderr=args_error or "Invalid command arguments",
                duration_ms=(time.time() - start) * 1000,
            )

        # Validate working directory stays within project (using is_relative_to
        # to prevent string-prefix bypass like /project matching /project-evil)
        work_dir = working_dir or str(self.project_dir)
        resolved_work_dir = Path(work_dir).resolve()
        if not resolved_work_dir.is_relative_to(self.project_dir):
            return ToolResult(
                tool=ToolName.SHELL,
                operation="execute",
                exit_code=1,
                stderr=f"Working directory outside project: {work_dir}",
                duration_ms=(time.time() - start) * 1000,
            )

        timeout_sec = None
        if timeout_ms:
            timeout_sec = min(timeout_ms, self.max_timeout_ms) / 1000.0

        # Sanitize environment variables
        safe_env = self._sanitize_env(env)

        try:
            # Use resource-limited execution if limiter is available
            if self.resource_limiter is not None:
                return_code, stdout, stderr, usage = await self._execute_with_resource_limits(
                    command, str(resolved_work_dir), safe_env, timeout_sec
                )
                return ToolResult(
                    tool=ToolName.SHELL,
                    operation="execute",
                    exit_code=return_code,
                    stdout=stdout,
                    stderr=stderr,
                    side_effects=self._detect_side_effects(command),
                    duration_ms=(time.time() - start) * 1000,
                    metadata={
                        "resource_usage": usage.model_dump(),
                        "limits_hit": usage.limits_hit,
                    },
                )

            # Standard execution without resource limits
            # Use create_subprocess_exec (not shell) to prevent shell injection
            cmd_parts = shlex.split(command)
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(resolved_work_dir),
                env=safe_env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                return ToolResult(
                    tool=ToolName.SHELL,
                    operation="execute",
                    exit_code=124,
                    stderr="Command timed out",
                    duration_ms=(time.time() - start) * 1000,
                )

            return ToolResult(
                tool=ToolName.SHELL,
                operation="execute",
                exit_code=process.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                side_effects=self._detect_side_effects(command),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool=ToolName.SHELL,
                operation="execute",
                exit_code=1,
                stderr=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _execute_with_resource_limits(
        self,
        command: str,
        cwd: str,
        env: Optional[dict[str, str]],
        timeout_sec: Optional[float],
    ) -> tuple[int, str, str, ResourceUsage]:
        """Execute command with resource limits.

        Args:
            command: Shell command to execute.
            cwd: Working directory.
            env: Environment variables.
            timeout_sec: Timeout in seconds.

        Returns:
            Tuple of (return_code, stdout, stderr, resource_usage).
        """
        # Use network timeout from limiter's limits
        network_timeout = self.resource_limiter.limits.network_timeout
        if timeout_sec is not None:
            network_timeout = min(network_timeout, timeout_sec)

        return_code, stdout, stderr, usage = await self.resource_limiter.execute_with_limits(
            cmd=command, cwd=cwd, env=env, limits=self.resource_limiter.limits
        )

        # Log resource usage if limits were hit
        if usage.limits_hit:
            logger.warning(
                "command_resource_limits_exceeded",
                command=command,
                limits_hit=usage.limits_hit,
                cpu_used=usage.cpu_time_seconds,
                memory_used=usage.memory_peak_bytes,
                wall_time=usage.wall_time_seconds,
            )
        else:
            logger.debug(
                "command_resource_usage",
                command=command,
                cpu_used=usage.cpu_time_seconds,
                memory_used=usage.memory_peak_bytes,
                wall_time=usage.wall_time_seconds,
            )

        return return_code, stdout, stderr, usage

    def _detect_side_effects(self, command: str) -> SideEffects:
        """Detect file operations from command."""
        effects = SideEffects()

        npm_install_match = re.search(
            r"npm|yarn|pnpm.*(?:install|add|remove|uninstall)", command
        )
        if npm_install_match:
            if "add" in command or "install" in command:
                effects.packages_added = ["<detected from npm/yarn/pnpm>"]
            elif "remove" in command or "uninstall" in command:
                effects.packages_removed = ["<detected from npm/yarn/pnpm>"]

        pip_match = re.search(r"pip.*(?:install|uninstall)", command)
        if pip_match:
            if "install" in command:
                effects.packages_added = ["<detected from pip>"]
            elif "uninstall" in command:
                effects.packages_removed = ["<detected from pip>"]

        if "rm " in command or "delete " in command:
            effects.files_deleted = ["<detected file deletion>"]

        if "git commit" in command or "git push" in command:
            effects.files_modified = ["<detected git operation>"]

        return effects
