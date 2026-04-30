"""Resource limits enforcement module for subprocess execution."""

from __future__ import annotations

import asyncio
import os
import resource
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# Default resource limits (in bytes/seconds)
_DEFAULT_CPU_SECONDS = 300
_DEFAULT_MEMORY_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
_DEFAULT_DISK_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
_DEFAULT_MAX_PROCESSES = 50
_DEFAULT_NETWORK_TIMEOUT = 60
_DEFAULT_WALL_TIME_SECONDS = 600


class ResourceLimits(BaseModel):
    """Resource limit specifications."""

    cpu_seconds: int = Field(default=_DEFAULT_CPU_SECONDS, ge=1)
    memory_bytes: int = Field(default=_DEFAULT_MEMORY_BYTES, ge=1024)
    disk_bytes: int = Field(default=_DEFAULT_DISK_BYTES, ge=1024)
    max_processes: int = Field(default=_DEFAULT_MAX_PROCESSES, ge=1)
    network_timeout: int = Field(default=_DEFAULT_NETWORK_TIMEOUT, ge=1)
    wall_time_seconds: int = Field(default=_DEFAULT_WALL_TIME_SECONDS, ge=1)


class ResourceUsage(BaseModel):
    """Actual resource usage metrics."""

    cpu_time_seconds: float = 0.0
    memory_peak_bytes: int = 0
    disk_written_bytes: int = 0
    processes_created: int = 0
    wall_time_seconds: float = 0.0
    limits_hit: list[str] = Field(default_factory=list)


class ResourceViolation(BaseModel):
    """Violation of a resource limit."""

    resource: str
    limit: int | float
    actual: int | float
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ResourceLimiter:
    """Enforce resource limits on subprocess execution."""

    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize resource limiter with optional custom limits.

        Args:
            limits: Custom resource limits. If None, uses standard defaults.
        """
        self.limits = limits or ResourceLimits()
        logger.debug(
            "resource_limiter_initialized",
            cpu_seconds=self.limits.cpu_seconds,
            memory_bytes=self.limits.memory_bytes,
            disk_bytes=self.limits.disk_bytes,
            max_processes=self.limits.max_processes,
            network_timeout=self.limits.network_timeout,
            wall_time_seconds=self.limits.wall_time_seconds,
        )

    def get_preexec_fn(self) -> Callable[[], None]:
        """Get a preexec_fn suitable for subprocess.Popen.

        Returns a function that sets OS-level resource limits using
        resource.setrlimit. This is called in the child process before
        execution begins.

        Returns:
            A callable with no arguments suitable for preexec_fn parameter.
        """

        def _set_limits() -> None:
            """Set resource limits in child process."""
            try:
                # CPU time limit (in seconds)
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (self.limits.cpu_seconds, self.limits.cpu_seconds),
                )

                # Memory limit (virtual address space)
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (self.limits.memory_bytes, self.limits.memory_bytes),
                )

                # File size limit (disk writes)
                resource.setrlimit(
                    resource.RLIMIT_FSIZE,
                    (self.limits.disk_bytes, self.limits.disk_bytes),
                )

                # Process count limit
                resource.setrlimit(
                    resource.RLIMIT_NPROC,
                    (self.limits.max_processes, self.limits.max_processes),
                )
            except ValueError as e:
                logger.error("failed_to_set_resource_limits", error=str(e))
                raise

        return _set_limits

    async def monitor_process(
        self, process: asyncio.subprocess.Process, limits: Optional[ResourceLimits] = None
    ) -> ResourceUsage:
        """Monitor a running asyncio subprocess for resource violations.

        Polls /proc/{pid}/stat and /proc/{pid}/status to track CPU and memory
        usage. Kills the process if any limit is exceeded.

        Args:
            process: The asyncio subprocess to monitor.
            limits: Resource limits to enforce. Uses self.limits if None.

        Returns:
            ResourceUsage object with collected metrics.
        """
        limits = limits or self.limits
        usage = ResourceUsage()
        start_time = time.time()
        initial_rss = 0
        last_cpu_ticks = 0
        pid = process.pid

        if pid is None:
            logger.warning("monitor_process_no_pid")
            return usage

        try:
            while process.returncode is None:
                try:
                    # Read /proc/{pid}/stat for CPU time
                    stat_path = f"/proc/{pid}/stat"
                    if os.path.exists(stat_path):
                        with open(stat_path, "r") as f:
                            stat_line = f.read()
                            # utime is field 13, stime is field 14 (0-indexed)
                            fields = stat_line.split()
                            if len(fields) > 14:
                                try:
                                    utime = int(fields[13])
                                    stime = int(fields[14])
                                    # Convert from clock ticks to seconds
                                    # (typically 100 ticks per second)
                                    clock_ticks = os.sysconf("SC_CLK_TCK")
                                    cpu_seconds = (utime + stime) / clock_ticks
                                    usage.cpu_time_seconds = max(
                                        usage.cpu_time_seconds, cpu_seconds
                                    )
                                except (ValueError, IndexError):
                                    pass

                    # Read /proc/{pid}/status for memory and process count
                    status_path = f"/proc/{pid}/status"
                    if os.path.exists(status_path):
                        with open(status_path, "r") as f:
                            for line in f:
                                if line.startswith("VmRSS:"):
                                    try:
                                        rss_kb = int(line.split()[1])
                                        rss_bytes = rss_kb * 1024
                                        usage.memory_peak_bytes = max(
                                            usage.memory_peak_bytes, rss_bytes
                                        )
                                        if initial_rss == 0:
                                            initial_rss = rss_bytes
                                    except (ValueError, IndexError):
                                        pass
                                elif line.startswith("Threads:"):
                                    try:
                                        threads = int(line.split()[1])
                                        usage.processes_created = max(
                                            usage.processes_created, threads
                                        )
                                    except (ValueError, IndexError):
                                        pass

                    # Check for limit violations
                    wall_time = time.time() - start_time
                    usage.wall_time_seconds = wall_time

                    if (
                        usage.cpu_time_seconds > limits.cpu_seconds
                        and "cpu" not in usage.limits_hit
                    ):
                        usage.limits_hit.append("cpu")
                        logger.warning(
                            "cpu_limit_exceeded",
                            limit=limits.cpu_seconds,
                            actual=usage.cpu_time_seconds,
                            pid=pid,
                        )
                        process.kill()
                        break

                    if (
                        usage.memory_peak_bytes > limits.memory_bytes
                        and "memory" not in usage.limits_hit
                    ):
                        usage.limits_hit.append("memory")
                        logger.warning(
                            "memory_limit_exceeded",
                            limit=limits.memory_bytes,
                            actual=usage.memory_peak_bytes,
                            pid=pid,
                        )
                        process.kill()
                        break

                    if (
                        wall_time > limits.wall_time_seconds
                        and "wall_time" not in usage.limits_hit
                    ):
                        usage.limits_hit.append("wall_time")
                        logger.warning(
                            "wall_time_limit_exceeded",
                            limit=limits.wall_time_seconds,
                            actual=wall_time,
                            pid=pid,
                        )
                        process.kill()
                        break

                    if (
                        usage.processes_created > limits.max_processes
                        and "processes" not in usage.limits_hit
                    ):
                        usage.limits_hit.append("processes")
                        logger.warning(
                            "process_limit_exceeded",
                            limit=limits.max_processes,
                            actual=usage.processes_created,
                            pid=pid,
                        )
                        process.kill()
                        break

                except (FileNotFoundError, IOError, PermissionError):
                    # Process may have exited or /proc files may be unavailable
                    pass

                # Sleep briefly before next poll
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error("monitor_process_error", error=str(e), pid=pid)

        return usage

    async def execute_with_limits(
        self,
        cmd: str,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        limits: Optional[ResourceLimits] = None,
    ) -> tuple[int, str, str, ResourceUsage]:
        """Execute a command with full resource limiting.

        Spawns a subprocess with OS-level resource limits via preexec_fn,
        monitors it for violations, and collects resource usage metrics.

        Args:
            cmd: Shell command to execute.
            cwd: Working directory for the command.
            env: Environment variables to pass to the subprocess.
            limits: Resource limits. Uses self.limits if None.

        Returns:
            Tuple of (return_code, stdout, stderr, resource_usage).
            If process is killed for limit violation, return_code will be
            negative (e.g., -9 for SIGKILL).
        """
        limits = limits or self.limits
        start_time = time.time()

        logger.debug(
            "execute_with_limits_start",
            cmd=cmd,
            cwd=cwd,
            cpu_limit=limits.cpu_seconds,
            memory_limit=limits.memory_bytes,
        )

        try:
            # Create subprocess with preexec_fn for OS-level limits
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or os.getcwd(),
                env=env,
                preexec_fn=self.get_preexec_fn(),
            )

            # Start monitoring task
            monitor_task = asyncio.create_task(self.monitor_process(process, limits))

            try:
                # Wait for process completion or network timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=limits.network_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "execute_network_timeout",
                    timeout=limits.network_timeout,
                    cmd=cmd,
                )
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                stdout = b""
                stderr = b"Command exceeded network timeout"

            # Wait for monitor task to complete
            try:
                usage = await asyncio.wait_for(monitor_task, timeout=5)
            except asyncio.TimeoutError:
                monitor_task.cancel()
                usage = ResourceUsage()

            return_code = process.returncode or 0
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            elapsed = time.time() - start_time
            logger.debug(
                "execute_with_limits_complete",
                cmd=cmd,
                return_code=return_code,
                elapsed_seconds=elapsed,
                cpu_used=usage.cpu_time_seconds,
                memory_used=usage.memory_peak_bytes,
                limits_hit=usage.limits_hit,
            )

            return return_code, stdout_str, stderr_str, usage

        except Exception as e:
            logger.error("execute_with_limits_error", cmd=cmd, error=str(e))
            return 1, "", str(e), ResourceUsage()

    @staticmethod
    def from_template(name: str) -> ResourceLimits:
        """Create ResourceLimits from a predefined template.

        Available templates:
        - "strict": 60s CPU, 512MB RAM, 1GB disk, 30 processes, 30s wall time
        - "standard": 300s CPU, 2GB RAM, 10GB disk, 50 processes, 600s wall time
        - "relaxed": 600s CPU, 4GB RAM, 20GB disk, 100 processes, 1200s wall time
        - "test": 30s CPU, 256MB RAM, 500MB disk, 10 processes, 60s wall time

        Args:
            name: Template name (case-insensitive).

        Returns:
            ResourceLimits instance for the requested template.

        Raises:
            ValueError: If template name is not recognized.
        """
        templates = {
            "strict": ResourceLimits(
                cpu_seconds=60,
                memory_bytes=512 * 1024 * 1024,  # 512 MB
                disk_bytes=1024 * 1024 * 1024,  # 1 GB
                max_processes=30,
                network_timeout=30,
                wall_time_seconds=120,
            ),
            "standard": ResourceLimits(
                cpu_seconds=300,
                memory_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
                disk_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
                max_processes=50,
                network_timeout=60,
                wall_time_seconds=600,
            ),
            "relaxed": ResourceLimits(
                cpu_seconds=600,
                memory_bytes=4 * 1024 * 1024 * 1024,  # 4 GB
                disk_bytes=20 * 1024 * 1024 * 1024,  # 20 GB
                max_processes=100,
                network_timeout=120,
                wall_time_seconds=1200,
            ),
            "test": ResourceLimits(
                cpu_seconds=30,
                memory_bytes=256 * 1024 * 1024,  # 256 MB
                disk_bytes=500 * 1024 * 1024,  # 500 MB
                max_processes=10,
                network_timeout=30,
                wall_time_seconds=60,
            ),
        }

        template_name = name.lower()
        if template_name not in templates:
            raise ValueError(
                f"Unknown template: {name}. Available: {', '.join(templates.keys())}"
            )

        limits = templates[template_name]
        logger.debug("resource_limits_from_template", template=template_name)
        return limits

    @staticmethod
    def check_violations(
        usage: ResourceUsage, limits: ResourceLimits
    ) -> list[ResourceViolation]:
        """Check for violations between actual usage and limits.

        Args:
            usage: Actual resource usage from a process.
            limits: Resource limits that were enforced.

        Returns:
            List of ResourceViolation objects. Empty if no violations.
        """
        violations: list[ResourceViolation] = []

        if usage.cpu_time_seconds > limits.cpu_seconds:
            violations.append(
                ResourceViolation(
                    resource="cpu",
                    limit=limits.cpu_seconds,
                    actual=usage.cpu_time_seconds,
                    message=f"CPU time {usage.cpu_time_seconds}s exceeded limit {limits.cpu_seconds}s",
                )
            )

        if usage.memory_peak_bytes > limits.memory_bytes:
            violations.append(
                ResourceViolation(
                    resource="memory",
                    limit=limits.memory_bytes,
                    actual=usage.memory_peak_bytes,
                    message=f"Memory {usage.memory_peak_bytes} bytes exceeded limit {limits.memory_bytes} bytes",
                )
            )

        if usage.disk_written_bytes > limits.disk_bytes:
            violations.append(
                ResourceViolation(
                    resource="disk",
                    limit=limits.disk_bytes,
                    actual=usage.disk_written_bytes,
                    message=f"Disk writes {usage.disk_written_bytes} bytes exceeded limit {limits.disk_bytes} bytes",
                )
            )

        if usage.processes_created > limits.max_processes:
            violations.append(
                ResourceViolation(
                    resource="processes",
                    limit=limits.max_processes,
                    actual=usage.processes_created,
                    message=f"Process count {usage.processes_created} exceeded limit {limits.max_processes}",
                )
            )

        if usage.wall_time_seconds > limits.wall_time_seconds:
            violations.append(
                ResourceViolation(
                    resource="wall_time",
                    limit=limits.wall_time_seconds,
                    actual=usage.wall_time_seconds,
                    message=f"Wall time {usage.wall_time_seconds}s exceeded limit {limits.wall_time_seconds}s",
                )
            )

        return violations
