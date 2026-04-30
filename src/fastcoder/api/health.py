"""Deep health check system for monitoring component health and availability."""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import structlog
from fastapi import APIRouter

from fastcoder.config.llm_key_store import LLMKeyStore
from fastcoder.llm.router import ModelRouter
from fastcoder.memory import MemoryStore
from fastcoder.types.story import Story, StoryState

logger = structlog.get_logger(__name__)


@dataclass
class ComponentHealthStatus:
    """Health status for a single component."""

    status: str  # "ok", "warn", "error"
    message: str
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class HealthChecker:
    """Monitors health of all critical system components."""

    # Thresholds
    DISK_SPACE_WARN_MB = 500
    HEALTH_CHECK_TIMEOUT_SEC = 5.0

    def __init__(
        self,
        router: ModelRouter,
        memory_store: MemoryStore,
        key_store: LLMKeyStore,
        project_dir: str,
        story_queue_accessor: Optional[Callable[[], list[Story]]] = None,
    ):
        """
        Initialize health checker.

        Args:
            router: LLM model router instance
            memory_store: Memory store instance
            key_store: LLM key store instance
            project_dir: Project directory path
            story_queue_accessor: Callable that returns list of active stories
        """
        self.router = router
        self.memory_store = memory_store
        self.key_store = key_store
        self.project_dir = Path(project_dir)
        self.story_queue_accessor = story_queue_accessor or (lambda: [])

    async def check_all(self) -> dict:
        """
        Check health of all components.

        Returns:
            Dictionary with overall status and per-component details
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
        }

        # Check each component
        checks = [
            ("llm_providers", self._check_llm_providers),
            ("memory_store", self._check_memory_store),
            ("key_store", self._check_key_store),
            ("disk_space", self._check_disk_space),
            ("story_queue", self._check_story_queue),
        ]

        for component_name, check_func in checks:
            try:
                status = await asyncio.wait_for(
                    check_func(), timeout=self.HEALTH_CHECK_TIMEOUT_SEC
                )
                results["components"][component_name] = status.to_dict()
            except asyncio.TimeoutError:
                results["components"][component_name] = ComponentHealthStatus(
                    status="error",
                    message=f"Health check timed out after {self.HEALTH_CHECK_TIMEOUT_SEC}s",
                    latency_ms=self.HEALTH_CHECK_TIMEOUT_SEC * 1000,
                ).to_dict()
            except Exception as e:
                logger.exception(f"Error checking {component_name}", component=component_name)
                results["components"][component_name] = ComponentHealthStatus(
                    status="error",
                    message=f"Unexpected error: {str(e)}",
                ).to_dict()

        # Determine overall status
        statuses = [c["status"] for c in results["components"].values()]
        critical_checks = ["llm_providers", "key_store"]
        critical_statuses = [
            results["components"].get(name, {}).get("status", "error")
            for name in critical_checks
        ]

        if "error" in critical_statuses:
            results["overall_status"] = "unhealthy"
        elif "error" in statuses:
            results["overall_status"] = "degraded"
        else:
            results["overall_status"] = "healthy"

        return results

    async def _check_llm_providers(self) -> ComponentHealthStatus:
        """Check if at least one LLM provider is healthy."""
        start_time = time.time()

        try:
            # Check health cache from router
            health_cache = getattr(self.router, "_health_cache", {})

            if not health_cache:
                # No providers checked yet, attempt to check
                providers = getattr(self.router, "_providers", {})
                if not providers:
                    return ComponentHealthStatus(
                        status="error",
                        message="No LLM providers configured",
                        latency_ms=(time.time() - start_time) * 1000,
                    )

                # Check at least one provider
                for provider_name, provider in list(providers.items())[:1]:
                    health = await provider.health_check()
                    health_cache[provider_name] = health

            # Verify at least one is healthy
            healthy_providers = [
                name for name, status in health_cache.items() if status.healthy
            ]

            if healthy_providers:
                return ComponentHealthStatus(
                    status="ok",
                    message=f"{len(healthy_providers)} of {len(health_cache)} providers healthy",
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={"healthy_providers": healthy_providers},
                )
            else:
                return ComponentHealthStatus(
                    status="error",
                    message=f"No healthy providers: {list(health_cache.keys())}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

        except Exception as e:
            return ComponentHealthStatus(
                status="error",
                message=f"Provider health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _check_memory_store(self) -> ComponentHealthStatus:
        """Check if memory store is loaded and accessible."""
        start_time = time.time()

        try:
            # Check if memory store has entries or can be accessed
            has_memories = any(
                len(entries) > 0
                for entries in self.memory_store.memories.values()
            )
            error_fixes_count = len(self.memory_store.error_fixes)

            status = "ok" if (has_memories or error_fixes_count > 0) else "warn"
            message = (
                f"Memory store: {sum(len(e) for e in self.memory_store.memories.values())} "
                f"memories, {error_fixes_count} error fixes"
            )

            return ComponentHealthStatus(
                status=status,
                message=message,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    "total_memories": sum(
                        len(e) for e in self.memory_store.memories.values()
                    ),
                    "error_fixes": error_fixes_count,
                },
            )

        except Exception as e:
            return ComponentHealthStatus(
                status="error",
                message=f"Memory store check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _check_key_store(self) -> ComponentHealthStatus:
        """Check if SQLite key store database is accessible."""
        start_time = time.time()

        try:
            # Attempt to query the database
            keys = self.key_store.get_all_keys()
            configured_providers = len(keys)

            status = "ok" if configured_providers > 0 else "warn"
            message = f"Key store accessible with {configured_providers} configured providers"

            return ComponentHealthStatus(
                status=status,
                message=message,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"configured_providers": list(keys.keys())},
            )

        except Exception as e:
            return ComponentHealthStatus(
                status="error",
                message=f"Key store not accessible: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _check_disk_space(self) -> ComponentHealthStatus:
        """Check available disk space."""
        start_time = time.time()

        try:
            stat = shutil.disk_usage(str(self.project_dir))
            available_mb = stat.free / (1024 * 1024)

            if available_mb < self.DISK_SPACE_WARN_MB:
                status = "warn"
                message = f"Low disk space: {available_mb:.1f} MB available"
            else:
                status = "ok"
                message = f"Disk space: {available_mb:.1f} MB available"

            return ComponentHealthStatus(
                status=status,
                message=message,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    "available_mb": round(available_mb, 1),
                    "total_mb": round(stat.total / (1024 * 1024), 1),
                    "used_mb": round(stat.used / (1024 * 1024), 1),
                },
            )

        except Exception as e:
            return ComponentHealthStatus(
                status="warn",
                message=f"Could not check disk space: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _check_story_queue(self) -> ComponentHealthStatus:
        """Check story queue and active stories."""
        start_time = time.time()

        try:
            stories = self.story_queue_accessor()
            active_stories = [
                s for s in stories
                if s.state not in (StoryState.DONE, StoryState.FAILED)
            ]
            completed_stories = [
                s for s in stories
                if s.state in (StoryState.DONE, StoryState.FAILED)
            ]

            return ComponentHealthStatus(
                status="ok",
                message=f"Queue: {len(active_stories)} active, {len(completed_stories)} completed",
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    "active_count": len(active_stories),
                    "completed_count": len(completed_stories),
                    "total_count": len(stories),
                },
            )

        except Exception as e:
            return ComponentHealthStatus(
                status="warn",
                message=f"Could not check story queue: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
            )


def create_health_routes(health_checker: HealthChecker) -> APIRouter:
    """
    Create FastAPI router with health check endpoints.

    Args:
        health_checker: HealthChecker instance

    Returns:
        FastAPI APIRouter with health endpoints
    """
    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("")
    async def health_check() -> dict:
        """
        Simple health check for load balancers.

        Returns:
            {"status": "healthy" | "degraded" | "unhealthy"}
        """
        result = await health_checker.check_all()
        return {"status": result["overall_status"]}

    @router.get("/deep")
    async def deep_health_check() -> dict:
        """
        Deep health check with per-component details.

        Returns:
            Full health report with component-level status
        """
        return await health_checker.check_all()

    return router
