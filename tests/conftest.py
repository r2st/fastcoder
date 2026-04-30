"""Pytest configuration and shared fixtures for fastcoder tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastcoder.llm.router import ModelRouter
from fastcoder.types.config import AgentConfig, ModelConfig, ProviderConfig
from fastcoder.types.story import (
    Priority,
    Story,
    StoryConstraints,
    StoryMetadata,
    StoryState,
    StorySubmission,
    StoryType,
)


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    """Create a sample AgentConfig with sensible test defaults."""
    from fastcoder.types.config import (
        LLMConfig,
        ProjectConfig,
        RoutingConfig,
        CostConfig,
        QualityConfig,
        SafetyConfig,
        ToolsConfig,
        ObservabilityConfig,
    )

    provider_config = ProviderConfig(
        name="test-provider",
        type="anthropic",
        api_key="test-key",
        models=[
            ModelConfig(
                id="claude-3-5-sonnet-20241022",
                tier="top",
                max_context_tokens=200000,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
            )
        ],
        enabled=True,
    )

    llm_config = LLMConfig(
        providers=[provider_config],
        routing=RoutingConfig(
            default_tier="top",
            default_provider="test-provider",
            default_model="claude-3-5-sonnet-20241022",
        ),
        cost=CostConfig(
            max_cost_per_story_usd=5.0,
            daily_budget_usd=100.0,
        ),
    )

    project_config = ProjectConfig(
        project_id="test-project",
        language="python",
    )

    return AgentConfig(
        project=project_config,
        llm=llm_config,
        tools=ToolsConfig(),
        safety=SafetyConfig(
            max_iterations_per_story=10,
            max_retries_per_stage=5,
        ),
        quality=QualityConfig(),
        observability=ObservabilityConfig(),
    )


@pytest.fixture
def sample_story_submission() -> StorySubmission:
    """Create a sample StorySubmission for testing."""
    return StorySubmission(
        story="As a developer, I want to add authentication to the API so that only authorized users can access endpoints",
        project_id="test-project",
        priority=Priority.HIGH,
        constraints=StoryConstraints(
            max_iterations=5,
            approval_gates=["pre_deploy"],
            target_branch="develop",
        ),
    )


@pytest.fixture
def sample_story() -> Story:
    """Create a sample Story object in RECEIVED state."""
    return Story(
        id="story-001",
        raw_text="Add authentication to API endpoints",
        project_id="test-project",
        priority=Priority.HIGH,
        state=StoryState.RECEIVED,
        constraints=StoryConstraints(
            max_iterations=5,
            approval_gates=["pre_deploy"],
        ),
        metadata=StoryMetadata(
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            total_tokens_used=0,
            total_cost_usd=0.0,
        ),
    )


@pytest.fixture
def mock_llm_router() -> AsyncMock:
    """Create a mock ModelRouter for testing LLM interactions."""
    router = AsyncMock(spec=ModelRouter)
    router.route.return_value = "claude-3-5-sonnet-20241022"
    router.get_provider.return_value = AsyncMock()
    return router


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary project directory structure for testing."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create basic project structure
    src_dir = project_dir / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").touch()

    tests_dir = project_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").touch()

    # Create a basic pyproject.toml
    (project_dir / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "test-project"
version = "0.1.0"
"""
    )

    yield project_dir


@pytest.fixture
async def app_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async FastAPI test client for integration testing."""
    app = FastAPI()

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client
