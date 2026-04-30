"""Execution plan types — produced by the Planner component."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskAction(str, Enum):
    CREATE_FILE = "create_file"
    MODIFY_FILE = "modify_file"
    DELETE_FILE = "delete_file"
    RUN_COMMAND = "run_command"


class TestingStrategy(str, Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    UNIT_INTEGRATION = "unit + integration"
    UNIT_INTEGRATION_E2E = "unit + integration + e2e"


class DeployStrategy(str, Enum):
    STAGING_FIRST = "staging_first"
    DIRECT_DEPLOY = "direct_deploy"
    PR_ONLY = "pr_only"
    NONE = "none"


class PlanTask(BaseModel):
    id: str
    action: TaskAction
    target: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    estimated_tokens: int = 2000


class ExecutionPlan(BaseModel):
    story_id: str
    tasks: list[PlanTask] = Field(default_factory=list)
    testing_strategy: TestingStrategy = TestingStrategy.UNIT_INTEGRATION
    deploy_strategy: DeployStrategy = DeployStrategy.STAGING_FIRST
    estimated_total_tokens: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    revision: int = 1
