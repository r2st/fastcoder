"""Agent configuration types."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    id: str
    tier: str = "mid"  # low, mid, top
    max_context_tokens: int = 128000
    cost_per_1k_input: float = 0.003
    cost_per_1k_output: float = 0.015


class ProviderConfig(BaseModel):
    name: str
    type: str  # anthropic, openai, ollama
    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    base_url: Optional[str] = None
    models: list[ModelConfig] = Field(default_factory=list)
    enabled: bool = True

    def __repr__(self) -> str:
        """Custom repr that redacts api_key to prevent accidental exposure in logs."""
        key_display = "***" if self.api_key else "None"
        return f"ProviderConfig(name={self.name!r}, type={self.type!r}, api_key={key_display}, enabled={self.enabled})"


class RoutingConfig(BaseModel):
    default_tier: str = "mid"
    default_provider: Optional[str] = None  # e.g. "anthropic", "openai", "google", "ollama"
    default_model: Optional[str] = None  # e.g. "claude-sonnet-4-6", "gpt-4o"
    task_overrides: dict[str, str] = Field(default_factory=dict)
    fallback_chain: list[str] = Field(default_factory=list)


class CostConfig(BaseModel):
    max_cost_per_story_usd: float = 5.0
    daily_budget_usd: float = 100.0
    monthly_budget_usd: float = 2000.0
    warning_threshold: float = 0.8
    escalation_action: str = "pause"  # pause, downgrade_model, notify_and_continue


class LLMConfig(BaseModel):
    providers: list[ProviderConfig] = Field(default_factory=list)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cost: CostConfig = Field(default_factory=CostConfig)


class SandboxToolsConfig(BaseModel):
    cpu_limit: str = "2"
    memory_limit_mb: int = 2048
    disk_limit_mb: int = 10240
    default_timeout_ms: int = 300000
    network_whitelist: list[str] = Field(
        default_factory=lambda: ["registry.npmjs.org", "pypi.org", "github.com"]
    )


class GitToolsConfig(BaseModel):
    auto_commit: bool = True
    commit_convention: str = "conventional"
    auto_pr: bool = True


class ShellToolsConfig(BaseModel):
    command_allowlist: list[str] = Field(
        default_factory=lambda: [
            "npm", "npx", "yarn", "pnpm", "pip", "python", "node",
            "pytest", "jest", "vitest", "eslint", "ruff", "mypy",
            "tsc", "cargo", "go", "make", "docker", "git",
            "cat", "ls", "find", "grep", "wc", "head", "tail",
            "mkdir", "cp", "mv", "rm", "touch", "chmod",
        ]
    )
    max_execution_time_ms: int = 300000


class ToolsConfig(BaseModel):
    sandbox: SandboxToolsConfig = Field(default_factory=SandboxToolsConfig)
    git: GitToolsConfig = Field(default_factory=GitToolsConfig)
    shell: ShellToolsConfig = Field(default_factory=ShellToolsConfig)


class SafetyConfig(BaseModel):
    max_iterations_per_story: int = 10
    max_retries_per_stage: int = 5
    approval_gates: dict[str, bool] = Field(
        default_factory=lambda: {
            "pre_code": False,
            "pre_deploy": True,
            "pre_production": True,
            "budget_exceeded": True,
            "ambiguity_detected": True,
        }
    )
    secret_scanning: bool = True
    sast_enabled: bool = True
    dependency_audit: bool = True


class QualityConfig(BaseModel):
    min_test_coverage: float = 80.0
    lint_required: bool = True
    type_check_required: bool = True
    sast_required: bool = True
    max_complexity: int = 15
    max_nesting_depth: int = 4


class ObservabilityConfig(BaseModel):
    log_level: str = "info"
    trace_llm_calls: bool = True
    metrics_enabled: bool = True
    audit_log_enabled: bool = True


class ProjectConfig(BaseModel):
    project_id: str = "default"
    project_dir: str = "."
    language: str = "python"
    framework: Optional[str] = None
    test_framework: Optional[str] = None
    package_manager: Optional[str] = None
    base_branch: str = "main"


class AgentConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
