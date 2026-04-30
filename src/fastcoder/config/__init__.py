"""Configuration management for the Autonomous Software Development Agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastcoder.config.llm_key_store import get_llm_key_store
from fastcoder.types.config import (
    AgentConfig,
    CostConfig,
    GitToolsConfig,
    LLMConfig,
    ModelConfig,
    ObservabilityConfig,
    ProjectConfig,
    ProviderConfig,
    QualityConfig,
    RoutingConfig,
    SafetyConfig,
    SandboxToolsConfig,
    ShellToolsConfig,
    ToolsConfig,
)


def load_config(overrides: Optional[dict] = None) -> AgentConfig:
    """Load AgentConfig from environment variables and/or config file.

    Loads sensible defaults matching the documentation:
    - Max iterations: 10, max retries: 5
    - Cost: $5/story, $100/day, $2000/month
    - Quality: 80% coverage, lint required
    - Safety: pre_deploy enabled, pre_production enabled (locked)
    - Anthropic provider with claude-sonnet-4-6 and claude-haiku models
    - OpenAI provider with gpt-4o and gpt-4o-mini
    - Ollama provider with llama3.2 (disabled by default)

    Args:
        overrides: Optional dict to override specific config values.

    Returns:
        AgentConfig with all settings loaded and applied.
    """
    overrides = overrides or {}

    # Load from file if exists
    config_file = Path(os.getenv("AGENT_CONFIG_FILE", ".agent.json"))
    file_config = {}
    if config_file.exists():
        try:
            with open(config_file) as f:
                file_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Merge priority: overrides > file_config > env > defaults
    # file_config may have settings at root level (legacy) or nested under
    # section keys (e.g., "project", "llm", "observability") when persisted
    # by the admin panel via _persist_config_non_secret.
    project_file = file_config.get("project", {})

    # Project config
    project_config = ProjectConfig(
        project_id=overrides.get(
            "project_id",
            project_file.get(
                "project_id",
                file_config.get(
                    "project_id",
                    os.getenv("AGENT_PROJECT_ID", "default"),
                ),
            ),
        ),
        project_dir=overrides.get(
            "project_dir",
            project_file.get(
                "project_dir",
                file_config.get(
                    "project_dir",
                    os.getenv("AGENT_PROJECT_DIR", "."),
                ),
            ),
        ),
        language=overrides.get(
            "language",
            project_file.get(
                "language",
                file_config.get(
                    "language",
                    os.getenv("AGENT_LANGUAGE", "python"),
                ),
            ),
        ),
        framework=overrides.get(
            "framework",
            project_file.get(
                "framework",
                file_config.get(
                    "framework",
                    os.getenv("AGENT_FRAMEWORK"),
                ),
            ),
        ),
        test_framework=overrides.get(
            "test_framework",
            project_file.get(
                "test_framework",
                file_config.get(
                    "test_framework",
                    os.getenv("AGENT_TEST_FRAMEWORK"),
                ),
            ),
        ),
        package_manager=overrides.get(
            "package_manager",
            project_file.get(
                "package_manager",
                file_config.get(
                    "package_manager",
                    os.getenv("AGENT_PACKAGE_MANAGER"),
                ),
            ),
        ),
        base_branch=overrides.get(
            "base_branch",
            project_file.get(
                "base_branch",
                file_config.get(
                    "base_branch",
                    os.getenv("AGENT_BASE_BRANCH", "main"),
                ),
            ),
        ),
    )

    # LLM config with defaults
    # API keys are loaded from the admin DB, not environment variables.
    providers = [
        ProviderConfig(
            name="anthropic",
            type="anthropic",
            api_key=None,
            enabled=False,
            models=[
                ModelConfig(
                    id="claude-sonnet-4-6",
                    tier="top",
                    max_context_tokens=200000,
                    cost_per_1k_input=0.003,
                    cost_per_1k_output=0.015,
                ),
                ModelConfig(
                    id="claude-haiku-4-5-20251001",
                    tier="low",
                    max_context_tokens=128000,
                    cost_per_1k_input=0.0008,
                    cost_per_1k_output=0.004,
                ),
            ],
        ),
        ProviderConfig(
            name="openai",
            type="openai",
            api_key=None,
            enabled=False,
            models=[
                ModelConfig(
                    id="gpt-4o",
                    tier="top",
                    max_context_tokens=128000,
                    cost_per_1k_input=0.005,
                    cost_per_1k_output=0.015,
                ),
                ModelConfig(
                    id="gpt-4o-mini",
                    tier="mid",
                    max_context_tokens=128000,
                    cost_per_1k_input=0.00015,
                    cost_per_1k_output=0.0006,
                ),
            ],
        ),
        ProviderConfig(
            name="google",
            type="gemini",
            api_key=None,
            enabled=False,
            models=[
                ModelConfig(
                    id="gemini-2-5-pro",
                    tier="top",
                    max_context_tokens=1000000,
                    cost_per_1k_input=0.00125,
                    cost_per_1k_output=0.01,
                ),
                ModelConfig(
                    id="gemini-2-5-flash",
                    tier="mid",
                    max_context_tokens=1000000,
                    cost_per_1k_input=0.00015,
                    cost_per_1k_output=0.0006,
                ),
                ModelConfig(
                    id="gemini-2-0-flash",
                    tier="low",
                    max_context_tokens=1000000,
                    cost_per_1k_input=0.0001,
                    cost_per_1k_output=0.0004,
                ),
            ],
        ),
        ProviderConfig(
            name="ollama",
            type="ollama",
            base_url="http://localhost:11434",
            enabled=False,
            models=[
                ModelConfig(
                    id="llama3.2",
                    tier="low",
                    max_context_tokens=128000,
                    cost_per_1k_input=0.0,
                    cost_per_1k_output=0.0,
                ),
            ],
        ),
    ]

    # Merge provider metadata from config file (never trust/persist api_key there)
    llm_file_config = file_config.get("llm", {})
    file_provider_overrides = llm_file_config.get("providers", [])
    if isinstance(file_provider_overrides, list):
        provider_by_name = {p.name: p for p in providers}
        for raw in file_provider_overrides:
            if not isinstance(raw, dict):
                continue
            name = raw.get("name")
            provider = provider_by_name.get(name)
            if not provider:
                continue

            if "type" in raw:
                provider.type = raw["type"]
            if "base_url" in raw:
                provider.base_url = raw["base_url"]
            if "enabled" in raw:
                provider.enabled = bool(raw["enabled"])
            if "models" in raw and isinstance(raw["models"], list):
                provider.models = [
                    ModelConfig(**m) for m in raw["models"] if isinstance(m, dict)
                ]

    # Inject provider API keys from admin DB.
    key_store = get_llm_key_store(project_config.project_dir)
    db_keys = key_store.get_all_keys()
    for provider in providers:
        key = db_keys.get(provider.name)
        if key:
            provider.api_key = key
            # Auto-enable cloud providers when key exists.
            if provider.type != "ollama":
                provider.enabled = True

    llm_config = LLMConfig(
        providers=providers,
        routing=RoutingConfig(
            default_tier=llm_file_config.get("routing", {}).get("default_tier", "mid"),
            default_provider=llm_file_config.get("routing", {}).get("default_provider"),
            default_model=llm_file_config.get("routing", {}).get("default_model"),
            task_overrides=overrides.get(
                "task_overrides",
                llm_file_config.get("routing", {}).get("task_overrides", {}),
            ),
            fallback_chain=llm_file_config.get("routing", {}).get(
                "fallback_chain",
                ["anthropic", "openai"],
            ),
        ),
        cost=CostConfig(
            max_cost_per_story_usd=overrides.get(
                "max_cost_per_story_usd",
                llm_file_config.get(
                    "cost",
                    {},
                ).get(
                    "max_cost_per_story_usd",
                    file_config.get("max_cost_per_story_usd", 5.0),
                ),
            ),
            daily_budget_usd=overrides.get(
                "daily_budget_usd",
                llm_file_config.get(
                    "cost",
                    {},
                ).get(
                    "daily_budget_usd",
                    file_config.get("daily_budget_usd", 100.0),
                ),
            ),
            monthly_budget_usd=overrides.get(
                "monthly_budget_usd",
                llm_file_config.get(
                    "cost",
                    {},
                ).get(
                    "monthly_budget_usd",
                    file_config.get("monthly_budget_usd", 2000.0),
                ),
            ),
            warning_threshold=llm_file_config.get("cost", {}).get(
                "warning_threshold", 0.8
            ),
            escalation_action=llm_file_config.get("cost", {}).get(
                "escalation_action", "pause"
            ),
        ),
    )

    # Tools config
    tools_config = ToolsConfig(
        sandbox=SandboxToolsConfig(),
        git=GitToolsConfig(),
        shell=ShellToolsConfig(),
    )

    # Safety config
    safety_file = file_config.get("safety", {})
    safety_config = SafetyConfig(
        max_iterations_per_story=overrides.get(
            "max_iterations",
            safety_file.get(
                "max_iterations_per_story",
                file_config.get("max_iterations", 10),
            ),
        ),
        max_retries_per_stage=overrides.get(
            "max_retries",
            safety_file.get(
                "max_retries_per_stage",
                file_config.get("max_retries", 5),
            ),
        ),
        approval_gates=safety_file.get(
            "approval_gates",
            {
                "pre_code": False,
                "pre_deploy": True,
                "pre_production": True,
                "budget_exceeded": True,
                "ambiguity_detected": True,
            },
        ),
    )

    # Quality config
    quality_file = file_config.get("quality", {})
    quality_config = QualityConfig(
        min_test_coverage=overrides.get(
            "min_test_coverage",
            quality_file.get(
                "min_test_coverage",
                file_config.get("min_test_coverage", 80.0),
            ),
        ),
        lint_required=quality_file.get("lint_required", True),
    )

    # Observability config
    obs_file = file_config.get("observability", {})
    observability_config = ObservabilityConfig(
        log_level=overrides.get(
            "log_level",
            obs_file.get(
                "log_level",
                file_config.get(
                    "log_level",
                    os.getenv("AGENT_LOG_LEVEL", "info"),
                ),
            ),
        ),
        trace_llm_calls=obs_file.get("trace_llm_calls", True),
        metrics_enabled=obs_file.get("metrics_enabled", True),
        audit_log_enabled=obs_file.get("audit_log_enabled", True),
    )

    config = AgentConfig(
        project=project_config,
        llm=llm_config,
        tools=tools_config,
        safety=safety_config,
        quality=quality_config,
        observability=observability_config,
    )

    return config


def validate_config(config: AgentConfig) -> list[str]:
    """Validate AgentConfig and return any errors.

    Checks:
    - API keys are set for enabled providers
    - Cost budgets are positive
    - Quality thresholds are reasonable
    - Max iterations and retries are positive

    Args:
        config: AgentConfig to validate.

    Returns:
        List of validation error strings. Empty if valid.
    """
    errors = []

    # Check providers
    enabled_providers = [p for p in config.llm.providers if p.enabled]

    for provider in enabled_providers:
        if provider.type != "ollama" and not provider.api_key:
            # Auto-disable providers without keys instead of blocking startup
            provider.enabled = False
            continue
        if not provider.models:
            errors.append(
                f"Provider '{provider.name}' has no models configured"
            )

    # Check costs
    if config.llm.cost.max_cost_per_story_usd <= 0:
        errors.append("max_cost_per_story_usd must be positive")
    if config.llm.cost.daily_budget_usd <= 0:
        errors.append("daily_budget_usd must be positive")
    if config.llm.cost.monthly_budget_usd <= 0:
        errors.append("monthly_budget_usd must be positive")

    # Check quality
    if not 0 <= config.quality.min_test_coverage <= 100:
        errors.append("min_test_coverage must be between 0 and 100")

    # Check safety
    if config.safety.max_iterations_per_story <= 0:
        errors.append("max_iterations_per_story must be positive")
    if config.safety.max_retries_per_stage <= 0:
        errors.append("max_retries_per_stage must be positive")

    return errors
