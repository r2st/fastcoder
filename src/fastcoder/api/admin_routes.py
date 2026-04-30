"""Admin API routes for managing agent configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fastcoder.config.llm_key_store import get_llm_key_store, _VALID_PROVIDERS
from fastcoder.config import load_config, validate_config
from fastcoder.types.config import (
    AgentConfig,
    CostConfig,
    GitToolsConfig,
    GraphifyConfig,
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


# Helper function to mask API keys
def mask_api_key(api_key: Optional[str]) -> Optional[str]:
    """Mask API key showing only last 4 characters.

    Args:
        api_key: The API key to mask, or None.

    Returns:
        Masked key (e.g., "sk-...xxxx") or None if input is None.
    """
    if not api_key:
        return None
    if len(api_key) <= 4:
        return "***"
    return f"{api_key[0]}-...{api_key[-4:]}"


def mask_config_api_keys(config: AgentConfig) -> AgentConfig:
    """Return a copy of config with all API keys masked.

    Args:
        config: The AgentConfig to mask.

    Returns:
        New AgentConfig with masked API keys in providers.
    """
    config_dict = config.model_dump()

    # Mask provider API keys
    if "llm" in config_dict and "providers" in config_dict["llm"]:
        for provider in config_dict["llm"]["providers"]:
            if "api_key" in provider:
                provider["api_key"] = mask_api_key(provider["api_key"])

    return AgentConfig(**config_dict)


# Response models for better API documentation
class ConfigResponse(BaseModel):
    """Response model for config endpoints."""

    success: bool = True
    data: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    message: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response for config validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)


def create_admin_router(config_holder: dict) -> APIRouter:
    """Create and return the admin API router with all config endpoints.

    Args:
        config_holder: Dict with key "config" holding the current AgentConfig.

    Returns:
        Configured APIRouter for admin endpoints.
    """
    router = APIRouter(prefix="/api/v1/admin", tags=["admin", "config"])

    def _sync_provider_keys_from_store(config: AgentConfig) -> None:
        """Hydrate provider api_key fields from persistent key storage."""
        key_store = get_llm_key_store(config.project.project_dir)
        keys = key_store.get_all_keys()
        for provider in config.llm.providers:
            provider.api_key = keys.get(provider.name)
            if provider.type != "ollama" and provider.api_key:
                provider.enabled = True

    def _persist_config_non_secret(config: AgentConfig) -> None:
        """Persist config to .agent.json without provider API keys."""
        config_file = Path(".agent.json")
        config_dict = config.model_dump()
        if "llm" in config_dict and "providers" in config_dict["llm"]:
            for provider in config_dict["llm"]["providers"]:
                provider.pop("api_key", None)

        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    # GET /config - return full AgentConfig with masked API keys
    @router.get("/config", response_model=dict)
    async def get_config() -> dict:
        """Get the full AgentConfig with API keys masked.

        Returns:
            Full AgentConfig as dict with masked API keys.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")
        _sync_provider_keys_from_store(config)

        masked_config = mask_config_api_keys(config)
        return masked_config.model_dump()

    # PUT /config - update full AgentConfig
    @router.put("/config", response_model=dict)
    async def update_config(body: dict) -> dict:
        """Update the full AgentConfig from JSON body.

        Validates config after update. Returns updated config with masked keys.

        Args:
            body: Dict containing updated config fields.

        Returns:
            Updated AgentConfig as dict.

        Raises:
            HTTPException: 400 if validation fails.
        """
        try:
            # Create new config from body
            new_config = AgentConfig(**body)

            # Validate
            errors = validate_config(new_config)
            if errors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Config validation failed: {'; '.join(errors)}",
                )

            # Update holder
            config_holder["config"] = new_config

            # Return masked version
            masked_config = mask_config_api_keys(new_config)
            return masked_config.model_dump()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # GET /config/project - return ProjectConfig
    @router.get("/config/project", response_model=dict)
    async def get_project_config() -> dict:
        """Get ProjectConfig.

        Returns:
            ProjectConfig as dict.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        return config.project.model_dump()

    # PUT /config/project - update ProjectConfig
    @router.put("/config/project", response_model=dict)
    async def update_project_config(body: dict) -> dict:
        """Update ProjectConfig.

        Args:
            body: Dict with project config updates.

        Returns:
            Updated ProjectConfig as dict.

        Raises:
            HTTPException: 400 if update fails.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            # Merge update with existing
            project_dict = config.project.model_dump()
            project_dict.update(body)
            config.project = ProjectConfig(**project_dict)

            _persist_config_non_secret(config)
            return config.project.model_dump()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # GET /config/server - return server-level settings
    @router.get("/config/server", response_model=dict)
    async def get_server_config() -> dict:
        """Get server-level settings (ports, log level, paths).

        Returns:
            Dict with server settings.
        """
        config = config_holder.get("config")
        return {
            "log_level": config.observability.log_level if config else os.environ.get("AGENT_LOG_LEVEL", "info"),
            "config_file": os.environ.get("AGENT_CONFIG_FILE", ".agent.json"),
            "api_port": int(os.environ.get("AGENT_PORT", "3000")),
            "admin_port": int(os.environ.get("AGENT_ADMIN_PORT", "3001")),
            "cors_origins": os.environ.get("AGENT_CORS_ORIGINS", ""),
            "admin_db_path": os.environ.get("AGENT_ADMIN_DB_PATH", ".agent_admin.db"),
        }

    # PUT /config/server - update server-level settings
    @router.put("/config/server", response_model=dict)
    async def update_server_config(body: dict) -> dict:
        """Update server-level settings.

        Updates are applied to the running config and persisted to .agent.json.
        Port/DB path changes require a restart to take effect.

        Args:
            body: Dict with server settings to update.

        Returns:
            Updated server settings dict.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        restart_required = False

        if "log_level" in body:
            level = body["log_level"].lower()
            if level in ("debug", "info", "warning", "error", "critical"):
                config.observability.log_level = level
                os.environ["AGENT_LOG_LEVEL"] = level
            else:
                raise HTTPException(status_code=400, detail=f"Invalid log level: {level}")

        if "config_file" in body:
            os.environ["AGENT_CONFIG_FILE"] = body["config_file"]

        if "api_port" in body:
            os.environ["AGENT_PORT"] = str(int(body["api_port"]))
            restart_required = True

        if "admin_port" in body:
            os.environ["AGENT_ADMIN_PORT"] = str(int(body["admin_port"]))
            restart_required = True

        if "cors_origins" in body:
            os.environ["AGENT_CORS_ORIGINS"] = body["cors_origins"]
            restart_required = True

        if "admin_db_path" in body:
            os.environ["AGENT_ADMIN_DB_PATH"] = body["admin_db_path"]
            restart_required = True

        _persist_config_non_secret(config)

        return {
            "log_level": config.observability.log_level,
            "config_file": os.environ.get("AGENT_CONFIG_FILE", ".agent.json"),
            "api_port": int(os.environ.get("AGENT_PORT", "3000")),
            "admin_port": int(os.environ.get("AGENT_ADMIN_PORT", "3001")),
            "cors_origins": os.environ.get("AGENT_CORS_ORIGINS", ""),
            "admin_db_path": os.environ.get("AGENT_ADMIN_DB_PATH", ".agent_admin.db"),
            "restart_required": restart_required,
        }

    # GET /config/llm - return LLMConfig with masked keys
    @router.get("/config/llm", response_model=dict)
    async def get_llm_config() -> dict:
        """Get LLMConfig with API keys masked.

        Returns:
            LLMConfig as dict with masked API keys.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")
        _sync_provider_keys_from_store(config)

        llm_dict = config.llm.model_dump()

        # Mask provider API keys
        for provider in llm_dict.get("providers", []):
            if "api_key" in provider:
                provider["api_key"] = mask_api_key(provider["api_key"])

        return llm_dict

    # PUT /config/llm/providers - update provider list
    @router.put("/config/llm/providers", response_model=dict)
    async def update_llm_providers(body: list[dict]) -> dict:
        """Update LLM providers list.

        Args:
            body: List of ProviderConfig dicts.

        Returns:
            Updated LLMConfig as dict.

        Raises:
            HTTPException: 400 if update fails.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            # Create new providers
            providers = [ProviderConfig(**p) for p in body]
            config.llm.providers = providers

            # Validate
            errors = validate_config(config)
            if errors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Config validation failed: {'; '.join(errors)}",
                )

            # Return masked
            llm_dict = config.llm.model_dump()
            for provider in llm_dict.get("providers", []):
                if "api_key" in provider:
                    provider["api_key"] = mask_api_key(provider["api_key"])

            return llm_dict

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # PUT /config/llm/providers/{provider_name}/toggle - enable/disable provider
    @router.put("/config/llm/providers/{provider_name}/toggle", response_model=dict)
    async def toggle_provider(provider_name: str) -> dict:
        """Enable or disable a specific provider.

        Toggles the enabled flag for the named provider.

        Args:
            provider_name: Name of the provider to toggle.

        Returns:
            Updated provider dict.

        Raises:
            HTTPException: 404 if provider not found, 400 if validation fails.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")
        _sync_provider_keys_from_store(config)

        # Find provider
        provider = None
        for p in config.llm.providers:
            if p.name == provider_name:
                provider = p
                break

        if not provider:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{provider_name}' not found",
            )

        # Toggle
        provider.enabled = not provider.enabled

        # Validate
        errors = validate_config(config)
        if errors:
            # Restore and return error
            provider.enabled = not provider.enabled
            raise HTTPException(
                status_code=400,
                detail=f"Config validation failed: {'; '.join(errors)}",
            )

        provider_dict = provider.model_dump()
        if "api_key" in provider_dict:
            provider_dict["api_key"] = mask_api_key(provider_dict["api_key"])

        # Persist immediately so provider enable/disable survives restart.
        _persist_config_non_secret(config)

        return provider_dict

    # PUT /config/llm/routing - update routing config
    @router.put("/config/llm/routing", response_model=dict)
    async def update_llm_routing(body: dict) -> dict:
        """Update LLM routing config.

        Args:
            body: Dict with routing config updates.

        Returns:
            Updated RoutingConfig as dict.

        Raises:
            HTTPException: 400 if update fails.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            # Merge with existing
            routing_dict = config.llm.routing.model_dump()
            routing_dict.update(body)
            config.llm.routing = RoutingConfig(**routing_dict)

            _persist_config_non_secret(config)
            return config.llm.routing.model_dump()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # PUT /config/llm/cost - update cost config
    @router.put("/config/llm/cost", response_model=dict)
    async def update_llm_cost(body: dict) -> dict:
        """Update LLM cost config.

        Args:
            body: Dict with cost config updates.

        Returns:
            Updated CostConfig as dict.

        Raises:
            HTTPException: 400 if update fails.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            # Merge with existing
            cost_dict = config.llm.cost.model_dump()
            cost_dict.update(body)
            config.llm.cost = CostConfig(**cost_dict)

            # Validate
            errors = validate_config(config)
            if errors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Config validation failed: {'; '.join(errors)}",
                )

            _persist_config_non_secret(config)
            return config.llm.cost.model_dump()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # GET /config/tools - return ToolsConfig
    @router.get("/config/tools", response_model=dict)
    async def get_tools_config() -> dict:
        """Get ToolsConfig.

        Returns:
            ToolsConfig as dict.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        return config.tools.model_dump()

    # PUT /config/tools - update ToolsConfig
    @router.put("/config/tools", response_model=dict)
    async def update_tools_config(body: dict) -> dict:
        """Update ToolsConfig.

        Args:
            body: Dict with tools config updates.

        Returns:
            Updated ToolsConfig as dict.

        Raises:
            HTTPException: 400 if update fails.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            # Handle nested configs
            tools_dict = config.tools.model_dump()

            # Merge updates
            if "sandbox" in body:
                tools_dict["sandbox"].update(body["sandbox"])
            if "git" in body:
                tools_dict["git"].update(body["git"])
            if "shell" in body:
                tools_dict["shell"].update(body["shell"])

            config.tools = ToolsConfig(**tools_dict)

            return config.tools.model_dump()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # GET /config/safety - return SafetyConfig
    @router.get("/config/safety", response_model=dict)
    async def get_safety_config() -> dict:
        """Get SafetyConfig.

        Returns:
            SafetyConfig as dict.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        return config.safety.model_dump()

    # PUT /config/safety - update SafetyConfig
    @router.put("/config/safety", response_model=dict)
    async def update_safety_config(body: dict) -> dict:
        """Update SafetyConfig.

        Args:
            body: Dict with safety config updates.

        Returns:
            Updated SafetyConfig as dict.

        Raises:
            HTTPException: 400 if update fails.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            # Merge with existing
            safety_dict = config.safety.model_dump()
            safety_dict.update(body)
            config.safety = SafetyConfig(**safety_dict)

            # Validate
            errors = validate_config(config)
            if errors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Config validation failed: {'; '.join(errors)}",
                )

            _persist_config_non_secret(config)
            return config.safety.model_dump()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # GET /config/quality - return QualityConfig
    @router.get("/config/quality", response_model=dict)
    async def get_quality_config() -> dict:
        """Get QualityConfig.

        Returns:
            QualityConfig as dict.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        return config.quality.model_dump()

    # PUT /config/quality - update QualityConfig
    @router.put("/config/quality", response_model=dict)
    async def update_quality_config(body: dict) -> dict:
        """Update QualityConfig.

        Args:
            body: Dict with quality config updates.

        Returns:
            Updated QualityConfig as dict.

        Raises:
            HTTPException: 400 if update fails.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            # Merge with existing
            quality_dict = config.quality.model_dump()
            quality_dict.update(body)
            config.quality = QualityConfig(**quality_dict)

            # Validate
            errors = validate_config(config)
            if errors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Config validation failed: {'; '.join(errors)}",
                )

            _persist_config_non_secret(config)
            return config.quality.model_dump()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # GET /config/observability - return ObservabilityConfig
    @router.get("/config/observability", response_model=dict)
    async def get_observability_config() -> dict:
        """Get ObservabilityConfig.

        Returns:
            ObservabilityConfig as dict.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        return config.observability.model_dump()

    # PUT /config/observability - update ObservabilityConfig
    @router.put("/config/observability", response_model=dict)
    async def update_observability_config(body: dict) -> dict:
        """Update ObservabilityConfig.

        Args:
            body: Dict with observability config updates.

        Returns:
            Updated ObservabilityConfig as dict.

        Raises:
            HTTPException: 400 if update fails.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            # Merge with existing
            obs_dict = config.observability.model_dump()
            obs_dict.update(body)
            config.observability = ObservabilityConfig(**obs_dict)

            _persist_config_non_secret(config)
            return config.observability.model_dump()

        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # GET /config/graphify - return GraphifyConfig
    @router.get("/config/graphify", response_model=dict)
    async def get_graphify_config() -> dict:
        """Get GraphifyConfig.

        Returns:
            GraphifyConfig as dict, plus runtime status fields:
              - installed: whether the `graphify` package is importable
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        try:
            import graphify  # noqa: F401
            installed = True
        except ImportError:
            installed = False

        return {
            **config.graphify.model_dump(),
            "installed": installed,
        }

    # PUT /config/graphify - update GraphifyConfig
    @router.put("/config/graphify", response_model=dict)
    async def update_graphify_config(body: dict) -> dict:
        """Update GraphifyConfig.

        Args:
            body: Dict with graphify config updates (enabled, min_corpus_words,
                  auto_rebuild_on_commit, semantic_extraction, query_token_budget,
                  cache_dir).

        Returns:
            Updated GraphifyConfig as dict.

        Raises:
            HTTPException: 400 if update fails or graphify package is not
                           installed when enabling.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            # Block enable if package missing — prevents silent no-op at runtime.
            if body.get("enabled") is True:
                try:
                    import graphify  # noqa: F401
                except ImportError:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Graphify package not installed. "
                            "Run: pip install 'fastcoder[graphify]'"
                        ),
                    )

            # Merge with existing — strip runtime-only fields the client may echo back.
            graphify_dict = config.graphify.model_dump()
            graphify_dict.update({k: v for k, v in body.items() if k != "installed"})
            config.graphify = GraphifyConfig(**graphify_dict)

            _persist_config_non_secret(config)
            return config.graphify.model_dump()

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")

    # ── Secure API Key Management ──

    @router.put("/config/llm/providers/{provider_name}/key")
    async def set_provider_api_key(provider_name: str, body: dict) -> dict:
        """Securely set an API key for a provider.

        The key is persisted in the admin database.
        The response confirms success and shows a masked version.

        Args:
            provider_name: Provider name (anthropic, openai, google, ollama).
            body: Dict with "api_key" field.

        Returns:
            Dict with provider name, masked key, and key_set flag.
        """
        # Validate provider name against whitelist to prevent injection
        if provider_name not in _VALID_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider name. Must be one of: {', '.join(sorted(_VALID_PROVIDERS))}",
            )

        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        api_key = body.get("api_key", "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="api_key is required and cannot be empty")
        # Security: reject excessively long keys (max 512 chars) and control characters
        if len(api_key) > 512:
            raise HTTPException(status_code=400, detail="API key too long (max 512 characters)")
        if any(not c.isprintable() for c in api_key):
            raise HTTPException(status_code=400, detail="API key contains invalid characters")

        # Find provider
        provider = None
        for p in config.llm.providers:
            if p.name == provider_name:
                provider = p
                break

        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

        # Validate key format (basic sanity checks per provider)
        validation_error = _validate_api_key_format(provider.type, api_key)
        if validation_error:
            raise HTTPException(status_code=400, detail=validation_error)

        # Persist key to DB and mirror in current runtime config
        key_store = get_llm_key_store(config.project.project_dir)
        key_store.set_key(provider_name, api_key)
        provider.api_key = api_key

        # Auto-enable the provider when a key is set
        if not provider.enabled:
            provider.enabled = True

        return {
            "provider": provider_name,
            "key_set": True,
            "masked_key": mask_api_key(api_key),
            "enabled": provider.enabled,
            "message": f"API key for {provider_name} set successfully",
        }

    @router.delete("/config/llm/providers/{provider_name}/key")
    async def clear_provider_api_key(provider_name: str) -> dict:
        """Clear/remove an API key for a provider.

        Also disables the provider since it can no longer make calls.

        Args:
            provider_name: Provider name.

        Returns:
            Dict confirming key removal.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")
        _sync_provider_keys_from_store(config)

        provider = None
        for p in config.llm.providers:
            if p.name == provider_name:
                provider = p
                break

        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

        key_store = get_llm_key_store(config.project.project_dir)
        key_store.clear_key(provider_name)
        provider.api_key = None
        provider.enabled = False

        return {
            "provider": provider_name,
            "key_set": False,
            "enabled": False,
            "message": f"API key for {provider_name} cleared. Provider disabled.",
        }

    @router.get("/config/llm/providers/{provider_name}/key/status")
    async def get_provider_key_status(provider_name: str) -> dict:
        """Check whether an API key is set for a provider (without revealing it).

        Args:
            provider_name: Provider name.

        Returns:
            Dict with key_set, masked_key, enabled, and provider info.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")
        _sync_provider_keys_from_store(config)

        provider = None
        for p in config.llm.providers:
            if p.name == provider_name:
                provider = p
                break

        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

        has_key = bool(provider.api_key)
        return {
            "provider": provider_name,
            "type": provider.type,
            "key_set": has_key,
            "masked_key": mask_api_key(provider.api_key) if has_key else None,
            "enabled": provider.enabled,
            "models": [m.model_dump() for m in provider.models],
        }

    @router.post("/config/llm/providers/{provider_name}/key/validate")
    async def validate_provider_api_key(provider_name: str) -> dict:
        """Validate that the stored API key works by making a lightweight test call.

        Args:
            provider_name: Provider name.

        Returns:
            Dict with valid flag, provider name, and error if any.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")
        _sync_provider_keys_from_store(config)

        provider = None
        for p in config.llm.providers:
            if p.name == provider_name:
                provider = p
                break

        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

        if not provider.api_key:
            return {
                "provider": provider_name,
                "valid": False,
                "error": "No API key set",
            }

        # Attempt a lightweight validation per provider type
        try:
            valid, error = await _test_provider_key(provider)
            return {
                "provider": provider_name,
                "valid": valid,
                "error": error,
            }
        except Exception as e:
            return {
                "provider": provider_name,
                "valid": False,
                "error": str(e),
            }

    @router.get("/config/llm/keys/summary")
    async def get_all_keys_summary() -> dict:
        """Get a summary of all provider key statuses and the default LLM.

        Returns:
            Dict with providers list (name, type, key_set, masked_key, enabled, models)
            and default_provider/default_model from routing config.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")
        _sync_provider_keys_from_store(config)

        providers = []
        for p in config.llm.providers:
            has_key = bool(p.api_key)
            providers.append({
                "name": p.name,
                "type": p.type,
                "key_set": has_key,
                "masked_key": mask_api_key(p.api_key) if has_key else None,
                "enabled": p.enabled,
                "models": [m.model_dump() for m in p.models],
            })

        return {
            "providers": providers,
            "default_provider": config.llm.routing.default_provider,
            "default_model": config.llm.routing.default_model,
            "default_tier": config.llm.routing.default_tier,
            "fallback_chain": config.llm.routing.fallback_chain,
        }

    # ── Default LLM Selection ──

    @router.put("/config/llm/default")
    async def set_default_llm(body: dict) -> dict:
        """Set the default LLM provider and model.

        Args:
            body: Dict with optional "provider" and "model" fields.

        Returns:
            Updated routing config with default provider/model.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        provider_name = body.get("provider")
        model_id = body.get("model")
        tier = body.get("tier")

        # Validate provider exists and is enabled
        if provider_name:
            provider = None
            for p in config.llm.providers:
                if p.name == provider_name:
                    provider = p
                    break
            if not provider:
                raise HTTPException(
                    status_code=404,
                    detail=f"Provider '{provider_name}' not found",
                )
            if not provider.enabled:
                provider.enabled = True
            # Validate model belongs to this provider
            if model_id:
                valid_models = [m.id for m in provider.models]
                if model_id not in valid_models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model '{model_id}' not found in provider '{provider_name}'. Available: {valid_models}",
                    )
            config.llm.routing.default_provider = provider_name

        if model_id:
            config.llm.routing.default_model = model_id

        if tier and tier in ("low", "mid", "top"):
            config.llm.routing.default_tier = tier

        _persist_config_non_secret(config)

        return {
            "default_provider": config.llm.routing.default_provider,
            "default_model": config.llm.routing.default_model,
            "default_tier": config.llm.routing.default_tier,
            "message": "Default LLM updated successfully",
        }

    @router.get("/config/llm/default")
    async def get_default_llm() -> dict:
        """Get the current default LLM provider and model.

        Returns:
            Dict with default_provider, default_model, default_tier.
        """
        config = config_holder.get("config")
        if not config:
            raise HTTPException(status_code=500, detail="Config not initialized")

        return {
            "default_provider": config.llm.routing.default_provider,
            "default_model": config.llm.routing.default_model,
            "default_tier": config.llm.routing.default_tier,
        }

    # POST /config/save - save config to .agent.json
    @router.post("/config/save", response_model=dict)
    async def save_config() -> dict:
        """Save current config to .agent.json file.

        Returns:
            Dict with success status and file path.

        Raises:
            HTTPException: 500 if save fails.
        """
        try:
            config = config_holder.get("config")
            if not config:
                raise HTTPException(status_code=500, detail="Config not initialized")

            config_file = Path(".agent.json")
            _persist_config_non_secret(config)

            return {
                "success": True,
                "message": f"Config saved to {config_file.absolute()}",
                "path": str(config_file.absolute()),
            }

        except IOError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save config: {str(e)}",
            )

    # POST /config/reload - reload config from file
    @router.post("/config/reload", response_model=dict)
    async def reload_config() -> dict:
        """Reload config from .agent.json file.

        Returns:
            Reloaded config with masked API keys.

        Raises:
            HTTPException: 500 if reload fails, 400 if validation fails.
        """
        try:
            reloaded_config = load_config()

            # Validate
            errors = validate_config(reloaded_config)
            if errors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Loaded config is invalid: {'; '.join(errors)}",
                )

            # Update holder
            config_holder["config"] = reloaded_config

            # Return masked
            masked_config = mask_config_api_keys(reloaded_config)
            return {
                "success": True,
                "message": "Config reloaded from file",
                "config": masked_config.model_dump(),
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reload config: {str(e)}",
            )

    return router


def _validate_api_key_format(provider_type: str, api_key: str) -> Optional[str]:
    """Basic format validation for API keys.

    Args:
        provider_type: The provider type (anthropic, openai, gemini, ollama).
        api_key: The key to validate.

    Returns:
        Error message string if invalid, None if OK.
    """
    if len(api_key) < 8:
        return "API key is too short (minimum 8 characters)"

    if provider_type == "anthropic":
        if not api_key.startswith("sk-ant-"):
            return "Anthropic keys should start with 'sk-ant-'"
    elif provider_type == "openai":
        if not api_key.startswith("sk-"):
            return "OpenAI keys should start with 'sk-'"
    elif provider_type == "gemini":
        # Google API keys are typically 39 chars, start with AIza
        if len(api_key) < 20:
            return "Google API key seems too short"
    # ollama doesn't need a key
    elif provider_type == "ollama":
        return "Ollama does not require an API key"

    return None


async def _test_provider_key(provider: "ProviderConfig") -> tuple[bool, Optional[str]]:
    """Test a provider's API key with a lightweight call.

    Args:
        provider: ProviderConfig with key and type.

    Returns:
        Tuple of (valid: bool, error: Optional[str]).
    """
    try:
        if provider.type == "anthropic":
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=provider.api_key)
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return (True, None)

        elif provider.type == "openai":
            import openai

            client = openai.AsyncOpenAI(api_key=provider.api_key)
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
            return (True, None)

        elif provider.type == "gemini":
            from google import genai

            client = genai.Client(api_key=provider.api_key)
            resp = client.models.list()
            return (True, None)

        elif provider.type == "ollama":
            import ollama

            client = ollama.AsyncClient(
                host=provider.base_url or "http://localhost:11434"
            )
            resp = await client.list()
            return (True, None)

        else:
            return (False, f"Unknown provider type: {provider.type}")

    except Exception as e:
        error_msg = str(e)
        # Log full error server-side for debugging
        import logging
        logging.getLogger(__name__).warning(
            "api_key_validation_failed",
            extra={"provider": provider.type, "error": error_msg},
        )
        # Return sanitized error to client — never expose raw exception details
        if "401" in error_msg or "auth" in error_msg.lower() or "invalid" in error_msg.lower():
            return (False, "Invalid API key — authentication failed")
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return (False, "Connection failed — check provider endpoint")
        if "404" in error_msg or "not found" in error_msg.lower():
            return (False, "Provider endpoint not found — check configuration")
        return (False, "Validation failed — check key and provider settings")
