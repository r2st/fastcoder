"""Intelligent model routing and provider selection."""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastcoder.types.config import ProviderConfig, RoutingConfig
from fastcoder.types.llm import (
    CompletionRequest,
    CompletionResponse,
    HealthStatus,
    ModelTier,
    RoutingContext,
    SelectedModel,
    TaskPurpose,
)

from .circuit_breaker import CircuitBreakerRegistry
from .providers import (
    LLMProvider,
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    OllamaProvider,
)

logger = structlog.get_logger(__name__)


@dataclass
class CostTracker:
    """Track costs per story."""

    story_id: Optional[str] = None
    total_cost_usd: float = 0.0
    call_count: int = 0

    def add_cost(self, cost_usd: float) -> None:
        """Record cost for a call."""
        self.total_cost_usd += cost_usd
        self.call_count += 1


class ModelRouter:
    """Intelligent router for selecting and routing to LLM providers."""

    # Routing rules for different tasks
    TASK_TIER_RULES: Dict[TaskPurpose, ModelTier] = {
        TaskPurpose.STORY_ANALYSIS: ModelTier.MID,
        TaskPurpose.PLANNING: ModelTier.TOP,
        TaskPurpose.CODE_GENERATION: ModelTier.TOP,  # Will adjust based on complexity
        TaskPurpose.TEST_GENERATION: ModelTier.MID,
        TaskPurpose.CODE_REVIEW: ModelTier.TOP,
        TaskPurpose.ERROR_ANALYSIS: ModelTier.TOP,
        TaskPurpose.DOCUMENTATION: ModelTier.LOW,
        TaskPurpose.COMMIT_MESSAGE: ModelTier.LOW,
        TaskPurpose.SELF_REFLECTION: ModelTier.MID,
    }

    def __init__(
        self,
        provider_configs: List[ProviderConfig],
        routing_config: Optional[RoutingConfig] = None,
    ):
        """Initialize router.

        Args:
            provider_configs: List of provider configurations
            routing_config: Routing configuration
        """
        self._provider_configs = provider_configs
        self._routing_config = routing_config or RoutingConfig()
        self._providers: Dict[str, LLMProvider] = {}
        self._health_cache: Dict[str, HealthStatus] = {}
        self._cost_trackers: Dict[Optional[str], CostTracker] = {}
        self._circuit_breakers = CircuitBreakerRegistry()

        # Initialize providers
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        for config in self._provider_configs:
            if not config.enabled:
                continue

            try:
                if config.type == "anthropic":
                    provider = AnthropicProvider(api_key=config.api_key)
                elif config.type == "openai":
                    provider = OpenAIProvider(
                        api_key=config.api_key, base_url=config.base_url
                    )
                elif config.type == "gemini":
                    provider = GeminiProvider(api_key=config.api_key)
                elif config.type == "ollama":
                    provider = OllamaProvider(
                        base_url=config.base_url or "http://localhost:11434"
                    )
                else:
                    logger.warning(f"Unknown provider type: {config.type}")
                    continue

                self._providers[config.name] = provider
                logger.info(f"Initialized provider: {config.name}")

            except Exception as e:
                logger.error(f"Failed to initialize provider {config.name}: {e}")

    async def _check_provider_health(self, provider_name: str) -> HealthStatus:
        """Check and cache provider health.

        Args:
            provider_name: Name of provider to check

        Returns:
            Health status
        """
        provider = self._providers.get(provider_name)
        if not provider:
            return HealthStatus(
                provider=provider_name,
                healthy=False,
                error="Provider not found",
            )

        health = await provider.health_check()
        self._health_cache[provider_name] = health
        return health

    def _select_tier(self, context: RoutingContext) -> ModelTier:
        """Select appropriate tier based on context.

        Args:
            context: Routing context

        Returns:
            Selected tier
        """
        # Use preferred tier if specified
        if context.preferred_tier:
            return context.preferred_tier

        # Get base tier from task
        base_tier = self.TASK_TIER_RULES.get(context.purpose, ModelTier.MID)

        # Adjust for code generation based on complexity
        if context.purpose == TaskPurpose.CODE_GENERATION:
            if context.complexity_score > 7.0:
                return ModelTier.TOP
            elif context.complexity_score < 3.0:
                return ModelTier.MID

        return base_tier

    def _find_model_in_tier(
        self,
        tier: ModelTier,
        context: RoutingContext,
        exclude_providers: List[str],
    ) -> Optional[tuple[str, str]]:
        """Find best available model in a tier.

        Args:
            tier: Model tier to search
            context: Routing context
            exclude_providers: Providers to exclude

        Returns:
            Tuple of (provider_name, model_id) or None
        """
        candidates = []

        for config in self._provider_configs:
            if not config.enabled or config.name in exclude_providers:
                continue

            # Check if provider is in health cache and healthy
            health = self._health_cache.get(config.name)
            if health and not health.healthy:
                continue

            for model in config.models:
                # Match tier
                if model.tier.lower() == tier.value:
                    cost_per_1k = (
                        model.cost_per_1k_input + model.cost_per_1k_output
                    ) / 2
                    candidates.append((config.name, model.id, cost_per_1k))

        if not candidates:
            return None

        # Sort by cost (prefer cheaper within tier)
        candidates.sort(key=lambda x: x[2])
        return (candidates[0][0], candidates[0][1])

    async def route(self, context: RoutingContext) -> SelectedModel:
        """Route request to best available model.

        Args:
            context: Routing context

        Returns:
            Selected model

        Raises:
            ValueError: If no suitable model found
        """
        # Check health of available providers
        for provider_name in self._providers.keys():
            if provider_name not in self._health_cache:
                await self._check_provider_health(provider_name)

        # Select tier
        tier = self._select_tier(context)

        # Find model in selected tier
        result = self._find_model_in_tier(tier, context, context.exclude_providers)
        if result:
            provider_name, model_id = result
            return self._create_selected_model(provider_name, model_id, tier)

        # Fallback to lower tier if cost allows
        if tier == ModelTier.TOP:
            tier = ModelTier.MID
            result = self._find_model_in_tier(tier, context, context.exclude_providers)
            if result:
                provider_name, model_id = result
                return self._create_selected_model(provider_name, model_id, tier)

        # Final fallback to lowest tier
        if tier != ModelTier.LOW:
            tier = ModelTier.LOW
            result = self._find_model_in_tier(tier, context, context.exclude_providers)
            if result:
                provider_name, model_id = result
                return self._create_selected_model(provider_name, model_id, tier)

        raise ValueError(f"No suitable model found for context: {context}")

    def _create_selected_model(
        self, provider_name: str, model_id: str, tier: ModelTier
    ) -> SelectedModel:
        """Create SelectedModel from configuration.

        Args:
            provider_name: Provider name
            model_id: Model ID
            tier: Model tier

        Returns:
            SelectedModel instance
        """
        provider_config = next(
            (c for c in self._provider_configs if c.name == provider_name), None
        )
        if not provider_config:
            raise ValueError(f"Provider not found: {provider_name}")

        model_config = next(
            (m for m in provider_config.models if m.id == model_id), None
        )
        if not model_config:
            raise ValueError(f"Model not found: {model_id}")

        return SelectedModel(
            provider=provider_name,
            model=model_id,
            tier=tier,
            estimated_cost_per_1k_tokens=(
                model_config.cost_per_1k_input + model_config.cost_per_1k_output
            )
            / 2,
            max_context_tokens=model_config.max_context_tokens,
        )

    def _calculate_cost(self, provider_name: str, model_id: str, usage) -> float:
        """Calculate actual cost based on per-model pricing.

        Args:
            provider_name: Name of the provider
            model_id: ID of the model
            usage: CompletionResponse usage object with prompt_tokens and completion_tokens

        Returns:
            Actual cost in USD
        """
        provider_config = next(
            (c for c in self._provider_configs if c.name == provider_name), None
        )
        if not provider_config:
            logger.warning(f"Provider not found for cost calculation: {provider_name}")
            return 0.0

        model_config = next(
            (m for m in provider_config.models if m.id == model_id), None
        )
        if not model_config:
            logger.warning(f"Model not found for cost calculation: {model_id}")
            return 0.0

        # Calculate cost: (prompt_tokens * input_price / 1000) + (completion_tokens * output_price / 1000)
        input_cost = (usage.prompt_tokens * model_config.cost_per_1k_input) / 1000.0
        output_cost = (usage.completion_tokens * model_config.cost_per_1k_output) / 1000.0
        total_cost = input_cost + output_cost

        return total_cost

    async def complete(
        self, request: CompletionRequest, context: RoutingContext
    ) -> CompletionResponse:
        """Route and execute a completion request.

        Args:
            request: Completion request
            context: Routing context for model selection

        Returns:
            Completion response

        Raises:
            ValueError: If routing fails
            Exception: If all fallbacks fail
        """
        # Route to best model
        selected = await self.route(context)

        # Get provider
        provider = self._providers.get(selected.provider)
        if not provider:
            raise ValueError(f"Provider not available: {selected.provider}")

        # Track cost
        cost_key = context.metadata.get("story_id") if hasattr(context, "metadata") else None
        if cost_key not in self._cost_trackers:
            self._cost_trackers[cost_key] = CostTracker(story_id=cost_key)

        # Check circuit breaker before calling provider
        breaker = self._circuit_breakers.get_breaker(selected.provider)
        if not breaker.can_execute():
            logger.warning(
                "circuit_breaker_open",
                provider=selected.provider,
                model=selected.model,
            )
            # Skip directly to fallback
            fallback_selected = await self.fallback(selected, context)
            fallback_provider = self._providers.get(fallback_selected.provider)
            if not fallback_provider:
                raise ValueError(f"No healthy provider available (circuit open for {selected.provider})")
            request.model = fallback_selected.model
            response = await fallback_provider.complete(request)
            self._circuit_breakers.get_breaker(fallback_selected.provider).record_success()
            fallback_cost = self._calculate_cost(fallback_selected.provider, fallback_selected.model, response.usage)
            self._cost_trackers[cost_key].add_cost(fallback_cost)
            return response

        try:
            # Update request with routed model
            request.model = selected.model

            # Execute request
            response = await provider.complete(request)

            # Record success with circuit breaker
            breaker.record_success()

            # Track cost using per-model pricing
            actual_cost = self._calculate_cost(selected.provider, selected.model, response.usage)
            self._cost_trackers[cost_key].add_cost(actual_cost)

            return response

        except Exception as e:
            # Record failure with circuit breaker
            breaker.record_failure()
            logger.error(f"Error with {selected.provider}: {e}")

            # Try fallback
            fallback_selected = await self.fallback(selected, context)
            if fallback_selected.provider == selected.provider:
                raise  # No fallback available

            fallback_provider = self._providers.get(fallback_selected.provider)
            if not fallback_provider:
                raise

            request.model = fallback_selected.model
            response = await fallback_provider.complete(request)

            # Track cost using per-model pricing
            fallback_cost = self._calculate_cost(fallback_selected.provider, fallback_selected.model, response.usage)
            self._cost_trackers[cost_key].add_cost(fallback_cost)

            return response

    async def fallback(
        self, failed_model: SelectedModel, context: RoutingContext
    ) -> SelectedModel:
        """Select fallback model after failure.

        Args:
            failed_model: Model that failed
            context: Routing context

        Returns:
            Fallback model

        Raises:
            ValueError: If no fallback available
        """
        # Add failed provider to exclusions
        exclude = context.exclude_providers + [failed_model.provider]

        # Try fallback chain from config
        if self._routing_config.fallback_chain:
            for fallback_provider in self._routing_config.fallback_chain:
                if fallback_provider not in exclude:
                    # Find any available model from fallback provider
                    for config in self._provider_configs:
                        if config.name == fallback_provider and config.models:
                            return self._create_selected_model(
                                config.name, config.models[0].id, failed_model.tier
                            )

        # Fallback to MID tier if we were on TOP
        if failed_model.tier == ModelTier.TOP:
            context.exclude_providers = exclude
            context.preferred_tier = ModelTier.MID
            return await self.route(context)

        # Last resort: try any available model
        for config in self._provider_configs:
            if config.enabled and config.name not in exclude and config.models:
                return self._create_selected_model(
                    config.name, config.models[0].id, ModelTier.LOW
                )

        raise ValueError("No fallback model available")

    def get_cost_for_story(self, story_id: Optional[str] = None) -> float:
        """Get accumulated cost for a story.

        Args:
            story_id: Story identifier

        Returns:
            Total cost in USD
        """
        tracker = self._cost_trackers.get(story_id)
        return tracker.total_cost_usd if tracker else 0.0

    def get_circuit_breaker_status(self) -> dict:
        """Get circuit breaker status for all providers."""
        return self._circuit_breakers.get_status()

    def get_provider_metrics(self, provider_name: str) -> Optional[dict]:
        """Get metrics for a provider.

        Args:
            provider_name: Provider name

        Returns:
            Metrics dict with usage stats
        """
        provider = self._providers.get(provider_name)
        if not provider:
            return None

        usage = provider.get_usage()
        return {
            "provider": usage.provider,
            "total_calls": usage.total_calls,
            "total_tokens": usage.total_tokens,
            "total_cost_usd": usage.total_cost_usd,
            "avg_latency_ms": usage.avg_latency_ms,
            "error_rate": usage.error_rate,
        }
