"""Circuit breaker pattern implementation for LLM provider resilience."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker state machine."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    """Number of consecutive failures before opening circuit."""

    recovery_timeout_seconds: int = 60
    """Time to wait before transitioning from OPEN to HALF_OPEN."""

    half_open_max_calls: int = 2
    """Maximum calls allowed in HALF_OPEN state before re-opening or closing."""


class CircuitBreaker:
    """Manages state transitions and request gating for a single provider."""

    def __init__(self, provider_name: str, config: CircuitBreakerConfig | None = None):
        """Initialize circuit breaker for a provider.

        Args:
            provider_name: Name of the LLM provider (e.g., 'anthropic', 'openai').
            config: Circuit breaker configuration. Defaults to CircuitBreakerConfig().
        """
        self.provider_name = provider_name
        self.config = config or CircuitBreakerConfig()

        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = CircuitState.CLOSED

        logger.info(
            "circuit_breaker_initialized",
            provider=provider_name,
            failure_threshold=self.config.failure_threshold,
            recovery_timeout_seconds=self.config.recovery_timeout_seconds,
        )

    def can_execute(self) -> bool:
        """Check if a request should be allowed based on circuit state.

        Returns:
            True if the request should be executed, False if it should be rejected.
        """
        current_state = self.get_state()

        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.OPEN:
            return False
        elif current_state == CircuitState.HALF_OPEN:
            # Allow limited calls in HALF_OPEN state
            return self._success_count < self.config.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful request and update state."""
        self._failure_count = 0
        self._success_count += 1
        self._last_failure_time = None

        current_state = self.get_state()

        if current_state == CircuitState.HALF_OPEN:
            if self._success_count >= self.config.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._success_count = 0
                logger.info(
                    "circuit_breaker_closed",
                    provider=self.provider_name,
                    reason="recovery_successful",
                )
        elif current_state == CircuitState.CLOSED:
            # Continue normal operation
            pass

    def record_failure(self) -> None:
        """Record a failed request and update state."""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()
        self._success_count = 0

        if self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "circuit_breaker_opened",
                provider=self.provider_name,
                failure_count=self._failure_count,
                threshold=self.config.failure_threshold,
            )

    def get_state(self) -> CircuitState:
        """Get the current circuit breaker state.

        Automatically transitions from OPEN to HALF_OPEN if recovery timeout
        has elapsed.

        Returns:
            Current CircuitState.
        """
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is None:
                return self._state

            elapsed = datetime.utcnow() - self._last_failure_time
            if elapsed >= timedelta(seconds=self.config.recovery_timeout_seconds):
                self._state = CircuitState.HALF_OPEN
                self._failure_count = 0
                self._success_count = 0
                logger.info(
                    "circuit_breaker_half_open",
                    provider=self.provider_name,
                    recovery_timeout_seconds=self.config.recovery_timeout_seconds,
                )

        return self._state

    def get_status(self) -> dict:
        """Get detailed status information about this circuit breaker.

        Returns:
            Dictionary with state, failure count, and last failure time.
        """
        return {
            "provider": self.provider_name,
            "state": self.get_state().value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "failure_threshold": self.config.failure_threshold,
        }


class CircuitBreakerRegistry:
    """Manages circuit breakers for multiple LLM providers."""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        """Initialize the circuit breaker registry.

        Args:
            config: Default circuit breaker configuration for all providers.
        """
        self.config = config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}

        logger.info(
            "circuit_breaker_registry_initialized",
            default_failure_threshold=self.config.failure_threshold,
        )

    def get_breaker(self, provider_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for the given provider.

        Args:
            provider_name: Name of the LLM provider.

        Returns:
            CircuitBreaker instance for the provider.
        """
        if provider_name not in self._breakers:
            self._breakers[provider_name] = CircuitBreaker(provider_name, self.config)
            logger.info(
                "circuit_breaker_created",
                provider=provider_name,
            )

        return self._breakers[provider_name]

    def get_healthy_providers(self) -> list[str]:
        """Get list of providers in CLOSED or HALF_OPEN state.

        Returns:
            List of provider names that are healthy (not in OPEN state).
        """
        healthy = []
        for provider_name, breaker in self._breakers.items():
            state = breaker.get_state()
            if state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                healthy.append(provider_name)

        return healthy

    def get_unhealthy_providers(self) -> list[str]:
        """Get list of providers currently in OPEN state.

        Returns:
            List of provider names that are unhealthy (in OPEN state).
        """
        unhealthy = []
        for provider_name, breaker in self._breakers.items():
            if breaker.get_state() == CircuitState.OPEN:
                unhealthy.append(provider_name)

        return unhealthy

    def get_status(self) -> dict:
        """Get comprehensive status of all circuit breakers.

        Returns:
            Dictionary mapping provider names to their status dictionaries,
            plus aggregate statistics.
        """
        breakers_status = {}
        for provider_name, breaker in self._breakers.items():
            breakers_status[provider_name] = breaker.get_status()

        closed_count = sum(
            1
            for breaker in self._breakers.values()
            if breaker.get_state() == CircuitState.CLOSED
        )
        open_count = sum(
            1
            for breaker in self._breakers.values()
            if breaker.get_state() == CircuitState.OPEN
        )
        half_open_count = sum(
            1
            for breaker in self._breakers.values()
            if breaker.get_state() == CircuitState.HALF_OPEN
        )

        return {
            "breakers": breakers_status,
            "summary": {
                "total_providers": len(self._breakers),
                "closed_count": closed_count,
                "open_count": open_count,
                "half_open_count": half_open_count,
                "healthy_providers": self.get_healthy_providers(),
                "unhealthy_providers": self.get_unhealthy_providers(),
            },
        }

    def reset(self, provider_name: str | None = None) -> None:
        """Reset circuit breaker state.

        Args:
            provider_name: Provider to reset. If None, resets all providers.
        """
        if provider_name is None:
            self._breakers.clear()
            logger.info("all_circuit_breakers_reset")
        else:
            if provider_name in self._breakers:
                del self._breakers[provider_name]
                logger.info(
                    "circuit_breaker_reset",
                    provider=provider_name,
                )
