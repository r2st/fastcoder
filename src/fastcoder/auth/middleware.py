"""Authentication middleware for FastAPI."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from functools import wraps
from typing import Any, Callable, Optional

import structlog
from fastapi import Request, HTTPException, status

from fastcoder.auth.scim_provider import SCIMProvider
from fastcoder.auth.sso_provider import SSOProvider
from fastcoder.auth.types import AuthDecision, AuthProvider, SCIMUser, UserRole

logger = structlog.get_logger(__name__)

# Session cookie configuration
AGENT_SESSION_COOKIE = "agent_session"


class AuthMiddleware:
    """Authentication and authorization middleware for FastAPI."""

    def __init__(
        self,
        sso_provider: SSOProvider,
        scim_provider: SCIMProvider,
        api_key_header: str = "X-API-Key",
    ):
        """Initialize the auth middleware.

        Args:
            sso_provider: SSO provider for session validation.
            scim_provider: SCIM provider for user and group lookup.
            api_key_header: Header name for API key authentication.
        """
        self.sso_provider = sso_provider
        self.scim_provider = scim_provider
        self.api_key_header = api_key_header
        # Store API keys as SHA-256 hashes to prevent plaintext exposure in memory
        self._api_keys: dict[str, tuple[str, list[UserRole]]] = {}  # hash -> (user_id, roles)

        logger.info(
            "auth_middleware_initialized",
            api_key_header=api_key_header,
        )

    async def authenticate(self, request: Request) -> AuthDecision:
        """Authenticate request using session, API key, or bearer token.

        Checks in order:
        1. Session cookie
        2. Bearer token (from Authorization header)
        3. API key (from custom header)

        Args:
            request: FastAPI request object.

        Returns:
            AuthDecision with authentication result.
        """
        # Try session authentication first
        session_id = request.cookies.get(AGENT_SESSION_COOKIE)
        if session_id:
            session = await self.sso_provider.validate_session(session_id)
            if session:
                user = await self.scim_provider.get_user(session.user_id)
                if user:
                    logger.debug(
                        "authentication_successful_via_session",
                        user_id=user.id,
                        session_id=session_id,
                    )
                    return AuthDecision(
                        allowed=True,
                        user=user,
                        reason="Authenticated via session",
                        required_roles=[],
                    )

        # Try bearer token authentication
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            session = await self.sso_provider.validate_session(token)
            if session:
                user = await self.scim_provider.get_user(session.user_id)
                if user:
                    logger.debug(
                        "authentication_successful_via_bearer",
                        user_id=user.id,
                    )
                    return AuthDecision(
                        allowed=True,
                        user=user,
                        reason="Authenticated via bearer token",
                        required_roles=[],
                    )

        # Try API key authentication
        api_key = request.headers.get(self.api_key_header)
        if api_key:
            user_id, roles = self._validate_api_key(api_key)
            if user_id:
                user = await self.scim_provider.get_user(user_id)
                if user:
                    logger.debug(
                        "authentication_successful_via_api_key",
                        user_id=user_id,
                    )
                    return AuthDecision(
                        allowed=True,
                        user=user,
                        reason="Authenticated via API key",
                        required_roles=roles,
                    )

        logger.warning(
            "authentication_failed",
            path=request.url.path,
            remote_addr=request.client.host if request.client else None,
        )
        return AuthDecision(
            allowed=False,
            user=None,
            reason="No valid authentication credentials found",
            required_roles=[],
        )

    def require_role(self, *roles: UserRole) -> Callable:
        """Decorator factory requiring all specified roles.

        Args:
            *roles: Required user roles.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(request: Request, *args: Any, **kwargs: Any) -> Any:
                decision = await self.authenticate(request)

                if not decision.allowed or not decision.user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                    )

                # Check if user has all required roles
                user_roles = set(decision.user.roles)
                required = set(roles)

                if not required.issubset(user_roles):
                    logger.warning(
                        "authorization_failed_insufficient_roles",
                        user_id=decision.user.id,
                        required_roles=list(required),
                        user_roles=list(user_roles),
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions",
                    )

                request.state.user = decision.user
                request.state.auth_decision = decision
                return await func(request, *args, **kwargs)

            return wrapper
        return decorator

    def require_any_role(self, *roles: UserRole) -> Callable:
        """Decorator factory requiring at least one of the specified roles.

        Args:
            *roles: User must have at least one of these roles.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(request: Request, *args: Any, **kwargs: Any) -> Any:
                decision = await self.authenticate(request)

                if not decision.allowed or not decision.user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                    )

                # Check if user has at least one required role
                user_roles = set(decision.user.roles)
                required = set(roles)

                if not user_roles.intersection(required):
                    logger.warning(
                        "authorization_failed_no_matching_roles",
                        user_id=decision.user.id,
                        required_roles=list(required),
                        user_roles=list(user_roles),
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions",
                    )

                request.state.user = decision.user
                request.state.auth_decision = decision
                return await func(request, *args, **kwargs)

            return wrapper
        return decorator

    @staticmethod
    def _hash_key(api_key: str) -> str:
        """Produce a constant-time-comparable SHA-256 hash of an API key."""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    def register_api_key(
        self,
        api_key: str,
        user_id: str,
        roles: Optional[list[UserRole]] = None,
    ) -> None:
        """Register an API key for authentication.

        The key is stored as a SHA-256 hash to prevent plaintext exposure
        if process memory is dumped.

        Args:
            api_key: The API key value.
            user_id: Associated user ID.
            roles: Roles for this API key.
        """
        if roles is None:
            roles = [UserRole.VIEWER]

        key_hash = self._hash_key(api_key)
        self._api_keys[key_hash] = (user_id, roles)
        logger.info("api_key_registered", user_id=user_id, roles=roles)

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key.

        Args:
            api_key: The API key to revoke.

        Returns:
            True if revoked, False if not found.
        """
        key_hash = self._hash_key(api_key)
        if key_hash in self._api_keys:
            user_id, _ = self._api_keys[key_hash]
            del self._api_keys[key_hash]
            logger.info("api_key_revoked", user_id=user_id)
            return True
        return False

    def _validate_api_key(self, api_key: str) -> tuple[Optional[str], list[UserRole]]:
        """Validate API key using constant-time comparison.

        Args:
            api_key: The API key to validate.

        Returns:
            Tuple of (user_id, roles) or (None, []) if invalid.
        """
        key_hash = self._hash_key(api_key)
        # Use constant-time comparison to prevent timing attacks
        for stored_hash, (user_id, roles) in self._api_keys.items():
            if hmac.compare_digest(key_hash, stored_hash):
                logger.debug("api_key_validated", user_id=user_id)
                return user_id, roles

        logger.debug("api_key_invalid")
        return None, []
