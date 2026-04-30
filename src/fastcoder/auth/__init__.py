"""SSO/SCIM Integration module for authentication and authorization.

This module provides:
- SSO authentication (SAML and OIDC)
- SCIM 2.0 user and group management
- Authentication middleware for FastAPI
- Role-based access control
"""

from __future__ import annotations

from fastcoder.auth.middleware import AuthMiddleware
from fastcoder.auth.scim_provider import SCIMProvider
from fastcoder.auth.scim_routes import create_scim_router
from fastcoder.auth.sso_provider import SSOProvider
from fastcoder.auth.auth_routes import create_auth_router
from fastcoder.auth.types import (
    AuthDecision,
    AuthProvider,
    LocalUser,
    OAuthProviderConfig,
    OIDCConfig,
    SAMLConfig,
    SCIMGroup,
    SCIMUser,
    SSOSession,
    UserRole,
)

__all__ = [
    # Type definitions
    "AuthProvider",
    "UserRole",
    "SCIMUser",
    "SCIMGroup",
    "SAMLConfig",
    "OIDCConfig",
    "SSOSession",
    "AuthDecision",
    "LocalUser",
    "OAuthProviderConfig",
    # Providers
    "SSOProvider",
    "SCIMProvider",
    "AuthMiddleware",
    "create_scim_router",
    "create_auth_router",
]
