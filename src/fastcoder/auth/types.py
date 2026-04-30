"""Authentication and authorization type definitions."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AuthProvider(str, Enum):
    """Supported authentication providers."""

    LOCAL = "local"
    SAML = "saml"
    OIDC = "oidc"
    API_KEY = "api_key"
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    APPLE = "apple"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"


class UserRole(str, Enum):
    """User roles for access control."""

    ADMIN = "admin"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class SCIMUser(BaseModel):
    """SCIM 2.0 compliant user representation."""

    id: str
    external_id: Optional[str] = None
    user_name: str
    display_name: str
    email: str
    active: bool = True
    roles: list[UserRole] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "2819c604-8471-4d6d-b6e8-4d1c3c6f2c1d",
                "external_id": "bjensen@example.com",
                "user_name": "bjensen@example.com",
                "display_name": "Barbara Jensen",
                "email": "bjensen@example.com",
                "active": True,
                "roles": ["developer"],
                "groups": ["developers", "backend-team"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-15T12:30:00Z",
            }
        }
    }


class SCIMGroup(BaseModel):
    """SCIM 2.0 compliant group representation."""

    id: str
    display_name: str
    members: list[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "a23b4f5d-e8c9-4f2e-9b7d-1c2b3a4d5e6f",
                "display_name": "Backend Team",
                "members": ["2819c604-8471-4d6d-b6e8-4d1c3c6f2c1d"],
            }
        }
    }


class LocalUser(BaseModel):
    """A user registered via email/password."""

    id: str
    email: str
    display_name: str
    password_hash: str
    active: bool = True
    roles: list[UserRole] = Field(default_factory=lambda: [UserRole.DEVELOPER])
    auth_provider: AuthProvider = AuthProvider.LOCAL
    oauth_provider_id: Optional[str] = None  # External ID from OAuth provider
    avatar_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OAuthProviderConfig(BaseModel):
    """Configuration for an OAuth 2.0 / OIDC provider (Google, GitHub, etc.)."""

    provider: AuthProvider
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: list[str] = Field(default_factory=lambda: ["openid", "profile", "email"])
    redirect_uri: str = ""  # Set at runtime


class SAMLConfig(BaseModel):
    """SAML 2.0 configuration."""

    entity_id: str
    sso_url: str
    slo_url: Optional[str] = None
    x509_cert: str
    name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    attribute_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            "display_name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
            "groups": "groups",
        }
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "entity_id": "https://myapp.example.com/metadata",
                "sso_url": "https://idp.example.com/sso",
                "slo_url": "https://idp.example.com/slo",
                "x509_cert": "-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
                "name_id_format": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
                "attribute_mapping": {"email": "email_attr", "display_name": "name_attr"},
            }
        }
    }


class OIDCConfig(BaseModel):
    """OpenID Connect configuration."""

    issuer: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: list[str] = Field(default_factory=lambda: ["openid", "profile", "email"])
    discovery_url: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "issuer": "https://accounts.google.com",
                "client_id": "client_id_here",
                "client_secret": "client_secret_here",
                "redirect_uri": "https://myapp.example.com/auth/callback",
                "scopes": ["openid", "profile", "email"],
                "discovery_url": "https://accounts.google.com/.well-known/openid-configuration",
            }
        }
    }


class SSOSession(BaseModel):
    """Active SSO session."""

    session_id: str
    user_id: str
    provider: AuthProvider
    issued_at: datetime
    expires_at: datetime
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session_abc123def456",
                "user_id": "2819c604-8471-4d6d-b6e8-4d1c3c6f2c1d",
                "provider": "oidc",
                "issued_at": "2024-01-15T12:00:00Z",
                "expires_at": "2024-01-15T20:00:00Z",
                "attributes": {"email": "bjensen@example.com", "groups": ["developers"]},
            }
        }
    }


class AuthDecision(BaseModel):
    """Authorization decision result."""

    allowed: bool
    user: Optional[SCIMUser] = None
    reason: str
    required_roles: list[UserRole] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "allowed": True,
                "user": {
                    "id": "2819c604-8471-4d6d-b6e8-4d1c3c6f2c1d",
                    "user_name": "bjensen@example.com",
                    "email": "bjensen@example.com",
                    "display_name": "Barbara Jensen",
                    "active": True,
                    "roles": ["developer"],
                    "groups": ["developers"],
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-15T12:30:00Z",
                },
                "reason": "User has required developer role",
                "required_roles": ["developer"],
            }
        }
    }
