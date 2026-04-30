"""Comprehensive tests for authentication and authorization modules."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from fastcoder.auth.middleware import AGENT_SESSION_COOKIE, AuthMiddleware
from fastcoder.auth.scim_provider import SCIMProvider
from fastcoder.auth.scim_routes import create_scim_router, set_scim_bearer_token
from fastcoder.auth.sso_provider import SSOProvider
from fastcoder.auth.types import (
    AuthDecision,
    AuthProvider,
    OIDCConfig,
    SAMLConfig,
    SCIMGroup,
    SCIMUser,
    SSOSession,
    UserRole,
)


# ============================================================================
# Tests for auth/types.py
# ============================================================================


class TestAuthProvider:
    """Test AuthProvider enum."""

    def test_auth_provider_values(self):
        """Test AuthProvider enum values."""
        assert AuthProvider.LOCAL.value == "local"
        assert AuthProvider.SAML.value == "saml"
        assert AuthProvider.OIDC.value == "oidc"
        assert AuthProvider.API_KEY.value == "api_key"

    def test_auth_provider_members(self):
        """Test all AuthProvider members exist."""
        members = {member.value for member in AuthProvider}
        assert "local" in members
        assert "saml" in members
        assert "oidc" in members
        assert "api_key" in members


class TestUserRole:
    """Test UserRole enum."""

    def test_user_role_values(self):
        """Test UserRole enum values."""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.DEVELOPER.value == "developer"
        assert UserRole.REVIEWER.value == "reviewer"
        assert UserRole.VIEWER.value == "viewer"

    def test_user_role_members(self):
        """Test all UserRole members exist."""
        members = {member.value for member in UserRole}
        assert "admin" in members
        assert "developer" in members
        assert "reviewer" in members
        assert "viewer" in members


class TestSCIMUser:
    """Test SCIMUser Pydantic model."""

    def test_scim_user_creation(self):
        """Test creating a valid SCIMUser."""
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="bjensen@example.com",
            display_name="Barbara Jensen",
            email="bjensen@example.com",
            created_at=now,
            updated_at=now,
        )
        assert user.id == "user-123"
        assert user.user_name == "bjensen@example.com"
        assert user.display_name == "Barbara Jensen"
        assert user.email == "bjensen@example.com"
        assert user.active is True
        assert user.roles == []
        assert user.groups == []

    def test_scim_user_with_roles_and_groups(self):
        """Test SCIMUser with roles and groups."""
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            external_id="ext-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            active=True,
            roles=[UserRole.DEVELOPER, UserRole.REVIEWER],
            groups=["developers", "backend-team"],
            created_at=now,
            updated_at=now,
        )
        assert user.external_id == "ext-123"
        assert len(user.roles) == 2
        assert UserRole.DEVELOPER in user.roles
        assert "developers" in user.groups

    def test_scim_user_defaults(self):
        """Test SCIMUser field defaults."""
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test",
            email="test@example.com",
            created_at=now,
            updated_at=now,
        )
        assert user.external_id is None
        assert user.active is True
        assert user.roles == []
        assert user.groups == []

    def test_scim_user_model_config(self):
        """Test SCIMUser model_config has json_schema_extra."""
        assert hasattr(SCIMUser, "model_config")
        assert "json_schema_extra" in SCIMUser.model_config

    def test_scim_user_serialization(self):
        """Test SCIMUser JSON serialization."""
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            active=True,
            roles=[UserRole.DEVELOPER],
            groups=["group1"],
            created_at=now,
            updated_at=now,
        )
        data = user.model_dump()
        assert data["id"] == "user-123"
        assert data["user_name"] == "test@example.com"
        assert data["active"] is True
        assert data["roles"] == [UserRole.DEVELOPER]


class TestSCIMGroup:
    """Test SCIMGroup Pydantic model."""

    def test_scim_group_creation(self):
        """Test creating a valid SCIMGroup."""
        group = SCIMGroup(
            id="group-123",
            display_name="Backend Team",
        )
        assert group.id == "group-123"
        assert group.display_name == "Backend Team"
        assert group.members == []

    def test_scim_group_with_members(self):
        """Test SCIMGroup with members."""
        group = SCIMGroup(
            id="group-123",
            display_name="Backend Team",
            members=["user-1", "user-2", "user-3"],
        )
        assert len(group.members) == 3
        assert "user-1" in group.members

    def test_scim_group_model_config(self):
        """Test SCIMGroup model_config has json_schema_extra."""
        assert hasattr(SCIMGroup, "model_config")
        assert "json_schema_extra" in SCIMGroup.model_config

    def test_scim_group_serialization(self):
        """Test SCIMGroup JSON serialization."""
        group = SCIMGroup(
            id="group-123",
            display_name="Backend Team",
            members=["user-1", "user-2"],
        )
        data = group.model_dump()
        assert data["id"] == "group-123"
        assert data["display_name"] == "Backend Team"
        assert data["members"] == ["user-1", "user-2"]


class TestSAMLConfig:
    """Test SAMLConfig Pydantic model."""

    def test_saml_config_creation(self):
        """Test creating a valid SAMLConfig."""
        config = SAMLConfig(
            entity_id="https://myapp.example.com/metadata",
            sso_url="https://idp.example.com/sso",
            x509_cert="-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
        )
        assert config.entity_id == "https://myapp.example.com/metadata"
        assert config.sso_url == "https://idp.example.com/sso"
        assert config.slo_url is None
        assert config.name_id_format == "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"

    def test_saml_config_with_all_fields(self):
        """Test SAMLConfig with all fields."""
        config = SAMLConfig(
            entity_id="https://myapp.example.com/metadata",
            sso_url="https://idp.example.com/sso",
            slo_url="https://idp.example.com/slo",
            x509_cert="cert",
            name_id_format="urn:oasis:names:tc:SAML:2.0:nameid-format:persistent",
            attribute_mapping={"email": "mail", "display_name": "cn"},
        )
        assert config.slo_url == "https://idp.example.com/slo"
        assert config.attribute_mapping["email"] == "mail"

    def test_saml_config_default_attribute_mapping(self):
        """Test SAMLConfig default attribute_mapping."""
        config = SAMLConfig(
            entity_id="https://myapp.example.com/metadata",
            sso_url="https://idp.example.com/sso",
            x509_cert="cert",
        )
        assert "email" in config.attribute_mapping
        assert "display_name" in config.attribute_mapping
        assert "groups" in config.attribute_mapping


class TestOIDCConfig:
    """Test OIDCConfig Pydantic model."""

    def test_oidc_config_creation(self):
        """Test creating a valid OIDCConfig."""
        config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
        )
        assert config.issuer == "https://accounts.google.com"
        assert config.client_id == "client_id"
        assert config.redirect_uri == "https://myapp.example.com/auth/callback"
        assert config.discovery_url is None

    def test_oidc_config_default_scopes(self):
        """Test OIDCConfig default scopes."""
        config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
        )
        assert config.scopes == ["openid", "profile", "email"]

    def test_oidc_config_custom_scopes(self):
        """Test OIDCConfig with custom scopes."""
        config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
            scopes=["openid", "custom_scope"],
        )
        assert config.scopes == ["openid", "custom_scope"]

    def test_oidc_config_with_discovery_url(self):
        """Test OIDCConfig with discovery_url."""
        config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
            discovery_url="https://accounts.google.com/.well-known/openid-configuration",
        )
        assert config.discovery_url is not None


class TestSSOSession:
    """Test SSOSession Pydantic model."""

    def test_sso_session_creation(self):
        """Test creating a valid SSOSession."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=8)
        session = SSOSession(
            session_id="session-123",
            user_id="user-123",
            provider=AuthProvider.OIDC,
            issued_at=now,
            expires_at=expires,
        )
        assert session.session_id == "session-123"
        assert session.user_id == "user-123"
        assert session.provider == AuthProvider.OIDC
        assert session.attributes == {}

    def test_sso_session_with_attributes(self):
        """Test SSOSession with attributes."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=8)
        session = SSOSession(
            session_id="session-123",
            user_id="user-123",
            provider=AuthProvider.SAML,
            issued_at=now,
            expires_at=expires,
            attributes={"email": "test@example.com", "groups": ["group1"]},
        )
        assert session.attributes["email"] == "test@example.com"

    def test_sso_session_default_attributes(self):
        """Test SSOSession default attributes."""
        now = datetime.now(timezone.utc)
        session = SSOSession(
            session_id="session-123",
            user_id="user-123",
            provider=AuthProvider.OIDC,
            issued_at=now,
            expires_at=now + timedelta(hours=8),
        )
        assert session.attributes == {}


class TestAuthDecision:
    """Test AuthDecision Pydantic model."""

    def test_auth_decision_allowed(self):
        """Test successful AuthDecision."""
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            created_at=now,
            updated_at=now,
            roles=[UserRole.DEVELOPER],
        )
        decision = AuthDecision(
            allowed=True,
            user=user,
            reason="User has required role",
            required_roles=[UserRole.DEVELOPER],
        )
        assert decision.allowed is True
        assert decision.user is user
        assert decision.reason == "User has required role"

    def test_auth_decision_denied(self):
        """Test denied AuthDecision."""
        decision = AuthDecision(
            allowed=False,
            user=None,
            reason="Invalid credentials",
        )
        assert decision.allowed is False
        assert decision.user is None

    def test_auth_decision_default_fields(self):
        """Test AuthDecision default fields."""
        decision = AuthDecision(
            allowed=False,
            reason="Test",
        )
        assert decision.user is None
        assert decision.required_roles == []


# ============================================================================
# Tests for auth/sso_provider.py
# ============================================================================


class TestSSOProviderInit:
    """Test SSOProvider initialization."""

    def test_init_no_config(self):
        """Test initialization without config (debug log)."""
        with patch("fastcoder.auth.sso_provider.logger") as mock_logger:
            provider = SSOProvider()
            assert provider.saml_config is None
            assert provider.oidc_config is None
            mock_logger.debug.assert_called_once()

    def test_init_with_saml_config(self):
        """Test initialization with SAML config only."""
        saml_config = SAMLConfig(
            entity_id="https://myapp.example.com/metadata",
            sso_url="https://idp.example.com/sso",
            x509_cert="cert",
        )
        with patch("fastcoder.auth.sso_provider.logger") as mock_logger:
            provider = SSOProvider(saml_config=saml_config)
            assert provider.saml_config is saml_config
            assert provider.oidc_config is None
            mock_logger.info.assert_called()

    def test_init_with_oidc_config(self):
        """Test initialization with OIDC config only."""
        oidc_config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
        )
        with patch("fastcoder.auth.sso_provider.logger") as mock_logger:
            provider = SSOProvider(oidc_config=oidc_config)
            assert provider.saml_config is None
            assert provider.oidc_config is oidc_config
            mock_logger.info.assert_called()

    def test_init_with_both_configs(self):
        """Test initialization with both SAML and OIDC configs."""
        saml_config = SAMLConfig(
            entity_id="https://myapp.example.com/metadata",
            sso_url="https://idp.example.com/sso",
            x509_cert="cert",
        )
        oidc_config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
        )
        provider = SSOProvider(saml_config=saml_config, oidc_config=oidc_config)
        assert provider.saml_config is saml_config
        assert provider.oidc_config is oidc_config


class TestSAMLFlow:
    """Test SAML authentication flow."""

    @pytest.mark.asyncio
    async def test_initiate_saml_login_success(self):
        """Test successful SAML login initiation."""
        saml_config = SAMLConfig(
            entity_id="https://myapp.example.com/metadata",
            sso_url="https://idp.example.com/sso",
            x509_cert="cert",
        )
        provider = SSOProvider(saml_config=saml_config)
        redirect_url = await provider.initiate_saml_login()
        assert redirect_url.startswith("https://idp.example.com/sso?")

    @pytest.mark.asyncio
    async def test_initiate_saml_login_not_configured(self):
        """Test SAML login initiation when not configured."""
        provider = SSOProvider()
        with pytest.raises(ValueError, match="SAML is not configured"):
            await provider.initiate_saml_login()

    @pytest.mark.asyncio
    async def test_handle_saml_response_success(self):
        """Test successful SAML response handling."""
        saml_config = SAMLConfig(
            entity_id="https://myapp.example.com/metadata",
            sso_url="https://idp.example.com/sso",
            x509_cert="cert",
        )
        provider = SSOProvider(saml_config=saml_config)

        # Create a valid base64-encoded XML response
        saml_xml = (
            "<Response>"
            "<Assertion>"
            "<Subject><NameID>user@example.com</NameID></Subject>"
            "</Assertion>"
            "</Response>"
        )
        saml_b64 = base64.b64encode(saml_xml.encode()).decode()

        session = await provider.handle_saml_response(saml_b64)
        assert session.user_id == "user@example.com"
        assert session.provider == AuthProvider.SAML
        assert session.attributes["email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_handle_saml_response_missing_nameid(self):
        """Test SAML response without NameID."""
        saml_config = SAMLConfig(
            entity_id="https://myapp.example.com/metadata",
            sso_url="https://idp.example.com/sso",
            x509_cert="cert",
        )
        provider = SSOProvider(saml_config=saml_config)

        saml_xml = (
            "<Response>"
            "<Assertion>"
            "<Subject></Subject>"
            "</Assertion>"
            "</Response>"
        )
        saml_b64 = base64.b64encode(saml_xml.encode()).decode()

        with pytest.raises(ValueError, match="missing NameID"):
            await provider.handle_saml_response(saml_b64)

    @pytest.mark.asyncio
    async def test_handle_saml_response_invalid_xml(self):
        """Test SAML response with invalid XML."""
        saml_config = SAMLConfig(
            entity_id="https://myapp.example.com/metadata",
            sso_url="https://idp.example.com/sso",
            x509_cert="cert",
        )
        provider = SSOProvider(saml_config=saml_config)

        saml_b64 = base64.b64encode(b"<invalid>xml").decode()

        with pytest.raises(ValueError, match="Invalid SAML response"):
            await provider.handle_saml_response(saml_b64)

    @pytest.mark.asyncio
    async def test_handle_saml_response_not_configured(self):
        """Test SAML response handling when not configured."""
        provider = SSOProvider()
        with pytest.raises(ValueError, match="SAML is not configured"):
            await provider.handle_saml_response("base64data")


class TestOIDCFlow:
    """Test OIDC authentication flow."""

    @pytest.mark.asyncio
    async def test_initiate_oidc_login_success(self):
        """Test successful OIDC login initiation."""
        oidc_config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
        )
        provider = SSOProvider(oidc_config=oidc_config)

        auth_url, state = await provider.initiate_oidc_login()
        assert auth_url.startswith("https://accounts.google.com/authorize?")
        assert "client_id=client_id" in auth_url
        assert "redirect_uri=" in auth_url
        assert state is not None
        assert state in provider._pkce_verifiers

    @pytest.mark.asyncio
    async def test_initiate_oidc_login_stores_pkce_verifier(self):
        """Test OIDC initiation stores PKCE verifier."""
        oidc_config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
        )
        provider = SSOProvider(oidc_config=oidc_config)

        auth_url, state = await provider.initiate_oidc_login()
        verifier, expires_at = provider._pkce_verifiers[state]
        assert verifier is not None
        assert isinstance(verifier, str)
        assert expires_at > datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_initiate_oidc_login_not_configured(self):
        """Test OIDC login initiation when not configured."""
        provider = SSOProvider()
        with pytest.raises(ValueError, match="OIDC is not configured"):
            await provider.initiate_oidc_login()

    @pytest.mark.asyncio
    async def test_handle_oidc_callback_success(self):
        """Test successful OIDC callback handling."""
        oidc_config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
        )
        provider = SSOProvider(oidc_config=oidc_config)

        # Initiate login to get state
        auth_url, state = await provider.initiate_oidc_login()

        # Mock the _simulate_id_token to return a properly formatted token
        payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "user-123", "email": "test@example.com", "name": "Test"}).encode()
        ).decode().rstrip("=")
        test_token = f"header.{payload}.signature"

        with patch.object(provider, '_simulate_id_token', return_value=test_token):
            # Handle callback
            session = await provider.handle_oidc_callback("auth_code", state)
            assert session.user_id == "user-123"
            assert session.provider == AuthProvider.OIDC

    @pytest.mark.asyncio
    async def test_handle_oidc_callback_invalid_state(self):
        """Test OIDC callback with invalid state."""
        oidc_config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
        )
        provider = SSOProvider(oidc_config=oidc_config)

        with pytest.raises(ValueError, match="Invalid or expired state"):
            await provider.handle_oidc_callback("auth_code", "invalid_state")

    @pytest.mark.asyncio
    async def test_handle_oidc_callback_expired_state(self):
        """Test OIDC callback with expired state."""
        oidc_config = OIDCConfig(
            issuer="https://accounts.google.com",
            client_id="client_id",
            client_secret="secret",
            redirect_uri="https://myapp.example.com/auth/callback",
        )
        provider = SSOProvider(oidc_config=oidc_config)

        # Create an expired state
        state = "test_state"
        expired_time = datetime.now(timezone.utc) - timedelta(minutes=1)
        provider._pkce_verifiers[state] = ("verifier", expired_time)

        with pytest.raises(ValueError, match="has expired"):
            await provider.handle_oidc_callback("auth_code", state)

    @pytest.mark.asyncio
    async def test_handle_oidc_callback_not_configured(self):
        """Test OIDC callback handling when not configured."""
        provider = SSOProvider()
        with pytest.raises(ValueError, match="OIDC is not configured"):
            await provider.handle_oidc_callback("code", "state")


class TestSSOSessionManagement:
    """Test SSO session management."""

    @pytest.mark.asyncio
    async def test_validate_session_valid(self):
        """Test validating a valid session."""
        provider = SSOProvider()
        session = provider._create_session(
            user_id="user-123",
            provider=AuthProvider.LOCAL,
            attributes={},
        )

        validated = await provider.validate_session(session.session_id)
        assert validated is session
        assert validated.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_validate_session_expired(self):
        """Test validating an expired session."""
        provider = SSOProvider()
        now = datetime.now(timezone.utc)
        session = SSOSession(
            session_id="expired-session",
            user_id="user-123",
            provider=AuthProvider.LOCAL,
            issued_at=now - timedelta(hours=10),
            expires_at=now - timedelta(hours=1),
        )
        provider._sessions[session.session_id] = session

        validated = await provider.validate_session(session.session_id)
        assert validated is None
        assert session.session_id not in provider._sessions

    @pytest.mark.asyncio
    async def test_validate_session_unknown(self):
        """Test validating an unknown session."""
        provider = SSOProvider()
        validated = await provider.validate_session("unknown-session")
        assert validated is None

    @pytest.mark.asyncio
    async def test_revoke_session_found(self):
        """Test revoking a found session."""
        provider = SSOProvider()
        session = provider._create_session(
            user_id="user-123",
            provider=AuthProvider.LOCAL,
            attributes={},
        )

        success = await provider.revoke_session(session.session_id)
        assert success is True
        assert session.session_id not in provider._sessions

    @pytest.mark.asyncio
    async def test_revoke_session_not_found(self):
        """Test revoking a non-existent session."""
        provider = SSOProvider()
        success = await provider.revoke_session("unknown-session")
        assert success is False


class TestSSOProviderHelpers:
    """Test SSOProvider helper methods."""

    def test_create_session(self):
        """Test _create_session method."""
        provider = SSOProvider()
        session = provider._create_session(
            user_id="user-123",
            provider=AuthProvider.OIDC,
            attributes={"email": "test@example.com"},
            ttl_hours=8,
        )
        assert session.user_id == "user-123"
        assert session.provider == AuthProvider.OIDC
        assert session.attributes["email"] == "test@example.com"
        assert session.session_id in provider._sessions

    def test_create_session_ttl(self):
        """Test _create_session TTL."""
        provider = SSOProvider()
        before = datetime.now(timezone.utc)
        session = provider._create_session(
            user_id="user-123",
            provider=AuthProvider.LOCAL,
            attributes={},
            ttl_hours=24,
        )
        after = datetime.now(timezone.utc)

        # Check expires_at is approximately 24 hours from now
        delta = (session.expires_at - session.issued_at).total_seconds()
        assert delta >= 24 * 3600 - 1  # Allow 1 second margin

    def test_extract_xml_text_found(self):
        """Test _extract_xml_text when tag is found."""
        provider = SSOProvider()
        xml = "<root><NameID>user@example.com</NameID></root>"
        element = __import__("defusedxml.ElementTree", fromlist=["fromstring"]).fromstring(xml)

        text = provider._extract_xml_text(element, "NameID")
        assert text == "user@example.com"

    def test_extract_xml_text_not_found(self):
        """Test _extract_xml_text when tag is not found."""
        provider = SSOProvider()
        xml = "<root><Other>value</Other></root>"
        element = __import__("defusedxml.ElementTree", fromlist=["fromstring"]).fromstring(xml)

        text = provider._extract_xml_text(element, "NameID")
        assert text is None

    def test_extract_xml_attribute_found(self):
        """Test _extract_xml_attribute when attribute is found."""
        provider = SSOProvider()
        xml = '<root><Attribute name="email">user@example.com</Attribute></root>'
        element = __import__("defusedxml.ElementTree", fromlist=["fromstring"]).fromstring(xml)

        attr = provider._extract_xml_attribute(element, "name")
        assert attr == "email"

    def test_extract_xml_attribute_not_found(self):
        """Test _extract_xml_attribute when attribute is not found."""
        provider = SSOProvider()
        xml = "<root><Attribute>value</Attribute></root>"
        element = __import__("defusedxml.ElementTree", fromlist=["fromstring"]).fromstring(xml)

        attr = provider._extract_xml_attribute(element, "missing")
        assert attr is None

    def test_simulate_id_token(self):
        """Test _simulate_id_token."""
        provider = SSOProvider()
        token = provider._simulate_id_token("test_code")
        assert isinstance(token, str)
        parts = token.split(".")
        assert len(parts) == 3

    def test_decode_id_token_valid(self):
        """Test _decode_id_token with valid token."""
        provider = SSOProvider()
        payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "user-123", "email": "test@example.com", "name": "Test"}).encode()
        ).decode().rstrip("=")
        token = f"header.{payload}.signature"

        claims = provider._decode_id_token(token)
        assert claims["sub"] == "user-123"
        assert claims["email"] == "test@example.com"
        assert claims["name"] == "Test"

    def test_decode_id_token_invalid_format(self):
        """Test _decode_id_token with invalid JWT format."""
        provider = SSOProvider()
        with pytest.raises(ValueError, match="Invalid token format"):
            provider._decode_id_token("only.two")

    def test_decode_id_token_missing_sub(self):
        """Test _decode_id_token missing sub claim."""
        provider = SSOProvider()
        payload = base64.urlsafe_b64encode(
            json.dumps({"email": "test@example.com"}).encode()
        ).decode().rstrip("=")
        token = f"header.{payload}.signature"

        with pytest.raises(ValueError, match="missing required 'sub' claim"):
            provider._decode_id_token(token)

    def test_decode_id_token_invalid_json(self):
        """Test _decode_id_token with invalid JSON payload."""
        provider = SSOProvider()
        payload = base64.urlsafe_b64encode(b"not valid json").decode().rstrip("=")
        token = f"header.{payload}.signature"

        with pytest.raises(ValueError, match="Failed to decode"):
            provider._decode_id_token(token)


# ============================================================================
# Tests for auth/middleware.py
# ============================================================================


class TestAuthMiddlewareInit:
    """Test AuthMiddleware initialization."""

    def test_middleware_init(self):
        """Test middleware initialization."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        assert middleware.sso_provider is sso_provider
        assert middleware.scim_provider is scim_provider
        assert middleware.api_key_header == "X-API-Key"

    def test_middleware_init_custom_header(self):
        """Test middleware initialization with custom API key header."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(
            sso_provider,
            scim_provider,
            api_key_header="Authorization"
        )
        assert middleware.api_key_header == "Authorization"


class TestAuthMiddlewareAuthenticate:
    """Test AuthMiddleware authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_via_session_cookie(self):
        """Test authentication via session cookie."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        # Create a session
        session = sso_provider._create_session(
            user_id="user-123",
            provider=AuthProvider.LOCAL,
            attributes={},
        )

        # Create user in SCIM
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            created_at=now,
            updated_at=now,
        )
        scim_provider._users["user-123"] = user

        # Mock request with session cookie
        request = MagicMock(spec=Request)
        request.cookies = {AGENT_SESSION_COOKIE: session.session_id}
        request.headers = {}
        request.client = None

        decision = await middleware.authenticate(request)
        assert decision.allowed is True
        assert decision.user is user
        assert decision.reason == "Authenticated via session"

    @pytest.mark.asyncio
    async def test_authenticate_via_bearer_token(self):
        """Test authentication via bearer token."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        session = sso_provider._create_session(
            user_id="user-123",
            provider=AuthProvider.OIDC,
            attributes={},
        )

        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            created_at=now,
            updated_at=now,
        )
        scim_provider._users["user-123"] = user

        request = MagicMock(spec=Request)
        request.cookies = {}
        request.headers = {"Authorization": f"Bearer {session.session_id}"}
        request.client = None

        decision = await middleware.authenticate(request)
        assert decision.allowed is True
        assert decision.user is user

    @pytest.mark.asyncio
    async def test_authenticate_via_api_key(self):
        """Test authentication via API key."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        # Register an API key
        api_key = "test-api-key-12345"
        middleware.register_api_key(api_key, "user-123", [UserRole.DEVELOPER])

        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            created_at=now,
            updated_at=now,
            roles=[UserRole.DEVELOPER],
        )
        scim_provider._users["user-123"] = user

        request = MagicMock(spec=Request)
        request.cookies = {}
        request.headers = {"X-API-Key": api_key}
        request.client = None

        decision = await middleware.authenticate(request)
        assert decision.allowed is True
        assert decision.user is user
        assert UserRole.DEVELOPER in decision.required_roles

    @pytest.mark.asyncio
    async def test_authenticate_failure(self):
        """Test authentication failure."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        request = MagicMock(spec=Request)
        request.cookies = {}
        request.headers = {}
        request.url.path = "/test"
        request.client = MagicMock()
        request.client.host = "192.168.1.1"

        decision = await middleware.authenticate(request)
        assert decision.allowed is False
        assert decision.user is None


class TestAuthMiddlewareRoleDecorators:
    """Test AuthMiddleware role requirement decorators."""

    @pytest.mark.asyncio
    async def test_require_role_success(self):
        """Test require_role decorator with sufficient roles."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        # Create user with required role
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            created_at=now,
            updated_at=now,
            roles=[UserRole.DEVELOPER],
        )
        scim_provider._users["user-123"] = user

        session = sso_provider._create_session(
            user_id="user-123",
            provider=AuthProvider.LOCAL,
            attributes={},
        )

        # Create a decorated function
        @middleware.require_role(UserRole.DEVELOPER)
        async def protected_endpoint(request: Request):
            return {"status": "success"}

        request = MagicMock(spec=Request)
        request.cookies = {AGENT_SESSION_COOKIE: session.session_id}
        request.headers = {}
        request.client = None
        request.state = MagicMock()

        result = await protected_endpoint(request)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_require_role_not_authenticated(self):
        """Test require_role decorator without authentication."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        @middleware.require_role(UserRole.ADMIN)
        async def protected_endpoint(request: Request):
            return {"status": "success"}

        request = MagicMock(spec=Request)
        request.cookies = {}
        request.headers = {}
        request.client = None

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_require_role_insufficient_role(self):
        """Test require_role decorator with insufficient roles."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        # Create user with VIEWER role only
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            created_at=now,
            updated_at=now,
            roles=[UserRole.VIEWER],
        )
        scim_provider._users["user-123"] = user

        session = sso_provider._create_session(
            user_id="user-123",
            provider=AuthProvider.LOCAL,
            attributes={},
        )

        @middleware.require_role(UserRole.ADMIN)
        async def protected_endpoint(request: Request):
            return {"status": "success"}

        request = MagicMock(spec=Request)
        request.cookies = {AGENT_SESSION_COOKIE: session.session_id}
        request.headers = {}
        request.client = None

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint(request)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_require_any_role_success(self):
        """Test require_any_role decorator with at least one role."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            created_at=now,
            updated_at=now,
            roles=[UserRole.VIEWER],
        )
        scim_provider._users["user-123"] = user

        session = sso_provider._create_session(
            user_id="user-123",
            provider=AuthProvider.LOCAL,
            attributes={},
        )

        @middleware.require_any_role(UserRole.ADMIN, UserRole.VIEWER)
        async def protected_endpoint(request: Request):
            return {"status": "success"}

        request = MagicMock(spec=Request)
        request.cookies = {AGENT_SESSION_COOKIE: session.session_id}
        request.headers = {}
        request.client = None
        request.state = MagicMock()

        result = await protected_endpoint(request)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_require_any_role_no_matching_role(self):
        """Test require_any_role decorator with no matching roles."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            created_at=now,
            updated_at=now,
            roles=[UserRole.VIEWER],
        )
        scim_provider._users["user-123"] = user

        session = sso_provider._create_session(
            user_id="user-123",
            provider=AuthProvider.LOCAL,
            attributes={},
        )

        @middleware.require_any_role(UserRole.ADMIN, UserRole.DEVELOPER)
        async def protected_endpoint(request: Request):
            return {"status": "success"}

        request = MagicMock(spec=Request)
        request.cookies = {AGENT_SESSION_COOKIE: session.session_id}
        request.headers = {}
        request.client = None

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint(request)
        assert exc_info.value.status_code == 403


class TestAuthMiddlewareAPIKey:
    """Test AuthMiddleware API key management."""

    def test_hash_key(self):
        """Test _hash_key method."""
        key1 = AuthMiddleware._hash_key("test-key-123")
        key2 = AuthMiddleware._hash_key("test-key-123")
        key3 = AuthMiddleware._hash_key("different-key")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 64  # SHA-256 hex is 64 chars

    def test_register_api_key_default_roles(self):
        """Test register_api_key with default roles."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        middleware.register_api_key("test-key", "user-123")
        key_hash = AuthMiddleware._hash_key("test-key")
        assert key_hash in middleware._api_keys
        user_id, roles = middleware._api_keys[key_hash]
        assert user_id == "user-123"
        assert roles == [UserRole.VIEWER]

    def test_register_api_key_custom_roles(self):
        """Test register_api_key with custom roles."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        middleware.register_api_key(
            "test-key",
            "user-123",
            roles=[UserRole.DEVELOPER, UserRole.REVIEWER],
        )
        key_hash = AuthMiddleware._hash_key("test-key")
        user_id, roles = middleware._api_keys[key_hash]
        assert len(roles) == 2

    def test_revoke_api_key_found(self):
        """Test revoke_api_key for registered key."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        middleware.register_api_key("test-key", "user-123")
        success = middleware.revoke_api_key("test-key")
        assert success is True

        key_hash = AuthMiddleware._hash_key("test-key")
        assert key_hash not in middleware._api_keys

    def test_revoke_api_key_not_found(self):
        """Test revoke_api_key for unregistered key."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        success = middleware.revoke_api_key("unknown-key")
        assert success is False

    def test_validate_api_key_valid(self):
        """Test _validate_api_key with valid key."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        middleware.register_api_key("test-key", "user-123", [UserRole.ADMIN])
        user_id, roles = middleware._validate_api_key("test-key")
        assert user_id == "user-123"
        assert UserRole.ADMIN in roles

    def test_validate_api_key_invalid(self):
        """Test _validate_api_key with invalid key."""
        sso_provider = SSOProvider()
        scim_provider = SCIMProvider()
        middleware = AuthMiddleware(sso_provider, scim_provider)

        user_id, roles = middleware._validate_api_key("invalid-key")
        assert user_id is None
        assert roles == []


# ============================================================================
# Tests for auth/scim_provider.py
# ============================================================================


class TestSCIMProviderInit:
    """Test SCIMProvider initialization."""

    def test_init_without_persistence(self):
        """Test initialization without persistence."""
        provider = SCIMProvider()
        assert provider.persistence_path is None
        assert len(provider._users) == 0
        assert len(provider._groups) == 0

    def test_init_with_persistence_file_not_found(self):
        """Test initialization with non-existent persistence file."""
        provider = SCIMProvider(persistence_path="/tmp/nonexistent-scim.json")
        assert provider.persistence_path == "/tmp/nonexistent-scim.json"
        assert len(provider._users) == 0

    def test_init_with_persistence_file_exists(self):
        """Test initialization loading existing data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "users": {},
                "groups": {}
            }
            json.dump(data, f)
            temp_path = f.name

        try:
            provider = SCIMProvider(persistence_path=temp_path)
            assert provider.persistence_path == temp_path
        finally:
            Path(temp_path).unlink()


class TestSCIMProviderUsers:
    """Test SCIM user operations."""

    @pytest.mark.asyncio
    async def test_list_users_empty(self):
        """Test listing users when none exist."""
        provider = SCIMProvider()
        result = await provider.list_users()
        assert result["totalResults"] == 0
        assert result["Resources"] == []

    @pytest.mark.asyncio
    async def test_list_users_multiple(self):
        """Test listing multiple users."""
        provider = SCIMProvider()
        now = datetime.now(timezone.utc)

        user1 = await provider.create_user({
            "userName": "user1@example.com",
            "displayName": "User One",
        })
        user2 = await provider.create_user({
            "userName": "user2@example.com",
            "displayName": "User Two",
        })

        result = await provider.list_users()
        assert result["totalResults"] == 2
        assert len(result["Resources"]) == 2

    @pytest.mark.asyncio
    async def test_list_users_with_filter(self):
        """Test listing users with filter."""
        provider = SCIMProvider()

        await provider.create_user({
            "userName": "alice@example.com",
            "displayName": "Alice",
        })
        await provider.create_user({
            "userName": "bob@example.com",
            "displayName": "Bob",
        })

        # Filter using display_name attribute (the actual Python attribute name)
        result = await provider.list_users(filter_expr='display_name eq "alice"')
        assert result["totalResults"] == 1
        assert result["Resources"][0]["displayName"] == "Alice"

    @pytest.mark.asyncio
    async def test_list_users_pagination(self):
        """Test listing users with pagination."""
        provider = SCIMProvider()

        for i in range(5):
            await provider.create_user({
                "userName": f"user{i}@example.com",
                "displayName": f"User {i}",
            })

        result = await provider.list_users(start_index=1, count=2)
        assert result["itemsPerPage"] == 2
        assert result["startIndex"] == 1
        assert result["totalResults"] == 5

    @pytest.mark.asyncio
    async def test_get_user_found(self):
        """Test getting a user that exists."""
        provider = SCIMProvider()
        created = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
        })

        user = await provider.get_user(created.id)
        assert user is not None
        assert user.id == created.id

    @pytest.mark.asyncio
    async def test_get_user_not_found(self):
        """Test getting a user that doesn't exist."""
        provider = SCIMProvider()
        user = await provider.get_user("nonexistent-id")
        assert user is None

    @pytest.mark.asyncio
    async def test_create_user_success(self):
        """Test creating a user successfully."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
            "email": "test@example.com",
            "active": True,
        })

        assert user.user_name == "test@example.com"
        assert user.display_name == "Test User"
        assert user.email == "test@example.com"
        assert user.active is True

    @pytest.mark.asyncio
    async def test_create_user_missing_username(self):
        """Test creating a user without userName."""
        provider = SCIMProvider()
        with pytest.raises(ValueError, match="userName is required"):
            await provider.create_user({"displayName": "Test"})

    @pytest.mark.asyncio
    async def test_create_user_with_emails_list(self):
        """Test creating a user with emails list."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
            "emails": [{"value": "test@example.com", "primary": True}],
        })
        assert user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_create_user_with_roles(self):
        """Test creating a user with roles."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
            "roles": ["developer", "reviewer"],
        })
        assert UserRole.DEVELOPER in user.roles
        assert UserRole.REVIEWER in user.roles

    @pytest.mark.asyncio
    async def test_create_user_with_groups(self):
        """Test creating a user with groups."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
            "groups": ["group1", "group2"],
        })
        assert "group1" in user.groups

    @pytest.mark.asyncio
    async def test_update_user_success(self):
        """Test updating a user successfully."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Old Name",
        })

        updated = await provider.update_user(user.id, {
            "displayName": "New Name",
            "userName": "test@example.com",
        })
        assert updated.display_name == "New Name"

    @pytest.mark.asyncio
    async def test_update_user_not_found(self):
        """Test updating a non-existent user."""
        provider = SCIMProvider()
        with pytest.raises(ValueError, match="not found"):
            await provider.update_user("nonexistent-id", {"displayName": "Test"})

    @pytest.mark.asyncio
    async def test_update_user_with_email(self):
        """Test updating user email from emails list."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test",
        })

        updated = await provider.update_user(user.id, {
            "emails": [{"value": "newemail@example.com"}],
            "userName": "test@example.com",
        })
        assert updated.email == "newemail@example.com"

    @pytest.mark.asyncio
    async def test_delete_user_found(self):
        """Test deleting a user that exists."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
        })

        success = await provider.delete_user(user.id)
        assert success is True

        deleted = await provider.get_user(user.id)
        assert deleted.active is False

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self):
        """Test deleting a non-existent user."""
        provider = SCIMProvider()
        success = await provider.delete_user("nonexistent-id")
        assert success is False

    @pytest.mark.asyncio
    async def test_patch_user_add_roles(self):
        """Test PATCH operation adding roles."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
        })

        patched = await provider.patch_user(user.id, [
            {"op": "add", "path": "roles", "value": ["admin"]}
        ])
        assert UserRole.ADMIN in patched.roles

    @pytest.mark.asyncio
    async def test_patch_user_remove_roles(self):
        """Test PATCH operation removing roles."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
            "roles": ["developer"],
        })

        patched = await provider.patch_user(user.id, [
            {"op": "remove", "path": "roles"}
        ])
        assert len(patched.roles) == 0

    @pytest.mark.asyncio
    async def test_patch_user_replace_active(self):
        """Test PATCH operation replacing active status."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
            "active": True,
        })

        patched = await provider.patch_user(user.id, [
            {"op": "replace", "path": "active", "value": False}
        ])
        assert patched.active is False

    @pytest.mark.asyncio
    async def test_patch_user_replace_display_name(self):
        """Test PATCH operation replacing displayName."""
        provider = SCIMProvider()
        user = await provider.create_user({
            "userName": "test@example.com",
            "displayName": "Old Name",
        })

        patched = await provider.patch_user(user.id, [
            {"op": "replace", "path": "displayName", "value": "New Name"}
        ])
        assert patched.display_name == "New Name"

    @pytest.mark.asyncio
    async def test_patch_user_not_found(self):
        """Test PATCH operation on non-existent user."""
        provider = SCIMProvider()
        with pytest.raises(ValueError, match="not found"):
            await provider.patch_user("nonexistent-id", [
                {"op": "replace", "path": "displayName", "value": "Test"}
            ])


class TestSCIMProviderGroups:
    """Test SCIM group operations."""

    @pytest.mark.asyncio
    async def test_list_groups_empty(self):
        """Test listing groups when none exist."""
        provider = SCIMProvider()
        result = await provider.list_groups()
        assert result["totalResults"] == 0
        assert result["Resources"] == []

    @pytest.mark.asyncio
    async def test_list_groups_multiple(self):
        """Test listing multiple groups."""
        provider = SCIMProvider()

        await provider.create_group({"displayName": "Group One"})
        await provider.create_group({"displayName": "Group Two"})

        result = await provider.list_groups()
        assert result["totalResults"] == 2

    @pytest.mark.asyncio
    async def test_list_groups_with_filter(self):
        """Test listing groups with filter."""
        provider = SCIMProvider()

        await provider.create_group({"displayName": "Developers"})
        await provider.create_group({"displayName": "Testers"})

        # Filter using display_name attribute (the actual Python attribute name)
        result = await provider.list_groups(filter_expr='display_name eq "developers"')
        assert result["totalResults"] == 1
        assert result["Resources"][0]["displayName"] == "Developers"

    @pytest.mark.asyncio
    async def test_list_groups_pagination(self):
        """Test listing groups with pagination."""
        provider = SCIMProvider()

        for i in range(5):
            await provider.create_group({"displayName": f"Group {i}"})

        result = await provider.list_groups(start_index=1, count=2)
        assert result["itemsPerPage"] == 2

    @pytest.mark.asyncio
    async def test_get_group_found(self):
        """Test getting a group that exists."""
        provider = SCIMProvider()
        created = await provider.create_group({"displayName": "Test Group"})

        group = await provider.get_group(created.id)
        assert group is not None
        assert group.id == created.id

    @pytest.mark.asyncio
    async def test_get_group_not_found(self):
        """Test getting a group that doesn't exist."""
        provider = SCIMProvider()
        group = await provider.get_group("nonexistent-id")
        assert group is None

    @pytest.mark.asyncio
    async def test_create_group_success(self):
        """Test creating a group successfully."""
        provider = SCIMProvider()
        group = await provider.create_group({"displayName": "Test Group"})

        assert group.display_name == "Test Group"
        assert group.members == []

    @pytest.mark.asyncio
    async def test_create_group_missing_display_name(self):
        """Test creating a group without displayName."""
        provider = SCIMProvider()
        with pytest.raises(ValueError, match="displayName is required"):
            await provider.create_group({})

    @pytest.mark.asyncio
    async def test_create_group_with_members(self):
        """Test creating a group with members."""
        provider = SCIMProvider()
        group = await provider.create_group({
            "displayName": "Test Group",
            "members": ["user1", "user2"],
        })
        assert len(group.members) == 2

    @pytest.mark.asyncio
    async def test_update_group_success(self):
        """Test updating a group successfully."""
        provider = SCIMProvider()
        group = await provider.create_group({"displayName": "Old Name"})

        updated = await provider.update_group(group.id, {
            "displayName": "New Name",
        })
        assert updated.display_name == "New Name"

    @pytest.mark.asyncio
    async def test_update_group_not_found(self):
        """Test updating a non-existent group."""
        provider = SCIMProvider()
        with pytest.raises(ValueError, match="not found"):
            await provider.update_group("nonexistent-id", {"displayName": "Test"})

    @pytest.mark.asyncio
    async def test_delete_group_found(self):
        """Test deleting a group that exists."""
        provider = SCIMProvider()
        group = await provider.create_group({"displayName": "Test Group"})

        success = await provider.delete_group(group.id)
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_group_not_found(self):
        """Test deleting a non-existent group."""
        provider = SCIMProvider()
        success = await provider.delete_group("nonexistent-id")
        assert success is False


class TestSCIMProviderPersistence:
    """Test SCIM provider persistence."""

    def test_save_and_load(self):
        """Test saving and loading SCIM data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        Path(temp_path).unlink()  # Delete so it's a new file

        try:
            # Create provider and add data
            provider1 = SCIMProvider(persistence_path=temp_path)
            user = SCIMUser(
                id="user-123",
                user_name="test@example.com",
                display_name="Test User",
                email="test@example.com",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            provider1._users["user-123"] = user
            provider1.save(temp_path)

            # Load in new provider
            provider2 = SCIMProvider(persistence_path=temp_path)
            loaded = provider2._users.get("user-123")
            assert loaded is not None
            assert loaded.user_name == "test@example.com"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "data.json"
            provider = SCIMProvider()
            user = SCIMUser(
                id="user-123",
                user_name="test@example.com",
                display_name="Test User",
                email="test@example.com",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            provider._users["user-123"] = user
            provider.save(str(path))
            assert path.exists()

    def test_load_corrupt_user_data(self):
        """Test loading with corrupt user data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                "users": {
                    "user-123": {
                        "id": "user-123",
                        # Missing required fields
                        "user_name": "test@example.com",
                    }
                },
                "groups": {}
            }
            json.dump(data, f)
            temp_path = f.name

        try:
            provider = SCIMProvider(persistence_path=temp_path)
            # Should log warning but not crash
            assert len(provider._users) == 0
        finally:
            Path(temp_path).unlink()


class TestSCIMProviderFiltering:
    """Test SCIM filter parsing."""

    def test_parse_filter_eq_operator(self):
        """Test parsing filter with eq operator."""
        provider = SCIMProvider()
        filter_func = provider._parse_scim_filter('user_name eq "test@example.com"')

        now = datetime.now(timezone.utc)
        user1 = SCIMUser(
            id="1", user_name="test@example.com", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now
        )
        user2 = SCIMUser(
            id="2", user_name="other@example.com", display_name="Other",
            email="other@example.com", created_at=now, updated_at=now
        )

        assert filter_func(user1) is True
        assert filter_func(user2) is False

    def test_parse_filter_ne_operator(self):
        """Test parsing filter with ne operator."""
        provider = SCIMProvider()
        filter_func = provider._parse_scim_filter('user_name ne "test@example.com"')

        now = datetime.now(timezone.utc)
        user1 = SCIMUser(
            id="1", user_name="test@example.com", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now
        )
        user2 = SCIMUser(
            id="2", user_name="other@example.com", display_name="Other",
            email="other@example.com", created_at=now, updated_at=now
        )

        assert filter_func(user1) is False
        assert filter_func(user2) is True

    def test_parse_filter_co_operator(self):
        """Test parsing filter with co (contains) operator."""
        provider = SCIMProvider()
        filter_func = provider._parse_scim_filter('display_name co "test"')

        now = datetime.now(timezone.utc)
        user1 = SCIMUser(
            id="1", user_name="user1", display_name="Test User",
            email="test@example.com", created_at=now, updated_at=now
        )
        user2 = SCIMUser(
            id="2", user_name="user2", display_name="Other User",
            email="other@example.com", created_at=now, updated_at=now
        )

        assert filter_func(user1) is True
        assert filter_func(user2) is False

    def test_parse_filter_sw_operator(self):
        """Test parsing filter with sw (startsWith) operator."""
        provider = SCIMProvider()
        filter_func = provider._parse_scim_filter('user_name sw "test"')

        now = datetime.now(timezone.utc)
        user1 = SCIMUser(
            id="1", user_name="test@example.com", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now
        )
        user2 = SCIMUser(
            id="2", user_name="other@example.com", display_name="Other",
            email="other@example.com", created_at=now, updated_at=now
        )

        assert filter_func(user1) is True
        assert filter_func(user2) is False

    def test_parse_filter_ew_operator(self):
        """Test parsing filter with ew (endsWith) operator."""
        provider = SCIMProvider()
        filter_func = provider._parse_scim_filter('user_name ew "example.com"')

        now = datetime.now(timezone.utc)
        user1 = SCIMUser(
            id="1", user_name="test@example.com", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now
        )
        user2 = SCIMUser(
            id="2", user_name="user@other.org", display_name="Other",
            email="other@other.org", created_at=now, updated_at=now
        )

        assert filter_func(user1) is True
        assert filter_func(user2) is False

    def test_parse_filter_invalid_expression(self):
        """Test parsing invalid filter expression."""
        provider = SCIMProvider()
        with pytest.raises(ValueError, match="Invalid or unsupported"):
            provider._parse_scim_filter("invalid filter expression")


class TestSCIMProviderPatchOperations:
    """Test SCIM PATCH operations."""

    def test_validate_patch_path_allowed_fields(self):
        """Test _validate_patch_path with allowed fields."""
        provider = SCIMProvider()
        # Should not raise
        provider._validate_patch_path("active")
        provider._validate_patch_path("displayName")
        provider._validate_patch_path("email")
        provider._validate_patch_path("roles")
        provider._validate_patch_path("groups")

    def test_validate_patch_path_disallowed_field(self):
        """Test _validate_patch_path with disallowed field."""
        provider = SCIMProvider()
        with pytest.raises(ValueError, match="Cannot patch field"):
            provider._validate_patch_path("userName")

    @pytest.mark.asyncio
    async def test_apply_patch_add_roles(self):
        """Test _apply_patch_add with roles."""
        provider = SCIMProvider()
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="1", user_name="test", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now
        )

        provider._apply_patch_add(user, "roles", ["admin", "developer"])
        assert UserRole.ADMIN in user.roles
        assert UserRole.DEVELOPER in user.roles

    @pytest.mark.asyncio
    async def test_apply_patch_add_groups(self):
        """Test _apply_patch_add with groups."""
        provider = SCIMProvider()
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="1", user_name="test", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now,
            groups=["existing-group"]
        )

        provider._apply_patch_add(user, "groups", ["new-group"])
        assert "new-group" in user.groups
        assert "existing-group" in user.groups

    @pytest.mark.asyncio
    async def test_apply_patch_remove_roles(self):
        """Test _apply_patch_remove with roles."""
        provider = SCIMProvider()
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="1", user_name="test", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now,
            roles=[UserRole.ADMIN, UserRole.DEVELOPER]
        )

        provider._apply_patch_remove(user, "roles")
        assert len(user.roles) == 0

    @pytest.mark.asyncio
    async def test_apply_patch_remove_groups(self):
        """Test _apply_patch_remove with groups."""
        provider = SCIMProvider()
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="1", user_name="test", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now,
            groups=["group1", "group2"]
        )

        provider._apply_patch_remove(user, "groups")
        assert len(user.groups) == 0

    @pytest.mark.asyncio
    async def test_apply_patch_replace_active(self):
        """Test _apply_patch_replace with active field."""
        provider = SCIMProvider()
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="1", user_name="test", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now,
            active=True
        )

        provider._apply_patch_replace(user, "active", False)
        assert user.active is False

    @pytest.mark.asyncio
    async def test_apply_patch_replace_display_name(self):
        """Test _apply_patch_replace with displayName."""
        provider = SCIMProvider()
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="1", user_name="test", display_name="Old Name",
            email="test@example.com", created_at=now, updated_at=now
        )

        provider._apply_patch_replace(user, "displayName", "New Name")
        assert user.display_name == "New Name"

    @pytest.mark.asyncio
    async def test_apply_patch_replace_roles(self):
        """Test _apply_patch_replace with roles."""
        provider = SCIMProvider()
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="1", user_name="test", display_name="Test",
            email="test@example.com", created_at=now, updated_at=now,
            roles=[UserRole.VIEWER]
        )

        provider._apply_patch_replace(user, "roles", ["admin"])
        assert UserRole.ADMIN in user.roles
        assert UserRole.VIEWER not in user.roles


class TestSCIMProviderConversion:
    """Test SCIM format conversion."""

    def test_user_to_scim(self):
        """Test _user_to_scim conversion."""
        provider = SCIMProvider()
        now = datetime.now(timezone.utc)
        user = SCIMUser(
            id="user-123",
            external_id="ext-123",
            user_name="test@example.com",
            display_name="Test User",
            email="test@example.com",
            active=True,
            roles=[UserRole.DEVELOPER],
            groups=["group1"],
            created_at=now,
            updated_at=now,
        )

        scim_data = provider._user_to_scim(user)
        assert scim_data["id"] == "user-123"
        assert scim_data["userName"] == "test@example.com"
        assert scim_data["displayName"] == "Test User"
        assert "urn:ietf:params:scim:schemas:core:2.0:User" in scim_data["schemas"]

    def test_group_to_scim(self):
        """Test _group_to_scim conversion."""
        provider = SCIMProvider()
        group = SCIMGroup(
            id="group-123",
            display_name="Test Group",
            members=["user-1", "user-2"],
        )

        scim_data = provider._group_to_scim(group)
        assert scim_data["id"] == "group-123"
        assert scim_data["displayName"] == "Test Group"
        assert len(scim_data["members"]) == 2
        assert "urn:ietf:params:scim:schemas:core:2.0:Group" in scim_data["schemas"]


# ============================================================================
# Tests for auth/scim_routes.py
# ============================================================================


class TestSCIMRoutes:
    """Test SCIM API routes."""

    @pytest.fixture
    def scim_app(self):
        """Create a FastAPI app with SCIM routes."""
        app = FastAPI()
        provider = SCIMProvider()
        router = create_scim_router(provider)
        app.include_router(router)
        set_scim_bearer_token("test-token")
        return app, provider, TestClient(app)

    def test_set_scim_bearer_token(self):
        """Test setting SCIM bearer token."""
        from fastcoder.auth.scim_routes import _SCIM_BEARER_TOKEN
        set_scim_bearer_token("my-token")
        # Note: Can't directly check _SCIM_BEARER_TOKEN as it's module-private,
        # but we verify indirectly through route tests

    def test_scim_routes_require_auth(self, scim_app):
        """Test SCIM routes require authentication."""
        app, provider, client = scim_app

        response = client.get("/scim/v2/Users")
        assert response.status_code == 401

    def test_scim_routes_invalid_token(self, scim_app):
        """Test SCIM routes reject invalid token."""
        app, provider, client = scim_app

        response = client.get(
            "/scim/v2/Users",
            headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401

    def test_scim_routes_valid_token(self, scim_app):
        """Test SCIM routes accept valid token."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.get("/scim/v2/Users", headers=headers)
        assert response.status_code == 200

    def test_list_users_empty(self, scim_app):
        """Test GET /scim/v2/Users when empty."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.get("/scim/v2/Users", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["totalResults"] == 0

    def test_list_users_with_users(self, scim_app):
        """Test GET /scim/v2/Users with users."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        # Create users
        import asyncio
        asyncio.run(provider.create_user({
            "userName": "user1@example.com",
            "displayName": "User One",
        }))

        response = client.get("/scim/v2/Users", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["totalResults"] >= 1

    def test_get_user_found(self, scim_app):
        """Test GET /scim/v2/Users/{id} when user exists."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        import asyncio
        user = asyncio.run(provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
        }))

        response = client.get(f"/scim/v2/Users/{user.id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == user.id

    def test_get_user_not_found(self, scim_app):
        """Test GET /scim/v2/Users/{id} when user doesn't exist."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.get("/scim/v2/Users/nonexistent-id", headers=headers)
        assert response.status_code == 404

    def test_create_user(self, scim_app):
        """Test POST /scim/v2/Users."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.post(
            "/scim/v2/Users",
            headers=headers,
            json={
                "userName": "new@example.com",
                "displayName": "New User",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["userName"] == "new@example.com"

    def test_create_user_missing_username(self, scim_app):
        """Test POST /scim/v2/Users without userName."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.post(
            "/scim/v2/Users",
            headers=headers,
            json={"displayName": "No Username"}
        )
        assert response.status_code == 400

    def test_update_user(self, scim_app):
        """Test PUT /scim/v2/Users/{id}."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        import asyncio
        user = asyncio.run(provider.create_user({
            "userName": "test@example.com",
            "displayName": "Old Name",
        }))

        response = client.put(
            f"/scim/v2/Users/{user.id}",
            headers=headers,
            json={
                "userName": "test@example.com",
                "displayName": "New Name",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["displayName"] == "New Name"

    def test_update_user_not_found(self, scim_app):
        """Test PUT /scim/v2/Users/{id} when user doesn't exist."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.put(
            "/scim/v2/Users/nonexistent-id",
            headers=headers,
            json={"displayName": "Test"}
        )
        assert response.status_code == 404

    def test_patch_user(self, scim_app):
        """Test PATCH /scim/v2/Users/{id}."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        import asyncio
        user = asyncio.run(provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test",
        }))

        response = client.patch(
            f"/scim/v2/Users/{user.id}",
            headers=headers,
            json={
                "Operations": [
                    {"op": "replace", "path": "displayName", "value": "Updated"}
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["displayName"] == "Updated"

    def test_delete_user(self, scim_app):
        """Test DELETE /scim/v2/Users/{id}."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        import asyncio
        user = asyncio.run(provider.create_user({
            "userName": "test@example.com",
            "displayName": "Test User",
        }))

        response = client.delete(f"/scim/v2/Users/{user.id}", headers=headers)
        assert response.status_code == 200

    def test_list_groups_empty(self, scim_app):
        """Test GET /scim/v2/Groups when empty."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.get("/scim/v2/Groups", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["totalResults"] == 0

    def test_get_group_found(self, scim_app):
        """Test GET /scim/v2/Groups/{id} when group exists."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        import asyncio
        group = asyncio.run(provider.create_group({
            "displayName": "Test Group",
        }))

        response = client.get(f"/scim/v2/Groups/{group.id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == group.id

    def test_get_group_not_found(self, scim_app):
        """Test GET /scim/v2/Groups/{id} when group doesn't exist."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.get("/scim/v2/Groups/nonexistent-id", headers=headers)
        assert response.status_code == 404

    def test_create_group(self, scim_app):
        """Test POST /scim/v2/Groups."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.post(
            "/scim/v2/Groups",
            headers=headers,
            json={
                "displayName": "New Group",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["displayName"] == "New Group"

    def test_update_group(self, scim_app):
        """Test PUT /scim/v2/Groups/{id}."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        import asyncio
        group = asyncio.run(provider.create_group({
            "displayName": "Old Name",
        }))

        response = client.put(
            f"/scim/v2/Groups/{group.id}",
            headers=headers,
            json={
                "displayName": "New Name",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["displayName"] == "New Name"

    def test_delete_group(self, scim_app):
        """Test DELETE /scim/v2/Groups/{id}."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        import asyncio
        group = asyncio.run(provider.create_group({
            "displayName": "Test Group",
        }))

        response = client.delete(f"/scim/v2/Groups/{group.id}", headers=headers)
        assert response.status_code == 200

    def test_service_provider_config(self, scim_app):
        """Test GET /scim/v2/ServiceProviderConfig."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.get("/scim/v2/ServiceProviderConfig", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "patch" in data
        assert data["patch"]["supported"] is True

    def test_schemas(self, scim_app):
        """Test GET /scim/v2/Schemas."""
        app, provider, client = scim_app
        headers = {"Authorization": "Bearer test-token"}

        response = client.get("/scim/v2/Schemas", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["totalResults"] >= 2
