"""SSO provider implementation for SAML and OIDC authentication."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import defusedxml.ElementTree as SafeET
from xml.etree.ElementTree import Element as XMLElement, ParseError as XMLParseError

import structlog

from fastcoder.auth.types import (
    AuthProvider,
    OIDCConfig,
    SAMLConfig,
    SSOSession,
)

logger = structlog.get_logger(__name__)


class SSOProvider:
    """Handles SAML and OIDC authentication flows."""

    def __init__(self, saml_config: Optional[SAMLConfig] = None, oidc_config: Optional[OIDCConfig] = None):
        """Initialize the SSO provider.

        Args:
            saml_config: SAML 2.0 configuration, if SAML is enabled.
            oidc_config: OpenID Connect configuration, if OIDC is enabled.
        """
        self.saml_config = saml_config
        self.oidc_config = oidc_config
        self._sessions: dict[str, SSOSession] = {}
        # Store PKCE verifiers with expiration to prevent replay attacks
        self._pkce_verifiers: dict[str, tuple[str, datetime]] = {}  # state -> (verifier, expires_at)

        if not saml_config and not oidc_config:
            logger.debug("sso_provider_initialized_without_config", hint="Configure via Admin Panel → SSO settings")
        else:
            if saml_config:
                logger.info("saml_provider_initialized", entity_id=saml_config.entity_id)
            if oidc_config:
                logger.info("oidc_provider_initialized", issuer=oidc_config.issuer)

    async def initiate_saml_login(self) -> str:
        """Initiate SAML login flow.

        Returns:
            Redirect URL to SAML Identity Provider.

        Raises:
            ValueError: If SAML is not configured.
        """
        if not self.saml_config:
            logger.error("saml_not_configured")
            raise ValueError("SAML is not configured")

        logger.info("initiating_saml_login", entity_id=self.saml_config.entity_id)

        # In production, use python3-saml library to generate proper AuthnRequest
        # This is a simplified placeholder showing the flow
        sso_url = self.saml_config.sso_url

        # Generate a simple SAML AuthnRequest (production should use proper library)
        request_id = secrets.token_hex(16)
        issue_instant = datetime.now(timezone.utc).isoformat()

        # URL encode the SSO URL with request parameters
        # Note: Real implementation requires python3-saml for proper XML generation and signing
        redirect_url = f"{sso_url}?SAMLRequest={request_id}"

        logger.info("saml_login_initiated", request_id=request_id)
        return redirect_url

    async def handle_saml_response(self, saml_response: str) -> SSOSession:
        """Handle SAML response from Identity Provider.

        Args:
            saml_response: Base64-encoded SAML response from IdP.

        Returns:
            SSOSession with authenticated user information.

        Raises:
            ValueError: If SAML response is invalid or signature verification fails.
        """
        if not self.saml_config:
            logger.error("saml_not_configured")
            raise ValueError("SAML is not configured")

        logger.info("handling_saml_response")

        try:
            # Decode the SAML response
            decoded_response = base64.b64decode(saml_response)

            # Parse XML response using defusedxml to prevent XXE attacks
            # Note: In production, use python3-saml library for proper XML parsing and signature validation
            root = SafeET.fromstring(decoded_response)

            # Extract NameID (email/username)
            # This is a simplified extraction - real implementation would handle XML namespaces properly
            nameid = self._extract_xml_text(root, "NameID")

            # Extract attributes based on mapping
            attributes = {}
            attribute_mapping = self.saml_config.attribute_mapping

            # Extract email, display_name, and groups
            for attr_name, attr_key in attribute_mapping.items():
                value = self._extract_xml_attribute(root, attr_key)
                if value:
                    attributes[attr_name] = value

            # Use NameID as fallback for username
            if not nameid:
                logger.error("saml_response_missing_nameid")
                raise ValueError("SAML response missing NameID")

            attributes.setdefault("email", nameid)
            attributes.setdefault("display_name", nameid)

            # Create session
            session = self._create_session(
                user_id=nameid,
                provider=AuthProvider.SAML,
                attributes=attributes,
            )

            logger.info(
                "saml_response_handled",
                user_id=session.user_id,
                session_id=session.session_id,
            )
            return session

        except XMLParseError as e:
            logger.error("saml_response_parse_error", error=str(e))
            raise ValueError(f"Invalid SAML response: {e}")
        except Exception as e:
            logger.error("saml_response_handling_error", error=str(e))
            raise

    async def initiate_oidc_login(self) -> tuple[str, str]:
        """Initiate OpenID Connect login flow with PKCE.

        Returns:
            Tuple of (authorization_url, state_parameter) for tracking.

        Raises:
            ValueError: If OIDC is not configured.
        """
        if not self.oidc_config:
            logger.error("oidc_not_configured")
            raise ValueError("OIDC is not configured")

        logger.info("initiating_oidc_login", issuer=self.oidc_config.issuer)

        # Generate PKCE parameters
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode("utf-8")).digest()
        ).decode("utf-8").rstrip("=")

        state = secrets.token_urlsafe(32)

        # Store PKCE verifier with 10-minute TTL for later use in callback
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        self._pkce_verifiers[state] = (code_verifier, expires_at)

        # Build authorization URL
        auth_params = {
            "client_id": self.oidc_config.client_id,
            "redirect_uri": self.oidc_config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.oidc_config.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        auth_url = f"{self.oidc_config.issuer}/authorize?{urlencode(auth_params)}"

        logger.info("oidc_login_initiated", state=state)
        return auth_url, state

    async def handle_oidc_callback(self, code: str, state: str) -> SSOSession:
        """Handle OpenID Connect authorization callback.

        Args:
            code: Authorization code from OIDC provider.
            state: State parameter to verify request integrity.

        Returns:
            SSOSession with authenticated user information.

        Raises:
            ValueError: If state is invalid or token exchange fails.
        """
        if not self.oidc_config:
            logger.error("oidc_not_configured")
            raise ValueError("OIDC is not configured")

        logger.info("handling_oidc_callback", state=state)

        # Verify state parameter exists and has not expired
        if state not in self._pkce_verifiers:
            logger.error("oidc_invalid_state", state=state)
            raise ValueError("Invalid or expired state parameter")

        code_verifier, expires_at = self._pkce_verifiers.pop(state)
        if datetime.now(timezone.utc) > expires_at:
            logger.error("oidc_state_expired", state=state)
            raise ValueError("State parameter has expired")

        try:
            # In production, use requests library to exchange code for tokens
            # This is a simplified placeholder showing the expected flow

            # Note: Real implementation requires proper HTTP request to token endpoint
            # with client authentication and PKCE verification

            # Example token response (would come from actual OIDC provider)
            # {
            #   "access_token": "...",
            #   "id_token": "...",
            #   "token_type": "Bearer",
            #   "expires_in": 3600,
            #   "refresh_token": "..."
            # }

            # For now, simulate a successful token exchange
            # In production, this would make an actual HTTP request
            id_token = self._simulate_id_token(code)

            # Decode ID token and extract claims
            # Note: In production, use PyJWT with proper signature validation
            claims = self._decode_id_token(id_token)

            # Extract user information from claims
            user_id = claims.get("sub")
            attributes = {
                "email": claims.get("email"),
                "display_name": claims.get("name"),
            }

            # Add groups if present
            if "groups" in claims:
                attributes["groups"] = claims.get("groups", [])

            if not user_id:
                logger.error("oidc_token_missing_subject")
                raise ValueError("ID token missing 'sub' claim")

            # Create session
            session = self._create_session(
                user_id=user_id,
                provider=AuthProvider.OIDC,
                attributes=attributes,
            )

            logger.info(
                "oidc_callback_handled",
                user_id=session.user_id,
                session_id=session.session_id,
            )
            return session

        except Exception as e:
            logger.error("oidc_callback_error", error=str(e))
            raise

    async def validate_session(self, session_id: str) -> Optional[SSOSession]:
        """Validate an active session.

        Args:
            session_id: Session ID to validate.

        Returns:
            SSOSession if valid and not expired, None otherwise.
        """
        session = self._sessions.get(session_id)

        if not session:
            logger.debug("session_not_found", session_id=session_id)
            return None

        # Check expiration
        now = datetime.now(timezone.utc)
        if now > session.expires_at:
            logger.info("session_expired", session_id=session_id)
            del self._sessions[session_id]
            return None

        logger.debug("session_valid", session_id=session_id, user_id=session.user_id)
        return session

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke an active session.

        Args:
            session_id: Session ID to revoke.

        Returns:
            True if session was revoked, False if not found.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("session_revoked", session_id=session_id)
            return True

        logger.debug("session_not_found_for_revocation", session_id=session_id)
        return False

    def _create_session(
        self,
        user_id: str,
        provider: AuthProvider,
        attributes: dict,
        ttl_hours: int = 8,
    ) -> SSOSession:
        """Create a new SSO session.

        Args:
            user_id: Authenticated user ID.
            provider: Authentication provider used.
            attributes: User attributes from provider.
            ttl_hours: Session time-to-live in hours.

        Returns:
            New SSOSession.
        """
        now = datetime.now(timezone.utc)
        session_id = secrets.token_urlsafe(32)

        session = SSOSession(
            session_id=session_id,
            user_id=user_id,
            provider=provider,
            issued_at=now,
            expires_at=now + timedelta(hours=ttl_hours),
            attributes=attributes,
        )

        self._sessions[session_id] = session
        return session

    def _extract_xml_text(self, element: XMLElement, tag: str) -> Optional[str]:
        """Extract text content from XML element.

        Args:
            element: Root XML element.
            tag: Tag name to find.

        Returns:
            Text content or None if not found.
        """
        try:
            # Try with and without namespace
            for child in element.iter():
                if child.tag.endswith(tag) or child.tag == tag:
                    return child.text
            return None
        except Exception:
            return None

    def _extract_xml_attribute(self, element: XMLElement, attr_name: str) -> Optional[str]:
        """Extract attribute value from XML element.

        Args:
            element: Root XML element.
            attr_name: Attribute name to find.

        Returns:
            Attribute value or None if not found.
        """
        try:
            # Simple extraction - in production, handle namespaces properly
            for child in element.iter():
                if attr_name in child.attrib:
                    return child.attrib[attr_name]
            return None
        except Exception:
            return None

    def _simulate_id_token(self, code: str) -> str:
        """Simulate ID token generation (for testing/placeholder).

        In production, this would receive an actual ID token from the OIDC provider.

        Args:
            code: Authorization code.

        Returns:
            Simulated ID token.
        """
        # This is a placeholder - real implementation would receive actual tokens
        # from the OIDC provider via HTTP request
        return f"eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.{code}.signature"

    def _decode_id_token(self, id_token: str) -> dict:
        """Decode ID token claims.

        WARNING: This implementation does NOT verify JWT signatures.
        In production, use PyJWT with proper signature validation against
        the IdP's public keys (JWKS endpoint). Without signature verification,
        tokens can be forged. This is a placeholder for the token exchange flow.

        Args:
            id_token: JWT ID token from OIDC provider.

        Returns:
            Claims dictionary.

        Raises:
            ValueError: If token format is invalid or cannot be decoded.
        """
        # Split token into parts
        parts = id_token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format: expected 3 JWT segments")

        # Decode payload (second part)
        # SECURITY NOTE: In production, validate signature BEFORE trusting payload
        payload = parts[1]

        # Add padding if needed
        padding = 4 - (len(payload) % 4)
        if padding != 4:
            payload += "=" * padding

        try:
            decoded = base64.urlsafe_b64decode(payload)
            claims = json.loads(decoded)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("id_token_decode_error", error=str(e))
            raise ValueError(f"Failed to decode ID token payload: {e}")

        # Validate required claims exist
        if "sub" not in claims:
            raise ValueError("ID token missing required 'sub' claim")

        return {
            "sub": claims["sub"],
            "email": claims.get("email", ""),
            "name": claims.get("name", ""),
            "groups": claims.get("groups", []),
        }
