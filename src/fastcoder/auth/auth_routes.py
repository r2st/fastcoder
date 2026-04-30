"""Authentication routes: email/password registration and login, OAuth 2.0 flows.

Supported providers:
- Email / Password (local registration)
- Google (OIDC)
- GitHub (OAuth 2.0)
- Microsoft (OIDC)
- Apple (OIDC)
- GitLab (OAuth 2.0)
- Bitbucket (OAuth 2.0)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, EmailStr, Field

from fastcoder.auth.types import (
    AuthProvider,
    LocalUser,
    OAuthProviderConfig,
    SSOSession,
    UserRole,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# In-memory stores (swap for a database in production)
# ────────────────────────────────────────────────────────────────────
_users: dict[str, LocalUser] = {}          # user_id  -> LocalUser
_users_by_email: dict[str, str] = {}       # email    -> user_id
_sessions: dict[str, SSOSession] = {}      # session_id -> SSOSession
_oauth_states: dict[str, dict] = {}        # state -> { provider, redirect_uri, created_at }

# Persistence path (JSON file next to the running server)
_DATA_DIR: Path = Path(os.environ.get("AGENT_DATA_DIR", "."))
_USERS_FILE: Path = _DATA_DIR / ".agent_users.json"

# Session cookie names — separate cookies for workspace and admin panel
SESSION_COOKIE = "agent_session"           # workspace / general
ADMIN_SESSION_COOKIE = "agent_admin_session"  # admin panel
SESSION_TTL_HOURS = 24

# Super-admin credentials file — generated once on first boot
_SUPER_ADMIN_FILE: Path = _DATA_DIR / ".agent_super_admin.json"
_super_admin_generated = False

# ────────────────────────────────────────────────────────────────────
# OAuth Provider Configs — persisted to .agent_oauth.json,
# configurable via Admin Panel
# ────────────────────────────────────────────────────────────────────

_OAUTH_FILE: Path = _DATA_DIR / ".agent_oauth.json"
_oauth_providers: dict[str, OAuthProviderConfig] = {}

# Well-known provider templates (default URLs/scopes)
OAUTH_PROVIDER_TEMPLATES: dict[str, dict] = {
    "google": {
        "label": "Google",
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
        "scopes": ["openid", "profile", "email"],
        "help_url": "https://console.cloud.google.com/apis/credentials",
    },
    "github": {
        "label": "GitHub",
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "scopes": ["read:user", "user:email"],
        "help_url": "https://github.com/settings/developers",
    },
    "microsoft": {
        "label": "Microsoft",
        "authorize_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "userinfo_url": "https://graph.microsoft.com/v1.0/me",
        "scopes": ["openid", "profile", "email", "User.Read"],
        "help_url": "https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps",
    },
    "apple": {
        "label": "Apple",
        "authorize_url": "https://appleid.apple.com/auth/authorize",
        "token_url": "https://appleid.apple.com/auth/token",
        "userinfo_url": "",
        "scopes": ["name", "email"],
        "help_url": "https://developer.apple.com/account/resources/identifiers",
    },
    "gitlab": {
        "label": "GitLab",
        "authorize_url": "https://gitlab.com/oauth/authorize",
        "token_url": "https://gitlab.com/oauth/token",
        "userinfo_url": "https://gitlab.com/api/v4/user",
        "scopes": ["read_user", "openid", "profile", "email"],
        "help_url": "https://gitlab.com/-/user_settings/applications",
    },
    "bitbucket": {
        "label": "Bitbucket",
        "authorize_url": "https://bitbucket.org/site/oauth2/authorize",
        "token_url": "https://bitbucket.org/site/oauth2/access_token",
        "userinfo_url": "https://api.bitbucket.org/2.0/user",
        "scopes": ["account", "email"],
        "help_url": "https://bitbucket.org/account/settings/app-authorizations/",
    },
}


def _save_oauth_configs() -> None:
    """Persist OAuth provider configs to disk (secrets included, file is 0600)."""
    try:
        data = {}
        for name, cfg in _oauth_providers.items():
            data[name] = cfg.model_dump(mode="json")
        _OAUTH_FILE.write_text(json.dumps(data, indent=2))
        os.chmod(str(_OAUTH_FILE), 0o600)
        logger.info(f"Saved {len(data)} OAuth providers to {_OAUTH_FILE}")
    except Exception as e:
        logger.warning(f"Could not save OAuth configs: {e}")


def _load_oauth_configs() -> None:
    """Load OAuth provider configs from disk."""
    global _oauth_providers
    if not _OAUTH_FILE.exists():
        _oauth_providers = {}
        return
    try:
        data = json.loads(_OAUTH_FILE.read_text())
        _oauth_providers = {}
        for name, cfg_data in data.items():
            _oauth_providers[name] = OAuthProviderConfig(**cfg_data)
        logger.info(f"Loaded {len(_oauth_providers)} OAuth providers from {_OAUTH_FILE}")
    except Exception as e:
        logger.warning(f"Could not load OAuth configs: {e}")
        _oauth_providers = {}


def _get_oauth_providers() -> dict[str, OAuthProviderConfig]:
    """Get currently configured OAuth providers."""
    if not _oauth_providers:
        _load_oauth_configs()
    return _oauth_providers


def _set_oauth_provider(name: str, config: OAuthProviderConfig) -> None:
    """Set/update an OAuth provider config and persist."""
    _oauth_providers[name] = config
    _save_oauth_configs()


def _remove_oauth_provider(name: str) -> bool:
    """Remove an OAuth provider config and persist."""
    if name in _oauth_providers:
        del _oauth_providers[name]
        _save_oauth_configs()
        return True
    return False


# ────────────────────────────────────────────────────────────────────
# Password hashing  (bcrypt-style using hashlib + salt)
# ────────────────────────────────────────────────────────────────────
def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations=260_000)
    return f"{salt}${h.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    parts = stored_hash.split("$", 1)
    if len(parts) != 2:
        return False
    salt, expected_hex = parts
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations=260_000)
    return secrets.compare_digest(h.hex(), expected_hex)


# ────────────────────────────────────────────────────────────────────
# Session helpers
# ────────────────────────────────────────────────────────────────────
def _create_session(user: LocalUser) -> SSOSession:
    now = datetime.now(timezone.utc)
    session = SSOSession(
        session_id=secrets.token_urlsafe(32),
        user_id=user.id,
        provider=user.auth_provider,
        issued_at=now,
        expires_at=now + timedelta(hours=SESSION_TTL_HOURS),
        attributes={
            "email": user.email,
            "display_name": user.display_name,
            "avatar_url": user.avatar_url or "",
            "roles": [r.value for r in user.roles],
        },
    )
    _sessions[session.session_id] = session
    return session


def _validate_session(session_id: str) -> Optional[SSOSession]:
    session = _sessions.get(session_id)
    if not session:
        return None
    if datetime.now(timezone.utc) > session.expires_at:
        del _sessions[session_id]
        return None
    return session


def _set_session_cookie(
    response: Response,
    session: SSOSession,
    cookie_name: str = SESSION_COOKIE,
) -> None:
    response.set_cookie(
        key=cookie_name,
        value=session.session_id,
        httponly=True,
        samesite="lax",
        secure=False,  # Set True in production with HTTPS
        max_age=SESSION_TTL_HOURS * 3600,
        path="/",
    )


def _resolve_cookie_name(context: str) -> str:
    """Map a context string ('admin' or 'workspace') to the right cookie name."""
    return ADMIN_SESSION_COOKIE if context == "admin" else SESSION_COOKIE


# ────────────────────────────────────────────────────────────────────
# Persistence helpers
# ────────────────────────────────────────────────────────────────────
def _save_users() -> None:
    """Persist users to disk."""
    try:
        data = {uid: u.model_dump(mode="json") for uid, u in _users.items()}
        _USERS_FILE.write_text(json.dumps(data, indent=2, default=str))
        os.chmod(str(_USERS_FILE), 0o600)
    except Exception as e:
        logger.warning(f"Could not save users: {e}")


def _load_users() -> None:
    """Load users from disk."""
    global _users, _users_by_email
    if not _USERS_FILE.exists():
        return
    try:
        data = json.loads(_USERS_FILE.read_text())
        for uid, udata in data.items():
            user = LocalUser(**udata)
            _users[uid] = user
            _users_by_email[user.email.lower()] = uid
        logger.info(f"Loaded {len(_users)} users from {_USERS_FILE}")
    except Exception as e:
        logger.warning(f"Could not load users: {e}")


# ────────────────────────────────────────────────────────────────────
# Super-admin bootstrap — runs once on first deployment
# ────────────────────────────────────────────────────────────────────
def _ensure_super_admin() -> None:
    """Generate a super-admin account on first startup if none exists.

    Credentials are written to .agent_super_admin.json (0600) and
    printed to stderr so the operator can grab them from deployment logs.
    """
    global _super_admin_generated

    # Already have at least one admin → nothing to do
    for u in _users.values():
        if UserRole.ADMIN in u.roles:
            return

    # If we previously generated one, read it from disk rather than
    # creating a second super-admin on every restart
    if _SUPER_ADMIN_FILE.exists():
        try:
            creds = json.loads(_SUPER_ADMIN_FILE.read_text())
            # Might have been deleted from users file — re-create if needed
            if creds.get("email") and creds["email"].lower() not in _users_by_email:
                _register_super_admin(creds["email"], creds["password"])
            return
        except Exception:
            pass  # file corrupt → re-generate

    # Generate fresh credentials
    password = secrets.token_urlsafe(20)
    email = "admin@autodev.local"

    _register_super_admin(email, password)

    # Persist credentials to disk (operator reference)
    try:
        _SUPER_ADMIN_FILE.write_text(json.dumps({
            "email": email,
            "password": password,
            "note": "Super-admin credentials generated at first deployment. "
                    "Change the password from the admin panel.",
        }, indent=2))
        os.chmod(str(_SUPER_ADMIN_FILE), 0o600)
    except Exception as e:
        logger.warning(f"Could not save super-admin credentials file: {e}")

    # Print to stderr for the operator to see in deployment logs
    import sys
    print(
        "\n"
        "  ╔══════════════════════════════════════════════════════════╗\n"
        "  ║           SUPER-ADMIN CREDENTIALS (first run)           ║\n"
        "  ╠══════════════════════════════════════════════════════════╣\n"
        f"  ║  Email:    {email:<45}║\n"
        f"  ║  Password: {password:<45}║\n"
        "  ╠══════════════════════════════════════════════════════════╣\n"
        "  ║  Save these now — password is NOT shown again.          ║\n"
        "  ║  Change it from Admin Panel → User Management.          ║\n"
        "  ╚══════════════════════════════════════════════════════════╝\n",
        file=sys.stderr,
    )
    _super_admin_generated = True


def _register_super_admin(email: str, password: str) -> LocalUser:
    """Create the super-admin user in the users store."""
    user_id = secrets.token_urlsafe(16)
    user = LocalUser(
        id=user_id,
        email=email.lower(),
        display_name="Super Admin",
        password_hash=_hash_password(password),
        active=True,
        roles=[UserRole.ADMIN],
        auth_provider=AuthProvider.LOCAL,
    )
    _users[user_id] = user
    _users_by_email[email.lower()] = user_id
    _save_users()
    logger.info(f"Super-admin created: {email}")
    return user


# ────────────────────────────────────────────────────────────────────
# Admin role check helper
# ────────────────────────────────────────────────────────────────────
def _get_session_user(request: Request, cookie_name: str | None = None) -> LocalUser:
    """Get the authenticated user from a session cookie or raise 401.

    If *cookie_name* is given, only that cookie is checked.
    Otherwise both the admin and workspace cookies are tried (admin first).
    """
    cookies_to_try = [cookie_name] if cookie_name else [ADMIN_SESSION_COOKIE, SESSION_COOKIE]
    for cname in cookies_to_try:
        session_id = request.cookies.get(cname)
        if not session_id:
            continue
        session = _validate_session(session_id)
        if not session:
            continue
        user = _users.get(session.user_id)
        if user:
            return user
    raise HTTPException(status_code=401, detail="Not authenticated")


def _has_valid_bearer_token(request: Request) -> bool:
    """Check if the request carries a valid Bearer API token.

    The API token is a shared admin secret — if the request already passed
    the ``_verify_token`` dependency with a valid Bearer token, the caller
    has full admin-level access (same privilege as every other admin config
    endpoint).
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header[7:]
    # Import lazily to avoid circular dependency
    from fastcoder.api import _get_api_token
    return secrets.compare_digest(token, _get_api_token())


def _require_admin(request: Request) -> LocalUser:
    """Get the authenticated user and verify admin access.

    Accepts either:
    1. A valid admin session cookie (returns the associated user), or
    2. A valid Bearer API token (returns a synthetic admin sentinel).

    The Bearer token path is needed because the admin panel SPA
    authenticates via the API token, not session cookies.
    """
    # Try session cookie first
    try:
        user = _get_session_user(request, cookie_name=ADMIN_SESSION_COOKIE)
        if UserRole.ADMIN not in user.roles:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user
    except HTTPException:
        pass

    # Fall back to Bearer token (admin panel SPA auth)
    if _has_valid_bearer_token(request):
        # Return a synthetic admin user for Bearer-token callers
        return LocalUser(
            id="__bearer_admin__",
            email="admin@local",
            display_name="Admin (API)",
            password_hash="",
            active=True,
            roles=[UserRole.ADMIN],
            auth_provider=AuthProvider.LOCAL,
        )

    raise HTTPException(status_code=401, detail="Not authenticated")


# ────────────────────────────────────────────────────────────────────
# Request / Response models
# ────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=100)
    context: str = "workspace"   # "workspace" or "admin"

class LoginRequest(BaseModel):
    email: str
    password: str
    context: str = "workspace"   # "workspace" or "admin"

class AuthResponse(BaseModel):
    success: bool
    user: Optional[dict] = None
    session_id: Optional[str] = None
    message: str = ""

class UserInfoResponse(BaseModel):
    id: str
    email: str
    display_name: str
    avatar_url: Optional[str] = None
    provider: str
    roles: list[str]


# ────────────────────────────────────────────────────────────────────
# Router factory
# ────────────────────────────────────────────────────────────────────
def create_auth_router() -> APIRouter:
    """Create the authentication router with registration, login, and OAuth endpoints."""

    router = APIRouter(prefix="/auth", tags=["authentication"])

    # Load persisted users and bootstrap super-admin on first run
    _load_users()
    _ensure_super_admin()

    # ── Available Providers ──────────────────────────────────────
    @router.get("/providers")
    async def list_providers() -> dict:
        """Return which auth providers are available."""
        providers = _get_oauth_providers()
        available = ["email"]  # Email/password is always available
        for name in providers:
            available.append(name)
        return {"providers": available}

    # ── Email/Password Registration ──────────────────────────────
    @router.post("/register", response_model=AuthResponse)
    async def register(body: RegisterRequest, response: Response) -> AuthResponse:
        email = body.email.strip().lower()

        # Validate email format
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
            raise HTTPException(status_code=400, detail="Invalid email address")

        # Check if email already exists
        if email in _users_by_email:
            raise HTTPException(status_code=409, detail="An account with this email already exists")

        # Validate password strength
        if len(body.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

        # Create user
        user_id = secrets.token_urlsafe(16)
        user = LocalUser(
            id=user_id,
            email=email,
            display_name=body.display_name.strip(),
            password_hash=_hash_password(body.password),
            auth_provider=AuthProvider.LOCAL,
        )
        _users[user_id] = user
        _users_by_email[email] = user_id
        _save_users()

        # Create session — set the right cookie based on context
        session = _create_session(user)
        cookie_name = _resolve_cookie_name(body.context)
        _set_session_cookie(response, session, cookie_name=cookie_name)

        logger.info(f"User registered: {email} (context={body.context})")
        return AuthResponse(
            success=True,
            session_id=session.session_id,
            user={"id": user.id, "email": user.email, "display_name": user.display_name},
            message="Registration successful",
        )

    # ── Email/Password Login ─────────────────────────────────────
    @router.post("/login", response_model=AuthResponse)
    async def login(body: LoginRequest, response: Response) -> AuthResponse:
        email = body.email.strip().lower()

        user_id = _users_by_email.get(email)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user = _users.get(user_id)
        if not user or not user.active:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # For OAuth-only users, they can't login with password
        if user.auth_provider != AuthProvider.LOCAL or not user.password_hash:
            raise HTTPException(
                status_code=401,
                detail=f"This account uses {user.auth_provider.value} sign-in. Please use that provider.",
            )

        if not _verify_password(body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # For admin context, verify user actually has admin role
        if body.context == "admin" and UserRole.ADMIN not in user.roles:
            raise HTTPException(status_code=403, detail="Admin access required. This account does not have admin privileges.")

        session = _create_session(user)
        cookie_name = _resolve_cookie_name(body.context)
        _set_session_cookie(response, session, cookie_name=cookie_name)

        logger.info(f"User logged in: {email} (context={body.context})")
        return AuthResponse(
            success=True,
            session_id=session.session_id,
            user={"id": user.id, "email": user.email, "display_name": user.display_name},
            message="Login successful",
        )

    # ── Session / Me ─────────────────────────────────────────────
    @router.get("/me")
    async def get_current_user(request: Request, context: str = "workspace") -> dict:
        """Return the current authenticated user, or 401.

        Query param ``context`` selects which session cookie to check:
        - ``workspace`` (default) → checks ``agent_session``
        - ``admin`` → checks ``agent_admin_session``
        """
        cookie_name = _resolve_cookie_name(context)
        session_id = request.cookies.get(cookie_name)
        if not session_id:
            raise HTTPException(status_code=401, detail="Not authenticated")

        session = _validate_session(session_id)
        if not session:
            raise HTTPException(status_code=401, detail="Session expired")

        user = _users.get(session.user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "avatar_url": user.avatar_url,
            "provider": user.auth_provider.value,
            "roles": [r.value for r in user.roles],
            "is_admin": UserRole.ADMIN in user.roles,
        }

    # ── Logout ───────────────────────────────────────────────────
    @router.post("/logout")
    async def logout(request: Request, response: Response, context: str = "workspace") -> dict:
        """Logout. Query param ``context`` = 'admin' clears admin session; default clears workspace."""
        cookie_name = _resolve_cookie_name(context)
        session_id = request.cookies.get(cookie_name)
        if session_id and session_id in _sessions:
            del _sessions[session_id]
        response.delete_cookie(cookie_name, path="/")
        return {"success": True, "message": "Logged out"}

    # ── OAuth: Initiate ──────────────────────────────────────────
    @router.get("/oauth/{provider}/login")
    async def oauth_login(provider: str, request: Request, context: str = "workspace") -> dict:
        """Initiate OAuth login. Returns the authorization URL to redirect to.

        Query param ``context`` = 'admin' | 'workspace' controls which session
        cookie is set after the callback completes.
        """
        providers = _get_oauth_providers()
        if provider not in providers:
            raise HTTPException(status_code=400, detail=f"Provider '{provider}' is not configured")

        config = providers[provider]
        state = secrets.token_urlsafe(32)

        # Determine redirect URI
        host = request.headers.get("host", "localhost:8000")
        scheme = request.headers.get("x-forwarded-proto", "http")
        redirect_uri = f"{scheme}://{host}/auth/oauth/{provider}/callback"

        # Store state for verification (include context for callback)
        _oauth_states[state] = {
            "provider": provider,
            "redirect_uri": redirect_uri,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "context": context,
        }

        # Build authorization URL
        params: dict[str, str] = {
            "client_id": config.client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": " ".join(config.scopes),
        }

        # Provider-specific tweaks
        if provider == "google":
            params["response_type"] = "code"
            params["access_type"] = "offline"
            params["prompt"] = "select_account"
        elif provider == "github":
            pass  # GitHub uses `code` response type by default
        elif provider == "microsoft":
            params["response_type"] = "code"
            params["response_mode"] = "query"
        elif provider == "apple":
            params["response_type"] = "code"
            params["response_mode"] = "form_post"
        elif provider in ("gitlab", "bitbucket"):
            params["response_type"] = "code"

        auth_url = f"{config.authorize_url}?{urlencode(params)}"
        return {"auth_url": auth_url, "state": state}

    # ── OAuth: Callback ──────────────────────────────────────────
    @router.get("/oauth/{provider}/callback")
    async def oauth_callback(
        provider: str,
        request: Request,
        response: Response,
        code: str = "",
        state: str = "",
        error: str = "",
    ):
        """Handle OAuth provider callback. Exchanges code for user info, creates/finds user, sets session."""
        if error:
            return _redirect_to_login(f"OAuth error: {error}")

        if not code or not state:
            return _redirect_to_login("Missing authorization code or state")

        # Verify state
        state_data = _oauth_states.pop(state, None)
        if not state_data or state_data["provider"] != provider:
            return _redirect_to_login("Invalid or expired state. Please try again.")

        # Check state age (10 minute max)
        created = datetime.fromisoformat(state_data["created_at"])
        if datetime.now(timezone.utc) - created > timedelta(minutes=10):
            return _redirect_to_login("Login attempt expired. Please try again.")

        providers = _get_oauth_providers()
        config = providers.get(provider)
        if not config:
            return _redirect_to_login(f"Provider '{provider}' not configured")

        redirect_uri = state_data["redirect_uri"]

        try:
            # Exchange code for access token
            token_data = await _exchange_code_for_token(config, code, redirect_uri)
            access_token = token_data.get("access_token")
            if not access_token:
                return _redirect_to_login("Failed to get access token")

            # Fetch user info from provider
            user_info = await _fetch_user_info(provider, config, access_token, token_data)

            email = user_info.get("email", "").strip().lower()
            display_name = user_info.get("name", "") or user_info.get("login", "") or email.split("@")[0]
            avatar_url = user_info.get("avatar_url") or user_info.get("picture") or ""
            provider_id = str(user_info.get("id") or user_info.get("sub") or email)

            if not email:
                # Some providers (GitHub) need a separate email call
                if provider == "github":
                    email = await _fetch_github_email(access_token)
                if not email:
                    return _redirect_to_login("Could not get email from provider")

            # Find or create user
            provider_enum = AuthProvider(provider)
            user = _find_or_create_oauth_user(email, display_name, avatar_url, provider_enum, provider_id)

            # Determine session context from stored state
            oauth_context = state_data.get("context", "workspace")

            # For admin context, verify user has admin role
            if oauth_context == "admin" and UserRole.ADMIN not in user.roles:
                return _redirect_to_login("Admin access required. This account does not have admin privileges.")

            # Create session and set the correct cookie
            session = _create_session(user)
            cookie_name = _resolve_cookie_name(oauth_context)
            redirect_url = "/"

            resp = RedirectResponse(url=redirect_url, status_code=302)
            _set_session_cookie(resp, session, cookie_name=cookie_name)
            logger.info(f"OAuth login success: {provider} / {email} (context={oauth_context})")
            return resp

        except Exception as e:
            logger.error(f"OAuth callback error: {e}", exc_info=True)
            return _redirect_to_login(f"Authentication failed. Please try again.")

    # ══════════════════════════════════════════════════════════════
    # Admin endpoints — all require ADMIN role
    # ══════════════════════════════════════════════════════════════

    # ── OAuth provider management ────────────────────────────────

    @router.get("/admin/oauth/templates")
    async def get_oauth_templates(request: Request) -> dict:
        """Return the well-known provider templates with default URLs."""
        _require_admin(request)
        return {"templates": OAUTH_PROVIDER_TEMPLATES}

    @router.get("/admin/oauth/providers")
    async def admin_list_oauth_providers(request: Request) -> dict:
        """Return all configured OAuth providers (secrets masked)."""
        _require_admin(request)
        providers = _get_oauth_providers()
        result = {}
        for name, cfg in providers.items():
            result[name] = {
                "provider": cfg.provider.value,
                "client_id": cfg.client_id,
                "client_secret_set": bool(cfg.client_secret),
                "client_secret_masked": _mask_secret(cfg.client_secret),
                "authorize_url": cfg.authorize_url,
                "token_url": cfg.token_url,
                "userinfo_url": cfg.userinfo_url,
                "scopes": cfg.scopes,
            }
        return {"providers": result}

    @router.put("/admin/oauth/providers/{provider_name}")
    async def admin_set_oauth_provider(provider_name: str, request: Request) -> dict:
        """Create or update an OAuth provider config."""
        _require_admin(request)
        body = await request.json()

        # Validate provider name
        valid_names = set(OAUTH_PROVIDER_TEMPLATES.keys())
        if provider_name not in valid_names:
            raise HTTPException(400, f"Unknown provider: {provider_name}. Valid: {sorted(valid_names)}")

        client_id = body.get("client_id", "").strip()
        client_secret = body.get("client_secret", "").strip()

        if not client_id:
            raise HTTPException(400, "client_id is required")

        # If updating and no new secret provided, keep the old one
        existing = _oauth_providers.get(provider_name)
        if not client_secret and existing:
            client_secret = existing.client_secret

        if not client_secret:
            raise HTTPException(400, "client_secret is required")

        # Use template defaults for URLs, but allow overrides
        template = OAUTH_PROVIDER_TEMPLATES[provider_name]
        provider_enum = AuthProvider(provider_name)

        config = OAuthProviderConfig(
            provider=provider_enum,
            client_id=client_id,
            client_secret=client_secret,
            authorize_url=body.get("authorize_url", "").strip() or template["authorize_url"],
            token_url=body.get("token_url", "").strip() or template["token_url"],
            userinfo_url=body.get("userinfo_url", "").strip() or template["userinfo_url"],
            scopes=body.get("scopes") or template["scopes"],
        )

        _set_oauth_provider(provider_name, config)
        logger.info(f"OAuth provider configured: {provider_name}")

        return {"success": True, "message": f"{template['label']} OAuth configured", "provider": provider_name}

    @router.delete("/admin/oauth/providers/{provider_name}")
    async def admin_remove_oauth_provider(provider_name: str, request: Request) -> dict:
        """Remove an OAuth provider config."""
        _require_admin(request)
        if _remove_oauth_provider(provider_name):
            logger.info(f"OAuth provider removed: {provider_name}")
            return {"success": True, "message": f"Provider '{provider_name}' removed"}
        raise HTTPException(404, f"Provider '{provider_name}' not found")

    # ── User management ──────────────────────────────────────────

    @router.get("/admin/auth/users")
    async def admin_list_users(request: Request) -> dict:
        """List all registered users (admin view)."""
        _require_admin(request)
        users_list = []
        for uid, user in _users.items():
            users_list.append({
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "provider": user.auth_provider.value,
                "active": user.active,
                "roles": [r.value for r in user.roles],
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "avatar_url": user.avatar_url,
            })
        return {"users": users_list, "total": len(users_list)}

    @router.post("/admin/auth/users")
    async def admin_create_user(request: Request) -> dict:
        """Create a new admin user (only admins can do this)."""
        admin = _require_admin(request)
        body = await request.json()

        email = (body.get("email") or "").strip().lower()
        password = (body.get("password") or "").strip()
        display_name = (body.get("display_name") or "").strip()
        role = (body.get("role") or "admin").strip().lower()

        if not email or not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
            raise HTTPException(400, "Valid email is required")
        if not password or len(password) < 8:
            raise HTTPException(400, "Password must be at least 8 characters")
        if not display_name:
            display_name = email.split("@")[0]
        if email in _users_by_email:
            raise HTTPException(409, "A user with this email already exists")

        # Map role string to UserRole
        valid_roles = {r.value: r for r in UserRole}
        if role not in valid_roles:
            raise HTTPException(400, f"Invalid role. Valid: {list(valid_roles.keys())}")

        user_id = secrets.token_urlsafe(16)
        user = LocalUser(
            id=user_id,
            email=email,
            display_name=display_name,
            password_hash=_hash_password(password),
            active=True,
            roles=[valid_roles[role]],
            auth_provider=AuthProvider.LOCAL,
        )
        _users[user_id] = user
        _users_by_email[email] = user_id
        _save_users()
        logger.info(f"Admin created user: {email} (role={role}) by {admin.email}")

        return {
            "success": True,
            "message": f"User '{email}' created with role '{role}'",
            "user": {"id": user_id, "email": email, "display_name": display_name, "role": role},
        }

    @router.put("/admin/auth/users/{user_id}/role")
    async def admin_set_user_role(user_id: str, request: Request) -> dict:
        """Change a user's role."""
        admin = _require_admin(request)
        body = await request.json()

        target = _users.get(user_id)
        if not target:
            raise HTTPException(404, "User not found")

        role = (body.get("role") or "").strip().lower()
        valid_roles = {r.value: r for r in UserRole}
        if role not in valid_roles:
            raise HTTPException(400, f"Invalid role. Valid: {list(valid_roles.keys())}")

        # Prevent the last admin from demoting themselves
        if target.id == admin.id and role != "admin":
            admin_count = sum(1 for u in _users.values() if UserRole.ADMIN in u.roles and u.active)
            if admin_count <= 1:
                raise HTTPException(400, "Cannot remove the last admin. Create another admin first.")

        target.roles = [valid_roles[role]]
        target.updated_at = datetime.now(timezone.utc)
        _save_users()
        logger.info(f"User {target.email} role changed to {role} by {admin.email}")

        return {"success": True, "message": f"Role updated to '{role}'"}

    @router.put("/admin/auth/users/{user_id}/active")
    async def admin_toggle_user_active(user_id: str, request: Request) -> dict:
        """Enable or disable a user account."""
        admin = _require_admin(request)
        body = await request.json()

        target = _users.get(user_id)
        if not target:
            raise HTTPException(404, "User not found")

        active = body.get("active", True)

        # Prevent disabling the last admin
        if not active and UserRole.ADMIN in target.roles:
            admin_count = sum(1 for u in _users.values() if UserRole.ADMIN in u.roles and u.active)
            if admin_count <= 1:
                raise HTTPException(400, "Cannot disable the last admin account.")

        target.active = bool(active)
        target.updated_at = datetime.now(timezone.utc)
        _save_users()
        logger.info(f"User {target.email} {'enabled' if active else 'disabled'} by {admin.email}")

        return {"success": True, "message": f"User {'enabled' if active else 'disabled'}"}

    @router.put("/admin/auth/users/{user_id}/password")
    async def admin_reset_user_password(user_id: str, request: Request) -> dict:
        """Reset a user's password (admin action)."""
        admin = _require_admin(request)
        body = await request.json()

        target = _users.get(user_id)
        if not target:
            raise HTTPException(404, "User not found")

        # Only local-auth users can have password reset
        if target.auth_provider != AuthProvider.LOCAL:
            raise HTTPException(400, f"This user authenticates via {target.auth_provider.value} — password cannot be set")

        new_password = (body.get("password") or "").strip()
        if not new_password or len(new_password) < 8:
            raise HTTPException(400, "New password must be at least 8 characters")

        target.password_hash = _hash_password(new_password)
        target.updated_at = datetime.now(timezone.utc)
        _save_users()
        logger.info(f"Password reset for {target.email} by {admin.email}")

        return {"success": True, "message": f"Password reset for {target.email}"}

    @router.delete("/admin/auth/users/{user_id}")
    async def admin_delete_user(user_id: str, request: Request) -> dict:
        """Delete a user account permanently."""
        admin = _require_admin(request)

        target = _users.get(user_id)
        if not target:
            raise HTTPException(404, "User not found")

        # Prevent deleting the last admin
        if UserRole.ADMIN in target.roles:
            admin_count = sum(1 for u in _users.values() if UserRole.ADMIN in u.roles and u.active)
            if admin_count <= 1:
                raise HTTPException(400, "Cannot delete the last admin account.")

        # Prevent self-deletion
        if target.id == admin.id:
            raise HTTPException(400, "Cannot delete your own account. Ask another admin.")

        del _users[user_id]
        _users_by_email.pop(target.email.lower(), None)
        # Invalidate their sessions
        to_remove = [sid for sid, s in _sessions.items() if s.user_id == user_id]
        for sid in to_remove:
            del _sessions[sid]
        _save_users()
        logger.info(f"User {target.email} deleted by {admin.email}")

        return {"success": True, "message": f"User '{target.email}' deleted"}

    return router


def _mask_secret(secret: str) -> str:
    """Mask a secret showing only last 4 characters."""
    if not secret:
        return ""
    if len(secret) <= 4:
        return "****"
    return f"****{secret[-4:]}"


# ────────────────────────────────────────────────────────────────────
# OAuth helpers
# ────────────────────────────────────────────────────────────────────
async def _exchange_code_for_token(
    config: OAuthProviderConfig, code: str, redirect_uri: str
) -> dict[str, Any]:
    """Exchange an OAuth authorization code for an access token."""
    import httpx

    data = {
        "client_id": config.client_id,
        "client_secret": config.client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    headers = {"Accept": "application/json"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(config.token_url, data=data, headers=headers)
        resp.raise_for_status()

        # GitHub returns application/x-www-form-urlencoded sometimes
        content_type = resp.headers.get("content-type", "")
        if "json" in content_type:
            return resp.json()
        else:
            # Parse form-encoded response
            from urllib.parse import parse_qs
            parsed = parse_qs(resp.text)
            return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}


async def _fetch_user_info(
    provider: str,
    config: OAuthProviderConfig,
    access_token: str,
    token_data: dict,
) -> dict[str, Any]:
    """Fetch user profile from the OAuth provider."""
    import httpx

    # Apple embeds user info in the ID token, not a userinfo endpoint
    if provider == "apple":
        id_token = token_data.get("id_token", "")
        if id_token:
            import base64
            parts = id_token.split(".")
            if len(parts) >= 2:
                payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
                return json.loads(base64.urlsafe_b64decode(payload))
        return {}

    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    # GitHub uses different header format
    if provider == "github":
        headers["Authorization"] = f"token {access_token}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(config.userinfo_url, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _fetch_github_email(access_token: str) -> str:
    """GitHub doesn't always return email in profile; fetch from emails API."""
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://api.github.com/user/emails",
            headers={"Authorization": f"token {access_token}", "Accept": "application/json"},
        )
        if resp.status_code == 200:
            emails = resp.json()
            # Prefer the primary verified email
            for e in emails:
                if e.get("primary") and e.get("verified"):
                    return e["email"]
            # Fallback to first verified
            for e in emails:
                if e.get("verified"):
                    return e["email"]
            # Fallback to first
            if emails:
                return emails[0].get("email", "")
    return ""


def _find_or_create_oauth_user(
    email: str,
    display_name: str,
    avatar_url: str,
    provider: AuthProvider,
    provider_id: str,
) -> LocalUser:
    """Find existing user by email or create a new one for OAuth login."""
    email_lower = email.lower()

    # Check if user already exists
    existing_uid = _users_by_email.get(email_lower)
    if existing_uid:
        user = _users[existing_uid]
        # Update avatar and display name if changed
        user.avatar_url = avatar_url or user.avatar_url
        user.display_name = display_name or user.display_name
        user.updated_at = datetime.now(timezone.utc)
        _save_users()
        return user

    # Create new user
    user_id = secrets.token_urlsafe(16)
    user = LocalUser(
        id=user_id,
        email=email_lower,
        display_name=display_name,
        password_hash="",  # OAuth users have no password
        auth_provider=provider,
        oauth_provider_id=provider_id,
        avatar_url=avatar_url,
    )
    _users[user_id] = user
    _users_by_email[email_lower] = user_id
    _save_users()
    return user


def _redirect_to_login(error_msg: str) -> RedirectResponse:
    """Redirect to login page with error message."""
    from urllib.parse import quote
    return RedirectResponse(url=f"/login?error={quote(error_msg)}", status_code=302)
